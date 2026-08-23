"""Default-off live wiring for Checkpoint V2 shadow evidence.

This module is imported lazily only when the operator explicitly sets a local
mode above OFF.  Server negotiation may still lower it to OFF.  OBSERVE and
RESTORE_TEST create one daemon sampler beside the ordinary Pier subprocess;
the sampler cannot change the process result, assignment state, or refill.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .api_client import ApiClient
from .checkpoint_activation_v2 import (
    CheckpointActivationV2,
    CheckpointV2ProtocolError,
    checkpoint_activation_from_assignment_v2,
)
from .checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from .checkpoint_container_bundle_v2 import build_checkpoint_container_bundle_v2
from .checkpoint_docker_runtime_v2 import (
    DockerCliLazyCheckpointExporterV2,
    docker_container_backend_v2,
)
from .checkpoint_runtime_v2 import CheckpointDataPlaneV2
from .checkpoint_shadow_v2 import (
    CheckpointShadowCoordinatorV2,
    CheckpointShadowRuntimeFactsV2,
)
from .checkpoint_v2 import (
    CheckpointV2Journal,
    checkpoint_machine_fingerprint,
)
from .providers import (
    DEEPSEEK_PROVIDER,
    DEEPSEEK_RUN_CONFIG_VERSION,
    DEEPSEEK_RUNTIME_PROFILE,
    DSH_RUN_CONFIG_VERSION,
    DSH_RUNTIME_PROFILE,
    KIMI_RUN_CONFIG_VERSION,
    KIMI_RUNTIME_PROFILE,
    ZCODE_RUN_CONFIG_VERSION,
    ZCODE_RUNTIME_PROFILE,
)
from .telemetry import RunnerTelemetry, platform_family


_OPENAI_CODEX_RUN_CONFIG_VERSION = "codex-chatgpt-subscription-v1"
_OPENAI_CODEX_RUNTIME_PROFILE = "public-pier-0.3.0-post3-codex-v1"


def checkpoint_live_activation_v2(
    assignment: Mapping[str, Any],
    *,
    local_mode: object,
) -> CheckpointActivationV2:
    """Negotiate before model start so authoritative modes cannot shadow-run."""

    activation = checkpoint_activation_from_assignment_v2(
        assignment, local_mode=local_mode,
    )
    if activation.authoritative:
        raise CheckpointV2ProtocolError(
            "authoritative checkpoint rollout requires the V2 owner runner"
        )
    return activation


def _runtime_config_v2(harness: str, provider: str) -> tuple[str, str]:
    if harness == "codex" and provider == "openai":
        return _OPENAI_CODEX_RUNTIME_PROFILE, _OPENAI_CODEX_RUN_CONFIG_VERSION
    if harness == "codex" and provider == DEEPSEEK_PROVIDER:
        return DEEPSEEK_RUNTIME_PROFILE, DEEPSEEK_RUN_CONFIG_VERSION
    if harness == "dsh" and provider == DEEPSEEK_PROVIDER:
        return DSH_RUNTIME_PROFILE, DSH_RUN_CONFIG_VERSION
    if harness == "kimi-code" and provider == "kimi-subscription":
        return KIMI_RUNTIME_PROFILE, KIMI_RUN_CONFIG_VERSION
    if harness == "zcode" and provider == "bigmodel-coding-plan":
        return ZCODE_RUNTIME_PROFILE, ZCODE_RUN_CONFIG_VERSION
    raise CheckpointV2ProtocolError("checkpoint live Harness/provider is unsupported")


def _runtime_digest_v2(
    *,
    contract,
    agent_version: str,
    runtime_profile: str,
    model_config_version: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix="dradar-checkpoint-v2-identity-") as raw:
        bundle = build_checkpoint_container_bundle_v2(
            Path(raw) / "helper.pyz",
        )
    payload = {
        "schema": "dradar-checkpoint-runtime-compatibility-v2",
        "client_version": __version__,
        "agent_version": agent_version,
        "runtime_profile": runtime_profile,
        "model_config_version": model_config_version,
        "helper_sha256": bundle.sha256,
        "contract": asdict(contract),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class LiveCheckpointShadowControllerV2:
    """Daemon lifecycle whose stop path is bounded and never owns mainline."""

    def __init__(
        self,
        coordinator: CheckpointShadowCoordinatorV2,
        *,
        initial_delay_sec: float = 30.0,
        interval_sec: float = 300.0,
        maximum_captures: int = 24,
    ) -> None:
        self.coordinator = coordinator
        self.initial_delay_sec = initial_delay_sec
        self.interval_sec = interval_sec
        self.maximum_captures = maximum_captures
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sync_elapsed_ms = 0
        self._closed_cleanly = False
        self._mainline_started_ns: int | None = None
        self._mainline_closed_ns: int | None = None

    async def _mainline_lifetime(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.1)

    def _run(self) -> None:
        async def lifecycle() -> None:
            try:
                await self.coordinator.run(
                    self._mainline_lifetime(),
                    initial_delay_sec=self.initial_delay_sec,
                    interval_sec=self.interval_sec,
                    maximum_captures=self.maximum_captures,
                )
            finally:
                await self.coordinator.release_local_snapshots()

        try:
            asyncio.run(lifecycle())
        except BaseException:
            # A shadow lane has no authority to surface an exception through
            # the ordinary Pier result or its upload path.
            pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="dradar-checkpoint-v2-shadow",
            daemon=True,
        )
        started = time.monotonic_ns()
        self._mainline_started_ns = started
        try:
            self._thread.start()
        finally:
            self._sync_elapsed_ms += max(
                0, (time.monotonic_ns() - started) // 1_000_000,
            )

    def close(self, timeout: float = 0.75) -> None:
        started = time.monotonic_ns()
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, min(float(timeout), 2.0)))
            self._closed_cleanly = not thread.is_alive()
        self._sync_elapsed_ms += max(
            0, (time.monotonic_ns() - started) // 1_000_000,
        )
        self._mainline_closed_ns = time.monotonic_ns()

    @property
    def impact_sample_id(self) -> str | None:
        if not self._closed_cleanly:
            return None
        value = getattr(self.coordinator, "impact_sample_id", None)
        return value if isinstance(value, str) else None

    @property
    def checkpoint_sync_elapsed_ms(self) -> int:
        return self._sync_elapsed_ms

    @property
    def mainline_elapsed_ms(self) -> int:
        started = self._mainline_started_ns
        closed = self._mainline_closed_ns
        if started is None or closed is None or closed < started:
            return 0
        return max(1, min(
            7 * 24 * 60 * 60 * 1000,
            (closed - started) // 1_000_000,
        ))


def build_live_checkpoint_shadow_v2(
    *,
    assignment: Mapping[str, Any],
    effective_assignment: Mapping[str, Any],
    local_mode: object,
    api: ApiClient,
    telemetry: RunnerTelemetry,
    home: Path,
    job_root: Path,
    initial_delay_sec: float = 30.0,
    interval_sec: float = 300.0,
    maximum_captures: int = 24,
) -> LiveCheckpointShadowControllerV2 | None:
    """Build only after both sides enable a non-authoritative shadow mode."""

    activation = checkpoint_live_activation_v2(
        assignment, local_mode=local_mode,
    )
    if not activation.capture_enabled:
        return None
    identity = assignment.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise CheckpointV2ProtocolError("checkpoint claim identity is unavailable")
    harness = identity.get("harness")
    provider = identity.get("provider")
    if not isinstance(harness, str) or not isinstance(provider, str):
        raise CheckpointV2ProtocolError("checkpoint claim identity is incomplete")
    contract = checkpoint_adapter_contract_v2(harness, provider)
    agent_version = effective_assignment.get("agent_version")
    if not isinstance(agent_version, str) or not agent_version:
        raise CheckpointV2ProtocolError("checkpoint runtime version is unavailable")
    runtime_profile, model_config_version = _runtime_config_v2(harness, provider)
    runtime_digest = _runtime_digest_v2(
        contract=contract,
        agent_version=agent_version,
        runtime_profile=runtime_profile,
        model_config_version=model_config_version,
    )
    backend = docker_container_backend_v2()
    machine_fingerprint = checkpoint_machine_fingerprint(Path(home))
    telemetry.configure_checkpoint_runtime(
        container_backend=backend,
        machine_fingerprint=machine_fingerprint,
    )
    telemetry.configure_checkpoint_observation_reporting(Path(home))
    exporter = DockerCliLazyCheckpointExporterV2(
        job_root=job_root,
        contract=contract,
    )
    recovery_capability = (
        "NONE" if contract.native_resume_required else "WORKSPACE_ONLY"
    )
    runtime = CheckpointShadowRuntimeFactsV2(
        harness=harness,
        provider=provider,
        agent_version=agent_version,
        runtime_profile=runtime_profile,
        model_config_version=model_config_version,
        checkpoint_abi=contract.checkpoint_abi,
        runtime_compatibility_digest=runtime_digest,
        recovery_capability=recovery_capability,
        native_state_schema=contract.native_state_schema,
        machine_fingerprint=machine_fingerprint,
        platform=platform_family(),
        container_backend=backend,
        capture_adapter_version=contract.exporter_version,
    )
    plane = CheckpointDataPlaneV2(
        activation=activation,
        storage_root=(
            Path(home).absolute()
            / "checkpoint-v2"
            / "shadow"
            / str(assignment["assignment_id"])
        ),
        shadow_budget_root=(
            Path(home).absolute() / "checkpoint-v2" / "shadow"
        ),
    )
    coordinator = CheckpointShadowCoordinatorV2(
        assignment=assignment,
        activation=activation,
        api=api,
        journal=CheckpointV2Journal(Path(home)),
        data_plane=plane,
        exporter=exporter,
        observation_sink=telemetry,
        runtime=runtime,
        restorer_factory=(
            exporter.restorer if activation.offline_restore_enabled else None
        ),
    )
    return LiveCheckpointShadowControllerV2(
        coordinator,
        initial_delay_sec=initial_delay_sec,
        interval_sec=interval_sec,
        maximum_captures=maximum_captures,
    )


__all__ = [
    "LiveCheckpointShadowControllerV2",
    "build_live_checkpoint_shadow_v2",
    "checkpoint_live_activation_v2",
]
