"""Live-side orchestration for the optional Checkpoint V2 shadow lane.

This module is the only bridge between identity negotiation, periodic capture,
offline restore testing, and the telemetry outbox.  It owns no assignment
state and exposes no paid-execution permit.  Callers must construct it lazily
through :func:`run_mainline_with_optional_checkpoint_shadow_v2`; OFF therefore
does not import a Harness adapter, create storage, start a thread, or call the
server.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

from . import __version__
from .api_client import ApiClient
from .checkpoint_activation_v2 import (
    CheckpointActivationV2,
    CheckpointV2ProtocolError,
)
from .checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneError,
    CheckpointDataPlaneV2,
    CheckpointObservationRuntimeV2,
    CheckpointObservationV2,
    CheckpointRestoreObservationV2,
    CheckpointRestoreRequestV2,
    HarnessCheckpointExporterV2,
    HarnessCheckpointRestorerV2,
    checkpoint_observation_payload_v2,
    checkpoint_restore_observation_payload_v2,
    new_capture_request_v2,
    next_shadow_generation_v2,
    run_mainline_with_periodic_shadow_captures_v2,
)
from .checkpoint_v2 import (
    CheckpointV2Journal,
    ExecutionIdentityV2,
    finalize_execution_identity_v2,
    new_operation_id,
)


_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9._/+:-]{1,160}")


class CheckpointShadowObservationSinkV2(Protocol):
    def record_checkpoint_observation(self, payload: dict) -> bool: ...


@dataclass(frozen=True)
class CheckpointShadowRuntimeFactsV2:
    harness: str
    provider: str
    agent_version: str
    runtime_profile: str
    model_config_version: str
    checkpoint_abi: str
    runtime_compatibility_digest: str
    recovery_capability: str
    native_state_schema: str | None
    machine_fingerprint: str
    platform: str
    container_backend: str
    capture_adapter_version: str
    client_version: str = __version__

    def __post_init__(self) -> None:
        for value, label in (
            (self.harness, "harness"),
            (self.provider, "provider"),
            (self.agent_version, "agent version"),
            (self.runtime_profile, "runtime profile"),
            (self.model_config_version, "model config version"),
            (self.checkpoint_abi, "checkpoint ABI"),
            (self.capture_adapter_version, "capture adapter version"),
        ):
            if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
                raise ValueError(f"checkpoint shadow {label} is invalid")
        if _DIGEST_RE.fullmatch(self.runtime_compatibility_digest) is None:
            raise ValueError("checkpoint shadow runtime digest is invalid")
        if _DIGEST_RE.fullmatch(self.machine_fingerprint) is None:
            raise ValueError("checkpoint shadow machine fingerprint is invalid")
        if self.recovery_capability not in {
            "NATIVE_VALID", "WORKSPACE_ONLY", "COMPLETED_UPLOAD_ONLY", "NONE",
        }:
            raise ValueError("checkpoint shadow recovery capability is invalid")
        if (
            self.native_state_schema is not None
            and _TOKEN_RE.fullmatch(self.native_state_schema) is None
        ):
            raise ValueError("checkpoint shadow native-state schema is invalid")
        if self.platform not in {"macos", "linux", "wsl", "windows", "other"}:
            raise ValueError("checkpoint shadow platform is invalid")
        if self.container_backend not in {
            "docker", "orbstack", "podman", "native", "other",
        }:
            raise ValueError("checkpoint shadow container backend is invalid")
        if re.fullmatch(r"[A-Za-z0-9._+-]{1,64}", self.client_version) is None:
            raise ValueError("checkpoint shadow client version is invalid")


def _stable_shadow_ids(
    assignment_id: str,
    identity_fingerprint: str,
    machine_fingerprint: str,
) -> tuple[str, str]:
    material = (
        f"dradar-checkpoint-shadow-v2:{assignment_id}:"
        f"{identity_fingerprint}:{machine_fingerprint}"
    ).encode("ascii")
    digest = hashlib.sha256(material).hexdigest()
    return f"shadow-{digest[:48]}", f"lineage-{digest[16:64]}"


class CheckpointShadowCoordinatorV2:
    """Run bounded shadow capture/restore work beside one authoritative run."""

    def __init__(
        self,
        *,
        assignment: Mapping[str, Any],
        activation: CheckpointActivationV2,
        api: ApiClient,
        journal: CheckpointV2Journal,
        data_plane: CheckpointDataPlaneV2,
        exporter: HarnessCheckpointExporterV2,
        observation_sink: CheckpointShadowObservationSinkV2,
        runtime: CheckpointShadowRuntimeFactsV2,
        restorer_factory: (
            Callable[[], HarnessCheckpointRestorerV2] | None
        ) = None,
    ) -> None:
        if not activation.capture_enabled or activation.authoritative:
            raise CheckpointV2ProtocolError(
                "shadow coordinator requires OBSERVE or RESTORE_TEST activation"
            )
        if data_plane.activation != activation:
            raise CheckpointV2ProtocolError(
                "checkpoint shadow data-plane activation differs"
            )
        if exporter.checkpoint_abi != runtime.checkpoint_abi:
            raise CheckpointV2ProtocolError(
                "checkpoint shadow exporter ABI differs from runtime identity"
            )
        if activation.offline_restore_enabled != (restorer_factory is not None):
            raise CheckpointV2ProtocolError(
                "checkpoint shadow restore adapter does not match rollout mode"
            )
        self.assignment = dict(assignment)
        self.activation = activation
        self.api = api
        self.journal = journal
        self.data_plane = data_plane
        self.exporter = exporter
        self.observation_sink = observation_sink
        self.runtime = runtime
        self.restorer_factory = restorer_factory
        self._identity: ExecutionIdentityV2 | None = None
        self._identity_lock = asyncio.Lock()

    async def _finalized_identity(self) -> ExecutionIdentityV2:
        if self._identity is not None:
            return self._identity
        async with self._identity_lock:
            if self._identity is None:
                receipt = await asyncio.to_thread(
                    finalize_execution_identity_v2,
                    self.assignment,
                    api=self.api,
                    journal=self.journal,
                    harness=self.runtime.harness,
                    provider=self.runtime.provider,
                    agent_version=self.runtime.agent_version,
                    runtime_profile=self.runtime.runtime_profile,
                    model_config_version=self.runtime.model_config_version,
                    checkpoint_abi=self.runtime.checkpoint_abi,
                    runtime_compatibility_digest=(
                        self.runtime.runtime_compatibility_digest
                    ),
                )
                self._identity = receipt.identity
                register = getattr(
                    self.observation_sink,
                    "register_checkpoint_cohort",
                    None,
                )
                if callable(register):
                    try:
                        register({
                            "assignment_id": receipt.identity.assignment_id,
                            "identity_fingerprint": receipt.identity.fingerprint,
                            "cohort": {
                                "platform": self.runtime.platform,
                                "container_backend": (
                                    self.runtime.container_backend
                                ),
                                "harness": receipt.identity.harness,
                                "provider": receipt.identity.provider,
                                "client_version": self.runtime.client_version,
                                "agent_version": receipt.identity.agent_version,
                                "runtime_profile": (
                                    receipt.identity.runtime_profile
                                ),
                                "model_config_version": (
                                    receipt.identity.model_config_version
                                ),
                                "runtime_compatibility_digest": (
                                    receipt.identity.runtime_compatibility_digest
                                ),
                                "checkpoint_core_abi": (
                                    receipt.identity.checkpoint_core_abi
                                ),
                                "checkpoint_abi": receipt.identity.checkpoint_abi,
                            },
                        })
                    except Exception:
                        # Local evidence registration is itself shadow-only.
                        # Capture remains useful to the mainline even when the
                        # operator later has to mark this cohort incomplete.
                        pass
        return self._identity

    def _record(self, payload: dict[str, object]) -> None:
        try:
            self.observation_sink.record_checkpoint_observation(dict(payload))
        except Exception:
            pass

    async def _observe_restore(
        self,
        capture_request: CheckpointCaptureRequestV2,
        capture_observation: CheckpointObservationV2,
    ) -> None:
        if (
            not self.activation.offline_restore_enabled
            or capture_observation.published is None
            or self.restorer_factory is None
        ):
            return
        restore_id = f"restore-{new_operation_id()[:48]}"
        restore_request = CheckpointRestoreRequestV2(
            published=capture_observation.published,
            expected_identity_fingerprint=capture_request.identity_fingerprint,
            restore_id=restore_id,
        )
        started = time.monotonic()
        try:
            restorer = self.restorer_factory()
            observation = await self.data_plane.observe_offline_restore(
                restore_request, restorer,
            )
            adapter_version = restorer.adapter_version
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            observation = CheckpointRestoreObservationV2(
                status="failed",
                restore_id=restore_id,
                stage="restore",
                code="restore_failed",
                failure_type=type(exc).__name__[:64] or "Exception",
            )
            adapter_version = "restore-adapter-unavailable/1"
        elapsed_ms = min(86_400_000, int((time.monotonic() - started) * 1000))
        try:
            payload = checkpoint_restore_observation_payload_v2(
                capture_request,
                restore_request,
                observation,
                self.activation,
                CheckpointObservationRuntimeV2(
                    assignment_id=self.assignment["assignment_id"],
                    operation_id=new_operation_id(),
                    elapsed_ms=elapsed_ms,
                    platform=self.runtime.platform,
                    container_backend=self.runtime.container_backend,
                    client_version=self.runtime.client_version,
                    adapter_version=adapter_version,
                ),
            )
        except Exception:
            return
        self._record(payload)

    async def capture(self, _sample_generation: int) -> CheckpointObservationV2:
        """Capture one generation; every failure remains shadow-only."""

        try:
            identity = await self._finalized_identity()
        except Exception as exc:
            return CheckpointObservationV2(
                status="failed",
                capture_id=None,
                stage="capture",
                code="identity_finalization_failed",
                failure_type=type(exc).__name__[:64] or "Exception",
            )
        checkpoint_id, lineage_id = _stable_shadow_ids(
            identity.assignment_id,
            identity.fingerprint,
            self.runtime.machine_fingerprint,
        )
        recovery_capability = self.runtime.recovery_capability
        native_state_schema = self.runtime.native_state_schema
        recovery_facts = getattr(self.exporter, "recovery_facts", None)
        if callable(recovery_facts):
            try:
                prepared = await recovery_facts()
                if (
                    not isinstance(prepared, tuple)
                    or len(prepared) != 2
                    or prepared[0] not in {
                        "NATIVE_VALID", "WORKSPACE_ONLY",
                        "COMPLETED_UPLOAD_ONLY", "NONE",
                    }
                    or (
                        prepared[1] is not None
                        and (
                            not isinstance(prepared[1], str)
                            or _TOKEN_RE.fullmatch(prepared[1]) is None
                        )
                    )
                ):
                    raise CheckpointDataPlaneError(
                        "capture", "recovery_facts_invalid",
                    )
                recovery_capability, native_state_schema = prepared
            except asyncio.CancelledError:
                raise
            except Exception:
                # The concrete exporter will emit a structured capture failure
                # if the task container or native state is still unavailable.
                pass
        try:
            generation = await asyncio.to_thread(
                next_shadow_generation_v2,
                self.data_plane.storage_root,
                checkpoint_id,
            )
        except Exception:
            generation = max(1, int(_sample_generation))
        request = new_capture_request_v2(
            checkpoint_id=checkpoint_id,
            checkpoint_lineage_id=lineage_id,
            snapshot_generation=generation,
            identity_fingerprint=identity.fingerprint,
            checkpoint_abi=identity.checkpoint_abi,
            recovery_capability=recovery_capability,
            native_state_schema=native_state_schema,
        )
        started = time.monotonic()
        observation = await self.data_plane.observe_capture(request, self.exporter)
        elapsed_ms = min(86_400_000, int((time.monotonic() - started) * 1000))
        try:
            payload = checkpoint_observation_payload_v2(
                request,
                observation,
                self.activation,
                CheckpointObservationRuntimeV2(
                    assignment_id=identity.assignment_id,
                    operation_id=new_operation_id(),
                    elapsed_ms=elapsed_ms,
                    platform=self.runtime.platform,
                    container_backend=self.runtime.container_backend,
                    client_version=self.runtime.client_version,
                    adapter_version=self.runtime.capture_adapter_version,
                ),
            )
        except Exception:
            pass
        else:
            self._record(payload)
        if observation.status == "sealed":
            await self._observe_restore(request, observation)
        return observation

    async def run(
        self,
        mainline: Awaitable[object],
        *,
        initial_delay_sec: float = 300.0,
        interval_sec: float = 300.0,
        maximum_captures: int = 24,
    ) -> object:
        return await run_mainline_with_periodic_shadow_captures_v2(
            mainline,
            self.capture,
            initial_delay_sec=initial_delay_sec,
            interval_sec=interval_sec,
            maximum_captures=maximum_captures,
        )


async def run_mainline_with_optional_checkpoint_shadow_v2(
    mainline: Awaitable[object],
    *,
    activation: CheckpointActivationV2,
    coordinator_factory: Callable[[], CheckpointShadowCoordinatorV2],
    initial_delay_sec: float = 300.0,
    interval_sec: float = 300.0,
    maximum_captures: int = 24,
) -> object:
    """Strict OFF no-op and fail-open construction boundary."""

    if not activation.capture_enabled:
        return await mainline
    if activation.authoritative:
        close = getattr(mainline, "close", None)
        if callable(close):
            close()
        raise CheckpointV2ProtocolError(
            "authoritative checkpoint assignment requires the V2 owner state machine"
        )
    try:
        coordinator = coordinator_factory()
    except Exception:
        return await mainline
    return await coordinator.run(
        mainline,
        initial_delay_sec=initial_delay_sec,
        interval_sec=interval_sec,
        maximum_captures=maximum_captures,
    )


__all__ = [
    "CheckpointShadowCoordinatorV2",
    "CheckpointShadowObservationSinkV2",
    "CheckpointShadowRuntimeFactsV2",
    "run_mainline_with_optional_checkpoint_shadow_v2",
]
