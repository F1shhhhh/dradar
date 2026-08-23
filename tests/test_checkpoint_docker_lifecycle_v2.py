"""Opt-in real Docker/OrbStack lifecycle gate for the V2 shared runtime."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from dradar.checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from dradar.checkpoint_docker_runtime_v2 import (
    DockerCliLazyCheckpointExporterV2,
    docker_container_backend_v2,
)
from dradar.checkpoint_pier_transport_v2 import (
    PierContainerCheckpointExporterV2,
    PierContainerCheckpointRestorerV2,
)
from dradar.checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneV2,
    CheckpointObservationRuntimeV2,
    CheckpointRestoreRequestV2,
    checkpoint_observation_payload_v2,
    checkpoint_restore_observation_payload_v2,
)
from dradar.checkpoint_v2 import negotiate_checkpoint_activation_v2
from dradar.telemetry import platform_family


IMAGE = os.environ.get("DRADAR_CHECKPOINT_V2_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(
    not IMAGE,
    reason="set DRADAR_CHECKPOINT_V2_DOCKER_IMAGE for the real container gate",
)


@dataclass
class _Result:
    return_code: int
    stdout: str
    stderr: str


class DockerPierEnvironment:
    default_user = "1000:1000"

    def __init__(self, container: str):
        self.container = container

    async def upload_file(self, source: Path, destination: str):
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "cp", os.fspath(source), f"{self.container}:{destination}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("docker upload failed")

    async def download_file(self, source: str, destination: Path):
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "cp", f"{self.container}:{source}", os.fspath(destination)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("docker download failed")

    async def exec(self, *, command, user, env, cwd, timeout_sec):
        arguments = ["docker", "exec", "-u", user, "-w", cwd]
        for key, value in sorted(env.items()):
            arguments.extend(("-e", f"{key}={value}"))
        arguments.extend((self.container, "/bin/sh", "-c", command))
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("docker root command timed out") from exc
        return _Result(result.returncode, result.stdout, result.stderr)


def _docker(container: str, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", "-u", "root", container, "/bin/sh", "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_container(container: str) -> None:
    started = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", container,
         "--entrypoint", "/bin/sh", IMAGE, "-c", "sleep 300"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert started.stdout.strip()


def _materialize_native_state(container: str, contract) -> None:
    commands = ["set -eu", "umask 077"]
    for artifact in contract.artifacts:
        source = shlex.quote(artifact.source_path)
        if artifact.kind == "directory":
            commands.extend((
                f"mkdir -p {source}",
                f"printf 'native-state\\n' > {source}/state.bin",
            ))
        else:
            parent = shlex.quote(os.path.dirname(artifact.source_path))
            commands.extend((
                f"mkdir -p {parent}",
                f"printf 'native-state\\n' > {source}",
            ))
    _docker(container, "; ".join(commands))


@pytest.mark.parametrize(
    ("harness", "provider"),
    [
        ("codex", "openai"),
        ("codex", "deepseek"),
        ("dsh", "deepseek"),
        ("kimi-code", "kimi-subscription"),
        ("zcode", "bigmodel-coding-plan"),
    ],
)
def test_real_container_native_capture_seal_download_restore(
    tmp_path: Path,
    harness: str,
    provider: str,
):
    assert IMAGE is not None
    container_backend = docker_container_backend_v2()
    platform = platform_family()
    container = f"dradar-checkpoint-v2-{uuid.uuid4().hex[:12]}"
    _start_container(container)
    try:
        _docker(container, """
set -eu
umask 077
mkdir -p /app /tmp/codex-home/sessions
git -C /app init -q
printf 'before\n' > /app/tracked.txt
git -C /app add tracked.txt
GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.invalid \
GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.invalid \
  git -C /app commit -q -m base
printf 'after\n' > /app/tracked.txt
printf 'untracked\n' > /app/new.txt
chmod 0700 /app
""")
        base = _docker(container, "git -C /app rev-parse HEAD").stdout.strip()
        environment = DockerPierEnvironment(container)
        contract = checkpoint_adapter_contract_v2(harness, provider)
        _materialize_native_state(container, contract)
        exporter = PierContainerCheckpointExporterV2(
            environment,
            contract=contract,
            base_commit=base,
            session_id="thread-0001",
        )
        request = CheckpointCaptureRequestV2(
            checkpoint_id="checkpoint-0001",
            checkpoint_lineage_id="lineage-0001",
            snapshot_generation=1,
            capture_id="capture-0001",
            identity_fingerprint="a" * 64,
            checkpoint_abi=contract.checkpoint_abi,
            recovery_capability="NATIVE_VALID",
            native_state_schema=contract.native_state_schema,
            captured_at="2026-08-23T12:00:00+00:00",
        )
        plane = CheckpointDataPlaneV2(
            activation=negotiate_checkpoint_activation_v2(
                local_mode="restore-test", server_mode="restore-test",
            ),
            storage_root=tmp_path / "host",
        )
        captured = asyncio.run(plane.observe_capture(request, exporter))
        assert captured.status == "sealed", captured
        assert captured.published is not None

        capture_wire = checkpoint_observation_payload_v2(
            request,
            captured,
            plane.activation,
            CheckpointObservationRuntimeV2(
                assignment_id="assignment-0001",
                operation_id="capture-op-0001",
                elapsed_ms=100,
                platform=platform,
                container_backend=container_backend,
                client_version="0.5.97",
                adapter_version=exporter.adapter_version,
            ),
        )
        assert capture_wire["observation_kind"] == "capture"
        assert capture_wire["authoritative"] is False
        # CURRENT may select a shadow generation for the next offline test;
        # local selection carries no assignment or paid-execution authority.
        assert capture_wire["selected_local"] is True

        # Preserve only the immutable base commit, then replace the container.
        # The restored state must come from the host-private sealed archive,
        # not from a still-running capture container or its writable layer.
        _docker(container, "git -C /app bundle create /tmp/base.bundle HEAD")
        bundle = tmp_path / "base.bundle"
        subprocess.run(
            ["docker", "cp", f"{container}:/tmp/base.bundle", os.fspath(bundle)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["docker", "rm", "-f", container],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _start_container(container)
        subprocess.run(
            ["docker", "cp", os.fspath(bundle), f"{container}:/tmp/base.bundle"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _docker(
            container,
            "git clone -q /tmp/base.bundle /restore && chmod 0700 /restore",
        )
        restorer = PierContainerCheckpointRestorerV2(
            environment,
            contract=contract,
            base_commit=base,
            worktree_path="/restore",
        )
        restore_request = CheckpointRestoreRequestV2(
            published=captured.published,
            expected_identity_fingerprint=request.identity_fingerprint,
            restore_id="restore-0001",
        )
        restored = asyncio.run(plane.observe_offline_restore(
            restore_request, restorer,
        ))
        assert restored.status == "verified"
        assert restored.paid_execution_authorized is False
        restore_wire = checkpoint_restore_observation_payload_v2(
            request,
            restore_request,
            restored,
            plane.activation,
            CheckpointObservationRuntimeV2(
                assignment_id="assignment-0001",
                operation_id="restore-op-0001",
                elapsed_ms=200,
                platform=platform,
                container_backend=container_backend,
                client_version="0.5.97",
                adapter_version=restorer.adapter_version,
            ),
        )
        assert restore_wire["observation_kind"] == "restore"
        assert restore_wire["source_capture_id"] == request.capture_id
        assert restore_wire["paid_execution_started"] is False
        assert restore_wire["authoritative"] is False
        result = _docker(
            container,
            "test \"$(cat /restore/tracked.txt)\" = after && "
            "test \"$(cat /restore/new.txt)\" = untracked && "
            "stat -c '%a' /run/dradar-checkpoint-v2",
        )
        assert result.stdout.strip() == "700"
        for artifact in contract.artifacts:
            restored_path = (
                "/run/dradar-checkpoint-v2/checkpoint-0001/restore/"
                f"restore-0001/state/{artifact.name}"
            )
            probe = "-d" if artifact.kind == "directory" else "-f"
            exists = subprocess.run(
                [
                    "docker", "exec", "-u", "root", container,
                    "/usr/bin/test", probe, restored_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).returncode == 0
            assert exists is artifact.restore
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def test_live_job_discovery_and_fresh_image_restore_for_native_kimi(
    tmp_path: Path,
) -> None:
    assert IMAGE is not None
    seed = f"dradar-checkpoint-v2-seed-{uuid.uuid4().hex[:12]}"
    live = f"dradar-checkpoint-v2-live-{uuid.uuid4().hex[:12]}"
    image_tag = f"dradar-checkpoint-v2-test:{uuid.uuid4().hex[:16]}"
    job_root = tmp_path / "job"
    job_root.mkdir(mode=0o700)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", seed,
         "--entrypoint", "/bin/sh", IMAGE, "-c", "sleep 300"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _docker(seed, """
set -eu
umask 077
mkdir -p /app
git -C /app init -q
printf 'before\n' > /app/tracked.txt
git -C /app add tracked.txt
GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.invalid \
GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.invalid \
  git -C /app commit -q -m base
""")
        subprocess.run(
            ["docker", "commit", seed, image_tag],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["docker", "rm", "-f", seed],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                "docker", "run", "-d", "--rm", "--name", live,
                "--mount", f"type=bind,src={job_root},dst=/logs",
                "--entrypoint", "/bin/sh", image_tag, "-c", "sleep 300",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        session = "session_123e4567-e89b-12d3-a456-426614174000"
        _docker(live, f"""
set -eu
umask 077
printf 'after\n' > /app/tracked.txt
printf 'untracked\n' > /app/new.txt
mkdir -p /tmp/dradar-kimi-home/sessions/{session}/agents/main
mkdir -p /logs/agent
printf '{{"step":2}}\n' > /tmp/dradar-kimi-home/sessions/{session}/state.json
printf '{{}}\n' > /tmp/dradar-kimi-home/sessions/{session}/agents/main/wire.jsonl
printf '%s\n' '{{"sessionId":"{session}","sessionDir":"/tmp/dradar-kimi-home/sessions/{session}","workDir":"/app"}}' > /tmp/dradar-kimi-home/session_index.jsonl
printf 'diagnostic-only\n' > /logs/agent/kimi-code.jsonl
""")
        contract = checkpoint_adapter_contract_v2(
            "kimi-code", "kimi-subscription",
        )
        exporter = DockerCliLazyCheckpointExporterV2(
            job_root=job_root,
            contract=contract,
            ready_timeout_sec=0,
        )
        capability, schema = asyncio.run(exporter.recovery_facts())
        assert capability == "NATIVE_VALID"
        assert schema == contract.native_state_schema
        request = CheckpointCaptureRequestV2(
            checkpoint_id="checkpoint-live-kimi",
            checkpoint_lineage_id="lineage-live-kimi",
            snapshot_generation=1,
            capture_id="capture-live-kimi",
            identity_fingerprint="a" * 64,
            checkpoint_abi=contract.checkpoint_abi,
            recovery_capability=capability,
            native_state_schema=schema,
            captured_at="2026-08-23T12:00:00+00:00",
        )
        plane = CheckpointDataPlaneV2(
            activation=negotiate_checkpoint_activation_v2(
                local_mode="restore-test", server_mode="restore-test",
            ),
            storage_root=tmp_path / "host-live",
        )
        captured = asyncio.run(plane.observe_capture(request, exporter))
        assert captured.status == "sealed", captured
        assert captured.published is not None
        assert not (
            captured.published.payload_root / "provider-state/stream"
        ).exists()
        restored = asyncio.run(plane.observe_offline_restore(
            CheckpointRestoreRequestV2(
                published=captured.published,
                expected_identity_fingerprint=request.identity_fingerprint,
                restore_id="restore-live-kimi",
            ),
            exporter.restorer(),
        ))
        assert restored.status == "verified"
        assert restored.evidence is not None
        assert restored.evidence.paid_execution_started is False
    finally:
        for owned_container in (live, seed):
            subprocess.run(
                ["docker", "rm", "-f", owned_container],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        subprocess.run(
            ["docker", "image", "rm", image_tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
