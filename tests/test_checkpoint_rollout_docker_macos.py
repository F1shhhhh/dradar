from __future__ import annotations

import asyncio
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from dradar.pier_checkpoint import DurableCheckpoint


_IMAGE_ENV = "DRADAR_CHECKPOINT_TEST_IMAGE"


def _run(command: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _require_orbstack_image() -> str:
    if sys.platform != "darwin":
        pytest.skip("the release gate targets macOS bind mounts")
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    info = _run(["docker", "info", "--format", "{{.OperatingSystem}}"])
    if info.returncode != 0 or "orbstack" not in info.stdout.lower():
        pytest.skip("the release gate requires OrbStack")
    image = os.environ.get(_IMAGE_ENV, "").strip()
    if not image:
        pytest.skip(f"set {_IMAGE_ENV} to an existing local task image")
    if _run(["docker", "image", "inspect", image]).returncode != 0:
        pytest.skip("the configured local task image is unavailable")
    return image


def test_real_orbstack_disabled_runtime_never_touches_agent_bind() -> None:
    image = _require_orbstack_image()
    fixture_parent = Path.home() / ".cache" / "dradar-checkpoint-tests"
    fixture_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fixture_parent.chmod(0o700)

    with tempfile.TemporaryDirectory(
        prefix="orbstack-rollout-disabled-", dir=fixture_parent,
    ) as temporary:
        trial = Path(temporary) / "trial"
        agent_dir = trial / "agent"
        agent_dir.mkdir(parents=True, mode=0o700)
        agent_dir.chmod(0o777)
        container = f"dradar-rollout-disabled-{uuid.uuid4().hex[:12]}"
        started = _run([
            "docker", "run", "--detach", "--rm", "--name", container,
            "--network", "none",
            "--mount",
            f"type=bind,src={agent_dir.resolve()},dst=/logs/agent",
            "--entrypoint", "/bin/bash", image, "--noprofile", "--norc",
            "-c", "trap 'exit 0' TERM INT; while :; do sleep 3600 & wait $!; done",
        ])
        assert started.returncode == 0, started.stderr[-2000:]
        try:
            before = _run([
                "docker", "exec", container, "/usr/bin/stat", "-c", "%u:%g:%a",
                "/logs/agent",
            ])
            assert before.returncode == 0, before.stderr[-2000:]
            manager = DurableCheckpoint(
                logs_dir=agent_dir,
                enabled=False,
                assignment_id="rollout-disabled-probe",
                task_id="no-model",
                model="none",
                effort="low",
                harness="release-gate",
                provider="none",
                agent_version="0.5.97",
            )

            async def scenario() -> None:
                assert await manager.start(object(), object(), {}) is None
                await manager.finish_durably(
                    object(), object(), {}, completed=False,
                    failure=RuntimeError("no-model release gate"),
                )

            asyncio.run(scenario())
            after = _run([
                "docker", "exec", container, "/usr/bin/stat", "-c", "%u:%g:%a",
                "/logs/agent",
            ])
            assert after.returncode == 0, after.stderr[-2000:]
            assert after.stdout == before.stdout
            assert stat.S_IMODE(agent_dir.stat().st_mode) == 0o777
            assert not (agent_dir / ".dradar-checkpoint-staging").exists()
            assert not (trial / "checkpoint").exists()
        finally:
            _run(["docker", "rm", "--force", container], timeout=30)
