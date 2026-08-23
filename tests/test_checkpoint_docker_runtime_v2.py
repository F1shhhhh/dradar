from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from dradar.checkpoint_docker_runtime_v2 import (
    DockerCliCheckpointEnvironmentV2,
    DockerCliDisposableCheckpointRestorerV2,
    DockerContainerIdentityV2,
    _material_native_artifacts_v2,
    _native_session_id_v2,
    discover_pier_container_v2,
)
from dradar.checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from dradar.checkpoint_runtime_v2 import CheckpointDataPlaneError


CONTAINER_A = "a" * 64
CONTAINER_B = "b" * 64
IMAGE = "sha256:" + "c" * 64


def _inspection(container_id: str, mount: Path, *, running: bool = True) -> dict:
    return {
        "Id": container_id,
        "Image": IMAGE,
        "State": {"Running": running},
        "Mounts": [{"Type": "bind", "Source": str(mount)}],
    }


def test_discovery_requires_one_exact_job_bound_container(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "jobs" / "assignment"
    logs = job / "trial" / "agent"
    logs.mkdir(parents=True)

    def run(arguments, **_kwargs):
        assert arguments[:3] == ["ps", "--quiet", "--no-trunc"]
        return SimpleNamespace(stdout=CONTAINER_A + "\n")

    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2._run_docker_v2", run,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2._inspect_containers_v2",
        lambda ids, stage: [_inspection(ids[0], logs)],
    )
    assert discover_pier_container_v2(job) == DockerContainerIdentityV2(
        CONTAINER_A, IMAGE,
    )


def test_discovery_rejects_ambiguous_job_ownership(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2._run_docker_v2",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=f"{CONTAINER_A}\n{CONTAINER_B}\n",
        ),
    )
    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2._inspect_containers_v2",
        lambda ids, stage: [_inspection(value, job) for value in ids],
    )
    with pytest.raises(CheckpointDataPlaneError) as exc:
        discover_pier_container_v2(job)
    assert exc.value.code == "task_container_ambiguous"


def test_pinned_environment_rejects_container_image_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    environment = DockerCliCheckpointEnvironmentV2(
        DockerContainerIdentityV2(CONTAINER_A, IMAGE),
        job_root=job,
    )
    changed = _inspection(CONTAINER_A, job)
    changed["Image"] = "sha256:" + "d" * 64
    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2._inspect_containers_v2",
        lambda ids, stage: [changed],
    )
    with pytest.raises(CheckpointDataPlaneError) as exc:
        environment._validate(stage="capture")
    assert exc.value.code == "container_identity_changed"


def test_disposable_restore_cleanup_failure_is_not_reported_as_verified(
    monkeypatch,
) -> None:
    capture_id = "capture-0001"
    exporter = SimpleNamespace(
        contract=SimpleNamespace(
            restorer_version="restore-adapter/1",
            checkpoint_abi="dradar-checkpoint-v2/codex/1",
        ),
        _runtime={
            capture_id: SimpleNamespace(
                identity=SimpleNamespace(image_id=IMAGE),
                base_commit="f" * 40,
            ),
        },
    )
    calls = []

    def run(arguments, **_kwargs):
        calls.append(arguments)
        if arguments[0] == "run":
            return SimpleNamespace(stdout=CONTAINER_A + "\n")
        raise CheckpointDataPlaneError("cleanup", "docker_transport_failed")

    class Restorer:
        def __init__(self, *_args, **_kwargs):
            pass

        async def restore_offline(self, _request):
            return SimpleNamespace(verified=True)

    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2._run_docker_v2", run,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_docker_runtime_v2.PierContainerCheckpointRestorerV2",
        Restorer,
    )
    request = SimpleNamespace(
        published=SimpleNamespace(capture_id=capture_id),
    )
    with pytest.raises(CheckpointDataPlaneError) as exc:
        asyncio.run(
            DockerCliDisposableCheckpointRestorerV2(exporter).restore_offline(
                request,
            )
        )
    assert exc.value.code == "restore_cleanup_failed"
    assert calls[-1] == ["rm", "-f", CONTAINER_A]


@pytest.mark.parametrize(
    ("harness", "provider", "stdout", "expected"),
    [
        (
            "codex", "openai",
            '{"type":"thread.started","thread_id":"thread-0001"}\n',
            "thread-0001",
        ),
        ("dsh", "deepseek", "dsh-session-0001\n", "dsh-session-0001"),
        (
            "zcode", "bigmodel-coding-plan", "zcode-session-0001\n",
            "zcode-session-0001",
        ),
        (
            "kimi-code", "kimi-subscription",
            '{"sessionId":"session_123e4567-e89b-12d3-a456-426614174000",'
            '"sessionDir":"/tmp/dradar-kimi-home/sessions/'
            'session_123e4567-e89b-12d3-a456-426614174000",'
            '"workDir":"/app"}\n',
            "session_123e4567-e89b-12d3-a456-426614174000",
        ),
    ],
)
def test_native_session_probes_are_harness_specific_and_bounded(
    harness: str,
    provider: str,
    stdout: str,
    expected: str,
) -> None:
    class Environment:
        async def exec(self, **_kwargs):
            return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    contract = checkpoint_adapter_contract_v2(harness, provider)
    assert asyncio.run(_native_session_id_v2(Environment(), contract)) == expected


def test_kimi_session_probe_rejects_traversal_alias() -> None:
    session = "session_123e4567-e89b-12d3-a456-426614174000"
    raw = (
        '{"sessionId":"' + session + '","sessionDir":"/tmp/'
        'dradar-kimi-home/sessions/../secrets/' + session
        + '","workDir":"/app"}\n'
    )

    class Environment:
        async def exec(self, **_kwargs):
            return SimpleNamespace(return_code=0, stdout=raw, stderr="")

    contract = checkpoint_adapter_contract_v2(
        "kimi-code", "kimi-subscription",
    )
    assert asyncio.run(_native_session_id_v2(Environment(), contract)) is None


def test_material_native_probe_requires_each_contract_artifact() -> None:
    class Environment:
        async def exec(self, *, command, **_kwargs):
            return SimpleNamespace(
                return_code=0 if "session" in command else 1,
                stdout="",
                stderr="",
            )

    contract = checkpoint_adapter_contract_v2(
        "kimi-code", "kimi-subscription",
    )
    present = asyncio.run(_material_native_artifacts_v2(Environment(), contract))
    assert present == frozenset({"sessions", "session-index"})
