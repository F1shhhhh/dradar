from __future__ import annotations

import asyncio
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from dradar.pier_checkpoint import (
    CheckpointError,
    CheckpointIncompatibleError,
    DurableCheckpoint,
    StatePath,
    _snapshot_script,
)


BASE_COMMIT = "a" * 40


class FakeAgent:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec_as_agent(self, _environment, *, command, env):
        del env
        self.commands.append(command)
        stdout = BASE_COMMIT + "\n" if "rev-parse HEAD" in command else ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


class FakeEnvironment:
    def __init__(self) -> None:
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []

    async def upload_file(self, source: Path | str, target: str) -> None:
        self.uploaded_files.append((Path(source), target))

    async def upload_dir(self, source: Path | str, target: str) -> None:
        self.uploaded_dirs.append((Path(source), target))


class FailingFinishAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.fail_finish = False

    async def exec_as_agent(self, environment, *, command, env):
        if self.fail_finish and "checkpoint/stop" in command:
            raise RuntimeError("snapshot did not stop")
        return await super().exec_as_agent(environment, command=command, env=env)


def _manager(
    logs_dir: Path,
    *,
    previous: Path | None = None,
    sensitive_values: tuple[str | bytes, ...] = (),
    state_paths: tuple[StatePath, ...] = (
        StatePath("sessions", "/tmp/provider/sessions"),
    ),
) -> DurableCheckpoint:
    return DurableCheckpoint(
        logs_dir=logs_dir,
        enabled=True,
        assignment_id="assignment-123",
        task_id="task-1",
        model="model-1",
        effort="high",
        resume_generation=1 if previous else 0,
        checkpoint_path=str(previous) if previous else None,
        harness="test-harness",
        provider="test-provider",
        agent_version="1.2.3",
        state_paths=state_paths,
        sensitive_values=sensitive_values,
    )


def _manifest(**overrides: object) -> dict:
    value = {
        "schema_version": 1,
        "checkpoint_id": "checkpoint-123",
        "assignment_id": "assignment-123",
        "phase": "paused",
        "created_at": "2026-08-22T00:00:00Z",
        "updated_at": "2026-08-22T00:00:01Z",
        "task_id": "task-1",
        "model": "model-1",
        "effort": "high",
        "harness": "test-harness",
        "provider": "test-provider",
        "agent_version": "1.2.3",
        "base_commit": BASE_COMMIT,
        "resume_generation": 0,
    }
    value.update(overrides)
    return value


def test_checkpoint_manifest_contains_runtime_identity_and_no_secret(
    tmp_path: Path,
) -> None:
    secret = "arbitrary-provider-value-without-a-known-prefix"
    manager = _manager(tmp_path / "agent", sensitive_values=(secret,))
    agent = FakeAgent()
    asyncio.run(manager.start(agent, FakeEnvironment(), {}))

    manifest = json.loads(
        (tmp_path / "agent/checkpoint/checkpoint.json").read_text()
    )
    assert manifest["harness"] == "test-harness"
    assert manifest["provider"] == "test-provider"
    assert manifest["agent_version"] == "1.2.3"
    assert secret not in json.dumps(manifest)
    assert secret not in (tmp_path / "agent/checkpoint/snapshot.sh").read_text()


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"assignment_id": "another-assignment"}, "assignment_id"),
        ({"task_id": "another-task"}, "task_id"),
        ({"model": "another-model"}, "model"),
        ({"effort": "low"}, "effort"),
        ({"harness": "another-harness"}, "harness"),
        ({"provider": "another-provider"}, "provider"),
        ({"agent_version": "9.9.9"}, "agent_version"),
        ({"base_commit": "b" * 40}, "base_commit"),
    ],
)
def test_checkpoint_restore_rejects_runtime_identity_mismatch(
    tmp_path: Path, override: dict, field: str,
) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest(**override)))

    with pytest.raises(CheckpointIncompatibleError, match=field):
        asyncio.run(
            _manager(tmp_path / "agent", previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )


def test_checkpoint_restore_rejects_corrupt_manifest(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text("not-json")

    with pytest.raises(CheckpointError, match="unreadable"):
        asyncio.run(
            _manager(tmp_path / "agent", previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )


@pytest.mark.parametrize("marker", ["invalid-secret", "invalid-snapshot"])
def test_checkpoint_restore_rejects_invalid_marker(
    tmp_path: Path, marker: str,
) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    (previous / marker).write_text("rejected\n", encoding="utf-8")

    with pytest.raises(CheckpointError, match="checkpoint"):
        asyncio.run(
            _manager(tmp_path / "agent", previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )


def test_checkpoint_restore_rejects_symlinked_root(tmp_path: Path) -> None:
    previous = tmp_path / "previous-real"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    linked = tmp_path / "previous-linked"
    linked.symlink_to(previous, target_is_directory=True)

    with pytest.raises(CheckpointError, match="root is a symlink"):
        asyncio.run(
            _manager(tmp_path / "agent", previous=linked).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )


def test_identity_mismatch_marks_only_new_attempt_incompatible(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    original = _manifest(provider="another-provider")
    (previous / "checkpoint.json").write_text(json.dumps(original))
    current_logs = tmp_path / "current-agent"

    with pytest.raises(CheckpointIncompatibleError, match="provider"):
        asyncio.run(
            _manager(current_logs, previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )

    assert json.loads((previous / "checkpoint.json").read_text()) == original
    current = json.loads(
        (current_logs / "checkpoint/checkpoint.json").read_text()
    )
    assert current["phase"] == "incompatible"
    assert current["provider"] == "test-provider"


def test_checkpoint_finish_deletes_exact_credential_and_marks_invalid(
    tmp_path: Path,
) -> None:
    secret = b"provider-secret-value-that-does-not-match-generic-regex"
    manager = _manager(tmp_path / "agent", sensitive_values=(secret,))
    agent = FakeAgent()
    environment = FakeEnvironment()
    asyncio.run(manager.start(agent, environment, {}))
    state = tmp_path / "agent/checkpoint/provider-state/sessions"
    state.mkdir(parents=True)
    (state / "session.jsonl").write_bytes(b"prefix:" + secret + b":suffix")

    asyncio.run(
        manager.finish(
            agent, environment, {}, completed=False, failure=RuntimeError("stop"),
        )
    )

    manifest = json.loads(
        (tmp_path / "agent/checkpoint/checkpoint.json").read_text()
    )
    assert manifest["phase"] == "invalid"
    assert manifest["failure_type"] == "CheckpointSecretDetected"
    assert not (tmp_path / "agent/checkpoint/provider-state").exists()


def test_snapshot_stop_failure_discards_all_payload_artifacts(
    tmp_path: Path,
) -> None:
    secret = b"provider-secret-value-not-covered-by-generic-pattern"
    manager = _manager(tmp_path / "agent", sensitive_values=(secret,))
    agent = FailingFinishAgent()
    environment = FakeEnvironment()
    asyncio.run(manager.start(agent, environment, {}))
    checkpoint = manager.host_dir
    (checkpoint / "workspace.patch").write_bytes(secret)
    (checkpoint / "untracked.tar.gz").write_bytes(secret)
    state = checkpoint / "provider-state/sessions"
    state.mkdir(parents=True)
    (state / "wire.jsonl").write_bytes(secret)
    agent.fail_finish = True

    asyncio.run(
        manager.finish(
            agent, environment, {}, completed=False, failure=KeyboardInterrupt(),
        )
    )

    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    assert manifest["phase"] == "invalid"
    assert manifest["failure_type"] == "CheckpointSnapshotInvalid"
    assert (checkpoint / "invalid-snapshot").is_file()
    assert not (checkpoint / "workspace.patch").exists()
    assert not (checkpoint / "untracked.tar.gz").exists()
    assert not (checkpoint / "provider-state").exists()


def test_checkpoint_restore_rejects_unsafe_untracked_archive(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    archive_path = previous / "untracked.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(CheckpointError, match="unsafe member"):
        asyncio.run(
            _manager(tmp_path / "agent", previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )


def test_checkpoint_restores_only_allowlisted_state_and_session_id(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous"
    state = previous / "provider-state/sessions"
    state.mkdir(parents=True)
    (state / "events.jsonl").write_text("{}\n")
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    (previous / "session-id").write_text("session-12345678\n")
    environment = FakeEnvironment()

    session_id = asyncio.run(
        _manager(tmp_path / "agent", previous=previous).start(
            FakeAgent(), environment, {},
        )
    )

    assert session_id == "session-12345678"
    assert environment.uploaded_dirs == [
        (state, "/tmp/provider/sessions"),
    ]


def test_checkpoint_restores_allowlisted_file_state(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    index = previous / "provider-state/session-index"
    index.parent.mkdir()
    index.write_text('{"sessionId":"session-12345678"}\n', encoding="utf-8")
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    environment = FakeEnvironment()

    asyncio.run(
        _manager(
            tmp_path / "agent",
            previous=previous,
            state_paths=(
                StatePath("session-index", "/tmp/provider/session_index.jsonl"),
            ),
        ).start(FakeAgent(), environment, {})
    )

    assert environment.uploaded_files == [
        (index, "/tmp/provider/session_index.jsonl"),
    ]


def test_checkpoint_finish_uses_snapshot_session_sidecar(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "agent")
    agent = FakeAgent()
    environment = FakeEnvironment()
    asyncio.run(manager.start(agent, environment, {}))
    (manager.host_dir / "session-id").write_text(
        "session-12345678\n", encoding="utf-8",
    )

    asyncio.run(
        manager.finish(
            agent, environment, {}, completed=False, failure=KeyboardInterrupt(),
        )
    )

    manifest = json.loads((manager.host_dir / "checkpoint.json").read_text())
    assert manifest["phase"] == "paused"
    assert manifest["session_id"] == "session-12345678"
    final_snapshot = agent.commands[-1]
    assert "touch /logs/agent/checkpoint/stop" in final_snapshot
    assert 'snapshot_state=$(awk \'{print $3}\'' in final_snapshot
    assert '[ "$snapshot_state" != Z ]' in final_snapshot
    assert "snapshot_group_running" in final_snapshot
    assert 'kill -TERM -- "-$snapshot_pid"' in final_snapshot
    assert 'kill -KILL -- "-$snapshot_pid"' in final_snapshot
    assert "DRADAR_SNAPSHOT_TOKEN" in final_snapshot
    assert final_snapshot.index("stop") < final_snapshot.index("--once")


def test_checkpoint_finish_marks_restore_rejection_incompatible(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    agent = FakeAgent()
    environment = FakeEnvironment()
    asyncio.run(manager.start(agent, environment, {}))

    asyncio.run(
        manager.finish(
            agent,
            environment,
            {},
            completed=False,
            failure=CheckpointIncompatibleError("native session missing"),
        )
    )

    manifest = json.loads((manager.host_dir / "checkpoint.json").read_text())
    assert manifest["phase"] == "incompatible"
    assert manifest["failure_type"] == "CheckpointIncompatibleError"


def test_restore_identity_rejection_never_requires_snapshot_script(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest(
        provider="another-provider",
    )))
    manager = _manager(tmp_path / "agent", previous=previous)
    agent = FakeAgent()
    environment = FakeEnvironment()

    with pytest.raises(CheckpointIncompatibleError) as caught:
        asyncio.run(manager.start(agent, environment, {}))
    asyncio.run(
        manager.finish(
            agent,
            environment,
            {},
            completed=False,
            failure=caught.value,
        )
    )

    manifest = json.loads((manager.host_dir / "checkpoint.json").read_text())
    assert manifest["phase"] == "incompatible"
    assert manifest["failure_type"] == "CheckpointIncompatibleError"
    assert not (manager.host_dir / "invalid-secret").exists()
    assert not (manager.host_dir / "invalid-snapshot").exists()
    assert not any("--once" in command for command in agent.commands)


def test_snapshot_script_captures_workspace_state_and_session(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    tracked = worktree / "tracked.txt"
    tracked.write_text("before\n")
    subprocess.run(["git", "-C", str(worktree), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(worktree), "-c", "user.name=DRadar Test",
            "-c", "user.email=test@dradar.invalid", "commit", "-qm", "base",
        ],
        check=True,
    )
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("after\n")
    (worktree / "new.txt").write_text("untracked\n")
    state = tmp_path / "native-sessions"
    state.mkdir()
    (state / "wire.jsonl").write_text("{}\n")
    session_id = tmp_path / "session-id-source"
    session_id.write_text("session-abcdefgh\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "base_commit").write_text(base + "\n")
    script = checkpoint / "snapshot.sh"
    script.write_text(
        _snapshot_script(
            checkpoint_dir=str(checkpoint),
            workdir=str(worktree),
            interval_sec=30,
            state_paths=(StatePath("sessions", str(state)),),
            session_probe=f"cat {session_id}",
        )
    )

    subprocess.run(["sh", str(script), "--once"], check=True)

    assert "tracked.txt" in (checkpoint / "workspace.patch").read_text()
    with tarfile.open(checkpoint / "untracked.tar.gz", "r:gz") as archive:
        assert "new.txt" in archive.getnames()
    assert (checkpoint / "provider-state/sessions/wire.jsonl").read_text() == "{}\n"
    assert (checkpoint / "session-id").read_text() == "session-abcdefgh\n"
    assert (checkpoint / "last_heartbeat").is_file()


def test_snapshot_once_fails_when_lock_is_held(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    (worktree / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(worktree), "-c", "user.name=DRadar Test",
            "-c", "user.email=test@dradar.invalid", "commit", "-qm", "base",
        ],
        check=True,
    )
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "base_commit").write_text(base + "\n", encoding="utf-8")
    (checkpoint / "snapshot.lock").mkdir()
    script = checkpoint / "snapshot.sh"
    script.write_text(
        _snapshot_script(
            checkpoint_dir=str(checkpoint),
            workdir=str(worktree),
            interval_sec=30,
            state_paths=(),
            session_probe=None,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(["sh", str(script), "--once"], check=False)

    assert result.returncode == 75
    assert not (checkpoint / "last_heartbeat").exists()
