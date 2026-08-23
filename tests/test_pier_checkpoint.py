from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import tarfile
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

import dradar.pier_checkpoint as checkpoint_mod
from dradar.pier_checkpoint import (
    AgentLogStore,
    AgentIdentity,
    CheckpointError,
    CheckpointIncompatibleError,
    CheckpointV2PaidExecutionGate,
    CheckpointV2PreProviderBarrier,
    DurableCheckpoint,
    StatePath,
    UnsafeAgentLog,
    _capture_script,
    _snapshot_payload_dir,
    _supervisor_script,
    _validate_archive,
    _validate_regular_tree,
)


BASE_COMMIT = "a" * 40


def _private_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_checkpoint_v2_paid_gate_blocks_until_exact_grant(tmp_path: Path) -> None:
    gate_dir = tmp_path / "paid-gate"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    contract = {
        "schema": "dradar-checkpoint-paid-gate-contract-v2",
        "assignment_id": "assignment-0001",
        "gate_nonce": "1" * 32,
        "action": "fresh",
        "session_id": "session-0001",
        "owner_epoch": 3,
        "reconcile_operation_id": "reconcile-operation-0001",
        "job_root": str(tmp_path / "work" / "jobs" / "aassignment-0001"),
    }
    _private_json(gate_dir / "contract.json", contract)
    gate = CheckpointV2PaidExecutionGate(str(gate_dir))

    async def exercise() -> None:
        waiter = asyncio.create_task(gate.authorize_provider_start(timeout_sec=2))
        request_path = gate_dir / "request.json"
        for _index in range(100):
            if request_path.exists():
                break
            await asyncio.sleep(0.01)
        assert request_path.is_file()
        assert not waiter.done()
        request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
        _private_json(gate_dir / "grant.json", {
            "schema": "dradar-checkpoint-paid-gate-grant-v2",
            "assignment_id": "assignment-0001",
            "gate_nonce": "1" * 32,
            "request_sha256": request_sha256,
            "owner_epoch": 3,
            "usage_segment_id": "usage-segment-0001",
            "paid_execution_authorized": True,
        })
        await waiter
        # Provider retries inside the same paid epoch do not ask for a second
        # owner transition or create another gate request.
        await gate.authorize_provider_start(timeout_sec=1)

    asyncio.run(exercise())


def test_checkpoint_v2_paid_gate_rejects_unbound_grant(tmp_path: Path) -> None:
    gate_dir = tmp_path / "paid-gate"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    _private_json(gate_dir / "contract.json", {
        "schema": "dradar-checkpoint-paid-gate-contract-v2",
        "assignment_id": "assignment-0001",
        "gate_nonce": "1" * 32,
        "action": "fresh",
        "session_id": "session-0001",
        "owner_epoch": 3,
        "reconcile_operation_id": "reconcile-operation-0001",
        "job_root": str(tmp_path / "work" / "jobs" / "aassignment-0001"),
    })
    _private_json(gate_dir / "grant.json", {
        "schema": "dradar-checkpoint-paid-gate-grant-v2",
        "assignment_id": "assignment-0001",
        "gate_nonce": "1" * 32,
        "request_sha256": "0" * 64,
        "owner_epoch": 3,
        "usage_segment_id": "usage-segment-0001",
        "paid_execution_authorized": True,
    })
    with pytest.raises(CheckpointError, match="grant is invalid"):
        asyncio.run(
            CheckpointV2PaidExecutionGate(str(gate_dir)).authorize_provider_start(
                timeout_sec=1,
            )
        )


def _v2_restore_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "generation-00000000000000000001"
    payload = root / "payload"
    sessions = payload / "provider-state" / "sessions"
    sessions.mkdir(parents=True, mode=0o700)
    for directory in (root, payload, payload / "provider-state", sessions):
        directory.chmod(0o700)
    files = {
        "workspace.patch": b"diff --git a/a b/a\n",
        "progress.json": json.dumps({
            "schema": "dradar-checkpoint-adapter-progress-v2",
            "harness": "codex",
            "provider": "openai",
            "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
            "base_commit": BASE_COMMIT,
            "captured_at": "2026-08-23T12:00:00+00:00",
            "session_id_present": True,
            "native_artifacts": ["sessions"],
            "recovery_capability": "NATIVE_VALID",
            "workspace_patch_bytes": 22,
            "untracked_files": 0,
            "untracked_bytes": 0,
        }, sort_keys=True, separators=(",", ":")).encode() + b"\n",
        "session-id": b"thread-session-0001\n",
        "provider-state/sessions/state.json": b'{"turn":2}\n',
    }
    archive_path = payload / "untracked.tar.gz"
    with tarfile.open(archive_path, "w:gz"):
        pass
    files["untracked.tar.gz"] = archive_path.read_bytes()
    for relative, data in files.items():
        path = payload / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(data)
        path.chmod(0o600)
    directories = ["provider-state", "provider-state/sessions"]
    manifest_files = []
    total = 0
    for relative, data in sorted(files.items()):
        total += len(data)
        manifest_files.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mode": 0o600,
        })
    manifest = {
        "schema": "dradar-checkpoint-export-v2",
        "protocol_version": 2,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
        "checkpoint_id": "checkpoint-0001",
        "checkpoint_lineage_id": "lineage-0001",
        "snapshot_generation": 1,
        "capture_id": "capture-0001",
        "identity_fingerprint": "a" * 64,
        "recovery_capability": "NATIVE_VALID",
        "native_state_schema": "codex-session/1",
        "captured_at": "2026-08-23T12:00:00+00:00",
        "capture_storage": "container_native",
        "directories": directories,
        "files": manifest_files,
        "file_count": len(manifest_files),
        "total_bytes": total,
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
    ).encode() + b"\n"
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "manifest.json").chmod(0o600)
    return root, hashlib.sha256(manifest_bytes).hexdigest()


def test_checkpoint_v2_resume_barrier_restores_before_paid_grant(
    tmp_path: Path,
) -> None:
    restore_root, manifest_sha256 = _v2_restore_fixture(tmp_path)
    gate_dir = tmp_path / "paid-gate-resume"
    gate_dir.mkdir(mode=0o700)
    gate_dir.chmod(0o700)
    receipt_sha256 = "e" * 64
    _private_json(gate_dir / "contract.json", {
        "schema": "dradar-checkpoint-paid-gate-contract-v2",
        "assignment_id": "assignment-0001",
        "gate_nonce": "2" * 32,
        "action": "resume",
        "session_id": "session-0001",
        "owner_epoch": 3,
        "reconcile_operation_id": "reconcile-operation-0001",
        "job_root": str(tmp_path / "work" / "jobs" / "aassignment-0001"),
        "restore_root": str(restore_root),
        "manifest_sha256": manifest_sha256,
        "identity_fingerprint": "a" * 64,
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
        "recovery_capability": "NATIVE_VALID",
        "native_state_schema": "codex-session/1",
        "restore_adapter_version": "codex-restorer-v2/1",
        "restore_receipt_sha256": receipt_sha256,
    })

    class _RestoreAgent:
        def __init__(self):
            self.commands = []
            self.root_commands = []

        async def exec_as_agent(self, _environment, command, env, **_kwargs):
            del env
            self.commands.append(command)
            return SimpleNamespace(
                stdout=(BASE_COMMIT + "\n") if "rev-parse HEAD" in command else "",
                return_code=0,
            )

        async def exec_as_root(self, _environment, command, env, **_kwargs):
            del env
            self.root_commands.append(command)
            return SimpleNamespace(stdout="", return_code=0)

    class _RestoreEnvironment:
        default_user = "1000:1000"

        def __init__(self):
            self.files = []
            self.directories = []

        async def upload_file(self, source, destination):
            self.files.append((Path(source), destination))

        async def upload_dir(self, source, destination):
            self.directories.append((Path(source), destination))

    barrier = CheckpointV2PreProviderBarrier(str(gate_dir))
    agent = _RestoreAgent()
    environment = _RestoreEnvironment()

    async def exercise() -> None:
        session_id = await barrier.restore_if_requested(
            agent,
            environment,
            {},
            state_paths=(StatePath("sessions", "/tmp/codex/sessions"),),
        )
        assert session_id == "thread-session-0001"
        waiter = asyncio.create_task(barrier.authorize_provider_start())
        request_path = gate_dir / "request.json"
        for _index in range(100):
            if request_path.exists():
                break
            await asyncio.sleep(0.01)
        request = json.loads(request_path.read_text())
        assert request["restore_receipt_sha256"] == receipt_sha256
        _private_json(gate_dir / "grant.json", {
            "schema": "dradar-checkpoint-paid-gate-grant-v2",
            "assignment_id": "assignment-0001",
            "gate_nonce": "2" * 32,
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "owner_epoch": 4,
            "usage_segment_id": "usage-segment-resume-0001",
            "paid_execution_authorized": True,
        })
        await waiter

    asyncio.run(exercise())
    assert any("git -C /app apply --check" in command for command in agent.commands)
    assert environment.directories == [
        (restore_root / "payload/provider-state/sessions", "/tmp/codex/sessions")
    ]


class FakeAgent:
    def __init__(
        self,
        *,
        uid: int | None = None,
        gid: int | None = None,
        groups: tuple[int, ...] | None = None,
    ) -> None:
        self.commands: list[str] = []
        self.root_commands: list[str] = []
        self.agent_exec_timeouts: list[int | None] = []
        self.uid = 12345 if uid is None else uid
        self.gid = 23456 if gid is None else gid
        self.groups = (self.gid,) if groups is None else groups

    async def exec_as_agent(
        self, environment, *, command, env, timeout_sec=None,
    ):
        del env
        self.commands.append(command)
        self.agent_exec_timeouts.append(timeout_sec)
        if "rev-parse HEAD" in command:
            stdout = BASE_COMMIT + "\n"
        elif "printf 'uid=%s" in command:
            uid, gid, groups = self.uid, self.gid, self.groups
            if environment.default_user is not None:
                user, group = str(environment.default_user).split(":", 1)
                uid, gid, groups = int(user), int(group), (int(group),)
            group_text = " ".join(str(value) for value in groups)
            stdout = f"uid={uid}\ngid={gid}\ngroups={group_text}\n"
        else:
            stdout = ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def exec_as_root(self, environment, *, command, env):
        self.root_commands.append(command)
        return await self.exec_as_agent(
            environment, command=command, env=env,
        )


class FakeEnvironment:
    def __init__(self) -> None:
        self.default_user: str | int | None = None
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []
        self.exec_calls: list[dict[str, object]] = []

    async def exec(self, **kwargs):
        self.exec_calls.append(dict(kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source: Path | str, target: str) -> None:
        self.uploaded_files.append((Path(source), target))

    async def upload_dir(self, source: Path | str, target: str) -> None:
        self.uploaded_dirs.append((Path(source), target))


class FakeDurableCheckpoint(DurableCheckpoint):
    """Exercise lifecycle logic without requiring a real Pier container."""

    fail_snapshot = False
    omit_provider_state = False

    async def _install_runtime(
        self, agent, environment, env, checkpoint_id: str,
    ) -> None:
        del agent, environment, env
        self.runtime_dir = PurePosixPath("/run/dradar-checkpoint") / checkpoint_id
        self.capture_sha256 = "a" * 64
        self.supervisor_sha256 = "b" * 64

    async def _snapshot_once(
        self, agent, environment, env, *, session_id: str | None = None,
    ) -> None:
        del agent, environment, env
        if self.fail_snapshot:
            raise RuntimeError("snapshot failed")
        generation = uuid.uuid4().hex
        self.staging_host_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.staging_host_dir.chmod(0o700)
        stage = self._stage_path(generation)
        stage.mkdir(mode=0o700)
        try:
            payload = _snapshot_payload_dir(self.host_dir)
        except CheckpointError:
            payload = self.host_dir
        if payload != self.host_dir and payload.is_dir():
            for source in payload.iterdir():
                target = stage / source.name
                if source.is_dir():
                    shutil.copytree(source, target)
                elif source.is_file():
                    shutil.copy2(source, target)
        (stage / "progress-summary.txt").write_text("test\n", encoding="utf-8")
        (stage / "last_heartbeat").write_text(
            "2026-08-23T00:00:00Z\n", encoding="utf-8",
        )
        if self.omit_provider_state:
            provider_state = stage / "provider-state"
            if provider_state.exists():
                shutil.rmtree(provider_state)
            (stage / "session-omitted-sensitive").write_text(
                "provider state omitted\n", encoding="utf-8",
            )
        else:
            (stage / "provider-state").mkdir(exist_ok=True)
        if not (stage / "workspace.patch").exists():
            (stage / "workspace.patch").write_bytes(b"")
        if not (stage / "untracked.tar.gz").exists():
            with tarfile.open(stage / "untracked.tar.gz", "w:gz"):
                pass
        (stage / checkpoint_mod._TRACKED_SCAN_ARTIFACT).write_bytes(b"")
        for current, directories, files in os.walk(stage):
            Path(current).chmod(0o700)
            for name in directories:
                (Path(current) / name).chmod(0o700)
            for name in files:
                (Path(current) / name).chmod(0o600)
        try:
            self._promote_stage(generation, session_id=session_id)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        self._verify_host_ownership()


def _manager(
    logs_dir: Path,
    *,
    previous: Path | None = None,
    sensitive_values: tuple[str | bytes, ...] = (),
    state_paths: tuple[StatePath, ...] = (
        StatePath("sessions", "/tmp/provider/sessions"),
    ),
) -> DurableCheckpoint:
    logs_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    logs_dir.parent.chmod(0o700)
    logs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    logs_dir.chmod(0o700)
    if previous is not None and not previous.is_symlink():
        previous.parent.chmod(0o700)
        previous.chmod(0o700)
    return FakeDurableCheckpoint(
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
        (tmp_path / "checkpoint/checkpoint.json").read_text()
    )
    assert manifest["harness"] == "test-harness"
    assert manifest["provider"] == "test-provider"
    assert manifest["agent_version"] == "1.2.3"
    assert secret not in json.dumps(manifest)
    capture = _capture_script(
        workdir="/app",
        state_paths=manager.state_paths,
        session_probe=manager.session_probe,
        agent_identity=AgentIdentity(
            os.getuid(), os.getgid(), _numeric_groups(),
        ),
    )
    assert secret not in capture


def _numeric_groups() -> tuple[int, ...]:
    return tuple(
        int(value) for value in subprocess.run(
            ["id", "-G"], check=True, capture_output=True, text=True,
        ).stdout.split()
    )


def test_capture_and_supervisor_keep_agent_and_root_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "getuid", lambda: 12345)
    monkeypatch.setattr(os, "getgid", lambda: 23456)
    manager = _manager(tmp_path / "agent")
    identity = AgentIdentity(34567, 45678, (45678, 56789))
    capture = _capture_script(
        workdir="/app",
        state_paths=manager.state_paths,
        session_probe="cat /tmp/session-id",
        agent_identity=identity,
    )
    capture_sha = hashlib.sha256(capture.encode()).hexdigest()
    supervisor = _supervisor_script(
        checkpoint_dir="/logs/agent/checkpoint",
        runtime_dir="/run/dradar-checkpoint/checkpoint-123",
        capture_sha256=capture_sha,
        agent_identity=identity,
        host_uid=manager.host_uid,
        host_gid=manager.host_gid,
    )
    assert "expected_uid=34567" in capture
    assert "expected_gid=45678" in capture
    assert "expected_groups='45678 56789'" in capture
    assert "git -C" in capture
    assert "tar -C" in capture
    assert "cat /tmp/session-id" in capture
    assert "chown" not in capture
    assert "HOST_UID = 12345" in supervisor
    assert "HOST_GID = 23456" in supervisor
    assert "O_NOFOLLOW" in supervisor
    assert "fcntl.flock" in supervisor
    assert ".supervisor-operation.lock" in supervisor
    assert ".snapshot-aborted-" in supervisor
    assert "dir_fd=" in supervisor
    assert "st_nlink != 1" in supervisor
    assert "os.fchown(checkpoint_fd" not in supervisor
    assert "validate_checkpoint_parent(checkpoint_fd)" in supervisor
    assert "stat.S_IMODE(metadata.st_mode) == 0o750" in supervisor
    assert "open_snapshot_lock(checkpoint_fd, checkpoint_device)" in supervisor
    assert "checkpoint snapshot lock ownership is unsafe" in supervisor
    assert (
        f"MAX_CHECKPOINT_FILES = {checkpoint_mod._MAX_CHECKPOINT_FILES}"
        in supervisor
    )
    assert (
        f"MAX_CHECKPOINT_DEPTH = {checkpoint_mod._MAX_CHECKPOINT_DEPTH}"
        in supervisor
    )
    assert (
        "MAX_CHECKPOINT_FILE_BYTES = "
        f"{checkpoint_mod._MAX_CHECKPOINT_FILE_BYTES}"
        in supervisor
    )
    assert (
        "MAX_CHECKPOINT_TOTAL_BYTES = "
        f"{checkpoint_mod._MAX_CHECKPOINT_TOTAL_BYTES}"
        in supervisor
    )
    assert "git -C" not in supervisor
    assert "tar -C" not in supervisor
    assert "cat /tmp/session-id" not in supervisor
    assert "snapshot.sh" not in supervisor
    assert "aloha" not in capture + supervisor


def _supervisor_functions(tmp_path: Path) -> dict[str, object]:
    script = _supervisor_script(
        checkpoint_dir=str(tmp_path / "checkpoint"),
        runtime_dir=str(tmp_path / "runtime"),
        capture_sha256="a" * 64,
        agent_identity=AgentIdentity(12345, 23456, (23456,)),
        host_uid=os.getuid(),
        host_gid=os.getgid(),
    )
    definitions, separator, _main = script.partition("if len(sys.argv)")
    assert separator
    namespace: dict[str, object] = {}
    exec(compile(definitions, "<checkpoint-supervisor>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("limit_name", "limit", "builder", "message"),
    [
        (
            "MAX_CHECKPOINT_FILES", 1,
            lambda root: [
                (root / name).write_bytes(b"x") for name in ("one", "two")
            ],
            "entry-count limit",
        ),
        (
            "MAX_CHECKPOINT_DEPTH", 1,
            lambda root: (root / "one" / "two").mkdir(parents=True),
            "depth limit",
        ),
        (
            "MAX_CHECKPOINT_FILE_BYTES", 3,
            lambda root: (root / "large").write_bytes(b"1234"),
            "oversized file",
        ),
        (
            "MAX_CHECKPOINT_TOTAL_BYTES", 5,
            lambda root: [
                (root / name).write_bytes(b"123") for name in ("one", "two")
            ],
            "total-size limit",
        ),
    ],
)
def test_supervisor_validation_enforces_tree_budgets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    limit_name: str,
    limit: int,
    builder,
    message: str,
) -> None:
    namespace = _supervisor_functions(tmp_path)
    namespace[limit_name] = limit
    root = tmp_path / "tree"
    root.mkdir()
    builder(root)
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(SystemExit) as caught:
            namespace["validate_tree"](root_fd, root.stat().st_dev)
    finally:
        os.close(root_fd)
    assert caught.value.code == 74
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["seize_tree", "delete_tree"])
def test_supervisor_mutations_enforce_tree_budgets(
    tmp_path: Path,
    operation: str,
) -> None:
    namespace = _supervisor_functions(tmp_path)
    namespace["MAX_CHECKPOINT_FILES"] = 1
    root = tmp_path / f"tree-{operation}"
    root.mkdir()
    (root / "one").write_bytes(b"1")
    (root / "two").write_bytes(b"2")
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        with pytest.raises(SystemExit) as caught:
            namespace[operation](root_fd, root.stat().st_dev)
    finally:
        os.close(root_fd)
    assert caught.value.code == 74


def test_supervisor_abort_delete_reaps_special_entry_without_opening_it(
    tmp_path: Path,
) -> None:
    namespace = _supervisor_functions(tmp_path)
    root = tmp_path / "tree-special-delete"
    root.mkdir()
    fifo = root / "fifo"
    os.mkfifo(fifo)
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        namespace["delete_tree"](root_fd, root.stat().st_dev)
    finally:
        os.close(root_fd)
    assert not os.path.lexists(fifo)


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO/openat semantics")
def test_supervisor_regular_to_fifo_race_never_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _supervisor_functions(tmp_path)
    root = tmp_path / "tree-seize-race"
    root.mkdir()
    victim = root / "victim"
    victim.write_bytes(b"safe")
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    real_open = os.open
    swapped = False

    def raced_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "victim" and dir_fd == root_fd and not swapped:
            assert flags & getattr(os, "O_NONBLOCK", 0)
            victim.unlink()
            os.mkfifo(victim)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", raced_open)
    try:
        with pytest.raises(SystemExit) as caught:
            namespace["seize_tree"](root_fd, root.stat().st_dev)
    finally:
        os.close(root_fd)
    assert caught.value.code == 74
    assert swapped is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO/openat semantics")
def test_host_copy_regular_to_fifo_race_never_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "seized"
    source.mkdir()
    victim = source / "victim"
    victim.write_bytes(b"safe")
    destination = tmp_path / "published"
    real_open = os.open
    swapped = False

    def raced_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "victim" and dir_fd is not None and not swapped:
            assert flags & getattr(os, "O_NONBLOCK", 0)
            victim.unlink()
            os.mkfifo(victim)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", raced_open)
    with pytest.raises(CheckpointError, match="changed during copy"):
        checkpoint_mod._copy_seized_tree(source, destination)
    assert swapped is True
    assert not destination.exists()


def test_checkpoint_demotes_root_image_to_numeric_host_identity(
    tmp_path: Path,
) -> None:
    class RootImageAgent(FakeAgent):
        async def exec_as_agent(
            self, environment, *, command, env, timeout_sec=None,
        ):
            if "printf 'uid=%s" in command:
                if environment.default_user is None:
                    self.uid, self.gid, self.groups = 0, 0, (0,)
                else:
                    user, group = str(environment.default_user).split(":", 1)
                    self.uid, self.gid = int(user), int(group)
                    self.groups = (self.gid,)
            return await super().exec_as_agent(
                environment, command=command, env=env,
                timeout_sec=timeout_sec,
            )

    manager = _manager(tmp_path / "agent")
    manager.host_uid = 34567
    manager.host_gid = 45678
    environment = FakeEnvironment()
    env: dict[str, str] = {}

    identity = asyncio.run(
        manager.prepare_agent_environment(RootImageAgent(), environment, env),
    )

    assert identity == AgentIdentity(34567, 45678, (45678,))
    assert environment.default_user == "34567:45678"
    assert env["HOME"] == "/tmp/dradar-agent-home"
    root_commands = [str(call.get("command")) for call in environment.exec_calls]
    assert any("chown 34567:45678 /tmp/dradar-agent-home" in command for command in root_commands)
    assert any("find -P /app -xdev" in command for command in root_commands)
    assert any(
        "chown 34567:45678 /logs/agent; chmod 700 /logs/agent" in command
        for command in root_commands
    )
    assert all("chmod 777" not in command for command in root_commands)
    assert all("aloha" not in command for command in root_commands)


def test_repeated_agent_preparation_reprobes_without_rechowning_worktree(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    environment = FakeEnvironment()
    agent = FakeAgent()
    env: dict[str, str] = {}

    first = asyncio.run(
        manager.prepare_agent_environment(agent, environment, env),
    )
    root_call_count = len(environment.exec_calls)
    second = asyncio.run(
        manager.prepare_agent_environment(agent, environment, env),
    )

    assert second == first
    assert len(environment.exec_calls) == root_call_count
    assert sum("find -P /app -xdev" in command for command in agent.commands) == 0


def test_root_maintenance_overrides_task_shell_environment(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    environment = FakeEnvironment()

    asyncio.run(
        manager.exec_root_maintenance(environment, "test -d /logs/agent"),
    )

    call = environment.exec_calls[-1]
    assert call["user"] == "root"
    assert call["cwd"] == "/"
    assert call["env"]["PATH"] == "/usr/sbin:/usr/bin:/sbin:/bin"
    assert call["env"]["BASH_ENV"] == "/dev/null"
    assert call["env"]["ENV"] == "/dev/null"
    assert call["env"]["CDPATH"] == ""
    assert "OPENAI_API_KEY" not in call["env"]
    assert str(call["command"]).startswith("/usr/bin/env -i ")
    assert "/bin/bash --noprofile --norc -c" in str(call["command"])


def test_root_maintenance_drops_inherited_bash_functions(
    tmp_path: Path,
) -> None:
    class FunctionInjectingEnvironment:
        async def exec(self, **kwargs):
            process_env = {
                "PATH": "/usr/bin:/bin",
                "BASH_FUNC_true%%": "() { return 97; }",
                **kwargs["env"],
            }
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", kwargs["command"]],
                env=process_env,
                capture_output=True,
                text=True,
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    manager = _manager(tmp_path / "agent")

    asyncio.run(
        manager.exec_root_maintenance(FunctionInjectingEnvironment(), "true"),
    )


def test_checkpoint_normalizes_pier_world_writable_host_logs_directory(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    manager.logs_dir.chmod(0o777)
    manager.trial_dir.chmod(0o755)

    asyncio.run(manager.start(FakeAgent(), FakeEnvironment(), {}))

    assert stat.S_IMODE(manager.logs_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(manager.trial_dir.stat().st_mode) == 0o700


def test_checkpoint_prepares_host_layout_before_agent_log_store_write(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "trial" / "agent")
    manager.logs_dir.chmod(0o777)
    manager.trial_dir.chmod(0o777)
    config = manager.logs_dir / "host-authored-config.toml"

    with pytest.raises(UnsafeAgentLog, match="parent is not host-private"):
        AgentLogStore(manager.logs_dir).replace_text(config, "enabled = true\n")

    manager.prepare_host_layout()
    manager.prepare_host_layout()

    assert manager.manifest_path is None
    assert not manager.host_dir.exists()
    assert stat.S_IMODE(manager.logs_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(manager.trial_dir.stat().st_mode) == 0o700
    assert AgentLogStore(manager.logs_dir).replace_text(
        config, "enabled = true\n",
    )
    assert config.read_text(encoding="utf-8") == "enabled = true\n"
    assert stat.S_IMODE(config.stat().st_mode) == 0o600


def test_disabled_checkpoint_host_layout_preflight_is_strict_noop(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "world-writable-trial"
    logs = trial / "not-the-agent-directory"
    logs.mkdir(parents=True, mode=0o777)
    trial.chmod(0o777)
    logs.chmod(0o777)
    manager = DurableCheckpoint(
        logs_dir=logs,
        enabled=False,
        assignment_id=None,
        task_id=None,
        model=None,
        effort=None,
        harness="test-harness",
        provider="test-provider",
        agent_version="1.2.3",
    )

    manager.prepare_host_layout()

    assert stat.S_IMODE(trial.stat().st_mode) == 0o777
    assert stat.S_IMODE(logs.stat().st_mode) == 0o777
    assert not manager.host_dir.exists()


def test_disabled_checkpoint_constructs_without_posix_host_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True)
    monkeypatch.delattr(os, "getuid")
    monkeypatch.delattr(os, "getgid")

    manager = DurableCheckpoint(
        logs_dir=logs,
        enabled=False,
        assignment_id=None,
        task_id=None,
        model=None,
        effort=None,
        harness="test-harness",
        provider="test-provider",
        agent_version="1.2.3",
    )

    assert manager.enabled is False
    assert manager.host_uid == 0
    assert manager.host_gid == 0


def test_enabled_checkpoint_fails_closed_without_posix_host_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True)
    monkeypatch.delattr(os, "getuid")
    monkeypatch.delattr(os, "getgid")

    with pytest.raises(CheckpointError, match="POSIX host ownership"):
        DurableCheckpoint(
            logs_dir=logs,
            enabled=True,
            assignment_id="assignment-1",
            task_id="task-1",
            model="gpt-5.6-sol",
            effort="high",
            harness="codex",
            provider="openai",
            agent_version="0.149.0",
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow directory semantics")
def test_checkpoint_host_layout_preflight_rejects_symlinked_agent_dir(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    trial.mkdir(mode=0o700)
    target = tmp_path / "outside-agent"
    target.mkdir(mode=0o700)
    logs = trial / "agent"
    logs.symlink_to(target, target_is_directory=True)
    manager = FakeDurableCheckpoint(
        logs_dir=logs,
        enabled=True,
        assignment_id="assignment-123",
        task_id="task-1",
        model="model-1",
        effort="high",
        harness="test-harness",
        provider="test-provider",
        agent_version="1.2.3",
    )

    with pytest.raises(CheckpointError, match="host layout is unreadable"):
        manager.prepare_host_layout()

    assert stat.S_IMODE(target.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "identity",
    [
        (0, 1, (1,)),
        (1, 0, (0,)),
        (1, 1, (1, 0)),
    ],
)
def test_agent_identity_rejects_root_uid_gid_or_group(
    identity: tuple[int, int, tuple[int, ...]],
) -> None:
    with pytest.raises(ValueError, match="root identity"):
        AgentIdentity(*identity)


def test_checkpoint_preflight_starts_host_periodic_task_without_root_process(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    agent = FakeAgent()
    environment = FakeEnvironment()

    asyncio.run(manager.start(agent, environment, {}))

    assert manager.snapshot_launch_attempted
    assert manager.snapshot_background_ready
    assert manager._periodic_task is not None
    assert not agent.root_commands
    assert not any("nohup" in command or "setsid" in command for command in agent.commands)


def test_runtime_install_protects_staging_entry_with_sticky_root_parent(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    manager.agent_identity = AgentIdentity(
        os.getuid(), os.getgid(), _numeric_groups(),
    )
    environment = FakeEnvironment()

    asyncio.run(
        DurableCheckpoint._install_runtime(
            manager,
            FakeAgent(),
            environment,
            {},
            "checkpoint-runtime-12345678",
        )
    )

    root_commands = "\n".join(
        str(call.get("command", "")) for call in environment.exec_calls
    )
    assert "/usr/bin/chmod 1770 /logs/agent" in root_commands
    assert f"/usr/bin/chown 0:{os.getgid()} /logs/agent" in root_commands
    assert f"-o 0 -g {os.getgid()} -m 0750" in root_commands


def test_host_handoff_verification_rejects_public_mode_and_symlink(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    manager.host_dir.mkdir(parents=True, mode=0o700)
    exposed = manager.host_dir / "exposed"
    exposed.write_text("private\n", encoding="utf-8")
    exposed.chmod(0o644)

    with pytest.raises(CheckpointError, match="not private"):
        manager._verify_host_ownership()

    exposed.chmod(0o600)
    linked = manager.host_dir / "linked"
    linked.symlink_to(exposed)
    with pytest.raises(CheckpointError, match="special file"):
        manager._verify_host_ownership()

    linked.unlink()
    fifo = manager.host_dir / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(CheckpointError, match="special file"):
        manager._verify_host_ownership()


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


def test_checkpoint_restore_rejects_snapshot_lock(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    (previous / "snapshot.lock").mkdir()

    with pytest.raises(CheckpointError, match="snapshot is incomplete"):
        asyncio.run(
            _manager(tmp_path / "agent", previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )


def test_checkpoint_restore_rejects_dangling_snapshot_lock(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    (previous / "snapshot.lock").symlink_to(previous / "missing")

    with pytest.raises(CheckpointError, match="special file"):
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

    with pytest.raises(CheckpointError, match="host-private storage"):
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
    current_logs = tmp_path / "current-trial" / "agent"

    with pytest.raises(CheckpointIncompatibleError, match="provider"):
        asyncio.run(
            _manager(current_logs, previous=previous).start(
                FakeAgent(), FakeEnvironment(), {},
            )
        )

    assert json.loads((previous / "checkpoint.json").read_text()) == original
    current = json.loads(
        (current_logs.parent / "checkpoint/checkpoint.json").read_text()
    )
    assert current["phase"] == "incompatible"
    assert current["provider"] == "test-provider"


def test_checkpoint_finish_detects_exact_credential_and_marks_invalid(
    tmp_path: Path,
) -> None:
    secret = b"provider-secret-value-that-does-not-match-generic-regex"
    manager = _manager(tmp_path / "agent", sensitive_values=(secret,))
    agent = FakeAgent()
    environment = FakeEnvironment()
    async def scenario() -> None:
        await manager.start(agent, environment, {})
        state = _snapshot_payload_dir(manager.host_dir) / "provider-state/sessions"
        state.mkdir(parents=True)
        state.parent.chmod(0o700)
        state.chmod(0o700)
        session = state / "session.jsonl"
        session.write_bytes(b"prefix:" + secret + b":suffix")
        session.chmod(0o600)
        await manager.finish(
            agent, environment, {}, completed=False, failure=RuntimeError("stop"),
        )
    asyncio.run(scenario())

    manifest = json.loads(
        (tmp_path / "checkpoint/checkpoint.json").read_text()
    )
    assert manifest["phase"] == "invalid"
    assert manifest["failure_type"] == "CheckpointSecretDetected"
    assert (manager.host_dir / "invalid-secret").is_file()


@pytest.mark.parametrize(
    ("session_id", "sensitive_values"),
    [
        (
            "provider-session-secret-123456789",
            (b"provider-session-secret-123456789",),
        ),
        (
            "sk-proj-genericcheckpointtoken123456789",
            (),
        ),
    ],
)
def test_checkpoint_finish_never_persists_credential_shaped_session_id(
    tmp_path: Path,
    session_id: str,
    sensitive_values: tuple[bytes, ...],
) -> None:
    manager = _manager(
        tmp_path / "agent", sensitive_values=sensitive_values,
    )
    agent = FakeAgent()
    environment = FakeEnvironment()

    async def scenario() -> None:
        await manager.start(agent, environment, {})
        await manager.finish(
            agent,
            environment,
            {},
            completed=False,
            failure=RuntimeError("stop"),
            session_id=session_id,
        )

    asyncio.run(scenario())

    manifest = json.loads((manager.host_dir / "checkpoint.json").read_text())
    assert manifest["phase"] == "invalid"
    assert manifest["failure_type"] == "CheckpointSecretDetected"
    assert "session_id" not in manifest
    secret_bytes = session_id.encode("utf-8")
    for current, _directories, files in os.walk(manager.host_dir):
        for name in files:
            assert secret_bytes not in (Path(current) / name).read_bytes()


def test_checkpoint_finish_omitted_provider_state_suppresses_native_session(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    manager.omit_provider_state = True
    agent = FakeAgent()
    environment = FakeEnvironment()

    async def scenario() -> None:
        await manager.start(agent, environment, {})
        await manager.finish(
            agent,
            environment,
            {},
            completed=False,
            failure=RuntimeError("interrupt"),
            session_id="zcode-session-1234",
        )

    asyncio.run(scenario())

    manifest = json.loads((manager.host_dir / "checkpoint.json").read_text())
    payload = _snapshot_payload_dir(manager.host_dir)
    assert manifest["phase"] == "paused"
    assert "session_id" not in manifest
    assert (payload / "session-omitted-sensitive").is_file()
    assert not (payload / "provider-state").exists()
    assert not (payload / "session-id").exists()


def test_runtime_tree_handoff_uses_captured_host_owner(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    manager.host_uid = 34567
    manager.host_gid = 45678
    environment = FakeEnvironment()

    asyncio.run(
        manager.return_runtime_tree_to_host_owner(
            environment,
            "/logs/agent/dsh-home",
        )
    )

    call = environment.exec_calls[-1]
    command = str(call["command"])
    assert call["user"] == "root"
    assert "/usr/bin/find -P /logs/agent/dsh-home -xdev" in command
    assert "/usr/bin/chown -h -- 34567:45678 {} +" in command
    assert "/usr/bin/chown -h -- 34567:45678 /logs/agent/dsh-home" in command
    assert "chown -R" not in command
    assert "aloha" not in command


@pytest.mark.parametrize(
    "remote_path",
    [
        "/logs/agent",
        "/logs/agent/.dradar-checkpoint-staging",
        "/logs/agent/.dradar-checkpoint-staging/child",
        "/tmp/runtime",
        "/logs/agent/../outside",
    ],
)
def test_runtime_tree_handoff_rejects_unsafe_path(
    tmp_path: Path,
    remote_path: str,
) -> None:
    manager = _manager(tmp_path / "agent")

    with pytest.raises(CheckpointError, match="handoff path is unsafe"):
        asyncio.run(
            manager.return_runtime_tree_to_host_owner(
                FakeEnvironment(),
                remote_path,
            )
        )


def test_snapshot_stop_failure_discards_all_payload_artifacts(
    tmp_path: Path,
) -> None:
    secret = b"provider-secret-value-not-covered-by-generic-pattern"
    manager = _manager(tmp_path / "agent", sensitive_values=(secret,))
    agent = FakeAgent()
    environment = FakeEnvironment()
    async def scenario() -> None:
        await manager.start(agent, environment, {})
        payload = _snapshot_payload_dir(manager.host_dir)
        (payload / "workspace.patch").write_bytes(secret)
        (payload / "untracked.tar.gz").write_bytes(secret)
        state = payload / "provider-state/sessions"
        state.mkdir(parents=True)
        (state / "wire.jsonl").write_bytes(secret)
        manager.fail_snapshot = True
        await manager.finish(
            agent, environment, {}, completed=False, failure=KeyboardInterrupt(),
        )
    asyncio.run(scenario())

    checkpoint = manager.host_dir
    payload = _snapshot_payload_dir(checkpoint)
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    assert manifest["phase"] == "invalid"
    assert manifest["failure_type"] == "CheckpointSnapshotInvalid"
    assert (checkpoint / "invalid-snapshot").is_file()
    assert (payload / "workspace.patch").exists()
    assert (payload / "untracked.tar.gz").exists()
    assert (payload / "provider-state").exists()


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


@pytest.mark.parametrize("archive_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_archive_rejects_oversized_long_name_control_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_format: int,
) -> None:
    archive_path = tmp_path / "long-name.tar.gz"
    with tarfile.open(archive_path, "w:gz", format=archive_format) as archive:
        member = tarfile.TarInfo("x" * 4096)
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    # Exercise the same branch as a multi-gigabyte compressed control record
    # without allocating one in the test process.
    monkeypatch.setattr(
        checkpoint_mod, "_MAX_ARCHIVE_CONTROL_READ_BYTES", 512,
    )
    with pytest.raises(CheckpointError, match="oversized control record"):
        _validate_archive(archive_path, ())


@pytest.mark.parametrize("archive_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_archive_accepts_bounded_long_name_control_record(
    tmp_path: Path, archive_format: int,
) -> None:
    archive_path = tmp_path / "bounded-long-name.tar.gz"
    with tarfile.open(archive_path, "w:gz", format=archive_format) as archive:
        member = tarfile.TarInfo("x" * 1024)
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    assert _validate_archive(archive_path, ()) is False


@pytest.mark.parametrize("owner_field", ["uname", "gname"])
def test_archive_rejects_exact_secret_in_owner_header(
    tmp_path: Path, owner_field: str,
) -> None:
    secret = b"opaque-owner-secret-123456789"
    archive_path = tmp_path / f"secret-{owner_field}.tar.gz"
    with tarfile.open(archive_path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("safe.txt")
        setattr(member, owner_field, secret.decode("ascii"))
        member.size = 4
        archive.addfile(member, io.BytesIO(b"safe"))

    assert _validate_archive(archive_path, (secret,)) is True


def test_archive_rejects_exact_secret_in_raw_tar_padding(tmp_path: Path) -> None:
    secret = b"opaque-provider-padding-value-123456789"
    raw_tar = bytearray(1024)
    raw_tar[700 : 700 + len(secret)] = secret
    archive_path = tmp_path / "secret-padding.tar.gz"
    with gzip.open(archive_path, "wb") as compressed:
        compressed.write(raw_tar)

    assert _validate_archive(archive_path, (secret,)) is True


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

    agent = FakeAgent(uid=12345, gid=23456, groups=(23456, 34567))
    session_id = asyncio.run(
        _manager(tmp_path / "agent", previous=previous).start(
            agent, environment, {},
        )
    )

    assert session_id == "session-12345678"
    assert environment.uploaded_dirs == [
        (state, "/tmp/provider/sessions"),
    ]
    ownership = [
        call for call in environment.exec_calls
        if f"chown -R -h -- {os.getuid()}:{os.getgid()} /tmp/provider/sessions"
        in str(call.get("command"))
    ]
    assert len(ownership) == 1
    assert ownership[0]["user"] == "root"
    assert "find -P /tmp/provider/sessions -type d -exec chmod 700" in str(
        ownership[0]["command"]
    )


def test_checkpoint_restores_allowlisted_file_state(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    index = previous / "provider-state/session-index"
    index.parent.mkdir()
    index.write_text('{"sessionId":"session-12345678"}\n', encoding="utf-8")
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    environment = FakeEnvironment()

    agent = FakeAgent(uid=12345, gid=23456, groups=(23456,))
    asyncio.run(
        _manager(
            tmp_path / "agent",
            previous=previous,
            state_paths=(
                StatePath("session-index", "/tmp/provider/session_index.jsonl"),
            ),
        ).start(agent, environment, {})
    )

    assert environment.uploaded_files == [
        (index, "/tmp/provider/session_index.jsonl"),
    ]
    ownership = [
        call for call in environment.exec_calls
        if (
            f"chown -h -- {os.getuid()}:{os.getgid()} "
            "/tmp/provider/session_index.jsonl"
        )
        in str(call.get("command"))
    ]
    assert len(ownership) == 1
    assert ownership[0]["user"] == "root"
    assert "chmod 600 -- /tmp/provider/session_index.jsonl" in str(
        ownership[0]["command"]
    )


def test_checkpoint_returns_uploaded_workspace_files_to_non_root_agent(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    (previous / "workspace.patch").write_text(
        "diff --git a/a b/a\n", encoding="utf-8",
    )
    with tarfile.open(previous / "untracked.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("note.txt")
        member.size = 4
        archive.addfile(member, io.BytesIO(b"safe"))
    environment = FakeEnvironment()
    agent = FakeAgent(uid=12345, gid=23456, groups=(23456,))

    asyncio.run(
        _manager(
            tmp_path / "agent", previous=previous, state_paths=(),
        ).start(agent, environment, {})
    )

    root_commands = [str(call.get("command")) for call in environment.exec_calls]
    assert any(
        f"chown -h -- {os.getuid()}:{os.getgid()} "
        "/tmp/dradar-checkpoint-restore/workspace.patch"
        in command
        for command in root_commands
    )
    assert any(
        f"chown -h -- {os.getuid()}:{os.getgid()} "
        "/tmp/dradar-checkpoint-restore/untracked.tar.gz"
        in command
        for command in root_commands
    )
    assert any(
        "git -C /app apply --binary /tmp/dradar-checkpoint-restore/workspace.patch"
        in command
        for command in agent.commands
    )
    assert any(
        "tar -xzf /tmp/dradar-checkpoint-restore/untracked.tar.gz -C /app"
        in command
        for command in agent.commands
    )


def test_checkpoint_finish_uses_snapshot_session_sidecar(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "agent")
    agent = FakeAgent()
    environment = FakeEnvironment()
    async def scenario() -> None:
        await manager.start(agent, environment, {})
        session_id_path = _snapshot_payload_dir(manager.host_dir) / "session-id"
        session_id_path.write_text(
            "session-12345678\n", encoding="utf-8",
        )
        session_id_path.chmod(0o600)
        await manager.finish(
            agent, environment, {}, completed=False, failure=KeyboardInterrupt(),
        )
    asyncio.run(scenario())

    manifest = json.loads((manager.host_dir / "checkpoint.json").read_text())
    assert manifest["phase"] == "paused"
    assert manifest["session_id"] == "session-12345678"
    assert manager._periodic_task is None


def test_checkpoint_finish_marks_restore_rejection_incompatible(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path / "agent")
    agent = FakeAgent()
    environment = FakeEnvironment()
    async def scenario() -> None:
        await manager.start(agent, environment, {})
        await manager.finish(
            agent,
            environment,
            {},
            completed=False,
            failure=CheckpointIncompatibleError("native session missing"),
        )
    asyncio.run(scenario())

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


def test_capture_script_captures_workspace_state_and_session(tmp_path: Path) -> None:
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
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(StatePath("sessions", str(state)),),
            session_probe=f"cat {session_id}",
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        )
    )

    subprocess.run(["sh", str(script), str(stage), base], check=True)

    assert "tracked.txt" in (stage / "workspace.patch").read_text()
    with tarfile.open(stage / "untracked.tar.gz", "r:gz") as archive:
        assert "new.txt" in archive.getnames()
    assert (stage / "provider-state/sessions/wire.jsonl").read_text() == "{}\n"
    assert (stage / "session-id").read_text() == "session-abcdefgh\n"
    assert (stage / "last_heartbeat").is_file()
    for path in (
        stage,
        stage / "workspace.patch",
        stage / "provider-state",
        stage / "provider-state/sessions/wire.jsonl",
        stage / "session-id",
        stage / "last_heartbeat",
    ):
        assert path.stat().st_uid == os.getuid()
        assert path.stat().st_gid == os.getgid()

    tracked.write_text("after second snapshot\n", encoding="utf-8")
    (state / "wire.jsonl").write_text('{"round":2}\n', encoding="utf-8")
    stage2 = tmp_path / "stage2"
    stage2.mkdir(mode=0o700)
    subprocess.run(["sh", str(script), str(stage2), base], check=True)

    assert "after second snapshot" in (stage2 / "workspace.patch").read_text()
    assert (
        stage2 / "provider-state/sessions/wire.jsonl"
    ).read_text() == '{"round":2}\n'


def test_sensitive_provider_state_degrades_to_workspace_only_checkpoint(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    tracked = worktree / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
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
    tracked.write_text("after\n", encoding="utf-8")
    state = tmp_path / "native-state"
    state.mkdir()
    (state / "rollout.jsonl").write_text(
        '{"credential":"eyJabcdefghij.abcdefghij.abcdefghij"}\n',
        encoding="utf-8",
    )
    session_source = tmp_path / "session-source"
    session_source.write_text("session-abcdefgh\n", encoding="utf-8")

    manager = _manager(tmp_path / "trial" / "agent")
    manager.host_dir.mkdir(mode=0o700)
    manager.staging_host_dir.mkdir(mode=0o700)
    generation = "c" * 32
    stage = manager._stage_path(generation)
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(StatePath("zcode-rollout", str(state)),),
            session_probe=f"cat {session_source}",
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )

    subprocess.run(["sh", str(script), str(stage), base], check=True)

    assert (stage / "session-omitted-sensitive").is_file()
    assert not (stage / "provider-state").exists()
    assert not (stage / "session-id").exists()
    manager._promote_stage(generation, session_id="session-explicit-1234")
    payload = _snapshot_payload_dir(manager.host_dir)
    assert (payload / "session-omitted-sensitive").is_file()
    assert not (payload / "provider-state").exists()
    assert not (payload / "session-id").exists()
    assert (payload / "workspace.patch").is_file()
    assert not (manager.host_dir / "invalid-secret").exists()


def test_provider_state_cannot_coexist_with_omission_marker(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "trial" / "agent")
    manager.host_dir.mkdir(mode=0o700)
    manager.staging_host_dir.mkdir(mode=0o700)
    generation = "d" * 32
    stage = manager._stage_path(generation)
    stage.mkdir(mode=0o700)
    (stage / "progress-summary.txt").write_text("test\n", encoding="utf-8")
    (stage / "last_heartbeat").write_text(
        "2026-08-23T00:00:00Z\n", encoding="utf-8",
    )
    (stage / "workspace.patch").write_bytes(b"")
    with tarfile.open(stage / "untracked.tar.gz", "w:gz"):
        pass
    (stage / checkpoint_mod._TRACKED_SCAN_ARTIFACT).write_bytes(b"")
    (stage / "session-omitted-sensitive").write_text(
        "provider state omitted\n", encoding="utf-8",
    )
    (stage / "provider-state").mkdir()
    for current, directories, files in os.walk(stage):
        Path(current).chmod(0o700)
        for name in directories:
            (Path(current) / name).chmod(0o700)
        for name in files:
            (Path(current) / name).chmod(0o600)

    with pytest.raises(CheckpointError, match="omission is inconsistent"):
        manager._promote_stage(generation)

    assert (manager.host_dir / "invalid-snapshot").is_file()
    assert not (manager.host_dir / "current-generation").exists()


def test_capture_disables_configured_textconv(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    (worktree / ".gitattributes").write_text(
        "tracked.txt diff=checkpoint-test\n", encoding="utf-8",
    )
    tracked = worktree / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(worktree), "add", ".gitattributes", "tracked.txt"],
        check=True,
    )
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
    tracked.write_text("after\n", encoding="utf-8")
    marker = tmp_path / "diff-driver-invoked"
    driver = tmp_path / "diff-driver.sh"
    driver.write_text(
        "#!/bin/sh\n: > \"$CHECKPOINT_DIFF_MARKER\"\ncat \"$1\"\n",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    subprocess.run(
        [
            "git", "-C", str(worktree), "config",
            "diff.checkpoint-test.textconv", str(driver),
        ],
        check=True,
    )
    environment = {**os.environ, "CHECKPOINT_DIFF_MARKER": str(marker)}
    # Prove the repository-local driver is live; the capture must suppress it.
    subprocess.run(
        ["git", "-C", str(worktree), "diff", "--textconv", base, "--"],
        check=True,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    assert marker.is_file()
    marker.unlink()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    capture = _capture_script(
        workdir=str(worktree),
        state_paths=(),
        session_probe=None,
        agent_identity=AgentIdentity(
            os.getuid(), os.getgid(), _numeric_groups(),
        ),
    )
    script.write_text(capture, encoding="utf-8")

    subprocess.run(
        ["sh", str(script), str(stage), base],
        check=True,
        env=environment,
    )

    assert not marker.exists()
    assert "tracked.txt" in (stage / "workspace.patch").read_text()
    assert "diff --no-ext-diff --no-textconv --binary" in capture


def test_capture_propagates_git_untracked_listing_failure(tmp_path: Path) -> None:
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
    (worktree / "untracked.txt").write_text("data\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    wrapper = fake_bin / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        "for argument in \"$@\"; do\n"
        "  [ \"$argument\" = ls-files ] && exit 42\n"
        "done\n"
        "exec \"$CHECKPOINT_REAL_GIT\" \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(),
            session_probe=None,
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )
    real_git = shutil.which("git")
    assert real_git is not None
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CHECKPOINT_REAL_GIT": real_git,
    }

    result = subprocess.run(
        ["sh", str(script), str(stage), base],
        check=False,
        env=environment,
    )

    assert result.returncode == 42
    assert not (stage / "untracked.tar.gz").exists()
    assert not (stage / ".untracked-files").exists()


def test_capture_rejects_untracked_fifo_before_tar(tmp_path: Path) -> None:
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
    os.mkfifo(worktree / "untracked-fifo")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(),
            session_probe=None,
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["sh", str(script), str(stage), base],
        check=False,
        timeout=5,
    )

    assert result.returncode == 74
    assert not (stage / "untracked.tar.gz").exists()


def test_capture_rejects_untracked_tree_over_entry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint_mod, "_MAX_CHECKPOINT_FILES", 1)
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
    (worktree / "one").write_bytes(b"1")
    (worktree / "two").write_bytes(b"2")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(),
            session_probe=None,
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["sh", str(script), str(stage), base], check=False,
    )

    assert result.returncode == 74
    assert not (stage / "untracked.tar.gz").exists()


def test_binary_tracked_secret_is_scanned_before_checkpoint_publication(
    tmp_path: Path,
) -> None:
    secret = b"opaque-provider-value-not-covered-by-generic-pattern"
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    tracked = worktree / "tracked.bin"
    tracked.write_bytes(b"before\x00binary\n")
    subprocess.run(["git", "-C", str(worktree), "add", "tracked.bin"], check=True)
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
    tracked.write_bytes(b"after\x00" + secret + b"\x00binary\n")
    manager = _manager(tmp_path / "trial" / "agent", sensitive_values=(secret,))
    manager.host_dir.mkdir(mode=0o700)
    manager.staging_host_dir.mkdir(mode=0o700)
    generation = "f" * 32
    stage = manager._stage_path(generation)
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(),
            session_probe=None,
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )

    subprocess.run(["sh", str(script), str(stage), base], check=True)

    # The resumable git binary patch zlib/base85-encodes these bytes, while the
    # short-lived scan representation exposes the exact reconstructable data.
    assert secret not in (stage / "workspace.patch").read_bytes()
    assert secret in (stage / checkpoint_mod._TRACKED_SCAN_ARTIFACT).read_bytes()
    with pytest.raises(CheckpointError, match="rejected credential data"):
        manager._promote_stage(generation)
    assert (manager.host_dir / "invalid-secret").is_file()
    assert not (manager.host_dir / "current-generation").exists()
    assert not (manager.host_dir / "snapshots" / generation).exists()


def test_tracked_secret_scan_is_derived_from_exact_binary_patch_not_live_tree(
    tmp_path: Path,
) -> None:
    secret = b"opaque-raced-binary-value-not-in-generic-pattern"
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    tracked = worktree / "tracked.bin"
    tracked.write_bytes(b"base\x00binary\n")
    subprocess.run(["git", "-C", str(worktree), "add", "tracked.bin"], check=True)
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
    tracked.write_bytes(b"before\x00" + secret + b"\x00binary\n")

    wrapper_dir = tmp_path / "git-wrapper"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "git"
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "binary_diff=0\n"
        "if [ \"${3-}\" = diff ]; then\n"
        "  for argument in \"$@\"; do\n"
        "    [ \"$argument\" = --binary ] && binary_diff=1\n"
        "  done\n"
        "fi\n"
        f"{shlex.quote(real_git)} \"$@\"\n"
        "status=$?\n"
        "if [ \"$status\" -eq 0 ] && [ \"$binary_diff\" -eq 1 ] "
        "&& [ ! -e \"$RACE_MARKER\" ]; then\n"
        "  printf 'safe-after-race\\000' > \"$RACE_FILE\"\n"
        "  : > \"$RACE_MARKER\"\n"
        "fi\n"
        "exit \"$status\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    manager = _manager(
        tmp_path / "trial" / "agent", sensitive_values=(secret,),
    )
    manager.host_dir.mkdir(mode=0o700)
    manager.staging_host_dir.mkdir(mode=0o700)
    generation = "e" * 32
    stage = manager._stage_path(generation)
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture-race.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(),
            session_probe=None,
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{wrapper_dir}:{environment.get('PATH', '')}",
            "RACE_FILE": str(tracked),
            "RACE_MARKER": str(tmp_path / "race-fired"),
        }
    )

    subprocess.run(
        ["sh", str(script), str(stage), base],
        check=True,
        env=environment,
    )

    assert tracked.read_bytes() == b"safe-after-race\x00"
    assert secret not in (stage / "workspace.patch").read_bytes()
    assert secret in (stage / checkpoint_mod._TRACKED_SCAN_ARTIFACT).read_bytes()
    with pytest.raises(CheckpointError, match="rejected credential data"):
        manager._promote_stage(generation)
    assert (manager.host_dir / "invalid-secret").is_file()
    assert not (manager.host_dir / "current-generation").exists()


def test_resume_does_not_carry_legacy_opaque_event_values(tmp_path: Path) -> None:
    secret = "opaque-provider-value-not-covered-by-generic-pattern"
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "checkpoint.json").write_text(json.dumps(_manifest()))
    (previous / "events.jsonl").write_text(
        json.dumps({"event": "legacy", "detail": {"note": secret}}) + "\n",
        encoding="utf-8",
    )

    manager = _manager(
        tmp_path / "current" / "agent",
        previous=previous,
        sensitive_values=(secret,),
    )
    asyncio.run(manager.start(FakeAgent(), FakeEnvironment(), {}))

    current_events = (manager.host_dir / "events.jsonl").read_text(
        encoding="utf-8",
    )
    assert secret not in current_events
    assert "checkpoint_started" in current_events


def test_regular_tree_rejects_symlink_fifo_and_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    regular = root / "regular"
    regular.write_text("data\n", encoding="utf-8")

    linked = root / "linked"
    linked.symlink_to(regular)
    with pytest.raises(CheckpointError, match="special file"):
        _validate_regular_tree(root, label="test tree")
    linked.unlink()

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(CheckpointError, match="special file"):
        _validate_regular_tree(root, label="test tree")
    fifo.unlink()

    hardlink = root / "hardlink"
    os.link(regular, hardlink)
    with pytest.raises(CheckpointError, match="multiply linked"):
        _validate_regular_tree(root, label="test tree")


def test_capture_rejects_special_provider_state(tmp_path: Path) -> None:
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
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    native = tmp_path / "native"
    native.mkdir()
    os.mkfifo(native / "fifo")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    script = tmp_path / "capture.sh"
    script.write_text(
        _capture_script(
            workdir=str(worktree),
            state_paths=(StatePath("sessions", str(native)),),
            session_probe=None,
            agent_identity=AgentIdentity(
                os.getuid(), os.getgid(), _numeric_groups(),
            ),
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["sh", str(script), str(stage), base], check=False,
    )

    assert result.returncode == 74


def test_generation_promotion_is_atomic_and_removes_old_payload(tmp_path: Path) -> None:
    manager = _manager(tmp_path / "agent")
    manager.host_dir.mkdir(parents=True, mode=0o700)
    manager.staging_host_dir.mkdir(parents=True, mode=0o700)
    first = "1" * 32
    first_stage = manager._stage_path(first)
    first_stage.mkdir(mode=0o700)
    (first_stage / "progress-summary.txt").write_text("first\n")
    (first_stage / "last_heartbeat").write_text("first\n")
    (first_stage / checkpoint_mod._TRACKED_SCAN_ARTIFACT).write_bytes(b"")
    manager._promote_stage(first)

    second = "2" * 32
    second_stage = manager._stage_path(second)
    second_stage.mkdir(mode=0o700)
    (second_stage / "progress-summary.txt").write_text("second\n")
    (second_stage / "last_heartbeat").write_text("second\n")
    (second_stage / checkpoint_mod._TRACKED_SCAN_ARTIFACT).write_bytes(b"")
    manager._promote_stage(second)

    assert (manager.host_dir / "current-generation").read_text().strip() == second
    assert _snapshot_payload_dir(manager.host_dir).name == second
    assert not (
        _snapshot_payload_dir(manager.host_dir)
        / checkpoint_mod._TRACKED_SCAN_ARTIFACT
    ).exists()
    assert not (manager.host_dir / "snapshots" / first).exists()


def test_stop_periodic_reaps_writer_before_propagating_caller_cancellation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "agent")
        manager._periodic_stop = asyncio.Event()
        entered = asyncio.Event()
        release = asyncio.Event()
        reaped = asyncio.Event()

        async def writer() -> None:
            entered.set()
            try:
                await release.wait()
            finally:
                reaped.set()

        manager._periodic_task = asyncio.create_task(writer())
        await entered.wait()
        stopping = asyncio.create_task(manager._stop_periodic())
        await asyncio.sleep(0)
        stopping.cancel()
        await asyncio.sleep(0)
        assert not stopping.done()
        assert not reaped.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert reaped.is_set()
        assert manager._periodic_task is None

    asyncio.run(scenario())


def test_stop_periodic_tears_down_environment_and_never_leaks_stubborn_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "agent")
        manager._periodic_stop = asyncio.Event()
        entered = asyncio.Event()
        environment_stopped = asyncio.Event()
        writer_done = asyncio.Event()

        class StoppableEnvironment:
            def __init__(self) -> None:
                self.stop_calls: list[bool] = []

            async def stop(self, delete: bool) -> None:
                self.stop_calls.append(delete)
                environment_stopped.set()

        environment = StoppableEnvironment()

        async def stubborn_writer() -> None:
            entered.set()
            try:
                while not environment_stopped.is_set():
                    try:
                        await environment_stopped.wait()
                    except asyncio.CancelledError:
                        # Reproduce an environment exec backend that consumes
                        # task cancellation until its container is stopped.
                        continue
            finally:
                writer_done.set()

        manager._periodic_environment = environment
        manager._periodic_task = asyncio.create_task(stubborn_writer())
        await entered.wait()
        monkeypatch.setattr(
            checkpoint_mod, "_PERIODIC_STOP_TIMEOUT_SEC", 0.01,
        )

        with pytest.raises(CheckpointError, match="forced cancellation"):
            await asyncio.wait_for(manager._stop_periodic(), timeout=1)

        assert environment.stop_calls == [False]
        assert writer_done.is_set()
        assert manager._periodic_task is None
        assert manager._periodic_environment is None

    asyncio.run(scenario())


def test_periodic_capture_timeout_cancels_paid_owner_immediately(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "agent")
        manager.interval_sec = 0.01
        agent = FakeAgent()
        environment = FakeEnvironment()
        original_snapshot = manager._snapshot_once
        periodic_failed = asyncio.Event()
        calls = 0

        async def fail_second_snapshot(*args, **kwargs) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await original_snapshot(*args, **kwargs)
                return
            periodic_failed.set()
            raise TimeoutError("periodic capture timed out")

        manager._snapshot_once = fail_second_snapshot  # type: ignore[method-assign]

        async def paid_owner() -> None:
            failure: BaseException | None = None
            try:
                await manager.start(agent, environment, {})
                await asyncio.Event().wait()
            except BaseException as exc:
                failure = exc
                raise
            finally:
                await manager.finish(
                    agent, environment, {}, completed=False, failure=failure,
                )

        owner = asyncio.create_task(paid_owner())
        await asyncio.wait_for(periodic_failed.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await owner

        assert calls == 2
        assert manager._periodic_task is None
        assert manager._owner_task is None
        manifest = json.loads(manager.manifest_path.read_text())
        assert manifest["phase"] == "invalid"
        assert manifest["failure_type"] == "CheckpointSnapshotInvalid"

    asyncio.run(scenario())


def test_snapshot_capture_has_hard_timeout_and_aborts_staging(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        class CaptureTimeoutAgent(FakeAgent):
            async def exec_as_agent(
                self, environment, *, command, env, timeout_sec=None,
            ):
                if "/usr/bin/timeout" in command:
                    self.commands.append(command)
                    self.agent_exec_timeouts.append(timeout_sec)
                    raise TimeoutError("capture command timed out")
                return await super().exec_as_agent(
                    environment,
                    command=command,
                    env=env,
                    timeout_sec=timeout_sec,
                )

        manager = _manager(tmp_path / "agent")
        manager.host_dir.mkdir(mode=0o700)
        manager.runtime_dir = PurePosixPath("/run/dradar-checkpoint/test")
        manager.supervisor_sha256 = "f" * 64
        manager.base_commit = BASE_COMMIT
        manager.agent_identity = AgentIdentity(
            os.getuid(), os.getgid(), _numeric_groups(),
        )
        actions: list[str] = []

        async def fake_exec_root(_environment, *, command: str) -> None:
            action, generation = command.rsplit(" ", 2)[-2:]
            actions.append(action)
            manager.staging_host_dir.mkdir(mode=0o700, exist_ok=True)
            internal_lock = manager.staging_host_dir / "snapshot.lock"
            stage = manager._stage_path(generation)
            if action == "prepare":
                internal_lock.mkdir(mode=0o700)
                stage.mkdir(mode=0o700)
            elif action == "abort":
                shutil.rmtree(stage, ignore_errors=True)
                if internal_lock.is_dir():
                    internal_lock.rmdir()

        manager._exec_root = fake_exec_root  # type: ignore[method-assign]
        agent = CaptureTimeoutAgent()

        with pytest.raises(TimeoutError, match="capture command timed out"):
            await DurableCheckpoint._snapshot_once(
                manager, agent, FakeEnvironment(), {},
            )

        capture_command = next(
            command for command in agent.commands
            if "/usr/bin/timeout" in command
        )
        assert "--signal=TERM" in capture_command
        assert (
            f"--kill-after={checkpoint_mod._CAPTURE_KILL_GRACE_SEC}s"
            in capture_command
        )
        assert f"{checkpoint_mod._CAPTURE_WORK_TIMEOUT_SEC}s" in capture_command
        assert agent.agent_exec_timeouts[-1] == (
            checkpoint_mod._CAPTURE_EXEC_TIMEOUT_SEC
        )
        assert actions == ["prepare", "abort"]
        assert not manager.staging_host_dir.exists() or not any(
            manager.staging_host_dir.iterdir()
        )
        assert (manager.host_dir / "snapshot.lock").is_dir()

    asyncio.run(scenario())


def test_snapshot_prepare_cancellation_aborts_container_staging(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "agent")
        manager.host_dir.mkdir(mode=0o700)
        manager.runtime_dir = PurePosixPath("/run/dradar-checkpoint/test")
        manager.supervisor_sha256 = "f" * 64
        manager.base_commit = BASE_COMMIT
        manager.agent_identity = AgentIdentity(
            os.getuid(), os.getgid(), _numeric_groups(),
        )
        prepared = asyncio.Event()
        never = asyncio.Event()
        generation_seen: list[str] = []

        async def fake_exec_root(_environment, *, command: str) -> None:
            action, generation = command.rsplit(" ", 2)[-2:]
            generation_seen[:] = [generation]
            manager.staging_host_dir.mkdir(mode=0o700, exist_ok=True)
            internal_lock = manager.staging_host_dir / "snapshot.lock"
            stage = manager._stage_path(generation)
            if action == "prepare":
                internal_lock.mkdir(mode=0o700)
                stage.mkdir(mode=0o700)
                prepared.set()
                await never.wait()
            elif action == "abort":
                shutil.rmtree(stage, ignore_errors=True)
                if internal_lock.is_dir():
                    internal_lock.rmdir()

        manager._exec_root = fake_exec_root  # type: ignore[method-assign]
        running = asyncio.create_task(
            DurableCheckpoint._snapshot_once(
                manager, FakeAgent(), FakeEnvironment(), {},
            )
        )
        await prepared.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        generation = generation_seen[0]
        assert not manager._stage_path(generation).exists()
        assert not (manager.staging_host_dir / "snapshot.lock").exists()
        # The host lock is deliberately sticky until the caller marks the
        # checkpoint invalid; scanners must never observe a partial commit.
        assert (manager.host_dir / "snapshot.lock").is_dir()

    asyncio.run(scenario())


def test_finish_cancellation_marks_invalid_after_final_writer_is_reaped(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "agent")
        agent = FakeAgent()
        environment = FakeEnvironment()
        original_snapshot = manager._snapshot_once
        final_started = asyncio.Event()
        never = asyncio.Event()
        final_reaped = asyncio.Event()
        calls = 0

        async def controlled_snapshot(*args, **kwargs) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await original_snapshot(*args, **kwargs)
                return
            final_started.set()
            try:
                await never.wait()
            finally:
                final_reaped.set()

        manager._snapshot_once = controlled_snapshot  # type: ignore[method-assign]
        await manager.start(agent, environment, {})
        finishing = asyncio.create_task(
            manager.finish(
                agent, environment, {}, completed=False,
                failure=KeyboardInterrupt(),
            )
        )
        await final_started.wait()
        finishing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finishing

        assert final_reaped.is_set()
        assert manager._periodic_task is None
        manifest = json.loads(manager.manifest_path.read_text())
        assert manifest["phase"] == "invalid"
        assert manifest["failure_type"] == "CheckpointSnapshotInvalid"
        assert (manager.host_dir / "invalid-snapshot").is_file()

    asyncio.run(scenario())


def test_finish_durably_reaps_finalizer_through_repeated_cancellation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = _manager(tmp_path / "agent")
        entered = asyncio.Event()
        release = asyncio.Event()
        reaped = asyncio.Event()

        async def controlled_finish(*args, **kwargs) -> None:
            entered.set()
            try:
                await release.wait()
            finally:
                reaped.set()

        manager.finish = controlled_finish  # type: ignore[method-assign]
        finishing = asyncio.create_task(
            manager.finish_durably(
                FakeAgent(), FakeEnvironment(), {},
                completed=False, failure=KeyboardInterrupt(),
            )
        )
        await entered.wait()
        finishing.cancel()
        await asyncio.sleep(0)
        finishing.cancel()
        await asyncio.sleep(0)
        assert not reaped.is_set()
        assert not finishing.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await finishing
        assert reaped.is_set()

    asyncio.run(scenario())
