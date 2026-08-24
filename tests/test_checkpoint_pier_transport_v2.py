from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dradar.checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from dradar.checkpoint_pier_transport_v2 import (
    PierContainerCheckpointExporterV2,
    PierContainerCheckpointRestorerV2,
)
from dradar.checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneV2,
    CheckpointRestoreRequestV2,
)
from dradar.checkpoint_v2 import negotiate_checkpoint_activation_v2


@dataclass
class _Result:
    return_code: int
    stdout: str = ""
    stderr: str = ""


class LocalPierEnvironment:
    """Public Pier surface emulator; root commands stay inside ``root``."""

    default_user = "1000:1000"

    def __init__(self, root: Path):
        self.root = root
        self.commands = []
        self.downloads = 0

    def _path(self, logical: str) -> Path:
        return self.root / logical.lstrip("/")

    async def upload_file(self, source: Path, destination: str):
        target = self._path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.parent.chmod(0o700)
        shutil.copyfile(source, target)
        target.chmod(0o600)

    async def download_file(self, source: str, destination: Path):
        self.downloads += 1
        shutil.copyfile(self._path(source), destination)

    @staticmethod
    def _logical_paths(command: str, suffix: str) -> list[str]:
        return sorted(set(re.findall(
            rf"(/(?:run/dradar-checkpoint-v2|tmp)/[A-Za-z0-9._/-]*{re.escape(suffix)})",
            command,
        )))

    async def exec(self, *, command, user, env, cwd, timeout_sec):
        assert user == "root"
        assert cwd == "/"
        assert env["HOME"] == "/root"
        assert timeout_sec <= 180
        self.commands.append(command)

        # Helper/spec installation. Derive the reviewed destination from the
        # exact operation IDs already embedded by the transport.
        if ".pyz.upload" in command and "/helper.pyz" in command:
            uploads = self._logical_paths(command, ".pyz.upload")
            specs = self._logical_paths(command, ".json.upload")
            helpers = self._logical_paths(command, "/helper.pyz")
            remote_specs = self._logical_paths(command, "/spec.json")
            assert len(uploads) == len(specs) == len(helpers) == len(remote_specs) == 1
            for source, destination in (
                (uploads[0], helpers[0]),
                (specs[0], remote_specs[0]),
            ):
                target = self._path(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.parent.chmod(0o700)
                shutil.copyfile(self._path(source), target)
                target.chmod(0o600)
                self._path(source).unlink(missing_ok=True)
            # Capture sealing requires a pre-existing private native export dir.
            import json
            spec = json.loads(self._path(remote_specs[0]).read_text())
            if spec["operation"] == "capture":
                export = self._path(spec["export_path"])
                export.parent.mkdir(parents=True, exist_ok=True)
                export.parent.chmod(0o700)
                current = self.root / "run/dradar-checkpoint-v2"
                while current != self.root / "run":
                    current.chmod(0o700)
                    current = current.parent
            return _Result(0)

        # Restore archive installation.
        if ".archive.upload" in command and "incoming.tar.gz" in command:
            uploads = self._logical_paths(command, ".archive.upload")
            destinations = self._logical_paths(command, "incoming.tar.gz")
            assert len(uploads) == len(destinations) == 1
            target = self._path(destinations[0])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.parent.chmod(0o700)
            shutil.copyfile(self._path(uploads[0]), target)
            target.chmod(0o600)
            self._path(uploads[0]).unlink(missing_ok=True)
            return _Result(0)

        # Execute only the installed checksum-pinned zipapp.
        match = re.search(
            r"/usr/bin/python3 (/run/dradar-checkpoint-v2/[^ ]+/helper[.]pyz) "
            r"(/run/dradar-checkpoint-v2/[^ ]+/spec[.]json)",
            command,
        )
        if match:
            result = subprocess.run(
                [sys.executable, os.fspath(self._path(match.group(1))),
                 os.fspath(self._path(match.group(2)))],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
            return _Result(result.returncode, result.stdout, result.stderr)

        # Exact cleanup/discard paths; no shell is executed on the host.
        if "/usr/bin/rm" in command:
            for logical in re.findall(
                r"(/(?:run/dradar-checkpoint-v2|tmp)/[A-Za-z0-9._/-]+)",
                command,
            ):
                path = self._path(logical.rstrip(".;'"))
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            return _Result(0)

        # Layout commands are represented by parent creation at install time.
        return _Result(0)


def _git(path: Path, *args: str) -> str:
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "Checkpoint Test",
        "GIT_AUTHOR_EMAIL": "checkpoint@example.invalid",
        "GIT_COMMITTER_NAME": "Checkpoint Test",
        "GIT_COMMITTER_EMAIL": "checkpoint@example.invalid",
    })
    result = subprocess.run(
        ["git", "-C", os.fspath(path), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _container(tmp_path: Path):
    root = tmp_path / "container"
    worktree = root / "app"
    worktree.mkdir(parents=True, mode=0o700)
    worktree.chmod(0o700)
    _git(worktree, "init", "-q")
    (worktree / "tracked.txt").write_text("before\n")
    _git(worktree, "add", "tracked.txt")
    _git(worktree, "commit", "-q", "-m", "base")
    base = _git(worktree, "rev-parse", "HEAD")
    (worktree / "tracked.txt").write_text("after\n")
    (worktree / "new.txt").write_text("new state\n")
    sessions = root / "tmp/codex-home/sessions"
    sessions.mkdir(parents=True)
    (sessions / "state.jsonl").write_text('{"step":2}\n')
    return root, worktree, base


def _request():
    return CheckpointCaptureRequestV2(
        checkpoint_id="checkpoint-0001",
        checkpoint_lineage_id="lineage-0001",
        snapshot_generation=1,
        capture_id="capture-0001",
        identity_fingerprint="a" * 64,
        checkpoint_abi="dradar-checkpoint-v2/codex/1",
        recovery_capability="NATIVE_VALID",
        native_state_schema="codex-sessions/1",
        captured_at="2026-08-23T12:00:00+00:00",
    )


def test_pier_transport_capture_download_publish_and_discard(tmp_path: Path):
    root, _worktree, base = _container(tmp_path)
    environment = LocalPierEnvironment(root)
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    exporter = PierContainerCheckpointExporterV2(
        environment,
        contract=contract,
        base_commit=base,
        session_id="thread-0001",
        _test_filesystem_root=root,
    )
    plane = CheckpointDataPlaneV2(
        activation=negotiate_checkpoint_activation_v2(
            local_mode="observe", server_mode="observe",
        ),
        storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "sealed"
    assert observed.published is not None
    assert observed.published.archive_path.is_file()
    assert environment.downloads == 1
    assert not environment._path(
        "/run/dradar-checkpoint-v2/checkpoint-0001/sealed/capture-0001.tar.gz"
    ).exists()


def test_pier_transport_install_failure_is_stage_specific_and_local_only(
    tmp_path: Path,
) -> None:
    root, _worktree, base = _container(tmp_path)

    class FailingInstallEnvironment(LocalPierEnvironment):
        async def exec(self, *, command, user, env, cwd, timeout_sec):
            if ".pyz.upload" in command and "/helper.pyz" in command:
                return _Result(
                    23,
                    stdout="untrusted install stdout",
                    stderr="untrusted install stderr",
                )
            return await super().exec(
                command=command,
                user=user,
                env=env,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )

    environment = FailingInstallEnvironment(root)
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    exporter = PierContainerCheckpointExporterV2(
        environment,
        contract=contract,
        base_commit=base,
        session_id="thread-0001",
        _test_filesystem_root=root,
    )
    plane = CheckpointDataPlaneV2(
        activation=negotiate_checkpoint_activation_v2(
            local_mode="observe", server_mode="observe",
        ),
        storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))

    assert observed.status == "failed"
    assert observed.stage == "capture"
    assert observed.code == "transport_helper_install_failed"
    diagnostic = (tmp_path / "host" / "diagnostics.jsonl").read_text()
    assert '"operation":"helper_install"' in diagnostic
    assert '"exit_code":23' in diagnostic
    assert "untrusted install" not in diagnostic


def test_pier_transport_offline_restore_never_starts_paid_execution(tmp_path: Path):
    source_root, source_worktree, base = _container(tmp_path / "source")
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    capture_environment = LocalPierEnvironment(source_root)
    exporter = PierContainerCheckpointExporterV2(
        capture_environment,
        contract=contract,
        base_commit=base,
        session_id="thread-0001",
        _test_filesystem_root=source_root,
    )
    capture_plane = CheckpointDataPlaneV2(
        activation=negotiate_checkpoint_activation_v2(
            local_mode="restore-test", server_mode="restore-test",
        ),
        storage_root=tmp_path / "host",
    )
    captured = asyncio.run(capture_plane.observe_capture(_request(), exporter))
    assert captured.published is not None

    restore_root = tmp_path / "restore-container"
    restore_root.mkdir(mode=0o700)
    restore_worktree = restore_root / "app"
    _git(tmp_path, "clone", "-q", os.fspath(source_worktree), os.fspath(restore_worktree))
    restore_worktree.chmod(0o700)
    restore_environment = LocalPierEnvironment(restore_root)
    restorer = PierContainerCheckpointRestorerV2(
        restore_environment,
        contract=contract,
        base_commit=base,
        _test_filesystem_root=restore_root,
    )
    request = CheckpointRestoreRequestV2(
        published=captured.published,
        expected_identity_fingerprint="a" * 64,
        restore_id="restore-0001",
    )
    restored = asyncio.run(capture_plane.observe_offline_restore(request, restorer))
    assert restored.status == "verified"
    assert restored.paid_execution_authorized is False
    assert restored.evidence is not None
    assert restored.evidence.paid_execution_started is False
    assert (restore_worktree / "tracked.txt").read_text() == "after\n"
    assert (restore_worktree / "new.txt").read_text() == "new state\n"
    assert restorer.last_state_root is not None
    assert restore_environment._path(restorer.last_state_root).joinpath(
        "sessions/state.jsonl"
    ).is_file()
