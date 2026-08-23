from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dradar.checkpoint_container_bundle_v2 import (
    CONTAINER_HELPER_SCHEMA_V2,
    build_checkpoint_container_bundle_v2,
)


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


def _fixture(tmp_path: Path):
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
    (worktree / "new.txt").write_text("untracked\n")
    sessions = root / "tmp/codex-home/sessions"
    sessions.mkdir(parents=True)
    (sessions / "state.jsonl").write_text('{"step":2}\n')
    sealed = root / "run/dradar-checkpoint-v2/checkpoint-0001/sealed"
    sealed.mkdir(parents=True, mode=0o700)
    for parent in (
        root / "run",
        root / "run/dradar-checkpoint-v2",
        root / "run/dradar-checkpoint-v2/checkpoint-0001",
        sealed,
    ):
        parent.chmod(0o700)
    return root, worktree, base


def _request(base: str):
    del base
    return {
        "checkpoint_id": "checkpoint-0001",
        "checkpoint_lineage_id": "lineage-0001",
        "snapshot_generation": 1,
        "capture_id": "capture-0001",
        "identity_fingerprint": "a" * 64,
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
        "recovery_capability": "NATIVE_VALID",
        "native_state_schema": "codex-sessions/1",
        "captured_at": "2026-08-23T12:00:00+00:00",
    }


def _run_helper(bundle: Path, spec_path: Path, spec: dict):
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    spec_path.chmod(0o600)
    result = subprocess.run(
        [sys.executable, os.fspath(bundle), os.fspath(spec_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, (result.stdout, result.stderr)
    return result, json.loads(lines[0])


def test_bundle_is_deterministic_and_runs_capture_then_offline_restore(
    tmp_path: Path,
) -> None:
    first = build_checkpoint_container_bundle_v2(
        tmp_path / "helper-1.pyz", allow_test_root=True,
    )
    second = build_checkpoint_container_bundle_v2(
        tmp_path / "helper-2.pyz", allow_test_root=True,
    )
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()

    root, source, base = _fixture(tmp_path)
    request = _request(base)
    capture_spec = {
        "schema": CONTAINER_HELPER_SCHEMA_V2,
        "operation": "capture",
        "filesystem_root": os.fspath(root),
        "harness": "codex",
        "provider": "openai",
        "worktree_path": "/app",
        "capture_root_path": (
            "/run/dradar-checkpoint-v2/checkpoint-0001/capture-0001"
        ),
        "export_path": (
            "/run/dradar-checkpoint-v2/checkpoint-0001/sealed/capture-0001.tar.gz"
        ),
        "base_commit": base,
        "captured_at": request["captured_at"],
        "session_id": "thread-0001",
        "sensitive_values": [],
        "request": request,
    }
    capture_result, captured = _run_helper(
        first.path, tmp_path / "capture-spec.json", capture_spec,
    )
    assert capture_result.returncode == 0, capture_result.stderr
    assert captured["ok"] is True
    assert captured["recovery_capability"] == "NATIVE_VALID"
    assert captured["export"]["remote_path"] == capture_spec["export_path"]
    assert not (tmp_path / "capture-spec.json").exists()
    assert not (root / capture_spec["capture_root_path"].lstrip("/")).exists()

    restore = tmp_path / "container/app-restore"
    _git(tmp_path, "clone", "-q", os.fspath(source), os.fspath(restore))
    restore.chmod(0o700)
    restore_spec = {
        "schema": CONTAINER_HELPER_SCHEMA_V2,
        "operation": "restore",
        "filesystem_root": os.fspath(root),
        "harness": "codex",
        "provider": "openai",
        "worktree_path": "/app-restore",
        "state_root_path": "/run/dradar-checkpoint-v2/restore/state",
        "storage_root_path": "/run/dradar-checkpoint-v2/restore/storage",
        "archive_path": capture_spec["export_path"],
        "base_commit": base,
        "expected_identity_fingerprint": request["identity_fingerprint"],
        "request": request,
        "export": captured["export"],
    }
    restore_result, restored = _run_helper(
        first.path, tmp_path / "restore-spec.json", restore_spec,
    )
    assert restore_result.returncode == 0, restore_result.stderr
    assert restored["ok"] is True
    assert restored["paid_execution_started"] is False
    assert restored["session_id"] == "thread-0001"
    assert (restore / "tracked.txt").read_text() == "after\n"
    assert (restore / "new.txt").read_text() == "untracked\n"
    assert (
        root / "run/dradar-checkpoint-v2/restore/state/sessions/state.jsonl"
    ).is_file()


def test_production_bundle_rejects_a_test_filesystem_root(tmp_path: Path) -> None:
    bundle = build_checkpoint_container_bundle_v2(tmp_path / "helper.pyz")
    root, _worktree, base = _fixture(tmp_path)
    spec = {
        "schema": CONTAINER_HELPER_SCHEMA_V2,
        "operation": "capture",
        "filesystem_root": os.fspath(root),
        "harness": "codex",
        "provider": "openai",
        "worktree_path": "/app",
        "capture_root_path": "/run/dradar-checkpoint-v2/cp/capture-0001",
        "export_path": "/run/dradar-checkpoint-v2/cp/sealed/capture-0001.tar.gz",
        "base_commit": base,
        "captured_at": "2026-08-23T12:00:00+00:00",
        "session_id": "thread-0001",
        "sensitive_values": [],
        "request": _request(base),
    }
    result, payload = _run_helper(bundle.path, tmp_path / "spec.json", spec)
    assert result.returncode == 64
    assert payload == {
        "schema": CONTAINER_HELPER_SCHEMA_V2,
        "ok": False,
        "stage": "capture",
        "code": "helper_filesystem_root_invalid",
    }
