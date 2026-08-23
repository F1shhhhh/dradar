from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from dradar.checkpoint_adapter_runtime_v2 import (
    create_adapter_capture_root_v2,
    restore_adapter_capture_offline_v2,
)
from dradar.checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from dradar.checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneError,
    publish_checkpoint_export_v2,
    seal_checkpoint_export_v2,
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


def _filesystem(tmp_path: Path):
    root = tmp_path / "container"
    worktree = root / "app"
    worktree.mkdir(parents=True, mode=0o700)
    worktree.chmod(0o700)
    _git(worktree, "init", "-q")
    (worktree / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(worktree, "add", "tracked.txt")
    _git(worktree, "commit", "-q", "-m", "base")
    base = _git(worktree, "rev-parse", "HEAD")
    (worktree / "tracked.txt").write_text("after\n", encoding="utf-8")
    (worktree / "generated").mkdir()
    (worktree / "generated" / "note.txt").write_text(
        "untracked progress\n", encoding="utf-8",
    )
    sessions = root / "tmp/codex-home/sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text(
        '{"event":"step","index":2}\n', encoding="utf-8",
    )
    return root, worktree, base


def _request(capability: str) -> CheckpointCaptureRequestV2:
    return CheckpointCaptureRequestV2(
        checkpoint_id="checkpoint-0001",
        checkpoint_lineage_id="lineage-0001",
        snapshot_generation=1,
        capture_id="capture-0001",
        identity_fingerprint="a" * 64,
        checkpoint_abi="dradar-checkpoint-v2/codex/1",
        recovery_capability=capability,
        native_state_schema="codex-sessions/1",
        captured_at="2026-08-23T12:00:00+00:00",
    )


def _capture_publish(tmp_path: Path):
    root, worktree, base = _filesystem(tmp_path)
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    summary = create_adapter_capture_root_v2(
        filesystem_root=root,
        worktree_path="/app",
        capture_root_path=(
            "/run/dradar-checkpoint-v2/checkpoint-0001/capture-0001"
        ),
        contract=contract,
        base_commit=base,
        captured_at="2026-08-23T12:00:00+00:00",
        session_id="thread-0001",
    )
    request = _request(summary.recovery_capability)
    native_root = root / "run/dradar-checkpoint-v2"
    sealed_root = native_root / "checkpoint-0001/sealed"
    sealed_root.mkdir(mode=0o700)
    archive = sealed_root / "capture-0001.tar.gz"
    exported = seal_checkpoint_export_v2(
        summary.capture_root,
        archive,
        request,
        container_export_root=native_root,
    )
    published = publish_checkpoint_export_v2(
        archive,
        tmp_path / "host-store",
        request,
        replace(
            exported,
            remote_path=(
                "/run/dradar-checkpoint-v2/checkpoint-0001/sealed/"
                "capture-0001.tar.gz"
            ),
        ),
        authoritative=False,
    )
    return root, worktree, base, contract, summary, request, published


def test_codex_adapter_capture_and_offline_restore_full_lifecycle(tmp_path: Path):
    root, worktree, base, contract, summary, request, published = _capture_publish(
        tmp_path,
    )
    assert summary.recovery_capability == "NATIVE_VALID"
    assert summary.present_artifacts == {"sessions"}
    assert summary.workspace_patch_bytes > 0
    assert summary.untracked_files == 1
    assert summary.untracked_bytes == len(b"untracked progress\n")
    assert (published.payload_root / "workspace.patch").is_file()

    restore = tmp_path / "restore-worktree"
    _git(tmp_path, "clone", "-q", os.fspath(worktree), os.fspath(restore))
    # Clone sees the committed base only, not source working-tree mutations.
    restore.chmod(0o755)
    state = tmp_path / "restored-provider-state"
    evidence = restore_adapter_capture_offline_v2(
        published=published,
        contract=contract,
        destination_worktree=restore,
        destination_state_root=state,
        expected_identity_fingerprint=request.identity_fingerprint,
        base_commit=base,
    )
    assert (restore / "tracked.txt").read_text() == "after\n"
    assert (restore / "generated/note.txt").read_text() == "untracked progress\n"
    assert (state / "sessions/rollout.jsonl").is_file()
    assert evidence.session_id == "thread-0001"
    assert evidence.recovery_capability == "NATIVE_VALID"
    assert evidence.restored_untracked_files == 1
    assert evidence.paid_execution_started is False


def test_capture_accepts_non_private_real_worktree(tmp_path: Path):
    root, worktree, base = _filesystem(tmp_path)
    worktree.chmod(0o755)
    contract = checkpoint_adapter_contract_v2("codex", "openai")

    summary = create_adapter_capture_root_v2(
        filesystem_root=root,
        worktree_path="/app",
        capture_root_path="/run/dradar-checkpoint-v2/cp/capture-public-worktree",
        contract=contract,
        base_commit=base,
        captured_at="2026-08-23T12:00:00+00:00",
        session_id="thread-0001",
    )

    assert summary.capture_root.stat().st_mode & 0o777 == 0o700


def test_capture_rejects_symlink_worktree(tmp_path: Path):
    root, worktree, base = _filesystem(tmp_path)
    real_worktree = root / "real-app"
    worktree.rename(real_worktree)
    worktree.symlink_to(real_worktree, target_is_directory=True)
    contract = checkpoint_adapter_contract_v2("codex", "openai")

    with pytest.raises(CheckpointDataPlaneError, match="adapter_directory_unsafe"):
        create_adapter_capture_root_v2(
            filesystem_root=root,
            worktree_path="/app",
            capture_root_path="/run/dradar-checkpoint-v2/cp/capture-symlink-worktree",
            contract=contract,
            base_commit=base,
            captured_at="2026-08-23T12:00:00+00:00",
            session_id="thread-0001",
        )


def test_capture_root_must_stay_below_container_native_storage(tmp_path: Path):
    root, _worktree, base = _filesystem(tmp_path)
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    with pytest.raises(
        CheckpointDataPlaneError, match="capture_root_outside_native_storage",
    ):
        create_adapter_capture_root_v2(
            filesystem_root=root,
            worktree_path="/app",
            capture_root_path="/logs/agent/checkpoint",
            contract=contract,
            base_commit=base,
            captured_at="2026-08-23T12:00:00+00:00",
            session_id="thread-0001",
        )


def test_untracked_secret_and_symlink_fail_capture_without_leaving_stage(
    tmp_path: Path,
):
    root, worktree, base = _filesystem(tmp_path)
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    secret = "opaque-provider-secret-123456789"
    (worktree / "generated/note.txt").write_text(secret, encoding="utf-8")
    capture_path = root / "run/dradar-checkpoint-v2/cp/capture-secret"
    with pytest.raises(CheckpointDataPlaneError, match="secret_detected"):
        create_adapter_capture_root_v2(
            filesystem_root=root,
            worktree_path="/app",
            capture_root_path="/run/dradar-checkpoint-v2/cp/capture-secret",
            contract=contract,
            base_commit=base,
            captured_at="2026-08-23T12:00:00+00:00",
            session_id="thread-0001",
            sensitive_values=(secret,),
        )
    assert not capture_path.exists()

    (worktree / "generated/note.txt").unlink()
    (worktree / "generated/link").symlink_to("../tracked.txt")
    with pytest.raises(CheckpointDataPlaneError, match="untracked_file_unsafe"):
        create_adapter_capture_root_v2(
            filesystem_root=root,
            worktree_path="/app",
            capture_root_path="/run/dradar-checkpoint-v2/cp/capture-link",
            contract=contract,
            base_commit=base,
            captured_at="2026-08-23T12:00:00+00:00",
            session_id="thread-0001",
        )


def test_restore_requires_a_clean_exact_base_worktree(tmp_path: Path):
    _root, source, base, contract, _summary, request, published = _capture_publish(
        tmp_path,
    )
    restore = tmp_path / "dirty-restore"
    _git(tmp_path, "clone", "-q", os.fspath(source), os.fspath(restore))
    restore.chmod(0o700)
    (restore / "local.txt").write_text("must not be overwritten\n")
    with pytest.raises(CheckpointDataPlaneError, match="restore_worktree_not_clean"):
        restore_adapter_capture_offline_v2(
            published=published,
            contract=contract,
            destination_worktree=restore,
            destination_state_root=tmp_path / "state",
            expected_identity_fingerprint=request.identity_fingerprint,
            base_commit=base,
        )
    assert not (tmp_path / "state").exists()
    assert (restore / "local.txt").read_text() == "must not be overwritten\n"


def test_capture_is_deterministic_for_same_repository_state(tmp_path: Path):
    root, _worktree, base = _filesystem(tmp_path)
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    first = create_adapter_capture_root_v2(
        filesystem_root=root,
        worktree_path="/app",
        capture_root_path="/run/dradar-checkpoint-v2/cp/capture-0001",
        contract=contract,
        base_commit=base,
        captured_at="2026-08-23T12:00:00+00:00",
        session_id="thread-0001",
    )
    second = create_adapter_capture_root_v2(
        filesystem_root=root,
        worktree_path="/app",
        capture_root_path="/run/dradar-checkpoint-v2/cp/capture-0002",
        contract=contract,
        base_commit=base,
        captured_at="2026-08-23T12:00:00+00:00",
        session_id="thread-0001",
    )
    for name in ("workspace.patch", "untracked.tar.gz", "progress.json"):
        assert hashlib.sha256((first.capture_root / name).read_bytes()).digest() == (
            hashlib.sha256((second.capture_root / name).read_bytes()).digest()
        )


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
def test_every_reviewed_harness_contract_builds_material_native_state(
    tmp_path: Path,
    harness: str,
    provider: str,
) -> None:
    root, _worktree, base = _filesystem(tmp_path)
    shutil_target = root / "tmp/codex-home/sessions"
    if shutil_target.exists():
        shutil.rmtree(shutil_target)
    contract = checkpoint_adapter_contract_v2(harness, provider)
    for artifact in contract.artifacts:
        logical = Path(*PurePosixPath(artifact.source_path).parts[1:])
        source = root / logical
        if artifact.kind == "directory":
            source.mkdir(parents=True, exist_ok=True)
            (source / "state.bin").write_bytes(b"material native state\n")
        else:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"material native state\n")
    summary = create_adapter_capture_root_v2(
        filesystem_root=root,
        worktree_path="/app",
        capture_root_path=f"/run/dradar-checkpoint-v2/{harness}/capture-0001",
        contract=contract,
        base_commit=base,
        captured_at="2026-08-23T12:00:00+00:00",
        session_id="session-0001",
    )
    assert summary.present_artifacts == contract.artifact_names
    assert summary.recovery_capability == "NATIVE_VALID"


@pytest.mark.parametrize(
    ("harness", "provider", "expected"),
    [
        ("codex", "openai", "WORKSPACE_ONLY"),
        ("codex", "deepseek", "WORKSPACE_ONLY"),
        ("dsh", "deepseek", "NONE"),
        ("kimi-code", "kimi-subscription", "NONE"),
        ("zcode", "bigmodel-coding-plan", "WORKSPACE_ONLY"),
    ],
)
def test_missing_native_state_degrades_per_harness_without_guessing(
    tmp_path: Path,
    harness: str,
    provider: str,
    expected: str,
) -> None:
    root, _worktree, base = _filesystem(tmp_path)
    # Remove the Codex fixture state; no Harness has native state in this case.
    sessions = root / "tmp/codex-home/sessions"
    if sessions.exists():
        for path in sorted(sessions.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        sessions.rmdir()
    contract = checkpoint_adapter_contract_v2(harness, provider)
    summary = create_adapter_capture_root_v2(
        filesystem_root=root,
        worktree_path="/app",
        capture_root_path=f"/run/dradar-checkpoint-v2/{harness}/capture-none",
        contract=contract,
        base_commit=base,
        captured_at="2026-08-23T12:00:00+00:00",
        session_id="session-0001",
    )
    assert summary.present_artifacts == frozenset()
    assert summary.recovery_capability == expected
