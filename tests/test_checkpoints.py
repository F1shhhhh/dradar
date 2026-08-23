import argparse
import json
import os
import threading
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from dradar import artifact_staging, checkpoints, pending, refill, runloop
from dradar.api_client import ApiClient


def _privatize_checkpoint_tree(trial: Path, checkpoint: Path) -> None:
    if os.name == "nt":
        return
    trial.chmod(0o700)
    for current, directories, files in os.walk(checkpoint, followlinks=False):
        Path(current).chmod(0o700)
        for name in directories:
            candidate = Path(current) / name
            if not candidate.is_symlink():
                candidate.chmod(0o700)
        for name in files:
            candidate = Path(current) / name
            if candidate.is_file() and not candidate.is_symlink():
                candidate.chmod(0o600)


def _make_checkpoint(
    home: Path,
    assignment_id: str,
    *,
    checkpoint_id: str = "checkpoint-12345678",
    phase: str = "paused",
    generation: int = 0,
    updated_at: str | None = None,
    suffix: str = "one",
    layout: str = "new",
    manifest_overrides: dict | None = None,
) -> checkpoints.Checkpoint:
    job = home / "work" / "jobs" / f"a{assignment_id}-{suffix}"
    trial = job / "task__trial"
    checkpoint = (
        trial / "checkpoint"
        if layout == "new"
        else trial / "agent" / "checkpoint"
    )
    checkpoint.mkdir(parents=True)
    heartbeat = updated_at or datetime.now(timezone.utc).isoformat()
    manifest = checkpoint / "checkpoint.json"
    payload = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "assignment_id": assignment_id,
        "phase": phase,
        "created_at": "2026-07-16T00:00:00Z",
        "updated_at": heartbeat,
        "last_heartbeat": heartbeat,
        "model": "gpt-test",
        "task_id": "task-1",
        "effort": "high",
        "harness": "codex",
        "provider": "openai",
        "agent_version": "1.2.3",
        "base_commit": "a" * 40,
        "resume_generation": generation,
        "resume_count": generation,
        "session_id": "thread-12345678",
        "workspace_patch": "workspace.patch",
        "untracked_archive": "untracked.tar.gz",
        "state_dir": "provider-state",
        "events_file": "events.jsonl",
    }
    payload.update(manifest_overrides or {})
    manifest.write_text(json.dumps(payload))
    _privatize_checkpoint_tree(trial, checkpoint)
    return next(item for item in checkpoints.scan(home)
                if item.manifest_path == manifest)


def _assignment(assignment_id: str, generation: int = 0) -> dict:
    return {
        "assignment_id": assignment_id,
        "nonce": "server-only-nonce",
        "task_id": "task-1",
        "model": "gpt-test",
        "effort": "high",
        "resume_generation": generation,
        "checkpoint_id": "checkpoint-12345678",
        "deep_swe_commit": None,
    }


def _args(**overrides):
    values = dict(
        dev_agent=None, keep=False, allow_task_drift=False,
        parallel=False, assignment=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_scan_reads_metadata_but_never_requires_nonce_or_account_token(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "1" * 32, generation=4)
    assert item.valid
    assert item.assignment_id == "1" * 32
    assert item.resume_generation == 4
    raw = item.manifest_path.read_text().lower()
    assert "nonce" not in raw
    assert "account_token" not in raw


def test_scan_discovers_sibling_checkpoint_layout(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "2" * 32, layout="new")

    assert item.valid
    assert item.checkpoint_dir == item.trial_dir / "checkpoint"
    assert item.trial_dir.name == "task__trial"
    assert item.job_dir == item.trial_dir.parent


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership boundary")
def test_scan_rejects_world_writable_sibling_checkpoint(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "0" * 32, layout="new")
    item.checkpoint_dir.chmod(0o777)

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert not loaded.valid
    assert loaded.phase == "invalid"
    assert "host-private" in (loaded.invalid_reason or "")


def test_scan_prefers_new_layout_and_deduplicates_legacy_copy(tmp_path: Path):
    assignment_id = "3" * 32
    legacy = _make_checkpoint(tmp_path, assignment_id, layout="legacy")
    current = _make_checkpoint(tmp_path, assignment_id, layout="new")

    loaded = checkpoints.scan(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].manifest_path == current.manifest_path
    assert loaded[0].manifest_path != legacy.manifest_path


@pytest.mark.parametrize("sibling_kind", ["empty", "dangling"])
def test_incomplete_sibling_does_not_hide_untrusted_legacy_checkpoint(
    tmp_path: Path, sibling_kind: str,
) -> None:
    legacy = _make_checkpoint(tmp_path, "4" * 32, layout="legacy")
    sibling = legacy.trial_dir / "checkpoint"
    if sibling_kind == "empty":
        sibling.mkdir(mode=0o700)
    else:
        sibling.symlink_to(legacy.trial_dir / "missing-checkpoint")

    loaded = checkpoints.scan(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].manifest_path == legacy.manifest_path
    assert loaded[0].checkpoint_dir == legacy.trial_dir / "agent" / "checkpoint"


@pytest.mark.skipif(os.name == "nt", reason="POSIX special file semantics")
def test_special_sibling_does_not_hide_untrusted_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    legacy = _make_checkpoint(tmp_path, "5" * 32, layout="legacy")
    os.mkfifo(legacy.trial_dir / "checkpoint")

    loaded = checkpoints.scan(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].manifest_path == legacy.manifest_path


def test_scan_reads_extended_harness_identity(tmp_path: Path):
    item = _make_checkpoint(
        tmp_path,
        "e" * 32,
        manifest_overrides={
            "harness": "kimi-code",
            "provider": "kimi-subscription",
            "agent_version": "0.36.1",
            "session_id": "session-12345678",
        },
    )
    assert item.harness == "kimi-code"
    assert item.provider == "kimi-subscription"
    assert item.agent_version == "0.36.1"
    assert item.session_id == "session-12345678"


def test_scan_omitted_provider_state_forces_workspace_only_resume(tmp_path: Path):
    item = _make_checkpoint(
        tmp_path,
        "a" * 32,
        manifest_overrides={
            "harness": "zcode",
            "provider": "bigmodel-coding-plan",
            "session_id": "zcode-session-1234",
        },
    )
    (item.checkpoint_dir / "session-id").write_text(
        "zcode-sidecar-1234\n", encoding="utf-8",
    )
    (item.checkpoint_dir / "session-omitted-sensitive").write_text(
        "provider state omitted\n", encoding="utf-8",
    )
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)

    loaded = checkpoints.scan(tmp_path)[0]

    assert loaded.valid
    assert loaded.session_id is None


def test_scan_reads_session_from_safe_sidecar(tmp_path: Path):
    item = _make_checkpoint(
        tmp_path,
        "d" * 32,
        manifest_overrides={
            "harness": "kimi-code",
            "provider": "kimi-subscription",
            "agent_version": "0.36.1",
        },
    )
    (item.checkpoint_dir / "session-id").write_text(
        "session_832d7f94-ab9a-4f83-b630-37a3dab65025\n",
        encoding="utf-8",
    )

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert loaded.session_id == "session_832d7f94-ab9a-4f83-b630-37a3dab65025"


def test_scan_rejects_symlinked_checkpoint_root(tmp_path: Path):
    aid = "b" * 32
    real_home = tmp_path / "real"
    real = _make_checkpoint(real_home, aid)
    linked_job = tmp_path / "work" / "jobs" / f"a{aid}-linked"
    checkpoint_parent = linked_job / "task__trial" / "agent"
    checkpoint_parent.mkdir(parents=True)
    (checkpoint_parent / "checkpoint").symlink_to(
        real.checkpoint_dir, target_is_directory=True,
    )

    assert checkpoints.scan(tmp_path) == []


def test_scan_rejects_manifest_assignment_mismatch(tmp_path: Path):
    job_assignment = "b" * 32
    item = _make_checkpoint(
        tmp_path,
        job_assignment,
        manifest_overrides={"assignment_id": "c" * 32},
    )

    assert not item.valid
    assert item.assignment_id == job_assignment
    assert "does not match job directory" in (item.invalid_reason or "")


def test_scan_reports_snapshot_failure_separately_from_secret(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "7" * 32, phase="running")
    (item.checkpoint_dir / "invalid-snapshot").write_text(
        "snapshot lock remained\n", encoding="utf-8",
    )

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert not loaded.valid
    assert loaded.phase == "invalid"
    assert "snapshot did not finish safely" in (loaded.invalid_reason or "")


@pytest.mark.parametrize("location", ["root", "payload"])
def test_scan_rejects_invalid_secret_marker_added_during_read(
    tmp_path: Path, monkeypatch, location: str,
):
    item = _make_checkpoint(tmp_path, "1" * 32, layout="new")
    generation = "a" * 32
    payload = item.checkpoint_dir / "snapshots" / generation
    payload.mkdir(parents=True)
    sidecar = payload / "session-id"
    sidecar.write_text("session-race-1234\n", encoding="utf-8")
    (item.checkpoint_dir / "current-generation").write_text(
        generation + "\n", encoding="ascii",
    )
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)
    real_read = checkpoints._read_regular_file

    def add_marker_during_read(path, *, max_bytes):
        data = real_read(path, max_bytes=max_bytes)
        if Path(path) == sidecar:
            marker_root = item.checkpoint_dir if location == "root" else payload
            (marker_root / "invalid-secret").write_text(
                "rejected\n", encoding="utf-8",
            )
        return data

    monkeypatch.setattr(checkpoints, "_read_regular_file", add_marker_during_read)

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "credential-shaped content" in (loaded.invalid_reason or "")


def test_scan_rejects_checkpoint_with_snapshot_lock(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "8" * 32, phase="running")
    (item.checkpoint_dir / "snapshot.lock").mkdir()

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert not loaded.valid
    assert loaded.phase == "invalid"
    assert loaded.assignment_id == item.assignment_id
    assert loaded.checkpoint_id == item.checkpoint_id
    assert "snapshot is incomplete" in (loaded.invalid_reason or "")


def test_scan_rejects_dangling_snapshot_lock(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "9" * 32, phase="running")
    (item.checkpoint_dir / "snapshot.lock").symlink_to(
        item.checkpoint_dir / "missing",
    )

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert not loaded.valid
    assert loaded.phase == "invalid"
    assert loaded.assignment_id == item.assignment_id
    assert loaded.checkpoint_id == item.checkpoint_id
    assert "snapshot is incomplete" in (loaded.invalid_reason or "")


def test_invalid_checkpoint_opens_refill_circuit_and_keeps_terminal_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "6" * 32
    original = _make_checkpoint(
        tmp_path,
        aid,
        phase="running",
        manifest_overrides={
            "harness": "zcode",
            "provider": "bigmodel-coding-plan",
            "model": "glm-5.3",
            "effort": "high",
        },
    )
    lock = original.checkpoint_dir / "snapshot.lock"
    lock.mkdir(mode=0o700)
    item = checkpoints.find_latest(tmp_path, aid)
    assert item is not None and not item.valid
    assignment = {
        **_assignment(aid),
        "agent": "zcode",
        "provider": "bigmodel-coding-plan",
        "model": "glm-5.3",
        "effort": "high",
    }
    refill.configure(
        tmp_path,
        volunteer_id="v1",
        refill_to=1,
        max_tasks=5,
        quota_tier="plus",
        max_estimated_quota_pct=None,
        active=[assignment],
        refill_harness="zcode",
        refill_model="glm-5.3",
        refill_effort="high",
    )
    client = _RecoveryClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)

    paused = runloop._pause_checkpoint_quietly(client, assignment)

    assert isinstance(paused, runloop._CheckpointPauseFailure)
    assert paused.family == "checkpoint_invalid"
    assert paused.discard_confirmed
    assert client.discards == [(
        aid, original.checkpoint_id, 0, "invalid",
    )]
    plan = refill.load(tmp_path)
    assert plan["status"] == refill.FAULTED_STATE
    assert plan["circuit"]["failure_family"] == "checkpoint_invalid"
    assert checkpoints.is_terminal(tmp_path, item)
    assert item.job_dir.is_dir()
    assert checkpoints.find_latest(tmp_path, aid) is None


@pytest.mark.parametrize("discard_status", [None, 409])
def test_invalid_pause_discard_failure_stays_retryable(
    tmp_path: Path, monkeypatch, discard_status: int | None,
) -> None:
    aid = "7" * 32
    original = _make_checkpoint(
        tmp_path,
        aid,
        phase="running",
        manifest_overrides={
            "harness": "zcode",
            "provider": "bigmodel-coding-plan",
            "model": "glm-5.3",
            "effort": "high",
        },
    )
    (original.checkpoint_dir / "snapshot.lock").mkdir(mode=0o700)
    assignment = {
        **_assignment(aid),
        "agent": "zcode",
        "provider": "bigmodel-coding-plan",
        "model": "glm-5.3",
        "effort": "high",
    }
    refill.configure(
        tmp_path,
        volunteer_id="v1",
        refill_to=1,
        max_tasks=5,
        quota_tier="plus",
        max_estimated_quota_pct=None,
        active=[assignment],
        refill_harness="zcode",
        refill_model="glm-5.3",
        refill_effort="high",
    )

    class OfflineDiscardClient(_RecoveryClient):
        def checkpoint_discard(
            self, assignment_id, checkpoint_id, generation, reason,
        ):
            self.discards.append((
                assignment_id, checkpoint_id, generation, reason,
            ))
            raise runloop.ApiError(
                "discard state is not confirmed", status_code=discard_status,
            )

    client = OfflineDiscardClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)

    paused = runloop._pause_checkpoint_quietly(client, assignment)

    assert isinstance(paused, runloop._CheckpointPauseFailure)
    assert paused.family == "checkpoint_invalid"
    assert not paused.discard_confirmed
    assert refill.load(tmp_path)["status"] == refill.FAULTED_STATE
    retryable = checkpoints.find_latest(tmp_path, aid)
    assert retryable is not None and not retryable.valid
    assert not checkpoints.is_terminal(tmp_path, retryable)


def test_scan_reads_atomically_published_generation(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "a" * 32)
    generation = "1" * 32
    payload = item.checkpoint_dir / "snapshots" / generation
    payload.mkdir(parents=True)
    (payload / "last_heartbeat").write_text(
        "2026-08-23T00:00:00Z\n", encoding="utf-8",
    )
    (payload / "session-id").write_text(
        "session-12345678\n", encoding="utf-8",
    )
    (item.checkpoint_dir / "current-generation").write_text(
        generation + "\n", encoding="ascii",
    )
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert loaded.valid
    assert loaded.session_id == "session-12345678"


def test_scan_rejects_generation_that_changes_during_payload_read(
    tmp_path: Path, monkeypatch,
):
    item = _make_checkpoint(tmp_path, "b" * 32)
    old_generation = "1" * 32
    new_generation = "2" * 32
    old_payload = item.checkpoint_dir / "snapshots" / old_generation
    new_payload = item.checkpoint_dir / "snapshots" / new_generation
    old_payload.mkdir(parents=True)
    new_payload.mkdir(parents=True)
    old_sidecar = old_payload / "session-id"
    old_sidecar.write_text("session-old-1234\n", encoding="utf-8")
    (new_payload / "session-id").write_text(
        "session-new-1234\n", encoding="utf-8",
    )
    pointer = item.checkpoint_dir / "current-generation"
    pointer.write_text(old_generation + "\n", encoding="ascii")
    real_read = checkpoints._read_regular_file

    def publish_during_read(path, *, max_bytes):
        payload = real_read(path, max_bytes=max_bytes)
        if Path(path) == old_sidecar:
            temporary = pointer.with_suffix(".tmp")
            temporary.write_text(new_generation + "\n", encoding="ascii")
            os.replace(temporary, pointer)
            old_sidecar.unlink()
            old_payload.rmdir()
        return payload

    monkeypatch.setattr(checkpoints, "_read_regular_file", publish_during_read)

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "generation changed" in (loaded.invalid_reason or "")


def test_scan_rejects_manifest_atomically_replaced_during_scan(
    tmp_path: Path, monkeypatch,
) -> None:
    item = _make_checkpoint(tmp_path, "f" * 32)
    manifest = item.manifest_path
    real_snapshot = checkpoints._read_regular_file_snapshot
    manifest_reads = 0

    def replace_before_final_read(path, *, max_bytes):
        nonlocal manifest_reads
        if Path(path) == manifest:
            manifest_reads += 1
            if manifest_reads == 2:
                replacement = json.loads(manifest.read_text(encoding="utf-8"))
                replacement["phase"] = "running"
                temporary = manifest.with_suffix(".replacement")
                temporary.write_text(json.dumps(replacement), encoding="utf-8")
                os.replace(temporary, manifest)
        return real_snapshot(path, max_bytes=max_bytes)

    monkeypatch.setattr(
        checkpoints, "_read_regular_file_snapshot", replace_before_final_read,
    )

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert loaded.phase == "invalid"
    assert "manifest changed" in (loaded.invalid_reason or "")


def test_scan_fails_closed_on_deep_manifest_without_crashing_all_discovery(
    tmp_path: Path,
) -> None:
    item = _make_checkpoint(tmp_path, "d" * 32)
    item.manifest_path.write_text(
        '{"nested":' * 2_000 + "0" + "}" * 2_000,
        encoding="utf-8",
    )

    loaded = checkpoints.scan(tmp_path)

    assert len(loaded) == 1
    assert not loaded[0].valid
    assert loaded[0].phase == "invalid"


def test_scan_bounds_checkpoint_tree_entries(
    tmp_path: Path, monkeypatch,
) -> None:
    item = _make_checkpoint(tmp_path, "e" * 32)
    monkeypatch.setattr(checkpoints, "MAX_CHECKPOINT_TREE_ENTRIES", 2)
    for index in range(3):
        (item.checkpoint_dir / f"extra-{index}").write_text("x")
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "entry-count limit" in (loaded.invalid_reason or "")


def test_scan_rejects_special_file_in_generation(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "c" * 32)
    generation = "2" * 32
    payload = item.checkpoint_dir / "snapshots" / generation
    payload.mkdir(parents=True)
    (item.checkpoint_dir / "current-generation").write_text(
        generation + "\n", encoding="ascii",
    )
    os.mkfifo(payload / "fifo")
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)

    loaded = next(
        found for found in checkpoints.scan(tmp_path)
        if found.manifest_path == item.manifest_path
    )

    assert not loaded.valid
    assert "special file" in (loaded.invalid_reason or "")


def test_scan_rejects_fifo_manifest_without_blocking(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "4" * 32)
    item.manifest_path.unlink()
    os.mkfifo(item.manifest_path)
    result = []

    worker = threading.Thread(
        target=lambda: result.extend(checkpoints.scan(tmp_path)), daemon=True,
    )
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "scanning a FIFO manifest must not block"
    assert len(result) == 1
    assert not result[0].valid
    assert "regular file" in (result[0].invalid_reason or "")


def test_scan_rejects_dangling_symlink_manifest(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "5" * 32)
    item.manifest_path.unlink()
    item.manifest_path.symlink_to(item.checkpoint_dir / "missing.json")

    loaded = checkpoints.scan(tmp_path)

    assert len(loaded) == 1
    assert not loaded[0].valid
    assert "regular file" in (loaded[0].invalid_reason or "")


def test_scan_rejects_multiply_linked_manifest(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "6" * 32)
    os.link(item.manifest_path, item.checkpoint_dir / "manifest-copy.json")

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "multiply linked" in (loaded.invalid_reason or "")


def test_scan_rejects_oversized_manifest(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "7" * 32)
    item.manifest_path.write_bytes(b" " * (checkpoints.MAX_MANIFEST_BYTES + 1))

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "too large" in (loaded.invalid_reason or "")


def test_scan_turns_manifest_open_error_into_invalid_checkpoint(
    tmp_path: Path, monkeypatch,
):
    item = _make_checkpoint(tmp_path, "8" * 32)
    real_open = checkpoints.os.open

    def deny_manifest(path, flags, *args, **kwargs):
        if Path(path) == item.manifest_path:
            raise PermissionError("denied for test")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(checkpoints.os, "open", deny_manifest)

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "opened safely" in (loaded.invalid_reason or "")


def test_scan_fails_closed_when_tree_walk_gets_permission_error(
    tmp_path: Path, monkeypatch,
):
    item = _make_checkpoint(tmp_path, "9" * 32)

    def denied_walk(*args, onerror=None, **kwargs):
        assert onerror is not None
        onerror(PermissionError("denied for test"))
        yield  # pragma: no cover

    monkeypatch.setattr(checkpoints.os, "walk", denied_walk)

    loaded = checkpoints.scan(tmp_path)[0]

    assert not loaded.valid
    assert "tree is unreadable" in (loaded.invalid_reason or "")


def test_dangling_terminal_marker_prevents_resume(tmp_path: Path):
    item = _make_checkpoint(tmp_path, "0" * 32)
    (item.job_dir / checkpoints.TERMINAL_MARKER).symlink_to(
        item.job_dir / "missing-terminal-evidence",
    )

    assert checkpoints.is_terminal(tmp_path, checkpoints.scan(tmp_path)[0])
    assert checkpoints.latest_by_assignment(tmp_path) == {}


def test_custom_checkpoint_runtime_identity_is_fail_closed(tmp_path: Path):
    item = _make_checkpoint(
        tmp_path,
        "f" * 32,
        manifest_overrides={
            "harness": "kimi-code",
            "provider": "kimi-subscription",
            "agent_version": "0.36.1",
        },
    )
    assignment = {
        **_assignment("f" * 32),
        "agent": "kimi-code",
        "provider": "kimi-subscription",
        "agent_version": "0.36.1",
    }
    assert runloop._checkpoint_identity_mismatches(item, assignment) == []
    assert runloop._checkpoint_identity_mismatches(
        item, {**assignment, "provider": "another-provider"},
    ) == ["provider"]
    assert runloop._checkpoint_identity_mismatches(
        item, {**assignment, "agent_version": "0.99.0"},
    ) == ["agent_version"]


def test_corrupt_manifest_infers_assignment_from_job_name(tmp_path: Path):
    assignment_id = "a" * 32
    item = _make_checkpoint(tmp_path, assignment_id)
    item.manifest_path.write_text("{broken")
    loaded = checkpoints.scan(tmp_path)[0]
    assert not loaded.valid
    assert loaded.phase == "invalid"
    assert loaded.assignment_id == assignment_id


def test_cleanup_removes_superseded_copies_but_can_keep_explicit_final_dir(tmp_path: Path):
    aid = "2" * 32
    old = _make_checkpoint(tmp_path, aid, suffix="old")
    new = _make_checkpoint(
        tmp_path, aid, checkpoint_id="checkpoint-abcdefgh", suffix="new",
        updated_at="2026-07-16T02:00:00Z",
    )
    checkpoints.cleanup_assignment(tmp_path, aid, keep_job_dir=new.job_dir)
    assert not old.job_dir.exists()
    assert new.job_dir.is_dir()
    checkpoints.cleanup_assignment(tmp_path, aid)
    assert not new.job_dir.exists()


def test_expiry_uses_checkpoint_heartbeat(tmp_path: Path):
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    item = _make_checkpoint(tmp_path, "3" * 32, updated_at=old)
    assert checkpoints.is_expired(item)


def test_assignment_lock_fences_a_second_worker_only_for_same_assignment(tmp_path: Path):
    with checkpoints.assignment_lock(tmp_path, "a1"):
        with pytest.raises(checkpoints.CheckpointBusy):
            with checkpoints.assignment_lock(tmp_path, "a1"):
                pass
        with checkpoints.assignment_lock(tmp_path, "a2"):
            pass


def test_terminal_evidence_is_listed_but_never_selected_for_resume(tmp_path: Path):
    aid = "c" * 32
    item = _make_checkpoint(tmp_path, aid)
    checkpoints.mark_terminal(tmp_path, item)
    assert len(checkpoints.scan(tmp_path)) == 1
    assert checkpoints.is_terminal(tmp_path, checkpoints.scan(tmp_path)[0])
    assert checkpoints.latest_by_assignment(tmp_path) == {}


def test_discarding_terminal_evidence_never_releases_server_lease(
        tmp_path: Path, monkeypatch, capsys):
    aid = "d" * 32
    item = _make_checkpoint(tmp_path, aid)
    checkpoints.mark_terminal(tmp_path, item)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(
        runloop, "_load_config",
        lambda: pytest.fail("terminal evidence removal must stay local"),
    )
    assert runloop.cmd_checkpoint_discard(
        argparse.Namespace(checkpoint_id=aid),
    ) == 0
    assert checkpoints.scan(tmp_path) == []
    assert "server lease left unchanged" in capsys.readouterr().out


class _RecoveryClient:
    def __init__(self, assignment):
        self.assignment = assignment
        self.resumes = []
        self.pauses = []
        self.discards = []

    def checkpoint_pause(self, assignment_id, checkpoint_id, generation):
        self.pauses.append((assignment_id, checkpoint_id, generation))
        return {"ok": True}

    def checkpoint_resume(self, assignment_id, checkpoint_id, generation, session_id=None):
        self.resumes.append((assignment_id, checkpoint_id, generation, session_id))
        resumed = dict(self.assignment, resume_generation=generation + 1)
        return {"assignment": resumed}

    def checkpoint_discard(self, assignment_id, checkpoint_id, generation, reason):
        self.discards.append((assignment_id, checkpoint_id, generation, reason))
        return {"ok": True}

    def get_assignment(self):
        return {"active": [self.assignment] if self.assignment else []}


def test_resume_one_passes_checkpoint_and_new_generation_to_runner(
    tmp_path: Path, monkeypatch,
):
    aid = "4" * 32
    item = _make_checkpoint(tmp_path, aid, generation=2)
    assignment = _assignment(aid, generation=2)
    client = _RecoveryClient(assignment)
    seen = {}
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    def fake_run(client_, resumed, tasks_root, args, local_commit, **kwargs):
        seen["assignment"] = resumed
        seen["checkpoint"] = kwargs["resume_checkpoint"]
        return "submitted"

    monkeypatch.setattr(runloop, "_run_and_submit", fake_run)
    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )
    assert outcome == "submitted"
    assert client.resumes[0][2] == 2
    assert seen["assignment"]["resume_generation"] == 3
    assert seen["checkpoint"].checkpoint_id == item.checkpoint_id


def test_legacy_materialization_is_refused_without_publishing_sibling(
    tmp_path: Path,
) -> None:
    item = _make_checkpoint(tmp_path, "a" * 32, layout="legacy")
    item.trial_dir.chmod(0o700)
    workspace = item.checkpoint_dir / "workspace.patch"
    workspace.write_bytes(b"before\n")

    with pytest.raises(ValueError, match="untrusted"):
        checkpoints.materialize_host_checkpoint(tmp_path, item)

    assert workspace.read_bytes() == b"before\n"
    assert not (item.trial_dir / "checkpoint").exists()


def test_legacy_materialization_never_launders_opaque_event_history(
    tmp_path: Path,
) -> None:
    secret = "opaque-provider-value-not-covered-by-generic-pattern"
    item = _make_checkpoint(tmp_path, "e" * 32, layout="legacy")
    item.trial_dir.chmod(0o700)
    (item.checkpoint_dir / "events.jsonl").write_text(
        json.dumps({"event": "legacy", "detail": {"note": secret}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="untrusted"):
        checkpoints.materialize_host_checkpoint(tmp_path, item)

    assert secret in (item.checkpoint_dir / "events.jsonl").read_text()
    assert not (item.trial_dir / "checkpoint").exists()


def test_invalid_checkpoint_cannot_be_materialized(
    tmp_path: Path,
) -> None:
    item = _make_checkpoint(tmp_path, "c" * 32, layout="legacy")
    (item.checkpoint_dir / "invalid-snapshot").write_text("rejected\n")
    item = next(
        found for found in checkpoints.scan(tmp_path)
        if found.assignment_id == item.assignment_id
    )

    with pytest.raises(ValueError, match="invalid checkpoint"):
        checkpoints.materialize_host_checkpoint(tmp_path, item)


def test_persist_generation_does_not_unlink_colliding_private_temp(
    tmp_path: Path, monkeypatch,
) -> None:
    item = _make_checkpoint(tmp_path, "b" * 32, layout="new", generation=1)
    fixed = "f" * 32

    class FixedUuid:
        hex = fixed

    monkeypatch.setattr(checkpoints.uuid, "uuid4", lambda: FixedUuid())
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("keep\n", encoding="utf-8")
    collision = item.checkpoint_dir / f".checkpoint.json.tmp-{fixed}"
    collision.symlink_to(sentinel)
    original = item.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="checkpoint changed"):
        checkpoints.persist_resume_generation(tmp_path, item, 2)

    assert collision.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert item.manifest_path.read_bytes() == original


def test_persist_generation_rejects_manifest_replacement_without_washing_phase(
    tmp_path: Path,
) -> None:
    item = _make_checkpoint(tmp_path, "1" * 32, generation=1)
    changed = json.loads(item.manifest_path.read_text(encoding="utf-8"))
    changed.update({
        "phase": "invalid",
        "model": "attacker-model",
        "resume_generation": 99,
    })
    replacement = item.manifest_path.with_suffix(".replacement")
    replacement.write_text(json.dumps(changed), encoding="utf-8")
    os.replace(replacement, item.manifest_path)

    with pytest.raises(ValueError, match="changed after discovery"):
        checkpoints.persist_resume_generation(tmp_path, item, 2)

    persisted = json.loads(item.manifest_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "invalid"
    assert persisted["model"] == "attacker-model"
    assert persisted["resume_generation"] == 99


def test_persist_generation_scrubs_session_suppressed_by_omission_marker(
    tmp_path: Path,
) -> None:
    item = _make_checkpoint(
        tmp_path,
        "2" * 32,
        generation=1,
        manifest_overrides={
            "harness": "zcode",
            "provider": "bigmodel-coding-plan",
            "session_id": "stale-zcode-session",
        },
    )
    payload = item.checkpoint_dir / "session-omitted-sensitive"
    payload.write_text("provider state omitted\n", encoding="utf-8")
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)
    workspace_only = checkpoints.find_latest(tmp_path, item.assignment_id)
    assert workspace_only is not None
    assert workspace_only.valid
    assert workspace_only.session_id is None

    persisted = checkpoints.persist_resume_generation(
        tmp_path, workspace_only, 2,
    )

    manifest = json.loads(persisted.manifest_path.read_text(encoding="utf-8"))
    assert "session_id" not in manifest
    assert persisted.session_id is None
    assert persisted.resume_generation == 2


def test_resume_uses_server_generation_and_persists_fence_before_runner(
    tmp_path: Path, monkeypatch,
):
    aid = "5" * 32
    item = _make_checkpoint(tmp_path, aid, generation=1)
    assignment = _assignment(aid, generation=2)
    client = _RecoveryClient(assignment)
    seen = {}
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    def fake_run(*args, **kwargs):
        migrated = kwargs["resume_checkpoint"]
        seen["persisted_generation"] = json.loads(
            migrated.manifest_path.read_text()
        )["resume_generation"]
        seen["checkpoint_dir"] = migrated.checkpoint_dir
        return "submitted"

    monkeypatch.setattr(runloop, "_run_and_submit", fake_run)

    assert runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    ) == "submitted"
    assert client.resumes[0][2] == 2
    assert seen["persisted_generation"] == 3
    assert seen["checkpoint_dir"] == item.trial_dir / "checkpoint"


def test_invalid_server_resume_response_is_compensated_with_pause(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "2" * 32
    item = _make_checkpoint(tmp_path, aid, generation=1)
    assignment = _assignment(aid, generation=1)

    class InvalidGenerationClient(_RecoveryClient):
        def checkpoint_resume(self, assignment_id, checkpoint_id, generation,
                              session_id=None):
            self.resumes.append((assignment_id, checkpoint_id, generation, session_id))
            return {"assignment": dict(
                self.assignment, resume_generation=generation + 2,
            )}

    client = InvalidGenerationClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "paused"
    assert client.pauses == [(aid, item.checkpoint_id, 2)]


@pytest.mark.parametrize(
    "response_override",
    [
        {"checkpoint_id": "another-checkpoint"},
        {"nonce": ""},
        {"nonce": "another-lease-nonce"},
        {"agent": "kimi-code"},
        {"provider": "another-provider"},
        {"model": "another-model"},
        {"effort": "low"},
        {"agent_version": "9.9.9"},
    ],
)
def test_resume_rejects_drifted_server_assignment_identity(
    tmp_path: Path, monkeypatch, response_override: dict,
) -> None:
    aid = "7" * 32
    item = _make_checkpoint(tmp_path, aid, generation=1)
    assignment = {
        **_assignment(aid, generation=1),
        "agent": "codex",
        "provider": "openai",
        "agent_version": "1.2.3",
    }

    class DriftedResponseClient(_RecoveryClient):
        def checkpoint_resume(
            self, assignment_id, checkpoint_id, generation, session_id=None,
        ):
            self.resumes.append(
                (assignment_id, checkpoint_id, generation, session_id),
            )
            return {"assignment": {
                **self.assignment,
                "resume_generation": generation + 1,
                **response_override,
            }}

    client = DriftedResponseClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        runloop,
        "_run_and_submit",
        lambda *_args, **_kwargs: pytest.fail("drifted response must not run"),
    )

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "paused"
    assert client.pauses == [(aid, item.checkpoint_id, 2)]


def test_local_fence_failure_after_server_resume_is_compensated_with_pause(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "3" * 32
    item = _make_checkpoint(tmp_path, aid, generation=1)
    assignment = _assignment(aid, generation=1)
    client = _RecoveryClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        checkpoints,
        "persist_resume_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("simulated local CAS failure")
        ),
    )

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "paused"
    assert client.pauses == [(aid, item.checkpoint_id, 2)]


def test_failed_resume_pause_uses_server_supported_invalid_discard(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "8" * 32
    item = _make_checkpoint(tmp_path, aid, generation=1)
    assignment = _assignment(aid, generation=1)

    class PauseRejectedClient(_RecoveryClient):
        def checkpoint_pause(self, assignment_id, checkpoint_id, generation):
            self.pauses.append((assignment_id, checkpoint_id, generation))
            raise runloop.ApiError("generation cannot be paused", status_code=409)

        def checkpoint_discard(
            self, assignment_id, checkpoint_id, generation, reason,
        ):
            assert reason in {
                "user_discard", "invalid", "expired", "incompatible",
            }
            return super().checkpoint_discard(
                assignment_id, checkpoint_id, generation, reason,
            )

    client = PauseRejectedClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        checkpoints,
        "persist_resume_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("simulated local CAS failure")
        ),
    )

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "paused"
    assert client.pauses == [(aid, item.checkpoint_id, 2)]
    assert client.discards == [(aid, item.checkpoint_id, 2, "invalid")]
    assert checkpoints.is_terminal(tmp_path, item)
    assert item.manifest_path.exists()


def test_legacy_checkpoint_is_never_sent_to_paid_resume(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "4" * 32
    item = _make_checkpoint(tmp_path, aid, layout="legacy")
    assignment = _assignment(aid)
    client = _RecoveryClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "legacy-checkpoint-unsupported"
    assert client.resumes == []
    assert client.discards == [(
        aid, item.checkpoint_id, item.resume_generation, "incompatible",
    )]
    assert checkpoints.is_terminal(tmp_path, item)


def test_legacy_checkpoint_discard_transport_failure_remains_retryable(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "6" * 32
    item = _make_checkpoint(tmp_path, aid, layout="legacy")
    assignment = _assignment(aid)

    class OfflineDiscardClient(_RecoveryClient):
        def checkpoint_discard(
            self, assignment_id, checkpoint_id, generation, reason,
        ):
            raise runloop.ApiError("network unavailable")

    client = OfflineDiscardClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "paused"
    assert client.resumes == []
    assert not checkpoints.is_terminal(tmp_path, item)
    assert checkpoints.find_latest(tmp_path, aid) is not None


def test_worker_child_skips_checkpoint_owned_by_healthy_runner(
    tmp_path: Path, monkeypatch,
):
    aid = "e" * 32
    item = _make_checkpoint(tmp_path, aid)
    assignment = _assignment(aid)

    class HealthyOwnerClient(_RecoveryClient):
        def checkpoint_resume(self, *args, **kwargs):
            raise runloop.ApiError(
                "assignment is still running with a healthy runner",
                status_code=409,
            )

    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    assert runloop._resume_one_checkpoint(
        HealthyOwnerClient(assignment),
        item,
        assignment,
        _args(parallel=True, worker_child=True),
        tmp_path / "tasks",
        None,
    ) == "busy"
    assert item.checkpoint_dir == item.trial_dir / "checkpoint"


def test_worker_child_does_not_hide_other_checkpoint_conflicts(
    tmp_path: Path, monkeypatch,
):
    aid = "f" * 32
    item = _make_checkpoint(tmp_path, aid)
    assignment = _assignment(aid)

    class MismatchedCheckpointClient(_RecoveryClient):
        def checkpoint_resume(self, *args, **kwargs):
            raise runloop.ApiError(
                "checkpoint does not own this assignment",
                status_code=409,
            )

    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    assert runloop._resume_one_checkpoint(
        MismatchedCheckpointClient(assignment),
        item,
        assignment,
        _args(parallel=True, worker_child=True),
        tmp_path / "tasks",
        None,
    ) == "paused"


def test_resume_registers_queued_then_announces_fenced_owner_after_success(
    tmp_path: Path, monkeypatch,
):
    aid = "d" * 32
    item = _make_checkpoint(tmp_path, aid, generation=2)
    assignment = dict(_assignment(aid, generation=2), batch_id="batch-1")
    events = []

    class Client(_RecoveryClient):
        def checkpoint_resume(self, *args, **kwargs):
            events.append("resume")
            return super().checkpoint_resume(*args, **kwargs)

    class Telemetry:
        session_id = "session-recovery"

        def bind_batch(self, batch_id):
            events.append(("batch", batch_id))

        def set_phase(self, phase, assignment_id=None, resume_generation=None):
            events.append(("phase", phase, assignment_id, resume_generation))

        def flush(self):
            events.append("flush")

    client = Client(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runloop, "_run_and_submit", lambda *a, **k: "submitted")

    assert runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", Telemetry(),
    ) == "submitted"
    assert events[:6] == [
        ("batch", "batch-1"),
        ("phase", "queued", None, None),
        "flush",
        "resume",
        ("phase", "running", aid, 3),
        "flush",
    ]


@pytest.mark.parametrize("ambiguous_status", [None, 502, 504])
def test_resume_replays_same_fence_after_committed_response_is_lost(
    tmp_path: Path, monkeypatch, ambiguous_status: int | None,
) -> None:
    aid = "9" * 32
    item = _make_checkpoint(tmp_path, aid, generation=2)
    assignment = dict(_assignment(aid, generation=2), batch_id="batch-1")

    class CommitThenLoseResponseClient(_RecoveryClient):
        def checkpoint_resume(
            self, assignment_id, checkpoint_id, generation, session_id=None,
        ):
            call = (assignment_id, checkpoint_id, generation, session_id)
            self.resumes.append(call)
            resumed = dict(self.assignment, resume_generation=generation + 1)
            if len(self.resumes) == 1:
                # Model the server committing N+1 and the network losing only
                # the response.  The next identical request is its supported
                # idempotent replay, not a new N+1 -> N+2 recovery.
                self.assignment = resumed
                raise runloop.ApiError(
                    "response lost after commit",
                    status_code=ambiguous_status,
                )
            assert call == self.resumes[0]
            return {"ok": True, "replayed": True, "assignment": resumed}

    class Telemetry:
        session_id = "session-response-loss"

        def bind_batch(self, _batch_id):
            pass

        def set_phase(self, *_args):
            pass

        def flush(self):
            pass

    client = CommitThenLoseResponseClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runloop, "_run_and_submit", lambda *a, **k: "submitted")

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", Telemetry(),
    )

    assert outcome == "submitted"
    assert client.resumes == [
        (aid, item.checkpoint_id, 2, "session-response-loss"),
        (aid, item.checkpoint_id, 2, "session-response-loss"),
    ]
    assert checkpoints.find_latest(tmp_path, aid).resume_generation == 3
    assert client.pauses == []
    assert client.discards == []


def test_resume_replays_truncated_http_200_with_same_idempotency_key(
    monkeypatch,
) -> None:
    calls: list[dict[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        form = urllib.parse.parse_qs(request.read().decode())
        calls.append(form)
        if len(calls) == 1:
            return httpx.Response(
                200,
                content=b'{"assignment":',
                headers={"content-type": "application/json"},
            )
        return httpx.Response(200, json={"assignment": {
            "assignment_id": "assignment-1",
            "checkpoint_id": "checkpoint-12345678",
            "resume_generation": 3,
        }})

    client = ApiClient(
        "https://dradar.invalid",
        "test-token",
        transport=httpx.MockTransport(handler),
        capabilities=(),
    )
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    response = runloop._resume_checkpoint_with_ambiguous_replay(
        client,
        assignment_id="assignment-1",
        checkpoint_id="checkpoint-12345678",
        generation=2,
        session_id="session-response-loss",
    )

    assert response["assignment"]["resume_generation"] == 3
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["resume_generation"] == ["2"]
    assert calls[0]["session_id"] == ["session-response-loss"]


def test_resume_replays_incomplete_json_success_before_compensation(
    monkeypatch,
) -> None:
    responses = [
        {"ok": True},
        {"assignment": {"assignment_id": "assignment-1"}},
        {"assignment": {
            "assignment_id": "assignment-1",
            "checkpoint_id": "checkpoint-12345678",
            "resume_generation": 3,
        }},
    ]

    class Client:
        def __init__(self):
            self.calls = []

        def checkpoint_resume(
            self, assignment_id, checkpoint_id, generation, session_id=None,
        ):
            self.calls.append(
                (assignment_id, checkpoint_id, generation, session_id),
            )
            return responses[len(self.calls) - 1]

    client = Client()
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    response = runloop._resume_checkpoint_with_ambiguous_replay(
        client,
        assignment_id="assignment-1",
        checkpoint_id="checkpoint-12345678",
        generation=2,
        session_id="session-response-loss",
    )

    assert response == responses[-1]
    assert client.calls == [
        (
            "assignment-1", "checkpoint-12345678", 2,
            "session-response-loss",
        ),
    ] * 3


def test_resume_without_session_never_replays_ambiguous_post(
    tmp_path: Path, monkeypatch,
) -> None:
    aid = "a" * 32
    item = _make_checkpoint(tmp_path, aid, generation=1)
    assignment = _assignment(aid, generation=1)

    class AmbiguousClient(_RecoveryClient):
        def checkpoint_resume(
            self, assignment_id, checkpoint_id, generation, session_id=None,
        ):
            self.resumes.append(
                (assignment_id, checkpoint_id, generation, session_id),
            )
            raise runloop.ApiError("response state unknown")

    client = AmbiguousClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop.time, "sleep", lambda _seconds: None)

    assert runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    ) == "paused"
    assert client.resumes == [(aid, item.checkpoint_id, 1, None)]


def test_checkpoint_recovery_uses_exponential_backoff(tmp_path: Path):
    aid = "7" * 32
    now = datetime.now(timezone.utc)
    item = _make_checkpoint(
        tmp_path, aid, generation=3, updated_at=now.isoformat(),
    )

    assert runloop._checkpoint_backoff_seconds(item, now=now) == 120


def test_checkpoint_recovery_limit_opens_circuit_without_resuming(
        tmp_path: Path, monkeypatch, capsys):
    aid = "8" * 32
    item = _make_checkpoint(
        tmp_path, aid, generation=runloop.MAX_CHECKPOINT_RESUMES,
    )
    assignment = _assignment(aid, generation=runloop.MAX_CHECKPOINT_RESUMES)
    client = _RecoveryClient(assignment)
    abort_file = tmp_path / "ACCOUNT_STOP"
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setenv("DRADAR_POOL_ABORT_FILE", str(abort_file))
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(), tmp_path / "tasks", None,
    )

    assert outcome == "recovery-exhausted"
    assert client.resumes == []
    assert checkpoints.is_terminal(tmp_path, item)
    assert abort_file.read_text() == "drain:checkpoint recovery safety limit reached"
    assert "5-resume safety limit" in capsys.readouterr().out


def test_completed_checkpoint_resume_recovers_missing_staged_patch_without_rerun(
    tmp_path: Path, monkeypatch,
):
    aid = "e" * 32
    item = _make_checkpoint(tmp_path, aid, phase="agent_completed")
    assignment = _assignment(aid)
    patch = item.trial_dir / "artifacts" / "model.patch"
    patch.parent.mkdir(parents=True)
    patch_bytes = b"diff --git a/app.py b/app.py\n-old\n+new\n"
    patch.write_bytes(patch_bytes)
    client = _RecoveryClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    # This is the narrow incident window: the model/Pier finished and wrote
    # the patch, then the runner paused before _upload_trial started.
    assert runloop._pause_checkpoint_quietly(client, assignment) is not None
    prepared = artifact_staging.ensure_staged_patch(item.trial_dir)
    assert client.pauses == [(aid, item.checkpoint_id, 0)]
    prepared.staged.unlink()  # paused/closed after completion, before upload

    class CompletedClient(_RecoveryClient):
        def __init__(self, assignment_):
            super().__init__(assignment_)
            self.submissions = []

        def submit(self, assignment_id, nonce, patch, trajectory, result, meta,
                   outcome="completed", resume_generation=None):
            self.submissions.append({
                "assignment_id": assignment_id,
                "patch": patch.read_bytes(),
                "meta": meta,
            })
            return {"submission_id": "s-recovered", "grade_status": "pending"}

    client = CompletedClient(assignment)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")

    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(yes=True), tmp_path / "tasks", None,
    )

    assert outcome == "submitted"
    assert client.resumes == []  # completed work uploads; the model never reruns
    assert client.submissions[0]["patch"] == patch_bytes
    assert client.submissions[0]["meta"]["artifact_staging_recovery"]["reason"] == (
        "source-present/staged-missing"
    )
    assert pending.load(tmp_path) == []


def test_completed_checkpoint_resume_recovers_workspace_patch_without_rerun(
    tmp_path: Path, monkeypatch, capsys,
):
    aid = "f" * 32
    item = _make_checkpoint(tmp_path, aid, phase="agent_completed")
    assignment = _assignment(aid)
    metadata = json.loads(item.manifest_path.read_text())
    metadata["workspace_patch"] = "workspace.patch"
    item.manifest_path.write_text(json.dumps(metadata))
    patch_bytes = b"diff --git a/model_answer.json b/model_answer.json\n-old\n+new\n"
    (item.checkpoint_dir / "workspace.patch").write_bytes(patch_bytes)
    _privatize_checkpoint_tree(item.trial_dir, item.checkpoint_dir)
    item = checkpoints.find_latest(tmp_path, aid)
    assert item is not None
    client = _RecoveryClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")

    class CompletedClient(_RecoveryClient):
        def __init__(self, assignment_):
            super().__init__(assignment_)
            self.submissions = []

        def submit(self, assignment_id, nonce, patch, trajectory, result, meta,
                   outcome="completed", resume_generation=None):
            self.submissions.append(patch.read_bytes())
            return {"submission_id": "s-workspace", "grade_status": "pending"}

    client = CompletedClient(assignment)
    outcome = runloop._resume_one_checkpoint(
        client, item, assignment, _args(yes=True), tmp_path / "tasks", None,
    )

    assert outcome == "submitted"
    assert client.resumes == []
    assert client.submissions == [patch_bytes]
    assert "uploading without rerunning" in capsys.readouterr().out
    assert pending.load(tmp_path) == []


def test_healthy_local_run_holds_assignment_lock_before_checkpoint_resume(
    tmp_path: Path, monkeypatch,
):
    """A checkpoint written by an active first run is not resumable locally."""
    aid = "b" * 32
    item = _make_checkpoint(tmp_path, aid)
    assignment = _assignment(aid)
    client = _RecoveryClient(assignment)
    entered = threading.Event()
    release = threading.Event()
    result = []
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "check_task_content_hash", lambda *a, **k: True)
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *a, **k: "base")
    monkeypatch.setattr(runloop, "_pause_checkpoint_quietly", lambda *a, **k: None)
    monkeypatch.setattr(runloop, "_mark_stopped_quietly", lambda *a, **k: None)

    def blocking_trial(*_a, **_kw):
        entered.set()
        assert release.wait(5)
        raise runloop.RunnerError("test stop")

    monkeypatch.setattr(runloop, "run_trial", blocking_trial)
    worker = threading.Thread(target=lambda: result.append(
        runloop._run_and_submit(
            client, assignment, tmp_path / "tasks", _args(), "base",
        )
    ))
    worker.start()
    assert entered.wait(5)
    try:
        assert runloop._resume_one_checkpoint(
            client, item, assignment, _args(), tmp_path / "tasks", None,
        ) == "busy"
        assert client.resumes == []
    finally:
        release.set()
        worker.join(5)
    assert result == ["failed"]


def test_invalid_checkpoint_discards_server_lease_but_keeps_terminal_evidence(
    tmp_path: Path, monkeypatch,
):
    aid = "5" * 32
    item = _make_checkpoint(tmp_path, aid)
    item.manifest_path.write_text("{broken")
    invalid = checkpoints.scan(tmp_path)[0]
    assignment = _assignment(aid)
    client = _RecoveryClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    outcome = runloop._resume_one_checkpoint(
        client, invalid, assignment, _args(), tmp_path / "tasks", None,
    )
    assert outcome == "checkpoint-invalid"
    assert client.discards[0][3] == "invalid"
    kept = checkpoints.scan(tmp_path)
    assert len(kept) == 1
    assert checkpoints.is_terminal(tmp_path, kept[0])
    assert checkpoints.find_latest(tmp_path, aid) is None


@pytest.mark.parametrize("discard_status", [None, 409])
def test_invalid_resume_discard_failure_only_retries_discard(
    tmp_path: Path, monkeypatch, discard_status: int | None,
) -> None:
    aid = "4" * 32
    item = _make_checkpoint(
        tmp_path,
        aid,
        manifest_overrides={
            "harness": "zcode",
            "provider": "bigmodel-coding-plan",
            "model": "glm-5.3",
            "effort": "low",
        },
    )
    item.manifest_path.write_text("{broken", encoding="utf-8")
    assignment = {
        **_assignment(aid),
        "agent": "zcode",
        "provider": "bigmodel-coding-plan",
        "model": "glm-5.3",
        "effort": "low",
    }
    refill.configure(
        tmp_path,
        volunteer_id="v1",
        refill_to=1,
        max_tasks=5,
        quota_tier="plus",
        max_estimated_quota_pct=None,
        active=[assignment],
        refill_harness="zcode",
        refill_model="glm-5.3",
        refill_effort="low",
    )

    class OfflineDiscardClient(_RecoveryClient):
        def checkpoint_discard(
            self, assignment_id, checkpoint_id, generation, reason,
        ):
            self.discards.append((
                assignment_id, checkpoint_id, generation, reason,
            ))
            raise runloop.ApiError(
                "discard state is not confirmed", status_code=discard_status,
            )

    client = OfflineDiscardClient(assignment)
    monkeypatch.setattr(runloop, "HOME", tmp_path)

    for expected_attempts in (1, 2):
        retryable = checkpoints.find_latest(tmp_path, aid)
        assert retryable is not None and not retryable.valid
        outcome = runloop._resume_one_checkpoint(
            client, retryable, assignment, _args(), tmp_path / "tasks", None,
        )
        assert outcome == "paused"
        assert len(client.discards) == expected_attempts
        assert client.resumes == []
        assert not checkpoints.is_terminal(tmp_path, retryable)
    assert refill.load(tmp_path)["status"] == refill.FAULTED_STATE


def test_cleanup_only_removes_server_settled_unprotected_jobs(
    tmp_path: Path, monkeypatch, capsys,
):
    _make_checkpoint(tmp_path, "6" * 32, suffix="active")
    _make_checkpoint(tmp_path, "7" * 32, suffix="pending")
    _make_checkpoint(tmp_path, "8" * 32, phase="agent_completed", suffix="kept")
    _make_checkpoint(tmp_path, "9" * 32, phase="agent_completed", suffix="settled")
    active = checkpoints.find_latest(tmp_path, "6" * 32)
    pending_item = checkpoints.find_latest(tmp_path, "7" * 32)
    kept = checkpoints.find_latest(tmp_path, "8" * 32)
    settled = checkpoints.find_latest(tmp_path, "9" * 32)
    assert active and pending_item and kept and settled
    checkpoints.mark_kept(tmp_path, kept)
    from dradar import pending
    pending.record(tmp_path, {"assignment_id": pending_item.assignment_id})

    client = _RecoveryClient(_assignment(active.assignment_id))
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_load_config", lambda: {})
    monkeypatch.setattr(runloop, "_client", lambda _cfg: client)

    dry = argparse.Namespace(dry_run=True, include_kept=False, yes=True)
    assert runloop.cmd_cleanup(dry) == 0
    assert all(item.job_dir.is_dir() for item in (active, pending_item, kept, settled))

    args = argparse.Namespace(dry_run=False, include_kept=False, yes=True)
    assert runloop.cmd_cleanup(args) == 0
    assert active.job_dir.is_dir()
    assert pending_item.job_dir.is_dir()
    assert kept.job_dir.is_dir()
    assert not settled.job_dir.exists()
    out = capsys.readouterr().out
    assert "protected: 1 active/resumable, 1 pending upload, 1 explicitly kept" in out

    include_kept = argparse.Namespace(dry_run=False, include_kept=True, yes=True)
    assert runloop.cmd_cleanup(include_kept) == 0
    assert not kept.job_dir.exists()
    assert active.job_dir.is_dir() and pending_item.job_dir.is_dir()


def test_cleanup_network_failure_deletes_nothing(tmp_path: Path, monkeypatch, capsys):
    item = _make_checkpoint(tmp_path, "a" * 32, phase="agent_completed")
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    monkeypatch.setattr(runloop, "_load_config", lambda: {})

    class Offline:
        def get_assignment(self):
            from dradar.api_client import ApiError
            raise ApiError("offline")

    monkeypatch.setattr(runloop, "_client", lambda _cfg: Offline())
    args = argparse.Namespace(dry_run=False, include_kept=True, yes=True)
    assert runloop.cmd_cleanup(args) == 1
    assert item.job_dir.is_dir()
    assert "nothing was deleted" in capsys.readouterr().out
