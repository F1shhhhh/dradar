import json
import os
import stat
import time

import pytest

from dradar.api_client import ApiError
from dradar.checkpoint_observation_v2 import (
    CheckpointObservationReporterV2,
    CheckpointObservationSpoolError,
    CheckpointObservationSpoolV2,
)


def _payload(operation_id="operation-0001", capture_id="capture-0001"):
    return {
        "observation_kind": "capture",
        "assignment_id": "assignment-0001",
        "operation_id": operation_id,
        "capture_id": capture_id,
        "checkpoint_id": "checkpoint-0001",
        "checkpoint_lineage_id": "lineage-0001",
        "snapshot_generation": 1,
        "rollout_mode": "observe",
        "status": "sealed",
        "stage": None,
        "failure_code": None,
        "failure_type": None,
        "identity_fingerprint": "a" * 64,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": "codex-openai/v1",
        "capture_storage": "container_native",
        "manifest_sha256": "b" * 64,
        "archive_sha256": "c" * 64,
        "archive_bytes": 1234,
        "file_count": 3,
        "payload_bytes": 999,
        "elapsed_ms": 42,
        "platform": "macos",
        "container_backend": "orbstack",
        "client_version": "0.5.97",
        "adapter_version": "codex-openai/v1",
        "remote_cleanup": "discarded",
        "authoritative": False,
        "selected_local": False,
    }


def _ack(payload):
    return {
        "ok": True,
        "assignment_id": payload["assignment_id"],
        "capture_id": payload["capture_id"],
        "status": payload["status"],
        "assignment_unchanged": True,
        "paid_execution_authorized": False,
    }


class FakeObservationApi:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.payloads = []

    def checkpoint_v2_observation(self, payload):
        self.payloads.append(payload)
        outcome = self.outcomes.pop(0) if self.outcomes else _ack(payload)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def checkpoint_v2_restore_observation(self, payload):
        self.payloads.append(payload)
        outcome = self.outcomes.pop(0) if self.outcomes else {
            "ok": True,
            "assignment_id": payload["assignment_id"],
            "restore_id": payload["restore_id"],
            "status": payload["status"],
            "assignment_unchanged": True,
            "paid_execution_authorized": False,
        }
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _restore_payload():
    return {
        "observation_kind": "restore",
        "assignment_id": "assignment-0001",
        "operation_id": "operation-restore-0001",
        "restore_id": "restore-0001",
        "source_capture_id": "capture-0001",
        "checkpoint_id": "checkpoint-0001",
        "checkpoint_lineage_id": "lineage-0001",
        "snapshot_generation": 1,
        "rollout_mode": "restore_test",
        "status": "verified",
        "stage": None,
        "failure_code": None,
        "failure_type": None,
        "identity_fingerprint": "a" * 64,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": "codex-openai/v1",
        "manifest_sha256": "b" * 64,
        "elapsed_ms": 91,
        "platform": "macos",
        "container_backend": "orbstack",
        "client_version": "0.5.97",
        "adapter_version": "codex-openai/v1",
        "paid_execution_started": False,
        "authoritative": False,
    }


def test_spool_is_private_canonical_idempotent_and_conflict_safe(tmp_path):
    spool = CheckpointObservationSpoolV2(tmp_path / "observations")
    payload = _payload()

    assert spool.persist(payload) is True
    assert spool.persist(dict(payload)) is False
    pending = spool.pending()
    assert len(pending) == 1
    path, decoded = pending[0]
    assert decoded == payload
    assert json.loads(path.read_text())["payload"] == payload
    if hasattr(os, "getuid"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    conflict = dict(payload, status="failed")
    with pytest.raises(CheckpointObservationSpoolError, match="conflicts"):
        spool.persist(conflict)
    assert spool.pending()[0][1] == payload


def test_spool_rejects_unknown_fields_multiline_text_and_symlink_root(tmp_path):
    spool = CheckpointObservationSpoolV2(tmp_path / "observations")
    with pytest.raises(CheckpointObservationSpoolError, match="wire fields"):
        spool.persist(dict(_payload(), prompt="do not store me"))
    bad = _payload()
    bad["failure_type"] = "trace\nwith local path"
    with pytest.raises(CheckpointObservationSpoolError, match="unbounded text"):
        spool.persist(bad)

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked-observations"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(CheckpointObservationSpoolError, match="unsafe"):
        CheckpointObservationSpoolV2(symlink).persist(_payload())


def test_reporter_only_deletes_after_explicit_non_authoritative_ack(tmp_path):
    payload = _payload()
    unsafe_ack = dict(_ack(payload), paid_execution_authorized=True)
    api = FakeObservationApi([unsafe_ack, _ack(payload)])
    reporter = CheckpointObservationReporterV2(api, tmp_path)

    assert reporter.record(payload) is True
    first = reporter.flush_once()
    assert first.persisted == 1
    assert first.retryable == 1
    assert len(reporter.spool.pending()) == 1

    second = reporter.flush_once()
    assert second.acknowledged == 1
    assert reporter.spool.pending() == []
    assert len(api.payloads) == 2


def test_transport_failure_survives_restart_and_replays_same_operation(tmp_path):
    payload = _payload()
    first_api = FakeObservationApi([ApiError("offline")])
    first = CheckpointObservationReporterV2(first_api, tmp_path)
    assert first.record(payload)
    result = first.flush_once()
    assert result.retryable == 1
    assert len(first.spool.pending()) == 1

    second_api = FakeObservationApi()
    second = CheckpointObservationReporterV2(second_api, tmp_path)
    replay = second.flush_once()
    assert replay.acknowledged == 1
    assert second_api.payloads == [payload]
    assert second.spool.pending() == []


def test_restore_observation_uses_distinct_route_and_ack_identity(tmp_path):
    payload = _restore_payload()
    api = FakeObservationApi()
    reporter = CheckpointObservationReporterV2(api, tmp_path)
    assert reporter.record(payload)
    result = reporter.flush_once()
    assert result.acknowledged == 1
    assert api.payloads == [payload]
    assert reporter.spool.pending() == []


def test_restore_source_pending_is_retryable_not_quarantined(tmp_path):
    payload = _restore_payload()
    api = FakeObservationApi([
        ApiError(
            "source pending",
            status_code=425,
            code="checkpoint_restore_source_pending",
        ),
    ])
    reporter = CheckpointObservationReporterV2(api, tmp_path)
    assert reporter.record(payload)
    first = reporter.flush_once()
    assert first.retryable == 1
    assert len(reporter.spool.pending()) == 1
    assert list(reporter.spool.rejected_root.glob("*.json")) == []
    second = reporter.flush_once()
    assert second.acknowledged == 1
    assert reporter.spool.pending() == []


def test_stable_semantic_rejection_is_quarantined_not_retried(tmp_path):
    payload = _payload()
    api = FakeObservationApi([
        ApiError("conflict", status_code=409, code="checkpoint_observation_conflict"),
    ])
    reporter = CheckpointObservationReporterV2(api, tmp_path)
    assert reporter.record(payload)
    result = reporter.flush_once()
    assert result.rejected == 1
    assert reporter.spool.pending() == []
    rejected = list(reporter.spool.rejected_root.glob("*.json"))
    assert len(rejected) == 1
    assert json.loads(rejected[0].read_text())["payload"] == payload


def test_old_server_404_is_retryable_for_future_upgrade(tmp_path):
    payload = _payload()
    api = FakeObservationApi([ApiError("not found", status_code=404)])
    reporter = CheckpointObservationReporterV2(api, tmp_path)
    assert reporter.record(payload)
    result = reporter.flush_once()
    assert result.retryable == 1
    assert len(reporter.spool.pending()) == 1


def test_record_is_nonblocking_and_bounded_when_memory_queue_is_full(tmp_path):
    api = FakeObservationApi()
    reporter = CheckpointObservationReporterV2(api, tmp_path, queue_size=1)
    started = time.monotonic()
    assert reporter.record(_payload()) is True
    assert reporter.record(_payload("operation-0002", "capture-0002")) is False
    assert time.monotonic() - started < 0.1
    assert reporter.stats.dropped == 1
    # No background thread was started: recording alone performs no disk I/O.
    assert not (tmp_path / "checkpoint-v2").exists()


def test_corrupt_pending_entry_is_never_sent_or_deleted(tmp_path):
    spool = CheckpointObservationSpoolV2(
        tmp_path / "checkpoint-v2" / "observations",
    )
    spool.persist(_payload())
    path = spool.pending()[0][0]
    path.write_text("{}")
    os.chmod(path, 0o600)
    api = FakeObservationApi()
    reporter = CheckpointObservationReporterV2(api, tmp_path)
    result = reporter.flush_once()
    assert result.retryable == 1
    assert api.payloads == []
    assert path.exists()


def test_background_close_has_a_hard_wait_bound(tmp_path):
    class SlowApi(FakeObservationApi):
        def checkpoint_v2_observation(self, payload):
            time.sleep(1)
            return _ack(payload)

    reporter = CheckpointObservationReporterV2(
        SlowApi(), tmp_path, idle_retry_sec=0.1,
    )
    reporter.start()
    assert reporter.record(_payload())
    started = time.monotonic()
    reporter.close(timeout=0.02)
    assert time.monotonic() - started < 0.15
