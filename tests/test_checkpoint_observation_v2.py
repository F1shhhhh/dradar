import json
import multiprocessing
import os
import stat
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dradar.api_client import ApiError
from dradar.checkpoint_observation_v2 import (
    CheckpointObservationReporterV2,
    CheckpointObservationSpoolError,
    CheckpointObservationSpoolV2,
    ObservationDeliveryResultV2,
    checkpoint_local_evidence_v2,
    checkpoint_mainline_assignment_digest_v2,
)


def _observation_persist_process(
    root: str,
    payload: dict,
    start,
    results,
) -> None:
    start.wait()
    spool = CheckpointObservationSpoolV2(Path(root))
    try:
        created = spool.persist(payload)
    except CheckpointObservationSpoolError as exc:
        results.put(("conflict", str(exc)))
    except Exception as exc:  # pragma: no cover - asserted by parent
        results.put(("error", f"{type(exc).__name__}:{exc}"))
    else:
        results.put(("created" if created else "replay", None))


def _observation_drop_process(
    root: str,
    assignment_id: str,
    count: int,
    start,
    results,
) -> None:
    start.wait()
    try:
        CheckpointObservationSpoolV2(Path(root)).record_delivery_drops(
            assignment_id, count,
        )
    except Exception as exc:  # pragma: no cover - asserted by parent
        results.put(("error", f"{type(exc).__name__}:{exc}"))
    else:
        results.put(("ok", count))


def _observation_lock_crash_process(
    root: str,
    ready,
    crash,
) -> None:
    from dradar.checkpoint_observation_v2 import _process_lock

    with _process_lock(Path(root)):
        ready.set()
        crash.wait(10)
        os._exit(94)


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


def _cohort_registration():
    return {
        "assignment_id": "assignment-0001",
        "runner_session_id": "session-0001",
        "identity_fingerprint": "a" * 64,
        "cohort": {
            "platform": "macos",
            "container_backend": "orbstack",
            "harness": "codex",
            "provider": "openai",
            "client_version": "0.5.97",
            "agent_version": "1.2.3",
            "runtime_profile": "codex-runtime-v1",
            "model_config_version": "codex-config-v1",
            "runtime_compatibility_digest": "d" * 64,
            "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
            "checkpoint_abi": "codex-openai/v1",
        },
    }


def _mainline_start():
    return {
        "assignment_id": "assignment-0001",
        "runner_session_id": "session-0001",
        "identity_fingerprint": "a" * 64,
        "attempt_id": "attempt-0001",
        "assignment_state_sha256": checkpoint_mainline_assignment_digest_v2({
            "assignment_id": "assignment-0001",
            "checkpoint_protocol_version": 1,
            "checkpoint_id": None,
            "execution_state": "running",
            "owner_epoch": 0,
            "resume_generation": 0,
        }),
        "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _mainline_completion(start):
    return {
        "completed_at": (
            datetime.fromisoformat(start["started_at"]) + timedelta(seconds=10)
        ).isoformat(),
        "mainline_elapsed_ms": 10_000,
        "checkpoint_sync_elapsed_ms": 25,
        "outcome": "completed",
        "result_preserved": True,
        "assignment_state_sha256_after": start["assignment_state_sha256"],
        "submission_preserved": True,
        "comparable": True,
        "exclusion_reason": None,
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


def test_spool_preserves_all_distinct_records_from_independent_processes(
    tmp_path,
):
    root = tmp_path / "observations"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = []
    expected = {}
    for index in range(12):
        payload = _payload(
            f"operation-process-{index:04d}",
            f"capture-process-{index:04d}",
        )
        expected[payload["operation_id"]] = payload
        process = ctx.Process(
            target=_observation_persist_process,
            args=(str(root), payload, start, results),
        )
        process.start()
        processes.append(process)
    start.set()
    observed = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert observed == [("created", None)] * len(processes)
    pending = {
        payload["operation_id"]: payload
        for _, payload in CheckpointObservationSpoolV2(root).pending(limit=32)
    }
    assert pending == expected
    assert list((root / "pending").glob(".*.tmp")) == []


def test_spool_replay_and_conflict_are_stable_across_processes(tmp_path):
    root = tmp_path / "observations"
    payload = _payload()
    spool = CheckpointObservationSpoolV2(root)
    assert spool.persist(payload) is True
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    variants = [dict(payload) for _ in range(5)] + [
        dict(payload, status="failed") for _ in range(3)
    ]
    processes = [
        ctx.Process(
            target=_observation_persist_process,
            args=(str(root), variant, start, results),
        )
        for variant in variants
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert sum(item[0] == "replay" for item in observed) == 5
    conflicts = [item for item in observed if item[0] == "conflict"]
    assert len(conflicts) == 3
    assert all("operation id conflicts" in item[1] for item in conflicts)
    assert spool.pending()[0][1] == payload


def test_spool_drop_accounting_is_atomic_across_processes(tmp_path):
    root = tmp_path / "observations"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    counts = list(range(1, 9))
    processes = [
        ctx.Process(
            target=_observation_drop_process,
            args=(str(root), "assignment-0001", count, start, results),
        )
        for count in counts
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert sorted(observed) == [("ok", count) for count in counts]
    health = json.loads(
        (root / "delivery-health" / "assignment-0001.json").read_text(),
    )
    assert health["dropped"] == sum(counts)


def test_spool_persists_every_delivery_health_counter(tmp_path):
    root = tmp_path / "observations"
    spool = CheckpointObservationSpoolV2(root)

    spool.record_delivery_health(
        "assignment-0001",
        ObservationDeliveryResultV2(
            persisted=2,
            acknowledged=1,
            retryable=3,
            rejected=4,
            dropped=5,
        ),
    )
    spool.record_delivery_health(
        "assignment-0001",
        ObservationDeliveryResultV2(
            persisted=7,
            acknowledged=6,
            retryable=5,
            rejected=4,
            dropped=3,
        ),
    )

    health = json.loads(
        (root / "delivery-health" / "assignment-0001.json").read_text(),
    )
    assert {
        field: health[field]
        for field in (
            "persisted", "acknowledged", "retryable", "rejected", "dropped",
        )
    } == {
        "persisted": 9,
        "acknowledged": 7,
        "retryable": 8,
        "rejected": 8,
        "dropped": 8,
    }


def test_spool_process_lock_is_released_after_hard_crash(tmp_path):
    root = tmp_path / "observations"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    crash = ctx.Event()
    process = ctx.Process(
        target=_observation_lock_crash_process,
        args=(str(root), ready, crash),
    )
    process.start()
    assert ready.wait(10)
    crash.set()
    process.join(timeout=15)
    assert process.exitcode == 94

    spool = CheckpointObservationSpoolV2(root)
    assert spool.persist(_payload()) is True
    assert spool.pending()[0][1] == _payload()


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


def test_rejected_spool_is_bounded_without_deleting_failure_evidence(tmp_path):
    root = tmp_path / "observations"
    spool = CheckpointObservationSpoolV2(
        root,
        max_rejected=1,
        max_rejected_bytes=16 * 1024,
    )
    first = _payload()
    second = _payload("operation-0002", "capture-0002")
    assert spool.persist(first)
    first_path = spool.pending()[0][0]
    spool.reject(first_path, first["operation_id"])
    assert spool.persist(second)
    second_path = spool.pending()[0][0]

    with pytest.raises(
        CheckpointObservationSpoolError, match="rejected spool is full",
    ):
        spool.reject(second_path, second["operation_id"])

    assert [payload for _, payload in spool.pending()] == [second]
    rejected = list(spool.rejected_root.glob("*.json"))
    assert len(rejected) == 1
    assert json.loads(rejected[0].read_text())["payload"] == first


def test_exact_rejected_replay_converges_without_duplicate_evidence(tmp_path):
    spool = CheckpointObservationSpoolV2(tmp_path / "observations")
    payload = _payload()
    assert spool.persist(payload)
    first_path = spool.pending()[0][0]
    spool.reject(first_path, payload["operation_id"])
    assert spool.persist(payload)
    replay_path = spool.pending()[0][0]

    spool.reject(replay_path, payload["operation_id"])

    assert spool.pending() == []
    rejected = list(spool.rejected_root.glob("*.json"))
    assert len(rejected) == 1
    assert json.loads(rejected[0].read_text())["payload"] == payload


def test_durable_reporter_handoff_survives_before_delivery(tmp_path):
    reporter = CheckpointObservationReporterV2(
        FakeObservationApi(), tmp_path, idle_retry_sec=3600,
    )
    payload = _payload()

    assert reporter.persist(payload) is True
    assert reporter.persist(payload) is True

    pending = reporter.spool.pending()
    assert len(pending) == 1
    assert pending[0][1] == payload
    assert reporter.stats.persisted == 1


@pytest.mark.parametrize("code", [
    "checkpoint_v2_kill_switch_active",
    "checkpoint_observation_not_authorized",
    "checkpoint_restore_observation_not_authorized",
])
def test_reversible_rollout_rejection_stays_pending_for_later_replay(
    tmp_path, code,
):
    payload = _payload()
    api = FakeObservationApi([
        ApiError("temporarily disabled", status_code=409, code=code),
        _ack(payload),
    ])
    reporter = CheckpointObservationReporterV2(api, tmp_path)
    assert reporter.record(payload)

    disabled = reporter.flush_once()
    assert disabled.retryable == 1
    assert len(reporter.spool.pending()) == 1
    assert list(reporter.spool.rejected_root.glob("*.json")) == []

    replay = CheckpointObservationReporterV2(api, tmp_path).flush_once()
    assert replay.acknowledged == 1
    assert reporter.spool.pending() == []


def test_reporter_delivery_health_survives_restart(tmp_path):
    payload = _payload()
    first_api = FakeObservationApi([
        ApiError("offline"),
    ])
    first = CheckpointObservationReporterV2(first_api, tmp_path)
    assert first.record(payload)
    first_result = first.flush_once()
    assert first_result == ObservationDeliveryResultV2(
        persisted=1, retryable=1,
    )

    second = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    assert second.flush_once() == ObservationDeliveryResultV2(acknowledged=1)

    health = json.loads(
        (
            tmp_path / "checkpoint-v2" / "observations" / "delivery-health"
            / "assignment-0001.json"
        ).read_text(),
    )
    assert health["persisted"] == 1
    assert health["retryable"] == 1
    assert health["acknowledged"] == 1
    assert health["rejected"] == 0
    assert health["dropped"] == 0


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


def test_cohort_registration_is_private_idempotent_and_conflict_safe(tmp_path):
    spool = CheckpointObservationSpoolV2(tmp_path / "observations")
    payload = _cohort_registration()
    assert spool.register_cohort(payload) is True
    assert spool.register_cohort(payload) is False
    path = next(spool.cohort_root.glob("*.json"))
    if hasattr(os, "getuid"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    changed = json.loads(json.dumps(payload))
    changed["cohort"]["platform"] = "linux"
    with pytest.raises(CheckpointObservationSpoolError, match="conflicts"):
        spool.register_cohort(changed)


def test_total_spool_budget_bounds_cohorts_and_exact_replay_survives(tmp_path):
    spool = CheckpointObservationSpoolV2(
        tmp_path / "observations", max_total_files=2,
    )
    first = _cohort_registration()
    second = json.loads(json.dumps(first))
    second.update({
        "assignment_id": "assignment-0002",
        "runner_session_id": "session-0002",
        "identity_fingerprint": "b" * 64,
    })

    assert spool.register_cohort(first) is True
    assert spool.register_cohort(first) is False
    with pytest.raises(
        CheckpointObservationSpoolError, match="total spool is full",
    ):
        spool.register_cohort(second)

    assert len(list(spool.cohort_root.glob("*.json"))) == 1


def test_total_spool_budget_counts_hidden_crash_temporaries(tmp_path):
    spool = CheckpointObservationSpoolV2(
        tmp_path / "observations", max_total_files=2,
    )
    spool._prepare()
    orphan = spool.pending_root / ".crash-left.tmp"
    orphan.write_bytes(b"partial")
    orphan.chmod(0o600)

    with pytest.raises(
        CheckpointObservationSpoolError, match="total spool is full",
    ):
        spool.persist(_payload())

    assert orphan.read_bytes() == b"partial"
    assert spool.pending() == []


def test_exact_replay_cannot_bypass_total_spool_budget(tmp_path):
    spool = CheckpointObservationSpoolV2(
        tmp_path / "observations", max_total_files=2,
    )
    registration = _cohort_registration()
    assert spool.register_cohort(registration) is True
    orphan = spool.pending_root / ".crash-left.tmp"
    orphan.write_bytes(b"partial")
    orphan.chmod(0o600)

    with pytest.raises(
        CheckpointObservationSpoolError, match="total spool is full",
    ):
        spool.register_cohort(registration)

    assert orphan.read_bytes() == b"partial"


def test_reporter_treats_exact_cohort_replay_as_durably_registered(tmp_path):
    reporter = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    registration = _cohort_registration()

    assert reporter.register_cohort(registration) is True
    assert reporter.register_cohort(registration) is True
    assert len(list(reporter.spool.cohort_root.glob("*.json"))) == 1


def test_mainline_impact_pair_is_private_idempotent_and_conflict_safe(tmp_path):
    spool = CheckpointObservationSpoolV2(tmp_path / "observations")
    start = _mainline_start()

    sample_id = spool.begin_mainline_impact(start)
    assert spool.begin_mainline_impact(dict(start)) == sample_id
    path = spool.mainline_impact_root / f"{sample_id}.json"
    opened = json.loads(path.read_text())
    assert opened["state"] == "started"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    completion = _mainline_completion(start)
    assert spool.complete_mainline_impact(sample_id, completion) is True
    assert spool.complete_mainline_impact(sample_id, completion) is False
    completed = json.loads(path.read_text())
    assert completed["state"] == "completed"
    assert completed["submission_preserved"] is True

    with pytest.raises(CheckpointObservationSpoolError, match="conflicts"):
        spool.complete_mainline_impact(
            sample_id, dict(completion, submission_preserved=False),
        )
    assert json.loads(path.read_text()) == completed

    with pytest.raises(CheckpointObservationSpoolError, match="fields"):
        spool.complete_mainline_impact(
            sample_id, dict(completion, assignment_id="assignment-evil"),
        )
    assert json.loads(path.read_text()) == completed


def test_mainline_impact_attempts_are_distinct_and_share_the_global_cap(tmp_path):
    spool = CheckpointObservationSpoolV2(
        tmp_path / "observations",
        max_total_files=3,
    )
    assert spool.register_cohort(_cohort_registration()) is True
    first = _mainline_start()
    first_id = spool.begin_mainline_impact(first)

    # The lock, cohort and first impact record exactly fill the global file
    # budget. Exact replay and completion remain possible at the cap.
    assert spool.begin_mainline_impact(dict(first)) == first_id
    completion = _mainline_completion(first)
    assert spool.complete_mainline_impact(first_id, completion) is True
    assert spool.complete_mainline_impact(first_id, completion) is False

    second = dict(first, attempt_id="attempt-0002")
    with pytest.raises(CheckpointObservationSpoolError, match="total spool is full"):
        spool.begin_mainline_impact(second)


def test_mainline_impact_completion_without_start_and_lock_contention_fail_open(
    tmp_path,
):
    spool = CheckpointObservationSpoolV2(tmp_path / "observations")
    start = _mainline_start()
    sample_id = "impact-" + "1" * 48
    with pytest.raises((CheckpointObservationSpoolError, FileNotFoundError)):
        spool.complete_mainline_impact(sample_id, _mainline_completion(start))

    spool._thread_lock.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(CheckpointObservationSpoolError, match="busy"):
            spool.begin_mainline_impact(start)
        assert time.monotonic() - started < 0.1
    finally:
        spool._thread_lock.release()


def test_local_evidence_emits_content_bound_mainline_impact_attestation(tmp_path):
    reporter = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    registration = _cohort_registration()
    assert reporter.register_cohort(registration) is True
    start = _mainline_start()
    sample_id = reporter.begin_mainline_impact(start)
    assert sample_id is not None
    assert reporter.complete_mainline_impact(
        sample_id, _mainline_completion(start),
    ) is True

    packet = checkpoint_local_evidence_v2(
        tmp_path,
        now=datetime.fromisoformat(start["started_at"]) + timedelta(seconds=20),
    )
    impacts = [
        item for item in packet["attestations"]
        if item["kind"] == "mainline_impact"
    ]
    assert len(impacts) == 1
    assert impacts[0]["metrics"] == {
        "paired_samples": 1,
        "incomplete_samples": 0,
        "excluded_samples": 0,
        "p95_added_latency_ratio": 0.0025,
        "result_mismatches": 0,
        "assignment_mutations": 0,
        "submission_losses": 0,
    }
    serialized = json.dumps(packet)
    assert str(tmp_path) not in serialized
    assert "assignment-0001" not in serialized


def test_incomplete_or_unregistered_mainline_sample_cannot_look_healthy(tmp_path):
    reporter = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    assert reporter.register_cohort(_cohort_registration()) is True
    sample_id = reporter.begin_mainline_impact(_mainline_start())
    assert sample_id is not None

    packet = checkpoint_local_evidence_v2(
        tmp_path, now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    impact = next(
        item for item in packet["attestations"]
        if item["kind"] == "mainline_impact"
    )
    assert impact["metrics"]["paired_samples"] == 0
    assert impact["metrics"]["incomplete_samples"] == 1

    other = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path / "other")
    unregistered_id = other.begin_mainline_impact(_mainline_start())
    assert unregistered_id is not None
    unregistered = checkpoint_local_evidence_v2(
        tmp_path / "other",
        now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert unregistered["unregistered_records"] == 1
    assert all(
        item["kind"] != "mainline_impact"
        for item in unregistered["attestations"]
    )


def test_local_evidence_reports_hidden_crash_temporary_as_scan_error(tmp_path):
    reporter = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    assert reporter.register_cohort(_cohort_registration()) is True
    hidden = reporter.spool.mainline_impact_root / ".impact-crash.tmp"
    hidden.write_bytes(b"partial")
    hidden.chmod(0o600)

    packet = checkpoint_local_evidence_v2(
        tmp_path, now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )

    assert packet["scan_errors"] == 1
    assert hidden.read_bytes() == b"partial"


def test_local_evidence_rejects_future_or_predating_mainline_samples(tmp_path):
    instant = datetime.now(timezone.utc).replace(microsecond=0)
    reporter = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    assert reporter.register_cohort(_cohort_registration()) is True
    future = dict(
        _mainline_start(),
        started_at=(instant + timedelta(minutes=5)).isoformat(),
    )
    assert reporter.begin_mainline_impact(future) is not None
    with pytest.raises(CheckpointObservationSpoolError, match="time is invalid"):
        checkpoint_local_evidence_v2(
            tmp_path, now=instant + timedelta(seconds=1),
        )

    other = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path / "other")
    assert other.register_cohort(_cohort_registration()) is True
    predating = dict(
        _mainline_start(),
        started_at=(instant - timedelta(minutes=5)).isoformat(),
    )
    assert other.begin_mainline_impact(predating) is not None
    with pytest.raises(CheckpointObservationSpoolError, match="time is invalid"):
        checkpoint_local_evidence_v2(
            tmp_path / "other", now=instant + timedelta(seconds=1),
        )


def test_local_evidence_mainline_metrics_are_scoped_to_requested_window(tmp_path):
    instant = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
    since = instant - timedelta(hours=1)
    reporter = CheckpointObservationReporterV2(FakeObservationApi(), tmp_path)
    registration = {
        **_cohort_registration(),
        "registered_at": (instant - timedelta(hours=2)).isoformat(),
    }
    assert reporter.register_cohort(registration) is True

    old = dict(
        _mainline_start(),
        attempt_id="attempt-old1",
        started_at=(instant - timedelta(minutes=90)).isoformat(),
    )
    old_id = reporter.begin_mainline_impact(old)
    assert old_id is not None
    assert reporter.complete_mainline_impact(
        old_id, _mainline_completion(old),
    ) is True

    recent = dict(
        _mainline_start(),
        attempt_id="attempt-new1",
        started_at=(instant - timedelta(minutes=30)).isoformat(),
    )
    recent_id = reporter.begin_mainline_impact(recent)
    assert recent_id is not None
    assert reporter.complete_mainline_impact(
        recent_id, _mainline_completion(recent),
    ) is True

    packet = checkpoint_local_evidence_v2(
        tmp_path, since=since, now=instant,
    )
    impact = next(
        item for item in packet["attestations"]
        if item["kind"] == "mainline_impact"
    )
    assert impact["observed_from"] == since.isoformat()
    assert impact["observed_until"] == instant.isoformat()
    assert impact["metrics"]["paired_samples"] == 1

    with pytest.raises(CheckpointObservationSpoolError, match="window is invalid"):
        checkpoint_local_evidence_v2(
            tmp_path, since=instant, now=instant,
        )
    with pytest.raises(CheckpointObservationSpoolError, match="window is invalid"):
        checkpoint_local_evidence_v2(
            tmp_path,
            since=datetime(2026, 8, 24, 11, 0),
            now=instant,
        )


def test_local_evidence_reports_assignment_scoped_backlog_and_drops(tmp_path):
    reporter = CheckpointObservationReporterV2(
        FakeObservationApi([ApiError("offline")]), tmp_path, queue_size=1,
    )
    assert reporter.register_cohort(_cohort_registration()) is True
    assert reporter.record(_payload()) is True
    assert reporter.record(_payload("operation-0002", "capture-0002")) is False
    flushed = reporter.flush_once()
    assert flushed.retryable == 1

    packet = checkpoint_local_evidence_v2(
        tmp_path,
        now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert packet["schema"] == "dradar-checkpoint-v2-local-evidence-v1"
    assert packet["unregistered_records"] == 0
    assert len(packet["attestations"]) == 1
    attestation = packet["attestations"][0]
    assert attestation["kind"] == "outbox_health"
    assert attestation["cohort"]["platform"] == "macos"
    assert attestation["metrics"]["observation_processes"] == 1
    assert attestation["metrics"]["pending_records"] == 1
    assert attestation["metrics"]["persisted_records"] == 1
    assert attestation["metrics"]["acknowledged_records"] == 0
    assert attestation["metrics"]["retryable_deliveries"] == 1
    assert attestation["metrics"]["rejected_deliveries"] == 0
    assert attestation["metrics"]["dropped_records"] == 1
    serialized = json.dumps(packet)
    assert str(tmp_path) not in serialized
    assert "offline" not in serialized


def test_local_evidence_counts_hidden_crash_residue_in_both_storage_lanes(
    tmp_path,
):
    reporter = CheckpointObservationReporterV2(
        FakeObservationApi(), tmp_path,
    )
    assert reporter.register_cohort(_cohort_registration()) is True
    for scope in ("shadow", "authoritative"):
        assignment = tmp_path / "checkpoint-v2" / scope / "assignment-0001"
        downloads = assignment / ".downloads"
        downloads.mkdir(parents=True, mode=0o700)
        (downloads / f"capture-0001-{scope}.tar.gz.part").write_bytes(
            b"partial",
        )
        checkpoint = assignment / "checkpoints" / "checkpoint-0001"
        generations = checkpoint / "generations"
        generations.mkdir(parents=True, mode=0o700)
        (checkpoint / f".incoming-capture-0001-{scope}.part").mkdir(
            mode=0o700,
        )
        (generations / f".retention-generation-{scope}").mkdir(mode=0o700)

    packet = checkpoint_local_evidence_v2(
        tmp_path,
        now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )

    assert packet["attestations"][0]["metrics"]["cleanup_residue"] == 6


def test_local_evidence_blocks_on_unreleased_shadow_generation(tmp_path):
    reporter = CheckpointObservationReporterV2(
        FakeObservationApi(), tmp_path,
    )
    assert reporter.register_cohort(_cohort_registration()) is True
    generation = (
        tmp_path / "checkpoint-v2" / "shadow" / "assignment-0001"
        / "checkpoints" / "checkpoint-0001" / "generations"
        / "generation-00000000000000000001"
    )
    generation.mkdir(parents=True, mode=0o700)

    packet = checkpoint_local_evidence_v2(
        tmp_path,
        now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert packet["attestations"][0]["metrics"]["cleanup_residue"] == 1

    generation.rmdir()
    packet = checkpoint_local_evidence_v2(
        tmp_path,
        now=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert packet["attestations"][0]["metrics"]["cleanup_residue"] == 0


def test_local_evidence_rejects_cross_session_assignment_cohort_drift(tmp_path):
    spool = CheckpointObservationSpoolV2(
        tmp_path / "checkpoint-v2" / "observations",
    )
    first = _cohort_registration()
    second = json.loads(json.dumps(first))
    second["runner_session_id"] = "session-0002"
    second["cohort"]["platform"] = "linux"
    assert spool.register_cohort(first)
    assert spool.register_cohort(second)
    with pytest.raises(CheckpointObservationSpoolError, match="drifted"):
        checkpoint_local_evidence_v2(
            tmp_path,
            now=datetime.now(timezone.utc) + timedelta(seconds=1),
        )
