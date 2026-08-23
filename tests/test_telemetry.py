import json

from dradar.api_client import ApiError
from dradar.telemetry import RunnerTelemetry


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.heartbeats = []
        self.closes = []
        self.checkpoint_observations = []

    def runner_heartbeat(self, payload):
        self.heartbeats.append(payload)
        response = self.responses.pop(0) if self.responses else {
            "accepted": True, "action": "continue", "batch_id": "batch-1",
            "next_heartbeat_sec": 60,
        }
        if isinstance(response, Exception):
            raise response
        return response

    def runner_close(self, payload):
        self.closes.append(payload)
        return {"ok": True}

    def checkpoint_v2_observation(self, payload):
        self.checkpoint_observations.append(payload)
        return {
            "ok": True,
            "assignment_id": payload["assignment_id"],
            "capture_id": payload["capture_id"],
            "status": payload["status"],
            "assignment_unchanged": True,
            "paid_execution_authorized": False,
        }


def _checkpoint_observation():
    return {
        "observation_kind": "capture",
        "assignment_id": "assignment-0001",
        "operation_id": "operation-0001",
        "capture_id": "capture-0001",
        "checkpoint_id": "checkpoint-0001",
        "checkpoint_lineage_id": "lineage-0001",
        "snapshot_generation": 1,
        "rollout_mode": "observe",
        "status": "failed",
        "stage": "capture",
        "failure_code": "capture_failed",
        "failure_type": "RuntimeError",
        "identity_fingerprint": "a" * 64,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": "codex-openai/v1",
        "capture_storage": "container_native",
        "manifest_sha256": None,
        "archive_sha256": None,
        "archive_bytes": None,
        "file_count": None,
        "payload_bytes": None,
        "elapsed_ms": 12,
        "platform": "macos",
        "container_backend": "orbstack",
        "client_version": "0.5.97",
        "adapter_version": "codex-openai/v1",
        "remote_cleanup": "not_needed",
        "authoritative": False,
        "selected_local": False,
    }


def test_payload_is_one_session_not_one_per_assignment_and_stays_small():
    client = FakeClient()
    telemetry = RunnerTelemetry(client, jitter=False, target_workers=20)
    telemetry.bind_batch("batch-1")
    telemetry.set_phase("running", "assignment-1", 3)
    assert telemetry._send_once() == 60
    telemetry.set_phase("running", "assignment-2")
    telemetry._send_once()

    assert {p["session_id"] for p in client.heartbeats} == {telemetry.session_id}
    assert [p["active_assignment_id"] for p in client.heartbeats] == [
        "assignment-1", "assignment-2"]
    assert [p["resume_generation"] for p in client.heartbeats] == [3, None]
    assert client.heartbeats[1]["seq"] > client.heartbeats[0]["seq"]
    assert len(json.dumps(client.heartbeats[-1]).encode()) < 1024
    assert set(client.heartbeats[-1]) == {
        "protocol_version", "client_version", "session_id", "batch_id", "seq",
        "phase", "active_assignment_id", "resume_generation",
        "client_monotonic_ms", "progress_counter",
        "platform", "target_workers",
    }
    assert client.heartbeats[-1]["target_workers"] == 20


def test_target_worker_count_is_bounded():
    client = FakeClient()
    for value in (0, 41):
        try:
            RunnerTelemetry(client, target_workers=value)
        except ValueError as exc:
            assert "between 1 and 40" in str(exc)
        else:
            raise AssertionError("out-of-range target worker count was accepted")


def test_checkpoint_runtime_identity_is_bounded_and_immutable():
    client = FakeClient()
    telemetry = RunnerTelemetry(client, jitter=False)
    telemetry.configure_checkpoint_runtime(
        container_backend="orbstack",
        machine_fingerprint="f" * 64,
    )
    telemetry._send_once()
    payload = client.heartbeats[-1]
    assert payload["container_backend"] == "orbstack"
    assert payload["checkpoint_machine_fingerprint"] == "f" * 64
    try:
        telemetry.configure_checkpoint_runtime(
            container_backend="docker",
            machine_fingerprint="e" * 64,
        )
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("checkpoint runtime identity drift was accepted")


def test_checkpoint_observation_reporting_is_opt_in_and_account_scoped(tmp_path):
    client = FakeClient()
    telemetry = RunnerTelemetry(client, jitter=False)
    payload = _checkpoint_observation()

    assert telemetry.record_checkpoint_observation(payload) is False
    assert not (tmp_path / "checkpoint-v2").exists()

    telemetry.configure_checkpoint_observation_reporting(tmp_path)
    assert telemetry.record_checkpoint_observation(payload) is True
    reporter = telemetry._checkpoint_observation_reporter
    assert reporter is not None
    result = reporter.flush_once()
    assert result.persisted == 1
    assert result.acknowledged == 1
    assert client.checkpoint_observations == [payload]

    telemetry.configure_checkpoint_observation_reporting(tmp_path)
    try:
        telemetry.configure_checkpoint_observation_reporting(tmp_path / "other")
    except ValueError as exc:
        assert "account boundary" in str(exc)
    else:
        raise AssertionError("checkpoint observation account boundary drift was accepted")


def test_server_can_slow_cadence_but_not_make_it_pathological():
    client = FakeClient([
        {"next_heartbeat_sec": 99999},
        {"next_heartbeat_sec": 1},
    ])
    telemetry = RunnerTelemetry(client, jitter=False)
    assert telemetry._send_once() == 600
    assert telemetry._send_once() == 30


def test_server_notices_are_bounded_validated_and_printed_once(capsys):
    notice = {
        "id": "dsh-usage-upgrade-20260814",
        "severity": "warning",
        "message": "当前任务继续运行。\n完成后刷新 CLI。",
    }
    client = FakeClient([
        {"next_heartbeat_sec": 60, "notices": [notice]},
        {"next_heartbeat_sec": 60, "notices": [notice]},
        {"next_heartbeat_sec": 60, "notices": [
            {"id": "bad", "severity": "unknown", "message": "ignored"},
            "not-an-object",
        ]},
    ])
    telemetry = RunnerTelemetry(client, jitter=False)

    telemetry._send_once()
    telemetry._send_once()
    telemetry._send_once()

    err = capsys.readouterr().err
    assert err.count("dsh-usage-upgrade-20260814") == 0
    assert err.count("server notice [warning]") == 1
    assert "当前任务继续运行。 完成后刷新 CLI。" in err
    assert "ignored" not in err


def test_three_failures_warn_once_then_recovery_is_visible(capsys):
    client = FakeClient([
        ApiError("offline"), ApiError("offline"), ApiError("offline"),
        ApiError("offline"), {"next_heartbeat_sec": 120},
    ])
    telemetry = RunnerTelemetry(client, jitter=False)
    for _ in range(5):
        telemetry._send_once()
    err = capsys.readouterr().err
    assert err.count("warning:") == 1
    assert "recovered" in err


def test_old_server_404_disables_future_traffic_silently(capsys):
    client = FakeClient([ApiError("not found", status_code=404)])
    telemetry = RunnerTelemetry(client, jitter=False)
    telemetry._send_once()
    telemetry._send_once()
    assert len(client.heartbeats) == 1
    assert capsys.readouterr().err == ""


def test_close_carries_only_session_batch_seq_and_reason():
    client = FakeClient()
    telemetry = RunnerTelemetry(client, jitter=False)
    telemetry.bind_batch("batch-1")
    telemetry._send_once()
    telemetry.close("paused")
    assert client.closes == [{
        "session_id": telemetry.session_id,
        "batch_id": "batch-1",
        "seq": 2,
        "reason": "paused",
    }]
