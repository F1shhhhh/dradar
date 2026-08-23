from types import SimpleNamespace

import pytest

from dradar import runloop


class ImpactTelemetry:
    def __init__(self):
        self.calls = []

    def complete_checkpoint_mainline_impact(self, sample_id, payload):
        self.calls.append((sample_id, payload))
        return True


def _assignment():
    return {
        "assignment_id": "assignment-0001",
        "checkpoint_protocol_version": 1,
        "checkpoint_id": None,
        "execution_state": "running",
        "owner_epoch": 0,
        "resume_generation": 0,
    }


def _art():
    return SimpleNamespace(
        checkpoint_impact_sample_id="impact-" + "1" * 48,
        checkpoint_sync_elapsed_ms=25,
        duration_sec=10.0,
    )


@pytest.mark.parametrize(
    ("upload_outcome", "pending_entries", "preserved", "comparable"),
    [
        ("submitted", [], True, True),
        ("upload-failed", [{"assignment_id": "assignment-0001"}], True, True),
        ("upload-failed", [], False, True),
        ("not-uploaded", [], False, False),
    ],
)
def test_mainline_impact_completion_tracks_durable_result_handoff(
    monkeypatch, upload_outcome, pending_entries, preserved, comparable,
):
    monkeypatch.setattr(runloop.pending, "load", lambda _home: pending_entries)
    telemetry = ImpactTelemetry()

    assert runloop._complete_checkpoint_mainline_impact(
        telemetry,
        _assignment(),
        _art(),
        outcome="completed",
        upload_outcome=upload_outcome,
    ) is True

    sample_id, payload = telemetry.calls[0]
    assert sample_id == "impact-" + "1" * 48
    assert payload["mainline_elapsed_ms"] == 10_000
    assert payload["checkpoint_sync_elapsed_ms"] == 25
    assert payload["result_preserved"] is preserved
    assert payload["submission_preserved"] is preserved
    assert payload["comparable"] is comparable
    assert payload["exclusion_reason"] == (
        None if comparable else "ordinary-result-not-submittable"
    )
    assert len(payload["assignment_state_sha256_after"]) == 64


def test_mainline_impact_absence_is_a_noop(monkeypatch):
    monkeypatch.setattr(runloop.pending, "load", lambda _home: [])
    telemetry = ImpactTelemetry()
    art = _art()
    art.checkpoint_impact_sample_id = None

    assert runloop._complete_checkpoint_mainline_impact(
        telemetry,
        _assignment(),
        art,
        outcome="completed",
        upload_outcome="submitted",
    ) is False
    assert telemetry.calls == []


def test_handled_nonresult_attempt_is_settled_as_explicit_exclusion(monkeypatch):
    monkeypatch.setattr(runloop.pending, "load", lambda _home: [])
    telemetry = ImpactTelemetry()
    controller = SimpleNamespace(
        impact_sample_id="impact-" + "2" * 48,
        checkpoint_sync_elapsed_ms=12,
        mainline_elapsed_ms=4_000,
    )

    assert runloop._complete_checkpoint_mainline_without_result(
        telemetry, _assignment(), controller,
    ) is True
    sample_id, payload = telemetry.calls[0]
    assert sample_id == "impact-" + "2" * 48
    assert payload["mainline_elapsed_ms"] == 4_000
    assert payload["checkpoint_sync_elapsed_ms"] == 12
    assert payload["outcome"] == "interrupted"
    assert payload["comparable"] is False
    assert payload["exclusion_reason"] == "ordinary-result-not-submittable"


def test_unclean_nonresult_attempt_remains_incomplete(monkeypatch):
    monkeypatch.setattr(runloop.pending, "load", lambda _home: [])
    telemetry = ImpactTelemetry()
    controller = SimpleNamespace(
        impact_sample_id=None,
        checkpoint_sync_elapsed_ms=12,
        mainline_elapsed_ms=4_000,
    )

    assert runloop._complete_checkpoint_mainline_without_result(
        telemetry, _assignment(), controller,
    ) is False
    assert telemetry.calls == []
