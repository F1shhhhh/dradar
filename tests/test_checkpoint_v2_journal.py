from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from dradar.api_client import ApiError
from dradar.checkpoint_v2 import (
    CHECKPOINT_CORE_ABI_V2,
    JOURNAL_DIR,
    CheckpointActivationV2,
    CheckpointV2CommandRejected,
    CheckpointV2Journal,
    CheckpointV2JournalError,
    CheckpointV2OperationConflict,
    CheckpointV2ProtocolError,
    CheckpointRolloutModeV2,
    CheckpointV2StateMachine,
    CheckpointFailureV2,
    ExecutionIdentityV2,
    FinalizedIdentityReceiptV2,
    PaidExecutionPermit,
    SealedCheckpointV2,
    UsageEventV2,
    UsageSegmentEvidenceV2,
    checkpoint_policy_v2,
    checkpoint_machine_fingerprint,
    checkpoint_activation_from_assignment_v2,
    completed_restore_receipt,
    finalize_execution_identity_v2,
    negotiate_checkpoint_activation_v2,
)


class FakeApi:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def checkpoint_v2_command(self, command, payload):
        self.calls.append((command, dict(payload)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.mark.parametrize(
    ("local", "server", "controlled", "effective"),
    [
        ("off", "on", False, CheckpointRolloutModeV2.OFF),
        ("observe", "on", False, CheckpointRolloutModeV2.OBSERVE),
        ("restore-test", "on", False, CheckpointRolloutModeV2.RESTORE_TEST),
        ("on", "observe", False, CheckpointRolloutModeV2.OBSERVE),
        ("canary", "canary", False, CheckpointRolloutModeV2.RESTORE_TEST),
        ("canary", "canary", True, CheckpointRolloutModeV2.CANARY),
        ("on", "on", False, CheckpointRolloutModeV2.ON),
    ],
)
def test_rollout_negotiation_requires_two_sided_authorization(
    local: str,
    server: str,
    controlled: bool,
    effective: CheckpointRolloutModeV2,
) -> None:
    activation = negotiate_checkpoint_activation_v2(
        local_mode=local,
        server_mode=server,
        controlled_account=controlled,
    )
    assert isinstance(activation, CheckpointActivationV2)
    assert activation.effective_mode == effective
    assert activation.capture_enabled is (
        effective >= CheckpointRolloutModeV2.OBSERVE
    )
    assert activation.offline_restore_enabled is (
        effective >= CheckpointRolloutModeV2.RESTORE_TEST
    )
    assert activation.paid_resume_enabled is (
        effective in {CheckpointRolloutModeV2.CANARY, CheckpointRolloutModeV2.ON}
    )
    assert activation.writer_failure_changes_assignment is False
    assert activation.failure_disposition == "continue_without_checkpoint"


@pytest.mark.parametrize("value", [None, "", "enabled", 1, object()])
def test_rollout_negotiation_rejects_unknown_values(value: object) -> None:
    with pytest.raises(CheckpointV2ProtocolError, match="rollout mode"):
        negotiate_checkpoint_activation_v2(local_mode=value, server_mode="off")


def test_assignment_rollout_is_a_server_side_cap_not_a_local_override() -> None:
    assignment = _assignment()
    assignment["checkpoint_v2_rollout_mode"] = "observe"
    assignment["checkpoint_v2_controlled_account"] = False
    activation = checkpoint_activation_from_assignment_v2(
        assignment, local_mode="on",
    )
    assert activation.effective_mode == CheckpointRolloutModeV2.OBSERVE
    assert activation.authoritative is False

    assignment["checkpoint_v2_rollout_mode"] = "canary"
    activation = checkpoint_activation_from_assignment_v2(
        assignment, local_mode="canary",
    )
    assert activation.effective_mode == CheckpointRolloutModeV2.RESTORE_TEST
    assignment["checkpoint_v2_controlled_account"] = True
    activation = checkpoint_activation_from_assignment_v2(
        assignment, local_mode="canary",
    )
    assert activation.effective_mode == CheckpointRolloutModeV2.CANARY


def _payload(**updates):
    value = {
        "assignment_id": "assignment-0001",
        "operation_id": "operation-0001",
        "session_id": "session-0001",
        "expected_owner_epoch": 0,
    }
    value.update(updates)
    return value


def _checkout_ack():
    return {
        "ok": True,
        "assignment_id": "assignment-0001",
        "owner_epoch": 1,
        "execution_state": "preparing",
        "paid_execution_authorized": False,
    }


def _usage_ledger_ack(
    *,
    complete: bool = True,
    segment_count: int = 1,
    totals: tuple[int, int, int] = (10, 2, 3),
    digest: str = "9" * 64,
) -> dict:
    return {
        "schema": "dradar-checkpoint-usage-ledger-v2",
        "usage_segment_count": segment_count,
        "finalized_segment_count": segment_count,
        "open_segment_count": 0,
        "complete": complete,
        "n_input_tokens": totals[0] if complete else None,
        "n_cache_tokens": totals[1] if complete else None,
        "n_output_tokens": totals[2] if complete else None,
        "ledger_sha256": digest,
    }


def _assignment(
    *,
    harness: str = "codex",
    provider: str = "openai",
    agent: str = "codex",
    runtime_digest: str = "b" * 64,
):
    return {
        "assignment_id": "assignment-0001",
        "checkpoint_protocol_version": 2,
        "checkpoint_v2_identity_protocol_version": 2,
        "checkpoint_v2_rollout_mode": "on",
        "owner_epoch": 0,
        "benchmark_id": "deep-swe",
        "task_content_hash": "a" * 64,
        "agent": agent,
        "provider": provider,
        "model": "model-v2",
        "effort": "high",
        "agent_version": "1.2.3",
        "execution_identity": {
            "benchmark_id": "deep-swe",
            "task_content_sha256": "a" * 64,
            "harness": harness,
            "provider": provider,
            "model": "model-v2",
            "effort": "high",
            "agent_version": "1.2.3",
            "runtime_profile": "runtime-profile-v2",
            "model_config_version": "model-config-v2",
            "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
            "checkpoint_abi": f"dradar-checkpoint-v2/{harness}/1",
            "identity_state": "FINAL",
            "identity_source": "claim_snapshot",
            "runtime_compatibility_digest": runtime_digest,
        },
    }


def _on_activation() -> CheckpointActivationV2:
    return negotiate_checkpoint_activation_v2(
        local_mode="on", server_mode="on",
    )


def _sealed_generation(
    generation: int,
    *,
    checkpoint_id: str | None = None,
    manifest_char: str | None = None,
) -> SealedCheckpointV2:
    return SealedCheckpointV2(
        checkpoint_id=checkpoint_id or f"checkpoint-{generation:04d}",
        checkpoint_lineage_id="lineage-0001",
        snapshot_generation=generation,
        capture_id=f"capture-{generation:04d}",
        manifest_schema=2,
        manifest_sha256=(manifest_char or str(generation)) * 64,
        compatibility_fingerprint="c" * 64,
        recovery_capability="NATIVE_VALID",
        native_state_schema="codex-session/1",
        storage_scope="machine_local",
        writer_machine_fingerprint="f" * 64,
        sync_state="local_only",
    )


def test_journal_writes_private_pending_then_reuses_ack_without_network(
    tmp_path: Path,
) -> None:
    journal = CheckpointV2Journal(tmp_path)
    payload = _payload()
    api = FakeApi([_checkout_ack()])

    first = journal.execute(api, "checkout", payload)
    second = journal.execute(
        FakeApi([AssertionError("acknowledged command must not use network")]),
        "checkout",
        payload,
    )
    assert first == second == _checkout_ack()
    entry = journal.load("assignment-0001", "operation-0001")
    assert entry.state == "ACKNOWLEDGED"
    assert entry.response == _checkout_ack()
    path = (
        tmp_path / JOURNAL_DIR / "assignment-0001" / "commands"
        / "operation-0001.json"
    )
    if hasattr(os, "getuid"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert api.calls == [("checkout", payload)]


def test_transport_failure_stays_pending_and_retries_same_operation(
    tmp_path: Path,
) -> None:
    journal = CheckpointV2Journal(tmp_path)
    payload = _payload()
    unavailable = FakeApi([ApiError("network unavailable")])
    with pytest.raises(ApiError):
        journal.execute(unavailable, "checkout", payload)
    assert journal.load(
        "assignment-0001", "operation-0001",
    ).state == "PENDING"

    recovered = FakeApi([_checkout_ack()])
    assert journal.execute(recovered, "checkout", payload) == _checkout_ack()
    assert recovered.calls == [("checkout", payload)]


def test_stable_4xx_is_rejected_without_persisting_free_form_detail(
    tmp_path: Path,
) -> None:
    journal = CheckpointV2Journal(tmp_path)
    payload = _payload()
    blocked = ApiError(
        "verbose server diagnostic must not enter the journal",
        status_code=409,
        code="checkpoint_circuit_open",
        detail={"code": "checkpoint_circuit_open", "detail": "verbose"},
    )
    with pytest.raises(ApiError):
        journal.execute(FakeApi([blocked]), "checkout", payload)
    entry = journal.load("assignment-0001", "operation-0001")
    assert entry.state == "REJECTED"
    assert entry.error_status == 409
    assert entry.error_code == "checkpoint_circuit_open"
    raw = entry.payload
    assert "verbose" not in json.dumps(raw)

    with pytest.raises(CheckpointV2CommandRejected) as caught:
        journal.execute(FakeApi([]), "checkout", payload)
    assert caught.value.status_code == 409
    assert caught.value.code == "checkpoint_circuit_open"


def test_operation_id_cannot_be_reused_for_a_different_request(
    tmp_path: Path,
) -> None:
    journal = CheckpointV2Journal(tmp_path)
    journal.begin("checkout", _payload())
    with pytest.raises(CheckpointV2OperationConflict):
        journal.begin("checkout", _payload(expected_owner_epoch=7))
    with pytest.raises(CheckpointV2OperationConflict):
        journal.begin("start", _payload())


def test_sensitive_fields_are_rejected_before_disk_or_network(tmp_path: Path) -> None:
    journal = CheckpointV2Journal(tmp_path)
    with pytest.raises(CheckpointV2JournalError, match="forbidden field"):
        journal.execute(
            FakeApi([AssertionError("unsafe payload must not use network")]),
            "failure",
            _payload(provider_token="do-not-store"),
        )


@pytest.mark.parametrize("command", ["start", "resume-commit"])
def test_paid_commands_require_explicit_server_authorization(
    tmp_path: Path, command: str,
) -> None:
    payload = _payload(operation_id=f"operation-{command.replace('-', '')}")
    response = {
        "ok": True,
        "assignment_id": "assignment-0001",
        "owner_epoch": 1,
        "execution_state": "running",
        "paid_execution_authorized": False,
    }
    journal = CheckpointV2Journal(tmp_path)
    with pytest.raises(
        CheckpointV2ProtocolError, match="not explicitly authorized",
    ):
        journal.execute(FakeApi([response]), command, payload)
    assert journal.load(
        "assignment-0001", payload["operation_id"],
    ).state == "PENDING"


@pytest.mark.parametrize("command", ["checkout", "resume-reserve", "failure"])
def test_free_only_commands_cannot_smuggle_paid_authorization(
    tmp_path: Path, command: str,
) -> None:
    payload = _payload(operation_id=f"operation-{command.replace('-', '')}")
    response = {
        "ok": True,
        "assignment_id": "assignment-0001",
        "owner_epoch": 1,
        "execution_state": "preparing",
        "paid_execution_authorized": True,
    }
    journal = CheckpointV2Journal(tmp_path)
    with pytest.raises(
        CheckpointV2ProtocolError, match="free-only",
    ):
        journal.execute(FakeApi([response]), command, payload)


def test_symlinked_journal_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / JOURNAL_DIR).symlink_to(target, target_is_directory=True)
    with pytest.raises(CheckpointV2JournalError, match="directory is unsafe"):
        CheckpointV2Journal(tmp_path).begin("checkout", _payload())


def test_machine_fingerprint_is_private_stable_and_not_host_derived(
    tmp_path: Path,
) -> None:
    first = checkpoint_machine_fingerprint(tmp_path)
    second = checkpoint_machine_fingerprint(tmp_path)
    assert first == second
    assert len(first) == 64
    identity_path = tmp_path / JOURNAL_DIR / "machine-identity.json"
    raw = identity_path.read_text(encoding="utf-8")
    assert "hostname" not in raw
    assert "username" not in raw
    assert str(tmp_path) not in raw
    if hasattr(os, "getuid"):
        assert stat.S_IMODE(identity_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("harness", "provider", "agent", "mode", "fallback_limit"),
    [
        ("codex", "openai", "codex", "native_preferred", 1),
        ("codex", "deepseek", "codex", "native_preferred", 1),
        ("dsh", "deepseek", "dsh-minimal", "native_required", 0),
        ("zcode", "bigmodel-coding-plan", "zcode", "native_preferred", 1),
        ("kimi-code", "kimi-subscription", "kimi-code", "native_required", 0),
        ("grok-build", "xai-subscription", "grok-build", "unsupported", 0),
    ],
)
def test_authoritative_identity_selects_an_exact_harness_policy(
    harness: str,
    provider: str,
    agent: str,
    mode: str,
    fallback_limit: int,
) -> None:
    identity = ExecutionIdentityV2.from_assignment(
        _assignment(harness=harness, provider=provider, agent=agent),
    )
    policy = checkpoint_policy_v2(identity)
    assert policy.recovery_mode == mode
    assert policy.workspace_fallback_limit == fallback_limit
    assert policy.completed_result_salvage is True


def test_impossible_or_drifting_harness_identity_is_fenced() -> None:
    with pytest.raises(CheckpointV2ProtocolError, match="impossible"):
        ExecutionIdentityV2.from_assignment(
            _assignment(harness="dsh", provider="bigmodel-coding-plan", agent="dsh-minimal"),
        )
    with pytest.raises(CheckpointV2ProtocolError, match="disagree on Harness"):
        ExecutionIdentityV2.from_assignment(
            _assignment(harness="zcode", provider="bigmodel-coding-plan", agent="dsh-minimal"),
        )


def test_runtime_identity_fingerprint_changes_with_compatibility_digest() -> None:
    left = ExecutionIdentityV2.from_assignment(_assignment(runtime_digest="b" * 64))
    right = ExecutionIdentityV2.from_assignment(_assignment(runtime_digest="c" * 64))
    assert left.fingerprint != right.fingerprint


def test_unknown_checkpoint_core_abi_is_fenced() -> None:
    assignment = _assignment()
    assignment["execution_identity"]["checkpoint_core_abi"] = (
        "dradar-checkpoint-core-v2/999"
    )
    with pytest.raises(CheckpointV2ProtocolError, match="core ABI"):
        ExecutionIdentityV2.from_assignment(assignment)


def test_observe_identity_is_readable_but_has_no_state_machine_authority(
    tmp_path: Path,
) -> None:
    assignment = _assignment()
    assignment["checkpoint_protocol_version"] = 1
    assignment["checkpoint_v2_rollout_mode"] = "observe"
    identity = ExecutionIdentityV2.from_assignment(assignment)
    assert identity.checkpoint_abi == "dradar-checkpoint-v2/codex/1"
    activation = negotiate_checkpoint_activation_v2(
        local_mode="observe", server_mode="observe",
    )
    with pytest.raises(
        CheckpointV2ProtocolError, match="state authority is not enabled",
    ):
        CheckpointV2StateMachine(
            assignment,
            api=FakeApi([]),
            journal=CheckpointV2Journal(tmp_path),
            activation=activation,
        )


def _provisional_observe_assignment() -> dict:
    assignment = _assignment()
    assignment["checkpoint_protocol_version"] = 1
    assignment["checkpoint_v2_rollout_mode"] = "observe"
    assignment["execution_identity"] = {
        **assignment["execution_identity"],
        "identity_state": "PROVISIONAL",
        "runtime_profile": None,
        "model_config_version": None,
        "runtime_compatibility_digest": None,
    }
    return assignment


def _final_identity_for(assignment: dict) -> ExecutionIdentityV2:
    finalized = dict(assignment)
    finalized["execution_identity"] = {
        **assignment["execution_identity"],
        "harness": "codex",
        "provider": "openai",
        "agent_version": "1.2.3",
        "runtime_profile": "runtime-profile-v2",
        "model_config_version": "model-config-v2",
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
        "runtime_compatibility_digest": "d" * 64,
        "identity_state": "FINAL",
    }
    return ExecutionIdentityV2.from_assignment(finalized)


def test_execution_identity_fingerprint_protocol_vector_is_stable() -> None:
    assert _final_identity_for(_provisional_observe_assignment()).fingerprint == (
        "b4d0c1a2c7944fcff50ae2b68f2836a1ce9d9f95ba82bc9e5bd1081dd089ef74"
    )


def _finalize_identity_ack(
    assignment: dict,
    **updates,
) -> dict:
    identity = _final_identity_for(assignment)
    response = {
        "ok": True,
        "assignment_id": assignment["assignment_id"],
        "identity_state": "FINAL",
        "checkpoint_protocol_version": 1,
        "checkpoint_v2_identity_protocol_version": 2,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
        "runtime_compatibility_digest": "d" * 64,
        "identity_fingerprint": identity.fingerprint,
        "assignment_ownership_unchanged": True,
        "paid_execution_authorized": False,
    }
    response.update(updates)
    return response


def _finalize_observe_identity(
    assignment: dict,
    *,
    api: FakeApi,
    journal: CheckpointV2Journal,
    operation_id: str = "identity-finalize-shadow-0001",
) -> FinalizedIdentityReceiptV2:
    return finalize_execution_identity_v2(
        assignment,
        api=api,
        journal=journal,
        harness="codex",
        provider="openai",
        agent_version="1.2.3",
        runtime_profile="runtime-profile-v2",
        model_config_version="model-config-v2",
        checkpoint_abi="dradar-checkpoint-v2/codex/1",
        runtime_compatibility_digest="d" * 64,
        operation_id=operation_id,
    )


def test_observe_identity_finalize_is_free_journaled_and_keeps_v1_ownership(
    tmp_path: Path,
) -> None:
    assignment = _provisional_observe_assignment()
    ack = _finalize_identity_ack(assignment)
    api = FakeApi([ack])
    journal = CheckpointV2Journal(tmp_path)

    receipt = _finalize_observe_identity(
        assignment, api=api, journal=journal,
    )
    assert receipt.identity == _final_identity_for(assignment)
    assert receipt.checkpoint_protocol_version == 1
    assert receipt.checkpoint_v2_identity_protocol_version == 2
    assert receipt.assignment_ownership_unchanged is True
    assert receipt.paid_execution_authorized is False
    assert api.calls[0][0] == "identity/finalize"

    # A process restart reuses the exact acknowledged operation without a
    # second HTTP request or any inference about owner state.
    assert _finalize_observe_identity(
        assignment, api=api, journal=journal,
    ) == receipt
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"paid_execution_authorized": True}, "free-only"),
        ({"checkpoint_protocol_version": 2}, "inconsistent"),
        ({"assignment_ownership_unchanged": False}, "inconsistent"),
        ({"identity_fingerprint": "f" * 64}, "fingerprint"),
    ],
)
def test_identity_finalize_rejects_authority_or_identity_drift(
    tmp_path: Path,
    updates: dict,
    message: str,
) -> None:
    assignment = _provisional_observe_assignment()
    api = FakeApi([_finalize_identity_ack(assignment, **updates)])
    with pytest.raises(CheckpointV2ProtocolError, match=message):
        _finalize_observe_identity(
            assignment,
            api=api,
            journal=CheckpointV2Journal(tmp_path),
            operation_id=f"identity-drift-{message}-0001",
        )


def test_identity_finalize_is_unreachable_when_assignment_snapshotted_off(
    tmp_path: Path,
) -> None:
    assignment = _provisional_observe_assignment()
    assignment["checkpoint_v2_rollout_mode"] = "off"
    api = FakeApi([])
    with pytest.raises(CheckpointV2ProtocolError, match="does not authorize"):
        _finalize_observe_identity(
            assignment, api=api, journal=CheckpointV2Journal(tmp_path),
        )
    assert api.calls == []


def test_typed_fresh_flow_issues_paid_permit_only_after_start(tmp_path: Path) -> None:
    api = FakeApi([
        _checkout_ack(),
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "owner_session_id": "session-0001",
            "owner_epoch": 1,
            "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
            "execution_state": "running",
            "usage_segment_id": "usage-segment-0001",
            "usage_schema": "dradar-checkpoint-usage-segment-v2",
            "paid_execution_authorized": True,
        },
    ])
    checkout_response = _checkout_ack()
    checkout_response.update({
        "owner_session_id": "session-0001",
        "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
    })
    api.outcomes[0] = checkout_response
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    free = machine.checkout(
        session_id="session-0001", operation_id="typed-checkout-0001",
    )
    assert not hasattr(free, "source")
    assert len(api.calls) == 1
    paid = machine.start_paid(free, operation_id="typed-start-0001")
    assert paid.source == "fresh"
    assert paid.usage_segment_id == "usage-segment-0001"
    assert paid.owner_epoch == free.owner_epoch == 1
    assert [command for command, _ in api.calls] == ["checkout", "start"]


def test_usage_evidence_is_content_bound_and_rejects_duplicate_events() -> None:
    permit = PaidExecutionPermit(
        assignment_id="assignment-0001",
        identity_fingerprint="f" * 64,
        session_id="session-0001",
        owner_epoch=3,
        owner_lease_expires_at="2030-01-01T00:00:00+00:00",
        source="fresh",
        usage_segment_id="usage-segment-0001",
    )
    later = UsageEventV2(
        event_id="b" * 64,
        occurred_at="2026-08-23T12:00:02+00:00",
        n_input_tokens=7,
        n_cache_tokens=2,
        n_output_tokens=3,
    )
    earlier = UsageEventV2(
        event_id="a" * 64,
        occurred_at="2026-08-23T12:00:01+00:00",
        n_input_tokens=5,
        n_cache_tokens=1,
        n_output_tokens=2,
    )
    first = UsageSegmentEvidenceV2(
        completeness="complete",
        evidence_kind="trajectory_bundle",
        events=(later, earlier),
    ).protocol_fields(permit)
    second = UsageSegmentEvidenceV2(
        completeness="complete",
        evidence_kind="trajectory_bundle",
        events=(earlier, later),
    ).protocol_fields(permit)
    assert first == second
    assert (
        first["n_input_tokens"],
        first["n_cache_tokens"],
        first["n_output_tokens"],
    ) == (12, 3, 5)
    with pytest.raises(CheckpointV2ProtocolError, match="duplicates"):
        UsageSegmentEvidenceV2(
            completeness="complete",
            evidence_kind="trajectory_bundle",
            events=(earlier, earlier),
        ).protocol_fields(permit)


def test_paid_pause_requires_exact_finalized_usage_receipt(
    tmp_path: Path,
) -> None:
    checkout_response = _checkout_ack()
    checkout_response.update({
        "owner_session_id": "session-0001",
        "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
    })
    api = FakeApi([
        checkout_response,
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "owner_session_id": "session-0001",
            "owner_epoch": 1,
            "owner_lease_expires_at": "2030-01-01T00:30:00+00:00",
            "execution_state": "running",
            "usage_segment_id": "usage-segment-pause",
            "usage_schema": "dradar-checkpoint-usage-segment-v2",
            "paid_execution_authorized": True,
        },
    ])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    free = machine.checkout(
        session_id="session-0001", operation_id="pause-checkout-0001",
    )
    paid = machine.start_paid(free, operation_id="pause-start-0001")
    with pytest.raises(CheckpointV2ProtocolError, match="usage receipt"):
        machine.pause_sealed(
            paid,
            checkpoint=_sealed_generation(1),
            operation_id="pause-without-usage-0001",
        )
    assert [command for command, _ in api.calls] == ["checkout", "start"]


def test_snapshot_storage_contract_fails_before_pause_network(tmp_path: Path) -> None:
    checkout_response = _checkout_ack()
    checkout_response.update({
        "owner_session_id": "session-0001",
        "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
    })
    api = FakeApi([checkout_response])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    free = machine.checkout(
        session_id="session-0001", operation_id="storage-checkout-0001",
    )
    inconsistent = SealedCheckpointV2(
        checkpoint_id="checkpoint-0001",
        checkpoint_lineage_id="lineage-0001",
        snapshot_generation=1,
        capture_id="capture-0001",
        manifest_schema=2,
        manifest_sha256="d" * 64,
        compatibility_fingerprint="c" * 64,
        recovery_capability="WORKSPACE_ONLY",
        native_state_schema=None,
        storage_scope="machine_local",
        writer_machine_fingerprint="f" * 64,
        sync_state="synced",
    )
    with pytest.raises(CheckpointV2ProtocolError, match="inconsistent"):
        machine.pause_sealed(
            free,
            checkpoint=inconsistent,
            operation_id="storage-pause-0001",
        )
    assert len(api.calls) == 1


def test_failure_code_layer_mismatch_fails_before_network(tmp_path: Path) -> None:
    checkout_response = _checkout_ack()
    checkout_response.update({
        "owner_session_id": "session-0001",
        "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
    })
    api = FakeApi([checkout_response])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    free = machine.checkout(
        session_id="session-0001", operation_id="failure-checkout-0001",
    )
    with pytest.raises(CheckpointV2ProtocolError, match="inconsistent"):
        machine.report_failure(
            free,
            failure=CheckpointFailureV2(
                stage="seal",
                code="checkpoint_security_quarantine",
                failure_layer="harness_adapter",
                recoverability="security_quarantine",
                cleanup_result="reaped",
                container_backend="docker",
            ),
            operation_id="failure-report-0001",
        )
    assert len(api.calls) == 1


def test_offline_restore_needs_explicit_commit_before_paid_permit(
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    api = FakeApi([
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "checkpoint_id": "checkpoint-0001",
            "snapshot_generation": 4,
            "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
            "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
            "compatibility_fingerprint": "c" * 64,
            "requester_machine_fingerprint": "f" * 64,
            "reservation_nonce": "1" * 32,
            "owner_session_id": "session-0002",
            "owner_epoch": 8,
            "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
            "execution_state": "resume_reserved",
            "paid_execution_authorized": False,
        },
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "checkpoint_id": "checkpoint-0001",
            "snapshot_generation": 4,
            "restore_receipt_sha256": None,
            "owner_session_id": "session-0002",
            "owner_epoch": 8,
            "owner_lease_expires_at": "2030-01-01T01:00:00+00:00",
            "execution_state": "running",
            "usage_segment_id": "usage-segment-0002",
            "usage_schema": "dradar-checkpoint-usage-segment-v2",
            "paid_execution_authorized": True,
        },
    ])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    offline = machine.reserve_offline_restore(
        session_id="session-0002",
        expected_owner_epoch=7,
        checkpoint_id="checkpoint-0001",
        snapshot_generation=4,
        manifest_sha256=digest,
        requester_machine_fingerprint="f" * 64,
        operation_id="typed-reserve-0001",
    )
    assert len(api.calls) == 1
    assert not hasattr(offline, "source")
    receipt = completed_restore_receipt(
        offline,
        restore_adapter_version="test-adapter-v2",
        restored_manifest_sha256=digest,
    )
    assert (
        receipt.receipt_sha256
        == "ca0c6bda17ea02560588837c2769e1ba7f9f883bca41f59d135140f571562f04"
    )
    api.outcomes[0]["restore_receipt_sha256"] = receipt.receipt_sha256
    paid = machine.commit_paid_resume(
        offline, receipt=receipt, operation_id="typed-commit-0001",
    )
    assert paid.source == "resume"
    assert paid.usage_segment_id == "usage-segment-0002"
    assert len(api.calls) == 2


def test_restore_receipt_rejects_wrong_manifest_before_paid_network(
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    api = FakeApi([
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "checkpoint_id": "checkpoint-0001",
            "snapshot_generation": 4,
            "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
            "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
            "compatibility_fingerprint": "c" * 64,
            "requester_machine_fingerprint": "f" * 64,
            "reservation_nonce": "1" * 32,
            "owner_session_id": "session-0002",
            "owner_epoch": 8,
            "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
            "execution_state": "resume_reserved",
            "paid_execution_authorized": False,
        },
    ])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    offline = machine.reserve_offline_restore(
        session_id="session-0002",
        expected_owner_epoch=7,
        checkpoint_id="checkpoint-0001",
        snapshot_generation=4,
        manifest_sha256=digest,
        requester_machine_fingerprint="f" * 64,
        operation_id="receipt-reserve-0001",
    )
    with pytest.raises(CheckpointV2ProtocolError, match="different checkpoint"):
        completed_restore_receipt(
            offline,
            restore_adapter_version="test-adapter-v2",
            restored_manifest_sha256="e" * 64,
        )
    assert len(api.calls) == 1


def test_completed_result_is_bound_before_checkpoint_finalize(
    tmp_path: Path,
) -> None:
    checkout_response = _checkout_ack()
    checkout_response.update({
        "owner_session_id": "session-0001",
        "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
    })
    api = FakeApi([
        checkout_response,
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "owner_session_id": "session-0001",
            "owner_epoch": 1,
            "owner_lease_expires_at": "2030-01-01T00:30:00+00:00",
            "execution_state": "running",
            "usage_segment_id": "usage-segment-result",
            "usage_schema": "dradar-checkpoint-usage-segment-v2",
            "paid_execution_authorized": True,
        },
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "usage_segment_id": "usage-segment-result",
            "usage_schema": "dradar-checkpoint-usage-segment-v2",
            "completeness": "complete",
            "evidence_sha256": None,
            "owner_session_id": "session-0001",
            "owner_epoch": 1,
            "owner_lease_expires_at": "2030-01-01T00:30:00+00:00",
            "execution_state": "running",
            "usage_ledger": _usage_ledger_ack(),
            "assignment_unchanged": True,
            "paid_execution_authorized": False,
        },
        {
            "ok": True,
            "assignment_id": "assignment-0001",
            "upload_intent_id": "e" * 64,
            "owner_session_id": "session-0001",
            "owner_epoch": 1,
            "owner_lease_expires_at": "2030-01-01T00:30:00+00:00",
            "execution_state": "result_ready",
            "usage_ledger": _usage_ledger_ack(),
            "paid_execution_authorized": False,
        },
    ])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    free = machine.checkout(
        session_id="session-0001", operation_id="result-checkout-0001",
    )
    paid = machine.start_paid(free, operation_id="result-start-0001")
    evidence = UsageSegmentEvidenceV2(
        completeness="complete",
        evidence_kind="trajectory_bundle",
        events=(UsageEventV2(
            event_id="a" * 64,
            occurred_at="2026-08-23T12:00:00+00:00",
            n_input_tokens=10,
            n_cache_tokens=2,
            n_output_tokens=3,
        ),),
    )
    api.outcomes[0]["evidence_sha256"] = evidence.protocol_fields(
        paid,
    )["evidence_sha256"]
    usage_receipt = machine.finalize_usage_segment(
        paid,
        evidence=evidence,
        operation_id="result-usage-finalize-0001",
    )
    completed = machine.declare_result_ready(
        paid,
        usage_receipt=usage_receipt,
        upload_intent_id="e" * 64,
        operation_id="result-ready-0001",
    )
    assert completed.upload_intent_id == "e" * 64
    assert completed.owner_epoch == paid.owner_epoch
    assert [command for command, _ in api.calls] == [
        "checkout", "start", "usage-finalize", "result-ready",
    ]


def test_retention_ack_is_exact_typed_and_journal_replayable(
    tmp_path: Path,
) -> None:
    first = _sealed_generation(1)
    second = _sealed_generation(2)
    operation_id = "retention-operation-0001"
    response = {
        "ok": True,
        "assignment_id": "assignment-0001",
        "operation_id": operation_id,
        "owner_epoch_observed": 7,
        "current_owner_epoch": 8,
        "delete_generations": [{
            "checkpoint_id": first.checkpoint_id,
            "snapshot_generation": first.snapshot_generation,
            "manifest_sha256": first.manifest_sha256,
            "already_released": False,
        }],
        "retain_generations": [{
            "checkpoint_id": second.checkpoint_id,
            "snapshot_generation": second.snapshot_generation,
            "manifest_sha256": second.manifest_sha256,
            "reason": "recovery_fallback_protected",
        }],
        "result_evidence_release": False,
        "submission_id": None,
        "assignment_unchanged": True,
        "paid_execution_authorized": False,
    }
    api = FakeApi([response])
    journal = CheckpointV2Journal(tmp_path)
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=journal, activation=_on_activation(),
    )

    acknowledged = machine.acknowledge_retention(
        owner_epoch_observed=7,
        generations=(first, second),
        operation_id=operation_id,
    )
    replayed = machine.acknowledge_retention(
        owner_epoch_observed=7,
        generations=(first, second),
        operation_id=operation_id,
    )

    assert replayed == acknowledged
    assert tuple(item.key for item in acknowledged.delete_generations) == ((
        first.checkpoint_id, first.snapshot_generation, first.manifest_sha256,
    ),)
    assert tuple(item.key for item in acknowledged.retain_generations) == ((
        second.checkpoint_id, second.snapshot_generation, second.manifest_sha256,
    ),)
    assert acknowledged.current_owner_epoch == 8
    assert acknowledged.result_evidence_release is False
    assert len(api.calls) == 1
    entry = journal.load("assignment-0001", operation_id)
    assert entry.state == "ACKNOWLEDGED"
    assert entry.response == response


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body, item: body.update(delete_generations=[], retain_generations=[]),
        lambda body, item: body.update(
            delete_generations=[item], retain_generations=[item],
        ),
        lambda body, item: body["delete_generations"][0].update(
            manifest_sha256="f" * 64,
        ),
        lambda body, item: body.update(operation_id="retention-operation-wrong"),
        lambda body, item: body.update(result_evidence_release=True),
    ],
)
def test_retention_ack_rejects_drifted_server_decision(
    tmp_path: Path,
    mutate,
) -> None:
    generation = _sealed_generation(1)
    item = {
        "checkpoint_id": generation.checkpoint_id,
        "snapshot_generation": generation.snapshot_generation,
        "manifest_sha256": generation.manifest_sha256,
    }
    response = {
        "ok": True,
        "assignment_id": "assignment-0001",
        "operation_id": "retention-operation-0002",
        "owner_epoch_observed": 3,
        "current_owner_epoch": 3,
        "delete_generations": [dict(item)],
        "retain_generations": [],
        "result_evidence_release": False,
        "submission_id": None,
        "assignment_unchanged": True,
        "paid_execution_authorized": False,
    }
    mutate(response, dict(item))
    machine = CheckpointV2StateMachine(
        _assignment(), api=FakeApi([response]),
        journal=CheckpointV2Journal(tmp_path), activation=_on_activation(),
    )
    with pytest.raises(CheckpointV2ProtocolError, match="evidence identity"):
        machine.acknowledge_retention(
            owner_epoch_observed=3,
            generations=(generation,),
            operation_id="retention-operation-0002",
        )


def test_retention_result_evidence_requires_exact_durable_submission_ack(
    tmp_path: Path,
) -> None:
    intent_id = "e" * 64
    operation_id = "retention-operation-result"
    response = {
        "ok": True,
        "assignment_id": "assignment-0001",
        "operation_id": operation_id,
        "owner_epoch_observed": 4,
        "current_owner_epoch": 6,
        "delete_generations": [],
        "retain_generations": [],
        "result_evidence_release": True,
        "submission_id": "submission-0001",
        "assignment_unchanged": True,
        "paid_execution_authorized": False,
    }
    machine = CheckpointV2StateMachine(
        _assignment(), api=FakeApi([response]),
        journal=CheckpointV2Journal(tmp_path), activation=_on_activation(),
    )
    acknowledged = machine.acknowledge_retention(
        owner_epoch_observed=4,
        generations=(),
        upload_intent_id=intent_id,
        operation_id=operation_id,
    )
    assert acknowledged.result_evidence_release is True
    assert acknowledged.upload_intent_id == intent_id
    assert acknowledged.submission_id == "submission-0001"


def test_permit_from_another_identity_is_rejected_before_network(
    tmp_path: Path,
) -> None:
    checkout_response = _checkout_ack()
    checkout_response.update({
        "owner_session_id": "session-0001",
        "owner_lease_expires_at": "2030-01-01T00:00:00+00:00",
    })
    api = FakeApi([checkout_response])
    machine = CheckpointV2StateMachine(
        _assignment(), api=api, journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    free = machine.checkout(
        session_id="session-0001", operation_id="permit-checkout-0001",
    )
    foreign = replace(free, identity_fingerprint="f" * 64)
    with pytest.raises(CheckpointV2ProtocolError, match="different execution identity"):
        machine.start_paid(foreign, operation_id="permit-start-0001")
    assert len(api.calls) == 1


def test_unsupported_grok_writer_is_fenced_before_checkout(tmp_path: Path) -> None:
    api = FakeApi([])
    machine = CheckpointV2StateMachine(
        _assignment(
            harness="grok-build",
            provider="xai-subscription",
            agent="grok-build",
        ),
        api=api,
        journal=CheckpointV2Journal(tmp_path),
        activation=_on_activation(),
    )
    with pytest.raises(CheckpointV2ProtocolError, match="no checkpoint v2 writer"):
        machine.checkout(session_id="session-0001")
    assert api.calls == []
