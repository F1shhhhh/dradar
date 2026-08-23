from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dradar.checkpoint_activation_v2 import negotiate_checkpoint_activation_v2
from dradar.api_client import ApiError
from dradar import pending
from dradar.checkpoint_owner_runtime_v2 import (
    AuthoritativeCheckpointRunV2,
    CheckpointV2OwnerLost,
    recover_completed_result_ready_v2,
    recover_completed_result_usage_v2,
    reconcile_orphaned_paid_gate_v2,
)
from dradar.checkpoint_activation_v2 import CheckpointV2ProtocolError
from dradar.checkpoint_v2 import ExecutionIdentityV2
from dradar.checkpoint_v2 import (
    CheckpointV2OrdinaryFallback,
    CompletedResultPermit,
    FinalizedUsageSegmentReceiptV2,
    PaidExecutionPermit,
)
from dradar.checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from dradar.checkpoint_live_v2 import _runtime_config_v2
from dradar.checkpoint_runtime_v2 import PublishedCheckpointV2
from dradar import runloop
from dradar.runner import TrialArtifacts
from dradar.runner import CheckpointV2PaidGateFaultedError


class _Telemetry:
    session_id = "session-owner-0001"

    def __init__(self) -> None:
        self.runtime = None
        self.phases = []
        self.flushes = 0
        self.observations = []

    def configure_checkpoint_runtime(self, **facts) -> None:
        self.runtime = facts

    def configure_checkpoint_observation_reporting(self, _home: Path) -> None:
        return None

    def set_phase(self, phase, assignment_id=None, resume_generation=None) -> None:
        self.phases.append((phase, assignment_id, resume_generation))

    def flush(self) -> None:
        self.flushes += 1

    def record_checkpoint_observation(self, payload) -> bool:
        self.observations.append(payload)
        return True


def _assignment() -> dict:
    return {
        "assignment_id": "assignment-owner-0001",
        "benchmark_id": "deep-swe",
        "task_id": "t1",
        "task_content_hash": "a" * 64,
        "agent": "codex",
        "provider": "openai",
        "model": "gpt-test",
        "effort": "high",
        "agent_version": "0.150.0",
        "resume_generation": 0,
        "checkpoint_protocol_version": 2,
        "checkpoint_v2_identity_protocol_version": 2,
        "checkpoint_v2_rollout_mode": "canary",
        "checkpoint_v2_controlled_account": True,
        "owner_epoch": 0,
        "execution_identity": {
            "benchmark_id": "deep-swe",
            "task_content_sha256": "a" * 64,
            "harness": "codex",
            "provider": "openai",
            "model": "gpt-test",
            "effort": "high",
            "agent_version": "0.150.0",
            "runtime_profile": None,
            "model_config_version": None,
            "checkpoint_core_abi": None,
            "checkpoint_abi": None,
            "runtime_compatibility_digest": None,
            "identity_state": "PROVISIONAL",
            "identity_source": "claim_snapshot",
        },
    }


class _Api:
    def __init__(self, assignment: dict) -> None:
        self.assignment = assignment
        self.calls = []

    def checkpoint_v2_command(self, command: str, payload: dict) -> dict:
        self.calls.append((command, dict(payload)))
        common = {
            "ok": True,
            "assignment_id": self.assignment["assignment_id"],
        }
        if command == "identity/finalize":
            finalized = {
                **self.assignment,
                "execution_identity": {
                    **self.assignment["execution_identity"],
                    "harness": payload["harness"],
                    "provider": payload["provider"],
                    "agent_version": payload["agent_version"],
                    "runtime_profile": payload["runtime_profile"],
                    "model_config_version": payload["model_config_version"],
                    "checkpoint_core_abi": payload["checkpoint_core_abi"],
                    "checkpoint_abi": payload["checkpoint_abi"],
                    "runtime_compatibility_digest": (
                        payload["runtime_compatibility_digest"]
                    ),
                    "identity_state": "FINAL",
                },
            }
            identity = ExecutionIdentityV2.from_assignment(finalized)
            return {
                **common,
                "identity_state": "FINAL",
                "checkpoint_protocol_version": 2,
                "checkpoint_v2_identity_protocol_version": 2,
                "checkpoint_core_abi": payload["checkpoint_core_abi"],
                "checkpoint_abi": payload["checkpoint_abi"],
                "runtime_compatibility_digest": (
                    payload["runtime_compatibility_digest"]
                ),
                "identity_fingerprint": identity.fingerprint,
                "assignment_ownership_unchanged": True,
                "paid_execution_authorized": False,
            }
        if command == "checkout":
            return {
                **common,
                "checkpoint_v2_authoritative_activated": True,
                "checkpoint_protocol_version": 2,
                "certification_id": "certification-owner-0001",
                "certification_digest": "c" * 64,
                "owner_session_id": payload["session_id"],
                "owner_epoch": 1,
                "owner_lease_expires_at": "2030-01-01T00:10:00+00:00",
                "execution_state": "preparing",
                "paid_execution_authorized": False,
            }
        if command == "start":
            return {
                **common,
                "certification_id": "certification-owner-0001",
                "certification_digest": "c" * 64,
                "owner_session_id": payload["session_id"],
                "owner_epoch": 1,
                "owner_lease_expires_at": "2030-01-01T00:20:00+00:00",
                "execution_state": "running",
                "usage_segment_id": "usage-segment-owner-0001",
                "usage_schema": "dradar-checkpoint-usage-segment-v2",
                "paid_execution_authorized": True,
            }
        if command == "paid-gate-reconcile":
            return {
                **common,
                "outcome": "faulted",
                "attempted_operation_id": payload["attempted_operation_id"],
                "attempted_command": payload["attempted_command"],
                "attempted_operation_matched": True,
                "paid_command_committed": True,
                "checkpoint_protocol_version": 2,
                "owner_epoch": payload["expected_owner_epoch"] + 1,
                "resume_generation": 0,
                "execution_state": "faulted",
                "assignment_unchanged": False,
                "assignment_invalidated": False,
                "assignment_restarted_fresh": False,
                "checkpoint_evidence_retained": True,
                "usage_segment_finalized_unavailable": (
                    "usage-segment-owner-0001"
                ),
                "circuit": {"state": "OPEN"},
                "paid_execution_authorized": False,
            }
        if command == "renew":
            return {
                **common,
                "owner_session_id": payload["session_id"],
                "owner_epoch": 1,
                "owner_lease_expires_at": "2030-01-01T00:30:00+00:00",
                "execution_state": "running",
                "paid_execution_authorized": False,
            }
        ledger = {
            "schema": "dradar-checkpoint-usage-ledger-v2",
            "usage_segment_count": 1,
            "finalized_segment_count": 1,
            "open_segment_count": 0,
            "complete": True,
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 3,
            "ledger_sha256": "9" * 64,
        }
        if command == "usage-finalize":
            segment_usage = {
                "ledger_scope": payload["ledger_scope"],
                "observed_event_count": payload["event_count"],
                "novel_event_count": payload["event_count"],
                "duplicate_event_count": 0,
                "n_input_tokens": payload["n_input_tokens"],
                "n_cache_tokens": payload["n_cache_tokens"],
                "n_output_tokens": payload["n_output_tokens"],
            }
            return {
                **common,
                "owner_session_id": payload["session_id"],
                "owner_epoch": 1,
                "owner_lease_expires_at": "2030-01-01T00:30:00+00:00",
                "execution_state": "running",
                "usage_segment_id": payload["usage_segment_id"],
                "usage_schema": payload["usage_schema"],
                "completeness": payload["completeness"],
                "evidence_sha256": payload["evidence_sha256"],
                "segment_usage": segment_usage,
                "usage_ledger": ledger,
                "assignment_unchanged": True,
                "paid_execution_authorized": False,
            }
        if command == "result-ready":
            return {
                **common,
                "owner_session_id": payload["session_id"],
                "owner_epoch": 1,
                "owner_lease_expires_at": "2030-01-01T00:40:00+00:00",
                "execution_state": "result_ready",
                "upload_intent_id": payload["upload_intent_id"],
                "usage_ledger": ledger,
                "paid_execution_authorized": False,
            }
        if command == "retention":
            return {
                **common,
                "operation_id": payload["operation_id"],
                "owner_epoch_observed": payload["owner_epoch_observed"],
                "current_owner_epoch": 2,
                "delete_generations": [],
                "retain_generations": [],
                "result_evidence_release": True,
                "upload_intent_id": payload["upload_intent_id"],
                "submission_id": "submission-owner-0001",
                "assignment_unchanged": True,
                "paid_execution_authorized": False,
            }
        if command == "failure":
            return {
                **common,
                "owner_epoch": 2,
                "execution_state": "faulted",
                "checkpoint_id": None,
                "owner_release_pending": False,
                "assignment_invalidated": False,
                "paid_execution_authorized": False,
            }
        raise AssertionError(command)


class _ResumeApi(_Api):
    def checkpoint_v2_command(self, command: str, payload: dict) -> dict:
        if command not in {
            "resume-reserve", "resume-commit", "resume-abort",
            "fresh-fallback",
        }:
            return super().checkpoint_v2_command(command, payload)
        self.calls.append((command, dict(payload)))
        common = {
            "ok": True,
            "assignment_id": self.assignment["assignment_id"],
        }
        if command == "fresh-fallback":
            return {
                **common,
                "checkpoint_v2_authoritative_activated": False,
                "checkpoint_protocol_version": 1,
                "fallback_to_ordinary": True,
                "fallback_observation_mode": "observe",
                "reason": payload["reason"],
                "assignment_restarted_fresh": True,
                "assignment_unchanged": False,
                "owner_epoch": payload["expected_owner_epoch"] + 1,
                "resume_generation": 1,
                "execution_state": "waiting",
                "checkpoint_evidence_retained": True,
                "paid_execution_authorized": False,
            }
        common.update({
            "checkpoint_id": payload["checkpoint_id"],
            "snapshot_generation": payload["snapshot_generation"],
        })
        if command == "resume-reserve":
            return {
                **common,
                "certification_id": "certification-owner-0001",
                "certification_digest": "c" * 64,
                "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
                "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
                "compatibility_fingerprint": self.assignment[
                    "checkpoint_v2_selected_generation"
                ]["compatibility_fingerprint"],
                "requester_machine_fingerprint": (
                    payload["requester_machine_fingerprint"]
                ),
                "reservation_nonce": "3" * 32,
                "owner_session_id": payload["session_id"],
                "owner_epoch": 3,
                "owner_lease_expires_at": "2030-01-01T00:10:00+00:00",
                "execution_state": "resume_reserved",
                "paid_execution_authorized": False,
            }
        if command == "resume-commit":
            return {
                **common,
                "certification_id": "certification-owner-0001",
                "certification_digest": "c" * 64,
                "restore_receipt_sha256": payload["restore_receipt_sha256"],
                "owner_session_id": payload["session_id"],
                "owner_epoch": 3,
                "owner_lease_expires_at": "2030-01-01T00:20:00+00:00",
                "execution_state": "running",
                "usage_segment_id": "usage-segment-resume-0001",
                "usage_schema": "dradar-checkpoint-usage-segment-v2",
                "paid_execution_authorized": True,
            }
        return {
            **common,
            "owner_epoch": 4,
            "execution_state": "paused",
            "paid_execution_authorized": False,
        }


@pytest.fixture
def owner(monkeypatch, tmp_path: Path) -> AuthoritativeCheckpointRunV2:
    assignment = _assignment()
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.docker_container_backend_v2",
        lambda: "docker",
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.checkpoint_machine_fingerprint",
        lambda _home: "b" * 64,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2._runtime_digest_v2",
        lambda **_facts: "c" * 64,
    )
    return AuthoritativeCheckpointRunV2(
        assignment=assignment,
        effective_assignment=assignment,
        activation=negotiate_checkpoint_activation_v2(
            local_mode="canary",
            server_mode="canary",
            controlled_account=True,
        ),
        api=_Api(assignment),
        telemetry=_Telemetry(),
        home=tmp_path,
        job_root=tmp_path / "work" / "jobs" / "aassignment-owner-0001",
        renew_interval_sec=600,
        initial_capture_delay_sec=86_400,
        capture_interval_sec=86_400,
    )


def _authorize_fresh_at_gate(
    owner: AuthoritativeCheckpointRunV2,
) -> PaidExecutionPermit:
    contract = json.loads((owner.gate_dir / "contract.json").read_text())
    owner._write_once(owner.gate_dir / "request.json", {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": owner.assignment["assignment_id"],
        "gate_nonce": contract["gate_nonce"],
        "action": "fresh",
        "restore_receipt_sha256": None,
    })

    class _LivePier:
        @staticmethod
        def poll():
            return None

    return owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)


def test_authoritative_owner_fresh_result_lifecycle_is_explicit(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    free = owner.prepare()
    assert free.owner_epoch == 1
    assert not hasattr(free, "usage_segment_id")

    paid = _authorize_fresh_at_gate(owner)
    assert paid.usage_segment_id == "usage-segment-owner-0001"
    owner.mainline_exited()
    receipt = owner.finalize_usage(
        n_input_tokens=10,
        n_cache_tokens=2,
        n_output_tokens=3,
        token_usage_events=[{
            "occurred_at": "2026-08-23T12:00:00+00:00",
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 3,
        }],
        request_usage_complete=True,
        request_usage_observed=True,
        occurred_at="2026-08-23T12:00:00+00:00",
    )
    completed = owner.declare_result_ready(upload_intent_id="e" * 64)
    submission_id = owner.release_after_submission()

    assert receipt.usage_ledger_complete is True
    assert completed.upload_intent_id == "e" * 64
    assert submission_id == "submission-owner-0001"
    assert [command for command, _payload in owner.api.calls] == [
        "identity/finalize", "checkout", "start", "usage-finalize",
        "result-ready", "retention",
    ]
    usage_payload = owner.api.calls[3][1]
    assert usage_payload["ledger_scope"] == "assignment_cumulative"
    assert usage_payload["evidence_kind"] == "trajectory_bundle"
    assert usage_payload["event_count"] == 1
    assert usage_payload["events"][0]["n_input_tokens"] == 10


def test_owner_authorizes_only_the_exact_pre_provider_gate_request(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    contract = json.loads((owner.gate_dir / "contract.json").read_text())
    request = {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": owner.assignment["assignment_id"],
        "gate_nonce": contract["gate_nonce"],
        "action": "fresh",
        "restore_receipt_sha256": None,
    }
    owner._write_once(owner.gate_dir / "request.json", request)

    class _LivePier:
        @staticmethod
        def poll():
            return None

    paid = owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)
    grant_path = owner.gate_dir / "grant.json"
    grant = json.loads(grant_path.read_text())
    intent_path = owner.gate_dir / "intent.json"
    intent = json.loads(intent_path.read_text())
    assert paid.usage_segment_id == "usage-segment-owner-0001"
    assert grant["request_sha256"] == hashlib.sha256(
        owner._canonical_bytes(request),
    ).hexdigest()
    assert grant["paid_execution_authorized"] is True
    assert stat.S_IMODE(intent_path.stat().st_mode) == 0o600
    assert intent["attempted_operation_id"] == owner._paid_gate_operation_id
    assert intent["reconcile_operation_id"] == (
        owner._paid_gate_reconcile_operation_id
    )
    assert intent["session_id"] == "session-owner-0001"
    assert intent["expected_owner_epoch"] == 1
    assert not any(
        fragment in json.dumps(intent).lower()
        for fragment in ("token", "secret", "credential", "authorization")
    )
    assert [command for command, _payload in owner.api.calls][-1] == "start"
    owner.mainline_exited()


def test_paid_gate_intent_fsync_failure_falls_back_before_start(
    monkeypatch,
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    free = owner.prepare()
    contract = json.loads((owner.gate_dir / "contract.json").read_text())
    request = {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": owner.assignment["assignment_id"],
        "gate_nonce": contract["gate_nonce"],
        "action": "fresh",
        "restore_receipt_sha256": None,
    }
    owner._write_once(owner.gate_dir / "request.json", request)
    original_command = owner.api.checkpoint_v2_command

    def command(name: str, payload: dict) -> dict:
        if name != "fresh-fallback":
            return original_command(name, payload)
        owner.api.calls.append((name, dict(payload)))
        return {
            "ok": True,
            "assignment_id": owner.assignment["assignment_id"],
            "checkpoint_v2_authoritative_activated": False,
            "checkpoint_protocol_version": 1,
            "fallback_to_ordinary": True,
            "fallback_observation_mode": "observe",
            "reason": "paid_gate_intent_failed",
            "assignment_restarted_fresh": True,
            "assignment_unchanged": False,
            "owner_epoch": free.owner_epoch + 1,
            "resume_generation": 1,
            "execution_state": "waiting",
            "checkpoint_evidence_retained": True,
            "paid_execution_authorized": False,
        }

    owner.api.checkpoint_v2_command = command
    original_write = owner._write_once

    def fail_intent(path: Path, value: dict) -> None:
        if path.name == "intent.json":
            raise OSError("simulated intent fsync failure")
        original_write(path, value)

    monkeypatch.setattr(owner, "_write_once", fail_intent)

    class _LivePier:
        @staticmethod
        def poll():
            return None

    with pytest.raises(CheckpointV2OrdinaryFallback) as raised:
        owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)
    assert raised.value.reason == "paid_gate_intent_failed"
    assert owner.permit is None
    assert owner.ordinary_fallback is True
    assert [name for name, _payload in owner.api.calls][-1] == "fresh-fallback"
    assert not any(name == "start" for name, _payload in owner.api.calls)


def test_paid_start_with_unwritable_grant_is_faulted_not_retried_fresh(
    monkeypatch,
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    contract = json.loads((owner.gate_dir / "contract.json").read_text())
    request = {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": owner.assignment["assignment_id"],
        "gate_nonce": contract["gate_nonce"],
        "action": "fresh",
        "restore_receipt_sha256": None,
    }
    owner._write_once(owner.gate_dir / "request.json", request)
    original_write = owner._write_once

    def fail_grant(path: Path, value: dict) -> None:
        if path.name == "grant.json":
            raise OSError("simulated grant fsync failure")
        original_write(path, value)

    monkeypatch.setattr(owner, "_write_once", fail_grant)

    class _LivePier:
        @staticmethod
        def poll():
            return None

    with pytest.raises(OSError, match="grant fsync"):
        owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)
    assert isinstance(owner.permit, PaidExecutionPermit)
    assert owner.paid_gate_reconcile_required is True
    assert (owner.gate_dir / "denial.json").is_file()

    response = owner.reconcile_ambiguous_paid_gate()
    assert response["execution_state"] == "faulted"
    assert response["assignment_invalidated"] is False
    assert owner.permit is None
    assert owner.paid_gate_reconcile_required is False
    assert [command for command, _payload in owner.api.calls] == [
        "identity/finalize", "checkout", "start", "paid-gate-reconcile",
    ]
    reconcile_payload = owner.api.calls[-1][1]
    assert reconcile_payload["attempted_operation_id"] == (
        owner._paid_gate_operation_id
    )
    assert reconcile_payload["attempted_command"] == "start"
    owner.mainline_exited()


def test_ambiguous_start_without_server_commit_reconciles_to_fresh_v1(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    contract = json.loads((owner.gate_dir / "contract.json").read_text())
    request = {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": owner.assignment["assignment_id"],
        "gate_nonce": contract["gate_nonce"],
        "action": "fresh",
        "restore_receipt_sha256": None,
    }
    owner._write_once(owner.gate_dir / "request.json", request)
    original = owner.api.checkpoint_v2_command

    def command(name: str, payload: dict) -> dict:
        if name == "start":
            owner.api.calls.append((name, dict(payload)))
            raise ApiError("simulated response loss")
        if name == "paid-gate-reconcile":
            owner.api.calls.append((name, dict(payload)))
            return {
                "ok": True,
                "assignment_id": owner.assignment["assignment_id"],
                "outcome": "fresh_fallback",
                "attempted_operation_id": payload["attempted_operation_id"],
                "attempted_command": "start",
                "attempted_operation_matched": False,
                "paid_command_committed": False,
                "checkpoint_protocol_version": 1,
                "owner_epoch": payload["expected_owner_epoch"] + 1,
                "resume_generation": 1,
                "execution_state": "waiting",
                "assignment_unchanged": False,
                "assignment_invalidated": False,
                "assignment_restarted_fresh": True,
                "checkpoint_evidence_retained": True,
                "paid_execution_authorized": False,
            }
        return original(name, payload)

    owner.api.checkpoint_v2_command = command

    class _LivePier:
        @staticmethod
        def poll():
            return None

    with pytest.raises(ApiError, match="response loss"):
        owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)
    response = owner.reconcile_ambiguous_paid_gate()
    assert response["outcome"] == "fresh_fallback"
    assert owner.permit is None
    assert owner.ordinary_fallback is True
    assert [name for name, _payload in owner.api.calls][-2:] == [
        "start", "paid-gate-reconcile",
    ]


def _orphan_assignment(
    owner: AuthoritativeCheckpointRunV2,
    *,
    execution_state: str = "preparing",
) -> dict:
    assert owner._finalized_assignment is not None
    return {
        **owner._finalized_assignment,
        "checkpoint_protocol_version": 2,
        "checkpoint_v2_rollout_mode": "canary",
        "checkpoint_v2_controlled_account": True,
        "owner_session_id": "session-owner-0001",
        "owner_epoch": 1,
        "resume_generation": 0,
        "execution_state": execution_state,
    }


def _install_orphan_reconcile_response(
    api: _Api,
    *,
    outcome: str,
) -> None:
    original = api.checkpoint_v2_command

    def command(name: str, payload: dict) -> dict:
        if name != "paid-gate-reconcile":
            return original(name, payload)
        api.calls.append((name, dict(payload)))
        common = {
            "ok": True,
            "assignment_id": api.assignment["assignment_id"],
            "outcome": outcome,
            "attempted_operation_id": payload["attempted_operation_id"],
            "attempted_command": payload["attempted_command"],
            "checkpoint_evidence_retained": True,
            "assignment_invalidated": False,
            "paid_execution_authorized": False,
            "resume_generation": 1 if outcome == "fresh_fallback" else 0,
        }
        if outcome == "fresh_fallback":
            return {
                **common,
                "attempted_operation_matched": False,
                "paid_command_committed": False,
                "checkpoint_protocol_version": 1,
                "owner_epoch": 2,
                "execution_state": "waiting",
                "assignment_unchanged": False,
                "assignment_restarted_fresh": True,
            }
        if outcome == "completed_result_preserved":
            return {
                **common,
                "attempted_operation_matched": True,
                "paid_command_committed": True,
                "checkpoint_protocol_version": 2,
                "owner_epoch": payload["expected_owner_epoch"],
                "execution_state": "result_ready",
                "assignment_unchanged": True,
                "assignment_restarted_fresh": False,
            }
        return {
            **common,
            "attempted_operation_matched": True,
            "paid_command_committed": True,
            "checkpoint_protocol_version": 2,
            "owner_epoch": 2,
            "execution_state": "faulted",
            "assignment_unchanged": False,
            "assignment_restarted_fresh": False,
            "usage_segment_finalized_unavailable": "usage-segment-owner-0001",
            "circuit": {"state": "OPEN"},
        }

    api.checkpoint_v2_command = command


def test_cross_process_orphan_before_launch_falls_back_without_paid_replay(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    _install_orphan_reconcile_response(owner.api, outcome="fresh_fallback")
    cleaned = []

    response = reconcile_orphaned_paid_gate_v2(
        _orphan_assignment(owner),
        activation=owner.activation,
        api=owner.api,
        home=owner.home,
        cleanup_containers=cleaned.append,
    )

    assert response is not None
    assert response["outcome"] == "fresh_fallback"
    assert cleaned == []
    assert owner.api.calls[-1][0] == "paid-gate-reconcile"
    assert owner.api.calls[-1][1]["attempted_operation_id"] is None
    assert not any(name == "start" for name, _payload in owner.api.calls)


def test_cross_process_orphan_blocks_the_unattributed_popen_gap(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    owner.record_pier_launch_intent(["pier", str(owner.gate_dir)])
    _install_orphan_reconcile_response(owner.api, outcome="fresh_fallback")

    with pytest.raises(
        Exception,
        match="launch outcome lacks process evidence",
    ):
        reconcile_orphaned_paid_gate_v2(
            _orphan_assignment(owner),
            activation=owner.activation,
            api=owner.api,
            home=owner.home,
            cleanup_containers=lambda _path: None,
        )
    assert owner.api.calls[-1][0] != "paid-gate-reconcile"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_cross_process_orphan_reaps_exact_pier_before_fresh_fallback(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(120)",
        str(owner.gate_dir),
    ]
    owner.record_pier_launch_intent(command)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        owner.register_pier_process(process, command)
        _install_orphan_reconcile_response(
            owner.api,
            outcome="fresh_fallback",
        )
        cleaned = []
        response = reconcile_orphaned_paid_gate_v2(
            _orphan_assignment(owner),
            activation=owner.activation,
            api=owner.api,
            home=owner.home,
            cleanup_containers=cleaned.append,
        )
        assert response is not None
        assert response["outcome"] == "fresh_fallback"
        assert process.poll() is not None
        assert cleaned == [owner.job_root.resolve()]
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_cross_process_orphan_preserves_completed_result_and_never_restarts(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(120)",
        str(owner.gate_dir),
    ]
    owner.record_pier_launch_intent(command)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        owner.register_pier_process(process, command)
        owner._paid_gate_operation_id = "paid-start-result-ready-0001"
        owner._persist_paid_gate_intent(
            request_sha256="a" * 64,
            attempted_operation_id=owner._paid_gate_operation_id,
        )
        os.killpg(process.pid, 15)
        process.wait(timeout=5)
        owner.mainline_exited()
        _install_orphan_reconcile_response(
            owner.api,
            outcome="completed_result_preserved",
        )
        cleaned = []
        response = reconcile_orphaned_paid_gate_v2(
            _orphan_assignment(owner, execution_state="result_ready"),
            activation=owner.activation,
            api=owner.api,
            home=owner.home,
            cleanup_containers=cleaned.append,
        )
        assert response is not None
        assert response["outcome"] == "completed_result_preserved"
        assert response["assignment_unchanged"] is True
        assert cleaned == [owner.job_root.resolve()]
        assert [name for name, _payload in owner.api.calls][-1] == (
            "paid-gate-reconcile"
        )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


@pytest.mark.parametrize(
    "disable_code",
    [
        "checkpoint_v2_certification_revoked",
        "checkpoint_v2_kill_switch_active",
    ],
)
def test_stable_prepaid_disable_restarts_same_lease_as_ordinary_fresh(
    monkeypatch,
    owner: AuthoritativeCheckpointRunV2,
    disable_code: str,
) -> None:
    free = owner.prepare()
    contract = json.loads((owner.gate_dir / "contract.json").read_text())
    request = {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": owner.assignment["assignment_id"],
        "gate_nonce": contract["gate_nonce"],
        "action": "fresh",
        "restore_receipt_sha256": None,
    }
    owner._write_once(owner.gate_dir / "request.json", request)
    original = owner.api.checkpoint_v2_command

    def disabled(command: str, payload: dict) -> dict:
        if command == "start":
            owner.api.calls.append((command, dict(payload)))
            raise ApiError(
                "authoritative checkpoint disabled",
                status_code=409,
                code=disable_code,
            )
        if command == "fresh-fallback":
            owner.api.calls.append((command, dict(payload)))
            return {
                "ok": True,
                "assignment_id": owner.assignment["assignment_id"],
                "checkpoint_v2_authoritative_activated": False,
                "checkpoint_protocol_version": 1,
                "fallback_to_ordinary": True,
                "fallback_observation_mode": "observe",
                "reason": "operator_disable",
                "assignment_restarted_fresh": True,
                "assignment_unchanged": False,
                "owner_epoch": free.owner_epoch + 1,
                "resume_generation": 1,
                "execution_state": "waiting",
                "checkpoint_evidence_retained": True,
                "paid_execution_authorized": False,
            }
        return original(command, payload)

    monkeypatch.setattr(owner.api, "checkpoint_v2_command", disabled)

    class _LivePier:
        @staticmethod
        def poll():
            return None

    with pytest.raises(CheckpointV2OrdinaryFallback) as raised:
        owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)
    assert raised.value.assignment_restarted_fresh is True
    assert raised.value.reason == "operator_disable"
    assert owner.permit is None
    assert owner.ordinary_fallback is True
    assert owner.paid_gate_reconcile_required is False
    denial = json.loads((owner.gate_dir / "denial.json").read_text())
    assert denial["code"] == disable_code
    assert [command for command, _payload in owner.api.calls][-2:] == [
        "start", "fresh-fallback",
    ]
    owner.mainline_exited()


def test_owner_restart_reserves_restores_then_commits_at_paid_gate(
    monkeypatch, tmp_path: Path,
) -> None:
    assignment = _assignment()
    contract = checkpoint_adapter_contract_v2("codex", "openai")
    runtime_profile, model_config_version = _runtime_config_v2("codex", "openai")
    assignment.update({
        "execution_state": "paused",
        "checkpoint_id": "checkpoint-resume-0001",
        "owner_epoch": 2,
    })
    assignment["execution_identity"] = {
        **assignment["execution_identity"],
        "runtime_profile": runtime_profile,
        "model_config_version": model_config_version,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": contract.checkpoint_abi,
        "runtime_compatibility_digest": "c" * 64,
        "identity_state": "FINAL",
    }
    identity = ExecutionIdentityV2.from_assignment(assignment)
    assignment["checkpoint_v2_selected_generation"] = {
        "descriptor_schema": "dradar-checkpoint-selected-generation-v2",
        "checkpoint_id": "checkpoint-resume-0001",
        "checkpoint_lineage_id": "lineage-resume-0001",
        "snapshot_generation": 5,
        "capture_id": "capture-resume-0001",
        "manifest_schema": 2,
        "manifest_sha256": "d" * 64,
        "compatibility_fingerprint": identity.fingerprint,
        "recovery_capability": "NATIVE_VALID",
        "native_state_schema": contract.native_state_schema,
        "storage_scope": "machine_local",
        "writer_machine_fingerprint": "b" * 64,
        "sync_state": "local_only",
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": contract.checkpoint_abi,
    }
    published_root = tmp_path / "published-generation"
    published = PublishedCheckpointV2(
        checkpoint_id="checkpoint-resume-0001",
        snapshot_generation=5,
        capture_id="capture-resume-0001",
        root=published_root,
        payload_root=published_root / "payload",
        archive_path=published_root / "export.tar.gz",
        manifest_sha256="d" * 64,
        archive_sha256="e" * 64,
        archive_bytes=100,
        file_count=4,
        payload_bytes=50,
        authoritative=True,
        selected=True,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.docker_container_backend_v2",
        lambda: "docker",
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.checkpoint_machine_fingerprint",
        lambda _home: "b" * 64,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2._runtime_digest_v2",
        lambda **_facts: "c" * 64,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.load_exact_published_checkpoint_v2",
        lambda *_args, **_kwargs: published,
    )
    api = _ResumeApi(assignment)
    owner = AuthoritativeCheckpointRunV2(
        assignment=assignment,
        effective_assignment=assignment,
        activation=negotiate_checkpoint_activation_v2(
            local_mode="canary", server_mode="canary", controlled_account=True,
        ),
        api=api,
        telemetry=_Telemetry(),
        home=tmp_path,
        job_root=tmp_path / "work/jobs/aassignment-owner-0001",
        renew_interval_sec=600,
        initial_capture_delay_sec=86_400,
        capture_interval_sec=86_400,
    )
    offline = owner.prepare()
    assert not isinstance(offline, PaidExecutionPermit)
    gate_contract = json.loads((owner.gate_dir / "contract.json").read_text())
    assert gate_contract["action"] == "resume"
    request = {
        "schema": "dradar-checkpoint-paid-gate-request-v2",
        "assignment_id": assignment["assignment_id"],
        "gate_nonce": gate_contract["gate_nonce"],
        "action": "resume",
        "restore_receipt_sha256": gate_contract["restore_receipt_sha256"],
    }
    owner._write_once(owner.gate_dir / "request.json", request)

    class _LivePier:
        @staticmethod
        def poll():
            return None

    paid = owner.authorize_at_paid_gate(_LivePier(), timeout_sec=1)
    assert paid.source == "resume"
    assert [command for command, _payload in api.calls] == [
        "identity/finalize", "resume-reserve", "resume-commit",
    ]
    owner.mainline_exited()


def test_paused_owner_invalid_descriptor_falls_back_before_provider_start(
    monkeypatch, tmp_path: Path,
) -> None:
    assignment = _assignment()
    assignment.update({
        "execution_state": "paused",
        "checkpoint_id": "checkpoint-resume-0001",
        "owner_epoch": 2,
    })
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.docker_container_backend_v2",
        lambda: "docker",
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.checkpoint_machine_fingerprint",
        lambda _home: "b" * 64,
    )
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2._runtime_digest_v2",
        lambda **_facts: "c" * 64,
    )
    api = _ResumeApi(assignment)
    owner = AuthoritativeCheckpointRunV2(
        assignment=assignment,
        effective_assignment=assignment,
        activation=negotiate_checkpoint_activation_v2(
            local_mode="canary", server_mode="canary", controlled_account=True,
        ),
        api=api,
        telemetry=_Telemetry(),
        home=tmp_path,
        job_root=tmp_path / "work/jobs/aassignment-owner-0001",
        renew_interval_sec=600,
        initial_capture_delay_sec=86_400,
        capture_interval_sec=86_400,
    )
    with pytest.raises(CheckpointV2OrdinaryFallback) as caught:
        owner.prepare()
    assert caught.value.assignment_restarted_fresh is True
    assert caught.value.reason == "selected_descriptor_invalid"
    assert owner.ordinary_fallback is True
    assert owner.permit is None
    assert [command for command, _payload in api.calls] == [
        "identity/finalize", "fresh-fallback",
    ]


def test_snapshot_absence_faults_without_invalidating_assignment(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    _authorize_fresh_at_gate(owner)
    assert owner.pause_last_sealed() is None
    command, payload = owner.api.calls[-1]
    assert command == "failure"
    assert payload["recoverability"] == "none"
    assert payload["code"] == "checkpoint_capture_failed"


def test_owner_renewal_failure_is_fail_stop_not_fail_open(
    owner: AuthoritativeCheckpointRunV2,
) -> None:
    owner.prepare()
    paid = _authorize_fresh_at_gate(owner)
    with owner._lock:
        owner._permit = replace(
            paid, owner_lease_expires_at="2000-01-01T00:00:00+00:00",
        )
        owner._fatal = RuntimeError("bounded test failure")
    with pytest.raises(CheckpointV2OwnerLost, match="paid execution must stop"):
        owner.raise_if_fatal()
    owner.mainline_exited()


class _UploadOwner:
    def __init__(self) -> None:
        self.permit = PaidExecutionPermit(
            assignment_id="assignment-upload-0001",
            identity_fingerprint="f" * 64,
            session_id="session-upload-0001",
            owner_epoch=3,
            owner_lease_expires_at="2030-01-01T00:00:00+00:00",
            source="fresh",
            usage_segment_id="usage-segment-upload-0001",
        )
        self.calls = []

    def completed_result_recovery_descriptor(
        self, *, usage_operation_id: str, result_operation_id: str,
    ) -> dict:
        return {
            "schema": "dradar-checkpoint-completed-result-recovery-v2",
            "assignment_id": self.permit.assignment_id,
            "execution_identity": {},
            "rollout_mode": "canary",
            "session_id": self.permit.session_id,
            "owner_epoch": self.permit.owner_epoch,
            "owner_lease_expires_at": self.permit.owner_lease_expires_at,
            "source": self.permit.source,
            "usage_segment_id": self.permit.usage_segment_id,
            "usage_operation_id": usage_operation_id,
            "result_operation_id": result_operation_id,
        }

    def finalize_usage(self, **facts):
        self.calls.append(("usage", facts))
        return FinalizedUsageSegmentReceiptV2(
            assignment_id=self.permit.assignment_id,
            identity_fingerprint=self.permit.identity_fingerprint,
            session_id=self.permit.session_id,
            owner_epoch=self.permit.owner_epoch,
            usage_segment_id=self.permit.usage_segment_id,
            completeness="complete",
            evidence_sha256="8" * 64,
            usage_ledger_sha256="9" * 64,
            usage_ledger_complete=True,
        )

    def declare_result_ready(
        self, *, upload_intent_id: str, operation_id: str | None = None,
    ):
        self.calls.append(("result-ready", upload_intent_id))
        self.permit = CompletedResultPermit(
            assignment_id=self.permit.assignment_id,
            identity_fingerprint=self.permit.identity_fingerprint,
            session_id=self.permit.session_id,
            owner_epoch=self.permit.owner_epoch,
            owner_lease_expires_at=self.permit.owner_lease_expires_at,
            upload_intent_id=upload_intent_id,
            usage_ledger_sha256="9" * 64,
        )
        return self.permit

    def release_after_submission(self) -> str:
        self.calls.append(("retention", None))
        return "submission-upload-0001"


class _RetentionFailingOwner(_UploadOwner):
    def release_after_submission(self) -> str:
        self.calls.append(("retention", None))
        raise RuntimeError("simulated response loss")


class _UploadClient:
    def __init__(self) -> None:
        self.calls = []

    def submit_checkpoint_v2(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {
            "submission_id": "submission-upload-0001",
            "grade_status": "pending",
        }


class _OwnerApiUploadClient(_Api):
    def submit_checkpoint_v2(self, *args, **kwargs):
        self.calls.append(("submit", {"args": args, "kwargs": kwargs}))
        return {
            "submission_id": "submission-owner-0001",
            "grade_status": "pending",
        }


class _RetryUploadClient(_UploadClient):
    def submit_checkpoint_v2(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise ApiError("already submitted", status_code=409)

    def checkpoint_v2_command(self, command: str, payload: dict):
        assert command == "retention"
        return {
            "ok": True,
            "assignment_id": payload["assignment_id"],
            "operation_id": payload["operation_id"],
            "owner_epoch_observed": payload["owner_epoch_observed"],
            "current_owner_epoch": payload["owner_epoch_observed"] + 1,
            "delete_generations": [],
            "retain_generations": [],
            "result_evidence_release": True,
            "upload_intent_id": payload["upload_intent_id"],
            "submission_id": "submission-upload-0001",
            "assignment_unchanged": True,
            "paid_execution_authorized": False,
        }


class _SubmitResponseLostClient(_UploadClient):
    def submit_checkpoint_v2(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise ApiError("simulated submit response loss", status_code=503)


def test_upload_uses_v2_result_ready_endpoint_and_retention_before_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    job_dir = home / "work" / "jobs" / "aassignment-upload-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.patch").write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runloop, "HOME", home)
    monkeypatch.setattr(runloop, "check_task_content_hash", lambda *_args: True)
    owner = _UploadOwner()
    client = _UploadClient()

    outcome = runloop._upload_trial(
        client,
        {
            "assignment_id": "assignment-upload-0001",
            "nonce": "nonce-upload-0001",
            "task_id": "t1",
            "trial_dir": str(trial_dir),
            "job_dir": str(job_dir),
            "meta": {
                "n_input_tokens": 10,
                "n_cache_tokens": 2,
                "n_output_tokens": 3,
                "token_usage_events": [{
                    "occurred_at": "2026-08-23T12:00:00+00:00",
                    "n_input_tokens": 10,
                    "n_cache_tokens": 2,
                    "n_output_tokens": 3,
                }],
                "request_usage_complete": True,
                "request_usage_observed": True,
            },
            "outcome": "completed",
            "keep": False,
        },
        checkpoint_v2_owner=owner,
    )

    assert outcome == "submitted"
    assert len(client.calls) == 1
    _args, kwargs = client.calls[0]
    assert kwargs["owner_epoch"] == 3
    assert kwargs["session_id"] == "session-upload-0001"
    assert len(kwargs["upload_intent_id"]) == 64
    assert [name for name, _facts in owner.calls] == [
        "usage", "result-ready", "retention",
    ]
    usage_facts = owner.calls[0][1]
    assert usage_facts["request_usage_complete"] is True
    assert usage_facts["request_usage_observed"] is True
    assert usage_facts["token_usage_events"][0]["n_output_tokens"] == 3
    assert not job_dir.exists()


def test_completed_result_free_commands_replay_from_exact_descriptor(
    owner: AuthoritativeCheckpointRunV2,
    tmp_path: Path,
) -> None:
    owner.prepare()
    _authorize_fresh_at_gate(owner)
    descriptor = owner.completed_result_recovery_descriptor(
        usage_operation_id="usage-recovery-operation-0001",
        result_operation_id="result-recovery-operation-0001",
    )
    recovery_api = _Api(_assignment())
    receipt = recover_completed_result_usage_v2(
        descriptor,
        api=recovery_api,
        home=tmp_path / "recovery-home",
        n_input_tokens=10,
        n_cache_tokens=2,
        n_output_tokens=3,
        token_usage_events=[{
            "occurred_at": "2026-08-23T12:00:00+00:00",
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 3,
        }],
        request_usage_complete=True,
        request_usage_observed=True,
    )
    completed = recover_completed_result_ready_v2(
        descriptor,
        usage_receipt=receipt,
        upload_intent_id="e" * 64,
        api=recovery_api,
        home=tmp_path / "recovery-home",
    )

    assert completed.upload_intent_id == "e" * 64
    assert [command for command, _payload in recovery_api.calls] == [
        "usage-finalize", "result-ready",
    ]
    assert recovery_api.calls[0][1]["operation_id"] == (
        "usage-recovery-operation-0001"
    )
    assert recovery_api.calls[1][1]["operation_id"] == (
        "result-recovery-operation-0001"
    )
    assert all(
        command not in {"start", "resume-commit"}
        for command, _payload in recovery_api.calls
    )

    tampered = json.loads(json.dumps(descriptor))
    tampered["execution_identity"]["provider"] = "different-provider"
    call_count = len(recovery_api.calls)
    with pytest.raises(
        CheckpointV2ProtocolError,
    ):
        recover_completed_result_usage_v2(
            tampered,
            api=recovery_api,
            home=tmp_path / "tampered-home",
            n_input_tokens=10,
            n_cache_tokens=2,
            n_output_tokens=3,
        )
    assert len(recovery_api.calls) == call_count
    owner.mainline_exited()


@pytest.mark.parametrize("crash_stage", ["usage-finalize", "result-ready"])
def test_completed_result_upload_recovers_after_free_command_response_loss(
    owner: AuthoritativeCheckpointRunV2,
    monkeypatch,
    tmp_path: Path,
    crash_stage: str,
) -> None:
    owner.prepare()
    _authorize_fresh_at_gate(owner)
    owner.mainline_exited()
    job_dir = tmp_path / "work" / "jobs" / "aassignment-owner-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.patch").write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    entry = {
        "assignment_id": "assignment-owner-0001",
        "nonce": "nonce-owner-0001",
        "task_id": "t1",
        "trial_dir": str(trial_dir),
        "job_dir": str(job_dir),
        "checkpoint_protocol_version": 2,
        "meta": {
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 3,
            "token_usage_events": [{
                "occurred_at": "2026-08-23T12:00:00+00:00",
                "n_input_tokens": 10,
                "n_cache_tokens": 2,
                "n_output_tokens": 3,
            }],
            "request_usage_complete": True,
            "request_usage_observed": True,
        },
        "outcome": "completed",
        "keep": False,
    }
    original_command = owner.api.checkpoint_v2_command
    response_lost = False

    def lose_exact_response(command: str, payload: dict):
        nonlocal response_lost
        response = original_command(command, payload)
        if command == crash_stage and not response_lost:
            response_lost = True
            raise ApiError(
                f"simulated {crash_stage} response loss",
                status_code=503,
            )
        return response

    owner.api.checkpoint_v2_command = lose_exact_response

    first = runloop._upload_trial(
        _UploadClient(), entry, checkpoint_v2_owner=owner,
    )
    assert first == "upload-failed"
    saved = pending.load(tmp_path)
    assert len(saved) == 1
    assert saved[0]["checkpoint_protocol_version"] == 2
    assert saved[0]["checkpoint_v2_recovery"]["usage_operation_id"]
    if crash_stage == "usage-finalize":
        assert "checkpoint_v2" not in saved[0]
    else:
        assert saved[0]["checkpoint_v2"]["result_ready_state"] == "pending"

    recovery_client = _OwnerApiUploadClient(_assignment())
    second = runloop._upload_trial(recovery_client, saved[0])
    assert second == "submitted"
    assert pending.load(tmp_path) == []
    assert not job_dir.exists()
    commands = [command for command, _payload in recovery_client.calls]
    expected = ["result-ready", "submit", "retention"]
    if crash_stage == "usage-finalize":
        expected.insert(0, "usage-finalize")
    assert commands == expected
    assert "start" not in commands
    assert "resume-commit" not in commands


def test_protocol_v2_pending_result_never_downgrades_to_legacy_submit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "work" / "jobs" / "aassignment-upload-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.patch").write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runloop, "HOME", tmp_path)

    class NoSubmitClient:
        def submit(self, *_args, **_kwargs):
            raise AssertionError("legacy submit must not be called")

        def submit_checkpoint_v2(self, *_args, **_kwargs):
            raise AssertionError("V2 submit must not precede result-ready")

    outcome = runloop._upload_trial(NoSubmitClient(), {
        "assignment_id": "assignment-upload-0001",
        "nonce": "nonce-upload-0001",
        "task_id": "t1",
        "trial_dir": str(trial_dir),
        "job_dir": str(job_dir),
        "checkpoint_protocol_version": 2,
        "meta": {},
        "outcome": "completed",
        "keep": False,
    })

    assert outcome == "upload-failed"
    assert pending.load(tmp_path)[0]["checkpoint_protocol_version"] == 2


def test_completed_result_retention_recovers_after_live_owner_process_is_gone(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    job_dir = home / "work" / "jobs" / "aassignment-upload-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.patch").write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runloop, "HOME", home)
    entry = {
        "assignment_id": "assignment-upload-0001",
        "nonce": "nonce-upload-0001",
        "task_id": "t1",
        "trial_dir": str(trial_dir),
        "job_dir": str(job_dir),
        "meta": {
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 3,
        },
        "outcome": "completed",
        "keep": False,
    }

    first = runloop._upload_trial(
        _UploadClient(),
        entry,
        checkpoint_v2_owner=_RetentionFailingOwner(),
    )
    assert first == "submitted-retention-pending"
    assert job_dir.exists()
    saved = pending.load(home)
    assert len(saved) == 1
    assert saved[0]["checkpoint_v2"]["retention_pending"] is True
    # Compatibility with V2 ledgers produced before the protocol marker was
    # copied into every pending upload entry.
    saved[0].pop("checkpoint_protocol_version")
    pending.record(home, saved[0])

    second = runloop._upload_trial(_RetryUploadClient(), saved[0])
    assert second == "submitted"
    assert pending.load(home) == []
    assert not job_dir.exists()


def test_v2_submit_response_loss_reuses_exact_intent_without_model_rerun(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    job_dir = home / "work" / "jobs" / "aassignment-upload-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.patch").write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runloop, "HOME", home)
    entry = {
        "assignment_id": "assignment-upload-0001",
        "nonce": "nonce-upload-0001",
        "task_id": "t1",
        "trial_dir": str(trial_dir),
        "job_dir": str(job_dir),
        "meta": {
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 3,
        },
        "outcome": "completed",
        "keep": False,
    }

    first_client = _SubmitResponseLostClient()
    first = runloop._upload_trial(
        first_client, entry, checkpoint_v2_owner=_UploadOwner(),
    )

    assert first == "upload-failed"
    saved = pending.load(home)
    assert len(saved) == 1
    facts = saved[0]["checkpoint_v2"]
    assert facts["result_ready_state"] == "acknowledged"
    assert first_client.calls[0][1]["upload_intent_id"] == (
        facts["upload_intent_id"]
    )

    retry_client = _RetryUploadClient()
    second = runloop._upload_trial(retry_client, saved[0])

    assert second == "submitted"
    assert retry_client.calls[0][1]["upload_intent_id"] == (
        facts["upload_intent_id"]
    )
    assert pending.load(home) == []
    assert not job_dir.exists()


def test_runloop_requires_explicit_authoritative_factory_for_protocol_v2(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    job_dir = home / "work" / "jobs" / "aassignment-upload-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    patch = artifacts / "model.patch"
    patch.write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    owner = _UploadOwner()
    assignment = {
        **_assignment(),
        "assignment_id": "assignment-upload-0001",
        "nonce": "nonce-upload-0001",
    }
    assignment["execution_identity"] = {
        **assignment["execution_identity"],
        "task_content_sha256": assignment["task_content_hash"],
    }
    art = TrialArtifacts(
        job_dir=job_dir,
        trial_dir=trial_dir,
        patch=patch,
        trajectory=None,
        result=None,
        returncode=0,
        duration_sec=1.0,
        log_path=home / "pier.log",
        codex_cli_version="0.150.0",
    )
    observed = {}

    def fake_run_trial(*_args, **kwargs):
        factory = kwargs.get("checkpoint_owner_factory")
        assert factory is not None
        observed["shadow"] = kwargs.get("checkpoint_shadow_factory")
        art.checkpoint_v2_owner = factory(assignment, job_dir)
        return art

    monkeypatch.setattr(runloop, "HOME", home)
    monkeypatch.setattr(runloop, "check_task_content_hash", lambda *_args: True)
    monkeypatch.setenv("DRADAR_CHECKPOINT_V2_MODE", "canary")
    monkeypatch.setattr(runloop, "run_trial", fake_run_trial)
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.AuthoritativeCheckpointRunV2",
        lambda **_kwargs: owner,
    )
    telemetry = _Telemetry()
    client = _UploadClient()
    args = argparse.Namespace(
        dev_agent=None,
        allow_task_drift=False,
        keep=False,
        yes=True,
        parallel=False,
        archive_session=False,
    )

    outcome = runloop._run_and_submit(
        client,
        assignment,
        tmp_path,
        args,
        "commit-test",
        telemetry=telemetry,
    )

    assert outcome == "submitted"
    assert observed["shadow"] is None
    assert len(client.calls) == 1


def test_runloop_retries_restore_preflight_failure_as_fresh_v1(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    job_dir = home / "work" / "jobs" / "aassignment-fallback-0001"
    trial_dir = job_dir / "trial"
    artifacts = trial_dir / "artifacts"
    artifacts.mkdir(parents=True)
    patch = artifacts / "model.patch"
    patch.write_text(
        "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0,0 +1 @@\n+x\n",
        encoding="utf-8",
    )
    assignment = {
        **_assignment(),
        "assignment_id": "assignment-fallback-0001",
        "nonce": "nonce-fallback-0001",
        "execution_state": "paused",
        "checkpoint_id": "checkpoint-fallback-0001",
        "owner_epoch": 2,
        "resume_generation": 7,
    }
    assignment["execution_identity"] = {
        **assignment["execution_identity"],
        "task_content_sha256": assignment["task_content_hash"],
    }
    art = TrialArtifacts(
        job_dir=job_dir,
        trial_dir=trial_dir,
        patch=patch,
        trajectory=None,
        result=None,
        returncode=0,
        duration_sec=1.0,
        log_path=home / "pier.log",
        codex_cli_version="0.150.0",
    )

    class _FallbackOwner:
        permit = None

        @staticmethod
        def build_ordinary_fallback_shadow():
            return object()

    fallback_owner = _FallbackOwner()
    calls = []

    def fake_run_trial(*_args, **kwargs):
        calls.append({
            "owner": kwargs.get("checkpoint_owner_factory"),
            "shadow": kwargs.get("checkpoint_shadow_factory"),
            "resume": kwargs.get("resume_checkpoint"),
            "protocol": assignment["checkpoint_protocol_version"],
        })
        if len(calls) == 1:
            factory = kwargs["checkpoint_owner_factory"]
            assert factory is not None
            assert factory(assignment, job_dir) is fallback_owner
            raise CheckpointV2OrdinaryFallback(
                assignment_restarted_fresh=True,
                owner_epoch=4,
                resume_generation=8,
                reason="restore_preflight_failed",
            )
        assert kwargs["checkpoint_owner_factory"] is None
        assert kwargs["checkpoint_shadow_factory"] is not None
        assert kwargs["resume_checkpoint"] is None
        return art

    monkeypatch.setattr(runloop, "HOME", home)
    monkeypatch.setattr(runloop, "check_task_content_hash", lambda *_args: True)
    monkeypatch.setenv("DRADAR_CHECKPOINT_V2_MODE", "canary")
    monkeypatch.setattr(runloop, "run_trial", fake_run_trial)
    monkeypatch.setattr(
        "dradar.checkpoint_owner_runtime_v2.AuthoritativeCheckpointRunV2",
        lambda **_kwargs: fallback_owner,
    )
    monkeypatch.setattr(
        runloop.artifact_staging, "ensure_staged_patch", lambda *_args: None,
    )
    monkeypatch.setattr(
        runloop.image_cache, "record_trial_images", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runloop, "summarize_result", lambda *_args: {})
    uploaded = []
    monkeypatch.setattr(
        runloop,
        "_upload_trial",
        lambda _client, entry, **_kwargs: uploaded.append(entry) or "submitted",
    )
    args = argparse.Namespace(
        dev_agent=None,
        allow_task_drift=False,
        keep=False,
        yes=True,
        parallel=False,
        archive_session=False,
    )
    outcome = runloop._run_and_submit(
        _UploadClient(),
        assignment,
        tmp_path,
        args,
        "commit-test",
        telemetry=_Telemetry(),
        resume_checkpoint=SimpleNamespace(
            checkpoint_dir=tmp_path / "old-checkpoint",
        ),
    )

    assert outcome == "submitted"
    assert len(calls) == 2
    assert calls[0]["resume"] == tmp_path / "old-checkpoint"
    assert calls[1]["protocol"] == 1
    assert assignment["checkpoint_protocol_version"] == 1
    assert assignment["checkpoint_id"] is None
    assert assignment["resume_generation"] == 8
    assert uploaded[0]["resume_generation"] == 8


def test_runloop_faults_refill_after_ambiguous_paid_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assignment = {
        **_assignment(),
        "checkpoint_protocol_version": 1,
        "assignment_id": "assignment-gate-fault-0001",
        "nonce": "nonce-gate-fault-0001",
    }
    assignment["execution_identity"] = {
        **assignment["execution_identity"],
        "task_content_sha256": assignment["task_content_hash"],
    }
    monkeypatch.setattr(runloop, "HOME", tmp_path / "home")
    monkeypatch.setattr(runloop, "check_task_content_hash", lambda *_args: True)
    monkeypatch.setenv("DRADAR_CHECKPOINT_V2_MODE", "off")
    monkeypatch.setattr(
        runloop,
        "run_trial",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CheckpointV2PaidGateFaultedError("ambiguous paid gate")
        ),
    )
    circuits = []
    monkeypatch.setattr(
        runloop.refill_plan,
        "open_circuit",
        lambda _home, observed_assignment, family: circuits.append(
            (observed_assignment["assignment_id"], family)
        ),
    )
    args = argparse.Namespace(
        dev_agent=None,
        allow_task_drift=False,
        keep=False,
        yes=True,
        parallel=False,
        archive_session=False,
    )
    outcome = runloop._run_and_submit(
        _UploadClient(), assignment, tmp_path, args, "commit-test",
    )
    assert outcome == "checkpoint-v2-faulted"
    assert circuits == [(
        "assignment-gate-fault-0001", "checkpoint_reconcile_ambiguous",
    )]
