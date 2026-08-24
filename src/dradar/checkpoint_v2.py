"""Fail-closed local command journal for checkpoint protocol v2.

The server owns assignment state.  This module only makes each client command
durable before the HTTP request leaves the process, so a timeout or restart
reuses the same ``operation_id`` instead of guessing whether the transition
committed.  Journal files contain bounded protocol facts only; prompts,
credentials, provider logs, and command lines are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .api_client import ApiClient, ApiError
from .checkpoint_activation_v2 import (
    CHECKPOINT_CORE_ABI_V2,
    CheckpointActivationV2,
    CheckpointRolloutModeV2,
    CheckpointV2ProtocolError,
    checkpoint_activation_from_assignment_v2,
    negotiate_checkpoint_activation_v2,
)
from .checkpoint_protocol_types_v2 import (
    CheckpointGenerationRefV2,
    CheckpointRetentionAcknowledgementV2,
)
from .checkpoints import assignment_lock
from .providers import (
    DEEPSEEK_PROVIDER,
    DSH_AGENT,
    GROK_AGENT,
    GROK_PROVIDER,
    KIMI_AGENT,
    KIMI_PROVIDER,
    ZCODE_AGENT,
    ZCODE_PROVIDER,
)
from .submission_intent import CHECKPOINT_V2_UPLOAD_INTENT_VERSION

JOURNAL_SCHEMA = "dradar-checkpoint-command-journal-v2"
JOURNAL_DIR = "checkpoint-v2-journal"
MACHINE_ID_FILE = "machine-identity.json"
USAGE_SEGMENT_SCHEMA_V2 = "dradar-checkpoint-usage-segment-v2"
MAX_JOURNAL_BYTES = 512 * 1024
MAX_PROTOCOL_OBJECT_BYTES = 256 * 1024
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_SESSION_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]{8,64}$")
_SENSITIVE_KEY_PARTS = (
    "token", "secret", "password", "credential", "api_key", "authorization",
    "proxy", "prompt", "command_line", "shell_command", "argv", "stdout", "stderr",
)
_SAFE_USAGE_COUNTER_KEYS = frozenset({
    "n_input_tokens", "n_cache_tokens", "n_output_tokens",
})
_STATES = frozenset({"PENDING", "ACKNOWLEDGED", "REJECTED"})
_PAID_AUTHORIZE_COMMANDS = frozenset({"start", "resume-commit"})
_FREE_ONLY_COMMANDS = frozenset({
    "identity/finalize",
    "checkout",
    "renew",
    "usage-finalize",
    "result-ready",
    "pause",
    "resume-reserve",
    "resume-abort",
    "fresh-fallback",
    "paid-gate-reconcile",
    "failure",
    "retention",
})
_CANONICAL_HARNESS_BY_AGENT = {
    "codex": "codex",
    DSH_AGENT: "dsh",
    GROK_AGENT: GROK_AGENT,
    KIMI_AGENT: KIMI_AGENT,
    ZCODE_AGENT: ZCODE_AGENT,
}
_ALLOWED_HARNESS_PROVIDERS = {
    "codex": frozenset({"openai", DEEPSEEK_PROVIDER}),
    "dsh": frozenset({DEEPSEEK_PROVIDER}),
    GROK_AGENT: frozenset({GROK_PROVIDER}),
    KIMI_AGENT: frozenset({KIMI_PROVIDER}),
    ZCODE_AGENT: frozenset({ZCODE_PROVIDER}),
}


class CheckpointV2OrdinaryFallback(CheckpointV2ProtocolError):
    """The server kept this run on V1 because its exact cohort is uncertified."""

    def __init__(
        self,
        observation_mode: str = "observe",
        *,
        assignment_restarted_fresh: bool = False,
        owner_epoch: int | None = None,
        resume_generation: int | None = None,
        reason: str = "checkpoint_v2_cohort_not_certified",
    ) -> None:
        super().__init__(
            "checkpoint v2 recovery declined; continue ordinary execution"
        )
        self.observation_mode = observation_mode
        self.assignment_restarted_fresh = assignment_restarted_fresh
        self.owner_epoch = owner_epoch
        self.resume_generation = resume_generation
        self.reason = reason


_FAILURE_STAGES = frozenset({
    "prepare", "capture", "seal", "finalize", "restore", "reconcile",
    "retention",
})
_FAILURE_LAYERS = frozenset({
    "checkpoint_core", "harness_adapter", "provider_runtime",
    "result_extractor", "task_pack",
})
_CONTAINER_BACKENDS = frozenset({
    "docker", "orbstack", "podman", "native", "other",
})
_FAILURE_CODES = frozenset({
    "checkpoint_prepare_failed",
    "checkpoint_capture_failed",
    "checkpoint_seal_failed",
    "checkpoint_finalize_failed",
    "checkpoint_manifest_invalid",
    "checkpoint_identity_mismatch",
    "checkpoint_security_quarantine",
    "checkpoint_restore_failed",
    "checkpoint_owner_fenced",
    "checkpoint_reconcile_ambiguous",
    "completed_result_extract_failed",
    "provider_runtime_failed",
    "task_pack_prepare_failed",
})


class CheckpointV2JournalError(RuntimeError):
    pass


class CheckpointV2OperationConflict(CheckpointV2JournalError):
    pass


class CheckpointV2CommandRejected(CheckpointV2JournalError):
    def __init__(self, status_code: int | None, code: str | None):
        super().__init__(
            "checkpoint v2 command was already rejected"
            + (f" ({code})" if code else "")
        )
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class CheckpointPolicyV2:
    harness: str
    provider: str
    recovery_mode: str
    workspace_fallback_limit: int
    completed_result_salvage: bool

    @property
    def supported(self) -> bool:
        return self.recovery_mode != "unsupported"


@dataclass(frozen=True)
class ExecutionIdentityV2:
    assignment_id: str
    benchmark_id: str
    task_content_sha256: str
    harness: str
    provider: str
    model: str
    effort: str
    agent_version: str
    runtime_profile: str
    model_config_version: str
    checkpoint_core_abi: str
    checkpoint_abi: str
    runtime_compatibility_digest: str
    fingerprint: str

    @classmethod
    def from_assignment(cls, assignment: Mapping[str, Any]) -> "ExecutionIdentityV2":
        if (
            assignment.get("checkpoint_protocol_version") != 2
            and assignment.get("checkpoint_v2_identity_protocol_version") != 2
        ):
            raise CheckpointV2ProtocolError(
                "assignment does not expose checkpoint identity protocol v2"
            )
        assignment_id = _validate_identifier(
            assignment.get("assignment_id"), "assignment_id",
        )
        raw = assignment.get("execution_identity")
        if not isinstance(raw, dict) or raw.get("identity_state") != "FINAL":
            raise CheckpointV2ProtocolError(
                "checkpoint v2 execution identity is not final"
            )

        def required_string(key: str) -> str:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise CheckpointV2ProtocolError(
                    f"checkpoint execution identity has invalid {key}"
                )
            return value

        benchmark_id = required_string("benchmark_id")
        task_hash = required_string("task_content_sha256")
        harness = required_string("harness")
        provider = required_string("provider")
        model = required_string("model")
        effort = required_string("effort")
        agent_version = required_string("agent_version")
        runtime_profile = required_string("runtime_profile")
        model_config_version = required_string("model_config_version")
        checkpoint_core_abi = required_string("checkpoint_core_abi")
        checkpoint_abi = required_string("checkpoint_abi")
        runtime_digest = required_string("runtime_compatibility_digest")
        if re.fullmatch(r"[0-9a-f]{64}", task_hash) is None:
            raise CheckpointV2ProtocolError("task content digest is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", runtime_digest) is None:
            raise CheckpointV2ProtocolError("runtime compatibility digest is invalid")
        allowed_providers = _ALLOWED_HARNESS_PROVIDERS.get(harness)
        if allowed_providers is None or provider not in allowed_providers:
            raise CheckpointV2ProtocolError(
                "checkpoint Harness/provider identity is impossible"
            )
        if checkpoint_core_abi != CHECKPOINT_CORE_ABI_V2:
            raise CheckpointV2ProtocolError(
                "checkpoint core ABI is unsupported"
            )
        expected_abi = f"dradar-checkpoint-v2/{harness}/1"
        if checkpoint_abi != expected_abi:
            raise CheckpointV2ProtocolError(
                "checkpoint ABI does not match its authoritative Harness"
            )
        top_level_agent = assignment.get("agent")
        if isinstance(top_level_agent, str):
            if _CANONICAL_HARNESS_BY_AGENT.get(top_level_agent) != harness:
                raise CheckpointV2ProtocolError(
                    "assignment and execution identity disagree on Harness"
                )
        for top_key, identity_value in (
            ("benchmark_id", benchmark_id),
            ("task_content_hash", task_hash),
            ("model", model),
            ("effort", effort),
            ("agent_version", agent_version),
        ):
            top_value = assignment.get(top_key)
            if top_value is not None and top_value != identity_value:
                raise CheckpointV2ProtocolError(
                    f"assignment and execution identity disagree on {top_key}"
                )
        top_provider = assignment.get("provider")
        if top_provider is not None and top_provider != provider:
            raise CheckpointV2ProtocolError(
                "assignment and execution identity disagree on provider"
            )
        fingerprint_payload = {
            "assignment_id": assignment_id,
            "benchmark_id": benchmark_id,
            "task_content_sha256": task_hash,
            "harness": harness,
            "provider": provider,
            "model": model,
            "effort": effort,
            "agent_version": agent_version,
            "runtime_profile": runtime_profile,
            "model_config_version": model_config_version,
            "checkpoint_core_abi": checkpoint_core_abi,
            "checkpoint_abi": checkpoint_abi,
            "runtime_compatibility_digest": runtime_digest,
        }
        fingerprint = hashlib.sha256(
            _canonical_bytes(
                fingerprint_payload, label="checkpoint execution identity",
            )
        ).hexdigest()
        return cls(
            assignment_id=assignment_id,
            benchmark_id=benchmark_id,
            task_content_sha256=task_hash,
            harness=harness,
            provider=provider,
            model=model,
            effort=effort,
            agent_version=agent_version,
            runtime_profile=runtime_profile,
            model_config_version=model_config_version,
            checkpoint_core_abi=checkpoint_core_abi,
            checkpoint_abi=checkpoint_abi,
            runtime_compatibility_digest=runtime_digest,
            fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class FinalizedIdentityReceiptV2:
    """A free identity handshake, never an assignment owner or paid permit."""

    identity: ExecutionIdentityV2
    checkpoint_protocol_version: int
    checkpoint_v2_identity_protocol_version: int = 2
    assignment_ownership_unchanged: bool = True
    paid_execution_authorized: bool = False


def checkpoint_policy_v2(identity: ExecutionIdentityV2) -> CheckpointPolicyV2:
    key = (identity.harness, identity.provider)
    if key == ("codex", "openai") or key == ("codex", DEEPSEEK_PROVIDER):
        return CheckpointPolicyV2(
            identity.harness, identity.provider, "native_preferred", 1, True,
        )
    if key == (ZCODE_AGENT, ZCODE_PROVIDER):
        return CheckpointPolicyV2(
            identity.harness, identity.provider, "native_preferred", 1, True,
        )
    if key in {
        ("dsh", DEEPSEEK_PROVIDER),
        (KIMI_AGENT, KIMI_PROVIDER),
    }:
        return CheckpointPolicyV2(
            identity.harness, identity.provider, "native_required", 0, True,
        )
    if key == (GROK_AGENT, GROK_PROVIDER):
        return CheckpointPolicyV2(
            identity.harness, identity.provider, "unsupported", 0, True,
        )
    raise CheckpointV2ProtocolError("checkpoint policy identity is unsupported")


@dataclass(frozen=True)
class FreePreparationPermit:
    assignment_id: str
    identity_fingerprint: str
    session_id: str
    owner_epoch: int
    owner_lease_expires_at: str


@dataclass(frozen=True)
class OfflineRestorePermit(FreePreparationPermit):
    checkpoint_id: str
    snapshot_generation: int
    manifest_sha256: str
    checkpoint_core_abi: str
    checkpoint_abi: str
    compatibility_fingerprint: str
    requester_machine_fingerprint: str
    reservation_nonce: str


@dataclass(frozen=True)
class RestoreReceiptV2:
    assignment_id: str
    checkpoint_id: str
    snapshot_generation: int
    manifest_sha256: str
    session_id: str
    owner_epoch: int
    reservation_nonce: str
    requester_machine_fingerprint: str
    restore_adapter_version: str
    receipt_sha256: str


@dataclass(frozen=True)
class SealedCheckpointV2:
    checkpoint_id: str
    checkpoint_lineage_id: str
    snapshot_generation: int
    capture_id: str
    manifest_schema: int
    manifest_sha256: str
    compatibility_fingerprint: str
    recovery_capability: str
    native_state_schema: str | None
    storage_scope: str
    writer_machine_fingerprint: str
    sync_state: str

    def protocol_fields(self) -> dict[str, Any]:
        for value, label in (
            (self.checkpoint_id, "checkpoint_id"),
            (self.checkpoint_lineage_id, "checkpoint_lineage_id"),
            (self.capture_id, "capture_id"),
        ):
            _validate_identifier(value, label)
        if (
            not isinstance(self.snapshot_generation, int)
            or isinstance(self.snapshot_generation, bool)
            or self.snapshot_generation < 0
            or not isinstance(self.manifest_schema, int)
            or isinstance(self.manifest_schema, bool)
            or self.manifest_schema < 2
        ):
            raise CheckpointV2ProtocolError("checkpoint generation/schema is invalid")
        for value, label in (
            (self.manifest_sha256, "manifest digest"),
            (self.compatibility_fingerprint, "compatibility fingerprint"),
            (self.writer_machine_fingerprint, "writer machine fingerprint"),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise CheckpointV2ProtocolError(f"checkpoint {label} is invalid")
        if self.recovery_capability not in {
            "NATIVE_VALID", "WORKSPACE_ONLY", "COMPLETED_UPLOAD_ONLY", "NONE",
        }:
            raise CheckpointV2ProtocolError("checkpoint recovery capability is invalid")
        if self.storage_scope not in {
            "machine_local", "account_synced", "server_managed",
        }:
            raise CheckpointV2ProtocolError("checkpoint storage scope is invalid")
        if (
            self.storage_scope == "machine_local"
            and self.sync_state != "local_only"
        ) or (
            self.storage_scope != "machine_local" and self.sync_state != "synced"
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint storage scope and sync state are inconsistent"
            )
        if (
            self.native_state_schema is not None
            and (
                not isinstance(self.native_state_schema, str)
                or not self.native_state_schema
                or len(self.native_state_schema) > 160
            )
        ):
            raise CheckpointV2ProtocolError("native checkpoint schema is invalid")
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_lineage_id": self.checkpoint_lineage_id,
            "snapshot_generation": self.snapshot_generation,
            "capture_id": self.capture_id,
            "manifest_schema": self.manifest_schema,
            "manifest_sha256": self.manifest_sha256,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "recovery_capability": self.recovery_capability,
            "native_state_schema": self.native_state_schema,
            "storage_scope": self.storage_scope,
            "writer_machine_fingerprint": self.writer_machine_fingerprint,
            "sync_state": self.sync_state,
            "process_reaped": True,
        }


@dataclass(frozen=True)
class SelectedCheckpointGenerationV2:
    """The only server-selected generation a client may attempt to restore.

    This is deliberately richer than ``checkpoint_id``.  Recovery is bound to
    one manifest, generation, runtime identity, storage location and ABI.  A
    missing, malformed, unreachable, or policy-incompatible descriptor is a
    safe no-restore outcome; callers must never search for a nearby snapshot.
    """

    checkpoint_id: str
    checkpoint_lineage_id: str
    snapshot_generation: int
    capture_id: str
    manifest_schema: int
    manifest_sha256: str
    compatibility_fingerprint: str
    recovery_capability: str
    native_state_schema: str | None
    storage_scope: str
    writer_machine_fingerprint: str
    sync_state: str
    checkpoint_core_abi: str
    checkpoint_abi: str

    @classmethod
    def from_assignment(
        cls, assignment: Mapping[str, Any],
    ) -> "SelectedCheckpointGenerationV2":
        if assignment.get("checkpoint_protocol_version") != 2:
            raise CheckpointV2ProtocolError(
                "assignment does not authorize checkpoint restore v2"
            )
        if assignment.get("execution_state") not in {
            "paused", "resume_reserved", "running",
        }:
            raise CheckpointV2ProtocolError(
                "assignment is not in a checkpoint recovery state"
            )
        identity = ExecutionIdentityV2.from_assignment(assignment)
        raw = assignment.get("checkpoint_v2_selected_generation")
        if not isinstance(raw, Mapping) or raw.get("descriptor_schema") != (
            "dradar-checkpoint-selected-generation-v2"
        ):
            raise CheckpointV2ProtocolError(
                "assignment has no authoritative checkpoint generation"
            )

        def identifier(key: str) -> str:
            return _validate_identifier(raw.get(key), key)

        checkpoint_id = identifier("checkpoint_id")
        checkpoint_lineage_id = identifier("checkpoint_lineage_id")
        capture_id = identifier("capture_id")
        generation = raw.get("snapshot_generation")
        manifest_schema = raw.get("manifest_schema")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
            or manifest_schema != 2
        ):
            raise CheckpointV2ProtocolError(
                "selected checkpoint generation/schema is unsupported"
            )

        def digest(key: str) -> str:
            value = raw.get(key)
            if not isinstance(value, str) or re.fullmatch(
                r"[0-9a-f]{64}", value,
            ) is None:
                raise CheckpointV2ProtocolError(
                    f"selected checkpoint has invalid {key}"
                )
            return value

        manifest_sha256 = digest("manifest_sha256")
        compatibility_fingerprint = digest("compatibility_fingerprint")
        writer_machine_fingerprint = digest("writer_machine_fingerprint")
        recovery_capability = raw.get("recovery_capability")
        storage_scope = raw.get("storage_scope")
        sync_state = raw.get("sync_state")
        native_state_schema = raw.get("native_state_schema")
        checkpoint_core_abi = raw.get("checkpoint_core_abi")
        checkpoint_abi = raw.get("checkpoint_abi")
        if recovery_capability not in {"NATIVE_VALID", "WORKSPACE_ONLY"}:
            raise CheckpointV2ProtocolError(
                "selected checkpoint is not resumable"
            )
        if storage_scope not in {
            "machine_local", "account_synced", "server_managed",
        } or (
            storage_scope == "machine_local" and sync_state != "local_only"
        ) or (
            storage_scope != "machine_local" and sync_state != "synced"
        ):
            raise CheckpointV2ProtocolError(
                "selected checkpoint storage contract is invalid"
            )
        if native_state_schema is not None and (
            not isinstance(native_state_schema, str)
            or not native_state_schema
            or len(native_state_schema) > 160
        ):
            raise CheckpointV2ProtocolError(
                "selected checkpoint native schema is invalid"
            )
        if (
            checkpoint_id != assignment.get("checkpoint_id")
            or compatibility_fingerprint != identity.fingerprint
            or checkpoint_core_abi != identity.checkpoint_core_abi
            or checkpoint_abi != identity.checkpoint_abi
        ):
            raise CheckpointV2ProtocolError(
                "selected checkpoint disagrees with execution identity"
            )
        policy = checkpoint_policy_v2(identity)
        if (
            policy.recovery_mode == "native_required"
            and recovery_capability != "NATIVE_VALID"
        ):
            raise CheckpointV2ProtocolError(
                "selected checkpoint lacks required native recovery state"
            )
        return cls(
            checkpoint_id=checkpoint_id,
            checkpoint_lineage_id=checkpoint_lineage_id,
            snapshot_generation=generation,
            capture_id=capture_id,
            manifest_schema=manifest_schema,
            manifest_sha256=manifest_sha256,
            compatibility_fingerprint=compatibility_fingerprint,
            recovery_capability=recovery_capability,
            native_state_schema=native_state_schema,
            storage_scope=storage_scope,
            writer_machine_fingerprint=writer_machine_fingerprint,
            sync_state=sync_state,
            checkpoint_core_abi=checkpoint_core_abi,
            checkpoint_abi=checkpoint_abi,
        )

    def assert_reachable_from(self, machine_fingerprint: str) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", machine_fingerprint) is None:
            raise CheckpointV2ProtocolError(
                "checkpoint requester machine fingerprint is invalid"
            )
        if (
            self.storage_scope == "machine_local"
            and self.writer_machine_fingerprint != machine_fingerprint
        ):
            raise CheckpointV2ProtocolError(
                "machine-local checkpoint is unreachable from this runner"
            )


@dataclass(frozen=True)
class CheckpointFailureV2:
    stage: str
    code: str
    failure_layer: str
    recoverability: str
    cleanup_result: str
    container_backend: str
    last_sealed_checkpoint_id: str | None = None
    last_sealed_generation: int | None = None

    def protocol_fields(self) -> dict[str, Any]:
        if self.stage not in _FAILURE_STAGES:
            raise CheckpointV2ProtocolError("checkpoint failure stage is invalid")
        if self.failure_layer not in _FAILURE_LAYERS:
            raise CheckpointV2ProtocolError("checkpoint failure layer is invalid")
        if self.code not in _FAILURE_CODES:
            raise CheckpointV2ProtocolError("checkpoint failure code is invalid")
        if self.container_backend not in _CONTAINER_BACKENDS:
            raise CheckpointV2ProtocolError("checkpoint container backend is invalid")
        required_layer = {
            "completed_result_extract_failed": "result_extractor",
            "provider_runtime_failed": "provider_runtime",
            "task_pack_prepare_failed": "task_pack",
            "checkpoint_identity_mismatch": "checkpoint_core",
            "checkpoint_security_quarantine": "checkpoint_core",
            "checkpoint_owner_fenced": "checkpoint_core",
        }.get(self.code)
        if required_layer is not None and self.failure_layer != required_layer:
            raise CheckpointV2ProtocolError(
                "checkpoint failure code and layer are inconsistent"
            )
        if (
            self.code.startswith("checkpoint_")
            and required_layer is None
            and self.failure_layer not in {"checkpoint_core", "harness_adapter"}
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint failure code and layer are inconsistent"
            )
        if self.recoverability not in {
            "last_sealed", "completed_result", "none", "security_quarantine",
        }:
            raise CheckpointV2ProtocolError("checkpoint recoverability is invalid")
        if self.cleanup_result not in {
            "reaped", "already_stopped", "cleanup_failed",
        }:
            raise CheckpointV2ProtocolError("checkpoint cleanup result is invalid")
        if (self.last_sealed_checkpoint_id is None) != (
            self.last_sealed_generation is None
        ):
            raise CheckpointV2ProtocolError(
                "last sealed checkpoint identity must be complete"
            )
        if self.last_sealed_checkpoint_id is not None:
            _validate_identifier(
                self.last_sealed_checkpoint_id, "last_sealed_checkpoint_id",
            )
            if (
                not isinstance(self.last_sealed_generation, int)
                or isinstance(self.last_sealed_generation, bool)
                or self.last_sealed_generation < 0
            ):
                raise CheckpointV2ProtocolError(
                    "last sealed checkpoint generation is invalid"
                )
        return {
            "stage": self.stage,
            "code": self.code,
            "failure_layer": self.failure_layer,
            "recoverability": self.recoverability,
            "cleanup_result": self.cleanup_result,
            "container_backend": self.container_backend,
            "last_sealed_checkpoint_id": self.last_sealed_checkpoint_id,
            "last_sealed_generation": self.last_sealed_generation,
        }


@dataclass(frozen=True)
class PaidExecutionPermit(FreePreparationPermit):
    source: str
    usage_segment_id: str


@dataclass(frozen=True)
class UsageEventV2:
    event_id: str
    occurred_at: str
    n_input_tokens: int
    n_cache_tokens: int
    n_output_tokens: int

    def protocol_fields(self) -> dict[str, Any]:
        if re.fullmatch(r"[0-9a-f]{64}", self.event_id) is None:
            raise CheckpointV2ProtocolError("usage event identity is invalid")
        try:
            instant = datetime.fromisoformat(
                self.occurred_at.replace("Z", "+00:00")
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise CheckpointV2ProtocolError(
                "usage event timestamp is invalid"
            ) from exc
        counters = (
            self.n_input_tokens,
            self.n_cache_tokens,
            self.n_output_tokens,
        )
        if (
            instant.tzinfo is None
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in counters
            )
            or self.n_cache_tokens > self.n_input_tokens
        ):
            raise CheckpointV2ProtocolError("usage event counters are invalid")
        return {
            "event_id": self.event_id,
            "occurred_at": self.occurred_at,
            "n_input_tokens": self.n_input_tokens,
            "n_cache_tokens": self.n_cache_tokens,
            "n_output_tokens": self.n_output_tokens,
        }


@dataclass(frozen=True)
class UsageSegmentEvidenceV2:
    completeness: str
    evidence_kind: str
    events: tuple[UsageEventV2, ...]
    ledger_scope: str = "segment_delta"

    def protocol_fields(
        self,
        permit: PaidExecutionPermit,
    ) -> dict[str, Any]:
        if self.completeness not in {"complete", "partial", "unavailable"}:
            raise CheckpointV2ProtocolError(
                "usage segment completeness is invalid"
            )
        if self.evidence_kind not in {
            "trajectory_bundle",
            "provider_request_ledger",
            "session_usage",
            "unavailable",
        }:
            raise CheckpointV2ProtocolError("usage evidence kind is invalid")
        if self.ledger_scope not in {
            "segment_delta", "assignment_cumulative",
        }:
            raise CheckpointV2ProtocolError("usage ledger scope is invalid")
        if not isinstance(self.events, tuple) or len(self.events) > 512:
            raise CheckpointV2ProtocolError("usage event inventory is invalid")
        materialized = [event.protocol_fields() for event in self.events]
        materialized.sort(key=lambda item: (
            item["occurred_at"], item["event_id"],
        ))
        event_ids = [item["event_id"] for item in materialized]
        if len(event_ids) != len(set(event_ids)):
            raise CheckpointV2ProtocolError("usage event inventory has duplicates")
        if self.completeness == "unavailable":
            if self.evidence_kind != "unavailable" or materialized:
                raise CheckpointV2ProtocolError(
                    "unavailable usage must not invent token events"
                )
            counters: tuple[int | None, int | None, int | None] = (
                None, None, None,
            )
        else:
            if self.evidence_kind == "unavailable":
                raise CheckpointV2ProtocolError(
                    "known usage requires a concrete evidence kind"
                )
            counters = tuple(
                sum(int(item[name]) for item in materialized)
                for name in (
                    "n_input_tokens", "n_cache_tokens", "n_output_tokens",
                )
            )
        evidence_snapshot = {
            "schema": USAGE_SEGMENT_SCHEMA_V2,
            "assignment_id": permit.assignment_id,
            "usage_segment_id": permit.usage_segment_id,
            "owner_epoch": permit.owner_epoch,
            "runner_session_id": permit.session_id,
            "completeness": self.completeness,
            "evidence_kind": self.evidence_kind,
            "ledger_scope": self.ledger_scope,
            "events": materialized,
        }
        evidence_sha256 = hashlib.sha256(_canonical_bytes(
            evidence_snapshot, label="checkpoint usage evidence",
        ).rstrip(b"\n")).hexdigest()
        return {
            "usage_segment_id": permit.usage_segment_id,
            "usage_schema": USAGE_SEGMENT_SCHEMA_V2,
            "completeness": self.completeness,
            "evidence_kind": self.evidence_kind,
            "ledger_scope": self.ledger_scope,
            "n_input_tokens": counters[0],
            "n_cache_tokens": counters[1],
            "n_output_tokens": counters[2],
            "event_count": len(materialized),
            "events": materialized,
            "evidence_sha256": evidence_sha256,
        }


@dataclass(frozen=True)
class FinalizedUsageSegmentReceiptV2:
    assignment_id: str
    identity_fingerprint: str
    session_id: str
    owner_epoch: int
    usage_segment_id: str
    completeness: str
    evidence_sha256: str
    usage_ledger_sha256: str
    usage_ledger_complete: bool


@dataclass(frozen=True)
class CompletedResultPermit(FreePreparationPermit):
    upload_intent_id: str
    usage_ledger_sha256: str


@dataclass(frozen=True)
class JournalEntry:
    assignment_id: str
    operation_id: str
    command: str
    request_sha256: str
    payload: dict[str, Any]
    state: str
    created_at: str
    updated_at: str
    response: dict[str, Any] | None = None
    error_status: int | None = None
    error_code: str | None = None


def new_operation_id() -> str:
    return uuid.uuid4().hex


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _contains_sensitive_key(value: object) -> bool:
    pending = [value]
    nodes = 0
    while pending:
        current = pending.pop()
        nodes += 1
        if nodes > 2_000:
            return True
        if isinstance(current, dict):
            for key, child in current.items():
                normalized = str(key).lower().replace("-", "_")
                if (
                    normalized not in _SAFE_USAGE_COUNTER_KEYS
                    and any(
                        part in normalized for part in _SENSITIVE_KEY_PARTS
                    )
                ):
                    return True
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(current)
    return False


def _canonical_bytes(value: Mapping[str, Any], *, label: str) -> bytes:
    if _contains_sensitive_key(value):
        raise CheckpointV2JournalError(f"{label} contains a forbidden field")
    try:
        encoded = json.dumps(
            dict(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointV2JournalError(f"{label} is not JSON serializable") from exc
    if len(encoded) > MAX_PROTOCOL_OBJECT_BYTES:
        raise CheckpointV2JournalError(f"{label} exceeds the bounded journal size")
    return encoded


def _request_sha256(command: str, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_bytes(
        {"command_type": command, "request": dict(payload)},
        label="checkpoint request",
    )
    return hashlib.sha256(encoded).hexdigest()


def restore_receipt_sha256(
    *,
    permit: OfflineRestorePermit,
    restore_adapter_version: str,
) -> str:
    payload = {
        "assignment_id": permit.assignment_id,
        "checkpoint_id": permit.checkpoint_id,
        "snapshot_generation": permit.snapshot_generation,
        "manifest_sha256": permit.manifest_sha256,
        "checkpoint_core_abi": permit.checkpoint_core_abi,
        "checkpoint_abi": permit.checkpoint_abi,
        "compatibility_fingerprint": permit.compatibility_fingerprint,
        "session_id": permit.session_id,
        "owner_epoch": permit.owner_epoch,
        "reservation_nonce": permit.reservation_nonce,
        "requester_machine_fingerprint": (
            permit.requester_machine_fingerprint
        ),
        "restore_adapter_version": restore_adapter_version,
    }
    return hashlib.sha256(
        _canonical_bytes(payload, label="checkpoint restore receipt")
    ).hexdigest()


def completed_restore_receipt(
    permit: OfflineRestorePermit,
    *,
    restore_adapter_version: str,
    restored_manifest_sha256: str,
) -> RestoreReceiptV2:
    """Create a receipt only for the exact manifest the adapter restored.

    Harness adapters call this after their offline restore and validation have
    completed.  It cannot prove semantic correctness by itself, but it makes a
    stale, partial, wrong-generation, or cross-session commit fail closed.
    """

    if (
        not isinstance(restore_adapter_version, str)
        or re.fullmatch(r"[A-Za-z0-9._/-]{1,160}", restore_adapter_version) is None
    ):
        raise CheckpointV2ProtocolError("restore adapter version is invalid")
    if restored_manifest_sha256 != permit.manifest_sha256:
        raise CheckpointV2ProtocolError(
            "restore adapter validated a different checkpoint manifest"
        )
    return RestoreReceiptV2(
        assignment_id=permit.assignment_id,
        checkpoint_id=permit.checkpoint_id,
        snapshot_generation=permit.snapshot_generation,
        manifest_sha256=permit.manifest_sha256,
        session_id=permit.session_id,
        owner_epoch=permit.owner_epoch,
        reservation_nonce=permit.reservation_nonce,
        requester_machine_fingerprint=permit.requester_machine_fingerprint,
        restore_adapter_version=restore_adapter_version,
        receipt_sha256=restore_receipt_sha256(
            permit=permit,
            restore_adapter_version=restore_adapter_version,
        ),
    )


def _validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise CheckpointV2JournalError(f"invalid {label}")
    return value


def _validate_session_identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or _SESSION_IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise CheckpointV2JournalError("invalid session_id")
    return value


def _require_private_directory(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointV2JournalError("checkpoint journal directory is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CheckpointV2JournalError("checkpoint journal directory is unsafe")
    if hasattr(os, "getuid"):
        if metadata.st_uid != os.getuid():
            raise CheckpointV2JournalError("checkpoint journal directory has wrong owner")
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise CheckpointV2JournalError("checkpoint journal directory is not private")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    if len(payload) > MAX_JOURNAL_BYTES:
        raise CheckpointV2JournalError("checkpoint journal entry is too large")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise CheckpointV2JournalError("checkpoint journal could not be committed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_entry(path: Path) -> dict[str, Any]:
    descriptor = None
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_JOURNAL_BYTES
            or path.is_symlink()
        ):
            raise CheckpointV2JournalError("checkpoint journal entry is unsafe")
        if hasattr(os, "getuid") and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise CheckpointV2JournalError("checkpoint journal entry is not private")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise CheckpointV2JournalError("checkpoint journal entry changed")
        chunks = []
        remaining = MAX_JOURNAL_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 16 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_JOURNAL_BYTES:
            raise CheckpointV2JournalError("checkpoint journal entry is too large")
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            or len(payload) != after.st_size
        ):
            raise CheckpointV2JournalError("checkpoint journal entry changed")
        value = json.loads(payload)
    except CheckpointV2JournalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointV2JournalError("checkpoint journal entry is unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise CheckpointV2JournalError("checkpoint journal entry is not an object")
    return value


def checkpoint_machine_fingerprint(home: Path) -> str:
    """Return a private, stable storage-location fingerprint.

    The persisted value is random.  It contains no hostname, username,
    hardware identifier, path, or provider information.  Copying the complete
    DRADAR_HOME together with its snapshots intentionally copies this identity;
    merely running on the same physical host with another home does not.
    """

    root = home.absolute() / JOURNAL_DIR
    _require_private_directory(root, create=True)
    path = root / MACHINE_ID_FILE
    if not path.exists():
        payload = _canonical_bytes(
            {
                "schema": "dradar-checkpoint-machine-identity-v1",
                "random_id": uuid.uuid4().hex,
            },
            label="checkpoint machine identity",
        )
        descriptor = None
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            _fsync_directory(root)
        except FileExistsError:
            pass
        except OSError as exc:
            raise CheckpointV2JournalError(
                "checkpoint machine identity could not be committed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
    value = _read_entry(path)
    random_id = value.get("random_id")
    if (
        value.get("schema") != "dradar-checkpoint-machine-identity-v1"
        or not isinstance(random_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", random_id) is None
    ):
        raise CheckpointV2JournalError(
            "checkpoint machine identity is invalid"
        )
    return hashlib.sha256(
        f"dradar-checkpoint-storage-v1:{random_id}".encode("ascii")
    ).hexdigest()


class CheckpointV2Journal:
    def __init__(
        self,
        home: Path,
        *,
        assignment_lock_already_held: bool = False,
    ):
        self.home = home.absolute()
        self.root = self.home / JOURNAL_DIR
        self.assignment_lock_already_held = bool(
            assignment_lock_already_held
        )
        self._thread_lock = threading.RLock()

    def _command_lock(self, assignment_id: str):
        if self.assignment_lock_already_held:
            # The process-wide assignment lock is held for the complete model
            # lifetime.  Reopening and flocking the same file is not reentrant
            # on macOS/Linux, so only serialize this owner's journal threads.
            return self._thread_lock
        return assignment_lock(self.home, assignment_id)

    def _entry_path(self, assignment_id: str, operation_id: str) -> Path:
        assignment_id = _validate_identifier(assignment_id, "assignment_id")
        operation_id = _validate_identifier(operation_id, "operation_id")
        _require_private_directory(self.root, create=True)
        assignment_dir = self.root / assignment_id
        _require_private_directory(assignment_dir, create=True)
        command_dir = assignment_dir / "commands"
        _require_private_directory(command_dir, create=True)
        return command_dir / f"{operation_id}.json"

    @staticmethod
    def _decode(value: dict[str, Any]) -> JournalEntry:
        if value.get("schema") != JOURNAL_SCHEMA:
            raise CheckpointV2JournalError("checkpoint journal schema is unsupported")
        assignment_id = _validate_identifier(
            value.get("assignment_id"), "assignment_id",
        )
        operation_id = _validate_identifier(value.get("operation_id"), "operation_id")
        command = value.get("command")
        request_sha256 = value.get("request_sha256")
        payload = value.get("payload")
        state = value.get("state")
        if not isinstance(command, str) or not command:
            raise CheckpointV2JournalError("checkpoint journal command is invalid")
        if not isinstance(request_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", request_sha256,
        ) is None:
            raise CheckpointV2JournalError("checkpoint request digest is invalid")
        if not isinstance(payload, dict) or state not in _STATES:
            raise CheckpointV2JournalError("checkpoint journal state is invalid")
        if _request_sha256(command, payload) != request_sha256:
            raise CheckpointV2JournalError("checkpoint journal request digest changed")
        response = value.get("response")
        if response is not None and not isinstance(response, dict):
            raise CheckpointV2JournalError("checkpoint journal response is invalid")
        if response is not None:
            _canonical_bytes(response, label="checkpoint command response")
        return JournalEntry(
            assignment_id=assignment_id,
            operation_id=operation_id,
            command=command,
            request_sha256=request_sha256,
            payload=dict(payload),
            state=state,
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
            response=dict(response) if response is not None else None,
            error_status=(
                value.get("error_status")
                if isinstance(value.get("error_status"), int) else None
            ),
            error_code=(
                value.get("error_code")
                if isinstance(value.get("error_code"), str) else None
            ),
        )

    @staticmethod
    def _encode(entry: JournalEntry) -> bytes:
        value = {
            "schema": JOURNAL_SCHEMA,
            "assignment_id": entry.assignment_id,
            "operation_id": entry.operation_id,
            "command": entry.command,
            "request_sha256": entry.request_sha256,
            "payload": entry.payload,
            "state": entry.state,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "response": entry.response,
            "error_status": entry.error_status,
            "error_code": entry.error_code,
        }
        return _canonical_bytes(value, label="checkpoint journal entry")

    def begin(self, command: str, payload: Mapping[str, Any]) -> JournalEntry:
        if not isinstance(command, str) or not command:
            raise CheckpointV2JournalError("checkpoint command is invalid")
        materialized = dict(payload)
        assignment_id = _validate_identifier(
            materialized.get("assignment_id"), "assignment_id",
        )
        operation_id = _validate_identifier(
            materialized.get("operation_id"), "operation_id",
        )
        request_sha256 = _request_sha256(command, materialized)
        path = self._entry_path(assignment_id, operation_id)
        with self._command_lock(assignment_id):
            if path.exists():
                existing = self._decode(_read_entry(path))
                if (
                    existing.command != command
                    or existing.request_sha256 != request_sha256
                ):
                    raise CheckpointV2OperationConflict(
                        "operation_id belongs to a different checkpoint command"
                    )
                return existing
            now = _now_iso()
            entry = JournalEntry(
                assignment_id=assignment_id,
                operation_id=operation_id,
                command=command,
                request_sha256=request_sha256,
                payload=materialized,
                state="PENDING",
                created_at=now,
                updated_at=now,
            )
            _atomic_write(path, self._encode(entry))
            return entry

    def _transition(
        self,
        current: JournalEntry,
        *,
        state: str,
        response: Mapping[str, Any] | None = None,
        error_status: int | None = None,
        error_code: str | None = None,
    ) -> JournalEntry:
        if state not in _STATES:
            raise CheckpointV2JournalError("checkpoint journal transition is invalid")
        path = self._entry_path(current.assignment_id, current.operation_id)
        with self._command_lock(current.assignment_id):
            latest = self._decode(_read_entry(path))
            if (
                latest.command != current.command
                or latest.request_sha256 != current.request_sha256
            ):
                raise CheckpointV2OperationConflict(
                    "checkpoint command changed during journal transition"
                )
            materialized_response = dict(response) if response is not None else None
            if materialized_response is not None:
                _canonical_bytes(
                    materialized_response, label="checkpoint command response",
                )
            if latest.state != "PENDING":
                exact_replay = (
                    latest.state == state
                    and latest.response == materialized_response
                    and latest.error_status == error_status
                    and latest.error_code == error_code
                )
                if exact_replay:
                    return latest
                raise CheckpointV2OperationConflict(
                    "checkpoint command already has a different terminal result"
                )
            updated = JournalEntry(
                assignment_id=latest.assignment_id,
                operation_id=latest.operation_id,
                command=latest.command,
                request_sha256=latest.request_sha256,
                payload=latest.payload,
                state=state,
                created_at=latest.created_at,
                updated_at=_now_iso(),
                response=materialized_response,
                error_status=error_status,
                error_code=error_code,
            )
            _atomic_write(path, self._encode(updated))
            return updated

    @staticmethod
    def _validate_response(
        entry: JournalEntry, response: Mapping[str, Any],
    ) -> None:
        if response.get("assignment_id") != entry.assignment_id:
            raise CheckpointV2ProtocolError(
                "checkpoint response assignment identity does not match"
            )
        if response.get("ok") is not True:
            raise CheckpointV2ProtocolError("checkpoint response is not acknowledged")
        if entry.command in _PAID_AUTHORIZE_COMMANDS:
            if response.get("paid_execution_authorized") is not True:
                raise CheckpointV2ProtocolError(
                    "paid execution was not explicitly authorized"
                )
        elif entry.command in _FREE_ONLY_COMMANDS:
            if response.get("paid_execution_authorized") is not False:
                raise CheckpointV2ProtocolError(
                    "free-only checkpoint command returned paid authorization"
                )

    def execute(
        self,
        api: ApiClient,
        command: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        entry = self.begin(command, payload)
        if entry.state == "ACKNOWLEDGED" and entry.response is not None:
            return dict(entry.response)
        if entry.state == "REJECTED":
            raise CheckpointV2CommandRejected(
                entry.error_status, entry.error_code,
            )
        try:
            response = api.checkpoint_v2_command(command, entry.payload)
        except ApiError as exc:
            # Ambiguous transport/server failures remain PENDING.  A restart
            # safely resends the exact operation id and lets the server ledger
            # decide whether this is a replay.
            if (
                exc.status_code is not None
                and exc.status_code != 429
                and exc.status_code < 500
            ):
                self._transition(
                    entry,
                    state="REJECTED",
                    error_status=exc.status_code,
                    error_code=exc.code,
                )
            raise
        self._validate_response(entry, response)
        acknowledged = self._transition(
            entry, state="ACKNOWLEDGED", response=response,
        )
        return dict(acknowledged.response or {})

    def load(self, assignment_id: str, operation_id: str) -> JournalEntry:
        path = self._entry_path(assignment_id, operation_id)
        with self._command_lock(assignment_id):
            return self._decode(_read_entry(path))


def finalize_execution_identity_v2(
    assignment: Mapping[str, Any],
    *,
    api: ApiClient,
    journal: CheckpointV2Journal,
    harness: str,
    provider: str,
    agent_version: str,
    runtime_profile: str,
    model_config_version: str,
    checkpoint_abi: str,
    runtime_compatibility_digest: str,
    operation_id: str | None = None,
) -> FinalizedIdentityReceiptV2:
    """Freeze a runtime identity without acquiring assignment ownership.

    OBSERVE and RESTORE_TEST assignments intentionally retain the legacy
    ownership protocol.  They still need one immutable identity before a
    shadow archive can be trusted.  This handshake is journaled like every V2
    command, but its response must explicitly deny paid execution and affirm
    that assignment ownership was not changed.
    """

    rollout_mode = assignment.get("checkpoint_v2_rollout_mode", "off")
    if (
        assignment.get("checkpoint_v2_identity_protocol_version") != 2
        or rollout_mode not in {"observe", "restore_test", "canary", "on"}
    ):
        raise CheckpointV2ProtocolError(
            "assignment does not authorize checkpoint identity protocol v2"
        )
    assignment_id = _validate_identifier(
        assignment.get("assignment_id"), "assignment_id",
    )
    ownership_protocol = assignment.get("checkpoint_protocol_version")
    if (
        not isinstance(ownership_protocol, int)
        or isinstance(ownership_protocol, bool)
        or ownership_protocol not in {1, 2}
    ):
        raise CheckpointV2ProtocolError(
            "assignment checkpoint ownership protocol is invalid"
        )
    provisional = assignment.get("execution_identity")
    if (
        not isinstance(provisional, dict)
        or provisional.get("identity_state") not in {"PROVISIONAL", "FINAL"}
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint v2 execution identity is not finalizable"
        )
    model = provisional.get("model")
    effort = provisional.get("effort")
    if not isinstance(model, str) or not model or not isinstance(effort, str) or not effort:
        raise CheckpointV2ProtocolError(
            "checkpoint v2 claim identity is incomplete"
        )
    payload = {
        "assignment_id": assignment_id,
        "operation_id": operation_id or new_operation_id(),
        "harness": harness,
        "provider": provider,
        "model": model,
        "effort": effort,
        "agent_version": agent_version,
        "runtime_profile": runtime_profile,
        "model_config_version": model_config_version,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": checkpoint_abi,
        "runtime_compatibility_digest": runtime_compatibility_digest,
    }
    response = journal.execute(api, "identity/finalize", payload)
    if (
        response.get("identity_state") != "FINAL"
        or response.get("checkpoint_protocol_version") != ownership_protocol
        or response.get("checkpoint_v2_identity_protocol_version") != 2
        or response.get("checkpoint_core_abi") != CHECKPOINT_CORE_ABI_V2
        or response.get("checkpoint_abi") != checkpoint_abi
        or response.get("runtime_compatibility_digest")
        != runtime_compatibility_digest
        or response.get("assignment_ownership_unchanged") is not True
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint identity finalization response is inconsistent"
        )

    finalized_assignment = dict(assignment)
    finalized_assignment["execution_identity"] = {
        **provisional,
        "harness": harness,
        "provider": provider,
        "model": model,
        "effort": effort,
        "agent_version": agent_version,
        "runtime_profile": runtime_profile,
        "model_config_version": model_config_version,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": checkpoint_abi,
        "runtime_compatibility_digest": runtime_compatibility_digest,
        "identity_state": "FINAL",
    }
    identity = ExecutionIdentityV2.from_assignment(finalized_assignment)
    if response.get("identity_fingerprint") != identity.fingerprint:
        raise CheckpointV2ProtocolError(
            "checkpoint identity fingerprint differs from the server"
        )
    return FinalizedIdentityReceiptV2(
        identity=identity,
        checkpoint_protocol_version=ownership_protocol,
    )


class CheckpointV2StateMachine:
    """Typed client-side façade over the server's authoritative transitions.

    Harness code receives either a free preparation/restore permit or an
    explicit paid permit.  It never infers authorization from HTTP success,
    assignment shape, or a locally present snapshot.
    """

    def __init__(
        self,
        assignment: Mapping[str, Any],
        *,
        api: ApiClient,
        journal: CheckpointV2Journal,
        activation: CheckpointActivationV2,
    ):
        self.identity = ExecutionIdentityV2.from_assignment(assignment)
        rollout_mode = assignment.get("checkpoint_v2_rollout_mode")
        ownership_protocol = assignment.get("checkpoint_protocol_version")
        if (
            ownership_protocol not in {1, 2}
            or rollout_mode not in {"canary", "on"}
            or activation.server_mode.wire_value != rollout_mode
            or not activation.authoritative
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint v2 state authority is not enabled for this assignment"
            )
        self.policy = checkpoint_policy_v2(self.identity)
        self.activation = activation
        self.api = api
        self.journal = journal
        self.ownership_protocol = ownership_protocol
        self.certification_id: str | None = None
        self.certification_digest: str | None = None
        owner_epoch = assignment.get("owner_epoch", 0)
        if not isinstance(owner_epoch, int) or isinstance(owner_epoch, bool):
            raise CheckpointV2ProtocolError("assignment owner epoch is invalid")
        self.initial_owner_epoch = owner_epoch

    def _payload(
        self,
        *,
        session_id: str,
        expected_owner_epoch: int,
        operation_id: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _validate_session_identifier(session_id)
        if (
            not isinstance(expected_owner_epoch, int)
            or isinstance(expected_owner_epoch, bool)
            or expected_owner_epoch < 0
        ):
            raise CheckpointV2ProtocolError("expected owner epoch is invalid")
        payload = {
            "assignment_id": self.identity.assignment_id,
            "operation_id": operation_id or new_operation_id(),
            "session_id": session_id,
            "expected_owner_epoch": expected_owner_epoch,
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def _assert_permit(self, permit: FreePreparationPermit) -> None:
        if (
            permit.assignment_id != self.identity.assignment_id
            or permit.identity_fingerprint != self.identity.fingerprint
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint permit belongs to a different execution identity"
            )

    @staticmethod
    def _owner_facts(
        response: Mapping[str, Any], *, expected_state: str,
    ) -> tuple[str, int, str]:
        session_id = response.get("owner_session_id")
        owner_epoch = response.get("owner_epoch")
        lease_expiry = response.get("owner_lease_expires_at")
        if (
            not isinstance(session_id, str)
            or not isinstance(owner_epoch, int)
            or isinstance(owner_epoch, bool)
            or not isinstance(lease_expiry, str)
            or not lease_expiry
            or response.get("execution_state") != expected_state
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint response has invalid owner facts"
            )
        return session_id, owner_epoch, lease_expiry

    @staticmethod
    def _usage_segment_id(response: Mapping[str, Any]) -> str:
        usage_segment_id = response.get("usage_segment_id")
        if (
            not isinstance(usage_segment_id, str)
            or _IDENTIFIER_RE.fullmatch(usage_segment_id) is None
            or response.get("usage_schema") != USAGE_SEGMENT_SCHEMA_V2
        ):
            raise CheckpointV2ProtocolError(
                "paid authorization omitted its usage segment identity"
            )
        return usage_segment_id

    @staticmethod
    def _certification_facts(
        response: Mapping[str, Any],
    ) -> tuple[str, str]:
        certification_id = response.get("certification_id")
        certification_digest = response.get("certification_digest")
        if (
            not isinstance(certification_id, str)
            or _IDENTIFIER_RE.fullmatch(certification_id) is None
            or not isinstance(certification_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", certification_digest) is None
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint paid owner omitted its exact certification"
            )
        return certification_id, certification_digest

    def _remember_certification(self, response: Mapping[str, Any]) -> None:
        certification_id, certification_digest = self._certification_facts(
            response,
        )
        if self.certification_id is not None and (
            self.certification_id != certification_id
            or self.certification_digest != certification_digest
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint certification changed within one owner epoch"
            )
        self.certification_id = certification_id
        self.certification_digest = certification_digest

    @staticmethod
    def _usage_ledger_facts(
        response: Mapping[str, Any],
    ) -> tuple[str, bool]:
        ledger = response.get("usage_ledger")
        if not isinstance(ledger, dict):
            raise CheckpointV2ProtocolError("usage ledger acknowledgement is missing")
        counts = (
            ledger.get("usage_segment_count"),
            ledger.get("finalized_segment_count"),
            ledger.get("open_segment_count"),
        )
        complete = ledger.get("complete")
        digest = ledger.get("ledger_sha256")
        if (
            ledger.get("schema") != "dradar-checkpoint-usage-ledger-v2"
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in counts
            )
            or counts[1] + counts[2] != counts[0]
            or not isinstance(complete, bool)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or (
                complete
                and (
                    counts[0] == 0
                    or counts[1] != counts[0]
                    or counts[2] != 0
                    or any(
                        not isinstance(ledger.get(name), int)
                        or isinstance(ledger.get(name), bool)
                        or ledger.get(name) < 0
                        for name in (
                            "n_input_tokens",
                            "n_cache_tokens",
                            "n_output_tokens",
                        )
                    )
                    or ledger.get("n_cache_tokens")
                    > ledger.get("n_input_tokens")
                )
            )
            or (
                not complete
                and any(
                    ledger.get(name) is not None
                    for name in (
                        "n_input_tokens",
                        "n_cache_tokens",
                        "n_output_tokens",
                    )
                )
            )
        ):
            raise CheckpointV2ProtocolError(
                "usage ledger acknowledgement is inconsistent"
            )
        return digest, complete

    def checkout(
        self,
        *,
        session_id: str,
        expected_owner_epoch: int | None = None,
        operation_id: str | None = None,
    ) -> FreePreparationPermit:
        if not self.policy.supported:
            raise CheckpointV2ProtocolError(
                "this Harness has no checkpoint v2 writer"
            )
        payload = self._payload(
            session_id=session_id,
            expected_owner_epoch=(
                self.initial_owner_epoch
                if expected_owner_epoch is None else expected_owner_epoch
            ),
            operation_id=operation_id,
        )
        response = self.journal.execute(self.api, "checkout", payload)
        if response.get("checkpoint_v2_authoritative_activated") is False:
            if (
                response.get("checkpoint_protocol_version") != 1
                or response.get("fallback_to_ordinary") is not True
                or response.get("reason")
                != "checkpoint_v2_cohort_not_certified"
                or response.get("assignment_unchanged") is not True
                or response.get("paid_execution_authorized") is not False
                or response.get("fallback_observation_mode") != "observe"
                or response.get("certification_id") is not None
                or response.get("certification_digest") is not None
            ):
                raise CheckpointV2ProtocolError(
                    "checkpoint ordinary fallback response is inconsistent"
                )
            raise CheckpointV2OrdinaryFallback("observe")
        if (
            response.get("checkpoint_v2_authoritative_activated") is not True
            or response.get("checkpoint_protocol_version") != 2
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint checkout did not activate authoritative ownership"
            )
        self._remember_certification(response)
        owner_session, owner_epoch, expiry = self._owner_facts(
            response, expected_state="preparing",
        )
        if owner_session != session_id:
            raise CheckpointV2ProtocolError(
                "checkpoint owner response changed the runner session"
            )
        return FreePreparationPermit(
            assignment_id=self.identity.assignment_id,
            identity_fingerprint=self.identity.fingerprint,
            session_id=owner_session,
            owner_epoch=owner_epoch,
            owner_lease_expires_at=expiry,
        )

    def start_paid(
        self,
        permit: FreePreparationPermit,
        *,
        operation_id: str | None = None,
    ) -> PaidExecutionPermit:
        self._assert_permit(permit)
        if isinstance(permit, OfflineRestorePermit):
            raise CheckpointV2ProtocolError(
                "offline restore requires resume-commit, not fresh start"
            )
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
        )
        response = self.journal.execute(self.api, "start", payload)
        self._remember_certification(response)
        session_id, owner_epoch, expiry = self._owner_facts(
            response, expected_state="running",
        )
        if session_id != permit.session_id or owner_epoch != permit.owner_epoch:
            raise CheckpointV2ProtocolError("fresh owner changed during paid start")
        usage_segment_id = self._usage_segment_id(response)
        return PaidExecutionPermit(
            assignment_id=self.identity.assignment_id,
            identity_fingerprint=self.identity.fingerprint,
            session_id=session_id,
            owner_epoch=owner_epoch,
            owner_lease_expires_at=expiry,
            source="fresh",
            usage_segment_id=usage_segment_id,
        )

    def renew_owner(
        self,
        permit: FreePreparationPermit,
        *,
        operation_id: str | None = None,
    ) -> FreePreparationPermit:
        """Renew one exact owner without changing its type, epoch, or state.

        The caller must persist and keep using the returned permit: the only
        mutable fact is the authoritative lease deadline. A renewal response
        can never upgrade a free owner into paid execution.
        """

        self._assert_permit(permit)
        expected_state = (
            "running"
            if isinstance(permit, PaidExecutionPermit)
            else "resume_reserved"
            if isinstance(permit, OfflineRestorePermit)
            else "result_ready"
            if isinstance(permit, CompletedResultPermit)
            else "preparing"
        )
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
        )
        response = self.journal.execute(self.api, "renew", payload)
        owner_session, owner_epoch, expiry = self._owner_facts(
            response, expected_state=expected_state,
        )
        if (
            owner_session != permit.session_id
            or owner_epoch != permit.owner_epoch
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint owner changed during lease renewal"
            )
        return replace(permit, owner_lease_expires_at=expiry)

    def _assert_usage_receipt(
        self,
        permit: PaidExecutionPermit,
        receipt: FinalizedUsageSegmentReceiptV2 | None,
    ) -> None:
        if (
            not isinstance(receipt, FinalizedUsageSegmentReceiptV2)
            or receipt.assignment_id != permit.assignment_id
            or receipt.identity_fingerprint != permit.identity_fingerprint
            or receipt.session_id != permit.session_id
            or receipt.owner_epoch != permit.owner_epoch
            or receipt.usage_segment_id != permit.usage_segment_id
            or re.fullmatch(r"[0-9a-f]{64}", receipt.evidence_sha256) is None
            or re.fullmatch(
                r"[0-9a-f]{64}", receipt.usage_ledger_sha256,
            ) is None
        ):
            raise CheckpointV2ProtocolError(
                "paid transition lacks its exact finalized usage receipt"
            )

    def pause_sealed(
        self,
        permit: FreePreparationPermit,
        *,
        checkpoint: SealedCheckpointV2,
        usage_receipt: FinalizedUsageSegmentReceiptV2 | None = None,
        operation_id: str | None = None,
    ) -> int:
        """Publish one already-sealed generation and release its owner."""

        self._assert_permit(permit)
        if isinstance(permit, PaidExecutionPermit):
            self._assert_usage_receipt(permit, usage_receipt)
        elif usage_receipt is not None:
            raise CheckpointV2ProtocolError(
                "free preparation has no paid usage segment"
            )
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
            extra=checkpoint.protocol_fields(),
        )
        response = self.journal.execute(self.api, "pause", payload)
        owner_epoch = response.get("owner_epoch")
        if (
            not isinstance(owner_epoch, int)
            or isinstance(owner_epoch, bool)
            or owner_epoch <= permit.owner_epoch
            or response.get("execution_state") != "paused"
            or response.get("checkpoint_id") != checkpoint.checkpoint_id
            or response.get("snapshot_generation")
            != checkpoint.snapshot_generation
            or response.get("manifest_sha256") != checkpoint.manifest_sha256
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint pause did not publish the exact sealed generation"
            )
        return owner_epoch

    def finalize_usage_segment(
        self,
        permit: PaidExecutionPermit,
        *,
        evidence: UsageSegmentEvidenceV2,
        operation_id: str | None = None,
    ) -> FinalizedUsageSegmentReceiptV2:
        """Finalize exactly one paid epoch before pause or result upload."""

        self._assert_permit(permit)
        if permit.source not in {"fresh", "resume"}:
            raise CheckpointV2ProtocolError("paid usage source is invalid")
        if not isinstance(evidence, UsageSegmentEvidenceV2):
            raise CheckpointV2ProtocolError("usage segment evidence is invalid")
        allowed_evidence = {
            "codex": {"trajectory_bundle", "session_usage"},
            "dsh": {"provider_request_ledger", "session_usage"},
            KIMI_AGENT: {"provider_request_ledger", "session_usage"},
            ZCODE_AGENT: {"provider_request_ledger", "session_usage"},
        }.get(self.identity.harness, set())
        if (
            evidence.completeness != "unavailable"
            and evidence.evidence_kind not in allowed_evidence
        ):
            raise CheckpointV2ProtocolError(
                "usage evidence kind does not match the Harness"
            )
        evidence_fields = evidence.protocol_fields(permit)
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
            extra=evidence_fields,
        )
        response = self.journal.execute(
            self.api, "usage-finalize", payload,
        )
        session_id, owner_epoch, _expiry = self._owner_facts(
            response, expected_state="running",
        )
        if (
            session_id != permit.session_id
            or owner_epoch != permit.owner_epoch
            or response.get("usage_segment_id") != permit.usage_segment_id
            or response.get("usage_schema") != USAGE_SEGMENT_SCHEMA_V2
            or response.get("completeness") != evidence.completeness
            or response.get("evidence_sha256")
            != evidence_fields["evidence_sha256"]
            or response.get("assignment_unchanged") is not True
        ):
            raise CheckpointV2ProtocolError(
                "usage finalization changed its paid segment identity"
            )
        segment_usage = response.get("segment_usage")
        materialized_events = evidence_fields["events"]
        observed_count = len(materialized_events)
        incoming_totals = tuple(
            sum(int(item[name]) for item in materialized_events)
            for name in (
                "n_input_tokens", "n_cache_tokens", "n_output_tokens",
            )
        )
        if not isinstance(segment_usage, Mapping):
            raise CheckpointV2ProtocolError(
                "usage finalization omitted its de-duplicated segment facts"
            )
        segment_counts = tuple(
            segment_usage.get(name) for name in (
                "observed_event_count", "novel_event_count",
                "duplicate_event_count",
            )
        )
        segment_totals = tuple(
            segment_usage.get(name) for name in (
                "n_input_tokens", "n_cache_tokens", "n_output_tokens",
            )
        )
        segment_valid = (
            segment_usage.get("ledger_scope") == evidence.ledger_scope
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in segment_counts
            )
            and segment_counts[0] == observed_count
            and segment_counts[1] + segment_counts[2] == segment_counts[0]
        )
        if evidence.completeness == "unavailable":
            segment_valid = (
                segment_valid
                and segment_counts == (0, 0, 0)
                and all(value is None for value in segment_totals)
            )
        else:
            segment_valid = (
                segment_valid
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value <= incoming_totals[index]
                    for index, value in enumerate(segment_totals)
                )
                and segment_totals[1] <= segment_totals[0]
            )
            if evidence.ledger_scope == "segment_delta":
                segment_valid = (
                    segment_valid
                    and segment_counts == (
                        observed_count, observed_count, 0,
                    )
                    and segment_totals == incoming_totals
                )
        if not segment_valid:
            raise CheckpointV2ProtocolError(
                "usage finalization de-duplication acknowledgement is inconsistent"
            )
        ledger_sha256, ledger_complete = self._usage_ledger_facts(response)
        return FinalizedUsageSegmentReceiptV2(
            assignment_id=self.identity.assignment_id,
            identity_fingerprint=self.identity.fingerprint,
            session_id=permit.session_id,
            owner_epoch=permit.owner_epoch,
            usage_segment_id=permit.usage_segment_id,
            completeness=evidence.completeness,
            evidence_sha256=evidence_fields["evidence_sha256"],
            usage_ledger_sha256=ledger_sha256,
            usage_ledger_complete=ledger_complete,
        )

    def report_failure(
        self,
        permit: FreePreparationPermit,
        *,
        failure: CheckpointFailureV2,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a typed failure; no free-form log or guessed layer is sent."""

        self._assert_permit(permit)
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
            extra=failure.protocol_fields(),
        )
        response = self.journal.execute(self.api, "failure", payload)
        owner_epoch = response.get("owner_epoch")
        if (
            not isinstance(owner_epoch, int)
            or isinstance(owner_epoch, bool)
            or owner_epoch < permit.owner_epoch
            or response.get("assignment_invalidated") is not False
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint failure response violated recovery invariants"
            )
        return dict(response)

    def reserve_offline_restore(
        self,
        *,
        session_id: str,
        expected_owner_epoch: int,
        checkpoint_id: str,
        snapshot_generation: int,
        manifest_sha256: str,
        requester_machine_fingerprint: str,
        operation_id: str | None = None,
    ) -> OfflineRestorePermit:
        _validate_identifier(checkpoint_id, "checkpoint_id")
        if (
            not isinstance(snapshot_generation, int)
            or isinstance(snapshot_generation, bool)
            or snapshot_generation < 0
        ):
            raise CheckpointV2ProtocolError("snapshot generation is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
            raise CheckpointV2ProtocolError("checkpoint manifest digest is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", requester_machine_fingerprint) is None:
            raise CheckpointV2ProtocolError("checkpoint machine fingerprint is invalid")
        payload = self._payload(
            session_id=session_id,
            expected_owner_epoch=expected_owner_epoch,
            operation_id=operation_id,
            extra={
                "checkpoint_id": checkpoint_id,
                "snapshot_generation": snapshot_generation,
                "manifest_sha256": manifest_sha256,
                "requester_machine_fingerprint": requester_machine_fingerprint,
            },
        )
        response = self.journal.execute(self.api, "resume-reserve", payload)
        self._remember_certification(response)
        owner_session, owner_epoch, expiry = self._owner_facts(
            response, expected_state="resume_reserved",
        )
        if (
            owner_session != session_id
            or response.get("checkpoint_id") != checkpoint_id
            or response.get("snapshot_generation") != snapshot_generation
            or response.get("checkpoint_core_abi")
            != self.identity.checkpoint_core_abi
            or response.get("checkpoint_abi") != self.identity.checkpoint_abi
            or response.get("requester_machine_fingerprint")
            != requester_machine_fingerprint
        ):
            raise CheckpointV2ProtocolError(
                "offline restore reservation changed checkpoint identity"
            )
        compatibility_fingerprint = response.get("compatibility_fingerprint")
        reservation_nonce = response.get("reservation_nonce")
        if (
            not isinstance(compatibility_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", compatibility_fingerprint) is None
            or not isinstance(reservation_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", reservation_nonce) is None
        ):
            raise CheckpointV2ProtocolError(
                "offline restore reservation omitted binding facts"
            )
        return OfflineRestorePermit(
            assignment_id=self.identity.assignment_id,
            identity_fingerprint=self.identity.fingerprint,
            session_id=owner_session,
            owner_epoch=owner_epoch,
            owner_lease_expires_at=expiry,
            checkpoint_id=checkpoint_id,
            snapshot_generation=snapshot_generation,
            manifest_sha256=manifest_sha256,
            checkpoint_core_abi=self.identity.checkpoint_core_abi,
            checkpoint_abi=self.identity.checkpoint_abi,
            compatibility_fingerprint=compatibility_fingerprint,
            requester_machine_fingerprint=requester_machine_fingerprint,
            reservation_nonce=reservation_nonce,
        )

    def fallback_to_ordinary_fresh(
        self,
        *,
        session_id: str,
        expected_owner_epoch: int,
        reason: str,
        operation_id: str | None = None,
    ) -> CheckpointV2OrdinaryFallback:
        """Fence recovery and reopen the same lease on ordinary protocol V1."""

        if reason not in {
            "cohort_not_certified",
            "selected_descriptor_invalid",
            "snapshot_unreachable",
            "archive_invalid",
            "restore_preflight_failed",
            "paid_gate_intent_failed",
            "operator_disable",
        }:
            raise CheckpointV2ProtocolError(
                "checkpoint fresh fallback reason is invalid"
            )
        payload = self._payload(
            session_id=session_id,
            expected_owner_epoch=expected_owner_epoch,
            operation_id=operation_id,
            extra={"reason": reason},
        )
        response = self.journal.execute(
            self.api, "fresh-fallback", payload,
        )
        owner_epoch = response.get("owner_epoch")
        resume_generation = response.get("resume_generation")
        if (
            response.get("checkpoint_v2_authoritative_activated") is not False
            or response.get("checkpoint_protocol_version") != 1
            or response.get("fallback_to_ordinary") is not True
            or response.get("fallback_observation_mode") != "observe"
            or response.get("reason") != reason
            or response.get("assignment_restarted_fresh") is not True
            or response.get("assignment_unchanged") is not False
            or response.get("execution_state") != "waiting"
            or response.get("checkpoint_evidence_retained") is not True
            or response.get("paid_execution_authorized") is not False
            or not isinstance(owner_epoch, int)
            or isinstance(owner_epoch, bool)
            or owner_epoch <= expected_owner_epoch
            or not isinstance(resume_generation, int)
            or isinstance(resume_generation, bool)
            or resume_generation < 1
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint fresh fallback acknowledgement is inconsistent"
            )
        return CheckpointV2OrdinaryFallback(
            "observe",
            assignment_restarted_fresh=True,
            owner_epoch=owner_epoch,
            resume_generation=resume_generation,
            reason=reason,
        )

    def reconcile_paid_gate(
        self,
        *,
        session_id: str,
        expected_owner_epoch: int,
        attempted_operation_id: str | None,
        attempted_command: str,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve an orphaned paid gate without replaying paid authority."""

        if attempted_command not in {"start", "resume-commit"}:
            raise CheckpointV2ProtocolError(
                "checkpoint paid-gate attempted command is invalid"
            )
        if attempted_operation_id is not None:
            _validate_identifier(
                attempted_operation_id, "attempted_operation_id",
            )
        payload = self._payload(
            session_id=session_id,
            expected_owner_epoch=expected_owner_epoch,
            operation_id=operation_id,
            extra={
                "attempted_operation_id": attempted_operation_id,
                "attempted_command": attempted_command,
            },
        )
        response = self.journal.execute(
            self.api, "paid-gate-reconcile", payload,
        )
        owner_epoch = response.get("owner_epoch")
        resume_generation = response.get("resume_generation")
        outcome = response.get("outcome")
        if (
            response.get("attempted_operation_id")
            != attempted_operation_id
            or response.get("attempted_command") != attempted_command
            or response.get("paid_execution_authorized") is not False
            or response.get("assignment_invalidated") is not False
            or response.get("checkpoint_evidence_retained") is not True
            or not isinstance(owner_epoch, int)
            or isinstance(owner_epoch, bool)
            or not isinstance(resume_generation, int)
            or isinstance(resume_generation, bool)
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint paid-gate reconciliation response is inconsistent"
            )
        if outcome == "fresh_fallback":
            valid = (
                response.get("paid_command_committed") is False
                and response.get("attempted_operation_matched") is False
                and response.get("checkpoint_protocol_version") == 1
                and response.get("assignment_restarted_fresh") is True
                and response.get("assignment_unchanged") is False
                and response.get("execution_state") == "waiting"
                and owner_epoch > expected_owner_epoch
            )
        elif outcome == "faulted":
            valid = (
                isinstance(response.get("paid_command_committed"), bool)
                and isinstance(response.get("attempted_operation_matched"), bool)
                and response.get("checkpoint_protocol_version") == 2
                and response.get("assignment_restarted_fresh") is False
                and response.get("assignment_unchanged") is False
                and response.get("execution_state") == "faulted"
                and owner_epoch > expected_owner_epoch
                and isinstance(response.get("circuit"), Mapping)
            )
        elif outcome == "completed_result_preserved":
            valid = (
                response.get("paid_command_committed") is True
                and response.get("checkpoint_protocol_version") == 2
                and response.get("assignment_restarted_fresh") is False
                and response.get("assignment_unchanged") is True
                and response.get("execution_state") == "result_ready"
                and owner_epoch == expected_owner_epoch
            )
        else:
            valid = False
        if not valid:
            raise CheckpointV2ProtocolError(
                "checkpoint paid-gate reconciliation outcome is invalid"
            )
        return dict(response)

    def declare_result_ready(
        self,
        permit: PaidExecutionPermit,
        *,
        usage_receipt: FinalizedUsageSegmentReceiptV2,
        upload_intent_id: str,
        operation_id: str | None = None,
    ) -> CompletedResultPermit:
        """End paid model execution and bind one completed upload identity."""

        self._assert_permit(permit)
        self._assert_usage_receipt(permit, usage_receipt)
        if re.fullmatch(r"[0-9a-f]{64}", upload_intent_id) is None:
            raise CheckpointV2ProtocolError("result upload intent digest is invalid")
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
            extra={
                "upload_intent_id": upload_intent_id,
                "intent_version": CHECKPOINT_V2_UPLOAD_INTENT_VERSION,
            },
        )
        response = self.journal.execute(self.api, "result-ready", payload)
        session_id, owner_epoch, expiry = self._owner_facts(
            response, expected_state="result_ready",
        )
        if (
            session_id != permit.session_id
            or owner_epoch != permit.owner_epoch
            or response.get("upload_intent_id") != upload_intent_id
        ):
            raise CheckpointV2ProtocolError(
                "result-ready changed owner or content identity"
            )
        ledger_sha256, _ledger_complete = self._usage_ledger_facts(response)
        if ledger_sha256 != usage_receipt.usage_ledger_sha256:
            raise CheckpointV2ProtocolError(
                "result-ready changed the finalized usage ledger"
            )
        return CompletedResultPermit(
            assignment_id=self.identity.assignment_id,
            identity_fingerprint=self.identity.fingerprint,
            session_id=session_id,
            owner_epoch=owner_epoch,
            owner_lease_expires_at=expiry,
            upload_intent_id=upload_intent_id,
            usage_ledger_sha256=ledger_sha256,
        )

    def commit_paid_resume(
        self,
        permit: OfflineRestorePermit,
        *,
        receipt: RestoreReceiptV2,
        operation_id: str | None = None,
    ) -> PaidExecutionPermit:
        self._assert_permit(permit)
        expected_receipt = completed_restore_receipt(
            permit,
            restore_adapter_version=receipt.restore_adapter_version,
            restored_manifest_sha256=receipt.manifest_sha256,
        )
        if receipt != expected_receipt:
            raise CheckpointV2ProtocolError(
                "restore receipt does not belong to this exact reservation"
            )
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
            extra={
                "checkpoint_id": permit.checkpoint_id,
                "snapshot_generation": permit.snapshot_generation,
                "manifest_sha256": permit.manifest_sha256,
                "requester_machine_fingerprint": (
                    permit.requester_machine_fingerprint
                ),
                "reservation_nonce": permit.reservation_nonce,
                "restore_adapter_version": receipt.restore_adapter_version,
                "restore_receipt_sha256": receipt.receipt_sha256,
            },
        )
        response = self.journal.execute(self.api, "resume-commit", payload)
        self._remember_certification(response)
        session_id, owner_epoch, expiry = self._owner_facts(
            response, expected_state="running",
        )
        if (
            session_id != permit.session_id
            or owner_epoch != permit.owner_epoch
            or response.get("checkpoint_id") != permit.checkpoint_id
            or response.get("snapshot_generation") != permit.snapshot_generation
            or response.get("restore_receipt_sha256") != receipt.receipt_sha256
        ):
            raise CheckpointV2ProtocolError(
                "resume commit changed owner or checkpoint identity"
            )
        usage_segment_id = self._usage_segment_id(response)
        return PaidExecutionPermit(
            assignment_id=self.identity.assignment_id,
            identity_fingerprint=self.identity.fingerprint,
            session_id=session_id,
            owner_epoch=owner_epoch,
            owner_lease_expires_at=expiry,
            source="resume",
            usage_segment_id=usage_segment_id,
        )

    def abort_offline_restore(
        self,
        permit: OfflineRestorePermit,
        *,
        operation_id: str | None = None,
    ) -> int:
        self._assert_permit(permit)
        payload = self._payload(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            operation_id=operation_id,
            extra={
                "checkpoint_id": permit.checkpoint_id,
                "snapshot_generation": permit.snapshot_generation,
                "manifest_sha256": permit.manifest_sha256,
                "requester_machine_fingerprint": (
                    permit.requester_machine_fingerprint
                ),
            },
        )
        response = self.journal.execute(self.api, "resume-abort", payload)
        owner_epoch = response.get("owner_epoch")
        if (
            not isinstance(owner_epoch, int)
            or isinstance(owner_epoch, bool)
            or owner_epoch <= permit.owner_epoch
            or response.get("execution_state") != "paused"
            or response.get("checkpoint_id") != permit.checkpoint_id
        ):
            raise CheckpointV2ProtocolError(
                "offline restore abort did not return to the exact pause"
            )
        return owner_epoch

    def acknowledge_retention(
        self,
        *,
        owner_epoch_observed: int,
        generations: tuple[SealedCheckpointV2, ...],
        upload_intent_id: str | None = None,
        operation_id: str | None = None,
    ) -> CheckpointRetentionAcknowledgementV2:
        """Obtain an exact server decision before deleting local evidence."""

        if (
            not isinstance(owner_epoch_observed, int)
            or isinstance(owner_epoch_observed, bool)
            or owner_epoch_observed < 0
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint retention owner epoch is invalid"
            )
        if not isinstance(generations, tuple) or len(generations) > 64:
            raise CheckpointV2ProtocolError(
                "checkpoint retention inventory is invalid"
            )
        inventory = []
        keys: set[tuple[str, int, str]] = set()
        for checkpoint in generations:
            if not isinstance(checkpoint, SealedCheckpointV2):
                raise CheckpointV2ProtocolError(
                    "checkpoint retention generation is invalid"
                )
            fields = checkpoint.protocol_fields()
            key = (
                fields["checkpoint_id"],
                fields["snapshot_generation"],
                fields["manifest_sha256"],
            )
            if key in keys:
                raise CheckpointV2ProtocolError(
                    "checkpoint retention inventory has a duplicate"
                )
            keys.add(key)
            inventory.append({
                "checkpoint_id": key[0],
                "snapshot_generation": key[1],
                "manifest_sha256": key[2],
            })
        if upload_intent_id is not None and re.fullmatch(
            r"[0-9a-f]{64}", upload_intent_id,
        ) is None:
            raise CheckpointV2ProtocolError(
                "checkpoint result evidence identity is invalid"
            )
        payload = {
            "assignment_id": self.identity.assignment_id,
            "operation_id": operation_id or new_operation_id(),
            "owner_epoch_observed": owner_epoch_observed,
            "generations": inventory,
            "upload_intent_id": upload_intent_id,
        }
        response = self.journal.execute(self.api, "retention", payload)

        def response_generations(
            name: str,
        ) -> tuple[CheckpointGenerationRefV2, ...]:
            raw = response.get(name)
            if not isinstance(raw, list):
                raise CheckpointV2ProtocolError(
                    "checkpoint retention response inventory is invalid"
                )
            observed: set[tuple[str, int, str]] = set()
            materialized: list[CheckpointGenerationRefV2] = []
            for item in raw:
                if not isinstance(item, dict):
                    raise CheckpointV2ProtocolError(
                        "checkpoint retention response generation is invalid"
                    )
                key = (
                    item.get("checkpoint_id"),
                    item.get("snapshot_generation"),
                    item.get("manifest_sha256"),
                )
                if (
                    not isinstance(key[0], str)
                    or _IDENTIFIER_RE.fullmatch(key[0]) is None
                    or not isinstance(key[1], int)
                    or isinstance(key[1], bool)
                    or key[1] < 0
                    or not isinstance(key[2], str)
                    or re.fullmatch(r"[0-9a-f]{64}", key[2]) is None
                    or key in observed
                ):
                    raise CheckpointV2ProtocolError(
                        "checkpoint retention response generation is invalid"
                    )
                observed.add(key)
                materialized.append(CheckpointGenerationRefV2(
                    checkpoint_id=key[0],
                    snapshot_generation=key[1],
                    manifest_sha256=key[2],
                ))
            return tuple(materialized)

        released = response_generations("delete_generations")
        retained = response_generations("retain_generations")
        released_keys = {item.key for item in released}
        retained_keys = {item.key for item in retained}
        current_owner_epoch = response.get("current_owner_epoch")
        result_evidence_release = response.get("result_evidence_release")
        submission_id = response.get("submission_id")
        if (
            released_keys & retained_keys
            or released_keys | retained_keys != keys
            or response.get("operation_id") != payload["operation_id"]
            or response.get("assignment_unchanged") is not True
            or not isinstance(current_owner_epoch, int)
            or isinstance(current_owner_epoch, bool)
            or current_owner_epoch < 0
            or response.get("owner_epoch_observed") != owner_epoch_observed
            or result_evidence_release is not (upload_intent_id is not None)
            or (
                result_evidence_release is True
                and (
                    not isinstance(submission_id, str)
                    or _IDENTIFIER_RE.fullmatch(submission_id) is None
                )
            )
            or (result_evidence_release is False and submission_id is not None)
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint retention response changed evidence identity"
            )
        return CheckpointRetentionAcknowledgementV2(
            assignment_id=self.identity.assignment_id,
            operation_id=payload["operation_id"],
            owner_epoch_observed=owner_epoch_observed,
            current_owner_epoch=current_owner_epoch,
            delete_generations=released,
            retain_generations=retained,
            result_evidence_release=result_evidence_release,
            upload_intent_id=upload_intent_id,
            submission_id=submission_id,
        )


def acknowledge_completed_result_retention_v2(
    *,
    assignment_id: str,
    owner_epoch_observed: int,
    upload_intent_id: str,
    operation_id: str,
    api: ApiClient,
    journal: CheckpointV2Journal,
) -> CheckpointRetentionAcknowledgementV2:
    """Replay result-only retention after the live owner process is gone.

    Checkpoint generations are intentionally omitted and therefore remain on
    disk. This narrow recovery primitive releases only the exact completed
    upload bytes after the server proves that their content-bound intent was
    consumed by a durable submission.
    """

    assignment_id = _validate_identifier(assignment_id, "assignment_id")
    operation_id = _validate_identifier(operation_id, "operation_id")
    if (
        not isinstance(owner_epoch_observed, int)
        or isinstance(owner_epoch_observed, bool)
        or owner_epoch_observed < 0
        or not isinstance(upload_intent_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", upload_intent_id) is None
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint completed-result retention identity is invalid"
        )
    payload = {
        "assignment_id": assignment_id,
        "operation_id": operation_id,
        "owner_epoch_observed": owner_epoch_observed,
        "generations": [],
        "upload_intent_id": upload_intent_id,
    }
    response = journal.execute(api, "retention", payload)
    current_owner_epoch = response.get("current_owner_epoch")
    submission_id = response.get("submission_id")
    if (
        response.get("operation_id") != operation_id
        or response.get("assignment_unchanged") is not True
        or response.get("owner_epoch_observed") != owner_epoch_observed
        or not isinstance(current_owner_epoch, int)
        or isinstance(current_owner_epoch, bool)
        or current_owner_epoch < 0
        or response.get("delete_generations") != []
        or response.get("retain_generations") != []
        or response.get("result_evidence_release") is not True
        or response.get("upload_intent_id") != upload_intent_id
        or not isinstance(submission_id, str)
        or _IDENTIFIER_RE.fullmatch(submission_id) is None
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint completed-result retention acknowledgement changed identity"
        )
    return CheckpointRetentionAcknowledgementV2(
        assignment_id=assignment_id,
        operation_id=operation_id,
        owner_epoch_observed=owner_epoch_observed,
        current_owner_epoch=current_owner_epoch,
        delete_generations=(),
        retain_generations=(),
        result_evidence_release=True,
        upload_intent_id=upload_intent_id,
        submission_id=submission_id,
    )
