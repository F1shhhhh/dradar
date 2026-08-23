"""Authoritative owner lifecycle for controlled Checkpoint V2 canaries.

This module is deliberately separate from the ordinary runner and from the
shadow observer.  Merely importing it changes no rollout state.  A caller must
first negotiate an authoritative CANARY/ON assignment, then explicitly prepare
and start this controller.  Snapshot failures remain fail-open; loss of the
server owner lease is fail-stop because continuing paid execution after the
fence expires could create a duplicate model session.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from . import __version__
from .api_client import ApiClient, ApiError
from .checkpoint_activation_v2 import (
    CheckpointActivationV2,
    CheckpointV2ProtocolError,
)
from .checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from .checkpoint_docker_runtime_v2 import (
    DockerCliLazyCheckpointExporterV2,
    docker_container_backend_v2,
)
from .checkpoint_live_v2 import _runtime_config_v2, _runtime_digest_v2
from .checkpoint_process_guard_v2 import (
    CheckpointV2ProcessGuardError,
    capture_pier_process_evidence_v2,
    process_exited_receipt_v2,
    read_private_json_v2,
    terminate_exact_orphaned_pier_v2,
    validate_pier_process_evidence_v2,
    validate_process_exited_receipt_v2,
)
from .checkpoint_runtime_v2 import (
    CheckpointDataPlaneV2,
    CheckpointObservationRuntimeV2,
    CheckpointObservationV2,
    PublishedCheckpointV2,
    apply_checkpoint_generation_retention_v2,
    checkpoint_observation_payload_v2,
    load_exact_published_checkpoint_v2,
    new_capture_request_v2,
    next_shadow_generation_v2,
)
from .checkpoint_v2 import (
    CheckpointFailureV2,
    CheckpointV2CommandRejected,
    CheckpointV2OrdinaryFallback,
    CheckpointV2Journal,
    CheckpointV2StateMachine,
    CompletedResultPermit,
    FinalizedUsageSegmentReceiptV2,
    FreePreparationPermit,
    OfflineRestorePermit,
    PaidExecutionPermit,
    RestoreReceiptV2,
    SealedCheckpointV2,
    SelectedCheckpointGenerationV2,
    UsageEventV2,
    UsageSegmentEvidenceV2,
    checkpoint_machine_fingerprint,
    completed_restore_receipt,
    finalize_execution_identity_v2,
    new_operation_id,
)
from .telemetry import RunnerTelemetry, platform_family


class CheckpointV2OwnerLost(RuntimeError):
    """The exact server owner can no longer be renewed safely."""


_PREPAID_DISABLE_CODES = frozenset({
    "checkpoint_v2_certification_revoked",
    "checkpoint_v2_kill_switch_active",
})


def _prepaid_disable_code(exc: BaseException) -> str | None:
    if isinstance(exc, (ApiError, CheckpointV2CommandRejected)):
        code = getattr(exc, "code", None)
        if code in _PREPAID_DISABLE_CODES:
            return code
    return None


def _stable_owner_ids(
    assignment_id: str,
    identity_fingerprint: str,
    machine_fingerprint: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        (
            f"dradar-checkpoint-owner-v2:{assignment_id}:"
            f"{identity_fingerprint}:{machine_fingerprint}"
        ).encode("ascii")
    ).hexdigest()
    return f"checkpoint-{digest[:48]}", f"lineage-{digest[16:64]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_ORPHANED_PAID_GATE_STATES = frozenset({
    "preparing", "resume_reserved", "running", "result_ready",
})
_PAID_GATE_CONTRACT_COMMON_FIELDS = frozenset({
    "schema", "assignment_id", "gate_nonce", "action", "session_id",
    "owner_epoch", "reconcile_operation_id", "job_root",
})
_PAID_GATE_CONTRACT_RESUME_FIELDS = frozenset({
    "restore_root", "manifest_sha256", "identity_fingerprint",
    "checkpoint_abi", "recovery_capability", "native_state_schema",
    "restore_adapter_version", "restore_receipt_sha256",
})


def _validate_private_gate_directory_v2(path: Path) -> None:
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise CheckpointV2ProtocolError(
            "checkpoint orphan paid-gate directory is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or canonical != path.absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or (
            os.name == "posix"
            and (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            )
        )
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint orphan paid-gate directory is unsafe"
        )


def _validate_orphan_contract_v2(
    value: Mapping[str, Any],
    *,
    assignment_id: str,
    home: Path,
) -> dict[str, Any]:
    action = value.get("action")
    expected_fields = _PAID_GATE_CONTRACT_COMMON_FIELDS | (
        _PAID_GATE_CONTRACT_RESUME_FIELDS if action == "resume" else frozenset()
    )
    job_root = value.get("job_root")
    try:
        canonical_job_root = Path(str(job_root)).resolve(strict=False)
        canonical_job_root.relative_to(
            (home / "work" / "jobs").resolve(strict=False)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointV2ProtocolError(
            "checkpoint orphan job root is unsafe"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != "dradar-checkpoint-paid-gate-contract-v2"
        or value.get("assignment_id") != assignment_id
        or re.fullmatch(r"[0-9a-f]{32}", str(value.get("gate_nonce"))) is None
        or action not in {"fresh", "resume"}
        or re.fullmatch(
            r"[A-Za-z0-9._:-]{8,64}", str(value.get("session_id"))
        ) is None
        or not isinstance(value.get("owner_epoch"), int)
        or isinstance(value.get("owner_epoch"), bool)
        or value["owner_epoch"] < 0
        or re.fullmatch(
            r"[A-Za-z0-9._-]{8,64}",
            str(value.get("reconcile_operation_id")),
        ) is None
        or not isinstance(job_root, str)
        or not Path(job_root).is_absolute()
        or canonical_job_root == (home / "work" / "jobs").resolve(strict=False)
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint orphan paid-gate contract is inconsistent"
        )
    if action == "resume" and (
        not isinstance(value.get("restore_root"), str)
        or not Path(value["restore_root"]).is_absolute()
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("manifest_sha256"))
        ) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("identity_fingerprint"))
        ) is None
        or not isinstance(value.get("checkpoint_abi"), str)
        or not 8 <= len(value["checkpoint_abi"]) <= 160
        or value.get("recovery_capability")
        not in {"NATIVE_VALID", "WORKSPACE_ONLY"}
        or (
            value.get("native_state_schema") is not None
            and (
                not isinstance(value.get("native_state_schema"), str)
                or not value["native_state_schema"]
                or len(value["native_state_schema"]) > 160
            )
        )
        or not isinstance(value.get("restore_adapter_version"), str)
        or not 1 <= len(value["restore_adapter_version"]) <= 160
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("restore_receipt_sha256"))
        ) is None
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint orphan resume contract is inconsistent"
        )
    result = dict(value)
    result["job_root"] = os.fspath(canonical_job_root)
    return result


def _stable_orphan_reconcile_operation_id_v2(
    assignment_id: str,
    session_id: str,
    owner_epoch: int,
) -> str:
    return hashlib.sha256(
        (
            "dradar-checkpoint-orphan-reconcile-v2:"
            f"{assignment_id}:{session_id}:{owner_epoch}"
        ).encode("utf-8")
    ).hexdigest()[:32]


def reconcile_orphaned_paid_gate_v2(
    assignment: Mapping[str, Any],
    *,
    activation: CheckpointActivationV2,
    api: ApiClient,
    home: Path,
    cleanup_containers: Callable[[Path], None],
) -> dict[str, Any] | None:
    """Fence an earlier process and reconcile its paid gate without replay.

    This is called only while the caller holds the assignment-wide process
    lock.  It never starts Pier or invokes a paid command.
    """

    state = assignment.get("execution_state")
    if (
        assignment.get("checkpoint_protocol_version") != 2
        or state not in _ORPHANED_PAID_GATE_STATES
    ):
        return None
    assignment_id = assignment.get("assignment_id")
    session_id = assignment.get("owner_session_id")
    owner_epoch = assignment.get("owner_epoch")
    resume_generation = assignment.get("resume_generation")
    if (
        re.fullmatch(r"[A-Za-z0-9._-]{8,64}", str(assignment_id)) is None
        or re.fullmatch(r"[A-Za-z0-9._:-]{8,64}", str(session_id)) is None
        or not isinstance(owner_epoch, int)
        or isinstance(owner_epoch, bool)
        or owner_epoch < 0
        or not isinstance(resume_generation, int)
        or isinstance(resume_generation, bool)
        or resume_generation < 0
        or not activation.authoritative
    ):
        raise CheckpointV2ProtocolError(
            "checkpoint orphan owner identity is incomplete"
        )

    home = Path(home).absolute()
    gate_parent = home / "checkpoint-v2" / "paid-gates" / assignment_id
    contract: dict[str, Any] | None = None
    gate_dir: Path | None = None
    if gate_parent.exists() or gate_parent.is_symlink():
        _validate_private_gate_directory_v2(gate_parent)
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for child in sorted(gate_parent.iterdir(), key=lambda item: item.name):
            _validate_private_gate_directory_v2(child)
            if re.fullmatch(r"[0-9a-f]{32}", child.name) is None:
                raise CheckpointV2ProtocolError(
                    "checkpoint orphan paid-gate name is invalid"
                )
            contract_path = child / "contract.json"
            if not contract_path.exists() and not contract_path.is_symlink():
                continue
            try:
                raw_contract = read_private_json_v2(contract_path)
            except CheckpointV2ProcessGuardError as exc:
                raise CheckpointV2ProtocolError(str(exc)) from exc
            candidate = _validate_orphan_contract_v2(
                raw_contract,
                assignment_id=assignment_id,
                home=home,
            )
            if (
                candidate["session_id"] == session_id
                and candidate["owner_epoch"] == owner_epoch
            ):
                candidates.append((child, candidate))
        if len(candidates) > 1:
            raise CheckpointV2ProtocolError(
                "checkpoint orphan owner has multiple paid gates"
            )
        if candidates:
            gate_dir, contract = candidates[0]

    if contract is None:
        if state not in {"preparing", "resume_reserved"}:
            raise CheckpointV2ProtocolError(
                "checkpoint paid owner has no local gate evidence"
            )
        # prepare()/resume-reserve() create the contract before Popen.  Its
        # absence therefore proves no detached Pier process was launched.
        attempted_command = (
            "start" if state == "preparing" else "resume-commit"
        )
        attempted_operation_id = None
        reconcile_operation_id = _stable_orphan_reconcile_operation_id_v2(
            assignment_id, session_id, owner_epoch,
        )
    else:
        if (
            (state == "preparing" and contract["action"] != "fresh")
            or (
                state == "resume_reserved"
                and contract["action"] != "resume"
            )
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint orphan action differs from owner state"
            )
        assert gate_dir is not None
        launch_path = gate_dir / "launch-intent.json"
        failed_path = gate_dir / "launch-failed.json"
        process_path = gate_dir / "process.json"
        exited_path = gate_dir / "process-exited.json"
        launch_exists = launch_path.exists() or launch_path.is_symlink()
        process_exists = process_path.exists() or process_path.is_symlink()

        if process_exists:
            if not launch_exists:
                raise CheckpointV2ProtocolError(
                    "checkpoint Pier process lacks its launch intent"
                )
            try:
                process_evidence = validate_pier_process_evidence_v2(
                    read_private_json_v2(process_path),
                    assignment_id=assignment_id,
                    gate_nonce=contract["gate_nonce"],
                    gate_dir=gate_dir,
                    home=home,
                )
                if exited_path.exists() or exited_path.is_symlink():
                    validate_process_exited_receipt_v2(
                        read_private_json_v2(exited_path),
                        evidence=process_evidence,
                    )
                terminate_exact_orphaned_pier_v2(
                    process_evidence,
                    gate_dir=gate_dir,
                )
                cleanup_containers(Path(process_evidence["job_root"]))
            except CheckpointV2ProcessGuardError as exc:
                raise CheckpointV2ProtocolError(str(exc)) from exc
        elif launch_exists:
            if not (failed_path.exists() or failed_path.is_symlink()):
                # This is the only Popen ambiguity window.  Without a PID/PGID
                # it is unsafe to start another process even though no paid
                # server command could yet have been issued.
                raise CheckpointV2ProtocolError(
                    "checkpoint Pier launch outcome lacks process evidence"
                )
            failed = read_private_json_v2(failed_path)
            if (
                set(failed) != {
                    "schema", "assignment_id", "gate_nonce", "code",
                }
                or failed.get("schema")
                != "dradar-checkpoint-pier-launch-failed-v2"
                or failed.get("assignment_id") != assignment_id
                or failed.get("gate_nonce") != contract["gate_nonce"]
                or re.fullmatch(
                    r"[a-z0-9_]{3,64}", str(failed.get("code"))
                ) is None
            ):
                raise CheckpointV2ProtocolError(
                    "checkpoint Pier launch failure evidence is invalid"
                )
            cleanup_containers(Path(contract["job_root"]))
        elif state in {"running", "result_ready"}:
            raise CheckpointV2ProtocolError(
                "checkpoint paid owner has no Pier launch evidence"
            )

        intent_path = gate_dir / "intent.json"
        if intent_path.exists() or intent_path.is_symlink():
            intent = read_private_json_v2(intent_path)
            if (
                set(intent) != {
                    "schema", "assignment_id", "gate_nonce", "action",
                    "session_id", "expected_owner_epoch",
                    "attempted_operation_id", "reconcile_operation_id",
                    "request_sha256",
                }
                or intent.get("schema")
                != "dradar-checkpoint-paid-gate-intent-v2"
                or intent.get("assignment_id") != assignment_id
                or intent.get("gate_nonce") != contract["gate_nonce"]
                or intent.get("action") != contract["action"]
                or intent.get("session_id") != session_id
                or intent.get("expected_owner_epoch") != owner_epoch
                or re.fullmatch(
                    r"[A-Za-z0-9._-]{8,64}",
                    str(intent.get("attempted_operation_id")),
                ) is None
                or intent.get("reconcile_operation_id")
                != contract["reconcile_operation_id"]
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(intent.get("request_sha256"))
                ) is None
            ):
                raise CheckpointV2ProtocolError(
                    "checkpoint paid-gate intent is inconsistent"
                )
            attempted_operation_id = intent["attempted_operation_id"]
        else:
            attempted_operation_id = None
        attempted_command = (
            "start" if contract["action"] == "fresh" else "resume-commit"
        )
        reconcile_operation_id = contract["reconcile_operation_id"]

    machine = CheckpointV2StateMachine(
        assignment,
        api=api,
        journal=CheckpointV2Journal(
            home,
            assignment_lock_already_held=True,
        ),
        activation=activation,
    )
    response = machine.reconcile_paid_gate(
        session_id=session_id,
        expected_owner_epoch=owner_epoch,
        attempted_operation_id=attempted_operation_id,
        attempted_command=attempted_command,
        operation_id=reconcile_operation_id,
    )
    if gate_dir is not None:
        reconciled_path = gate_dir / "reconciled.json"
        if not reconciled_path.exists() and not reconciled_path.is_symlink():
            AuthoritativeCheckpointRunV2._write_once(reconciled_path, {
                "schema": "dradar-checkpoint-paid-gate-reconciled-v2",
                "assignment_id": assignment_id,
                "gate_nonce": contract["gate_nonce"],
                "reconcile_operation_id": reconcile_operation_id,
                "outcome": response["outcome"],
                "owner_epoch": response["owner_epoch"],
                "resume_generation": response["resume_generation"],
            })
    return response


class AuthoritativeCheckpointRunV2:
    """Own one paid execution epoch and its bounded background maintenance."""

    def __init__(
        self,
        *,
        assignment: Mapping[str, Any],
        effective_assignment: Mapping[str, Any],
        activation: CheckpointActivationV2,
        api: ApiClient,
        telemetry: RunnerTelemetry,
        home: Path,
        job_root: Path,
        renew_interval_sec: float = 60.0,
        initial_capture_delay_sec: float = 30.0,
        capture_interval_sec: float = 300.0,
        maximum_captures: int = 24,
        assignment_lock_held: bool = False,
    ) -> None:
        if not activation.authoritative or not activation.paid_resume_enabled:
            raise CheckpointV2ProtocolError(
                "authoritative owner requires negotiated CANARY or ON mode"
            )
        for value, label, lower, upper in (
            (renew_interval_sec, "renew interval", 1.0, 600.0),
            (initial_capture_delay_sec, "initial capture delay", 0.0, 86_400.0),
            (capture_interval_sec, "capture interval", 1.0, 86_400.0),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"checkpoint owner {label} is invalid")
        if (
            not isinstance(maximum_captures, int)
            or isinstance(maximum_captures, bool)
            or not 1 <= maximum_captures <= 10_000
        ):
            raise ValueError("checkpoint owner capture limit is invalid")

        identity = assignment.get("execution_identity")
        if not isinstance(identity, Mapping):
            raise CheckpointV2ProtocolError(
                "checkpoint owner claim identity is unavailable"
            )
        harness = identity.get("harness")
        provider = identity.get("provider")
        agent_version = effective_assignment.get("agent_version")
        if (
            not isinstance(harness, str)
            or not isinstance(provider, str)
            or not isinstance(agent_version, str)
            or not agent_version
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint owner runtime identity is incomplete"
            )
        contract = checkpoint_adapter_contract_v2(harness, provider)
        runtime_profile, model_config_version = _runtime_config_v2(
            harness, provider,
        )
        runtime_digest = _runtime_digest_v2(
            contract=contract,
            agent_version=agent_version,
            runtime_profile=runtime_profile,
            model_config_version=model_config_version,
        )
        backend = docker_container_backend_v2()
        machine_fingerprint = checkpoint_machine_fingerprint(Path(home))
        telemetry.configure_checkpoint_runtime(
            container_backend=backend,
            machine_fingerprint=machine_fingerprint,
        )
        telemetry.configure_checkpoint_observation_reporting(Path(home))

        self.assignment = dict(assignment)
        self.effective_assignment = dict(effective_assignment)
        self.activation = activation
        self.api = api
        self.telemetry = telemetry
        self.home = Path(home).absolute()
        self.job_root = Path(job_root).absolute()
        self.contract = contract
        self.harness = harness
        self.provider = provider
        self.agent_version = agent_version
        self.runtime_profile = runtime_profile
        self.model_config_version = model_config_version
        self.runtime_digest = runtime_digest
        self.container_backend = backend
        self.machine_fingerprint = machine_fingerprint
        self.journal = CheckpointV2Journal(
            self.home,
            assignment_lock_already_held=assignment_lock_held,
        )
        self.exporter = DockerCliLazyCheckpointExporterV2(
            job_root=self.job_root,
            contract=contract,
        )
        self.data_plane = CheckpointDataPlaneV2(
            activation=activation,
            storage_root=(
                self.home
                / "checkpoint-v2"
                / "authoritative"
                / str(assignment["assignment_id"])
            ),
        )
        self.renew_interval_sec = float(renew_interval_sec)
        self.initial_capture_delay_sec = float(initial_capture_delay_sec)
        self.capture_interval_sec = float(capture_interval_sec)
        self.maximum_captures = maximum_captures

        self.state_machine: CheckpointV2StateMachine | None = None
        self._permit: FreePreparationPermit | None = None
        self._usage_receipt: FinalizedUsageSegmentReceiptV2 | None = None
        self._last_sealed: SealedCheckpointV2 | None = None
        self._sealed: list[SealedCheckpointV2] = []
        self._published: list[PublishedCheckpointV2] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._fatal: BaseException | None = None
        self._resume_selected: SelectedCheckpointGenerationV2 | None = None
        self._resume_published: PublishedCheckpointV2 | None = None
        self._restore_receipt: RestoreReceiptV2 | None = None
        self._paid_gate_operation_id: str | None = None
        self._paid_gate_reconcile_operation_id = new_operation_id()
        self._paid_gate_reconcile_required = False
        self._pier_process: Any | None = None
        self._pier_process_evidence: dict[str, Any] | None = None
        self._finalized_assignment: dict[str, Any] | None = None
        self._ordinary_fallback = False
        self._gate_nonce = new_operation_id()
        self._gate_action = (
            "resume" if assignment.get("execution_state") == "paused" else "fresh"
        )
        self._gate_dir = (
            self.home
            / "checkpoint-v2"
            / "paid-gates"
            / str(assignment["assignment_id"])
            / self._gate_nonce
        )

    @property
    def permit(self) -> FreePreparationPermit | None:
        with self._lock:
            return self._permit

    @property
    def ordinary_fallback(self) -> bool:
        return self._ordinary_fallback

    @property
    def offline_restore_pending(self) -> bool:
        with self._lock:
            return isinstance(self._permit, OfflineRestorePermit)

    @property
    def paid_gate_reconcile_required(self) -> bool:
        with self._lock:
            return self._paid_gate_reconcile_required

    @property
    def last_sealed(self) -> SealedCheckpointV2 | None:
        with self._lock:
            return self._last_sealed

    @property
    def published_generations(self) -> tuple[PublishedCheckpointV2, ...]:
        with self._lock:
            return tuple(self._published)

    @property
    def gate_dir(self) -> Path:
        return self._gate_dir

    def pier_agent_kwargs(self) -> dict[str, str]:
        if self._permit is None:
            raise CheckpointV2ProtocolError(
                "checkpoint v2 owner must prepare before Pier command construction"
            )
        return {"checkpoint_v2_gate_dir": os.fspath(self._gate_dir)}

    @staticmethod
    def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8") + b"\n"

    @staticmethod
    def _write_once(path: Path, value: Mapping[str, Any]) -> None:
        data = AuthoritativeCheckpointRunV2._canonical_bytes(value)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise CheckpointV2ProtocolError(
                        "checkpoint v2 gate write failed"
                    )
                view = view[written:]
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if os.name == "posix":
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    @staticmethod
    def _read_private_json(path: Path) -> tuple[dict[str, Any], bytes]:
        try:
            metadata = path.lstat()
            data = path.read_bytes()
        except OSError as exc:
            raise CheckpointV2ProtocolError(
                "checkpoint v2 gate request is unreadable"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or len(data) > 4096
            or (
                os.name == "posix"
                and (
                    metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                )
            )
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint v2 gate request is unsafe"
            )
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointV2ProtocolError(
                "checkpoint v2 gate request is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise CheckpointV2ProtocolError(
                "checkpoint v2 gate request is invalid"
            )
        return value, data

    def _create_paid_gate(self) -> None:
        permit = self._permit
        if permit is None:
            raise CheckpointV2ProtocolError(
                "checkpoint v2 paid gate has no free owner"
            )
        parents = [
            self.home / "checkpoint-v2",
            self.home / "checkpoint-v2" / "paid-gates",
            self.home / "checkpoint-v2" / "paid-gates"
            / str(self.assignment["assignment_id"]),
            self._gate_dir,
        ]
        for path in parents:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISDIR(metadata.st_mode)
                or (
                    os.name == "posix"
                    and (
                        metadata.st_uid != os.getuid()
                        or stat.S_IMODE(metadata.st_mode) != 0o700
                    )
                )
            ):
                raise CheckpointV2ProtocolError(
                    "checkpoint v2 paid gate directory is unsafe"
                )
        contract: dict[str, Any] = {
            "schema": "dradar-checkpoint-paid-gate-contract-v2",
            "assignment_id": self.assignment["assignment_id"],
            "gate_nonce": self._gate_nonce,
            "action": self._gate_action,
            "session_id": permit.session_id,
            "owner_epoch": permit.owner_epoch,
            "reconcile_operation_id": (
                self._paid_gate_reconcile_operation_id
            ),
            "job_root": os.fspath(self.job_root),
        }
        if self._gate_action == "resume":
            selected = self._resume_selected
            published = self._resume_published
            receipt = self._restore_receipt
            if selected is None or published is None or receipt is None:
                raise CheckpointV2ProtocolError(
                    "checkpoint v2 resume gate lacks exact restore evidence"
                )
            contract.update({
                "restore_root": os.fspath(published.root),
                "manifest_sha256": selected.manifest_sha256,
                "identity_fingerprint": selected.compatibility_fingerprint,
                "checkpoint_abi": selected.checkpoint_abi,
                "recovery_capability": selected.recovery_capability,
                "native_state_schema": selected.native_state_schema,
                "restore_adapter_version": receipt.restore_adapter_version,
                "restore_receipt_sha256": receipt.receipt_sha256,
            })
        self._write_once(self._gate_dir / "contract.json", contract)

    def record_pier_launch_intent(self, command: list[str]) -> None:
        """Fence the Popen gap without persisting Provider command contents."""

        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint Pier launch command is invalid"
            )
        self._write_once(self._gate_dir / "launch-intent.json", {
            "schema": "dradar-checkpoint-pier-launch-intent-v2",
            "assignment_id": self.assignment["assignment_id"],
            "gate_nonce": self._gate_nonce,
            "command_sha256": hashlib.sha256(
                self._canonical_bytes({"argv": command})
            ).hexdigest(),
            "job_root": os.fspath(self.job_root),
        })

    def record_pier_launch_failed(self, *, code: str) -> None:
        if re.fullmatch(r"[a-z0-9_]{3,64}", code) is None:
            raise CheckpointV2ProtocolError(
                "checkpoint Pier launch failure code is invalid"
            )
        self._write_once(self._gate_dir / "launch-failed.json", {
            "schema": "dradar-checkpoint-pier-launch-failed-v2",
            "assignment_id": self.assignment["assignment_id"],
            "gate_nonce": self._gate_nonce,
            "code": code,
        })

    def register_pier_process(
        self,
        pier_process: Any,
        command: list[str],
    ) -> None:
        """Durably bind the exact detached Pier group before paid authority."""

        with self._lock:
            if self._pier_process is not None or self._pier_process_evidence is not None:
                raise CheckpointV2ProtocolError(
                    "checkpoint Pier process registration is one-shot"
                )
        launch, _launch_bytes = self._read_private_json(
            self._gate_dir / "launch-intent.json"
        )
        expected_command_sha256 = hashlib.sha256(
            self._canonical_bytes({"argv": command})
        ).hexdigest()
        if (
            set(launch) != {
                "schema", "assignment_id", "gate_nonce", "command_sha256",
                "job_root",
            }
            or launch.get("schema")
            != "dradar-checkpoint-pier-launch-intent-v2"
            or launch.get("assignment_id") != self.assignment["assignment_id"]
            or launch.get("gate_nonce") != self._gate_nonce
            or launch.get("command_sha256") != expected_command_sha256
            or launch.get("job_root") != os.fspath(self.job_root)
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint Pier launch intent is inconsistent"
            )
        try:
            evidence = capture_pier_process_evidence_v2(
                pier_process,
                command,
                assignment_id=str(self.assignment["assignment_id"]),
                gate_nonce=self._gate_nonce,
                gate_dir=self._gate_dir,
                job_root=self.job_root,
                home=self.home,
            )
        except BaseException:
            poll = getattr(pier_process, "poll", None)
            if callable(poll) and poll() is not None:
                try:
                    self.record_pier_launch_failed(
                        code="exited_before_process_evidence",
                    )
                except BaseException:
                    pass
            raise
        self._write_once(self._gate_dir / "process.json", evidence)
        with self._lock:
            self._pier_process = pier_process
            self._pier_process_evidence = evidence

    def _persist_paid_gate_intent(
        self,
        *,
        request_sha256: str,
        attempted_operation_id: str,
    ) -> None:
        permit = self._permit
        if permit is None:
            raise CheckpointV2ProtocolError(
                "checkpoint v2 paid gate intent has no owner"
            )
        self._write_once(self._gate_dir / "intent.json", {
            "schema": "dradar-checkpoint-paid-gate-intent-v2",
            "assignment_id": self.assignment["assignment_id"],
            "gate_nonce": self._gate_nonce,
            "action": self._gate_action,
            "session_id": permit.session_id,
            "expected_owner_epoch": permit.owner_epoch,
            "attempted_operation_id": attempted_operation_id,
            "reconcile_operation_id": self._paid_gate_reconcile_operation_id,
            "request_sha256": request_sha256,
        })

    def prepare(self) -> FreePreparationPermit:
        """Finalize identity and acquire only a free PREPARING owner."""

        if self.state_machine is not None or self._permit is not None:
            raise CheckpointV2ProtocolError(
                "checkpoint owner preparation is one-shot"
            )
        # Register the immutable backend/machine cohort before checkout.  The
        # heartbeat transport is best effort; checkout itself remains the
        # authoritative fail-closed proof that the session reached the server.
        self.telemetry.set_phase(
            "preparing", self.assignment["assignment_id"],
            self.assignment.get("resume_generation"),
        )
        self.telemetry.flush()
        receipt = finalize_execution_identity_v2(
            self.assignment,
            api=self.api,
            journal=self.journal,
            harness=self.harness,
            provider=self.provider,
            agent_version=self.agent_version,
            runtime_profile=self.runtime_profile,
            model_config_version=self.model_config_version,
            checkpoint_abi=self.contract.checkpoint_abi,
            runtime_compatibility_digest=self.runtime_digest,
        )
        finalized_assignment = {
            **self.effective_assignment,
            "checkpoint_protocol_version": self.assignment.get(
                "checkpoint_protocol_version", 1,
            ),
            "checkpoint_v2_identity_protocol_version": 2,
            "checkpoint_v2_rollout_mode": self.activation.server_mode.wire_value,
            "checkpoint_v2_controlled_account": (
                self.activation.controlled_account
            ),
            "execution_identity": {
                **asdict(receipt.identity),
                "identity_state": "FINAL",
                "identity_source": "runtime_finalize",
            },
        }
        self._finalized_assignment = finalized_assignment
        machine = CheckpointV2StateMachine(
            finalized_assignment,
            api=self.api,
            journal=self.journal,
            activation=self.activation,
        )
        if self.assignment.get("execution_state") == "paused":
            selected_assignment = {
                **finalized_assignment,
                "execution_state": "paused",
                "checkpoint_id": self.assignment.get("checkpoint_id"),
                "checkpoint_v2_selected_generation": self.assignment.get(
                    "checkpoint_v2_selected_generation"
                ),
            }
            try:
                selected = SelectedCheckpointGenerationV2.from_assignment(
                    selected_assignment,
                )
            except Exception:
                self._ordinary_fallback = True
                raise machine.fallback_to_ordinary_fresh(
                    session_id=self.telemetry.session_id,
                    expected_owner_epoch=machine.initial_owner_epoch,
                    reason="selected_descriptor_invalid",
                )
            try:
                selected.assert_reachable_from(self.machine_fingerprint)
            except Exception:
                self._ordinary_fallback = True
                raise machine.fallback_to_ordinary_fresh(
                    session_id=self.telemetry.session_id,
                    expected_owner_epoch=machine.initial_owner_epoch,
                    reason="snapshot_unreachable",
                )
            try:
                permit = machine.reserve_offline_restore(
                    session_id=self.telemetry.session_id,
                    expected_owner_epoch=machine.initial_owner_epoch,
                    checkpoint_id=selected.checkpoint_id,
                    snapshot_generation=selected.snapshot_generation,
                    manifest_sha256=selected.manifest_sha256,
                    requester_machine_fingerprint=self.machine_fingerprint,
                )
            except (ApiError, CheckpointV2CommandRejected) as exc:
                code = getattr(exc, "code", None)
                fallback_reason = {
                    "checkpoint_v2_cohort_not_certified": (
                        "cohort_not_certified"
                    ),
                    "checkpoint_snapshot_unreachable": "snapshot_unreachable",
                    "checkpoint_machine_identity_mismatch": (
                        "snapshot_unreachable"
                    ),
                    "checkpoint_record_mismatch": (
                        "selected_descriptor_invalid"
                    ),
                }.get(code)
                if fallback_reason is None:
                    raise
                self._ordinary_fallback = True
                raise machine.fallback_to_ordinary_fresh(
                    session_id=self.telemetry.session_id,
                    expected_owner_epoch=machine.initial_owner_epoch,
                    reason=fallback_reason,
                )
            try:
                published = load_exact_published_checkpoint_v2(
                    self.data_plane.storage_root,
                    checkpoint_id=selected.checkpoint_id,
                    checkpoint_lineage_id=selected.checkpoint_lineage_id,
                    snapshot_generation=selected.snapshot_generation,
                    capture_id=selected.capture_id,
                    manifest_sha256=selected.manifest_sha256,
                    expected_identity_fingerprint=(
                        selected.compatibility_fingerprint
                    ),
                    expected_checkpoint_core_abi=selected.checkpoint_core_abi,
                    expected_checkpoint_abi=selected.checkpoint_abi,
                    expected_recovery_capability=selected.recovery_capability,
                    expected_native_state_schema=selected.native_state_schema,
                )
            except Exception:
                self._ordinary_fallback = True
                raise machine.fallback_to_ordinary_fresh(
                    session_id=self.telemetry.session_id,
                    expected_owner_epoch=permit.owner_epoch,
                    reason="archive_invalid",
                )
            receipt = completed_restore_receipt(
                permit,
                restore_adapter_version=self.contract.restorer_version,
                restored_manifest_sha256=selected.manifest_sha256,
            )
            self._resume_selected = selected
            self._resume_published = published
            self._restore_receipt = receipt
        else:
            try:
                permit = machine.checkout(session_id=self.telemetry.session_id)
            except CheckpointV2OrdinaryFallback:
                self._ordinary_fallback = True
                raise
        with self._lock:
            self.state_machine = machine
            self._permit = permit
        self._create_paid_gate()
        self._start_renewal()
        return permit

    def build_ordinary_fallback_shadow(self):
        """Keep collecting harmless evidence after certification declines V2.

        Checkout leaves the assignment on protocol V1.  Capping the local
        assignment snapshot to OBSERVE prevents the shadow controller from
        acquiring ownership while preserving the exact finalized identity.
        """

        finalized = self._finalized_assignment
        if finalized is None or self._permit is not None:
            raise CheckpointV2ProtocolError(
                "checkpoint ordinary fallback is not available"
            )
        from .checkpoint_live_v2 import build_live_checkpoint_shadow_v2

        shadow_assignment = {
            **finalized,
            "checkpoint_protocol_version": 1,
            "checkpoint_v2_rollout_mode": "observe",
            "checkpoint_v2_controlled_account": False,
        }
        return build_live_checkpoint_shadow_v2(
            assignment=shadow_assignment,
            effective_assignment=self.effective_assignment,
            local_mode="observe",
            api=self.api,
            telemetry=self.telemetry,
            home=self.home,
            job_root=self.job_root,
        )

    def fallback_after_restore_preflight(
        self,
    ) -> CheckpointV2OrdinaryFallback:
        """Downgrade a free offline restore after Pier failed before Provider."""

        with self._lock:
            machine = self.state_machine
            permit = self._permit
        if machine is None or not isinstance(permit, OfflineRestorePermit):
            raise CheckpointV2ProtocolError(
                "checkpoint restore preflight fallback has no offline owner"
            )
        fallback = machine.fallback_to_ordinary_fresh(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            reason="restore_preflight_failed",
        )
        with self._lock:
            if self._permit != permit:
                raise CheckpointV2ProtocolError(
                    "checkpoint restore owner changed during fresh fallback"
                )
            self._permit = None
            self._ordinary_fallback = True
        self._stop.set()
        return fallback

    def _start_renewal(self) -> None:
        if self._renew_thread is not None:
            return
        self._renew_thread = threading.Thread(
            target=self._renew_loop,
            name="dradar-checkpoint-v2-owner-renew",
            daemon=True,
        )
        self._renew_thread.start()

    def _start_capture(self) -> None:
        if self._capture_thread is not None:
            return
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="dradar-checkpoint-v2-owner-capture",
            daemon=True,
        )
        self._capture_thread.start()

    def authorize_at_paid_gate(
        self, pier_process: Any, *, timeout_sec: float,
    ) -> PaidExecutionPermit:
        """Wait for Pier's pre-Provider barrier, then issue one exact grant."""

        request_path = self._gate_dir / "request.json"
        deadline = time.monotonic() + max(1.0, min(float(timeout_sec), 3600.0))
        while not request_path.exists() and not request_path.is_symlink():
            self.raise_if_fatal()
            poll = getattr(pier_process, "poll", None)
            if callable(poll) and poll() is not None:
                raise CheckpointV2ProtocolError(
                    "Pier exited before reaching the checkpoint v2 paid gate"
                )
            if time.monotonic() >= deadline:
                raise CheckpointV2ProtocolError(
                    "Pier did not reach the checkpoint v2 paid gate in time"
                )
            time.sleep(0.1)
        request, request_bytes = self._read_private_json(request_path)
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        if (
            set(request) != {
                "schema", "assignment_id", "gate_nonce", "action",
                "restore_receipt_sha256",
            }
            or request.get("schema")
            != "dradar-checkpoint-paid-gate-request-v2"
            or request.get("assignment_id") != self.assignment["assignment_id"]
            or request.get("gate_nonce") != self._gate_nonce
            or request.get("action") != self._gate_action
            or (
                self._gate_action == "fresh"
                and request.get("restore_receipt_sha256") is not None
            )
            or (
                self._gate_action == "resume"
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(request.get("restore_receipt_sha256")),
                ) is None
            )
        ):
            raise CheckpointV2ProtocolError(
                "checkpoint v2 paid gate request changed execution identity"
            )
        operation_id = self._paid_gate_operation_id or new_operation_id()
        self._paid_gate_operation_id = operation_id
        try:
            self._persist_paid_gate_intent(
                request_sha256=request_sha256,
                attempted_operation_id=operation_id,
            )
        except BaseException as intent_error:
            with self._lock:
                machine = self.state_machine
                permit = self._permit
            if machine is None or permit is None:
                raise
            try:
                fallback = machine.fallback_to_ordinary_fresh(
                    session_id=permit.session_id,
                    expected_owner_epoch=permit.owner_epoch,
                    reason="paid_gate_intent_failed",
                )
            except BaseException:
                # No paid command was attempted, but without a durable intent
                # the owner cannot prove enough state for an automatic retry.
                self._stop.set()
                raise intent_error
            with self._lock:
                self._permit = None
                self._ordinary_fallback = True
                self._paid_gate_reconcile_required = False
            self._stop.set()
            raise fallback from intent_error
        try:
            if self._gate_action == "fresh":
                with self._lock:
                    machine = self.state_machine
                    permit = self._permit
                if machine is None or not isinstance(
                    permit, FreePreparationPermit,
                ):
                    raise CheckpointV2ProtocolError(
                        "checkpoint v2 fresh gate has no free owner"
                    )
                paid = machine.start_paid(
                    permit, operation_id=operation_id,
                )
                with self._lock:
                    self._permit = paid
                self.telemetry.set_phase(
                    "running", self.assignment["assignment_id"],
                    self.assignment.get("resume_generation"),
                )
                self.telemetry.flush()
                self._start_capture()
            else:
                with self._lock:
                    machine = self.state_machine
                    permit = self._permit
                    receipt = self._restore_receipt
                if (
                    machine is None
                    or not isinstance(permit, OfflineRestorePermit)
                    or receipt is None
                    or request.get("restore_receipt_sha256")
                    != receipt.receipt_sha256
                ):
                    raise CheckpointV2ProtocolError(
                        "checkpoint v2 resume gate lacks its exact restore receipt"
                    )
                paid = machine.commit_paid_resume(
                    permit, receipt=receipt, operation_id=operation_id,
                )
                with self._lock:
                    self._permit = paid
                self.telemetry.set_phase(
                    "running", self.assignment["assignment_id"],
                    self.assignment.get("resume_generation"),
                )
                self.telemetry.flush()
                self._start_capture()
            self._write_once(self._gate_dir / "grant.json", {
                "schema": "dradar-checkpoint-paid-gate-grant-v2",
                "assignment_id": self.assignment["assignment_id"],
                "gate_nonce": self._gate_nonce,
                "request_sha256": request_sha256,
                "owner_epoch": paid.owner_epoch,
                "usage_segment_id": paid.usage_segment_id,
                "paid_execution_authorized": True,
            })
            with self._lock:
                self._paid_gate_reconcile_required = False
            return paid
        except BaseException as exc:
            disable_code = _prepaid_disable_code(exc)
            with self._lock:
                self._paid_gate_reconcile_required = disable_code is None
            try:
                self._write_once(self._gate_dir / "denial.json", {
                    "schema": "dradar-checkpoint-paid-gate-denial-v2",
                    "assignment_id": self.assignment["assignment_id"],
                    "gate_nonce": self._gate_nonce,
                    "request_sha256": request_sha256,
                    "code": disable_code or "authorization_failed",
                })
            except BaseException:
                pass
            if disable_code is not None:
                with self._lock:
                    machine = self.state_machine
                    permit = self._permit
                if machine is None or permit is None:
                    raise
                fallback = machine.fallback_to_ordinary_fresh(
                    session_id=permit.session_id,
                    expected_owner_epoch=permit.owner_epoch,
                    reason="operator_disable",
                )
                with self._lock:
                    self._permit = None
                    self._ordinary_fallback = True
                self._stop.set()
                raise fallback
            raise

    def reconcile_ambiguous_paid_gate(self) -> dict[str, Any]:
        """Resolve a paid-gate crash without ever replaying paid authority.

        The server atomically inspects whether the original command committed.
        Absence falls back to ordinary fresh execution; a possible paid epoch
        is faulted with unavailable usage.  The reconciliation command itself
        is free-only and replay-safe across another client process.
        """

        with self._lock:
            machine = self.state_machine
            permit = self._permit
            operation_id = self._paid_gate_operation_id
            required = self._paid_gate_reconcile_required
        if machine is None or permit is None or not required or operation_id is None:
            raise CheckpointV2ProtocolError(
                "checkpoint paid gate has no ambiguous authorization to reconcile"
            )
        response = machine.reconcile_paid_gate(
            session_id=permit.session_id,
            expected_owner_epoch=permit.owner_epoch,
            attempted_operation_id=operation_id,
            attempted_command=(
                "start" if self._gate_action == "fresh" else "resume-commit"
            ),
            operation_id=self._paid_gate_reconcile_operation_id,
        )
        with self._lock:
            self._permit = None
            self._paid_gate_reconcile_required = False
            self._ordinary_fallback = response.get("outcome") == "fresh_fallback"
        self._stop.set()
        return response

    def _renew_loop(self) -> None:
        while not self._stop.wait(self.renew_interval_sec):
            try:
                with self._lock:
                    machine = self.state_machine
                    permit = self._permit
                if machine is None or permit is None:
                    raise CheckpointV2ProtocolError(
                        "checkpoint owner permit disappeared"
                    )
                renewed = machine.renew_owner(permit)
                with self._lock:
                    if self._permit != permit:
                        raise CheckpointV2ProtocolError(
                            "checkpoint owner changed during renewal"
                        )
                    self._permit = renewed
            except BaseException as exc:
                with self._lock:
                    if self._fatal is None:
                        self._fatal = exc
                self._stop.set()
                return

    def _capture_loop(self) -> None:
        if self._stop.wait(self.initial_capture_delay_sec):
            return
        for _index in range(self.maximum_captures):
            if self._stop.is_set():
                return
            try:
                asyncio.run(self._capture_once())
            except BaseException:
                # The data plane normally reduces failures to typed
                # observations.  An unexpected observer exception is still
                # non-authoritative and cannot fail the paid mainline.
                pass
            if self._stop.wait(self.capture_interval_sec):
                return

    async def _capture_once(self) -> CheckpointObservationV2:
        with self._lock:
            machine = self.state_machine
            permit = self._permit
        if machine is None or not isinstance(permit, PaidExecutionPermit):
            return CheckpointObservationV2(
                status="skipped", capture_id=None,
            )
        checkpoint_id, lineage_id = _stable_owner_ids(
            machine.identity.assignment_id,
            machine.identity.fingerprint,
            self.machine_fingerprint,
        )
        try:
            recovery_capability, native_state_schema = (
                await self.exporter.recovery_facts()
            )
        except Exception:
            recovery_capability = (
                "NONE" if self.contract.native_resume_required
                else "WORKSPACE_ONLY"
            )
            native_state_schema = self.contract.native_state_schema
        try:
            generation = await asyncio.to_thread(
                next_shadow_generation_v2,
                self.data_plane.storage_root,
                checkpoint_id,
            )
        except Exception:
            generation = 1
        request = new_capture_request_v2(
            checkpoint_id=checkpoint_id,
            checkpoint_lineage_id=lineage_id,
            snapshot_generation=generation,
            identity_fingerprint=machine.identity.fingerprint,
            checkpoint_abi=machine.identity.checkpoint_abi,
            recovery_capability=recovery_capability,
            native_state_schema=native_state_schema,
        )
        started = time.monotonic()
        observation = await self.data_plane.observe_capture(
            request, self.exporter,
        )
        elapsed_ms = min(
            86_400_000, int((time.monotonic() - started) * 1000),
        )
        try:
            payload = checkpoint_observation_payload_v2(
                request,
                observation,
                self.activation,
                CheckpointObservationRuntimeV2(
                    assignment_id=machine.identity.assignment_id,
                    operation_id=new_operation_id(),
                    elapsed_ms=elapsed_ms,
                    platform=platform_family(),
                    container_backend=self.container_backend,
                    client_version=__version__,
                    adapter_version=self.contract.exporter_version,
                ),
            )
        except Exception:
            payload = None
        if payload is not None:
            self.telemetry.record_checkpoint_observation(payload)
        if observation.status == "sealed" and observation.published is not None:
            sealed = SealedCheckpointV2(
                checkpoint_id=request.checkpoint_id,
                checkpoint_lineage_id=request.checkpoint_lineage_id,
                snapshot_generation=request.snapshot_generation,
                capture_id=request.capture_id,
                manifest_schema=2,
                manifest_sha256=observation.published.manifest_sha256,
                compatibility_fingerprint=machine.identity.fingerprint,
                recovery_capability=request.recovery_capability,
                native_state_schema=request.native_state_schema,
                storage_scope="machine_local",
                writer_machine_fingerprint=self.machine_fingerprint,
                sync_state="local_only",
            )
            with self._lock:
                self._last_sealed = sealed
                self._sealed.append(sealed)
                self._published.append(observation.published)
        return observation

    def raise_if_fatal(self) -> None:
        with self._lock:
            fatal = self._fatal
        if fatal is not None:
            raise CheckpointV2OwnerLost(
                "checkpoint v2 owner lease could not be renewed; "
                "paid execution must stop"
            ) from fatal

    def mainline_exited(self, *, timeout: float = 2.0) -> None:
        """Stop background work without releasing result/checkpoint evidence."""

        with self._lock:
            pier_process = self._pier_process
            process_evidence = self._pier_process_evidence
        if pier_process is not None and process_evidence is not None:
            poll = getattr(pier_process, "poll", None)
            returncode = poll() if callable(poll) else None
            if isinstance(returncode, int) and not isinstance(returncode, bool):
                exited_path = self._gate_dir / "process-exited.json"
                if not exited_path.exists() and not exited_path.is_symlink():
                    self._write_once(
                        exited_path,
                        process_exited_receipt_v2(
                            process_evidence,
                            returncode=returncode,
                        ),
                    )
        self._stop.set()
        for thread in (self._renew_thread, self._capture_thread):
            if thread is not None:
                thread.join(timeout=max(0.0, min(float(timeout), 5.0)))

    def finalize_usage(
        self,
        *,
        n_input_tokens: object,
        n_cache_tokens: object,
        n_output_tokens: object,
        token_usage_events: object = None,
        request_usage_complete: object = None,
        request_usage_observed: object = None,
        occurred_at: str | None = None,
        complete: bool = True,
    ) -> FinalizedUsageSegmentReceiptV2:
        with self._lock:
            machine = self.state_machine
            permit = self._permit
            existing = self._usage_receipt
        if existing is not None:
            return existing
        if machine is None or not isinstance(permit, PaidExecutionPermit):
            raise CheckpointV2ProtocolError(
                "checkpoint usage has no paid owner"
            )
        counters = (n_input_tokens, n_cache_tokens, n_output_tokens)
        counters_valid = all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in counters
        ) and int(n_cache_tokens) <= int(n_input_tokens)
        raw_events = (
            token_usage_events if isinstance(token_usage_events, list) else []
        )
        events_valid = (
            isinstance(token_usage_events, list)
            and 0 < len(raw_events) <= 512
        )
        materialized: list[UsageEventV2] = []
        event_totals = [0, 0, 0]
        if events_valid:
            for source_sequence, raw in enumerate(raw_events):
                if not isinstance(raw, Mapping):
                    events_valid = False
                    break
                raw_counters = tuple(
                    raw.get(name) for name in (
                        "n_input_tokens", "n_cache_tokens",
                        "n_output_tokens",
                    )
                )
                observed_at = raw.get("occurred_at")
                try:
                    instant = datetime.fromisoformat(
                        observed_at.replace("Z", "+00:00")
                    )
                except (AttributeError, TypeError, ValueError):
                    events_valid = False
                    break
                if (
                    instant.tzinfo is None
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        for value in raw_counters
                    )
                    or int(raw_counters[1]) > int(raw_counters[0])
                ):
                    events_valid = False
                    break
                identity = {
                    "schema": "dradar-checkpoint-usage-event-identity-v2",
                    "assignment_id": machine.identity.assignment_id,
                    "harness": self.harness,
                    "provider": self.provider,
                    "ledger_scope": self.contract.usage_ledger_scope,
                    "source_sequence": source_sequence,
                    "occurred_at": observed_at,
                    "n_input_tokens": int(raw_counters[0]),
                    "n_cache_tokens": int(raw_counters[1]),
                    "n_output_tokens": int(raw_counters[2]),
                }
                if self.contract.usage_ledger_scope == "segment_delta":
                    # Segment-local ledgers (Kimi/ZCode) can restart their
                    # source sequence at zero. Bind those event identities to
                    # the paid epoch so two distinct requests with identical
                    # counters and timestamps cannot collapse into one.
                    identity["usage_segment_id"] = permit.usage_segment_id
                event_id = hashlib.sha256(json.dumps(
                    identity, sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                event = UsageEventV2(
                    event_id=event_id,
                    occurred_at=str(observed_at),
                    n_input_tokens=int(raw_counters[0]),
                    n_cache_tokens=int(raw_counters[1]),
                    n_output_tokens=int(raw_counters[2]),
                )
                materialized.append(event)
                for index, value in enumerate(raw_counters):
                    event_totals[index] += int(value)
        events_valid = (
            events_valid
            and counters_valid
            and tuple(event_totals) == tuple(int(value) for value in counters)
        )
        if not events_valid or request_usage_observed is not True:
            evidence = UsageSegmentEvidenceV2(
                completeness="unavailable",
                evidence_kind="unavailable",
                events=(),
            )
        else:
            evidence_kind = (
                "trajectory_bundle"
                if self.harness == "codex"
                else "provider_request_ledger"
            )
            evidence = UsageSegmentEvidenceV2(
                completeness=(
                    "complete"
                    if complete and request_usage_complete is True
                    else "partial"
                ),
                evidence_kind=evidence_kind,
                events=tuple(materialized),
                ledger_scope=self.contract.usage_ledger_scope,
            )
        receipt = machine.finalize_usage_segment(permit, evidence=evidence)
        with self._lock:
            self._usage_receipt = receipt
        return receipt

    def declare_result_ready(
        self,
        *,
        upload_intent_id: str,
    ) -> CompletedResultPermit:
        with self._lock:
            machine = self.state_machine
            permit = self._permit
            usage_receipt = self._usage_receipt
        if (
            machine is None
            or not isinstance(permit, PaidExecutionPermit)
            or usage_receipt is None
        ):
            raise CheckpointV2ProtocolError(
                "completed result lacks finalized usage ownership"
            )
        completed = machine.declare_result_ready(
            permit,
            usage_receipt=usage_receipt,
            upload_intent_id=upload_intent_id,
        )
        with self._lock:
            self._permit = completed
        return completed

    def pause_last_sealed(self) -> int | None:
        """Release to the newest sealed generation, or fault without invalidating."""

        self.mainline_exited()
        with self._lock:
            machine = self.state_machine
            permit = self._permit
            sealed = self._last_sealed
            usage_receipt = self._usage_receipt
        if machine is None or permit is None:
            return None
        if isinstance(permit, OfflineRestorePermit):
            return machine.abort_offline_restore(permit)
        if sealed is not None:
            if isinstance(permit, PaidExecutionPermit) and usage_receipt is None:
                usage_receipt = self.finalize_usage(
                    n_input_tokens=None,
                    n_cache_tokens=None,
                    n_output_tokens=None,
                    complete=False,
                )
            return machine.pause_sealed(
                permit,
                checkpoint=sealed,
                usage_receipt=usage_receipt,
            )
        failure = CheckpointFailureV2(
            stage="capture",
            code="checkpoint_capture_failed",
            failure_layer="checkpoint_core",
            recoverability="none",
            cleanup_result="reaped",
            container_backend=self.container_backend,
        )
        machine.report_failure(permit, failure=failure)
        return None

    def release_after_submission(self) -> str:
        """Delete local evidence only after the server proves durable submit."""

        with self._lock:
            machine = self.state_machine
            permit = self._permit
            sealed = tuple(self._sealed)
            published = tuple(self._published)
        if machine is None or not isinstance(permit, CompletedResultPermit):
            raise CheckpointV2ProtocolError(
                "checkpoint result retention lacks completed owner proof"
            )
        acknowledgement = machine.acknowledge_retention(
            owner_epoch_observed=permit.owner_epoch,
            generations=sealed,
            upload_intent_id=permit.upload_intent_id,
        )
        if (
            not acknowledgement.result_evidence_release
            or acknowledgement.upload_intent_id != permit.upload_intent_id
            or acknowledgement.submission_id is None
        ):
            raise CheckpointV2ProtocolError(
                "server did not release the exact completed result evidence"
            )
        apply_checkpoint_generation_retention_v2(
            self.data_plane.storage_root,
            acknowledgement,
            published,
        )
        return acknowledgement.submission_id


__all__ = [
    "AuthoritativeCheckpointRunV2",
    "CheckpointV2OwnerLost",
    "reconcile_orphaned_paid_gate_v2",
]
