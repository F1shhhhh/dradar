"""Crash-tolerant, non-authoritative Checkpoint V2 observation delivery.

The checkpoint writer is an optional shadow experiment.  Its telemetry must
therefore obey a stricter availability rule than ordinary command journals:
recording or sending an observation may fail, but it may never delay or veto
the paid run, its result upload, or assignment cleanup.

``CheckpointObservationReporterV2.record`` is deliberately non-blocking.  It
accepts only the small, reviewed wire schema and places a canonical copy into
a bounded in-memory queue.  A daemon thread persists records as private,
one-file-per-operation spool entries and retries them across client restarts.
Only an explicit server acknowledgement proving that the assignment was not
changed removes a pending entry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

from .api_client import ApiError


OBSERVATION_SPOOL_SCHEMA_V2 = "dradar-checkpoint-v2-observation-spool-v1"
COHORT_REGISTRY_SCHEMA_V2 = "dradar-checkpoint-v2-cohort-registration-v1"
DELIVERY_HEALTH_SCHEMA_V2 = "dradar-checkpoint-v2-delivery-health-v1"
MAINLINE_IMPACT_SCHEMA_V2 = "dradar-checkpoint-v2-mainline-impact-v1"
LOCAL_EVIDENCE_SCHEMA_V2 = "dradar-checkpoint-v2-local-evidence-v1"
EVIDENCE_ATTESTATION_SCHEMA_V2 = (
    "dradar-checkpoint-v2-evidence-attestation-v1"
)
MAX_OBSERVATION_RECORD_BYTES_V2 = 16 * 1024
MAX_PENDING_OBSERVATIONS_V2 = 512
MAX_PENDING_OBSERVATION_BYTES_V2 = 8 * 1024 * 1024
MAX_REJECTED_OBSERVATIONS_V2 = 512
MAX_REJECTED_OBSERVATION_BYTES_V2 = 8 * 1024 * 1024
MAX_OBSERVATION_SPOOL_FILES_V2 = 16_384
MAX_OBSERVATION_SPOOL_BYTES_V2 = 64 * 1024 * 1024

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]{8,64}")
_CAPTURE_WIRE_FIELDS = frozenset({
    "observation_kind",
    "assignment_id",
    "operation_id",
    "capture_id",
    "checkpoint_id",
    "checkpoint_lineage_id",
    "snapshot_generation",
    "rollout_mode",
    "status",
    "stage",
    "failure_code",
    "failure_type",
    "identity_fingerprint",
    "checkpoint_core_abi",
    "checkpoint_abi",
    "capture_storage",
    "manifest_sha256",
    "archive_sha256",
    "archive_bytes",
    "file_count",
    "payload_bytes",
    "elapsed_ms",
    "platform",
    "container_backend",
    "client_version",
    "adapter_version",
    "remote_cleanup",
    "authoritative",
    "selected_local",
})
_RESTORE_WIRE_FIELDS = frozenset({
    "observation_kind",
    "assignment_id",
    "operation_id",
    "restore_id",
    "source_capture_id",
    "checkpoint_id",
    "checkpoint_lineage_id",
    "snapshot_generation",
    "rollout_mode",
    "status",
    "stage",
    "failure_code",
    "failure_type",
    "identity_fingerprint",
    "checkpoint_core_abi",
    "checkpoint_abi",
    "manifest_sha256",
    "elapsed_ms",
    "platform",
    "container_backend",
    "client_version",
    "adapter_version",
    "paid_execution_started",
    "authoritative",
})
_STABLE_REJECTION_STATUS = frozenset({400, 409, 410, 422})
_RETRYABLE_APPLICATION_CODES = frozenset({
    # A fleet or cohort downgrade is reversible.  The sealed local record must
    # remain in the private outbox so it can be published if the experiment is
    # re-enabled; treating these 409s as semantic corruption would discard the
    # most useful incident-boundary evidence.
    "checkpoint_v2_kill_switch_active",
    "checkpoint_observation_not_authorized",
    "checkpoint_restore_observation_not_authorized",
    "checkpoint_restore_source_pending",
})
CHECKPOINT_COHORT_FIELDS_V2 = (
    "platform",
    "container_backend",
    "harness",
    "provider",
    "client_version",
    "agent_version",
    "runtime_profile",
    "model_config_version",
    "runtime_compatibility_digest",
    "checkpoint_core_abi",
    "checkpoint_abi",
)
_DELIVERY_FIELDS = (
    "persisted", "acknowledged", "retryable", "rejected", "dropped",
)
_MAINLINE_ASSIGNMENT_FIELDS = (
    "assignment_id",
    "checkpoint_protocol_version",
    "checkpoint_id",
    "execution_state",
    "owner_epoch",
    "resume_generation",
)
_MAINLINE_OUTCOMES = frozenset({"completed", "interrupted"})


class CheckpointObservationSpoolError(RuntimeError):
    """The local observation spool rejected an unsafe or conflicting entry."""


class _ObservationApi(Protocol):
    def checkpoint_v2_observation(
        self, payload: dict[str, Any],
    ) -> dict[str, Any]: ...

    def checkpoint_v2_restore_observation(
        self, payload: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ObservationDeliveryResultV2:
    persisted: int = 0
    acknowledged: int = 0
    retryable: int = 0
    rejected: int = 0
    dropped: int = 0

    def plus(self, other: "ObservationDeliveryResultV2") -> "ObservationDeliveryResultV2":
        return ObservationDeliveryResultV2(
            persisted=self.persisted + other.persisted,
            acknowledged=self.acknowledged + other.acknowledged,
            retryable=self.retryable + other.retryable,
            rejected=self.rejected + other.rejected,
            dropped=self.dropped + other.dropped,
        )


def _canonical_record(payload: dict[str, Any]) -> bytes:
    if not isinstance(payload, dict):
        raise CheckpointObservationSpoolError(
            "checkpoint observation wire fields are invalid",
        )
    kind = payload.get("observation_kind")
    expected_fields = (
        _CAPTURE_WIRE_FIELDS if kind == "capture"
        else _RESTORE_WIRE_FIELDS if kind == "restore"
        else None
    )
    if expected_fields is None or set(payload) != expected_fields:
        raise CheckpointObservationSpoolError(
            "checkpoint observation wire fields are invalid",
        )
    for key, value in payload.items():
        if value is not None and not isinstance(value, (str, int, bool)):
            raise CheckpointObservationSpoolError(
                "checkpoint observation contains a composite value",
            )
        if isinstance(value, str) and (
            len(value) > 200 or "\x00" in value or "\r" in value or "\n" in value
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint observation contains unbounded text",
            )
    identity_keys = [
        "assignment_id", "operation_id", "checkpoint_id",
        "checkpoint_lineage_id",
    ]
    identity_keys.extend(
        ["capture_id"]
        if kind == "capture"
        else ["restore_id", "source_capture_id"]
    )
    for key in identity_keys:
        value = payload.get(key)
        if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
            raise CheckpointObservationSpoolError(
                f"checkpoint observation {key} is invalid",
            )
    if payload.get("checkpoint_core_abi") != "dradar-checkpoint-core-v2/1":
        raise CheckpointObservationSpoolError(
            "checkpoint observation core ABI is invalid",
        )
    record = {
        "schema": OBSERVATION_SPOOL_SCHEMA_V2,
        "payload": payload,
    }
    try:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint observation is not canonical JSON",
        ) from exc
    if len(encoded) > MAX_OBSERVATION_RECORD_BYTES_V2:
        raise CheckpointObservationSpoolError(
            "checkpoint observation record is too large",
        )
    return encoded


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def checkpoint_mainline_assignment_digest_v2(
    assignment: dict[str, Any],
) -> str:
    """Hash only checkpoint authority fields; never persist task/user data."""

    if not isinstance(assignment, dict):
        raise CheckpointObservationSpoolError(
            "checkpoint mainline assignment snapshot is invalid",
        )
    assignment_id = assignment.get("assignment_id")
    if (
        not isinstance(assignment_id, str)
        or _IDENTIFIER_RE.fullmatch(assignment_id) is None
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint mainline assignment identity is invalid",
        )
    snapshot = {field: assignment.get(field) for field in _MAINLINE_ASSIGNMENT_FIELDS}
    for field, value in snapshot.items():
        if value is not None and not isinstance(value, (str, int)):
            raise CheckpointObservationSpoolError(
                f"checkpoint mainline assignment {field} is invalid",
            )
        if isinstance(value, bool):
            raise CheckpointObservationSpoolError(
                f"checkpoint mainline assignment {field} is invalid",
            )
        if isinstance(value, str) and (
            len(value) > 200 or "\x00" in value or "\r" in value or "\n" in value
        ):
            raise CheckpointObservationSpoolError(
                f"checkpoint mainline assignment {field} is invalid",
            )
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _mainline_sample_id(
    assignment_id: str,
    runner_session_id: str,
    identity_fingerprint: str,
    attempt_id: str,
) -> str:
    encoded = (
        f"dradar-checkpoint-v2-mainline:{assignment_id}:"
        f"{runner_session_id}:{identity_fingerprint}:{attempt_id}"
    ).encode("ascii")
    return f"impact-{hashlib.sha256(encoded).hexdigest()[:48]}"


def _canonical_mainline_impact(value: dict[str, Any]) -> bytes:
    common = {
        "schema", "sample_id", "assignment_id", "runner_session_id",
        "identity_fingerprint", "attempt_id", "started_at", "state",
        "assignment_state_sha256",
    }
    state = value.get("state") if isinstance(value, dict) else None
    completed = {
        "completed_at", "mainline_elapsed_ms", "checkpoint_sync_elapsed_ms",
        "outcome", "result_preserved", "assignment_state_sha256_after",
        "submission_preserved", "comparable", "exclusion_reason",
    }
    expected = common if state == "started" else common | completed
    if (
        not isinstance(value, dict)
        or state not in {"started", "completed"}
        or set(value) != expected
        or value.get("schema") != MAINLINE_IMPACT_SCHEMA_V2
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint mainline impact fields are invalid",
        )
    for field in (
        "sample_id", "assignment_id", "runner_session_id", "attempt_id",
    ):
        item = value[field]
        if not isinstance(item, str) or _IDENTIFIER_RE.fullmatch(item) is None:
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact identity is invalid",
            )
    for field in ("identity_fingerprint", "assignment_state_sha256"):
        item = value[field]
        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(ch not in "0123456789abcdef" for ch in item)
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact digest is invalid",
            )
    if value["sample_id"] != _mainline_sample_id(
        value["assignment_id"],
        value["runner_session_id"],
        value["identity_fingerprint"],
        value["attempt_id"],
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint mainline impact sample identity is invalid",
        )
    try:
        started_at = datetime.fromisoformat(value["started_at"])
    except (TypeError, ValueError) as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint mainline impact timestamp is invalid",
        ) from exc
    if started_at.tzinfo is None:
        raise CheckpointObservationSpoolError(
            "checkpoint mainline impact timestamp is invalid",
        )
    if state == "completed":
        after = value["assignment_state_sha256_after"]
        if (
            not isinstance(after, str)
            or len(after) != 64
            or any(ch not in "0123456789abcdef" for ch in after)
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact digest is invalid",
            )
        try:
            completed_at = datetime.fromisoformat(value["completed_at"])
        except (TypeError, ValueError) as exc:
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact timestamp is invalid",
            ) from exc
        if completed_at.tzinfo is None or completed_at < started_at:
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact timestamp is invalid",
            )
        for field in ("mainline_elapsed_ms", "checkpoint_sync_elapsed_ms"):
            metric = value[field]
            if (
                not isinstance(metric, int)
                or isinstance(metric, bool)
                or not 0 <= metric <= 7 * 24 * 60 * 60 * 1000
            ):
                raise CheckpointObservationSpoolError(
                    "checkpoint mainline impact duration is invalid",
                )
        if value["mainline_elapsed_ms"] <= 0:
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact duration is invalid",
            )
        if value["outcome"] not in _MAINLINE_OUTCOMES or any(
            not isinstance(value[field], bool)
            for field in (
                "result_preserved", "submission_preserved", "comparable",
            )
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact result is invalid",
            )
        exclusion_reason = value["exclusion_reason"]
        if (
            value["comparable"] is True and exclusion_reason is not None
        ) or (
            value["comparable"] is False
            and (
                not isinstance(exclusion_reason, str)
                or _IDENTIFIER_RE.fullmatch(exclusion_reason) is None
            )
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact exclusion is invalid",
            )
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > MAX_OBSERVATION_RECORD_BYTES_V2:
        raise CheckpointObservationSpoolError(
            "checkpoint mainline impact record is too large",
        )
    return encoded


def _canonical_cohort_registration(value: dict[str, Any]) -> bytes:
    expected = {
        "schema", "assignment_id", "runner_session_id",
        "identity_fingerprint", "registered_at", "cohort",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise CheckpointObservationSpoolError(
            "checkpoint cohort registration fields are invalid",
        )
    if value["schema"] != COHORT_REGISTRY_SCHEMA_V2:
        raise CheckpointObservationSpoolError(
            "checkpoint cohort registration schema is invalid",
        )
    for field in ("assignment_id", "runner_session_id"):
        item = value[field]
        if not isinstance(item, str) or _IDENTIFIER_RE.fullmatch(item) is None:
            raise CheckpointObservationSpoolError(
                "checkpoint cohort registration identity is invalid",
            )
    fingerprint = value["identity_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(ch not in "0123456789abcdef" for ch in fingerprint)
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint cohort fingerprint is invalid",
        )
    try:
        registered_at = datetime.fromisoformat(value["registered_at"])
    except (TypeError, ValueError) as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint cohort timestamp is invalid",
        ) from exc
    if registered_at.tzinfo is None:
        raise CheckpointObservationSpoolError(
            "checkpoint cohort timestamp is invalid",
        )
    cohort = value["cohort"]
    if (
        not isinstance(cohort, dict)
        or set(cohort) != set(CHECKPOINT_COHORT_FIELDS_V2)
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint cohort tuple is invalid",
        )
    for field, item in cohort.items():
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 160
            or "\x00" in item
            or "\r" in item
            or "\n" in item
        ):
            raise CheckpointObservationSpoolError(
                f"checkpoint cohort {field} is invalid",
            )
    if cohort["checkpoint_core_abi"] != "dradar-checkpoint-core-v2/1":
        raise CheckpointObservationSpoolError(
            "checkpoint cohort core ABI is invalid",
        )
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint cohort registration is not canonical JSON",
        ) from exc
    if len(encoded) > MAX_OBSERVATION_RECORD_BYTES_V2:
        raise CheckpointObservationSpoolError(
            "checkpoint cohort registration is too large",
        )
    return encoded


def _canonical_delivery_health(value: dict[str, Any]) -> bytes:
    expected = {
        "schema", "first_observed_at", "last_observed_at",
        *_DELIVERY_FIELDS,
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise CheckpointObservationSpoolError(
            "checkpoint delivery health fields are invalid",
        )
    if value["schema"] != DELIVERY_HEALTH_SCHEMA_V2:
        raise CheckpointObservationSpoolError(
            "checkpoint delivery health schema is invalid",
        )
    for field in _DELIVERY_FIELDS:
        metric = value[field]
        if not isinstance(metric, int) or isinstance(metric, bool) or metric < 0:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health metric is invalid",
            )
    for field in ("first_observed_at", "last_observed_at"):
        try:
            parsed = datetime.fromisoformat(value[field])
        except (TypeError, ValueError) as exc:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health timestamp is invalid",
            ) from exc
        if parsed.tzinfo is None:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health timestamp is invalid",
            )
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > MAX_OBSERVATION_RECORD_BYTES_V2:
        raise CheckpointObservationSpoolError(
            "checkpoint delivery health is too large",
        )
    return encoded


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _private_directory(path: Path, *, create: bool) -> None:
    created = False
    if create and not path.exists():
        try:
            path.mkdir(parents=True, mode=0o700)
            created = True
        except FileExistsError:
            pass
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool is unavailable",
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool is unsafe",
        )
    if created:
        try:
            os.chmod(path, 0o700)
            metadata = path.lstat()
        except OSError as exc:
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool permissions failed",
            ) from exc
    if hasattr(os, "getuid") and (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool is not private",
        )


@contextmanager
def _process_lock(root: Path) -> Iterator[None]:
    _private_directory(root, create=True)
    path = root / ".lock"
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool lock failed",
        ) from exc
    windows_lock = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool lock is unsafe",
            )
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - exercised on Windows CI
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            windows_lock = True
        yield
    finally:
        if windows_lock:  # pragma: no cover - exercised on Windows CI
            try:
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        os.close(descriptor)


@contextmanager
def _try_process_lock(root: Path) -> Iterator[bool]:
    """Try the shared evidence lock once; optional mainline work never waits."""

    _private_directory(root, create=True)
    path = root / ".lock"
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool lock failed",
        ) from exc
    windows_lock = False
    acquired = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool lock is unsafe",
            )
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
        except ImportError:  # pragma: no cover - exercised on Windows CI
            import msvcrt

            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                windows_lock = True
                acquired = True
            except OSError:
                acquired = False
        yield acquired
    finally:
        if acquired and windows_lock:  # pragma: no cover - Windows CI
            try:
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        elif acquired:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        os.close(descriptor)


def _read_private_record(path: Path) -> tuple[dict[str, Any], bytes]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > MAX_OBSERVATION_RECORD_BYTES_V2
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool entry is unsafe",
            )
        if hasattr(os, "getuid") and (
            before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool entry is not private",
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool entry changed",
            )
        chunks: list[bytes] = []
        remaining = MAX_OBSERVATION_RECORD_BYTES_V2 + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 16 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(encoded) > MAX_OBSERVATION_RECORD_BYTES_V2
            or len(encoded) != after.st_size
            or (after.st_dev, after.st_ino, after.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_mtime_ns)
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint observation spool entry changed",
            )
        value = json.loads(encoded)
    except CheckpointObservationSpoolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool entry is unreadable",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not isinstance(value, dict)
        or value.get("schema") != OBSERVATION_SPOOL_SCHEMA_V2
        or not isinstance(value.get("payload"), dict)
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool schema is unsupported",
        )
    canonical = _canonical_record(value["payload"])
    if canonical != encoded:
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool entry is not canonical",
        )
    return value["payload"], encoded


def _atomic_private_record(path: Path, encoded: bytes) -> None:
    _private_directory(path.parent, create=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short checkpoint evidence write")
            view = view[written:]
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint evidence record could not be persisted",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_private_json(path: Path) -> tuple[dict[str, Any], bytes]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > MAX_OBSERVATION_RECORD_BYTES_V2
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint evidence record is unsafe",
            )
        if hasattr(os, "getuid") and (
            before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint evidence record is not private",
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint evidence record changed",
            )
        chunks: list[bytes] = []
        remaining = MAX_OBSERVATION_RECORD_BYTES_V2 + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 16 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(encoded) > MAX_OBSERVATION_RECORD_BYTES_V2
            or len(encoded) != after.st_size
            or (after.st_dev, after.st_ino, after.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_mtime_ns)
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint evidence record changed",
            )
        value = json.loads(encoded)
    except CheckpointObservationSpoolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint evidence record is unreadable",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise CheckpointObservationSpoolError(
            "checkpoint evidence record is not an object",
        )
    return value, encoded


def _private_spool_usage(root: Path) -> tuple[int, int]:
    """Return exact private-tree usage or refuse an unsafe/unreadable tree."""

    files = 0
    total_bytes = 0
    try:
        for current, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False,
        ):
            current_path = Path(current)
            for name in directory_names:
                path = current_path / name
                metadata = path.lstat()
                if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                    raise CheckpointObservationSpoolError(
                        "checkpoint observation spool tree is unsafe",
                    )
                if hasattr(os, "getuid") and (
                    metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o077
                ):
                    raise CheckpointObservationSpoolError(
                        "checkpoint observation spool tree is not private",
                    )
            for name in file_names:
                path = current_path / name
                metadata = path.lstat()
                if (
                    path.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise CheckpointObservationSpoolError(
                        "checkpoint observation spool tree is unsafe",
                    )
                if hasattr(os, "getuid") and (
                    metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    raise CheckpointObservationSpoolError(
                        "checkpoint observation spool tree is not private",
                    )
                files += 1
                total_bytes += metadata.st_size
    except CheckpointObservationSpoolError:
        raise
    except OSError as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint observation spool tree is unreadable",
        ) from exc
    return files, total_bytes


class CheckpointObservationSpoolV2:
    """A bounded private outbox, safe for multiple local worker processes."""

    def __init__(
        self,
        root: Path,
        *,
        max_pending: int = MAX_PENDING_OBSERVATIONS_V2,
        max_pending_bytes: int = MAX_PENDING_OBSERVATION_BYTES_V2,
        max_rejected: int = MAX_REJECTED_OBSERVATIONS_V2,
        max_rejected_bytes: int = MAX_REJECTED_OBSERVATION_BYTES_V2,
        max_total_files: int = MAX_OBSERVATION_SPOOL_FILES_V2,
        max_total_bytes: int = MAX_OBSERVATION_SPOOL_BYTES_V2,
    ) -> None:
        if not 1 <= max_pending <= 10_000:
            raise ValueError("checkpoint observation pending limit is invalid")
        if not MAX_OBSERVATION_RECORD_BYTES_V2 <= max_pending_bytes <= 64 * 1024 * 1024:
            raise ValueError("checkpoint observation byte limit is invalid")
        if not 1 <= max_rejected <= 10_000:
            raise ValueError("checkpoint observation rejected limit is invalid")
        if not MAX_OBSERVATION_RECORD_BYTES_V2 <= max_rejected_bytes <= 64 * 1024 * 1024:
            raise ValueError(
                "checkpoint observation rejected byte limit is invalid"
            )
        if not 2 <= max_total_files <= 100_000:
            raise ValueError("checkpoint observation total file limit is invalid")
        if not 2 * MAX_OBSERVATION_RECORD_BYTES_V2 <= max_total_bytes <= 1024 * 1024 * 1024:
            raise ValueError("checkpoint observation total byte limit is invalid")
        self.root = Path(root).absolute()
        self.pending_root = self.root / "pending"
        self.rejected_root = self.root / "rejected"
        self.cohort_root = self.root / "cohorts"
        self.delivery_health_root = self.root / "delivery-health"
        self.mainline_impact_root = self.root / "mainline-impact"
        self.max_pending = max_pending
        self.max_pending_bytes = max_pending_bytes
        self.max_rejected = max_rejected
        self.max_rejected_bytes = max_rejected_bytes
        self.max_total_files = max_total_files
        self.max_total_bytes = max_total_bytes
        self._thread_lock = threading.Lock()

    def _prepare(self) -> None:
        _private_directory(self.root, create=True)
        _private_directory(self.pending_root, create=True)
        _private_directory(self.rejected_root, create=True)
        _private_directory(self.cohort_root, create=True)
        _private_directory(self.delivery_health_root, create=True)
        _private_directory(self.mainline_impact_root, create=True)

    def begin_mainline_impact(self, payload: dict[str, Any]) -> str:
        """Crash-durably open one shadow/mainline pair without waiting on locks."""

        value = dict(payload)
        value.update({
            "schema": MAINLINE_IMPACT_SCHEMA_V2,
            "state": "started",
        })
        assignment_id = value.get("assignment_id")
        session_id = value.get("runner_session_id")
        fingerprint = value.get("identity_fingerprint")
        attempt_id = value.get("attempt_id")
        if not all(isinstance(item, str) for item in (
            assignment_id, session_id, fingerprint, attempt_id,
        )):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact identity is invalid",
            )
        value["sample_id"] = _mainline_sample_id(
            assignment_id, session_id, fingerprint, attempt_id,
        )
        value.setdefault("started_at", _now_iso())
        encoded = _canonical_mainline_impact(value)
        if not self._thread_lock.acquire(blocking=False):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact spool is busy",
            )
        try:
            with _try_process_lock(self.root) as acquired:
                if not acquired:
                    raise CheckpointObservationSpoolError(
                        "checkpoint mainline impact spool is busy",
                    )
                self._prepare()
                target = self.mainline_impact_root / f"{value['sample_id']}.json"
                if target.exists() or target.is_symlink():
                    existing, raw = _read_private_json(target)
                    if (
                        _canonical_mainline_impact(existing) == raw
                        and all(
                            existing.get(field) == value.get(field)
                            for field in (
                                "sample_id", "assignment_id", "runner_session_id",
                                "identity_fingerprint", "attempt_id",
                                "assignment_state_sha256",
                            )
                        )
                    ):
                        self._assert_total_capacity_unlocked(
                            additional_files=0, additional_bytes=0,
                        )
                        return value["sample_id"]
                    raise CheckpointObservationSpoolError(
                        "checkpoint mainline impact sample conflicts",
                    )
                self._assert_total_capacity_unlocked(
                    additional_files=1, additional_bytes=len(encoded),
                )
                _atomic_private_record(target, encoded)
                return value["sample_id"]
        finally:
            self._thread_lock.release()

    def complete_mainline_impact(
        self,
        sample_id: str,
        payload: dict[str, Any],
    ) -> bool:
        """Complete one exact pair; crashes leave the start marker visible."""

        if not isinstance(sample_id, str) or _IDENTIFIER_RE.fullmatch(sample_id) is None:
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact sample identity is invalid",
            )
        completion_fields = {
            "mainline_elapsed_ms", "checkpoint_sync_elapsed_ms", "outcome",
            "result_preserved", "assignment_state_sha256_after",
            "submission_preserved", "comparable", "exclusion_reason",
        }
        payload_fields = frozenset(payload) if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload_fields not in {
                frozenset(completion_fields),
                frozenset(completion_fields | {"completed_at"}),
            }
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact completion fields are invalid",
            )
        if not self._thread_lock.acquire(blocking=False):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact spool is busy",
            )
        try:
            with _try_process_lock(self.root) as acquired:
                if not acquired:
                    raise CheckpointObservationSpoolError(
                        "checkpoint mainline impact spool is busy",
                    )
                self._prepare()
                target = self.mainline_impact_root / f"{sample_id}.json"
                existing, raw = _read_private_json(target)
                if _canonical_mainline_impact(existing) != raw:
                    raise CheckpointObservationSpoolError(
                        "checkpoint mainline impact sample changed",
                    )
                completed = {
                    **existing,
                    **dict(payload),
                    "schema": MAINLINE_IMPACT_SCHEMA_V2,
                    "sample_id": sample_id,
                    "state": "completed",
                }
                completed.setdefault("completed_at", _now_iso())
                encoded = _canonical_mainline_impact(completed)
                if existing.get("state") == "completed":
                    if raw == encoded:
                        self._assert_total_capacity_unlocked(
                            additional_files=0, additional_bytes=0,
                        )
                        return False
                    raise CheckpointObservationSpoolError(
                        "checkpoint mainline impact completion conflicts",
                    )
                self._assert_total_capacity_unlocked(
                    additional_files=0,
                    additional_bytes=max(0, len(encoded) - len(raw)),
                )
                _atomic_private_record(target, encoded)
                return True
        finally:
            self._thread_lock.release()

    def register_cohort(self, payload: dict[str, Any]) -> bool:
        """Persist one exact finalized runtime/session tuple for local audit."""

        value = dict(payload)
        value["schema"] = COHORT_REGISTRY_SCHEMA_V2
        assignment_id = value.get("assignment_id")
        session_id = value.get("runner_session_id")
        if (
            not isinstance(assignment_id, str)
            or _IDENTIFIER_RE.fullmatch(assignment_id) is None
            or not isinstance(session_id, str)
            or _IDENTIFIER_RE.fullmatch(session_id) is None
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint cohort registration identity is invalid",
            )
        with self._thread_lock, _process_lock(self.root):
            self._prepare()
            target = self.cohort_root / f"{assignment_id}.{session_id}.json"
            if target.exists() or target.is_symlink():
                existing, raw = _read_private_json(target)
                value["registered_at"] = existing.get("registered_at")
                encoded = _canonical_cohort_registration(value)
                if _canonical_cohort_registration(existing) == encoded == raw:
                    self._assert_total_capacity_unlocked(
                        additional_files=0, additional_bytes=0,
                    )
                    return False
                raise CheckpointObservationSpoolError(
                    "checkpoint cohort registration conflicts",
                )
            value.setdefault("registered_at", _now_iso())
            encoded = _canonical_cohort_registration(value)
            self._assert_total_capacity_unlocked(
                additional_files=1, additional_bytes=len(encoded),
            )
            _atomic_private_record(target, encoded)
            return True

    def record_delivery_health(
        self,
        assignment_id: str,
        delta: ObservationDeliveryResultV2,
    ) -> None:
        """Atomically persist assignment-scoped delivery counters.

        These counters are evidence, not control state.  Keeping every field
        crash-persistent prevents a client restart from making a lossy outbox
        look healthy merely because its in-memory statistics were reset.
        """

        if _IDENTIFIER_RE.fullmatch(assignment_id) is None:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery assignment is invalid",
            )
        if not isinstance(delta, ObservationDeliveryResultV2):
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health delta is invalid",
            )
        raw_values = {field: getattr(delta, field) for field in _DELIVERY_FIELDS}
        if (
            any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in raw_values.values()
            )
            or not any(raw_values.values())
            or any(value > 1_000_000 for value in raw_values.values())
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health delta is invalid",
            )
        values = {field: int(value) for field, value in raw_values.items()}
        with self._thread_lock, _process_lock(self.root):
            self._prepare()
            now = _now_iso()
            target = self.delivery_health_root / f"{assignment_id}.json"
            target_exists = target.exists() or target.is_symlink()
            if target_exists:
                current, raw = _read_private_json(target)
                if _canonical_delivery_health(current) != raw:
                    raise CheckpointObservationSpoolError(
                        "checkpoint delivery health changed",
                    )
            else:
                current = {
                    "schema": DELIVERY_HEALTH_SCHEMA_V2,
                    "first_observed_at": now,
                    "last_observed_at": now,
                    **{field: 0 for field in _DELIVERY_FIELDS},
                }
            updated = {**current, "last_observed_at": now}
            for field, value in values.items():
                total = int(current[field]) + value
                if total > 2**63 - 1:
                    raise CheckpointObservationSpoolError(
                        "checkpoint delivery health counter overflowed",
                    )
                updated[field] = total
            encoded = _canonical_delivery_health(updated)
            self._assert_total_capacity_unlocked(
                additional_files=0 if target_exists else 1,
                additional_bytes=(
                    max(0, len(encoded) - len(raw))
                    if target_exists else len(encoded)
                ),
            )
            _atomic_private_record(target, encoded)

    def record_delivery_drops(
        self, assignment_id: str, count: int,
    ) -> None:
        """Compatibility wrapper for assignment-scoped validation drops."""

        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery drop count is invalid",
            )
        self.record_delivery_health(
            assignment_id,
            ObservationDeliveryResultV2(dropped=count),
        )

    def _pending_paths_unlocked(self) -> list[Path]:
        self._prepare()
        paths: list[tuple[int, str, Path]] = []
        for path in self.pending_root.iterdir():
            if path.name.startswith(".") or path.suffix != ".json":
                continue
            try:
                metadata = path.lstat()
            except OSError:
                continue
            paths.append((metadata.st_mtime_ns, path.name, path))
        paths.sort()
        return [path for _, _, path in paths]

    def _rejected_paths_unlocked(self) -> list[Path]:
        self._prepare()
        paths: list[tuple[int, str, Path]] = []
        for path in self.rejected_root.iterdir():
            if path.name.startswith(".") or path.suffix != ".json":
                continue
            try:
                metadata = path.lstat()
            except OSError:
                continue
            paths.append((metadata.st_mtime_ns, path.name, path))
        paths.sort()
        return [path for _, _, path in paths]

    def _assert_total_capacity_unlocked(
        self, *, additional_files: int, additional_bytes: int,
    ) -> None:
        """Bound every evidence file, including crash-left hidden temporaries."""

        if additional_files < 0 or additional_bytes < 0:
            raise CheckpointObservationSpoolError(
                "checkpoint observation capacity delta is invalid",
            )
        self._prepare()
        files, total_bytes = _private_spool_usage(self.root)
        if (
            files + additional_files > self.max_total_files
            or total_bytes + additional_bytes > self.max_total_bytes
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint observation total spool is full",
            )

    def persist(self, payload: dict[str, Any]) -> bool:
        """Persist once; return False for an exact already-persisted replay."""

        encoded = _canonical_record(payload)
        operation_id = str(payload["operation_id"])
        with self._thread_lock, _process_lock(self.root):
            self._prepare()
            target = self.pending_root / f"{operation_id}.json"
            if target.exists() or target.is_symlink():
                _, existing = _read_private_record(target)
                if existing == encoded:
                    self._assert_total_capacity_unlocked(
                        additional_files=0, additional_bytes=0,
                    )
                    return False
                raise CheckpointObservationSpoolError(
                    "checkpoint observation operation id conflicts",
                )
            paths = self._pending_paths_unlocked()
            total = 0
            for path in paths:
                try:
                    metadata = path.lstat()
                except OSError:
                    continue
                total += metadata.st_size
            if len(paths) >= self.max_pending or total + len(encoded) > self.max_pending_bytes:
                raise CheckpointObservationSpoolError(
                    "checkpoint observation spool is full",
                )
            self._assert_total_capacity_unlocked(
                additional_files=1, additional_bytes=len(encoded),
            )
            temporary = self.pending_root / f".{operation_id}.{uuid.uuid4().hex}.tmp"
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short checkpoint observation spool write")
                    view = view[written:]
                if hasattr(os, "fchmod"):
                    os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                os.replace(temporary, target)
                _fsync_directory(self.pending_root)
            except OSError as exc:
                raise CheckpointObservationSpoolError(
                    "checkpoint observation could not be persisted",
                ) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            return True

    def pending(self, *, limit: int = 16) -> list[tuple[Path, dict[str, Any]]]:
        if not 1 <= limit <= 512:
            raise ValueError("checkpoint observation replay limit is invalid")
        result: list[tuple[Path, dict[str, Any]]] = []
        with self._thread_lock, _process_lock(self.root):
            for path in self._pending_paths_unlocked()[:limit]:
                payload, _ = _read_private_record(path)
                result.append((path, payload))
        return result

    def acknowledge(self, path: Path, operation_id: str) -> None:
        expected = self.pending_root / f"{operation_id}.json"
        if Path(path) != expected:
            raise CheckpointObservationSpoolError(
                "checkpoint observation acknowledgement target is invalid",
            )
        with self._thread_lock, _process_lock(self.root):
            payload, _ = _read_private_record(expected)
            if payload.get("operation_id") != operation_id:
                raise CheckpointObservationSpoolError(
                    "checkpoint observation acknowledgement identity changed",
                )
            expected.unlink()
            _fsync_directory(self.pending_root)

    def reject(self, path: Path, operation_id: str) -> None:
        expected = self.pending_root / f"{operation_id}.json"
        if Path(path) != expected:
            raise CheckpointObservationSpoolError(
                "checkpoint observation rejection target is invalid",
            )
        with self._thread_lock, _process_lock(self.root):
            payload, _ = _read_private_record(expected)
            if payload.get("operation_id") != operation_id:
                raise CheckpointObservationSpoolError(
                    "checkpoint observation rejection identity changed",
                )
            self._prepare()
            target = self.rejected_root / expected.name
            if target.exists() or target.is_symlink():
                _, pending_raw = _read_private_record(expected)
                _, rejected_raw = _read_private_record(target)
                if pending_raw != rejected_raw:
                    raise CheckpointObservationSpoolError(
                        "checkpoint observation rejection identity conflicts",
                    )
                expected.unlink()
                _fsync_directory(self.pending_root)
                return
            rejected = self._rejected_paths_unlocked()
            rejected_bytes = 0
            for item in rejected:
                try:
                    rejected_bytes += item.lstat().st_size
                except OSError:
                    continue
            pending_bytes = expected.lstat().st_size
            if (
                len(rejected) >= self.max_rejected
                or rejected_bytes + pending_bytes > self.max_rejected_bytes
            ):
                raise CheckpointObservationSpoolError(
                    "checkpoint observation rejected spool is full",
                )
            os.replace(expected, target)
            _fsync_directory(self.pending_root)
            _fsync_directory(self.rejected_root)


class CheckpointObservationReporterV2:
    """Best-effort background delivery with a crash-persistent local outbox."""

    def __init__(
        self,
        client: _ObservationApi,
        home: Path,
        *,
        queue_size: int = 128,
        replay_batch: int = 16,
        idle_retry_sec: float = 30.0,
    ) -> None:
        if not 1 <= queue_size <= 4096:
            raise ValueError("checkpoint observation queue size is invalid")
        if not 1 <= replay_batch <= 512:
            raise ValueError("checkpoint observation replay batch is invalid")
        if not 0.1 <= idle_retry_sec <= 3600:
            raise ValueError("checkpoint observation retry delay is invalid")
        self.client = client
        self.spool = CheckpointObservationSpoolV2(
            Path(home) / "checkpoint-v2" / "observations",
        )
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_size)
        self._replay_batch = replay_batch
        self._idle_retry_sec = float(idle_retry_sec)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._stats = ObservationDeliveryResultV2()
        self._unpersisted_health: dict[str, ObservationDeliveryResultV2] = {}

    def _add_stats(self, value: ObservationDeliveryResultV2) -> None:
        with self._stats_lock:
            self._stats = self._stats.plus(value)

    def _queue_health(
        self,
        payload: object,
        delta: ObservationDeliveryResultV2,
    ) -> None:
        assignment_id = (
            payload.get("assignment_id") if isinstance(payload, dict) else None
        )
        if (
            isinstance(assignment_id, str)
            and _IDENTIFIER_RE.fullmatch(assignment_id) is not None
        ):
            with self._stats_lock:
                self._unpersisted_health[assignment_id] = (
                    self._unpersisted_health.get(
                        assignment_id, ObservationDeliveryResultV2(),
                    ).plus(delta)
                )

    def _record_drop(self, payload: object) -> None:
        delta = ObservationDeliveryResultV2(dropped=1)
        self._add_stats(delta)
        self._queue_health(payload, delta)

    def _flush_health(self) -> None:
        with self._stats_lock:
            pending = self._unpersisted_health
            self._unpersisted_health = {}
        if not pending:
            return
        failed: dict[str, ObservationDeliveryResultV2] = {}
        for assignment_id, delta in pending.items():
            try:
                self.spool.record_delivery_health(assignment_id, delta)
            except Exception:
                failed[assignment_id] = delta
        if failed:
            with self._stats_lock:
                for assignment_id, delta in failed.items():
                    self._unpersisted_health[assignment_id] = (
                        self._unpersisted_health.get(
                            assignment_id, ObservationDeliveryResultV2(),
                        ).plus(delta)
                    )

    def register_cohort(self, payload: dict[str, Any]) -> bool:
        """Persist exact finalized cohort facts; failure remains shadow-only."""

        try:
            self.spool.register_cohort(payload)
            return True
        except Exception:
            self._record_drop(payload)
            self._wake.set()
            return False

    def begin_mainline_impact(self, payload: dict[str, Any]) -> str | None:
        """Open a crash marker off the paid thread after cohort registration."""

        try:
            return self.spool.begin_mainline_impact(payload)
        except Exception:
            return None

    def complete_mainline_impact(
        self,
        sample_id: str,
        payload: dict[str, Any],
    ) -> bool:
        """Finish a pair without ever delaying on the shared evidence lock."""

        try:
            self.spool.complete_mainline_impact(sample_id, payload)
            return True
        except Exception:
            return False

    @property
    def stats(self) -> ObservationDeliveryResultV2:
        with self._stats_lock:
            return self._stats

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="dradar-checkpoint-v2-observations",
            daemon=True,
        )
        self._thread.start()

    def record(self, payload: dict[str, Any]) -> bool:
        """Queue without I/O or waiting; every failure is a quiet data drop."""

        try:
            encoded = _canonical_record(payload)
            canonical = json.loads(encoded)["payload"]
            self._queue.put_nowait(canonical)
        except (CheckpointObservationSpoolError, queue.Full, ValueError, TypeError):
            self._record_drop(payload)
            return False
        self._wake.set()
        return True

    def persist(self, payload: dict[str, Any]) -> bool:
        """Durably enqueue one shadow record outside the paid mainline.

        This path is used only by the isolated shadow coordinator when local
        snapshot retention depends on a crash-safe evidence handoff.  Exact
        replay is success: the operation is already durable.  Delivery remains
        asynchronous and never grants assignment or paid-execution authority.
        """

        try:
            encoded = _canonical_record(payload)
            canonical = json.loads(encoded)["payload"]
            created = self.spool.persist(canonical)
        except (CheckpointObservationSpoolError, ValueError, TypeError):
            self._record_drop(payload)
            self._wake.set()
            return False
        if created:
            delta = ObservationDeliveryResultV2(persisted=1)
            self._add_stats(delta)
            self._queue_health(canonical, delta)
        self._wake.set()
        return True

    def _persist_queued(self) -> ObservationDeliveryResultV2:
        result = ObservationDeliveryResultV2()
        for _ in range(self._queue.qsize()):
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                created = self.spool.persist(payload)
            except Exception:
                delta = ObservationDeliveryResultV2(dropped=1)
                result = result.plus(delta)
                self._queue_health(payload, delta)
            else:
                if created:
                    delta = ObservationDeliveryResultV2(persisted=1)
                    result = result.plus(delta)
                    self._queue_health(payload, delta)
            finally:
                self._queue.task_done()
        return result

    @staticmethod
    def _acknowledges_without_authority(
        response: object, payload: dict[str, Any],
    ) -> bool:
        kind = payload.get("observation_kind")
        identity_matches = (
            response.get("capture_id") == payload.get("capture_id")
            if kind == "capture"
            else response.get("restore_id") == payload.get("restore_id")
        ) if isinstance(response, dict) else False
        return (
            isinstance(response, dict)
            and response.get("ok") is True
            and response.get("assignment_unchanged") is True
            and response.get("paid_execution_authorized") is False
            and response.get("assignment_id") == payload.get("assignment_id")
            and identity_matches
            and response.get("status") == payload.get("status")
        )

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("observation_kind") == "capture":
            return self.client.checkpoint_v2_observation(payload)
        if payload.get("observation_kind") == "restore":
            return self.client.checkpoint_v2_restore_observation(payload)
        raise CheckpointObservationSpoolError(
            "checkpoint observation kind is invalid",
        )

    @staticmethod
    def _is_stable_rejection(exc: ApiError) -> bool:
        return (
            exc.status_code in _STABLE_REJECTION_STATUS
            and exc.code not in _RETRYABLE_APPLICATION_CODES
        )

    def flush_once(self) -> ObservationDeliveryResultV2:
        """Persist queued data and attempt a bounded replay batch."""

        result = self._persist_queued()
        try:
            pending = self.spool.pending(limit=self._replay_batch)
        except Exception:
            result = result.plus(ObservationDeliveryResultV2(retryable=1))
            self._add_stats(result)
            self._flush_health()
            return result
        for path, payload in pending:
            try:
                response = self._send(payload)
            except ApiError as exc:
                if self._is_stable_rejection(exc):
                    try:
                        self.spool.reject(path, str(payload["operation_id"]))
                    except Exception:
                        delta = ObservationDeliveryResultV2(retryable=1)
                    else:
                        delta = ObservationDeliveryResultV2(rejected=1)
                else:
                    delta = ObservationDeliveryResultV2(retryable=1)
                result = result.plus(delta)
                self._queue_health(payload, delta)
                continue
            except Exception:
                delta = ObservationDeliveryResultV2(retryable=1)
                result = result.plus(delta)
                self._queue_health(payload, delta)
                continue
            if not self._acknowledges_without_authority(response, payload):
                delta = ObservationDeliveryResultV2(retryable=1)
                result = result.plus(delta)
                self._queue_health(payload, delta)
                continue
            try:
                self.spool.acknowledge(path, str(payload["operation_id"]))
            except Exception:
                delta = ObservationDeliveryResultV2(retryable=1)
            else:
                delta = ObservationDeliveryResultV2(acknowledged=1)
            result = result.plus(delta)
            self._queue_health(payload, delta)
        self._add_stats(result)
        self._flush_health()
        return result

    def _loop(self) -> None:
        while True:
            result = self.flush_once()
            if self._stop.is_set() and self._queue.empty():
                return
            delay = 0.25 if result.persisted else self._idle_retry_sec
            self._wake.wait(delay)
            self._wake.clear()

    def close(self, *, timeout: float = 0.5) -> None:
        """Request a bounded drain; never wait long enough to delay submission."""

        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, min(float(timeout), 2.0)))


def _private_json_files(path: Path) -> list[Path]:
    _private_directory(path, create=False)
    result: list[Path] = []
    for item in path.iterdir():
        if item.name.startswith("."):
            continue
        if item.suffix != ".json":
            raise CheckpointObservationSpoolError(
                "checkpoint evidence directory contains an unknown entry",
            )
        metadata = item.lstat()
        if item.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise CheckpointObservationSpoolError(
                "checkpoint evidence directory contains an unsafe entry",
            )
        result.append(item)
    return sorted(result, key=lambda item: item.name)


def _local_diagnostic_counts(home: Path, assignment_id: str) -> tuple[int, int]:
    path = (
        home / "checkpoint-v2" / "shadow" / assignment_id
        / "diagnostics.jsonl"
    )
    if not path.exists():
        return 0, 0
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 4 * 1024 * 1024
        ):
            return 1, 1
        if hasattr(os, "getuid") and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            return 1, 1
        total = 0
        unstructured = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                total += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    unstructured += 1
                    continue
                if (
                    not isinstance(value, dict)
                    or value.get("schema")
                    != "dradar-checkpoint-local-diagnostic-v2"
                    or not isinstance(value.get("stage"), str)
                    or not isinstance(value.get("code"), str)
                    or not isinstance(value.get("failure_type"), str)
                    or not isinstance(value.get("diagnostic"), dict)
                ):
                    unstructured += 1
        return total, unstructured
    except (OSError, UnicodeError):
        return 1, 1


def _local_cleanup_residue(home: Path, assignment_id: str) -> int:
    total = 0
    for scope in ("shadow", "authoritative"):
        root = home / "checkpoint-v2" / scope / assignment_id
        for relative in (".downloads", ".restore"):
            path = root / relative
            try:
                if path.is_dir() and not path.is_symlink():
                    total += sum(1 for _item in path.iterdir())
                elif path.exists() or path.is_symlink():
                    total += 1
            except OSError:
                total += 1

        checkpoints = root / "checkpoints"
        try:
            if not checkpoints.exists() and not checkpoints.is_symlink():
                continue
            if checkpoints.is_symlink() or not checkpoints.is_dir():
                total += 1
                continue
            for checkpoint_root in checkpoints.iterdir():
                if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
                    total += 1
                    continue
                for child in checkpoint_root.iterdir():
                    if child.name in {"CURRENT", "PUBLICATION.lock", "generations"}:
                        continue
                    # Incomplete incoming/CURRENT/retention marker writes and
                    # unknown transaction artifacts are all promotion-blocking
                    # residue.  The audit is read-only and never removes them.
                    total += 1
                generations = checkpoint_root / "generations"
                if not generations.exists() and not generations.is_symlink():
                    continue
                if generations.is_symlink() or not generations.is_dir():
                    total += 1
                    continue
                total += sum(
                    1 for generation in generations.iterdir()
                    if (
                        re.fullmatch(
                            r"generation-[0-9]{20}", generation.name,
                        ) is None
                        or scope == "shadow"
                    )
                )
        except OSError:
            total += 1
    return total


def _checkpoint_local_evidence_unlocked_v2(
    home: Path,
    *,
    since: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build strict assignment-scoped outbox-health attestations locally.

    The scan is read-only and emits counts/digests only.  It never includes a
    local path, hostname, username, command, prompt, log line, or credential.
    Passing the resulting JSON to the server report remains an explicit human
    review action and cannot enable Checkpoint V2 by itself.
    """

    home = Path(home).absolute()
    root = home / "checkpoint-v2" / "observations"
    raw_instant = now or datetime.now(timezone.utc)
    if raw_instant.tzinfo is None or since is not None and since.tzinfo is None:
        raise CheckpointObservationSpoolError(
            "checkpoint local evidence window is invalid",
        )
    instant = raw_instant.astimezone(timezone.utc)
    window_start = since.astimezone(timezone.utc) if since is not None else None
    if (
        window_start is not None
        and (
            window_start >= instant
            or instant - window_start > timedelta(days=365)
        )
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint local evidence window is invalid",
        )
    if not root.exists():
        return {
            "schema": LOCAL_EVIDENCE_SCHEMA_V2,
            "generated_at": instant.replace(microsecond=0).isoformat(),
            "attestations": [],
            "unregistered_records": 0,
            "scan_errors": 0,
        }
    _private_directory(root, create=False)
    cohort_root = root / "cohorts"
    pending_root = root / "pending"
    rejected_root = root / "rejected"
    health_root = root / "delivery-health"
    impact_root = root / "mainline-impact"
    for path in (cohort_root, pending_root, rejected_root, health_root):
        if not path.exists():
            raise CheckpointObservationSpoolError(
                "checkpoint local evidence is incomplete",
            )
    spool_files, spool_bytes = _private_spool_usage(root)
    spool_capacity_pressure = int(
        spool_files * 10 >= MAX_OBSERVATION_SPOOL_FILES_V2 * 9
        or spool_bytes * 10 >= MAX_OBSERVATION_SPOOL_BYTES_V2 * 9
    )
    allowed_root_entries = {
        ".lock", "cohorts", "pending", "rejected", "delivery-health",
        "mainline-impact",
    }
    scan_errors = 0
    for item in root.iterdir():
        if item.name not in allowed_root_entries:
            scan_errors += 1
            continue
        if item.name == ".lock":
            try:
                metadata = item.lstat()
                if (
                    item.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or hasattr(os, "getuid") and (
                        metadata.st_uid != os.getuid()
                        or stat.S_IMODE(metadata.st_mode) != 0o600
                    )
                ):
                    scan_errors += 1
            except OSError:
                scan_errors += 1
    for directory in (
        cohort_root, pending_root, rejected_root, health_root, impact_root,
    ):
        if not directory.exists() and not directory.is_symlink():
            continue
        _private_directory(directory, create=False)
        scan_errors += sum(
            item.name.startswith(".") for item in directory.iterdir()
        )

    registrations: list[tuple[dict[str, Any], str]] = []
    for path in _private_json_files(cohort_root):
        value, encoded = _read_private_json(path)
        if _canonical_cohort_registration(value) != encoded:
            raise CheckpointObservationSpoolError(
                "checkpoint cohort registration changed",
            )
        registrations.append((value, hashlib.sha256(encoded).hexdigest()))

    observation_records: list[tuple[str, dict[str, Any], str]] = []
    for state, directory in (("pending", pending_root), ("rejected", rejected_root)):
        for path in _private_json_files(directory):
            payload, encoded = _read_private_record(path)
            observation_records.append((
                state, payload, hashlib.sha256(encoded).hexdigest(),
            ))

    health: dict[str, tuple[dict[str, Any], str]] = {}
    for path in _private_json_files(health_root):
        value, encoded = _read_private_json(path)
        if _canonical_delivery_health(value) != encoded:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health changed",
            )
        assignment_id = path.stem
        if _IDENTIFIER_RE.fullmatch(assignment_id) is None:
            raise CheckpointObservationSpoolError(
                "checkpoint delivery health identity is invalid",
            )
        health[assignment_id] = (value, hashlib.sha256(encoded).hexdigest())

    impact_records: list[tuple[dict[str, Any], str]] = []
    if impact_root.exists() or impact_root.is_symlink():
        _private_directory(impact_root, create=False)
        for path in _private_json_files(impact_root):
            value, encoded = _read_private_json(path)
            if _canonical_mainline_impact(value) != encoded:
                raise CheckpointObservationSpoolError(
                    "checkpoint mainline impact sample changed",
                )
            if path.stem != value["sample_id"]:
                raise CheckpointObservationSpoolError(
                    "checkpoint mainline impact sample identity changed",
                )
            impact_records.append((
                value, hashlib.sha256(encoded).hexdigest(),
            ))

    groups: dict[tuple[str, ...], list[tuple[dict[str, Any], str]]] = {}
    assignment_ids: set[str] = set()
    assignment_cohorts: dict[str, tuple[str, ...]] = {}
    assignment_fingerprints: dict[str, str] = {}
    session_registrations: dict[
        tuple[str, str], tuple[tuple[str, ...], str, str, datetime]
    ] = {}
    for registration in registrations:
        value = registration[0]
        key = tuple(
            value["cohort"][field] for field in CHECKPOINT_COHORT_FIELDS_V2
        )
        assignment_id = value["assignment_id"]
        fingerprint = value["identity_fingerprint"]
        if (
            assignment_id in assignment_cohorts
            and assignment_cohorts[assignment_id] != key
        ) or (
            assignment_id in assignment_fingerprints
            and assignment_fingerprints[assignment_id] != fingerprint
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint assignment cohort registration drifted",
            )
        if datetime.fromisoformat(value["registered_at"]) > instant:
            raise CheckpointObservationSpoolError(
                "checkpoint cohort registration is in the future",
            )
        assignment_cohorts[assignment_id] = key
        assignment_fingerprints[assignment_id] = fingerprint
        session_key = (assignment_id, value["runner_session_id"])
        current_session = session_registrations.get(session_key)
        session_value = (
            key,
            fingerprint,
            registration[1],
            datetime.fromisoformat(value["registered_at"]).astimezone(
                timezone.utc,
            ),
        )
        if current_session is not None and current_session != session_value:
            raise CheckpointObservationSpoolError(
                "checkpoint runner session cohort registration drifted",
            )
        session_registrations[session_key] = session_value
        groups.setdefault(key, []).append(registration)
        assignment_ids.add(assignment_id)
    unregistered_records = sum(
        payload.get("assignment_id") not in assignment_ids
        for _, payload, _ in observation_records
    ) + sum(assignment_id not in assignment_ids for assignment_id in health)
    impact_groups: dict[
        tuple[str, ...], list[tuple[dict[str, Any], str, str]]
    ] = {}
    for sample, digest in impact_records:
        registration = session_registrations.get((
            sample["assignment_id"], sample["runner_session_id"],
        ))
        if (
            registration is None
            or registration[1] != sample["identity_fingerprint"]
        ):
            unregistered_records += 1
            continue
        sample_started = datetime.fromisoformat(sample["started_at"]).astimezone(
            timezone.utc,
        )
        if sample_started < registration[3] or sample_started > instant:
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact sample time is invalid",
            )
        if (
            sample["state"] == "completed"
            and datetime.fromisoformat(sample["completed_at"]).astimezone(
                timezone.utc,
            ) > instant
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint mainline impact sample time is invalid",
            )
        if window_start is not None and sample_started < window_start:
            continue
        impact_groups.setdefault(registration[0], []).append((
            sample, digest, registration[2],
        ))

    attestations: list[dict[str, Any]] = []
    for key in sorted(groups):
        cohort_registrations = groups[key]
        cohort = dict(zip(CHECKPOINT_COHORT_FIELDS_V2, key, strict=True))
        assignments = {
            value["assignment_id"] for value, _ in cohort_registrations
        }
        sessions = {
            value["runner_session_id"] for value, _ in cohort_registrations
        }
        registered_at = [
            datetime.fromisoformat(value["registered_at"]).astimezone(timezone.utc)
            for value, _ in cohort_registrations
        ]
        evidence_from = min(registered_at)
        if window_start is not None and evidence_from <= window_start:
            evidence_from = window_start
        pending = [
            (payload, digest) for state, payload, digest in observation_records
            if state == "pending" and payload.get("assignment_id") in assignments
        ]
        rejected = [
            (payload, digest) for state, payload, digest in observation_records
            if state == "rejected" and payload.get("assignment_id") in assignments
        ]
        delivery_totals = {
            field: sum(
                int(health[assignment_id][0][field])
                for assignment_id in assignments if assignment_id in health
            )
            for field in _DELIVERY_FIELDS
        }
        diagnostics = 0
        unstructured_diagnostics = 0
        cleanup_residue = 0
        for assignment_id in assignments:
            total, unstructured = _local_diagnostic_counts(home, assignment_id)
            diagnostics += total
            unstructured_diagnostics += unstructured
            cleanup_residue += _local_cleanup_residue(home, assignment_id)
        cleanup_residue += sum(
            payload.get("remote_cleanup") == "failed"
            for payload, _ in (*pending, *rejected)
        )
        pending_ages = [
            max(0, int((instant - datetime.fromtimestamp(
                (root / "pending" / f"{payload['operation_id']}.json").stat().st_mtime,
                timezone.utc,
            )).total_seconds()))
            for payload, _ in pending
        ]
        source = {
            "cohort": cohort,
            "assignment_ids": sorted(assignments),
            "runner_session_ids": sorted(sessions),
            "registration_digests": sorted(
                digest for _, digest in cohort_registrations
            ),
            "pending_digests": sorted(digest for _, digest in pending),
            "rejected_digests": sorted(digest for _, digest in rejected),
            "health_digests": sorted(
                health[assignment_id][1]
                for assignment_id in assignments if assignment_id in health
            ),
            "diagnostic_records": diagnostics,
            "unstructured_diagnostics": unstructured_diagnostics,
            "cleanup_residue": cleanup_residue,
            "spool_files": spool_files,
            "spool_bytes": spool_bytes,
            "spool_file_limit": MAX_OBSERVATION_SPOOL_FILES_V2,
            "spool_byte_limit": MAX_OBSERVATION_SPOOL_BYTES_V2,
            "spool_capacity_pressure": spool_capacity_pressure,
        }
        artifact = hashlib.sha256(json.dumps(
            source, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")).hexdigest()
        attestations.append({
            "schema": EVIDENCE_ATTESTATION_SCHEMA_V2,
            "attestation_id": f"local-outbox-{artifact[:40]}",
            "kind": "outbox_health",
            "cohort": cohort,
            "observed_from": evidence_from.replace(microsecond=0).isoformat(),
            "observed_until": instant.replace(microsecond=0).isoformat(),
            "artifact_sha256": artifact,
            "metrics": {
                "observation_processes": len(sessions),
                "assignment_count": len(assignments),
                "pending_records": len(pending),
                "oldest_pending_seconds": max(pending_ages, default=0),
                "rejected_records": len(rejected),
                "persisted_records": delivery_totals["persisted"],
                "acknowledged_records": delivery_totals["acknowledged"],
                "retryable_deliveries": delivery_totals["retryable"],
                "rejected_deliveries": delivery_totals["rejected"],
                "dropped_records": delivery_totals["dropped"],
                "diagnostic_records": diagnostics,
                "unstructured_diagnostics": unstructured_diagnostics,
                "cleanup_residue": cleanup_residue,
                "spool_files": spool_files,
                "spool_bytes": spool_bytes,
                "spool_file_limit": MAX_OBSERVATION_SPOOL_FILES_V2,
                "spool_byte_limit": MAX_OBSERVATION_SPOOL_BYTES_V2,
                "spool_capacity_pressure": spool_capacity_pressure,
            },
        })
    for key in sorted(impact_groups):
        samples = impact_groups[key]
        cohort = dict(zip(CHECKPOINT_COHORT_FIELDS_V2, key, strict=True))
        cohort_registered_at = min(
            datetime.fromisoformat(value["registered_at"]).astimezone(
                timezone.utc,
            )
            for value, _ in groups[key]
        )
        evidence_from = cohort_registered_at
        if window_start is not None and evidence_from <= window_start:
            evidence_from = window_start
        settled = [
            sample for sample, _, _ in samples if sample["state"] == "completed"
        ]
        completed = [sample for sample in settled if sample["comparable"]]
        ratios = sorted(
            sample["checkpoint_sync_elapsed_ms"] / sample["mainline_elapsed_ms"]
            for sample in completed
        )
        p95_ratio = (
            ratios[max(0, math.ceil(0.95 * len(ratios)) - 1)]
            if ratios else 0.0
        )
        source = {
            "cohort": cohort,
            "sample_digests": sorted(digest for _, digest, _ in samples),
            "registration_digests": sorted({
                registration_digest for _, _, registration_digest in samples
            }),
        }
        artifact = hashlib.sha256(json.dumps(
            source, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")).hexdigest()
        attestations.append({
            "schema": EVIDENCE_ATTESTATION_SCHEMA_V2,
            "attestation_id": f"local-impact-{artifact[:40]}",
            "kind": "mainline_impact",
            "cohort": cohort,
            "observed_from": evidence_from.replace(microsecond=0).isoformat(),
            "observed_until": instant.replace(microsecond=0).isoformat(),
            "artifact_sha256": artifact,
            "metrics": {
                "paired_samples": len(completed),
                "incomplete_samples": len(samples) - len(settled),
                "excluded_samples": len(settled) - len(completed),
                "p95_added_latency_ratio": round(p95_ratio, 6),
                "result_mismatches": sum(
                    not sample["result_preserved"] for sample in completed
                ),
                "assignment_mutations": sum(
                    sample["assignment_state_sha256_after"]
                    != sample["assignment_state_sha256"]
                    for sample in completed
                ),
                "submission_losses": sum(
                    not sample["submission_preserved"] for sample in completed
                ),
            },
        })
    return {
        "schema": LOCAL_EVIDENCE_SCHEMA_V2,
        "generated_at": instant.replace(microsecond=0).isoformat(),
        "attestations": attestations,
        "unregistered_records": unregistered_records,
        "scan_errors": scan_errors,
    }


def checkpoint_local_evidence_v2(
    home: Path,
    *,
    since: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build one process-consistent, read-only local evidence snapshot."""

    home = Path(home).absolute()
    root = home / "checkpoint-v2" / "observations"
    if not root.exists():
        return _checkpoint_local_evidence_unlocked_v2(
            home, since=since, now=now,
        )
    _private_directory(root, create=False)
    lock_path = root / ".lock"
    try:
        metadata = lock_path.lstat()
    except OSError as exc:
        raise CheckpointObservationSpoolError(
            "checkpoint local evidence is incomplete",
        ) from exc
    if (
        lock_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != 1
        or hasattr(os, "getuid") and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        )
    ):
        raise CheckpointObservationSpoolError(
            "checkpoint local evidence lock is unsafe",
        )
    with _process_lock(root):
        return _checkpoint_local_evidence_unlocked_v2(
            home, since=since, now=now,
        )


def cmd_checkpoint_audit(args) -> int:
    """CLI wrapper for the privacy-bounded local evidence scan."""

    from .local_config import HOME

    try:
        hours = getattr(args, "hours", 24 * 30)
        if (
            not isinstance(hours, int)
            or isinstance(hours, bool)
            or not 1 <= hours <= 8760
        ):
            raise CheckpointObservationSpoolError(
                "checkpoint local evidence hours are invalid",
            )
        instant = datetime.now(timezone.utc).replace(microsecond=0)
        report = checkpoint_local_evidence_v2(
            HOME,
            since=instant - timedelta(hours=hours),
            now=instant,
        )
    except (OSError, ValueError, CheckpointObservationSpoolError) as exc:
        print(f"checkpoint evidence audit refused: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CHECKPOINT_COHORT_FIELDS_V2",
    "COHORT_REGISTRY_SCHEMA_V2",
    "CheckpointObservationReporterV2",
    "CheckpointObservationSpoolError",
    "CheckpointObservationSpoolV2",
    "MAX_OBSERVATION_RECORD_BYTES_V2",
    "MAX_OBSERVATION_SPOOL_BYTES_V2",
    "MAX_OBSERVATION_SPOOL_FILES_V2",
    "MAX_PENDING_OBSERVATIONS_V2",
    "MAX_PENDING_OBSERVATION_BYTES_V2",
    "MAX_REJECTED_OBSERVATIONS_V2",
    "MAX_REJECTED_OBSERVATION_BYTES_V2",
    "MAINLINE_IMPACT_SCHEMA_V2",
    "OBSERVATION_SPOOL_SCHEMA_V2",
    "ObservationDeliveryResultV2",
    "checkpoint_mainline_assignment_digest_v2",
    "checkpoint_local_evidence_v2",
    "cmd_checkpoint_audit",
]
