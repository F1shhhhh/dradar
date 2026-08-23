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

import json
import os
import queue
import re
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from .api_client import ApiError


OBSERVATION_SPOOL_SCHEMA_V2 = "dradar-checkpoint-v2-observation-spool-v1"
MAX_OBSERVATION_RECORD_BYTES_V2 = 16 * 1024
MAX_PENDING_OBSERVATIONS_V2 = 512
MAX_PENDING_OBSERVATION_BYTES_V2 = 8 * 1024 * 1024

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


class CheckpointObservationSpoolV2:
    """A bounded private outbox, safe for multiple local worker processes."""

    def __init__(
        self,
        root: Path,
        *,
        max_pending: int = MAX_PENDING_OBSERVATIONS_V2,
        max_pending_bytes: int = MAX_PENDING_OBSERVATION_BYTES_V2,
    ) -> None:
        if not 1 <= max_pending <= 10_000:
            raise ValueError("checkpoint observation pending limit is invalid")
        if not MAX_OBSERVATION_RECORD_BYTES_V2 <= max_pending_bytes <= 64 * 1024 * 1024:
            raise ValueError("checkpoint observation byte limit is invalid")
        self.root = Path(root).absolute()
        self.pending_root = self.root / "pending"
        self.rejected_root = self.root / "rejected"
        self.max_pending = max_pending
        self.max_pending_bytes = max_pending_bytes
        self._thread_lock = threading.Lock()

    def _prepare(self) -> None:
        _private_directory(self.root, create=True)
        _private_directory(self.pending_root, create=True)
        _private_directory(self.rejected_root, create=True)

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
                target = self.rejected_root / f"{operation_id}.{uuid.uuid4().hex}.json"
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

    def _add_stats(self, value: ObservationDeliveryResultV2) -> None:
        with self._stats_lock:
            self._stats = self._stats.plus(value)

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
            self._add_stats(ObservationDeliveryResultV2(dropped=1))
            return False
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
                result = result.plus(ObservationDeliveryResultV2(dropped=1))
            else:
                if created:
                    result = result.plus(ObservationDeliveryResultV2(persisted=1))
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

    def flush_once(self) -> ObservationDeliveryResultV2:
        """Persist queued data and attempt a bounded replay batch."""

        result = self._persist_queued()
        try:
            pending = self.spool.pending(limit=self._replay_batch)
        except Exception:
            result = result.plus(ObservationDeliveryResultV2(retryable=1))
            self._add_stats(result)
            return result
        for path, payload in pending:
            try:
                response = self._send(payload)
            except ApiError as exc:
                if exc.status_code in _STABLE_REJECTION_STATUS:
                    try:
                        self.spool.reject(path, str(payload["operation_id"]))
                    except Exception:
                        result = result.plus(ObservationDeliveryResultV2(retryable=1))
                    else:
                        result = result.plus(ObservationDeliveryResultV2(rejected=1))
                else:
                    result = result.plus(ObservationDeliveryResultV2(retryable=1))
                continue
            except Exception:
                result = result.plus(ObservationDeliveryResultV2(retryable=1))
                continue
            if not self._acknowledges_without_authority(response, payload):
                result = result.plus(ObservationDeliveryResultV2(retryable=1))
                continue
            try:
                self.spool.acknowledge(path, str(payload["operation_id"]))
            except Exception:
                result = result.plus(ObservationDeliveryResultV2(retryable=1))
            else:
                result = result.plus(ObservationDeliveryResultV2(acknowledged=1))
        self._add_stats(result)
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


__all__ = [
    "CheckpointObservationReporterV2",
    "CheckpointObservationSpoolError",
    "CheckpointObservationSpoolV2",
    "MAX_OBSERVATION_RECORD_BYTES_V2",
    "MAX_PENDING_OBSERVATIONS_V2",
    "OBSERVATION_SPOOL_SCHEMA_V2",
    "ObservationDeliveryResultV2",
]
