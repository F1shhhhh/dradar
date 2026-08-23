"""Discovery, validation, locking, and garbage collection for Pier checkpoints.

Current checkpoints are published in a host-private sibling of Pier's
bind-mounted ``agent`` directory.  The old agent-mounted layout is discovered
only so it can be released fail-closed; its bytes are never resumed or copied.
This module deliberately stores no server token or assignment nonce: the CLI
re-fetches the authenticated assignment before every recovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1
DEFAULT_TTL_DAYS = 7
KEEP_MARKER = ".dradar-keep"
TERMINAL_MARKER = ".dradar-terminal-evidence"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_NODES = 10_000
MAX_MANIFEST_DEPTH = 64
MAX_CHECKPOINT_TREE_ENTRIES = 20_000
MAX_CHECKPOINT_TREE_DEPTH = 64
MAX_CHECKPOINT_FILE_BYTES = 256 * 1024 * 1024
MAX_CHECKPOINT_TREE_BYTES = 512 * 1024 * 1024
_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "credential", "api_key", "auth")
_ASSIGNMENT_FROM_JOB = re.compile(r"^a([0-9a-f]{32})(?:-|$)")


@dataclass(frozen=True)
class Checkpoint:
    manifest_path: Path
    checkpoint_dir: Path
    trial_dir: Path
    job_dir: Path
    assignment_id: str | None
    checkpoint_id: str | None
    phase: str
    resume_generation: int
    task_id: str | None
    model: str | None
    effort: str | None
    updated_at: datetime
    valid: bool
    invalid_reason: str | None = None
    harness: str | None = None
    provider: str | None = None
    agent_version: str | None = None
    session_id: str | None = None
    manifest_sha256: str | None = None
    manifest_identity: tuple[int, int, int, int] | None = None
    payload_pointer: tuple[bytes, tuple[int, int, int, int]] | None = None

    @property
    def size_bytes(self) -> int:
        def raise_walk_error(error: OSError) -> None:
            raise error

        total = 0
        if not self.job_dir.is_dir():
            return 0
        for root, _dirs, files in os.walk(
            self.job_dir, followlinks=False, onerror=raise_walk_error,
        ):
            for name in files:
                total += (Path(root) / name).lstat().st_size
        return total


class CheckpointBusy(RuntimeError):
    pass


def _contains_sensitive_key(value) -> bool:
    pending = [(value, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if visited > MAX_MANIFEST_NODES or depth > MAX_MANIFEST_DEPTH:
            return True
        if isinstance(current, dict):
            for key, child in current.items():
                normalized = str(key).lower().replace("-", "_")
                if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                    return True
                pending.append((child, depth + 1))
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return False


def _parse_time(value: object, fallback: float) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback, timezone.utc)


def _infer_assignment_id(job_dir: Path) -> str | None:
    matched = _ASSIGNMENT_FROM_JOB.match(job_dir.name)
    return matched.group(1) if matched else None


def _path_uses_symlink(root: Path, path: Path) -> bool:
    """Return true when ``path`` escapes ``root`` lexically or crosses a link."""

    lexical_root = root.absolute()
    lexical_path = path.absolute()
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        return True
    current = lexical_root
    if current.is_symlink():
        return True
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


class _UnsafeCheckpointFile(ValueError):
    pass


def _read_regular_file_snapshot(
    path: Path, *, max_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int]]:
    """Read a small metadata file without following or blocking on special files."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise _UnsafeCheckpointFile("file is unreadable") from exc
    if not stat.S_ISREG(before.st_mode):
        raise _UnsafeCheckpointFile("file is not a regular file")
    if before.st_nlink != 1:
        raise _UnsafeCheckpointFile("file is multiply linked")
    if before.st_size > max_bytes:
        raise _UnsafeCheckpointFile("file is too large")

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _UnsafeCheckpointFile("file could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise _UnsafeCheckpointFile("file changed before it was opened")
        if not stat.S_ISREG(opened.st_mode):
            raise _UnsafeCheckpointFile("opened file is not a regular file")
        if opened.st_nlink != 1:
            raise _UnsafeCheckpointFile("opened file is multiply linked")
        if opened.st_size > max_bytes:
            raise _UnsafeCheckpointFile("opened file is too large")

        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise _UnsafeCheckpointFile("file grew beyond the size limit")

        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or len(payload) != after.st_size
        ):
            raise _UnsafeCheckpointFile("file changed while it was read")
        identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        return payload, identity
    except OSError as exc:
        raise _UnsafeCheckpointFile("file could not be read safely") from exc
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    return _read_regular_file_snapshot(path, max_bytes=max_bytes)[0]


def _snapshot_payload_dir(
    checkpoint_dir: Path,
) -> tuple[Path, str | None, tuple[bytes, tuple[int, int, int, int]] | None]:
    pointer = checkpoint_dir / "current-generation"
    if not _lexists(pointer):
        return checkpoint_dir, None, None
    try:
        pointer_bytes, pointer_identity = _read_regular_file_snapshot(
            pointer, max_bytes=128,
        )
        generation = pointer_bytes.decode("ascii").strip()
    except (UnicodeError, _UnsafeCheckpointFile):
        return checkpoint_dir, "checkpoint generation pointer is invalid", None
    if re.fullmatch(r"[A-Za-z0-9._-]{8,64}", generation) is None:
        return checkpoint_dir, "checkpoint generation pointer is invalid", None
    payload = checkpoint_dir / "snapshots" / generation
    if _path_uses_symlink(checkpoint_dir, payload) or not payload.is_dir():
        return checkpoint_dir, "checkpoint snapshot generation is missing", None
    return payload, None, (pointer_bytes, pointer_identity)


def _special_tree_error(
    root: Path, *, require_private: bool = False,
) -> str | None:
    def raise_walk_error(error: OSError) -> None:
        raise error

    try:
        root_metadata = root.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            return "checkpoint root is not a directory"
        root_device = root_metadata.st_dev
        expected_uid = os.getuid() if require_private else None
        expected_gid = os.getgid() if require_private else None

        def private_error(metadata: os.stat_result) -> str | None:
            if not require_private:
                return None
            if metadata.st_uid != expected_uid or metadata.st_gid != expected_gid:
                return "checkpoint tree is not owned by DRadar"
            expected_mode = 0o700 if stat.S_ISDIR(metadata.st_mode) else 0o600
            if stat.S_IMODE(metadata.st_mode) != expected_mode:
                return "checkpoint tree is not host-private"
            return None

        ownership_error = private_error(root_metadata)
        if ownership_error:
            return ownership_error
        entry_count = 0
        total_bytes = 0
        for current, directories, files in os.walk(
            root, followlinks=False, onerror=raise_walk_error,
        ):
            try:
                depth = len(Path(current).relative_to(root).parts)
            except ValueError:
                return "checkpoint tree escaped its root"
            if depth > MAX_CHECKPOINT_TREE_DEPTH:
                return "checkpoint tree exceeds the depth limit"
            for name in (*directories, *files):
                entry_count += 1
                if entry_count > MAX_CHECKPOINT_TREE_ENTRIES:
                    return "checkpoint tree exceeds the entry-count limit"
                metadata = (Path(current) / name).lstat()
                if metadata.st_dev != root_device:
                    return "checkpoint crossed a filesystem boundary"
                if not (
                    stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISREG(metadata.st_mode)
                ):
                    return "checkpoint contains a special file"
                ownership_error = private_error(metadata)
                if ownership_error:
                    return ownership_error
                if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
                    return "checkpoint contains a multiply linked file"
                if stat.S_ISREG(metadata.st_mode):
                    if metadata.st_size > MAX_CHECKPOINT_FILE_BYTES:
                        return "checkpoint contains an oversized file"
                    total_bytes += metadata.st_size
                    if total_bytes > MAX_CHECKPOINT_TREE_BYTES:
                        return "checkpoint tree exceeds the total-size limit"
    except OSError:
        return "checkpoint tree is unreadable"
    return None


def _host_private_layout_error(
    trial_dir: Path, checkpoint_dir: Path,
) -> str | None:
    """Validate the mount/ownership boundary promised by the new layout."""

    if checkpoint_dir != trial_dir / "checkpoint":
        return "checkpoint is not in the host-private sibling layout"
    try:
        trial_metadata = trial_dir.lstat()
        checkpoint_metadata = checkpoint_dir.lstat()
    except OSError:
        return "checkpoint host layout is unreadable"
    if (
        not stat.S_ISDIR(trial_metadata.st_mode)
        or not stat.S_ISDIR(checkpoint_metadata.st_mode)
        or trial_metadata.st_dev != checkpoint_metadata.st_dev
        or trial_metadata.st_uid != os.getuid()
        or trial_metadata.st_gid != os.getgid()
        or stat.S_IMODE(trial_metadata.st_mode) != 0o700
    ):
        return "checkpoint trial directory is not host-private"
    return _special_tree_error(checkpoint_dir, require_private=True)


def _load(path: Path, *, trial_dir: Path, job_dir: Path) -> Checkpoint:
    checkpoint_dir = path.parent
    snapshot_lock = checkpoint_dir / "snapshot.lock"
    if _lexists(snapshot_lock):
        return Checkpoint(
            path, checkpoint_dir, trial_dir, job_dir,
            _infer_assignment_id(job_dir), None, "invalid", 0,
            None, None, None, datetime.now(timezone.utc), False,
            "checkpoint snapshot is incomplete",
        )
    try:
        fallback = path.lstat().st_mtime
    except OSError:
        fallback = datetime.now(timezone.utc).timestamp()
    try:
        manifest_bytes, manifest_identity = _read_regular_file_snapshot(
            path, max_bytes=MAX_MANIFEST_BYTES,
        )
        manifest = manifest_bytes.decode("utf-8")
        raw = json.loads(manifest)
        if not isinstance(raw, dict):
            raise ValueError("manifest is not an object")
    except (
        UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError,
    ) as exc:
        return Checkpoint(
            path, checkpoint_dir, trial_dir, job_dir,
            _infer_assignment_id(job_dir), None, "invalid", 0,
            None, None, None, _parse_time(None, fallback), False, str(exc),
        )

    inferred_assignment_id = _infer_assignment_id(job_dir)
    manifest_assignment_id = raw.get("assignment_id")
    assignment_id = (
        manifest_assignment_id
        if isinstance(manifest_assignment_id, str) and manifest_assignment_id
        else inferred_assignment_id
    )
    checkpoint_id = raw.get("checkpoint_id")
    phase = raw.get("phase") if isinstance(raw.get("phase"), str) else "invalid"
    generation = raw.get("resume_generation", 0)
    errors = []
    if raw.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema {raw.get('schema_version')!r}")
    if _contains_sensitive_key(raw):
        errors.append("manifest contains a sensitive field")
    if not assignment_id:
        errors.append("missing assignment_id")
    elif (
        inferred_assignment_id is not None
        and assignment_id != inferred_assignment_id
    ):
        errors.append("manifest assignment_id does not match job directory")
        assignment_id = inferred_assignment_id
    if not isinstance(checkpoint_id, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]{8,64}", checkpoint_id
    ):
        errors.append("invalid checkpoint_id")
        checkpoint_id = None
    if not isinstance(generation, int) or generation < 0:
        errors.append("invalid resume_generation")
        generation = 0
    if phase not in {"running", "paused", "agent_completed", "incompatible", "invalid"}:
        errors.append(f"invalid phase {phase!r}")
        phase = "invalid"
    payload_dir, payload_error, pointer_before = _snapshot_payload_dir(
        checkpoint_dir,
    )
    if payload_error:
        errors.append(payload_error)
        phase = "invalid"
    if checkpoint_dir == trial_dir / "checkpoint":
        tree_error = _host_private_layout_error(trial_dir, checkpoint_dir)
    else:
        tree_error = _special_tree_error(checkpoint_dir)
    if tree_error:
        errors.append(tree_error)
        phase = "invalid"
    if (
        _lexists(checkpoint_dir / "invalid-secret")
        or _lexists(payload_dir / "invalid-secret")
    ):
        errors.append("credential-shaped content was rejected")
        phase = "invalid"
    if _lexists(checkpoint_dir / "invalid-snapshot"):
        errors.append("checkpoint snapshot did not finish safely")
        phase = "invalid"
    heartbeat = payload_dir / "last_heartbeat"
    manifest_time = _parse_time(
        raw.get("updated_at") or raw.get("last_heartbeat"), fallback,
    )
    heartbeat_time = (
        datetime.fromtimestamp(heartbeat.stat().st_mtime, timezone.utc)
        if heartbeat.is_file() else manifest_time
    )
    session_id = None
    sidecar = payload_dir / "session-id"
    try:
        candidate = _read_regular_file(
            sidecar, max_bytes=512,
        ).decode("utf-8").strip()
        if re.fullmatch(r"[A-Za-z0-9._:-]{8,160}", candidate):
            session_id = candidate
    except (UnicodeDecodeError, _UnsafeCheckpointFile):
        pass
    if session_id is None:
        for candidate in (raw.get("session_id"), raw.get("root_thread_id")):
            if (
                isinstance(candidate, str)
                and re.fullmatch(r"[A-Za-z0-9._:-]{8,160}", candidate)
            ):
                session_id = candidate
                break
    _payload_after, pointer_error_after, pointer_after = _snapshot_payload_dir(
        checkpoint_dir,
    )
    if pointer_error_after:
        errors.append(pointer_error_after)
        phase = "invalid"
    if pointer_after != pointer_before:
        errors.append("checkpoint generation changed while it was scanned")
        phase = "invalid"
    if (
        _lexists(checkpoint_dir / "invalid-secret")
        or _lexists(payload_dir / "invalid-secret")
    ):
        errors.append("credential-shaped content was rejected")
        phase = "invalid"
    if _lexists(checkpoint_dir / "invalid-snapshot"):
        errors.append("checkpoint snapshot did not finish safely")
        phase = "invalid"
    if _lexists(snapshot_lock):
        errors.append("checkpoint snapshot is incomplete")
        phase = "invalid"
    try:
        manifest_after, manifest_identity_after = _read_regular_file_snapshot(
            path, max_bytes=MAX_MANIFEST_BYTES,
        )
    except _UnsafeCheckpointFile:
        errors.append("checkpoint manifest changed while it was scanned")
        phase = "invalid"
    else:
        if (
            manifest_after != manifest_bytes
            or manifest_identity_after != manifest_identity
        ):
            errors.append("checkpoint manifest changed while it was scanned")
            phase = "invalid"
    return Checkpoint(
        path, checkpoint_dir, trial_dir, job_dir, assignment_id, checkpoint_id,
        phase, generation,
        raw.get("task_id") if isinstance(raw.get("task_id"), str) else None,
        raw.get("model") if isinstance(raw.get("model"), str) else None,
        raw.get("effort") if isinstance(raw.get("effort"), str) else None,
        max(manifest_time, heartbeat_time),
        not errors and phase != "invalid",
        "; ".join(errors) if errors else None,
        raw.get("harness") if isinstance(raw.get("harness"), str) else None,
        raw.get("provider") if isinstance(raw.get("provider"), str) else None,
        (
            raw.get("agent_version")
            if isinstance(raw.get("agent_version"), str) else None
        ),
        session_id if isinstance(session_id, str) else None,
        hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_identity,
        pointer_before,
    )


def scan(home: Path) -> list[Checkpoint]:
    root = home / "work" / "jobs"
    if root.is_symlink() or not root.is_dir():
        return []
    found = []
    candidates: dict[Path, tuple[Path, Path, Path]] = {}
    layouts = (
        ("*/*/checkpoint/checkpoint.json", 1, 2),
        ("*/*/agent/checkpoint/checkpoint.json", 2, 3),
    )
    for pattern, trial_parent, job_parent in layouts:
        for path in root.glob(pattern):
            trial_dir = path.parents[trial_parent]
            job_dir = path.parents[job_parent]
            if _path_uses_symlink(root, path.parent):
                continue
            candidates.setdefault(
                trial_dir.absolute(), (path, trial_dir, job_dir),
            )
    for path, trial_dir, job_dir in candidates.values():
        try:
            found.append(_load(path, trial_dir=trial_dir, job_dir=job_dir))
        except (OSError, IndexError, RecursionError):
            continue
    seen_jobs = {item.job_dir.resolve() for item in found}
    for marker in root.glob(f"*/{TERMINAL_MARKER}"):
        if _path_uses_symlink(root, marker):
            continue
        try:
            job_dir = marker.parent
            if job_dir.resolve() in seen_jobs:
                continue
            trials = sorted(path for path in job_dir.glob("*__*") if path.is_dir())
            trial_dir = trials[0] if trials else job_dir
            updated_at = datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc)
            found.append(Checkpoint(
                marker, marker.parent, trial_dir, job_dir,
                _infer_assignment_id(job_dir), None, "terminal", 0,
                None, None, None, updated_at,
                False, "terminal local evidence",
            ))
        except OSError:
            # A concurrent cleanup may remove the marker between glob/stat.
            continue
    return sorted(found, key=lambda item: item.updated_at, reverse=True)


def latest_by_assignment(home: Path) -> dict[str, Checkpoint]:
    latest: dict[str, Checkpoint] = {}
    for item in scan(home):
        if is_terminal(home, item):
            continue
        if item.assignment_id and item.assignment_id not in latest:
            latest[item.assignment_id] = item
    return latest


def find_latest(home: Path, assignment_id: str) -> Checkpoint | None:
    return latest_by_assignment(home).get(assignment_id)


def materialize_host_checkpoint(home: Path, item: Checkpoint) -> Checkpoint:
    """Accept only checkpoints that were born in host-private storage.

    A legacy checkpoint below the model-writable ``agent`` bind has no
    trustworthy scan-to-copy boundary: an orphaned root container can switch
    its generation or inject data after discovery.  Copying it after the
    server fence would merely launder untrusted bytes into the trusted sibling.
    Keep that evidence in place and require a fresh checkpoint-capable run.
    """

    if not item.valid:
        raise ValueError("invalid checkpoint cannot be resumed")
    expected_new = item.trial_dir / "checkpoint"
    if item.checkpoint_dir == expected_new:
        return item
    expected_legacy = item.trial_dir / "agent" / "checkpoint"
    if item.checkpoint_dir == expected_legacy:
        _safe_job_dir(home, item)
        raise ValueError(
            "legacy agent-mounted checkpoints are untrusted and cannot be resumed",
        )
    raise ValueError("checkpoint layout is not recognized")


def _safe_job_dir(home: Path, item: Checkpoint) -> Path:
    lexical_root = (home / "work" / "jobs").absolute()
    lexical_job = item.job_dir.absolute()
    if _path_uses_symlink(lexical_root, lexical_job):
        raise ValueError(f"checkpoint path crosses a symlink: {lexical_job}")
    root = lexical_root.resolve()
    job_dir = lexical_job.resolve()
    if job_dir == root or root not in job_dir.parents:
        raise ValueError(f"checkpoint path escaped jobs directory: {job_dir}")
    return job_dir


_RESUME_STABLE_FIELDS = (
    "assignment_id", "checkpoint_id", "phase", "resume_generation",
    "task_id", "model", "effort", "harness", "provider", "agent_version",
    "session_id", "manifest_sha256", "manifest_identity", "payload_pointer",
)
_RESUME_MANIFEST_FIELDS = frozenset({
    "schema_version", "checkpoint_id", "assignment_id", "phase",
    "created_at", "updated_at", "last_heartbeat", "task_id", "model",
    "effort", "harness", "provider", "agent_version", "base_commit",
    "resume_generation", "resume_count", "workspace_patch",
    "untracked_archive", "state_dir", "events_file", "session_id",
    "failure_type",
})


def _same_checkpoint_snapshot(left: Checkpoint, right: Checkpoint) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in _RESUME_STABLE_FIELDS
    )


def revalidate_host_checkpoint(home: Path, item: Checkpoint) -> Checkpoint:
    """Re-scan a new-layout checkpoint immediately before trusted recovery."""

    _safe_job_dir(home, item)
    if item.checkpoint_dir != item.trial_dir / "checkpoint":
        raise ValueError("legacy checkpoint cannot be trusted")
    fresh = _load(
        item.manifest_path, trial_dir=item.trial_dir, job_dir=item.job_dir,
    )
    if not fresh.valid or not _same_checkpoint_snapshot(item, fresh):
        raise ValueError(
            fresh.invalid_reason or "checkpoint changed after discovery",
        )
    return fresh


def _canonical_resume_manifest(
    payload: object, item: Checkpoint, generation: int,
) -> dict[str, object]:
    if not isinstance(payload, dict) or _contains_sensitive_key(payload):
        raise ValueError("checkpoint manifest is unsafe")
    value = dict(payload)
    root_thread_id = value.pop("root_thread_id", None)
    if "session_id" not in value and isinstance(root_thread_id, str):
        value["session_id"] = root_thread_id
    unexpected = set(value) - _RESUME_MANIFEST_FIELDS
    if unexpected:
        raise ValueError("checkpoint manifest contains unsupported fields")
    required_strings = (
        "checkpoint_id", "assignment_id", "phase", "created_at", "updated_at",
        "last_heartbeat", "task_id", "model", "effort", "harness", "provider",
        "agent_version", "base_commit", "workspace_patch", "untracked_archive",
        "state_dir", "events_file",
    )
    if any(not isinstance(value.get(field), str) for field in required_strings):
        raise ValueError("checkpoint manifest is incomplete")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint manifest schema is unsupported")
    if value.get("assignment_id") != item.assignment_id:
        raise ValueError("checkpoint assignment changed before resume")
    if value.get("checkpoint_id") != item.checkpoint_id:
        raise ValueError("checkpoint id changed before resume")
    if value.get("phase") != item.phase:
        raise ValueError("checkpoint phase changed before resume")
    if value.get("resume_generation") != item.resume_generation:
        raise ValueError("checkpoint generation changed before resume")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("base_commit"))):
        raise ValueError("checkpoint base commit is invalid")
    resume_count = value.get("resume_count")
    if (
        not isinstance(resume_count, int)
        or isinstance(resume_count, bool)
        or resume_count < 0
    ):
        raise ValueError("checkpoint resume count is invalid")
    for field, expected in {
        "workspace_patch": "workspace.patch",
        "untracked_archive": "untracked.tar.gz",
        "state_dir": "provider-state",
        "events_file": "events.jsonl",
    }.items():
        if value.get(field) != expected:
            raise ValueError("checkpoint artifact mapping is invalid")
    session_id = value.get("session_id")
    if session_id is not None and (
        not isinstance(session_id, str)
        or re.fullmatch(r"[A-Za-z0-9._:-]{8,160}", session_id) is None
    ):
        raise ValueError("checkpoint session id is invalid")
    failure_type = value.get("failure_type")
    if failure_type is not None and (
        not isinstance(failure_type, str)
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", failure_type) is None
    ):
        raise ValueError("checkpoint failure type is invalid")
    value["resume_generation"] = generation
    value["phase"] = "paused"
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    return value


def persist_resume_generation(
    home: Path, item: Checkpoint, generation: int,
) -> Checkpoint:
    """Fence a server-accepted resume locally before starting paid work."""

    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("resume generation must be a non-negative integer")
    if not item.valid or item.phase not in {"paused", "running"}:
        raise ValueError("checkpoint is not resumable")
    job_dir = _safe_job_dir(home, item)
    manifest = item.manifest_path.absolute()
    if _path_uses_symlink(job_dir, manifest):
        raise ValueError(f"checkpoint manifest crosses a symlink: {manifest}")
    if item.checkpoint_dir != item.trial_dir / "checkpoint":
        raise ValueError("legacy checkpoint cannot be resumed")
    if item.manifest_sha256 is None or item.manifest_identity is None:
        raise ValueError("checkpoint scan identity is unavailable")
    try:
        manifest_bytes, manifest_identity = _read_regular_file_snapshot(
            manifest, max_bytes=MAX_MANIFEST_BYTES,
        )
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (
        UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError,
    ) as exc:
        raise ValueError("checkpoint manifest is unreadable") from exc
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != item.manifest_sha256
        or manifest_identity != item.manifest_identity
    ):
        raise ValueError("checkpoint manifest changed after discovery")
    fresh = _load(
        manifest, trial_dir=item.trial_dir, job_dir=item.job_dir,
    )
    if not fresh.valid or not _same_checkpoint_snapshot(item, fresh):
        raise ValueError("checkpoint changed after discovery")
    canonical = _canonical_resume_manifest(payload, item, generation)
    serialized = (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode()
    parent_fd = os.open(
        manifest.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = f".checkpoint.json.tmp-{uuid.uuid4().hex}"
    temporary_created = False
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        try:
            view = memoryview(serialized)
            while view:
                written = os.write(temporary_fd, view)
                view = view[written:]
            os.fchmod(temporary_fd, 0o600)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        manifest_before_replace, identity_before_replace = (
            _read_regular_file_snapshot(
                manifest, max_bytes=MAX_MANIFEST_BYTES,
            )
        )
        if (
            manifest_before_replace != manifest_bytes
            or identity_before_replace != manifest_identity
        ):
            raise ValueError("checkpoint manifest changed before fencing")
        os.replace(
            temporary_name,
            manifest.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    except BaseException:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(parent_fd)
    persisted = _load(
        manifest, trial_dir=item.trial_dir, job_dir=item.job_dir,
    )
    stable_after = (
        "assignment_id", "checkpoint_id", "task_id", "model", "effort",
        "harness", "provider", "agent_version", "session_id", "payload_pointer",
    )
    if (
        not persisted.valid
        or persisted.phase != "paused"
        or persisted.resume_generation != generation
        or any(
            getattr(persisted, field) != getattr(item, field)
            for field in stable_after
        )
    ):
        raise ValueError("checkpoint generation fence could not be verified")
    return persisted


def remove(home: Path, item: Checkpoint) -> None:
    shutil.rmtree(_safe_job_dir(home, item), ignore_errors=True)


def mark_kept(home: Path, item: Checkpoint) -> None:
    """Protect a settled job from the default ``dradar cleanup`` sweep."""
    marker = _safe_job_dir(home, item) / KEEP_MARKER
    marker.touch(mode=0o600, exist_ok=True)


def mark_terminal(home: Path, item: Checkpoint) -> None:
    """Keep diagnostics locally while excluding this item from recovery."""
    mark_terminal_job(home, _safe_job_dir(home, item))


def mark_terminal_job(home: Path, job_dir: Path) -> None:
    root = (home / "work" / "jobs").resolve()
    job_dir = job_dir.resolve()
    if job_dir == root or root not in job_dir.parents:
        raise ValueError(f"job path escaped jobs directory: {job_dir}")
    (job_dir / KEEP_MARKER).touch(mode=0o600, exist_ok=True)
    (job_dir / TERMINAL_MARKER).touch(mode=0o600, exist_ok=True)


def is_terminal(home: Path, item: Checkpoint) -> bool:
    return _lexists(_safe_job_dir(home, item) / TERMINAL_MARKER)


def is_kept(home: Path, item: Checkpoint) -> bool:
    return (_safe_job_dir(home, item) / KEEP_MARKER).is_file()


def cleanup_assignment(
    home: Path, assignment_id: str, *, keep_job_dir: Path | None = None
) -> None:
    keep = keep_job_dir.resolve() if keep_job_dir else None
    seen: set[Path] = set()
    for item in scan(home):
        if item.assignment_id != assignment_id:
            continue
        job = _safe_job_dir(home, item)
        if job in seen or (keep is not None and job == keep):
            continue
        seen.add(job)
        shutil.rmtree(job, ignore_errors=True)


def prune_superseded(home: Path, assignment_id: str, keep: Checkpoint) -> int:
    removed = 0
    for item in scan(home):
        if item.assignment_id != assignment_id or item.job_dir == keep.job_dir:
            continue
        remove(home, item)
        removed += 1
    return removed


def is_expired(item: Checkpoint, ttl_days: int = DEFAULT_TTL_DAYS) -> bool:
    age = datetime.now(timezone.utc) - item.updated_at
    return age.total_seconds() > max(1, ttl_days) * 86400


@contextmanager
def assignment_lock(home: Path, assignment_id: str) -> Iterator[None]:
    """Non-blocking per-assignment process lock; workers remain independent."""
    lock_dir = home / "checkpoint-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{assignment_id}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
    os.lseek(fd, 0, os.SEEK_SET)
    windows_lock = False
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - exercised on Windows runners
            import msvcrt

            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                windows_lock = True
            except OSError as exc:
                raise CheckpointBusy(
                    f"checkpoint {assignment_id} is already resuming"
                ) from exc
        except BlockingIOError as exc:
            raise CheckpointBusy(f"checkpoint {assignment_id} is already resuming") from exc
        yield
    finally:
        if windows_lock:  # pragma: no cover - exercised on Windows runners
            try:
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        os.close(fd)
