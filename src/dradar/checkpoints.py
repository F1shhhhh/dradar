"""Discovery, validation, locking, and garbage collection for Pier checkpoints.

Pier writes each checkpoint below a trial's bind-mounted ``agent`` directory.
This module deliberately stores no server token or assignment nonce: the CLI
re-fetches the authenticated assignment before every recovery.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1
DEFAULT_TTL_DAYS = 7
KEEP_MARKER = ".dradar-keep"
TERMINAL_MARKER = ".dradar-terminal-evidence"
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

    @property
    def size_bytes(self) -> int:
        total = 0
        if not self.job_dir.is_dir():
            return 0
        for root, _dirs, files in os.walk(self.job_dir, followlinks=False):
            for name in files:
                try:
                    total += (Path(root) / name).lstat().st_size
                except OSError:
                    pass
        return total


class CheckpointBusy(RuntimeError):
    pass


def _contains_sensitive_key(value) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(child) for child in value)
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


def _load(path: Path) -> Checkpoint:
    checkpoint_dir = path.parent
    # .../<job>/<trial>/agent/checkpoint/checkpoint.json
    trial_dir = path.parents[2]
    job_dir = path.parents[3]
    fallback = path.stat().st_mtime
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("manifest is not an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
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
    if (checkpoint_dir / "invalid-secret").is_file():
        errors.append("credential-shaped content was rejected")
        phase = "invalid"
    if (checkpoint_dir / "invalid-snapshot").is_file():
        errors.append("checkpoint snapshot did not finish safely")
        phase = "invalid"

    heartbeat = checkpoint_dir / "last_heartbeat"
    manifest_time = _parse_time(
        raw.get("updated_at") or raw.get("last_heartbeat"), fallback,
    )
    heartbeat_time = (
        datetime.fromtimestamp(heartbeat.stat().st_mtime, timezone.utc)
        if heartbeat.is_file() else manifest_time
    )
    session_id = raw.get("session_id")
    if not isinstance(session_id, str):
        sidecar = checkpoint_dir / "session-id"
        if (
            sidecar.is_file()
            and not sidecar.is_symlink()
            and sidecar.stat().st_size <= 512
        ):
            candidate = sidecar.read_text(
                encoding="utf-8", errors="replace",
            ).strip()
            if re.fullmatch(r"[A-Za-z0-9._:-]{8,160}", candidate):
                session_id = candidate
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
    )


def scan(home: Path) -> list[Checkpoint]:
    root = home / "work" / "jobs"
    if root.is_symlink() or not root.is_dir():
        return []
    found = []
    for path in root.glob("*/*/agent/checkpoint/checkpoint.json"):
        if _path_uses_symlink(root, path):
            continue
        try:
            found.append(_load(path))
        except (OSError, IndexError):
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


def persist_resume_generation(
    home: Path, item: Checkpoint, generation: int,
) -> None:
    """Fence a server-accepted resume locally before starting paid work."""

    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("resume generation must be a non-negative integer")
    job_dir = _safe_job_dir(home, item)
    manifest = item.manifest_path.absolute()
    if _path_uses_symlink(job_dir, manifest):
        raise ValueError(f"checkpoint manifest crosses a symlink: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint manifest is unreadable") from exc
    if not isinstance(payload, dict) or _contains_sensitive_key(payload):
        raise ValueError("checkpoint manifest is unsafe")
    if payload.get("assignment_id") != item.assignment_id:
        raise ValueError("checkpoint assignment changed before resume")
    payload["resume_generation"] = generation
    payload["phase"] = "paused"
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    os.replace(temporary, manifest)


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
    return (_safe_job_dir(home, item) / TERMINAL_MARKER).is_file()


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
