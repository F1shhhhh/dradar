"""Local pending-upload ledger: a trial that finished but failed to upload
must not just print a path and be forgotten. The volunteer's most expensive
artifact — a real trial that already burned real quota — gets a safety net.

On an upload failure (network drop, timeout, server 5xx), _run_and_submit
records everything needed to retry WITHOUT re-running the trial: the raw
trial_dir (patch/trajectory/result live under it, untouched by scrubbing —
scrubbing writes to a fresh tempdir and never mutates the originals) plus the
already-built client_meta and outcome. `dradar retry-upload` (and an
automatic scan at the top of `dradar go`) replays the upload later.

Entries are self-pruning: a retry that gets back 409 specifically saying
"already submitted" (some earlier attempt actually landed) or 410 (lease
expired — unsalvageable, the cell already reopened for someone else) removes
the entry. A 409 recovery-generation conflict is not success and stays queued.
Anything else keeps it for the next retry.
"""

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_FILENAME = "pending_uploads.json"
_LOCK_FILENAME = "pending_uploads.lock"
_PROCESS_LOCK = threading.Lock()


def _path(home: Path) -> Path:
    return home / _FILENAME


@contextmanager
def _locked(home: Path) -> Iterator[None]:
    """Serialize the ledger's read-modify-write cycle across all workers."""
    home.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        fd = os.open(home / _LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        windows_lock = False
        try:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows CI exercises callers
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                windows_lock = True
            yield
        finally:
            if windows_lock:  # pragma: no cover
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


def _load_unlocked(home: Path) -> list[dict]:
    path = _path(home)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def load(home: Path) -> list[dict]:
    # _save_unlocked commits with os.replace, so an unlocked reader still
    # sees either the complete old file or the complete new one. Keeping this
    # read-only also lets status/inspection work when DRADAR_HOME is mounted
    # read-only and avoids creating a lock file merely to report no entries.
    return _load_unlocked(home)


def assignment_ids(home: Path) -> set[str]:
    """Assignments with durable completed work that must never be rerun.

    A blocked/superseded upload is intentionally included: it still represents
    paid work whose ownership must be resolved explicitly, not a license to
    start the model again.
    """
    return {
        str(entry["assignment_id"])
        for entry in load(home)
        if entry.get("assignment_id")
    }


def _save_unlocked(home: Path, entries: list[dict]) -> None:
    # A safety-net ledger that isn't itself crash-safe defeats the point: a
    # plain write_text() truncates the file before writing, so a kill/OOM/
    # power-loss mid-write leaves truncated JSON and load() would then drop
    # EVERY pending entry, not just the one being saved. Write-to-temp +
    # atomic rename means the file on disk is always either the old or the
    # new complete version, never a partial one.
    home.mkdir(parents=True, exist_ok=True)
    path = _path(home)
    fd, raw_tmp = tempfile.mkstemp(
        dir=home, prefix=f".{_FILENAME}.", suffix=".tmp",
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(entries, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(home, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def record(home: Path, entry: dict) -> None:
    """Add or replace (by assignment_id) a pending-upload entry."""
    with _locked(home):
        entries = [
            e for e in _load_unlocked(home)
            if e.get("assignment_id") != entry.get("assignment_id")
        ]
        entries.append(entry)
        _save_unlocked(home, entries)


def remove(home: Path, assignment_id: str) -> None:
    with _locked(home):
        entries = [
            e for e in _load_unlocked(home)
            if e.get("assignment_id") != assignment_id
        ]
        _save_unlocked(home, entries)
