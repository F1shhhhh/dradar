"""Provider-neutral, credential-free checkpoints for paid Pier harnesses.

The container writes only to an untrusted staging directory below
``/logs/agent``. DRadar copies a validated snapshot into the host-only sibling
``<trial>/checkpoint`` before publishing it. Credentials are injected again
on resume and are never named in the manifest.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
_GENERIC_SECRET_RE = re.compile(
    rb"(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"
    rb"|ghp_[A-Za-z0-9]{20,}"
    rb"|github_pat_[A-Za-z0-9_]{20,}"
    rb"|gAAAAA[A-Za-z0-9_-]{40,}"
    rb"|eyJ[A-Za-z0-9_-]{10,}[.][A-Za-z0-9_-]{10,}"
    rb"[.][A-Za-z0-9_-]{10,})",
)
_ROOT_EXEC_ENV = {
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "BASH_ENV": "/dev/null",
    "ENV": "/dev/null",
    "CDPATH": "",
    "PYTHONPATH": "",
    "PYTHONHOME": "",
    "PYTHONINSPECT": "",
    "PYTHONSTARTUP": "",
    "PYTHONWARNINGS": "ignore",
    "PYTHONBREAKPOINT": "0",
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "",
    "SHELLOPTS": "",
    "BASHOPTS": "",
    "IFS": " \t\n",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}
_CAPTURE_WORK_TIMEOUT_SEC = 120
_CAPTURE_KILL_GRACE_SEC = 10
_CAPTURE_EXEC_TIMEOUT_SEC = (
    _CAPTURE_WORK_TIMEOUT_SEC + _CAPTURE_KILL_GRACE_SEC + 20
)
_PERIODIC_STOP_TIMEOUT_SEC = 180.0
_MAX_CHECKPOINT_FILES = 20_000
_MAX_CHECKPOINT_DEPTH = 64
_MAX_CHECKPOINT_FILE_BYTES = 256 * 1024 * 1024
_MAX_CHECKPOINT_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 20_000
_MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_CONTROL_READ_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_STREAM_BYTES = (
    _MAX_ARCHIVE_TOTAL_BYTES + _MAX_ARCHIVE_MEMBERS * 1024
)
_SENSITIVE_KEY_PARTS = (
    "token", "secret", "password", "credential", "api_key", "auth",
)
_FIXED_ARTIFACTS = {
    "workspace_patch": "workspace.patch",
    "untracked_archive": "untracked.tar.gz",
    "state_dir": "provider-state",
    "events_file": "events.jsonl",
}
_TRACKED_SCAN_ARTIFACT = ".tracked-worktree.scan"
_MAX_AGENT_LOG_BYTES = 64 * 1024 * 1024


class CheckpointError(RuntimeError):
    """A checkpoint is corrupt or cannot be safely restored."""


class CheckpointIncompatibleError(CheckpointError):
    """A valid checkpoint belongs to a different runtime identity."""


class UnsafeAgentLog(ValueError):
    """A model-writable host log could not be handled without following it."""


class AgentLogStore:
    """Safely consume direct children of Pier's host-side agent log directory.

    Pier intentionally makes ``/logs/agent`` writable by arbitrary container
    users, and mounted Docker logs are recursively returned to the host UID/GID
    before post-run conversion.  Keep that compatibility boundary intact: the
    directory may be 0777, but its parent and inode must remain host-owned and
    stable.  Individual reads are bounded ``O_NOFOLLOW`` snapshots.  Writes
    create a host-owned 0600 inode and atomically replace the directory entry,
    so a symlink, FIFO, hardlink, or raced replacement can never redirect a
    read or redaction into an unrelated host file.
    """

    REJECTED = "[DRADAR rejected an unsafe agent log]\n"

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.uid = os.getuid()

    @staticmethod
    def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    @staticmethod
    def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
        )

    def _leaf(self, path: Path) -> str:
        candidate = Path(path)
        if (
            candidate.parent != self.logs_dir
            or candidate.name in {"", ".", ".."}
            or candidate != self.logs_dir / candidate.name
        ):
            raise UnsafeAgentLog("agent log is outside the logs directory")
        return candidate.name

    def _open_dir(self) -> tuple[int, tuple[int, ...]]:
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_flags |= (
            getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            parent_before = self.logs_dir.parent.lstat()
            parent_fd = os.open(self.logs_dir.parent, parent_flags)
        except OSError as exc:
            raise UnsafeAgentLog("agent logs parent directory is unsafe") from exc
        try:
            parent_opened = os.fstat(parent_fd)
            if (
                not stat.S_ISDIR(parent_before.st_mode)
                or not stat.S_ISDIR(parent_opened.st_mode)
                or (parent_opened.st_dev, parent_opened.st_ino)
                != (parent_before.st_dev, parent_before.st_ino)
                or parent_opened.st_uid != self.uid
                or stat.S_IMODE(parent_opened.st_mode) & 0o022
            ):
                raise UnsafeAgentLog("agent logs parent is not host-private")
            observed = os.stat(
                self.logs_dir.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(self.logs_dir.name, flags, dir_fd=parent_fd)
        except (OSError, UnsafeAgentLog) as exc:
            os.close(parent_fd)
            if isinstance(exc, UnsafeAgentLog):
                raise
            raise UnsafeAgentLog("agent logs directory is unsafe") from exc
        os.close(parent_fd)
        try:
            opened = os.fstat(descriptor)
            mode = stat.S_IMODE(opened.st_mode)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (observed.st_dev, observed.st_ino)
                or opened.st_uid != self.uid
                or mode & 0o700 != 0o700
            ):
                raise UnsafeAgentLog("agent logs directory is not host-owned")
            identity = self._directory_identity(opened)
            self._verify_dir(descriptor, identity)
            return descriptor, identity
        except BaseException:
            os.close(descriptor)
            raise

    def _verify_dir(self, descriptor: int, expected: tuple[int, ...]) -> None:
        try:
            current = self.logs_dir.lstat()
            actual = os.fstat(descriptor)
        except OSError as exc:
            raise UnsafeAgentLog("agent logs directory changed") from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or not stat.S_ISDIR(actual.st_mode)
            or self._directory_identity(current) != expected
            or self._directory_identity(actual) != expected
        ):
            raise UnsafeAgentLog("agent logs directory changed")

    def read_text(
        self, path: Path, *, max_bytes: int = _MAX_AGENT_LOG_BYTES,
    ) -> tuple[str, tuple[int, ...]] | None:
        leaf = self._leaf(path)
        directory_fd, directory_identity = self._open_dir()
        try:
            try:
                observed = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                self._verify_dir(directory_fd, directory_identity)
                return None
            except OSError as exc:
                raise UnsafeAgentLog("agent log is unreadable") from exc
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_size > max_bytes
                or stat.S_IMODE(observed.st_mode) & 0o7000
            ):
                raise UnsafeAgentLog("agent log is not a bounded regular file")
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            try:
                file_fd = os.open(leaf, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise UnsafeAgentLog("agent log could not be opened safely") from exc
            try:
                opened = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                    or opened.st_size > max_bytes
                    or stat.S_IMODE(opened.st_mode) & 0o7000
                ):
                    raise UnsafeAgentLog("agent log changed before it was opened")
                chunks: list[bytes] = []
                remaining = max_bytes + 1
                while remaining:
                    chunk = os.read(file_fd, min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                after = os.fstat(file_fd)
                if (
                    len(payload) > max_bytes
                    or len(payload) != after.st_size
                    or self._fingerprint(after) != self._fingerprint(opened)
                ):
                    raise UnsafeAgentLog("agent log changed while it was read")
            except OSError as exc:
                raise UnsafeAgentLog("agent log could not be read safely") from exc
            finally:
                os.close(file_fd)
            self._verify_dir(directory_fd, directory_identity)
            return payload.decode("utf-8", errors="replace"), self._fingerprint(after)
        finally:
            os.close(directory_fd)

    def replace_text(
        self,
        path: Path,
        text: str,
        *,
        expected: tuple[int, ...] | None = None,
    ) -> bool:
        leaf = self._leaf(path)
        payload = text.encode("utf-8")
        if len(payload) > _MAX_AGENT_LOG_BYTES:
            raise UnsafeAgentLog("replacement agent log is too large")
        directory_fd, directory_identity = self._open_dir()
        temporary = f".dradar-log-{uuid.uuid4().hex}.tmp"
        temporary_exists = False
        try:
            matched = True
            if expected is not None:
                try:
                    current = os.stat(
                        leaf, dir_fd=directory_fd, follow_symlinks=False,
                    )
                except OSError:
                    matched = False
                else:
                    matched = self._fingerprint(current) == expected
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            try:
                temporary_fd = os.open(
                    temporary, flags, 0o600, dir_fd=directory_fd,
                )
                temporary_exists = True
            except OSError as exc:
                raise UnsafeAgentLog(
                    "safe agent log temporary could not be created",
                ) from exc
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(temporary_fd, view)
                    if written <= 0:
                        raise OSError("short write")
                    view = view[written:]
                os.fchmod(temporary_fd, 0o600)
                os.fsync(temporary_fd)
                created = os.fstat(temporary_fd)
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_nlink != 1
                    or created.st_uid != self.uid
                    or stat.S_IMODE(created.st_mode) != 0o600
                    or created.st_size != len(payload)
                ):
                    raise UnsafeAgentLog("safe agent log temporary is invalid")
            except OSError as exc:
                raise UnsafeAgentLog(
                    "safe agent log temporary could not be written",
                ) from exc
            finally:
                os.close(temporary_fd)
            try:
                if expected is not None:
                    try:
                        current = os.stat(
                            leaf, dir_fd=directory_fd, follow_symlinks=False,
                        )
                    except OSError:
                        matched = False
                    else:
                        matched = matched and self._fingerprint(current) == expected
                os.replace(
                    temporary,
                    leaf,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                temporary_exists = False
                os.fsync(directory_fd)
                published = os.stat(
                    leaf, dir_fd=directory_fd, follow_symlinks=False,
                )
            except OSError as exc:
                raise UnsafeAgentLog(
                    "agent log could not be atomically replaced",
                ) from exc
            if (
                not stat.S_ISREG(published.st_mode)
                or published.st_nlink != 1
                or published.st_uid != self.uid
                or stat.S_IMODE(published.st_mode) != 0o600
                or (published.st_dev, published.st_ino)
                != (created.st_dev, created.st_ino)
                or published.st_size != len(payload)
            ):
                raise UnsafeAgentLog("published agent log is invalid")
            self._verify_dir(directory_fd, directory_identity)
            return matched
        finally:
            if temporary_exists:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except OSError:
                    pass
            os.close(directory_fd)

    def redact_texts(
        self,
        paths: list[Path],
        sensitive_values: tuple[str, ...],
        marker: str,
        *,
        retain_paths: Iterable[Path] | None = None,
    ) -> dict[Path, str]:
        retained = set(paths if retain_paths is None else retain_paths)
        safe: dict[Path, str] = {}
        rejected = False
        for path in paths:
            try:
                snapshot = self.read_text(path)
            except UnsafeAgentLog:
                self.replace_text(path, self.REJECTED)
                rejected = True
                continue
            if snapshot is None:
                continue
            text, identity = snapshot
            redacted = text
            for value in sensitive_values:
                if value and value in redacted:
                    redacted = redacted.replace(value, marker)
            matched = self.replace_text(path, redacted, expected=identity)
            if redacted != text or not matched:
                rejected = True
            elif path in retained:
                safe[path] = text
        if rejected:
            raise ValueError(
                "credential material or an unsafe entry reached agent output; "
                "logs were sanitized and the run was rejected"
            )
        return safe


@dataclass(frozen=True)
class AgentIdentity:
    """Numeric container identity used by Pier's unprivileged agent."""

    uid: int
    gid: int
    groups: tuple[int, ...]

    def __post_init__(self) -> None:
        values = (self.uid, self.gid, *self.groups)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError("agent identity must contain non-negative integers")
        if not self.groups or self.gid not in self.groups:
            raise ValueError("agent supplementary groups must include the primary gid")
        if self.uid == 0 or self.gid == 0 or 0 in self.groups:
            raise ValueError("checkpoint agent must not have root identity or groups")


@dataclass(frozen=True)
class StatePath:
    """One credential-free provider path copied into the checkpoint."""

    name: str
    remote_path: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", self.name):
            raise ValueError("checkpoint state name is invalid")
        if not self.remote_path.startswith("/"):
            raise ValueError("checkpoint state path must be absolute")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contains_sensitive_key(value: Any) -> bool:
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


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    if _contains_sensitive_key(data):
        raise CheckpointError("checkpoint manifest contains a sensitive field name")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("checkpoint manifest is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointIncompatibleError("checkpoint schema is unsupported")
    if _contains_sensitive_key(value):
        raise CheckpointError("checkpoint manifest contains a sensitive field name")
    checkpoint_id = value.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or _ID_RE.fullmatch(checkpoint_id) is None:
        raise CheckpointError("checkpoint id is invalid")
    for name in ("assignment_id", "phase", "created_at"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise CheckpointError(f"checkpoint is missing {name}")
    generation = value.get("resume_generation", 0)
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise CheckpointError("checkpoint generation is invalid")
    for name, expected in _FIXED_ARTIFACTS.items():
        if value.get(name, expected) != expected:
            raise CheckpointError(f"checkpoint {name} path is unsafe")
    return value


def _lexists(path: Path) -> bool:
    """Like ``os.path.lexists`` without silently following a dangling link."""

    return os.path.lexists(path)


def _validate_regular_tree(path: Path, *, label: str) -> None:
    """Allow only same-filesystem regular files and directories.

    ``lstat`` is intentional: symlinks (including dangling ones), FIFOs,
    sockets, and device nodes are checkpoint corruption, not absent data.
    """

    try:
        root_metadata = path.lstat()
    except OSError as exc:
        raise CheckpointError(f"{label} is unreadable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) and not stat.S_ISREG(
        root_metadata.st_mode,
    ):
        raise CheckpointError(f"{label} contains a special file")
    root_device = root_metadata.st_dev
    if stat.S_ISREG(root_metadata.st_mode):
        if root_metadata.st_nlink != 1:
            raise CheckpointError(f"{label} contains a multiply linked file")
        return
    def fail_walk(error: OSError) -> None:
        raise error

    try:
        walker = os.walk(path, followlinks=False, onerror=fail_walk)
        for current, directories, files in walker:
            base = Path(current)
            for name in (*directories, *files):
                candidate = base / name
                try:
                    metadata = candidate.lstat()
                except OSError as exc:
                    raise CheckpointError(
                        f"{label} changed during validation",
                    ) from exc
                if metadata.st_dev != root_device:
                    raise CheckpointError(f"{label} crossed a filesystem boundary")
                if stat.S_ISDIR(metadata.st_mode):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise CheckpointError(f"{label} contains a special file")
                if metadata.st_nlink != 1:
                    raise CheckpointError(f"{label} contains a multiply linked file")
    except CheckpointError:
        raise
    except OSError as exc:
        raise CheckpointError(f"{label} is unreadable") from exc


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    """Fields that must remain stable while a seized inode is copied."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _copy_seized_tree(source: Path, destination: Path) -> None:
    """Copy seized agent inodes into a new host-owned, private inode tree.

    An agent may retain writable file descriptors after root changes ownership.
    Publishing the original staging inode would therefore be unsafe.  Every
    source node is opened without following links, copied to a fresh inode, and
    fingerprinted before and after the copy.  Writes through a retained source
    descriptor change ctime/mtime and fail the snapshot instead of reaching the
    published generation.
    """

    if _lexists(destination):
        raise CheckpointError("checkpoint copy destination already exists")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    source_parent_fd = os.open(
        source.parent, os.O_RDONLY | directory | nofollow | cloexec,
    )
    try:
        observed_source = os.stat(
            source.name, dir_fd=source_parent_fd, follow_symlinks=False,
        )
        if not stat.S_ISDIR(observed_source.st_mode):
            raise CheckpointError("checkpoint seized staging is not a directory")
        source_fd = os.open(
            source.name,
            os.O_RDONLY | directory | nofollow | cloexec,
            dir_fd=source_parent_fd,
        )
        opened_source = os.fstat(source_fd)
        if (opened_source.st_dev, opened_source.st_ino) != (
            observed_source.st_dev, observed_source.st_ino,
        ):
            os.close(source_fd)
            raise CheckpointError("checkpoint seized staging changed before copy")
    finally:
        os.close(source_parent_fd)
    try:
        destination.mkdir(mode=0o700)
        destination_fd = os.open(
            destination, os.O_RDONLY | directory | nofollow | cloexec,
        )
    except BaseException:
        os.close(source_fd)
        raise

    copied_entries = 0
    copied_bytes = 0

    def copy_directory(
        source_dir_fd: int, destination_dir_fd: int, *, depth: int,
    ) -> None:
        nonlocal copied_entries, copied_bytes
        if depth > _MAX_CHECKPOINT_DEPTH:
            raise CheckpointError("checkpoint copy exceeds the depth limit")
        directory_before = _metadata_fingerprint(os.fstat(source_dir_fd))
        names = sorted(os.listdir(source_dir_fd))
        for name in names:
            copied_entries += 1
            if copied_entries > _MAX_CHECKPOINT_FILES:
                raise CheckpointError("checkpoint copy exceeds the entry-count limit")
            observed = os.stat(name, dir_fd=source_dir_fd, follow_symlinks=False)
            if observed.st_dev != directory_before[0]:
                raise CheckpointError("checkpoint copy crossed a filesystem boundary")
            if stat.S_ISDIR(observed.st_mode):
                child_source_fd = os.open(
                    name,
                    os.O_RDONLY | directory | nofollow | cloexec,
                    dir_fd=source_dir_fd,
                )
                actual = os.fstat(child_source_fd)
                if (actual.st_dev, actual.st_ino) != (
                    observed.st_dev, observed.st_ino,
                ):
                    os.close(child_source_fd)
                    raise CheckpointError("checkpoint source changed during copy")
                os.mkdir(name, mode=0o700, dir_fd=destination_dir_fd)
                child_destination_fd = os.open(
                    name,
                    os.O_RDONLY | directory | nofollow | cloexec,
                    dir_fd=destination_dir_fd,
                )
                try:
                    copy_directory(
                        child_source_fd, child_destination_fd, depth=depth + 1,
                    )
                finally:
                    os.close(child_destination_fd)
                    os.close(child_source_fd)
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise CheckpointError(
                    "checkpoint copy contains a special or multiply linked file",
                )
            copied_bytes += observed.st_size
            if observed.st_size > _MAX_CHECKPOINT_FILE_BYTES:
                raise CheckpointError("checkpoint copy contains an oversized file")
            if copied_bytes > _MAX_CHECKPOINT_TOTAL_BYTES:
                raise CheckpointError("checkpoint copy exceeds the total-size limit")
            child_source_fd = os.open(
                name,
                os.O_RDONLY | nofollow | nonblock | cloexec,
                dir_fd=source_dir_fd,
            )
            try:
                before = os.fstat(child_source_fd)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or (before.st_dev, before.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    raise CheckpointError("checkpoint source changed during copy")
                child_destination_fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | cloexec,
                    0o600,
                    dir_fd=destination_dir_fd,
                )
                try:
                    while True:
                        chunk = os.read(child_source_fd, 1024 * 1024)
                        if not chunk:
                            break
                        view = memoryview(chunk)
                        while view:
                            written = os.write(child_destination_fd, view)
                            view = view[written:]
                    os.fsync(child_destination_fd)
                    os.fchmod(child_destination_fd, 0o600)
                finally:
                    os.close(child_destination_fd)
                after = os.fstat(child_source_fd)
                if _metadata_fingerprint(before) != _metadata_fingerprint(after):
                    raise CheckpointError("checkpoint source changed during copy")
            finally:
                os.close(child_source_fd)
        if sorted(os.listdir(source_dir_fd)) != names:
            raise CheckpointError("checkpoint source changed during copy")
        directory_after = _metadata_fingerprint(os.fstat(source_dir_fd))
        if directory_before != directory_after:
            raise CheckpointError("checkpoint source changed during copy")
        os.fchmod(destination_dir_fd, 0o700)
        os.fsync(destination_dir_fd)

    try:
        copy_directory(source_fd, destination_fd, depth=0)
    except BaseException:
        os.close(destination_fd)
        os.close(source_fd)
        shutil.rmtree(destination, ignore_errors=True)
        raise
    else:
        os.close(destination_fd)
        os.close(source_fd)
    _validate_regular_tree(destination, label="checkpoint copied staging")


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CheckpointError("checkpoint artifact path is invalid")
    if root.is_symlink():
        raise CheckpointError("checkpoint root is a symlink")
    canonical_root = root.resolve()
    path = root / relative
    if path.is_symlink():
        raise CheckpointError("checkpoint artifact is a symlink")
    canonical = path.resolve()
    if canonical_root not in canonical.parents:
        raise CheckpointError("checkpoint artifact escaped its directory")
    if _lexists(path):
        _validate_regular_tree(path, label="checkpoint artifact")
    return path


def _snapshot_payload_dir(root: Path) -> Path:
    """Resolve the atomically published snapshot generation.

    Checkpoints produced before generation publishing stored payloads directly
    below ``root`` and remain readable.  Once a pointer exists, it is strict:
    a dangling link, malformed generation, or missing directory fails closed.
    """

    _validate_regular_tree(root, label="checkpoint root")
    pointer = root / "current-generation"
    if not _lexists(pointer):
        return root
    _validate_regular_tree(pointer, label="checkpoint generation pointer")
    if not pointer.is_file() or pointer.stat().st_size > 128:
        raise CheckpointError("checkpoint generation pointer is invalid")
    generation = pointer.read_text(encoding="ascii", errors="strict").strip()
    if _ID_RE.fullmatch(generation) is None:
        raise CheckpointError("checkpoint generation pointer is invalid")
    snapshots = _safe_path(root, "snapshots")
    payload = _safe_path(snapshots, generation)
    if not payload.is_dir():
        raise CheckpointError("checkpoint snapshot generation is missing")
    return payload


def _bytes_contains_secret(
    value: bytes, needles: tuple[bytes, ...], *, generic: bool = True,
) -> bool:
    return any(needle in value for needle in needles) or (
        generic and _GENERIC_SECRET_RE.search(value) is not None
    )


def _stream_contains_any(
    handle: Any,
    needles: tuple[bytes, ...],
    *,
    generic: bool = True,
    max_bytes: int | None = None,
) -> bool:
    longest_exact = max((len(value) for value in needles), default=0)
    # Generic credential shapes are unbounded.  Retaining 8 KiB catches tokens
    # split at a read boundary without retaining an entire large artifact.
    overlap = max(longest_exact - 1, 8192 if generic else 0)
    previous = b""
    consumed = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return False
        consumed += len(chunk)
        if max_bytes is not None and consumed > max_bytes:
            raise CheckpointError("checkpoint artifact exceeds the size limit")
        combined = previous + chunk
        if _bytes_contains_secret(combined, needles, generic=generic):
            return True
        previous = combined[-overlap:] if overlap else b""


class _BoundedGzipReader:
    """Seekable gzip reader that rejects tar control-record allocation bombs.

    ``tarfile`` reads PAX and GNU long-name records before yielding a member.
    Without this boundary, a tiny gzip stream can declare a multi-gigabyte
    control record and make ``tarfile`` allocate it before our member limits
    run.  Ordinary payload reads stay chunked by ``_stream_contains_any``.
    """

    def __init__(self, stream: gzip.GzipFile) -> None:
        self._stream = stream

    def tell(self) -> int:
        return int(self._stream.tell())

    def read(self, size: int = -1) -> bytes:
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > _MAX_ARCHIVE_CONTROL_READ_BYTES
        ):
            raise CheckpointError(
                "checkpoint archive requested an oversized control record",
            )
        current = self.tell()
        if current + size > _MAX_ARCHIVE_STREAM_BYTES:
            raise CheckpointError("checkpoint archive exceeds the stream-size limit")
        data = self._stream.read(size)
        if self.tell() > _MAX_ARCHIVE_STREAM_BYTES:
            raise CheckpointError("checkpoint archive exceeds the stream-size limit")
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise CheckpointError("checkpoint archive seek is invalid")
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self.tell() + offset
        else:
            raise CheckpointError("checkpoint archive end-relative seek is unsafe")
        if target < 0 or target > _MAX_ARCHIVE_STREAM_BYTES:
            raise CheckpointError("checkpoint archive seek exceeds the size limit")
        result = int(self._stream.seek(offset, whence))
        if result != target:
            raise CheckpointError("checkpoint archive seek was inconsistent")
        return result


def _validate_archive(path: Path, needles: tuple[bytes, ...]) -> bool:
    """Validate a generated untracked archive and scan its regular files."""

    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_CHECKPOINT_FILE_BYTES
        ):
            raise CheckpointError("checkpoint archive metadata is unsafe")
        before = _metadata_fingerprint(metadata)
        with path.open("rb") as raw_archive:
            opened = os.fstat(raw_archive.fileno())
            if _metadata_fingerprint(opened) != before:
                raise CheckpointError("checkpoint archive changed before validation")
            with gzip.GzipFile(fileobj=raw_archive, mode="rb") as inflated:
                bounded = _BoundedGzipReader(inflated)
                # Tar metadata is not limited to the fields exposed by
                # ``TarInfo``.  GNU/PAX control records, owner names, global
                # headers, padding, and unknown extension records all survive
                # decompression and could otherwise carry an exact credential
                # into a published checkpoint.  Scan the complete bounded tar
                # byte stream first, then rewind for the structural parser.
                if _stream_contains_any(
                    bounded,
                    needles,
                    max_bytes=_MAX_ARCHIVE_STREAM_BYTES,
                ):
                    return True
                bounded.seek(0)
                archive_context = tarfile.open(
                    fileobj=bounded, mode="r:",
                )
                with archive_context as archive:
                    if _archive_contains_rejected_data(archive, needles):
                        return True
            after = os.fstat(raw_archive.fileno())
            if _metadata_fingerprint(after) != before:
                raise CheckpointError("checkpoint archive changed during validation")
    except CheckpointError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise CheckpointError("checkpoint archive is unreadable") from exc
    return False


def _archive_contains_rejected_data(
    archive: tarfile.TarFile, needles: tuple[bytes, ...],
) -> bool:
    """Validate one already bounded tar stream and scan regular members."""

    member_count = 0
    expanded_bytes = 0
    for member in archive:
        member_count += 1
        if member_count > _MAX_ARCHIVE_MEMBERS:
            raise CheckpointError("checkpoint archive has too many members")
        relative = Path(member.name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise CheckpointError("checkpoint archive contains an unsafe member")
        metadata_fields = [
            member.name,
            member.linkname,
            member.uname,
            member.gname,
        ]
        metadata_fields.extend(archive.pax_headers)
        metadata_fields.extend(archive.pax_headers.values())
        metadata_fields.extend(member.pax_headers)
        metadata_fields.extend(member.pax_headers.values())
        if any(
            _bytes_contains_secret(os.fsencode(value), needles)
            for value in metadata_fields
            if value
        ):
            return True
        if member.isfile():
            if member.size < 0 or member.size > _MAX_ARCHIVE_MEMBER_BYTES:
                raise CheckpointError(
                    "checkpoint archive contains an oversized member",
                )
            expanded_bytes += member.size
            if expanded_bytes > _MAX_ARCHIVE_TOTAL_BYTES:
                raise CheckpointError(
                    "checkpoint archive exceeds the total-size limit",
                )
            source = archive.extractfile(member)
            if source is None:
                raise CheckpointError("checkpoint archive member is unreadable")
            with source:
                if _stream_contains_any(
                    source, needles, max_bytes=member.size,
                ):
                    return True
    return False


def _path_contains_any(path: Path, needles: tuple[bytes, ...]) -> bool:
    _validate_regular_tree(path, label="checkpoint secret scan")
    candidates = [path]
    if path.is_dir():
        candidates.extend(path.rglob("*"))
    for candidate in candidates:
        if _bytes_contains_secret(os.fsencode(candidate.name), needles):
            return True
        try:
            attribute_names = os.listxattr(candidate, follow_symlinks=False)
        except (AttributeError, NotImplementedError):
            attribute_names = []
        except OSError as exc:
            raise CheckpointError("checkpoint xattrs are unreadable") from exc
        for attribute_name in attribute_names:
            encoded_name = os.fsencode(attribute_name)
            try:
                attribute_value = os.getxattr(
                    candidate, attribute_name, follow_symlinks=False,
                )
            except OSError as exc:
                raise CheckpointError("checkpoint xattrs are unreadable") from exc
            if _bytes_contains_secret(encoded_name, needles) or _bytes_contains_secret(
                attribute_value, needles,
            ):
                return True
        metadata = candidate.lstat()
        if stat.S_ISREG(metadata.st_mode):
            with candidate.open("rb") as handle:
                if _stream_contains_any(handle, needles):
                    return True
    return False


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("checkpoint_enabled must be a boolean")


def _untracked_preflight_program() -> str:
    """Return a bounded, descriptor-relative validator for Git's NUL list."""

    return f'''import os
import stat
import sys

MAX_FILES = {_MAX_CHECKPOINT_FILES}
MAX_DEPTH = {_MAX_CHECKPOINT_DEPTH}
MAX_FILE_BYTES = {_MAX_CHECKPOINT_FILE_BYTES}
MAX_TOTAL_BYTES = {_MAX_CHECKPOINT_TOTAL_BYTES}
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)
CLOEXEC = getattr(os, "O_CLOEXEC", 0)

def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(74)

def checked_path(root_fd, root_device, raw):
    if not raw or os.path.isabs(raw):
        fail("untracked checkpoint path is unsafe")
    parts = raw.split(b"/")
    if (
        len(parts) - 1 > MAX_DEPTH
        or any(part in (b"", b".", b"..") for part in parts)
    ):
        fail("untracked checkpoint path is unsafe")
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or observed.st_dev != root_device
            ):
                fail("untracked checkpoint path crosses an unsafe directory")
            child_fd = os.open(
                part, os.O_RDONLY | DIRECTORY | NOFOLLOW | CLOEXEC,
                dir_fd=parent_fd,
            )
            actual = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(actual.st_mode)
                or actual.st_dev != root_device
                or (actual.st_dev, actual.st_ino)
                != (observed.st_dev, observed.st_ino)
            ):
                os.close(child_fd)
                fail("untracked checkpoint path changed during validation")
            os.close(parent_fd)
            parent_fd = child_fd
        metadata = os.stat(
            parts[-1], dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_dev != root_device
        ):
            fail("untracked checkpoint contains a special or linked file")
        return metadata.st_size
    except OSError:
        fail("untracked checkpoint path changed during validation")
    finally:
        os.close(parent_fd)

if len(sys.argv) != 3:
    fail("invalid untracked checkpoint validator invocation")
root_fd = os.open(
    os.fsencode(sys.argv[1]), os.O_RDONLY | DIRECTORY | NOFOLLOW | CLOEXEC,
)
list_fd = os.open(
    os.fsencode(sys.argv[2]), os.O_RDONLY | NOFOLLOW | CLOEXEC,
)
try:
    root_metadata = os.fstat(root_fd)
    list_metadata = os.fstat(list_fd)
    if not stat.S_ISDIR(root_metadata.st_mode):
        fail("checkpoint worktree is unsafe")
    if not stat.S_ISREG(list_metadata.st_mode) or list_metadata.st_nlink != 1:
        fail("untracked checkpoint list is unsafe")
    pending = b""
    count = 0
    total_bytes = 0
    while True:
        chunk = os.read(list_fd, 64 * 1024)
        if not chunk:
            break
        pending += chunk
        parts = pending.split(b"\\0")
        pending = parts.pop()
        for raw in parts:
            count += 1
            if count > MAX_FILES:
                fail("untracked checkpoint exceeds the entry-count limit")
            size = checked_path(root_fd, root_metadata.st_dev, raw)
            if size > MAX_FILE_BYTES:
                fail("untracked checkpoint contains an oversized file")
            total_bytes += size
            if total_bytes > MAX_TOTAL_BYTES:
                fail("untracked checkpoint exceeds the total-size limit")
    if pending:
        fail("untracked checkpoint list is not NUL terminated")
finally:
    os.close(list_fd)
    os.close(root_fd)
'''


def _capture_script(
    *, workdir: str, state_paths: Iterable[StatePath],
    session_probe: str | None, agent_identity: AgentIdentity,
) -> str:
    """Build the trusted capture program that always runs as the agent.

    It has no ownership-changing or publication capability.  Git, tar, provider
    state reads, secret filtering, and the provider session probe all happen
    after the numeric Pier agent identity (including groups) is re-verified.
    """

    copy_lines: list[str] = []
    for item in state_paths:
        source = shlex.quote(item.remote_path)
        target = shlex.quote(item.name)
        copy_lines.extend([
            f"if [ -e {source} ] || [ -L {source} ]; then",
            f"  validate_regular_tree {source}",
            f"  if [ -d {source} ]; then cp -R {source} \"$state_tmp\"/{target};",
            f"  else cp {source} \"$state_tmp\"/{target}; fi",
            f"  validate_regular_tree \"$state_tmp\"/{target}",
            "fi",
        ])
    probe = ""
    if session_probe:
        probe = f"""
if [ ! -e "$staging/session-omitted-sensitive" ]; then
  session_id=$({session_probe} 2>/dev/null || true)
  case "$session_id" in
    ''|*[!A-Za-z0-9._:-]*) ;;
    *)
      if [ "${{#session_id}}" -ge 8 ] && [ "${{#session_id}}" -le 160 ]; then
        printf '%s\\n' "$session_id" > "$staging/session-id"
      fi ;;
  esac
fi
"""
    expected_groups = " ".join(str(value) for value in agent_identity.groups)
    copies = "\n".join(copy_lines)
    untracked_preflight = shlex.quote(_untracked_preflight_program())
    return f"""#!/bin/sh
set -eu
umask 077
workdir={shlex.quote(workdir)}
expected_uid={agent_identity.uid}
expected_gid={agent_identity.gid}
expected_groups={shlex.quote(expected_groups)}
secret_re='(sk-(ant-|proj-)?[A-Za-z0-9_-]{{16,}}|ghp_[A-Za-z0-9]{{20,}}|github_pat_[A-Za-z0-9_]{{20,}}|gAAAAA[A-Za-z0-9_-]{{40,}}|eyJ[A-Za-z0-9_-]{{10,}}[.][A-Za-z0-9_-]{{10,}}[.][A-Za-z0-9_-]{{10,}})'
[ "$#" -eq 2 ] || exit 64
staging=$1
base=$2
[ "$(id -u)" = "$expected_uid" ] || exit 77
[ "$(id -g)" = "$expected_gid" ] || exit 77
[ "$(id -G)" = "$expected_groups" ] || exit 77
case "$base" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]* ) ;;
  *) exit 64 ;;
esac
[ "${{#base}}" -eq 40 ] || exit 64
[ -d "$staging" ] && [ ! -L "$staging" ] || exit 74
validate_regular_tree() {{
  candidate=$1
  [ ! -L "$candidate" ] || return 74
  [ -f "$candidate" ] || [ -d "$candidate" ] || return 74
  special=$(find -P "$candidate" -xdev \
    ! \\( -type f -o -type d \\) -print -quit)
  [ -z "$special" ] || return 74
  multiply_linked=$(find -P "$candidate" -xdev -type f ! -links 1 -print -quit)
  [ -z "$multiply_linked" ] || return 74
}}
validate_regular_tree "$staging"
# Git deliberately omits FIFOs, sockets and device nodes from its untracked
# listing. Reject them before any archive command can race into opening one.
workspace_special=$(find -P "$workdir" -xdev \
  ! \\( -type f -o -type d -o -type l \\) -print -quit)
[ -z "$workspace_special" ] || exit 74
git -C "$workdir" diff --no-ext-diff --no-textconv --binary "$base" -- \
  > "$staging/workspace.patch"
# ``--binary`` may zlib/base85-encode the exact bytes that git apply will
# reconstruct. Derive the short-lived plaintext scan from that exact patch,
# not from a second read of the concurrently changing live worktree.
scan_index="$staging/.tracked-scan.index"
rm -f "$scan_index" "$scan_index.lock"
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
  GIT_NO_REPLACE_OBJECTS=1 GIT_INDEX_FILE="$scan_index" \
  git -C "$workdir" read-tree "$base"
if [ -s "$staging/workspace.patch" ]; then
  GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 GIT_INDEX_FILE="$scan_index" \
    git -C "$workdir" apply --cached --binary --whitespace=nowarn \
      "$staging/workspace.patch"
fi
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
  GIT_NO_REPLACE_OBJECTS=1 GIT_INDEX_FILE="$scan_index" \
  git -C "$workdir" diff --cached --no-ext-diff --no-textconv --text \
    "$base" -- > "$staging/{_TRACKED_SCAN_ARTIFACT}"
rm -f "$scan_index" "$scan_index.lock"
if LC_ALL=C grep -aEq "$secret_re" \
    "$staging/workspace.patch" "$staging/{_TRACKED_SCAN_ARTIFACT}"; then
  rm -f "$staging/workspace.patch" "$staging/{_TRACKED_SCAN_ARTIFACT}"
  printf 'credential-shaped content detected in workspace patch\\n' \
    > "$staging/invalid-secret"
fi
git -C "$workdir" status --short > "$staging/progress-summary.txt"
printf '\\nChanged files:\\n' >> "$staging/progress-summary.txt"
git -C "$workdir" diff --no-ext-diff --no-textconv --stat "$base" -- \
  >> "$staging/progress-summary.txt"
untracked_tmp="$staging/.untracked-files.tmp"
untracked_list="$staging/.untracked-files"
rm -f "$untracked_tmp" "$untracked_list"
git -C "$workdir" ls-files --others --exclude-standard -z \
  > "$untracked_tmp"
mv "$untracked_tmp" "$untracked_list"
/usr/bin/python3 -c {untracked_preflight} "$workdir" "$untracked_list"
tar -C "$workdir" --null -czf "$staging/untracked.tar.gz" \
  --exclude='.env' --exclude='.env.*' --exclude='auth.json' \
  --exclude='*.pem' --exclude='*.key' --exclude='credentials*' \
  --exclude='token*' --exclude='secret*' --exclude='password*' \
  --exclude='*.p12' --exclude='*.pfx' --files-from="$untracked_list"
rm -f "$untracked_list"
if tar -xOzf "$staging/untracked.tar.gz" 2>/dev/null | \
    LC_ALL=C grep -aEq "$secret_re"; then
  rm -f "$staging/untracked.tar.gz"
  printf 'credential-shaped content detected in untracked files\\n' \
    > "$staging/invalid-secret"
fi
state_tmp="$staging/provider-state"
mkdir "$state_tmp"
{copies}
validate_regular_tree "$state_tmp"
if LC_ALL=C grep -aErq "$secret_re" "$state_tmp" 2>/dev/null; then
  rm -rf "$state_tmp"
  printf 'provider state omitted because it contained credential-shaped content\\n' \
    > "$staging/session-omitted-sensitive"
fi
{probe}
date -u +%Y-%m-%dT%H:%M:%SZ > "$staging/last_heartbeat"
validate_regular_tree "$staging"
"""


def _supervisor_script(
    *, checkpoint_dir: str, runtime_dir: str, capture_sha256: str,
    agent_identity: AgentIdentity, host_uid: int, host_gid: int,
) -> str:
    """Build the root-only, descriptor-relative staging supervisor."""

    for name, value in (("host_uid", host_uid), ("host_gid", host_gid)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not re.fullmatch(r"[0-9a-f]{64}", capture_sha256):
        raise ValueError("capture sha256 is invalid")
    return f'''#!/usr/bin/python3
import hashlib
import fcntl
import os
import re
import stat
import sys

CHECKPOINT = {checkpoint_dir!r}
RUNTIME = {runtime_dir!r}
CAPTURE_SHA = {capture_sha256!r}
AGENT_UID = {agent_identity.uid}
AGENT_GID = {agent_identity.gid}
HOST_UID = {host_uid}
HOST_GID = {host_gid}
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
NONBLOCK = getattr(os, "O_NONBLOCK", 0)
CLOEXEC = getattr(os, "O_CLOEXEC", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)
MAX_CHECKPOINT_FILES = {_MAX_CHECKPOINT_FILES}
MAX_CHECKPOINT_DEPTH = {_MAX_CHECKPOINT_DEPTH}
MAX_CHECKPOINT_FILE_BYTES = {_MAX_CHECKPOINT_FILE_BYTES}
MAX_CHECKPOINT_TOTAL_BYTES = {_MAX_CHECKPOINT_TOTAL_BYTES}

def fail(message, code=74):
    print(message, file=sys.stderr)
    raise SystemExit(code)

def open_directory(path):
    try:
        fd = os.open(path, os.O_RDONLY | DIRECTORY | NOFOLLOW)
    except OSError:
        fail("unsafe checkpoint directory")
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        fail("unsafe checkpoint directory")
    return fd

def open_child_directory(parent_fd, name, expected):
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(observed.st_mode):
        fail("checkpoint contains a special file")
    child_fd = os.open(
        name, os.O_RDONLY | DIRECTORY | NOFOLLOW, dir_fd=parent_fd,
    )
    actual = os.fstat(child_fd)
    if (actual.st_dev, actual.st_ino) != (observed.st_dev, observed.st_ino):
        os.close(child_fd)
        fail("checkpoint changed during validation")
    if actual.st_dev != expected:
        os.close(child_fd)
        fail("checkpoint crossed a filesystem boundary")
    return child_fd

def validate_checkpoint_parent(checkpoint_fd):
    metadata = os.fstat(checkpoint_fd)
    if not (
        metadata.st_uid == 0
        and metadata.st_gid == AGENT_GID
        and stat.S_IMODE(metadata.st_mode) == 0o750
    ):
        fail("checkpoint staging parent ownership is unsafe")

def open_snapshot_lock(checkpoint_fd, device):
    lock_fd = open_child_directory(checkpoint_fd, "snapshot.lock", device)
    metadata = os.fstat(lock_fd)
    if (
        metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(lock_fd)
        fail("checkpoint snapshot lock ownership is unsafe")
    return lock_fd

def new_budget():
    # Mutable pair: [entry count, aggregate regular-file bytes].
    return [0, 0]

def account_entry(metadata, depth, budget):
    if depth > MAX_CHECKPOINT_DEPTH:
        fail("checkpoint tree exceeds the depth limit")
    budget[0] += 1
    if budget[0] > MAX_CHECKPOINT_FILES:
        fail("checkpoint tree exceeds the entry-count limit")
    if stat.S_ISREG(metadata.st_mode):
        if metadata.st_size > MAX_CHECKPOINT_FILE_BYTES:
            fail("checkpoint tree contains an oversized file")
        budget[1] += metadata.st_size
        if budget[1] > MAX_CHECKPOINT_TOTAL_BYTES:
            fail("checkpoint tree exceeds the total-size limit")

def validate_tree(fd, device, depth=0, budget=None):
    if depth > MAX_CHECKPOINT_DEPTH:
        fail("checkpoint tree exceeds the depth limit")
    if budget is None:
        budget = new_budget()
    with os.scandir(fd) as entries:
        for entry in entries:
            name = entry.name
            metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
            account_entry(metadata, depth, budget)
            if metadata.st_dev != device:
                fail("checkpoint crossed a filesystem boundary")
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = open_child_directory(fd, name, device)
                try:
                    validate_tree(child_fd, device, depth + 1, budget)
                finally:
                    os.close(child_fd)
            elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                fail("checkpoint contains a special or multiply linked file")

def seize_tree_contents(fd, device, depth, budget):
    if depth > MAX_CHECKPOINT_DEPTH:
        fail("checkpoint tree exceeds the depth limit")
    with os.scandir(fd) as entries:
        for entry in entries:
            name = entry.name
            metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
            account_entry(metadata, depth, budget)
            if metadata.st_dev != device:
                fail("checkpoint crossed a filesystem boundary")
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = open_child_directory(fd, name, device)
                try:
                    os.fchmod(child_fd, 0)
                    seize_tree_contents(
                        child_fd, device, depth + 1, budget,
                    )
                    os.fchown(child_fd, HOST_UID, HOST_GID)
                    os.fchmod(child_fd, 0o700)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                file_fd = os.open(
                    name,
                    os.O_RDONLY | NOFOLLOW | NONBLOCK | CLOEXEC,
                    dir_fd=fd,
                )
                try:
                    actual = os.fstat(file_fd)
                    if (
                        not stat.S_ISREG(actual.st_mode)
                        or actual.st_nlink != 1
                        or actual.st_dev != device
                        or actual.st_size != metadata.st_size
                        or (actual.st_dev, actual.st_ino)
                        != (metadata.st_dev, metadata.st_ino)
                    ):
                        fail("checkpoint changed during seize")
                    os.fchmod(file_fd, 0)
                    os.fchown(file_fd, HOST_UID, HOST_GID)
                    os.fchmod(file_fd, 0o600)
                finally:
                    os.close(file_fd)
            else:
                fail("checkpoint contains a special or multiply linked file")

def seize_tree(fd, device):
    validate_tree(fd, device)
    seize_tree_contents(fd, device, 0, new_budget())
    validate_tree(fd, device)

def delete_tree(fd, device, depth=0, budget=None):
    if depth > MAX_CHECKPOINT_DEPTH:
        fail("checkpoint tree exceeds the depth limit")
    if budget is None:
        budget = new_budget()
    with os.scandir(fd) as entries:
        for entry in entries:
            name = entry.name
            metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
            account_entry(metadata, depth, budget)
            if metadata.st_dev != device:
                fail("checkpoint crossed a filesystem boundary")
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = open_child_directory(fd, name, device)
                try:
                    delete_tree(child_fd, device, depth + 1, budget)
                finally:
                    os.close(child_fd)
                os.rmdir(name, dir_fd=fd)
            else:
                # Descriptor-relative unlinking never follows the entry. Abort
                # must still be able to reap an attacker-created FIFO, socket,
                # symlink or hard link, subject to the same count/depth/device
                # budget as a regular staging tree.
                os.unlink(name, dir_fd=fd)

def validate_runtime():
    runtime_fd = open_directory(RUNTIME)
    try:
        capture_fd = os.open("capture", os.O_RDONLY | NOFOLLOW, dir_fd=runtime_fd)
        try:
            metadata = os.fstat(capture_fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != 0
                or metadata.st_gid != 0
                or stat.S_IMODE(metadata.st_mode) != 0o555
            ):
                fail("capture runtime metadata is unsafe")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(capture_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != CAPTURE_SHA:
                fail("capture runtime checksum mismatch")
        finally:
            os.close(capture_fd)
    except BaseException:
        os.close(runtime_fd)
        raise
    return runtime_fd

def open_operation_lock(runtime_fd):
    name = ".supervisor-operation.lock"
    flags = os.O_RDWR | os.O_CREAT | NOFOLLOW
    lock_fd = os.open(name, flags, 0o600, dir_fd=runtime_fd)
    metadata = os.fstat(lock_fd)
    runtime_device = os.fstat(runtime_fd).st_dev
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_dev != runtime_device
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(lock_fd)
        fail("checkpoint supervisor lock is unsafe")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    return lock_fd

def validate_abort_marker(runtime_fd, name):
    try:
        metadata = os.stat(name, dir_fd=runtime_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_dev != os.fstat(runtime_fd).st_dev
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        fail("checkpoint abort marker is unsafe")
    return True

def ensure_abort_marker(runtime_fd, name):
    try:
        marker_fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW,
            0o600, dir_fd=runtime_fd,
        )
    except FileExistsError:
        if not validate_abort_marker(runtime_fd, name):
            fail("checkpoint abort marker disappeared")
        return
    try:
        os.fchmod(marker_fd, 0o600)
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)

if len(sys.argv) != 3 or sys.argv[1] not in {{"prepare", "seize", "release", "abort"}}:
    fail("invalid supervisor action", 64)
action, generation = sys.argv[1:]
if re.fullmatch(r"[A-Za-z0-9._-]{{8,64}}", generation) is None:
    fail("invalid checkpoint generation", 64)
stage_name = ".snapshot-stage-" + generation
abort_name = ".snapshot-aborted-" + generation
runtime_fd = validate_runtime()
operation_fd = open_operation_lock(runtime_fd)
try:
    if action == "abort":
        # This root-only tombstone closes the cancel-before-prepare race:
        # a delayed prepare process takes the same lock and must refuse this
        # generation even after abort has already observed no staging tree.
        ensure_abort_marker(runtime_fd, abort_name)
    elif action == "prepare" and validate_abort_marker(runtime_fd, abort_name):
        fail("checkpoint generation was already aborted", 75)
    checkpoint_fd = open_directory(CHECKPOINT)
    try:
      checkpoint_device = os.fstat(checkpoint_fd).st_dev
      if action == "prepare":
        validate_checkpoint_parent(checkpoint_fd)
        # The staging root is root-owned for the whole checkpoint lifetime.
        # Clear only stale supervisor entries, then create root-authorized
        # names; the group-traversing model cannot rename either child.
        delete_tree(checkpoint_fd, checkpoint_device)
        validate_tree(checkpoint_fd, checkpoint_device)
        try:
            os.mkdir("snapshot.lock", 0o700, dir_fd=checkpoint_fd)
        except OSError:
            fail("checkpoint snapshot lock is busy", 75)
        lock_fd = open_snapshot_lock(
            checkpoint_fd, checkpoint_device,
        )
        os.close(lock_fd)
        try:
            os.mkdir(stage_name, 0o700, dir_fd=checkpoint_fd)
        except OSError:
            fail("checkpoint staging already exists", 75)
        stage_fd = open_child_directory(
            checkpoint_fd, stage_name, checkpoint_device,
        )
        try:
            os.fchown(stage_fd, AGENT_UID, AGENT_GID)
            os.fchmod(stage_fd, 0o700)
            stage_metadata = os.fstat(stage_fd)
            if (
                (stage_metadata.st_uid, stage_metadata.st_gid)
                != (AGENT_UID, AGENT_GID)
                or stat.S_IMODE(stage_metadata.st_mode) != 0o700
            ):
                fail("checkpoint staging owner is unsafe")
        finally:
            os.close(stage_fd)
      elif action == "seize":
        validate_checkpoint_parent(checkpoint_fd)
        lock_fd = open_snapshot_lock(checkpoint_fd, checkpoint_device)
        os.close(lock_fd)
        stage_fd = open_child_directory(checkpoint_fd, stage_name, checkpoint_device)
        try:
            metadata = os.fstat(stage_fd)
            if (metadata.st_uid, metadata.st_gid) != (AGENT_UID, AGENT_GID):
                fail("checkpoint staging owner is unsafe")
            os.fchmod(stage_fd, 0)
            os.fchown(stage_fd, 0, 0)
            seize_tree(stage_fd, checkpoint_device)
            os.fchown(stage_fd, HOST_UID, HOST_GID)
            os.fchmod(stage_fd, 0o700)
        finally:
            os.close(stage_fd)
      elif action == "release":
        validate_checkpoint_parent(checkpoint_fd)
        lock_fd = open_snapshot_lock(checkpoint_fd, checkpoint_device)
        os.close(lock_fd)
        try:
            stage_fd = open_child_directory(checkpoint_fd, stage_name, checkpoint_device)
        except FileNotFoundError:
            fail("checkpoint staging disappeared before release", 75)
        try:
            delete_tree(stage_fd, checkpoint_device)
        finally:
            os.close(stage_fd)
        os.rmdir(stage_name, dir_fd=checkpoint_fd)
        os.rmdir("snapshot.lock", dir_fd=checkpoint_fd)
        validate_checkpoint_parent(checkpoint_fd)
      else:
        validate_checkpoint_parent(checkpoint_fd)
        try:
            lock_fd = open_snapshot_lock(
                checkpoint_fd, checkpoint_device,
            )
        except FileNotFoundError:
            # Abort may win before delayed prepare creates the root lock.
            pass
        else:
            os.close(lock_fd)
        delete_tree(checkpoint_fd, checkpoint_device)
        validate_checkpoint_parent(checkpoint_fd)
    finally:
        os.close(checkpoint_fd)
finally:
    os.close(operation_fd)
    os.close(runtime_fd)
'''


class DurableCheckpoint:
    """Lifecycle manager shared by custom paid harness adapters."""

    # Only this untrusted staging root is visible to the model container.  The
    # published checkpoint lives in the host-only sibling of ``logs_dir``.
    REMOTE_STAGING_DIR = PurePosixPath(
        "/logs/agent/.dradar-checkpoint-staging",
    )

    def __init__(
        self,
        *,
        logs_dir: Path,
        enabled: str | bool,
        assignment_id: str | None,
        task_id: str | None,
        model: str | None,
        effort: str | None,
        resume_generation: str | int = 0,
        checkpoint_path: str | None = None,
        harness: str,
        provider: str,
        agent_version: str,
        interval_sec: str | int = 30,
        state_paths: Iterable[StatePath] = (),
        sensitive_values: Iterable[str | bytes] = (),
        session_probe: str | None = None,
        workdir: str = "/app",
    ) -> None:
        self.enabled = _parse_bool(enabled)
        self.assignment_id = assignment_id
        self.task_id = task_id
        self.model = model
        self.effort = effort
        self.resume_generation = int(resume_generation)
        if self.resume_generation < 0:
            raise ValueError("checkpoint generation must be non-negative")
        self.previous_dir = Path(checkpoint_path) if checkpoint_path else None
        self.harness = harness
        self.provider = provider
        self.agent_version = agent_version
        self.interval_sec = max(10, min(int(interval_sec), 300))
        self.state_paths = tuple(state_paths)
        self.sensitive_values = tuple(
            raw if isinstance(raw, bytes) else raw.encode("utf-8")
            for raw in sensitive_values
            if isinstance(raw, (str, bytes)) and len(raw) >= 8
        )
        self.session_probe = session_probe
        self.workdir = workdir
        self.host_uid = os.getuid()
        self.host_gid = os.getgid()
        self.logs_dir = logs_dir
        self.trial_dir = logs_dir.parent
        self.host_dir = self.trial_dir / "checkpoint"
        self.staging_host_dir = logs_dir / ".dradar-checkpoint-staging"
        self.manifest_path: Path | None = None
        self.previous: dict[str, Any] | None = None
        self.session_id: str | None = None
        self.agent_identity: AgentIdentity | None = None
        self.runtime_dir: PurePosixPath | None = None
        self.capture_sha256: str | None = None
        self.supervisor_sha256: str | None = None
        self.base_commit: str | None = None
        self._periodic_task: asyncio.Task[None] | None = None
        self._periodic_environment: Any | None = None
        self._owner_task: asyncio.Task[Any] | None = None
        self._periodic_stop: asyncio.Event | None = None
        self._snapshot_mutex: asyncio.Lock | None = None
        self._periodic_failure: BaseException | None = None
        self.snapshot_launch_attempted = False
        self.snapshot_background_ready = False

    def _normalize_host_layout(self) -> None:
        """Privatize Pier's host-owned trial and agent bind directories.

        Public Pier deliberately changes the mounted agent directory to 0777
        before launching an agent so arbitrary image users can write logs.
        Durable checkpoints instead select the DRadar host UID numerically, so
        the two host-owned directories can and must be returned to 0700 before
        any untrusted model process starts.
        """

        descriptors: list[int] = []
        try:
            for path in (self.trial_dir, self.logs_dir):
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                try:
                    descriptor = os.open(path, flags)
                except OSError as exc:
                    raise CheckpointError(
                        "checkpoint host layout is unreadable",
                    ) from exc
                descriptors.append(descriptor)
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != self.host_uid
                    or metadata.st_gid != self.host_gid
                ):
                    raise CheckpointError(
                        "checkpoint host layout is not owned by DRadar",
                    )
                os.fchmod(descriptor, 0o700)
            trial_metadata = os.fstat(descriptors[0])
            logs_metadata = os.fstat(descriptors[1])
            if trial_metadata.st_dev != logs_metadata.st_dev:
                raise CheckpointError(
                    "checkpoint host layout crosses a filesystem",
                )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def _validate_host_layout(self) -> None:
        """Verify that publication is outside the container's agent bind."""

        if self.logs_dir.name != "agent":
            raise CheckpointError("checkpoint logs directory is not the Pier agent dir")
        try:
            logs_metadata = self.logs_dir.lstat()
            trial_metadata = self.trial_dir.lstat()
        except OSError as exc:
            raise CheckpointError("checkpoint host layout is unreadable") from exc
        if (
            not stat.S_ISDIR(logs_metadata.st_mode)
            or not stat.S_ISDIR(trial_metadata.st_mode)
            or self.logs_dir.is_symlink()
            or self.trial_dir.is_symlink()
        ):
            raise CheckpointError("checkpoint host layout crosses a symlink")
        if logs_metadata.st_dev != trial_metadata.st_dev:
            raise CheckpointError("checkpoint host layout crosses a filesystem")
        if (
            logs_metadata.st_uid != self.host_uid
            or logs_metadata.st_gid != self.host_gid
            or stat.S_IMODE(logs_metadata.st_mode) & 0o077
        ):
            raise CheckpointError("checkpoint logs directory is not host-private")
        if (
            trial_metadata.st_uid != self.host_uid
            or trial_metadata.st_gid != self.host_gid
            or stat.S_IMODE(trial_metadata.st_mode) & 0o022
        ):
            raise CheckpointError("checkpoint trial directory is not host-private")
        expected = self.logs_dir.parent / "checkpoint"
        if self.host_dir != expected or self.host_dir == self.logs_dir:
            raise CheckpointError("checkpoint publication path is unsafe")
        if _lexists(self.host_dir):
            _validate_regular_tree(self.host_dir, label="checkpoint publication root")
            metadata = self.host_dir.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != self.host_uid
                or metadata.st_gid != self.host_gid
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise CheckpointError("checkpoint publication root is not host-private")

    def _validate_previous_host_layout(self) -> None:
        """Accept only a host-private source, never the old agent bind."""

        if self.previous_dir is None:
            return
        if self.previous_dir.parent.name == "agent":
            raise CheckpointError(
                "legacy agent-mounted checkpoint must be migrated before restore",
            )
        try:
            root_metadata = self.previous_dir.lstat()
            parent_metadata = self.previous_dir.parent.lstat()
        except OSError as exc:
            raise CheckpointError("previous checkpoint host layout is unreadable") from exc
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or self.previous_dir.is_symlink()
            or self.previous_dir.parent.is_symlink()
            or root_metadata.st_dev != parent_metadata.st_dev
            or root_metadata.st_uid != self.host_uid
            or root_metadata.st_gid != self.host_gid
            or parent_metadata.st_uid != self.host_uid
            or parent_metadata.st_gid != self.host_gid
            or stat.S_IMODE(root_metadata.st_mode) & 0o077
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise CheckpointError(
                "previous checkpoint is not in host-private storage",
            )

    def prepare_host_layout(self) -> None:
        """Normalize and verify the host layout before trusted local writes.

        Some adapters must publish fixed host-authored inputs through
        ``AgentLogStore`` before :meth:`start` seals ``/logs/agent`` for the
        untrusted model process.  Keep that early write on the exact same
        no-follow, numeric-owner, and private-mode boundary as checkpoint
        publication.  Ordinary non-checkpoint Pier runs retain their existing
        compatibility permissions and remain subject to ``AgentLogStore``'s
        own fail-closed validation.
        """

        if not self.enabled:
            return
        self._normalize_host_layout()
        self._validate_host_layout()

    def _event(self, event: str, **detail: Any) -> None:
        if self.manifest_path is None:
            return
        payload = {"at": _utc_now(), "event": event, "detail": detail}
        if _contains_sensitive_key(payload):
            raise CheckpointError("checkpoint event contains a sensitive field name")
        path = self.host_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _update(self, **changes: Any) -> None:
        if self.manifest_path is None:
            return
        value = _load_manifest(self.manifest_path)
        value.update(changes)
        value["updated_at"] = _utc_now()
        _write_manifest(self.manifest_path, value)

    async def _base_commit(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> str:
        result = await agent.exec_as_agent(
            environment,
            command=f"git -C {shlex.quote(self.workdir)} rev-parse HEAD",
            env=env,
        )
        value = (result.stdout or "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise CheckpointIncompatibleError("task worktree has no stable base commit")
        return value

    async def _exec_root(
        self,
        environment: Any,
        command: str,
        *,
        timeout_sec: int = 120,
    ) -> Any:
        """Run one fixed maintenance command without provider/agent env state."""

        clean_command = (
            "/usr/bin/env -i "
            "PATH=/usr/sbin:/usr/bin:/sbin:/bin "
            "HOME=/root LANG=C LC_ALL=C BASH_ENV=/dev/null ENV=/dev/null "
            "CDPATH= GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null "
            "/bin/bash --noprofile --norc -c "
            + shlex.quote(command)
        )
        result = await environment.exec(
            command=clean_command,
            user="root",
            env=dict(_ROOT_EXEC_ENV),
            cwd="/",
            timeout_sec=timeout_sec,
        )
        return_code = getattr(result, "return_code", None)
        if return_code != 0:
            raise CheckpointError(
                f"checkpoint root maintenance failed with exit {return_code!r}",
            )
        return result

    async def exec_root_maintenance(
        self,
        environment: Any,
        command: str,
        *,
        timeout_sec: int = 120,
    ) -> Any:
        """Run fixed adapter maintenance without task/provider environment."""

        return await self._exec_root(
            environment, command, timeout_sec=timeout_sec,
        )

    async def return_runtime_tree_to_host_owner(
        self,
        environment: Any,
        remote_path: str,
    ) -> None:
        """Return one agent runtime subtree to the captured host identity.

        Some task images run as root when checkpointing is disabled, while
        checkpoint-enabled runs deliberately make ``/logs/agent`` a
        root-owned sticky directory.  Therefore the current owner of the bind
        mount is not a trustworthy proxy for the host process identity.  Use
        the UID/GID captured when this manager was created and never follow
        model-controlled symlinks during the handoff.
        """

        candidate = PurePosixPath(remote_path)
        logs_root = self.REMOTE_STAGING_DIR.parent
        if (
            not candidate.is_absolute()
            or ".." in candidate.parts
            or candidate == logs_root
            or not candidate.is_relative_to(logs_root)
            or candidate == self.REMOTE_STAGING_DIR
            or candidate.is_relative_to(self.REMOTE_STAGING_DIR)
        ):
            raise CheckpointError("runtime ownership handoff path is unsafe")
        quoted_path = shlex.quote(candidate.as_posix())
        owner = f"{self.host_uid}:{self.host_gid}"
        await self._exec_root(
            environment,
            command=(
                "set -eu; "
                f"test -d {quoted_path}; test ! -L {quoted_path}; "
                f"/usr/bin/find -P {quoted_path} -xdev "
                f"-exec /usr/bin/chown -h -- {owner} {{}} +; "
                f"/usr/bin/chown -h -- {owner} {quoted_path}"
            ),
        )

    async def _probe_agent_identity(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> tuple[int, int, tuple[int, ...]]:
        result = await agent.exec_as_agent(
            environment,
            command=(
                "set -eu; "
                "printf 'uid=%s\\n' \"$(id -u)\"; "
                "printf 'gid=%s\\n' \"$(id -g)\"; "
                "printf 'groups=%s\\n' \"$(id -G)\""
            ),
            env=env,
        )
        fields: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            name, separator, value = line.partition("=")
            if separator and name in {"uid", "gid", "groups"}:
                fields[name] = value.strip()
        try:
            uid = int(fields["uid"])
            gid = int(fields["gid"])
            raw_groups = tuple(int(value) for value in fields["groups"].split())
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError("Pier agent numeric identity is unavailable") from exc
        groups = tuple(dict.fromkeys(raw_groups))
        return uid, gid, groups

    async def _agent_identity(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> AgentIdentity:
        uid, gid, groups = await self._probe_agent_identity(
            agent, environment, env,
        )
        try:
            return AgentIdentity(uid=uid, gid=gid, groups=groups)
        except ValueError as exc:
            raise CheckpointError("Pier agent numeric identity is invalid") from exc

    async def prepare_agent_environment(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> AgentIdentity:
        """Establish a numeric non-root model/maintenance trust boundary.

        Many benchmark images default to root when ``[agent].user`` is absent.
        The host-private checkpoint supervisor is meaningful only when model
        commands cannot rewrite root-owned runtime helpers.  On Linux Pier the
        logs and subscription mounts belong to the non-root DRadar host user,
        so run the model with that same numeric UID/GID.  This also preserves
        shared OAuth file ownership without assuming a username such as
        ``aloha``.
        """

        remote_home = "/tmp/dradar-agent-home"
        env.setdefault("HOME", remote_home)
        if self.host_uid == 0 or self.host_gid == 0:
            raise CheckpointError(
                "checkpoint execution requires a non-root DRadar host user",
            )
        if self.agent_identity is not None:
            prepared = await self._agent_identity(agent, environment, env)
            if prepared != self.agent_identity:
                raise CheckpointError("Pier agent numeric identity changed")
            return prepared
        await self._probe_agent_identity(agent, environment, env)
        workdir = PurePosixPath(self.workdir)
        if not workdir.is_absolute() or ".." in workdir.parts:
            raise CheckpointError("checkpoint workdir is unsafe")
        numeric_user = f"{self.host_uid}:{self.host_gid}"
        quoted_workdir = shlex.quote(workdir.as_posix())
        quoted_home = shlex.quote(remote_home)
        await self._exec_root(
            environment,
            command=(
                "set -eu; "
                f"test -d {quoted_workdir}; test ! -L {quoted_workdir}; "
                "test -d /logs/agent; test ! -L /logs/agent; "
                f"mkdir -p {quoted_home}; "
                f"chown {numeric_user} {quoted_home}; chmod 700 {quoted_home}; "
                f"find -P {quoted_workdir} -xdev -exec "
                f"chown -h -- {numeric_user} {{}} +; "
                f"chown {numeric_user} /logs/agent; chmod 700 /logs/agent"
            ),
        )
        try:
            environment.default_user = numeric_user
        except (AttributeError, TypeError) as exc:
            raise CheckpointError(
                "Pier environment cannot select a non-root agent user",
            ) from exc
        env.setdefault("HOME", remote_home)
        prepared = await self._agent_identity(agent, environment, env)
        if prepared.uid != self.host_uid or prepared.gid != self.host_gid:
            raise CheckpointError("Pier failed to enter the non-root agent identity")
        probe = await agent.exec_as_agent(
            environment,
            command=(
                f"test -r {quoted_workdir} && test -w {quoted_workdir} "
                "&& test -w /logs/agent"
            ),
            env=env,
        )
        if getattr(probe, "return_code", None) != 0:
            raise CheckpointError("non-root Pier agent cannot access its work paths")
        self.agent_identity = prepared
        return prepared

    async def _make_restored_path_agent_owned(
        self,
        environment: Any,
        remote_path: str,
        *,
        recursive: bool,
    ) -> None:
        """Return a root-created upload to the numeric container agent.

        Pier's Docker upload helpers do not preserve the source owner.  Keep
        the operation inside the task container and use the identity sampled
        from the running image instead of assuming a username or UID.
        """

        identity = self.agent_identity
        if identity is None:
            raise CheckpointError("Pier agent numeric identity is unavailable")
        quoted_path = shlex.quote(remote_path)
        owner = f"{identity.uid}:{identity.gid}"
        if recursive:
            command = (
                "set -eu; "
                f"test ! -L {quoted_path}; "
                f"chown -R -h -- {owner} {quoted_path}; "
                f"find -P {quoted_path} -type d -exec chmod 700 -- {{}} +; "
                f"find -P {quoted_path} -type f -exec chmod 600 -- {{}} +"
            )
        else:
            command = (
                "set -eu; "
                f"test -f {quoted_path}; test ! -L {quoted_path}; "
                f"chown -h -- {owner} {quoted_path}; "
                f"chmod 600 -- {quoted_path}"
            )
        await self._exec_root(environment, command)

    def _validate_previous(self, value: dict[str, Any], base_commit: str) -> None:
        expected = {
            "assignment_id": self.assignment_id,
            "task_id": self.task_id,
            "model": self.model,
            "effort": self.effort,
            "harness": self.harness,
            "provider": self.provider,
            "agent_version": self.agent_version,
            "base_commit": base_commit,
        }
        mismatched = [
            name for name, wanted in expected.items() if value.get(name) != wanted
        ]
        if mismatched:
            raise CheckpointIncompatibleError(
                "checkpoint runtime identity mismatch: " + ", ".join(mismatched)
            )

    async def _restore_workspace(
        self, agent: Any, environment: Any, env: dict[str, str], previous_dir: Path,
    ) -> None:
        previous_dir = _snapshot_payload_dir(previous_dir)
        remote_restore = "/tmp/dradar-checkpoint-restore"
        await agent.exec_as_agent(
            environment,
            command=f"rm -rf {remote_restore} && mkdir -p {remote_restore}",
            env=env,
        )
        patch = _safe_path(previous_dir, "workspace.patch")
        if patch.is_file() and patch.stat().st_size:
            remote_patch = f"{remote_restore}/workspace.patch"
            await environment.upload_file(patch, remote_patch)
            await self._make_restored_path_agent_owned(
                environment, remote_patch, recursive=False,
            )
            await agent.exec_as_agent(
                environment,
                command=(
                    f"git -C {shlex.quote(self.workdir)} apply --binary {remote_patch}"
                ),
                env=env,
            )
        archive = _safe_path(previous_dir, "untracked.tar.gz")
        if archive.is_file() and archive.stat().st_size:
            if _validate_archive(archive, self.sensitive_values):
                raise CheckpointError("checkpoint archive contains a credential")
            remote_archive = f"{remote_restore}/untracked.tar.gz"
            await environment.upload_file(archive, remote_archive)
            await self._make_restored_path_agent_owned(
                environment, remote_archive, recursive=False,
            )
            await agent.exec_as_agent(
                environment,
                command=(
                    f"tar -xzf {remote_archive} -C {shlex.quote(self.workdir)}"
                ),
                env=env,
            )

    async def _restore_state(
        self, agent: Any, environment: Any, env: dict[str, str], previous_dir: Path,
    ) -> None:
        previous_dir = _snapshot_payload_dir(previous_dir)
        state_root = _safe_path(previous_dir, "provider-state")
        if not state_root.is_dir():
            return
        for item in self.state_paths:
            source = _safe_path(state_root, item.name)
            if not source.exists():
                continue
            if _path_contains_any(source, self.sensitive_values):
                raise CheckpointError("checkpoint provider state contains a credential")
            remote = item.remote_path
            await agent.exec_as_agent(
                environment,
                command=(
                    f"rm -rf {shlex.quote(remote)} && mkdir -p "
                    f"{shlex.quote(str(PurePosixPath(remote).parent))}"
                ),
                env=env,
            )
            if source.is_dir():
                await environment.upload_dir(source, remote)
                await self._make_restored_path_agent_owned(
                    environment, remote, recursive=True,
                )
            elif source.is_file():
                await environment.upload_file(source, remote)
                await self._make_restored_path_agent_owned(
                    environment, remote, recursive=False,
                )

    def _previous_session_id(self, previous_dir: Path) -> str | None:
        previous_dir = _snapshot_payload_dir(previous_dir)
        # Provider-state omission deliberately downgrades recovery to the
        # workspace only.  A bare native session id without its matching
        # provider files is not a valid resume contract.
        if self._marker_present(previous_dir, "session-omitted-sensitive"):
            return None
        candidates = [self.previous.get("session_id") if self.previous else None]
        path = _safe_path(previous_dir, "session-id")
        if path.is_file() and path.stat().st_size <= 512:
            candidates.insert(0, path.read_text(encoding="utf-8", errors="replace").strip())
        for candidate in candidates:
            if isinstance(candidate, str) and _SESSION_ID_RE.fullmatch(candidate):
                return candidate
        return None

    def _remove_sensitive_artifacts(self) -> bool:
        """Revalidate the immutable published generation and mark it sticky."""

        payload_dir = _snapshot_payload_dir(self.host_dir)
        archive = payload_dir / "untracked.tar.gz"
        rejected = (
            self._marker_present(payload_dir, "invalid-secret")
            or _path_contains_any(payload_dir, self.sensitive_values)
            or (
                archive.is_file()
                and _validate_archive(archive, self.sensitive_values)
            )
        )
        if rejected:
            self._mark_invalid_secret(
                "credential-shaped content detected in published checkpoint",
            )
        return rejected

    def _discard_untrusted_artifacts(self) -> None:
        """Remove host-only temporary copies and reject residual staging."""

        snapshots = self.host_dir / "snapshots"
        if _lexists(snapshots):
            _validate_regular_tree(snapshots, label="checkpoint snapshots cleanup")
            if not snapshots.is_dir():
                raise CheckpointError("checkpoint snapshots cleanup root is unsafe")
            for candidate in snapshots.iterdir():
                if not candidate.name.startswith(".copy-"):
                    continue
                _validate_regular_tree(
                    candidate, label="checkpoint temporary generation",
                )
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
        pointer_temporary = self.host_dir / ".current-generation.tmp"
        if _lexists(pointer_temporary):
            _validate_regular_tree(
                pointer_temporary, label="checkpoint pointer temporary",
            )
            if not pointer_temporary.is_file():
                raise CheckpointError("checkpoint pointer temporary is unsafe")
            pointer_temporary.unlink()
        if _lexists(self.staging_host_dir):
            _validate_regular_tree(
                self.staging_host_dir, label="checkpoint staging cleanup",
            )
            if not self.staging_host_dir.is_dir():
                raise CheckpointError("checkpoint staging cleanup root is unsafe")
            if any(self.staging_host_dir.iterdir()):
                raise CheckpointError(
                    "checkpoint staging cleanup left untrusted artifacts",
                )

    def _invalidate_snapshot(self, reason: str) -> None:
        cleanup_error: BaseException | None = None
        try:
            self._discard_untrusted_artifacts()
        except BaseException as exc:
            cleanup_error = exc
        marker_error: BaseException | None = None
        try:
            self._mark_invalid_snapshot(reason)
        except BaseException as exc:
            marker_error = exc
        if cleanup_error is not None or marker_error is not None:
            raise CheckpointError(
                "checkpoint invalidation left an unsafe residual state",
            ) from (cleanup_error or marker_error)

    @staticmethod
    def _marker_present(root: Path, name: str) -> bool:
        marker = root / name
        if not _lexists(marker):
            return False
        _validate_regular_tree(marker, label=f"checkpoint {name} marker")
        metadata = marker.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CheckpointError(f"checkpoint {name} marker is unsafe")
        return True

    def _snapshot_lock_present(self) -> bool:
        lock = self.host_dir / "snapshot.lock"
        if not _lexists(lock):
            return False
        _validate_regular_tree(lock, label="checkpoint snapshot lock")
        metadata = lock.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise CheckpointError("checkpoint snapshot lock is unsafe")
        return True

    def _create_snapshot_lock(self) -> None:
        if self._snapshot_lock_present():
            raise CheckpointError("checkpoint snapshot lock is busy")
        try:
            (self.host_dir / "snapshot.lock").mkdir(mode=0o700)
        except OSError as exc:
            raise CheckpointError("checkpoint snapshot lock could not be created") from exc

    def _release_snapshot_lock(self) -> None:
        lock = self.host_dir / "snapshot.lock"
        if not self._snapshot_lock_present():
            raise CheckpointError("checkpoint snapshot lock disappeared")
        metadata = lock.lstat()
        if (
            metadata.st_uid != self.host_uid
            or metadata.st_gid != self.host_gid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise CheckpointError("checkpoint snapshot lock ownership is unsafe")
        lock.rmdir()

    def _mark_invalid_secret(self, reason: str) -> None:
        """Persist a host-only sticky marker that no generation can clear."""

        marker = self.host_dir / "invalid-secret"
        if _lexists(marker):
            if not self._marker_present(self.host_dir, "invalid-secret"):
                raise CheckpointError("checkpoint invalid-secret marker is unsafe")
            return
        marker.write_text(reason + "\n", encoding="utf-8")
        marker.chmod(0o600)

    def _mark_invalid_snapshot(self, reason: str) -> None:
        """Create the fail-closed marker without overwriting an unreadable one."""

        marker = self.host_dir / "invalid-snapshot"
        if _lexists(marker):
            if not self._marker_present(self.host_dir, "invalid-snapshot"):
                raise CheckpointError("checkpoint invalid marker is unsafe")
            return
        marker.write_text(reason + "\n", encoding="utf-8")
        marker.chmod(0o600)

    def _verify_host_ownership(self) -> None:
        """Verify the container handed every private checkpoint entry back."""

        _validate_regular_tree(self.host_dir, label="checkpoint handoff")
        candidates = [self.host_dir]
        def fail_walk(error: OSError) -> None:
            raise error

        try:
            for root, directories, files in os.walk(
                self.host_dir,
                followlinks=False,
                onerror=fail_walk,
            ):
                base = Path(root)
                candidates.extend(base / name for name in directories)
                candidates.extend(base / name for name in files)
        except OSError as exc:
            raise CheckpointError("checkpoint ownership tree is unreadable") from exc
        for candidate in candidates:
            metadata = candidate.lstat()
            if not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise CheckpointError("checkpoint handoff contains a special file")
            if metadata.st_uid != self.host_uid or metadata.st_gid != self.host_gid:
                raise CheckpointError("checkpoint ownership handoff is incomplete")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode & 0o077:
                raise CheckpointError("checkpoint handoff is not private")
            if stat.S_ISDIR(metadata.st_mode) and mode & 0o700 != 0o700:
                raise CheckpointError("checkpoint directory is not owner-accessible")
            if stat.S_ISREG(metadata.st_mode) and mode & 0o600 != 0o600:
                raise CheckpointError("checkpoint file is not owner-accessible")

    async def _install_runtime(
        self,
        agent: Any,
        environment: Any,
        env: dict[str, str],
        checkpoint_id: str,
    ) -> None:
        """Install capture helpers; their output remains untrusted until copied."""

        del agent, env
        if self.agent_identity is None:
            raise CheckpointError("Pier agent identity was not captured")
        runtime = PurePosixPath("/run/dradar-checkpoint") / checkpoint_id
        capture_text = _capture_script(
            workdir=self.workdir,
            state_paths=self.state_paths,
            session_probe=self.session_probe,
            agent_identity=self.agent_identity,
        )
        capture_sha = hashlib.sha256(capture_text.encode("utf-8")).hexdigest()
        supervisor_text = _supervisor_script(
            checkpoint_dir=str(self.REMOTE_STAGING_DIR),
            runtime_dir=str(runtime),
            capture_sha256=capture_sha,
            agent_identity=self.agent_identity,
            host_uid=self.host_uid,
            host_gid=self.host_gid,
        )
        supervisor_sha = hashlib.sha256(
            supervisor_text.encode("utf-8"),
        ).hexdigest()
        capture_upload = f"/tmp/dradar-checkpoint-{checkpoint_id}-capture.upload"
        supervisor_upload = (
            f"/tmp/dradar-checkpoint-{checkpoint_id}-supervisor.upload"
        )
        staging = shlex.quote(str(self.REMOTE_STAGING_DIR))
        agent_logs = shlex.quote(str(self.REMOTE_STAGING_DIR.parent))
        await self._exec_root(
            environment,
            command=(
                "set -eu; umask 077; "
                "test -x /usr/bin/python3; "
                "test -x /usr/bin/timeout; "
                "test \"$(/usr/bin/stat -c '%u' /usr /usr/bin)\" = "
                "\"0\n0\"; "
                "runtime_root=/run/dradar-checkpoint; "
                "if [ -e \"$runtime_root\" ] || [ -L \"$runtime_root\" ]; then "
                "[ -d \"$runtime_root\" ] && [ ! -L \"$runtime_root\" ]; "
                "else /usr/bin/install -d -o 0 -g 0 -m 0711 \"$runtime_root\"; fi; "
                "/usr/bin/chown 0:0 \"$runtime_root\"; "
                "/usr/bin/chmod 0711 \"$runtime_root\"; "
                f"/usr/bin/rm -rf {shlex.quote(str(runtime))}; "
                f"/usr/bin/install -d -o 0 -g 0 -m 0711 "
                f"{shlex.quote(str(runtime))}; "
                # The sticky, root-owned logs directory remains writable to
                # the numeric agent group for ordinary output, but that agent
                # cannot rename/unlink the root-owned staging entry itself.
                f"/usr/bin/chown 0:{self.agent_identity.gid} {agent_logs}; "
                f"/usr/bin/chmod 1770 {agent_logs}; "
                f"/usr/bin/rm -rf {staging}; "
                f"/usr/bin/install -d -o 0 -g {self.agent_identity.gid} "
                f"-m 0750 {staging}; "
                f"test \"$(/usr/bin/stat -c '%u:%g:%a' {agent_logs})\" = "
                f"0:{self.agent_identity.gid}:1770; "
                f"test \"$(/usr/bin/stat -c '%u:%g:%a' {staging})\" = "
                f"0:{self.agent_identity.gid}:750; "
                f"/usr/bin/rm -f {shlex.quote(capture_upload)} "
                f"{shlex.quote(supervisor_upload)}"
            ),
        )
        with tempfile.TemporaryDirectory(prefix="dradar-checkpoint-runtime-") as tmp:
            capture_local = Path(tmp) / "capture"
            supervisor_local = Path(tmp) / "supervisor"
            capture_local.write_text(capture_text, encoding="utf-8")
            supervisor_local.write_text(supervisor_text, encoding="utf-8")
            capture_local.chmod(0o600)
            supervisor_local.chmod(0o600)
            await environment.upload_file(capture_local, capture_upload)
            await environment.upload_file(supervisor_local, supervisor_upload)
        await self._exec_root(
            environment,
            command=(
                "set -eu; umask 077; "
                f"test -f {shlex.quote(capture_upload)} "
                f"&& test ! -L {shlex.quote(capture_upload)}; "
                f"test -f {shlex.quote(supervisor_upload)} "
                f"&& test ! -L {shlex.quote(supervisor_upload)}; "
                f"test \"$(/usr/bin/sha256sum {shlex.quote(capture_upload)} | "
                f"/usr/bin/awk '{{print $1}}')\" = {capture_sha}; "
                f"test \"$(/usr/bin/sha256sum {shlex.quote(supervisor_upload)} | "
                f"/usr/bin/awk '{{print $1}}')\" = {supervisor_sha}; "
                f"/usr/bin/install -o 0 -g 0 -m 0555 "
                f"{shlex.quote(capture_upload)} "
                f"{shlex.quote(str(runtime / 'capture'))}; "
                f"/usr/bin/install -o 0 -g 0 -m 0700 "
                f"{shlex.quote(supervisor_upload)} "
                f"{shlex.quote(str(runtime / 'supervisor'))}; "
                f"/usr/bin/rm -f {shlex.quote(capture_upload)} "
                f"{shlex.quote(supervisor_upload)}; "
                f"test \"$(/usr/bin/stat -c '%u:%g:%a' "
                f"{shlex.quote(str(runtime / 'capture'))})\" = 0:0:555; "
                f"test \"$(/usr/bin/stat -c '%u:%g:%a' "
                f"{shlex.quote(str(runtime / 'supervisor'))})\" = 0:0:700; "
                f"test \"$(/usr/bin/sha256sum "
                f"{shlex.quote(str(runtime / 'capture'))} | "
                f"/usr/bin/awk '{{print $1}}')\" = {capture_sha}; "
                f"test \"$(/usr/bin/sha256sum "
                f"{shlex.quote(str(runtime / 'supervisor'))} | "
                f"/usr/bin/awk '{{print $1}}')\" = {supervisor_sha}"
            ),
        )
        self.runtime_dir = runtime
        self.capture_sha256 = capture_sha
        self.supervisor_sha256 = supervisor_sha

    def _supervisor_command(self, action: str, generation: str) -> str:
        if (
            self.runtime_dir is None
            or self.supervisor_sha256 is None
            or action not in {"prepare", "seize", "release", "abort"}
            or _ID_RE.fullmatch(generation) is None
        ):
            raise CheckpointError("checkpoint supervisor is not initialized")
        supervisor = self.runtime_dir / "supervisor"
        quoted = shlex.quote(str(supervisor))
        return (
            "set -eu; umask 077; "
            "test -x /usr/bin/python3; "
            f"test -f {quoted} && test ! -L {quoted}; "
            f"test \"$(/usr/bin/stat -c '%u:%g:%a' {quoted})\" = 0:0:700; "
            f"test \"$(/usr/bin/sha256sum {quoted} | "
            f"/usr/bin/awk '{{print $1}}')\" = "
            f"{self.supervisor_sha256}; "
            "/usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin "
            "HOME=/root LANG=C LC_ALL=C "
            f"/usr/bin/python3 -I -S {quoted} {action} "
            f"{shlex.quote(generation)}"
        )

    def _stage_path(self, generation: str) -> Path:
        if _ID_RE.fullmatch(generation) is None:
            raise CheckpointError("checkpoint generation is invalid")
        return self.staging_host_dir / f".snapshot-stage-{generation}"

    def _validate_staged_snapshot(self, stage: Path) -> None:
        _validate_regular_tree(stage, label="checkpoint staging")
        for name in ("progress-summary.txt", "last_heartbeat"):
            candidate = _safe_path(stage, name)
            if not candidate.is_file():
                raise CheckpointError(f"checkpoint staging is missing {name}")
        if self._marker_present(stage, "invalid-secret"):
            self._mark_invalid_secret(
                "credential-shaped content detected in checkpoint staging",
            )
            raise CheckpointError("checkpoint contains rejected credential data")
        if self._marker_present(stage, "session-omitted-sensitive"):
            # This marker means the capture program already removed native
            # provider state before publication.  The workspace artifacts are
            # still a safe checkpoint, but the marker must never coexist with
            # the supposedly omitted directory.
            provider_state = _safe_path(stage, "provider-state")
            if _lexists(provider_state):
                self._mark_invalid_snapshot(
                    "checkpoint provider state conflicts with omission marker",
                )
                raise CheckpointError(
                    "checkpoint provider-state omission is inconsistent",
                )
        tracked_scan = _safe_path(stage, _TRACKED_SCAN_ARTIFACT)
        if not tracked_scan.is_file():
            raise CheckpointError(
                "checkpoint staging is missing the tracked-worktree scan",
            )
        archive = _safe_path(stage, "untracked.tar.gz")
        archive_rejected = archive.is_file() and _validate_archive(
            archive, self.sensitive_values,
        )
        tree_rejected = _path_contains_any(stage, self.sensitive_values)
        if archive_rejected or tree_rejected:
            self._mark_invalid_secret(
                "credential-shaped content detected in checkpoint staging",
            )
            raise CheckpointError("checkpoint contains rejected credential data")

    def _promote_stage(
        self, generation: str, *, session_id: str | None = None,
    ) -> None:
        """Copy untrusted staging into a host-only generation, then publish it."""

        stage = self._stage_path(generation)
        snapshots = self.host_dir / "snapshots"
        if _lexists(snapshots):
            _validate_regular_tree(snapshots, label="checkpoint snapshots")
            if not snapshots.is_dir():
                raise CheckpointError("checkpoint snapshots path is not a directory")
        else:
            snapshots.mkdir(mode=0o700)
        snapshots.chmod(0o700)
        destination = snapshots / generation
        if _lexists(destination):
            raise CheckpointError("checkpoint generation already exists")
        copied = snapshots / f".copy-{generation}"
        if _lexists(copied):
            raise CheckpointError("checkpoint generation copy already exists")
        try:
            _copy_seized_tree(stage, copied)
            native_state_omitted = self._marker_present(
                copied, "session-omitted-sensitive",
            )
            sidecar = copied / "session-id"
            if native_state_omitted:
                if _lexists(sidecar):
                    _validate_regular_tree(
                        sidecar, label="checkpoint staged session id",
                    )
                    if not sidecar.is_file():
                        raise CheckpointError("checkpoint staged session id is unsafe")
                    sidecar.unlink()
            elif (
                isinstance(session_id, str)
                and _SESSION_ID_RE.fullmatch(session_id)
            ):
                if _lexists(sidecar):
                    _validate_regular_tree(
                        sidecar, label="checkpoint staged session id",
                    )
                    if not sidecar.is_file():
                        raise CheckpointError("checkpoint staged session id is unsafe")
                temporary_session = copied / ".session-id.tmp"
                if _lexists(temporary_session):
                    raise CheckpointError(
                        "checkpoint staged session temporary is unsafe",
                    )
                temporary_session.write_text(session_id + "\n", encoding="utf-8")
                temporary_session.chmod(0o600)
                os.replace(temporary_session, sidecar)
            self._validate_staged_snapshot(copied)
            # This file exists only to expose the unencoded tracked bytes to
            # the exact-secret scanner.  It is never a resumable artifact.
            (copied / _TRACKED_SCAN_ARTIFACT).unlink()
            os.replace(copied, destination)
        except BaseException:
            if _lexists(copied):
                _validate_regular_tree(copied, label="checkpoint failed generation copy")
                shutil.rmtree(copied)
            raise
        snapshots_fd = os.open(
            snapshots,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(snapshots_fd)
        finally:
            os.close(snapshots_fd)
        pointer = self.host_dir / "current-generation"
        if _lexists(pointer):
            _validate_regular_tree(pointer, label="checkpoint generation pointer")
            if not pointer.is_file():
                raise CheckpointError("checkpoint generation pointer is unsafe")
        temporary = self.host_dir / ".current-generation.tmp"
        if _lexists(temporary):
            _validate_regular_tree(temporary, label="checkpoint generation temporary")
            if not temporary.is_file():
                raise CheckpointError("checkpoint generation temporary is unsafe")
            temporary.unlink()
        temporary.write_text(generation + "\n", encoding="ascii")
        temporary.chmod(0o600)
        os.replace(temporary, pointer)
        directory_fd = os.open(
            self.host_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        for candidate in snapshots.iterdir():
            if candidate.name in {generation, f".copy-{generation}"}:
                continue
            _validate_regular_tree(candidate, label="old checkpoint generation")
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()

    async def _snapshot_once(
        self,
        agent: Any,
        environment: Any,
        env: dict[str, str],
        *,
        session_id: str | None = None,
    ) -> None:
        if (
            self.runtime_dir is None
            or self.base_commit is None
            or self.agent_identity is None
        ):
            raise CheckpointError("checkpoint runtime is not initialized")
        generation = uuid.uuid4().hex
        self._create_snapshot_lock()
        prepare_attempted = False
        try:
            # ``environment.exec`` may be cancelled after the remote command
            # has already created its lock/stage. Treat invocation itself as a
            # possible side effect and always run the idempotent abort path.
            prepare_attempted = True
            await self._exec_root(
                environment,
                command=self._supervisor_command("prepare", generation),
            )
            capture = shlex.quote(str(self.runtime_dir / "capture"))
            stage = shlex.quote(
                str(
                    self.REMOTE_STAGING_DIR
                    / f".snapshot-stage-{generation}"
                ),
            )
            await agent.exec_as_agent(
                environment,
                command=(
                    "/usr/bin/timeout --signal=TERM "
                    f"--kill-after={_CAPTURE_KILL_GRACE_SEC}s "
                    f"{_CAPTURE_WORK_TIMEOUT_SEC}s "
                    f"/bin/sh {capture} {stage} {shlex.quote(self.base_commit)}"
                ),
                env=env,
                timeout_sec=_CAPTURE_EXEC_TIMEOUT_SEC,
            )
            await self._exec_root(
                environment,
                command=self._supervisor_command("seize", generation),
            )
            self._promote_stage(generation, session_id=session_id)
            await self._exec_root(
                environment,
                command=self._supervisor_command("release", generation),
            )
            self._verify_host_ownership()
            # Releasing this host-only lock is the commit point. No fallible
            # validation may run after scanners can observe the generation.
            self._release_snapshot_lock()
        except BaseException as primary:
            cleanup_failure: BaseException | None = None
            if prepare_attempted:
                async def abort_staging() -> None:
                    # A cancelled remote prepare can finish just after the
                    # first abort. Re-run the idempotent descriptor cleanup
                    # until both the generation and internal lock are absent.
                    for _attempt in range(3):
                        await self._exec_root(
                            environment,
                            command=self._supervisor_command("abort", generation),
                        )
                        stage = self._stage_path(generation)
                        internal_lock = self.staging_host_dir / "snapshot.lock"
                        if not _lexists(stage) and not _lexists(internal_lock):
                            return
                        await asyncio.sleep(0)
                    raise CheckpointError(
                        "checkpoint staging remained after abort",
                    )

                cleanup_task = asyncio.create_task(
                    abort_staging(),
                    name=f"dradar-checkpoint-abort-{generation}",
                )
                try:
                    while not cleanup_task.done():
                        try:
                            await asyncio.shield(cleanup_task)
                        except asyncio.CancelledError:
                            # The host lock stays sticky, but sensitive staging
                            # still must be descriptor-cleaned before teardown.
                            continue
                    cleanup_task.result()
                except BaseException as exc:
                    cleanup_failure = exc
            if cleanup_failure is not None:
                raise CheckpointError(
                    "checkpoint staging cleanup failed; snapshot remains invalid",
                ) from cleanup_failure
            if isinstance(primary, asyncio.CancelledError):
                raise primary
            raise

    async def _periodic_loop(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> None:
        assert self._periodic_stop is not None
        assert self._snapshot_mutex is not None
        try:
            while True:
                try:
                    await asyncio.wait_for(
                        self._periodic_stop.wait(), timeout=self.interval_sec,
                    )
                    return
                except TimeoutError:
                    pass
                async with self._snapshot_mutex:
                    await self._snapshot_once(agent, environment, env)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._periodic_failure = exc
            try:
                try:
                    self._invalidate_snapshot("checkpoint periodic snapshot failed")
                except BaseException as invalidation_error:
                    self._periodic_failure = invalidation_error
                self._update(
                    phase="invalid", failure_type="CheckpointSnapshotInvalid",
                )
                self._event("invalid", failure_type=type(exc).__name__)
            finally:
                # Once checkpointing fails there is no safe way to keep a paid
                # model turn running. Cancel its owning harness task so the
                # normal harness finally path stops the model and records the
                # invalid checkpoint immediately instead of burning quota until
                # the outer watchdog.
                owner = self._owner_task
                if (
                    owner is not None
                    and owner is not asyncio.current_task()
                    and not owner.done()
                ):
                    owner.cancel("checkpoint periodic snapshot failed")

    async def _stop_periodic(self) -> None:
        task = self._periodic_task
        if task is None:
            return
        if self._periodic_stop is not None:
            self._periodic_stop.set()
        caller_cancellation: asyncio.CancelledError | None = None

        async def wait_until(deadline: float) -> bool:
            nonlocal caller_cancellation
            while not task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=remaining,
                    )
                except asyncio.CancelledError as exc:
                    # Do not return while a snapshot can still mutate staging
                    # or hold the host lock. Preserve cancellation for the
                    # caller only after the background task is fully reaped.
                    if caller_cancellation is None:
                        caller_cancellation = exc
                    continue
                except TimeoutError:
                    return task.done()
            return True

        loop = asyncio.get_running_loop()
        finished = await wait_until(loop.time() + _PERIODIC_STOP_TIMEOUT_SEC)
        forced = False
        if not finished:
            forced = True
            task.cancel()
            finished = await wait_until(loop.time() + _PERIODIC_STOP_TIMEOUT_SEC)
            if not finished:
                # An environment backend may swallow task cancellation while
                # an exec call is still attached. A stopped harness must never
                # return while that writer can later seize/promote staging or
                # mutate the host lock. Stop this task's own environment, then
                # keep re-cancelling/reaping without a final escape hatch.
                # A pathological backend may therefore delay teardown, but it
                # cannot leak a live checkpoint writer past this method.
                try:
                    self._invalidate_snapshot(
                        "checkpoint periodic snapshot cleanup required "
                        "environment teardown",
                    )
                except BaseException:
                    pass
                environment = self._periodic_environment
                stop = getattr(environment, "stop", None)
                if callable(stop):
                    teardown = asyncio.create_task(
                        stop(delete=False),
                        name="dradar-checkpoint-environment-stop",
                    )
                    while not teardown.done():
                        try:
                            await asyncio.shield(teardown)
                        except asyncio.CancelledError as exc:
                            if caller_cancellation is None:
                                caller_cancellation = exc
                            continue
                        except BaseException:
                            break
                    if not teardown.cancelled():
                        try:
                            teardown.result()
                        except BaseException:
                            pass
                while not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=_PERIODIC_STOP_TIMEOUT_SEC,
                        )
                    except asyncio.CancelledError as exc:
                        if caller_cancellation is None:
                            caller_cancellation = exc
                    except TimeoutError:
                        continue
                    except BaseException:
                        break

        self._periodic_task = None
        self._periodic_environment = None
        self._owner_task = None
        unexpected_cancel = False
        try:
            task.result()
        except asyncio.CancelledError:
            unexpected_cancel = not forced
        if caller_cancellation is not None:
            raise caller_cancellation
        if forced:
            raise CheckpointError(
                "checkpoint periodic snapshot required forced cancellation",
            )
        if unexpected_cancel:
            raise CheckpointError(
                "checkpoint periodic snapshot was cancelled unexpectedly",
            )

    async def start(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> str | None:
        if not self.enabled:
            return None
        if not isinstance(self.assignment_id, str) or not self.assignment_id:
            raise CheckpointError("checkpoint assignment id is required")
        self.prepare_host_layout()
        self._validate_previous_host_layout()
        self.agent_identity = await self.prepare_agent_environment(
            agent, environment, env,
        )
        base_commit = await self._base_commit(agent, environment, env)
        self.base_commit = base_commit
        previous = None
        if self.previous_dir is not None:
            _validate_regular_tree(
                self.previous_dir, label="previous checkpoint root",
            )
            previous_lock = self.previous_dir / "snapshot.lock"
            if _lexists(previous_lock):
                _validate_regular_tree(
                    previous_lock, label="previous checkpoint snapshot lock",
                )
                raise CheckpointError("checkpoint snapshot is incomplete")
            previous_payload = _snapshot_payload_dir(self.previous_dir)
            if self._marker_present(
                self.previous_dir, "invalid-secret",
            ) or self._marker_present(previous_payload, "invalid-secret"):
                raise CheckpointError("checkpoint contains rejected credential data")
            if self._marker_present(self.previous_dir, "invalid-snapshot"):
                raise CheckpointError("checkpoint snapshot did not finish safely")
            previous_path = _safe_path(self.previous_dir, "checkpoint.json")
            previous = _load_manifest(previous_path)
            try:
                self._validate_previous(previous, base_commit)
            except CheckpointIncompatibleError as exc:
                if _lexists(self.host_dir):
                    _validate_regular_tree(
                        self.host_dir, label="checkpoint output root",
                    )
                    shutil.rmtree(self.host_dir)
                self.host_dir.mkdir(parents=True, mode=0o700)
                self.manifest_path = self.host_dir / "checkpoint.json"
                now = _utc_now()
                incompatible = {
                    "schema_version": SCHEMA_VERSION,
                    "checkpoint_id": previous["checkpoint_id"],
                    "assignment_id": self.assignment_id,
                    "phase": "incompatible",
                    "created_at": previous.get("created_at", now),
                    "updated_at": now,
                    "last_heartbeat": now,
                    "task_id": self.task_id,
                    "model": self.model,
                    "effort": self.effort,
                    "harness": self.harness,
                    "provider": self.provider,
                    "agent_version": self.agent_version,
                    "base_commit": base_commit,
                    "resume_generation": self.resume_generation,
                    "resume_count": int(previous.get("resume_count", 0)),
                    "failure_type": type(exc).__name__,
                    **_FIXED_ARTIFACTS,
                }
                _write_manifest(self.manifest_path, incompatible)
                self._event(
                    "restore_rejected", failure_type=type(exc).__name__,
                )
                raise

        if _lexists(self.host_dir):
            _validate_regular_tree(self.host_dir, label="checkpoint output root")
            shutil.rmtree(self.host_dir)
        self.host_dir.mkdir(parents=True, mode=0o700)
        self.host_dir.chmod(0o700)
        self.manifest_path = self.host_dir / "checkpoint.json"
        # Historical event streams are diagnostics, not resume state.  Older
        # checkpoint writers allowed arbitrary detail values under benign JSON
        # keys, so carrying the bytes forward could republish an opaque
        # credential.  Start a fresh trusted event stream on every resume.
        now = _utc_now()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": (
                previous["checkpoint_id"] if previous is not None else uuid.uuid4().hex
            ),
            "assignment_id": self.assignment_id,
            "phase": "running",
            "created_at": previous.get("created_at", now) if previous else now,
            "updated_at": now,
            "last_heartbeat": now,
            "task_id": self.task_id,
            "model": self.model,
            "effort": self.effort,
            "harness": self.harness,
            "provider": self.provider,
            "agent_version": self.agent_version,
            "base_commit": base_commit,
            "resume_generation": self.resume_generation,
            "resume_count": int(previous.get("resume_count", 0)) + 1 if previous else 0,
            **_FIXED_ARTIFACTS,
        }
        _write_manifest(self.manifest_path, manifest)
        base_commit_path = self.host_dir / "base_commit"
        base_commit_path.write_text(base_commit + "\n", encoding="utf-8")
        base_commit_path.chmod(0o600)
        self.previous = previous
        self._event("checkpoint_started", resumed=previous is not None)
        if previous is not None and self.previous_dir is not None:
            try:
                await self._restore_workspace(
                    agent, environment, env, self.previous_dir,
                )
                await self._restore_state(agent, environment, env, self.previous_dir)
            except BaseException as exc:
                self._update(phase="incompatible", failure_type=type(exc).__name__)
                self._event("restore_rejected", failure_type=type(exc).__name__)
                raise
            self.session_id = self._previous_session_id(self.previous_dir)
            self._event(
                "checkpoint_restored", native_session_available=bool(self.session_id),
            )
        try:
            await self._install_runtime(
                agent, environment, env, manifest["checkpoint_id"],
            )
            await self._snapshot_once(agent, environment, env)
        except BaseException as exc:
            self._invalidate_snapshot(
                "checkpoint snapshot preflight failed before model start",
            )
            self._update(phase="invalid", failure_type="CheckpointSnapshotInvalid")
            self._event(
                "invalid",
                native_session_available=False,
                failure_type=type(exc).__name__,
            )
            raise
        self.snapshot_launch_attempted = True
        try:
            self._periodic_stop = asyncio.Event()
            self._snapshot_mutex = asyncio.Lock()
            self._periodic_failure = None
            self._owner_task = asyncio.current_task()
            self._periodic_environment = environment
            self._periodic_task = asyncio.create_task(
                self._periodic_loop(agent, environment, dict(env)),
                name=f"dradar-checkpoint-{manifest['checkpoint_id']}",
            )
            # Force one scheduling boundary so immediate task-construction
            # failures are handled here rather than by a harness finally block.
            await asyncio.sleep(0)
            if self._periodic_task.done():
                await self._periodic_task
                raise CheckpointError("checkpoint periodic task stopped at readiness")
            self.snapshot_background_ready = True
        except BaseException as exc:
            stop_failure: BaseException | None = None
            try:
                await self._stop_periodic()
            except BaseException as stop_exc:
                stop_failure = stop_exc
            self.snapshot_background_ready = False
            self._owner_task = None
            self._invalidate_snapshot(
                "checkpoint periodic task did not become ready",
            )
            self._update(phase="invalid", failure_type="CheckpointSnapshotInvalid")
            self._event("invalid", failure_type=type(exc).__name__)
            if stop_failure is not None and not isinstance(
                exc, asyncio.CancelledError,
            ):
                raise stop_failure from exc
            raise
        return self.session_id

    async def finish(
        self,
        agent: Any,
        environment: Any,
        env: dict[str, str],
        *,
        completed: bool,
        failure: BaseException | None,
        session_id: str | None = None,
    ) -> None:
        if not self.enabled or self.manifest_path is None:
            return
        selected_session = session_id or self.session_id
        if not (
            isinstance(selected_session, str)
            and _SESSION_ID_RE.fullmatch(selected_session)
        ):
            selected_session = None
        elif _bytes_contains_secret(
            selected_session.encode("utf-8"), self.sensitive_values,
        ):
            # Session identifiers are persisted in both the generation and
            # manifest. Treat an exact provider credential or generic token
            # shape as rejected data before the final snapshot can copy it.
            self._mark_invalid_secret(
                "credential-shaped checkpoint session id was rejected",
            )
            selected_session = None
        snapshot_stopped = True
        deferred_cancellation: asyncio.CancelledError | None = None
        snapshot_failure: BaseException | None = None
        if self.snapshot_launch_attempted:
            try:
                await self._stop_periodic()
            except asyncio.CancelledError as exc:
                # _stop_periodic has already reaped the writer. Finish the
                # final snapshot/manifest transition before re-propagating.
                deferred_cancellation = exc
            except BaseException as exc:
                snapshot_failure = exc

            if snapshot_failure is None:
                try:
                    if self._periodic_failure is not None:
                        raise self._periodic_failure
                    if self._snapshot_mutex is None:
                        raise CheckpointError(
                            "checkpoint periodic lock is unavailable",
                        )
                    async with self._snapshot_mutex:
                        await self._snapshot_once(
                            agent,
                            environment,
                            env,
                            session_id=selected_session,
                        )
                except asyncio.CancelledError as exc:
                    if deferred_cancellation is None:
                        deferred_cancellation = exc
                    snapshot_failure = exc
                except BaseException as exc:
                    snapshot_failure = exc

            if snapshot_failure is not None:
                snapshot_stopped = False
                self._invalidate_snapshot(
                    "checkpoint snapshot did not stop cleanly",
                )
        if self.snapshot_launch_attempted and not self.snapshot_background_ready:
            snapshot_stopped = False
            self._invalidate_snapshot(
                "checkpoint snapshot background did not become ready",
            )
        try:
            if snapshot_stopped:
                self._remove_sensitive_artifacts()
        except BaseException:
            snapshot_stopped = False
            self._invalidate_snapshot(
                "checkpoint artifact validation failed",
            )
        try:
            payload_dir = _snapshot_payload_dir(self.host_dir)
        except CheckpointError:
            payload_dir = self.host_dir
            snapshot_stopped = False
            self._mark_invalid_snapshot(
                "checkpoint generation could not be resolved at finish",
            )
        native_state_omitted = False
        try:
            native_state_omitted = self._marker_present(
                payload_dir, "session-omitted-sensitive",
            )
        except CheckpointError:
            snapshot_stopped = False
            self._mark_invalid_snapshot(
                "checkpoint provider-state omission marker is unsafe",
            )
        if native_state_omitted:
            selected_session = None
        if selected_session is None and not native_state_omitted:
            sidecar = payload_dir / "session-id"
            try:
                if _lexists(sidecar):
                    _validate_regular_tree(sidecar, label="checkpoint session id")
                    metadata = sidecar.lstat()
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_size > 512
                    ):
                        raise CheckpointError(
                            "checkpoint session id is unsafe",
                        )
                    candidate = sidecar.read_text(
                        encoding="utf-8", errors="replace",
                    ).strip()
                    if _SESSION_ID_RE.fullmatch(candidate):
                        selected_session = candidate
            except BaseException:
                snapshot_stopped = False
                self._mark_invalid_snapshot(
                    "checkpoint session id could not be validated",
                )
        try:
            invalid_secret = self._marker_present(
                self.host_dir, "invalid-secret",
            ) or self._marker_present(payload_dir, "invalid-secret")
            invalid_snapshot = self._marker_present(
                self.host_dir, "invalid-snapshot",
            )
        except CheckpointError:
            self._mark_invalid_snapshot("checkpoint marker validation failed")
            invalid_secret = False
            invalid_snapshot = True
        invalid = invalid_secret or invalid_snapshot
        phase = (
            "invalid"
            if invalid
            else (
                "incompatible"
                if isinstance(failure, CheckpointIncompatibleError)
                else ("agent_completed" if completed else "paused")
            )
        )
        changes: dict[str, Any] = {"phase": phase}
        if (
            not invalid
            and isinstance(selected_session, str)
            and _SESSION_ID_RE.fullmatch(selected_session)
        ):
            changes["session_id"] = selected_session
        if invalid_secret:
            changes["failure_type"] = "CheckpointSecretDetected"
        elif invalid_snapshot:
            changes["failure_type"] = "CheckpointSnapshotInvalid"
        elif failure is not None:
            changes["failure_type"] = type(failure).__name__
        self._update(**changes)
        self._event(
            phase,
            native_session_available=bool(changes.get("session_id")),
            failure_type=type(failure).__name__ if failure else None,
        )
        if deferred_cancellation is not None:
            raise deferred_cancellation

    async def finish_durably(
        self,
        agent: Any,
        environment: Any,
        env: dict[str, str],
        *,
        completed: bool,
        failure: BaseException | None,
        session_id: str | None = None,
    ) -> None:
        """Reap finalization through repeated caller cancellation.

        Harness ``finally`` blocks are themselves cancellable.  Running the
        final snapshot in a child task and repeatedly shielding it ensures a
        second cancellation cannot strand a live periodic writer or skip the
        terminal manifest transition.  Cancellation is re-propagated only
        after the finalizer has reached a terminal state.
        """

        finalizer = asyncio.create_task(
            self.finish(
                agent,
                environment,
                env,
                completed=completed,
                failure=failure,
                session_id=session_id,
            ),
            name=f"dradar-checkpoint-finalize-{self.assignment_id or 'unknown'}",
        )
        deferred_cancellation: asyncio.CancelledError | None = None
        while not finalizer.done():
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError as exc:
                if deferred_cancellation is None:
                    deferred_cancellation = exc
                continue
            except BaseException:
                break
        if finalizer.cancelled():
            finalizer_error: BaseException | None = CheckpointError(
                "checkpoint finalizer was cancelled unexpectedly",
            )
        else:
            try:
                finalizer_error = finalizer.exception()
            except asyncio.CancelledError:
                finalizer_error = CheckpointError(
                    "checkpoint finalizer was cancelled unexpectedly",
                )
        if finalizer_error is not None:
            if deferred_cancellation is not None:
                raise deferred_cancellation from finalizer_error
            raise finalizer_error
        if deferred_cancellation is not None:
            raise deferred_cancellation


__all__ = [
    "AgentLogStore",
    "CheckpointError",
    "CheckpointIncompatibleError",
    "DurableCheckpoint",
    "StatePath",
    "UnsafeAgentLog",
]
