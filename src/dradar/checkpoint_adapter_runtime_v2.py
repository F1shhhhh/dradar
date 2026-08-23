"""Container-side Harness capture and offline restore primitives for V2.

These functions contain no Provider CLI invocation and no assignment API.
Capture runs against an already-running Harness filesystem and writes only to
the container-native V2 staging root.  Restore targets a disposable offline
worktree/state root; it never starts a model or grants paid execution.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

from .checkpoint_adapters_v2 import (
    COMMON_CAPTURE_FILES_V2,
    PROVIDER_STATE_DIR_V2,
    SESSION_ID_FILE_V2,
    HarnessCheckpointContractV2,
    NativeStateArtifactV2,
    recovery_capability_for_capture_v2,
    validate_adapter_capture_root_v2,
)
from .checkpoint_runtime_v2 import (
    CheckpointDataPlaneError,
    CheckpointPackageLimitsV2,
    DEFAULT_PACKAGE_LIMITS_V2,
    PublishedCheckpointV2,
    revalidate_published_checkpoint_v2,
)


ADAPTER_PROGRESS_SCHEMA_V2 = "dradar-checkpoint-adapter-progress-v2"
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")
_SESSION_RE = re.compile(r"[A-Za-z0-9._:-]{8,160}")
_GENERIC_SECRET_RE = re.compile(
    rb"(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"
    rb"|ghp_[A-Za-z0-9]{20,}"
    rb"|github_pat_[A-Za-z0-9_]{20,}"
    rb"|gAAAAA[A-Za-z0-9_-]{40,}"
    rb"|eyJ[A-Za-z0-9_-]{10,}[.][A-Za-z0-9_-]{10,}"
    rb"[.][A-Za-z0-9_-]{10,})"
)


@dataclass(frozen=True)
class AdapterCaptureSummaryV2:
    capture_root: Path
    base_commit: str
    present_artifacts: frozenset[str]
    session_id: str | None
    recovery_capability: str
    workspace_patch_bytes: int
    untracked_files: int
    untracked_bytes: int


@dataclass(frozen=True)
class OfflineAdapterRestoreV2:
    worktree: Path
    provider_state_root: Path
    base_commit: str
    session_id: str | None
    recovery_capability: str
    restored_untracked_files: int
    paid_execution_started: bool = False


def _resolve_container_path(filesystem_root: Path, path: str | PurePosixPath) -> Path:
    logical = PurePosixPath(path)
    if not logical.is_absolute() or ".." in logical.parts:
        raise CheckpointDataPlaneError("capture", "adapter_path_invalid")
    return Path(filesystem_root).joinpath(*logical.parts[1:])


def _private_directory(path: Path, *, stage: str, create: bool) -> None:
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
        raise CheckpointDataPlaneError(stage, "adapter_directory_unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CheckpointDataPlaneError(stage, "adapter_directory_unsafe")
    if created:
        try:
            path.chmod(0o700)
            metadata = path.lstat()
        except OSError as exc:
            raise CheckpointDataPlaneError(stage, "adapter_permissions_failed") from exc
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise CheckpointDataPlaneError(stage, "adapter_directory_not_private")


def _safe_directory(path: Path, *, stage: str) -> None:
    """Require a real directory without imposing checkpoint-storage modes.

    Harness worktrees are ordinary task inputs, not secret checkpoint storage.
    Images and bind-mount backends commonly expose them as 0755, so requiring
    0700 here would make capture platform-dependent.  Symlinks and non-
    directories remain forbidden to keep path resolution deterministic.
    """

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError(stage, "adapter_directory_unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CheckpointDataPlaneError(stage, "adapter_directory_unsafe")


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/nonexistent-dradar-checkpoint-v2",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run_git_to_file(
    worktree: Path,
    arguments: list[str],
    destination: Path,
    *,
    max_bytes: int,
    timeout_sec: float = 30.0,
    stage: str = "capture",
) -> int:
    if stage not in {"capture", "restore"}:
        raise ValueError("checkpoint git stage is invalid")
    if not 0.1 <= timeout_sec <= 300:
        raise ValueError("checkpoint git timeout is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(destination, flags, 0o600)
    stderr_path = destination.with_name(f".{destination.name}.stderr")
    stderr_descriptor = os.open(stderr_path, flags, 0o600)
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            ["git", "-C", os.fspath(worktree), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=stderr_descriptor,
            env=_git_environment(),
            close_fds=True,
        )
        while process.poll() is None:
            if (
                os.fstat(descriptor).st_size > max_bytes
                or os.fstat(stderr_descriptor).st_size > 64 * 1024
            ):
                process.kill()
                process.wait()
                raise CheckpointDataPlaneError(stage, "adapter_git_output_limit")
            if time.monotonic() - started > timeout_sec:
                process.kill()
                process.wait()
                raise CheckpointDataPlaneError(stage, "adapter_git_timeout")
            time.sleep(0.01)
        if process.returncode != 0:
            raise CheckpointDataPlaneError(stage, "adapter_git_failed")
        size = os.fstat(descriptor).st_size
        if size > max_bytes:
            raise CheckpointDataPlaneError(stage, "adapter_git_output_limit")
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        return size
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        os.close(descriptor)
        os.close(stderr_descriptor)
        stderr_path.unlink(missing_ok=True)


def _git_text(
    worktree: Path,
    arguments: list[str],
    *,
    max_bytes: int,
    stage: str = "capture",
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="dradar-checkpoint-git-") as raw:
        path = Path(raw) / "output"
        _run_git_to_file(
            worktree, arguments, path, max_bytes=max_bytes, stage=stage,
        )
        return path.read_bytes()


def _validate_base_commit(
    worktree: Path, base_commit: str, *, stage: str = "capture",
) -> None:
    if _COMMIT_RE.fullmatch(base_commit) is None:
        raise CheckpointDataPlaneError(stage, "adapter_base_commit_invalid")
    observed = _git_text(
        worktree,
        ["rev-parse", "--verify", f"{base_commit}^{{commit}}"],
        max_bytes=256,
        stage=stage,
    ).strip()
    if observed.decode("ascii", errors="ignore") != base_commit:
        raise CheckpointDataPlaneError(stage, "adapter_base_commit_mismatch")


def _safe_relative_path(raw: bytes, *, limits: CheckpointPackageLimitsV2) -> PurePosixPath:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckpointDataPlaneError("capture", "untracked_path_encoding") from exc
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) > limits.max_depth
        or len(text.encode("utf-8")) > limits.max_path_bytes
    ):
        raise CheckpointDataPlaneError("capture", "untracked_path_invalid")
    return path


class _ScanningReader:
    def __init__(
        self,
        raw: BinaryIO,
        *,
        sensitive_values: tuple[bytes, ...],
        expected_size: int,
    ) -> None:
        self.raw = raw
        self.sensitive_values = sensitive_values
        self.expected_size = expected_size
        self.total = 0
        self.overlap = b""

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.total += len(chunk)
        if self.total > self.expected_size:
            raise CheckpointDataPlaneError("capture", "untracked_file_changed")
        scan = self.overlap + chunk
        if _GENERIC_SECRET_RE.search(scan) or any(
            value and value in scan for value in self.sensitive_values
        ):
            raise CheckpointDataPlaneError("capture", "secret_detected")
        self.overlap = scan[-512:]
        return chunk


def _write_untracked_archive(
    worktree: Path,
    paths: list[PurePosixPath],
    destination: Path,
    *,
    sensitive_values: tuple[bytes, ...],
    limits: CheckpointPackageLimitsV2,
) -> tuple[int, int]:
    total_bytes = 0
    directories: set[str] = set()
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w:", format=tarfile.PAX_FORMAT) as archive:
                for relative in paths:
                    path = worktree.joinpath(*relative.parts)
                    try:
                        metadata = path.lstat()
                    except OSError as exc:
                        raise CheckpointDataPlaneError(
                            "capture", "untracked_file_unavailable",
                        ) from exc
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or path.is_symlink()
                        or metadata.st_nlink != 1
                        or metadata.st_size > limits.max_file_bytes
                    ):
                        raise CheckpointDataPlaneError(
                            "capture", "untracked_file_unsafe",
                        )
                    total_bytes += metadata.st_size
                    if total_bytes > limits.max_total_bytes:
                        raise CheckpointDataPlaneError(
                            "capture", "untracked_total_limit",
                        )
                    parents: list[PurePosixPath] = []
                    parent = relative.parent
                    while parent != PurePosixPath("."):
                        parents.append(parent)
                        parent = parent.parent
                    for directory in reversed(parents):
                        name = directory.as_posix()
                        if name in directories:
                            continue
                        info = tarfile.TarInfo(name)
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o700
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        info.pax_headers = {}
                        archive.addfile(info)
                        directories.add(name)
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    descriptor = os.open(path, flags)
                    try:
                        opened = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(opened.st_mode)
                            or opened.st_nlink != 1
                            or (opened.st_dev, opened.st_ino, opened.st_size)
                            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
                        ):
                            raise CheckpointDataPlaneError(
                                "capture", "untracked_file_changed",
                            )
                        info = tarfile.TarInfo(relative.as_posix())
                        info.size = opened.st_size
                        info.mode = 0o600
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        info.pax_headers = {}
                        with os.fdopen(os.dup(descriptor), "rb") as source:
                            scanner = _ScanningReader(
                                source,
                                sensitive_values=sensitive_values,
                                expected_size=opened.st_size,
                            )
                            archive.addfile(info, scanner)
                            if scanner.total != opened.st_size:
                                raise CheckpointDataPlaneError(
                                    "capture", "untracked_file_changed",
                                )
                        after = os.fstat(descriptor)
                        current = path.lstat()
                        if (
                            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                            or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                        ):
                            raise CheckpointDataPlaneError(
                                "capture", "untracked_file_changed",
                            )
                    finally:
                        os.close(descriptor)
        raw.flush()
        os.fsync(raw.fileno())
    destination.chmod(0o600)
    if destination.stat().st_size > COMMON_CAPTURE_FILES_V2["untracked.tar.gz"]:
        raise CheckpointDataPlaneError("capture", "untracked_archive_size_limit")
    return len(paths), total_bytes


def _copy_native_artifact(
    source: Path,
    destination: Path,
    artifact: NativeStateArtifactV2,
) -> bool:
    if not source.exists() and not source.is_symlink():
        return False
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError("capture", "native_artifact_unavailable") from exc
    if artifact.kind == "file":
        if (
            not stat.S_ISREG(metadata.st_mode)
            or source.is_symlink()
            or metadata.st_nlink != 1
            or metadata.st_size > artifact.max_bytes
        ):
            raise CheckpointDataPlaneError("capture", "native_artifact_unsafe")
        if artifact.restore and metadata.st_size == 0:
            return False
        shutil.copyfile(source, destination, follow_symlinks=False)
        destination.chmod(0o600)
    else:
        if not stat.S_ISDIR(metadata.st_mode) or source.is_symlink():
            raise CheckpointDataPlaneError("capture", "native_artifact_unsafe")
        total = 0
        entries = 0
        material = False
        try:
            for current, directory_names, file_names in os.walk(
                source, topdown=True, followlinks=False,
            ):
                current_path = Path(current)
                for name in directory_names:
                    candidate = current_path / name
                    candidate_metadata = candidate.lstat()
                    entries += 1
                    if (
                        candidate.is_symlink()
                        or not stat.S_ISDIR(candidate_metadata.st_mode)
                    ):
                        raise CheckpointDataPlaneError(
                            "capture", "native_artifact_unsafe",
                        )
                for name in file_names:
                    candidate = current_path / name
                    candidate_metadata = candidate.lstat()
                    entries += 1
                    if (
                        candidate.is_symlink()
                        or not stat.S_ISREG(candidate_metadata.st_mode)
                        or candidate_metadata.st_nlink != 1
                    ):
                        raise CheckpointDataPlaneError(
                            "capture", "native_artifact_unsafe",
                        )
                    total += candidate_metadata.st_size
                    material = material or candidate_metadata.st_size > 0
                    if total > artifact.max_bytes or entries > 20_000:
                        raise CheckpointDataPlaneError(
                            "capture", "native_artifact_size_limit",
                        )
        except OSError as exc:
            raise CheckpointDataPlaneError(
                "capture", "native_artifact_unavailable",
            ) from exc
        if artifact.restore and not material:
            return False
        try:
            shutil.copytree(source, destination, symlinks=True)
        except (OSError, shutil.Error) as exc:
            raise CheckpointDataPlaneError("capture", "native_artifact_copy_failed") from exc
        destination.chmod(0o700)
    return True


def _validate_clean_restore_worktree(worktree: Path, base_commit: str) -> None:
    head = _git_text(
        worktree, ["rev-parse", "HEAD"], max_bytes=256, stage="restore",
    ).strip().decode("ascii", errors="ignore")
    if head != base_commit:
        raise CheckpointDataPlaneError("restore", "restore_base_commit_mismatch")
    status = _git_text(
        worktree,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        max_bytes=4 * 1024 * 1024,
        stage="restore",
    )
    if status:
        raise CheckpointDataPlaneError("restore", "restore_worktree_not_clean")


def create_adapter_capture_root_v2(
    *,
    filesystem_root: Path,
    worktree_path: str,
    capture_root_path: str,
    contract: HarnessCheckpointContractV2,
    base_commit: str,
    captured_at: str,
    session_id: str | None,
    sensitive_values: Iterable[str | bytes] = (),
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
) -> AdapterCaptureSummaryV2:
    """Create the exact reviewed adapter tree on a container filesystem."""

    filesystem_root = Path(filesystem_root)
    worktree = _resolve_container_path(filesystem_root, worktree_path)
    capture_root = _resolve_container_path(filesystem_root, capture_root_path)
    native_root = _resolve_container_path(filesystem_root, "/run/dradar-checkpoint-v2")
    try:
        capture_root.relative_to(native_root)
    except ValueError as exc:
        raise CheckpointDataPlaneError("capture", "capture_root_outside_native_storage") from exc
    _private_directory(native_root, stage="capture", create=True)
    _private_directory(capture_root.parent, stage="capture", create=True)
    if capture_root.exists() or capture_root.is_symlink():
        raise CheckpointDataPlaneError("capture", "adapter_capture_root_exists")
    _safe_directory(worktree, stage="capture")
    if capture_root.parent.lstat().st_dev != native_root.lstat().st_dev:
        raise CheckpointDataPlaneError("capture", "capture_storage_cross_device")
    if session_id is not None and _SESSION_RE.fullmatch(session_id) is None:
        raise CheckpointDataPlaneError("capture", "adapter_session_id_invalid")
    needles = tuple(
        value if isinstance(value, bytes) else value.encode("utf-8")
        for value in sensitive_values
        if isinstance(value, (str, bytes)) and len(value) >= 8
    )
    capture_root.mkdir(mode=0o700)
    capture_root.chmod(0o700)
    try:
        _validate_base_commit(worktree, base_commit)
        patch = capture_root / "workspace.patch"
        patch_bytes = _run_git_to_file(
            worktree,
            ["diff", "--binary", "--no-ext-diff", "--full-index", base_commit, "--"],
            patch,
            max_bytes=COMMON_CAPTURE_FILES_V2["workspace.patch"],
        )
        raw_untracked = _git_text(
            worktree,
            ["ls-files", "--others", "--exclude-standard", "-z", "--"],
            max_bytes=4 * 1024 * 1024,
        )
        raw_paths = [value for value in raw_untracked.split(b"\0") if value]
        if len(raw_paths) > limits.max_files:
            raise CheckpointDataPlaneError("capture", "untracked_entry_limit")
        paths = sorted(
            (_safe_relative_path(value, limits=limits) for value in raw_paths),
            key=PurePosixPath.as_posix,
        )
        if len(paths) != len(set(paths)):
            raise CheckpointDataPlaneError("capture", "untracked_path_duplicate")
        untracked_count, untracked_bytes = _write_untracked_archive(
            worktree,
            paths,
            capture_root / "untracked.tar.gz",
            sensitive_values=needles,
            limits=limits,
        )
        provider_state = capture_root / PROVIDER_STATE_DIR_V2
        provider_state.mkdir(mode=0o700)
        present: set[str] = set()
        for artifact in contract.artifacts:
            source = _resolve_container_path(filesystem_root, artifact.source_path)
            if _copy_native_artifact(
                source,
                provider_state / artifact.name,
                artifact,
            ):
                present.add(artifact.name)
        if not present:
            provider_state.rmdir()
        if session_id is not None:
            (capture_root / SESSION_ID_FILE_V2).write_text(
                session_id + "\n", encoding="utf-8",
            )
            (capture_root / SESSION_ID_FILE_V2).chmod(0o600)
        capability = recovery_capability_for_capture_v2(
            contract,
            present_artifacts=frozenset(present),
            has_session_id=session_id is not None,
        )
        progress = {
            "schema": ADAPTER_PROGRESS_SCHEMA_V2,
            "harness": contract.harness,
            "provider": contract.provider,
            "checkpoint_abi": contract.checkpoint_abi,
            "base_commit": base_commit,
            "captured_at": captured_at,
            "session_id_present": session_id is not None,
            "native_artifacts": sorted(present),
            "recovery_capability": capability,
            "workspace_patch_bytes": patch_bytes,
            "untracked_files": untracked_count,
            "untracked_bytes": untracked_bytes,
        }
        progress_path = capture_root / "progress.json"
        progress_path.write_text(
            json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        progress_path.chmod(0o600)
        validated = validate_adapter_capture_root_v2(capture_root, contract)
        if validated != frozenset(present):
            raise CheckpointDataPlaneError("capture", "adapter_inventory_mismatch")
        return AdapterCaptureSummaryV2(
            capture_root=capture_root,
            base_commit=base_commit,
            present_artifacts=frozenset(present),
            session_id=session_id,
            recovery_capability=capability,
            workspace_patch_bytes=patch_bytes,
            untracked_files=untracked_count,
            untracked_bytes=untracked_bytes,
        )
    except BaseException:
        shutil.rmtree(capture_root, ignore_errors=True)
        raise


def _restore_untracked_archive(
    archive_path: Path,
    worktree: Path,
    *,
    limits: CheckpointPackageLimitsV2,
) -> int:
    seen: set[str] = set()
    restored = 0
    total = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            name = member.name.rstrip("/")
            relative = _safe_relative_path(name.encode("utf-8"), limits=limits)
            if name in seen:
                raise CheckpointDataPlaneError("restore", "untracked_path_duplicate")
            seen.add(name)
            target = worktree.joinpath(*relative.parts)
            if member.isdir():
                if member.issym() or member.islnk() or stat.S_IMODE(member.mode) != 0o700:
                    raise CheckpointDataPlaneError("restore", "untracked_member_invalid")
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or stat.S_IMODE(member.mode) != 0o600
                or member.size > limits.max_file_bytes
            ):
                raise CheckpointDataPlaneError("restore", "untracked_member_invalid")
            total += member.size
            if total > limits.max_total_bytes:
                raise CheckpointDataPlaneError("restore", "untracked_total_limit")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(target, flags, 0o600)
            try:
                source = archive.extractfile(member)
                if source is None:
                    raise CheckpointDataPlaneError("restore", "untracked_member_unreadable")
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise CheckpointDataPlaneError("restore", "untracked_member_truncated")
                    view = memoryview(chunk)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise CheckpointDataPlaneError("restore", "untracked_write_failed")
                        view = view[written:]
                    remaining -= len(chunk)
                if source.read(1):
                    raise CheckpointDataPlaneError("restore", "untracked_member_oversized")
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            restored += 1
    return restored


def restore_adapter_capture_offline_v2(
    *,
    published: PublishedCheckpointV2,
    contract: HarnessCheckpointContractV2,
    destination_worktree: Path,
    destination_state_root: Path,
    expected_identity_fingerprint: str,
    base_commit: str,
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
) -> OfflineAdapterRestoreV2:
    """Restore into disposable roots and prove no paid Provider process ran."""

    revalidate_published_checkpoint_v2(
        published,
        expected_identity_fingerprint=expected_identity_fingerprint,
        expected_checkpoint_abi=contract.checkpoint_abi,
        limits=limits,
    )
    worktree = Path(destination_worktree)
    state_root = Path(destination_state_root)
    _safe_directory(worktree, stage="restore")
    if state_root.exists() or state_root.is_symlink():
        raise CheckpointDataPlaneError("restore", "restore_state_root_exists")
    state_root.mkdir(mode=0o700, parents=True)
    state_root.chmod(0o700)
    try:
        _validate_base_commit(worktree, base_commit, stage="restore")
        _validate_clean_restore_worktree(worktree, base_commit)
        payload = published.payload_root
        progress = json.loads((payload / "progress.json").read_text(encoding="utf-8"))
        captured_artifacts = validate_adapter_capture_root_v2(payload, contract)
        if (
            not isinstance(progress, dict)
            or progress.get("schema") != ADAPTER_PROGRESS_SCHEMA_V2
            or progress.get("harness") != contract.harness
            or progress.get("provider") != contract.provider
            or progress.get("checkpoint_abi") != contract.checkpoint_abi
            or progress.get("base_commit") != base_commit
            or progress.get("native_artifacts") != sorted(captured_artifacts)
            or progress.get("session_id_present")
            is not (payload / SESSION_ID_FILE_V2).exists()
        ):
            raise CheckpointDataPlaneError("restore", "adapter_progress_invalid")
        patch = payload / "workspace.patch"
        if patch.stat().st_size:
            for arguments in (
                ["apply", "--check", "--binary", os.fspath(patch)],
                ["apply", "--binary", os.fspath(patch)],
            ):
                with tempfile.TemporaryDirectory(prefix="dradar-checkpoint-restore-") as raw:
                    _run_git_to_file(
                        worktree,
                        arguments,
                        Path(raw) / "output",
                        max_bytes=64 * 1024,
                        stage="restore",
                    )
        restored_untracked = _restore_untracked_archive(
            payload / "untracked.tar.gz", worktree, limits=limits,
        )
        provider_source = payload / PROVIDER_STATE_DIR_V2
        present: set[str] = set()
        if provider_source.exists():
            artifact_by_name = {item.name: item for item in contract.artifacts}
            for source in provider_source.iterdir():
                artifact = artifact_by_name.get(source.name)
                if artifact is None or not artifact.restore:
                    continue
                destination = state_root / source.name
                if artifact.kind == "directory":
                    shutil.copytree(source, destination, symlinks=True)
                    destination.chmod(0o700)
                else:
                    shutil.copyfile(source, destination, follow_symlinks=False)
                    destination.chmod(0o600)
                present.add(source.name)
        session_id: str | None = None
        session_path = payload / SESSION_ID_FILE_V2
        if session_path.exists():
            session_id = session_path.read_text(encoding="utf-8").strip()
            if _SESSION_RE.fullmatch(session_id) is None:
                raise CheckpointDataPlaneError("restore", "adapter_session_id_invalid")
        capability = recovery_capability_for_capture_v2(
            contract,
            present_artifacts=frozenset(present),
            has_session_id=session_id is not None,
        )
        if capability != progress.get("recovery_capability"):
            raise CheckpointDataPlaneError("restore", "adapter_capability_mismatch")
        return OfflineAdapterRestoreV2(
            worktree=worktree,
            provider_state_root=state_root,
            base_commit=base_commit,
            session_id=session_id,
            recovery_capability=capability,
            restored_untracked_files=restored_untracked,
        )
    except BaseException:
        shutil.rmtree(state_root, ignore_errors=True)
        raise


__all__ = [
    "ADAPTER_PROGRESS_SCHEMA_V2",
    "AdapterCaptureSummaryV2",
    "OfflineAdapterRestoreV2",
    "create_adapter_capture_root_v2",
    "restore_adapter_capture_offline_v2",
]
