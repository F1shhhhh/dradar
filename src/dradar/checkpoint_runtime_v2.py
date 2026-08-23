"""Optional, container-native data plane for checkpoint protocol v2.

The normal trial/result path is authoritative.  This module deliberately has
no API client and cannot invalidate, release, refill, pause, or resume an
assignment.  In ``OBSERVE`` mode it may capture and verify a local shadow
snapshot, but every ordinary failure is converted into a bounded observation
result so the caller can continue without checkpoint support.

Harness adapters must capture into a container-owned Linux filesystem and
return one sealed archive below ``/run/dradar-checkpoint-v2``.  The host treats
that archive as untrusted: it verifies the complete manifest and every member
before an atomic, host-private publication.  No bind-mount UID/GID or sticky
bit is part of the trust boundary.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable, Iterable, Mapping, Protocol

from .checkpoint_activation_v2 import (
    CHECKPOINT_CORE_ABI_V2,
    CheckpointActivationV2,
)
from .checkpoint_protocol_types_v2 import (
    CheckpointGenerationRefV2,
    CheckpointRetentionAcknowledgementV2,
)


EXPORT_SCHEMA_V2 = "dradar-checkpoint-export-v2"
MANIFEST_NAME = "manifest.json"
PAYLOAD_ROOT = "payload"
CONTAINER_EXPORT_ROOT = PurePosixPath("/run/dradar-checkpoint-v2")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ABI_RE = re.compile(r"^[A-Za-z0-9._/-]{8,160}$")
_NATIVE_SCHEMA_RE = re.compile(r"^[A-Za-z0-9._/-]{1,160}$")
_GENERIC_SECRET_RE = re.compile(
    rb"(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"
    rb"|ghp_[A-Za-z0-9]{20,}"
    rb"|github_pat_[A-Za-z0-9_]{20,}"
    rb"|gAAAAA[A-Za-z0-9_-]{40,}"
    rb"|eyJ[A-Za-z0-9_-]{10,}[.][A-Za-z0-9_-]{10,}"
    rb"[.][A-Za-z0-9_-]{10,})"
)
_MANIFEST_FIELDS = frozenset({
    "schema",
    "protocol_version",
    "checkpoint_core_abi",
    "checkpoint_abi",
    "checkpoint_id",
    "checkpoint_lineage_id",
    "snapshot_generation",
    "capture_id",
    "identity_fingerprint",
    "recovery_capability",
    "native_state_schema",
    "captured_at",
    "capture_storage",
    "directories",
    "files",
    "file_count",
    "total_bytes",
})
_RECOVERY_CAPABILITIES = frozenset({
    "NATIVE_VALID", "WORKSPACE_ONLY", "COMPLETED_UPLOAD_ONLY", "NONE",
})


@dataclass(frozen=True)
class CheckpointPackageLimitsV2:
    max_files: int = 20_000
    max_depth: int = 64
    max_path_bytes: int = 1024
    max_file_bytes: int = 256 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    max_manifest_bytes: int = 2 * 1024 * 1024
    max_archive_bytes: int = 640 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.max_files,
            self.max_depth,
            self.max_path_bytes,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_manifest_bytes,
            self.max_archive_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError("checkpoint package limits must be positive integers")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("checkpoint file limit exceeds total limit")


DEFAULT_PACKAGE_LIMITS_V2 = CheckpointPackageLimitsV2()


@dataclass(frozen=True)
class CheckpointRetentionPolicyV2:
    """Bound optional shadow storage without deleting authoritative evidence.

    Shadow generations have no assignment authority and may be sampled again,
    so retaining the newest two is enough to exercise fallback and corruption
    handling.  CANARY/ON generations are deliberately never pruned here: an
    authoritative server decision must identify the exact superseded
    generation before it can be removed.
    """

    shadow_generations: int = 2
    minimum_free_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.shadow_generations, int)
            or isinstance(self.shadow_generations, bool)
            or not 2 <= self.shadow_generations <= 32
        ):
            raise ValueError(
                "checkpoint shadow retention must keep between 2 and 32 generations"
            )
        if (
            not isinstance(self.minimum_free_bytes, int)
            or isinstance(self.minimum_free_bytes, bool)
            or self.minimum_free_bytes < 0
            or self.minimum_free_bytes > 1024 * 1024 * 1024 * 1024
        ):
            raise ValueError("checkpoint minimum free-space reserve is invalid")


DEFAULT_RETENTION_POLICY_V2 = CheckpointRetentionPolicyV2()


class CheckpointDataPlaneError(RuntimeError):
    """Typed checkpoint failure with optional local-only diagnostics.

    ``stage`` and ``code`` are the only fields exported to aggregate server
    telemetry.  ``diagnostic`` is deliberately restricted to bounded scalar
    facts (exit code, byte counts and content digests) so callers may persist
    useful local evidence without copying arbitrary command output, paths or
    Provider data into the protocol.
    """

    def __init__(
        self,
        stage: str,
        code: str,
        *,
        diagnostic: Mapping[str, str | int | bool] | None = None,
    ) -> None:
        if stage not in {
            "capture", "seal", "download", "verify", "publish", "cleanup",
            "restore", "retention",
        }:
            raise ValueError("checkpoint data-plane stage is invalid")
        if re.fullmatch(r"[a-z][a-z0-9_]{2,63}", code) is None:
            raise ValueError("checkpoint data-plane code is invalid")
        bounded: dict[str, str | int | bool] = {}
        if diagnostic is not None:
            if not isinstance(diagnostic, Mapping) or len(diagnostic) > 16:
                raise ValueError("checkpoint diagnostic is invalid")
            for key, value in diagnostic.items():
                if (
                    not isinstance(key, str)
                    or re.fullmatch(r"[a-z][a-z0-9_]{1,47}", key) is None
                    or any(
                        marker in key
                        for marker in (
                            "token", "secret", "password", "credential",
                            "authorization", "api_key",
                        )
                    )
                    or not isinstance(value, (str, int, bool))
                    or isinstance(value, str) and len(value) > 160
                ):
                    raise ValueError("checkpoint diagnostic is invalid")
                bounded[key] = value
        super().__init__(f"checkpoint {stage} failed ({code})")
        self.stage = stage
        self.code = code
        self.diagnostic = bounded


@dataclass(frozen=True)
class CheckpointCaptureRequestV2:
    checkpoint_id: str
    checkpoint_lineage_id: str
    snapshot_generation: int
    capture_id: str
    identity_fingerprint: str
    checkpoint_abi: str
    recovery_capability: str
    native_state_schema: str | None
    captured_at: str

    def validate(self) -> None:
        for value, label in (
            (self.checkpoint_id, "checkpoint id"),
            (self.checkpoint_lineage_id, "checkpoint lineage id"),
            (self.capture_id, "capture id"),
        ):
            if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
                raise CheckpointDataPlaneError("capture", f"invalid_{label.replace(' ', '_')}")
        if (
            not isinstance(self.snapshot_generation, int)
            or isinstance(self.snapshot_generation, bool)
            or self.snapshot_generation < 0
        ):
            raise CheckpointDataPlaneError("capture", "invalid_generation")
        if _DIGEST_RE.fullmatch(self.identity_fingerprint) is None:
            raise CheckpointDataPlaneError("capture", "invalid_identity")
        if _ABI_RE.fullmatch(self.checkpoint_abi) is None:
            raise CheckpointDataPlaneError("capture", "invalid_adapter_abi")
        if self.recovery_capability not in _RECOVERY_CAPABILITIES:
            raise CheckpointDataPlaneError("capture", "invalid_capability")
        if (
            self.native_state_schema is not None
            and _NATIVE_SCHEMA_RE.fullmatch(self.native_state_schema) is None
        ):
            raise CheckpointDataPlaneError("capture", "invalid_native_schema")
        try:
            parsed = datetime.fromisoformat(self.captured_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise CheckpointDataPlaneError("capture", "invalid_capture_time") from exc
        if parsed.tzinfo is None:
            raise CheckpointDataPlaneError("capture", "invalid_capture_time")


@dataclass(frozen=True)
class ContainerSealedExportV2:
    capture_id: str
    remote_path: str
    archive_sha256: str
    archive_size: int
    manifest_sha256: str
    capture_storage: str

    def validate(self, request: CheckpointCaptureRequestV2) -> None:
        if self.capture_id != request.capture_id:
            raise CheckpointDataPlaneError("seal", "capture_identity_mismatch")
        if self.capture_storage != "container_native":
            raise CheckpointDataPlaneError("seal", "unsafe_capture_storage")
        remote = PurePosixPath(self.remote_path)
        if (
            not remote.is_absolute()
            or ".." in remote.parts
            or remote == CONTAINER_EXPORT_ROOT
            or not remote.is_relative_to(CONTAINER_EXPORT_ROOT)
        ):
            raise CheckpointDataPlaneError("seal", "unsafe_remote_export")
        if _DIGEST_RE.fullmatch(self.archive_sha256) is None:
            raise CheckpointDataPlaneError("seal", "invalid_archive_digest")
        if _DIGEST_RE.fullmatch(self.manifest_sha256) is None:
            raise CheckpointDataPlaneError("seal", "invalid_manifest_digest")
        if (
            not isinstance(self.archive_size, int)
            or isinstance(self.archive_size, bool)
            or self.archive_size <= 0
        ):
            raise CheckpointDataPlaneError("seal", "invalid_archive_size")


@dataclass(frozen=True)
class PublishedCheckpointV2:
    checkpoint_id: str
    snapshot_generation: int
    capture_id: str
    root: Path
    payload_root: Path
    archive_path: Path
    manifest_sha256: str
    archive_sha256: str
    archive_bytes: int
    file_count: int
    payload_bytes: int
    authoritative: bool
    selected: bool


@dataclass(frozen=True)
class CheckpointObservationV2:
    status: str
    capture_id: str | None
    stage: str | None = None
    code: str | None = None
    failure_type: str | None = None
    published: PublishedCheckpointV2 | None = None
    remote_cleanup: str = "not_needed"

    @property
    def mainline_may_continue(self) -> bool:
        return True


@dataclass(frozen=True)
class CheckpointRestoreRequestV2:
    published: PublishedCheckpointV2
    expected_identity_fingerprint: str
    restore_id: str


@dataclass(frozen=True)
class CheckpointRestoreEvidenceV2:
    restore_id: str
    manifest_sha256: str
    identity_fingerprint: str
    restore_adapter_version: str
    paid_execution_started: bool


@dataclass(frozen=True)
class CheckpointRestoreObservationV2:
    status: str
    restore_id: str
    stage: str | None = None
    code: str | None = None
    failure_type: str | None = None
    evidence: CheckpointRestoreEvidenceV2 | None = None

    @property
    def paid_execution_authorized(self) -> bool:
        # Offline restore never authorizes a model.  The control-plane receipt
        # and resume-commit transaction remain a separate later step.
        return False


@dataclass(frozen=True)
class CheckpointRetentionApplyResultV2:
    """Local effect of one exact, server-acknowledged release decision."""

    deleted_generations: tuple[CheckpointGenerationRefV2, ...]
    already_absent_generations: tuple[CheckpointGenerationRefV2, ...]
    retained_generations: tuple[CheckpointGenerationRefV2, ...]


@dataclass(frozen=True)
class CheckpointObservationRuntimeV2:
    assignment_id: str
    operation_id: str
    elapsed_ms: int
    platform: str
    container_backend: str
    client_version: str
    adapter_version: str


class HarnessCheckpointExporterV2(Protocol):
    adapter_version: str
    checkpoint_abi: str

    async def capture_and_seal(
        self, request: CheckpointCaptureRequestV2,
    ) -> ContainerSealedExportV2:
        """Capture in container-native storage and return a sealed export."""

    async def download_export(
        self,
        export: ContainerSealedExportV2,
        destination: Path,
        *,
        max_bytes: int,
    ) -> None:
        """Copy one sealed regular file to an exclusive host destination."""

    async def discard_export(self, export: ContainerSealedExportV2) -> None:
        """Delete only this capture's sealed container export."""


class HarnessCheckpointRestorerV2(Protocol):
    adapter_version: str
    checkpoint_abi: str

    async def restore_offline(
        self, request: CheckpointRestoreRequestV2,
    ) -> CheckpointRestoreEvidenceV2:
        """Restore and self-check without starting a paid model process."""


@dataclass
class _CopyBudget:
    limits: CheckpointPackageLimitsV2
    file_count: int = 0
    total_bytes: int = 0
    entry_count: int = 0

    def account_directory(self, *, depth: int, path: str) -> None:
        self.entry_count += 1
        if self.entry_count > self.limits.max_files:
            raise CheckpointDataPlaneError("seal", "entry_count_limit")
        if depth > self.limits.max_depth:
            raise CheckpointDataPlaneError("seal", "depth_limit")
        if len(path.encode("utf-8")) > self.limits.max_path_bytes:
            raise CheckpointDataPlaneError("seal", "path_length_limit")

    def account_file(self, *, size: int, depth: int, path: str) -> None:
        self.file_count += 1
        self.entry_count += 1
        self.total_bytes += size
        if self.entry_count > self.limits.max_files:
            raise CheckpointDataPlaneError("seal", "entry_count_limit")
        if depth > self.limits.max_depth:
            raise CheckpointDataPlaneError("seal", "depth_limit")
        if len(path.encode("utf-8")) > self.limits.max_path_bytes:
            raise CheckpointDataPlaneError("seal", "path_length_limit")
        if size > self.limits.max_file_bytes:
            raise CheckpointDataPlaneError("seal", "file_size_limit")
        if self.total_bytes > self.limits.max_total_bytes:
            raise CheckpointDataPlaneError("seal", "total_size_limit")


def _metadata_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8") + b"\n"


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointDataPlaneError("verify", "archive_open_failed") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CheckpointDataPlaneError("verify", "archive_not_regular")
        if max_bytes is not None and before.st_size > max_bytes:
            raise CheckpointDataPlaneError("verify", "archive_size_limit")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise CheckpointDataPlaneError("verify", "archive_size_limit")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise CheckpointDataPlaneError("verify", "archive_changed")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _copy_regular_file_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int,
    stage: str,
) -> None:
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    read_flags |= getattr(os, "O_CLOEXEC", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    write_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(source, read_flags)
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
            or before.st_size > max_bytes
        ):
            raise CheckpointDataPlaneError(stage, "archive_copy_source_unsafe")
        destination_fd = os.open(destination, write_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > expected_size or copied > max_bytes:
                raise CheckpointDataPlaneError(stage, "archive_copy_size_mismatch")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise CheckpointDataPlaneError(stage, "archive_copy_failed")
                view = view[written:]
        after = os.fstat(source_fd)
        if (
            copied != expected_size
            or digest.hexdigest() != expected_sha256
            or _metadata_fingerprint(before) != _metadata_fingerprint(after)
        ):
            raise CheckpointDataPlaneError(stage, "archive_copy_digest_mismatch")
        os.fchmod(destination_fd, 0o600)
        os.fsync(destination_fd)
    except CheckpointDataPlaneError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise CheckpointDataPlaneError(stage, "archive_copy_failed") from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _contains_secret(chunk: bytes, sensitive_values: tuple[bytes, ...]) -> bool:
    return any(value and value in chunk for value in sensitive_values) or bool(
        _GENERIC_SECRET_RE.search(chunk)
    )


def _validate_nested_untracked_archive(
    path: Path,
    *,
    sensitive_values: tuple[bytes, ...],
    limits: CheckpointPackageLimitsV2,
    stage: str,
) -> None:
    """Inspect the nested untracked-worktree archive before trusting it.

    Scanning only the outer ``untracked.tar.gz`` bytes cannot detect a token,
    traversal path, or link encoded inside the compressed stream.  The same
    bounded inspection therefore runs during container seal, host verification
    and restore-time revalidation.  It never extracts the nested archive.
    """

    if stage not in {"seal", "verify", "restore"}:
        raise ValueError("nested checkpoint archive stage is invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limits.max_file_bytes
        ):
            raise CheckpointDataPlaneError(stage, "nested_archive_unsafe")
        seen: set[str] = set()
        entry_count = 0
        total_bytes = 0
        with os.fdopen(os.dup(descriptor), "rb") as raw:
            with tarfile.open(fileobj=raw, mode="r|gz") as archive:
                for member in archive:
                    entry_count += 1
                    if entry_count > limits.max_files:
                        raise CheckpointDataPlaneError(
                            stage, "nested_archive_entry_limit",
                        )
                    name = member.name.rstrip("/")
                    relative = PurePosixPath(name)
                    if (
                        not name
                        or relative.is_absolute()
                        or any(part in {"", ".", ".."} for part in relative.parts)
                        or len(relative.parts) > limits.max_depth
                        or len(name.encode("utf-8")) > limits.max_path_bytes
                        or name in seen
                    ):
                        raise CheckpointDataPlaneError(
                            stage, "nested_archive_path_invalid",
                        )
                    seen.add(name)
                    if member.isdir():
                        if (
                            member.issym()
                            or member.islnk()
                            or stat.S_IMODE(member.mode) != 0o700
                        ):
                            raise CheckpointDataPlaneError(
                                stage, "nested_archive_member_invalid",
                            )
                        continue
                    if (
                        not member.isfile()
                        or member.issym()
                        or member.islnk()
                        or stat.S_IMODE(member.mode) != 0o600
                        or member.size < 0
                        or member.size > limits.max_file_bytes
                    ):
                        raise CheckpointDataPlaneError(
                            stage, "nested_archive_member_invalid",
                        )
                    total_bytes += member.size
                    if total_bytes > limits.max_total_bytes:
                        raise CheckpointDataPlaneError(
                            stage, "nested_archive_total_limit",
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise CheckpointDataPlaneError(
                            stage, "nested_archive_member_unreadable",
                        )
                    remaining = member.size
                    overlap = b""
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise CheckpointDataPlaneError(
                                stage, "nested_archive_member_truncated",
                            )
                        scan = overlap + chunk
                        if _contains_secret(scan, sensitive_values):
                            raise CheckpointDataPlaneError(
                                stage, "secret_detected",
                            )
                        overlap = scan[-512:]
                        remaining -= len(chunk)
                    if source.read(1):
                        raise CheckpointDataPlaneError(
                            stage, "nested_archive_member_oversized",
                        )
        after = os.fstat(descriptor)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise CheckpointDataPlaneError(stage, "nested_archive_changed")
    except CheckpointDataPlaneError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
        raise CheckpointDataPlaneError(stage, "nested_archive_invalid") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _copy_source_tree(
    source_root: Path,
    destination_root: Path,
    *,
    limits: CheckpointPackageLimitsV2,
    sensitive_values: tuple[bytes, ...],
) -> tuple[list[str], list[dict[str, object]], int]:
    """Copy a hostile mutable tree into new private inodes and hash it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        source_fd = os.open(source_root, flags)
    except OSError as exc:
        raise CheckpointDataPlaneError("capture", "source_open_failed") from exc
    budget = _CopyBudget(limits)
    directories: list[str] = []
    files: list[dict[str, object]] = []
    root_metadata = os.fstat(source_fd)
    if not stat.S_ISDIR(root_metadata.st_mode):
        os.close(source_fd)
        raise CheckpointDataPlaneError("capture", "source_not_directory")
    destination_root.mkdir(mode=0o700)
    destination_root.chmod(0o700)

    def copy_directory(
        descriptor: int,
        destination: Path,
        relative: PurePosixPath,
        depth: int,
    ) -> None:
        before = os.fstat(descriptor)
        if before.st_dev != root_metadata.st_dev:
            raise CheckpointDataPlaneError("seal", "filesystem_boundary")
        names = sorted(os.listdir(descriptor))
        for name in names:
            if name in {"", ".", ".."} or "/" in name or "\x00" in name:
                raise CheckpointDataPlaneError("seal", "unsafe_path")
            rel = relative / name
            rel_text = rel.as_posix()
            if len(rel_text.encode("utf-8")) > limits.max_path_bytes:
                raise CheckpointDataPlaneError("seal", "path_length_limit")
            try:
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise CheckpointDataPlaneError("seal", "source_changed") from exc
            if observed.st_dev != root_metadata.st_dev:
                raise CheckpointDataPlaneError("seal", "filesystem_boundary")
            target = destination / name
            if stat.S_ISDIR(observed.st_mode):
                budget.account_directory(depth=depth + 1, path=rel_text)
                if depth + 1 > limits.max_depth:
                    raise CheckpointDataPlaneError("seal", "depth_limit")
                try:
                    child_fd = os.open(name, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise CheckpointDataPlaneError("seal", "unsafe_directory") from exc
                try:
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (observed.st_dev, observed.st_ino)
                    ):
                        raise CheckpointDataPlaneError("seal", "source_changed")
                    target.mkdir(mode=0o700)
                    target.chmod(0o700)
                    directories.append(rel_text)
                    copy_directory(child_fd, target, rel, depth + 1)
                    _fsync_directory(target)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise CheckpointDataPlaneError("seal", "unsafe_file_type")
            budget.account_file(size=observed.st_size, depth=depth + 1, path=rel_text)
            read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            read_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
            try:
                input_fd = os.open(name, read_flags, dir_fd=descriptor)
            except OSError as exc:
                raise CheckpointDataPlaneError("seal", "unsafe_source_file") from exc
            output_fd: int | None = None
            try:
                opened = os.fstat(input_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    raise CheckpointDataPlaneError("seal", "source_changed")
                output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                output_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                output_fd = os.open(target, output_flags, 0o600)
                digest = hashlib.sha256()
                copied = 0
                overlap = b""
                while True:
                    chunk = os.read(input_fd, 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > observed.st_size or copied > limits.max_file_bytes:
                        raise CheckpointDataPlaneError("seal", "source_changed")
                    scan = overlap + chunk
                    if _contains_secret(scan, sensitive_values):
                        raise CheckpointDataPlaneError("seal", "secret_detected")
                    overlap = scan[-512:]
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(output_fd, view)
                        if written <= 0:
                            raise CheckpointDataPlaneError("seal", "copy_failed")
                        view = view[written:]
                if copied != observed.st_size:
                    raise CheckpointDataPlaneError("seal", "source_changed")
                os.fchmod(output_fd, 0o600)
                os.fsync(output_fd)
                after = os.fstat(input_fd)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    _metadata_fingerprint(opened) != _metadata_fingerprint(after)
                    or (current.st_dev, current.st_ino)
                    != (opened.st_dev, opened.st_ino)
                    or current.st_mtime_ns != opened.st_mtime_ns
                    or current.st_size != opened.st_size
                ):
                    raise CheckpointDataPlaneError("seal", "source_changed")
                if rel_text == "untracked.tar.gz":
                    _validate_nested_untracked_archive(
                        target,
                        sensitive_values=sensitive_values,
                        limits=limits,
                        stage="seal",
                    )
                files.append({
                    "path": rel_text,
                    "size": copied,
                    "sha256": digest.hexdigest(),
                    "mode": 0o600,
                })
            finally:
                if output_fd is not None:
                    os.close(output_fd)
                os.close(input_fd)
        if sorted(os.listdir(descriptor)) != names:
            raise CheckpointDataPlaneError("seal", "source_changed")
        after = os.fstat(descriptor)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise CheckpointDataPlaneError("seal", "source_changed")

    try:
        copy_directory(source_fd, destination_root, PurePosixPath(), 0)
    except BaseException:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise
    finally:
        os.close(source_fd)
    files.sort(key=lambda value: str(value["path"]))
    directories.sort()
    return directories, files, budget.total_bytes


def _manifest_for(
    request: CheckpointCaptureRequestV2,
    directories: list[str],
    files: list[dict[str, object]],
    total_bytes: int,
) -> dict[str, object]:
    return {
        "schema": EXPORT_SCHEMA_V2,
        "protocol_version": 2,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": request.checkpoint_abi,
        "checkpoint_id": request.checkpoint_id,
        "checkpoint_lineage_id": request.checkpoint_lineage_id,
        "snapshot_generation": request.snapshot_generation,
        "capture_id": request.capture_id,
        "identity_fingerprint": request.identity_fingerprint,
        "recovery_capability": request.recovery_capability,
        "native_state_schema": request.native_state_schema,
        "captured_at": request.captured_at,
        "capture_storage": "container_native",
        "directories": directories,
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _write_deterministic_archive(
    sealed_root: Path,
    manifest_bytes: bytes,
    destination: Path,
    *,
    directories: list[str],
    files: list[dict[str, object]],
) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_private_directory(destination.parent, stage="seal")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w:", format=tarfile.PAX_FORMAT) as archive:
                    manifest_info = tarfile.TarInfo(MANIFEST_NAME)
                    manifest_info.size = len(manifest_bytes)
                    manifest_info.mode = 0o600
                    manifest_info.uid = manifest_info.gid = 0
                    manifest_info.uname = manifest_info.gname = ""
                    manifest_info.mtime = 0
                    manifest_info.pax_headers = {}
                    import io
                    archive.addfile(manifest_info, io.BytesIO(manifest_bytes))

                    payload_info = tarfile.TarInfo(PAYLOAD_ROOT)
                    payload_info.type = tarfile.DIRTYPE
                    payload_info.mode = 0o700
                    payload_info.uid = payload_info.gid = 0
                    payload_info.uname = payload_info.gname = ""
                    payload_info.mtime = 0
                    payload_info.pax_headers = {}
                    archive.addfile(payload_info)
                    for relative in directories:
                        info = tarfile.TarInfo(f"{PAYLOAD_ROOT}/{relative}")
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o700
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        info.pax_headers = {}
                        archive.addfile(info)
                    for entry in files:
                        relative = str(entry["path"])
                        path = sealed_root / Path(relative)
                        info = tarfile.TarInfo(f"{PAYLOAD_ROOT}/{relative}")
                        info.size = int(entry["size"])
                        info.mode = 0o600
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        info.pax_headers = {}
                        with path.open("rb") as source:
                            archive.addfile(info, source)
            raw.flush()
            os.fsync(raw.fileno())
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


@contextmanager
def _container_seal_lock(destination_archive: Path):
    """Serialize one final export without a crash-sticky lock directory."""

    lock_path = destination_archive.with_name(
        f".{destination_archive.name}.lock"
    )
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CheckpointDataPlaneError("seal", "seal_lock_failed") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            lock_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise CheckpointDataPlaneError("seal", "seal_lock_unsafe")
        if os.name == "posix" and metadata.st_uid != os.getuid():
            raise CheckpointDataPlaneError("seal", "seal_lock_unsafe")
        os.fchmod(descriptor, 0o600)
        if os.name == "nt":  # pragma: no cover - container runtime is POSIX
            import msvcrt
            if metadata.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":  # pragma: no cover - container runtime is POSIX
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def seal_checkpoint_export_v2(
    source_root: Path,
    destination_archive: Path,
    request: CheckpointCaptureRequestV2,
    *,
    sensitive_values: Iterable[str | bytes] = (),
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
    container_export_root: Path = Path("/run/dradar-checkpoint-v2"),
) -> ContainerSealedExportV2:
    """Container-side reference capture/seal/export implementation.

    The caller must place both ``source_root`` and ``destination_archive`` on
    the container's native filesystem.  The function copies mutable adapter
    output into fresh private inodes, builds a deterministic archive, and only
    then atomically exposes the final export path.
    """

    request.validate()
    source_root = Path(source_root)
    destination_archive = Path(destination_archive)
    container_export_root = Path(container_export_root)
    _assert_private_directory(container_export_root, stage="seal")
    try:
        destination_archive.relative_to(container_export_root)
    except ValueError as exc:
        raise CheckpointDataPlaneError("seal", "export_outside_container_storage") from exc
    _assert_private_directory(destination_archive.parent, stage="seal")
    try:
        if source_root.lstat().st_dev != destination_archive.parent.lstat().st_dev:
            raise CheckpointDataPlaneError("seal", "container_storage_cross_device")
    except OSError as exc:
        raise CheckpointDataPlaneError("seal", "container_storage_unavailable") from exc
    needles = tuple(
        value if isinstance(value, bytes) else value.encode("utf-8")
        for value in sensitive_values
        if isinstance(value, (str, bytes)) and len(value) >= 8
    )
    with _container_seal_lock(destination_archive):
        if destination_archive.exists() or destination_archive.is_symlink():
            return _recover_existing_sealed_export_v2(
                destination_archive,
                request,
                sensitive_values=needles,
                limits=limits,
                container_export_root=container_export_root,
            )
        # The lock provides live-writer exclusion.  A unique transaction root
        # means kill -9/power loss cannot leave a pathname that blocks the
        # exact capture on restart; abandoned roots remain inert evidence.
        seal_parent = destination_archive.parent / (
            f".sealed-{request.capture_id}-{uuid.uuid4().hex}.part"
        )
        payload = seal_parent / PAYLOAD_ROOT
        seal_parent.mkdir(mode=0o700, parents=False)
        seal_parent.chmod(0o700)
        try:
            directories, files, total_bytes = _copy_source_tree(
                source_root,
                payload,
                limits=limits,
                sensitive_values=needles,
            )
            manifest = _manifest_for(request, directories, files, total_bytes)
            manifest_bytes = _canonical_json(manifest)
            if len(manifest_bytes) > limits.max_manifest_bytes:
                raise CheckpointDataPlaneError("seal", "manifest_size_limit")
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            _write_deterministic_archive(
                payload,
                manifest_bytes,
                destination_archive,
                directories=directories,
                files=files,
            )
            archive_sha256, archive_size = _sha256_file(
                destination_archive,
                max_bytes=limits.max_archive_bytes,
            )
            return ContainerSealedExportV2(
                capture_id=request.capture_id,
                remote_path=destination_archive.as_posix(),
                archive_sha256=archive_sha256,
                archive_size=archive_size,
                manifest_sha256=manifest_sha256,
                capture_storage="container_native",
            )
        except BaseException:
            destination_archive.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(seal_parent, ignore_errors=True)


def _recover_existing_sealed_export_v2(
    archive_path: Path,
    request: CheckpointCaptureRequestV2,
    *,
    sensitive_values: tuple[bytes, ...],
    limits: CheckpointPackageLimitsV2,
    container_export_root: Path,
) -> ContainerSealedExportV2:
    """Resume only an exact, fully verified atomic seal from a dead writer."""

    try:
        archive_sha256, archive_size = _sha256_file(
            archive_path, max_bytes=limits.max_archive_bytes,
        )
        with archive_path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as zipped:
                with tarfile.open(fileobj=zipped, mode="r|") as archive:
                    first = next(iter(archive), None)
                    if (
                        first is None
                        or first.name.rstrip("/") != MANIFEST_NAME
                        or not first.isfile()
                        or first.issym()
                        or first.islnk()
                        or first.size > limits.max_manifest_bytes
                        or stat.S_IMODE(first.mode) != 0o600
                    ):
                        raise CheckpointDataPlaneError(
                            "seal", "existing_export_manifest_invalid",
                        )
                    source = archive.extractfile(first)
                    if source is None:
                        raise CheckpointDataPlaneError(
                            "seal", "existing_export_manifest_invalid",
                        )
                    manifest_bytes = source.read(limits.max_manifest_bytes + 1)
                    if len(manifest_bytes) != first.size:
                        raise CheckpointDataPlaneError(
                            "seal", "existing_export_manifest_invalid",
                        )
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _parse_manifest(manifest_bytes, request, limits=limits)
        recovered = ContainerSealedExportV2(
            capture_id=request.capture_id,
            remote_path=archive_path.as_posix(),
            archive_sha256=archive_sha256,
            archive_size=archive_size,
            manifest_sha256=manifest_sha256,
            capture_storage="container_native",
        )
        verification_root = container_export_root / (
            f".recover-{request.capture_id}-{uuid.uuid4().hex}.part"
        )
        try:
            _extract_verified_archive(
                archive_path,
                verification_root,
                request,
                recovered,
                limits=limits,
                sensitive_values=sensitive_values,
            )
        finally:
            shutil.rmtree(verification_root, ignore_errors=True)
        return recovered
    except CheckpointDataPlaneError:
        raise
    except (OSError, EOFError, tarfile.TarError, gzip.BadGzipFile) as exc:
        raise CheckpointDataPlaneError(
            "seal", "existing_export_invalid",
        ) from exc


def _assert_private_directory(path: Path, *, stage: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError(stage, "storage_unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CheckpointDataPlaneError(stage, "unsafe_storage_root")
    if os.name == "posix":
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CheckpointDataPlaneError(stage, "unsafe_storage_permissions")


def _ensure_private_directory(path: Path, *, stage: str) -> None:
    if path.exists() or path.is_symlink():
        _assert_private_directory(path, stage=stage)
        return
    try:
        path.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        # Another process may have created the same storage level after our
        # initial lstat.  Never chmod that raced path (it could have been
        # replaced by a symlink); the authoritative postcondition check below
        # accepts only our uid's non-symlink private directory.
        pass
    except OSError as exc:
        raise CheckpointDataPlaneError(stage, "storage_create_failed") from exc
    _assert_private_directory(path, stage=stage)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_relative_path(value: object, *, kind: str, limits: CheckpointPackageLimitsV2) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointDataPlaneError("verify", f"invalid_{kind}_path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != value
        or value.startswith("./")
        or len(value.encode("utf-8")) > limits.max_path_bytes
    ):
        raise CheckpointDataPlaneError("verify", f"invalid_{kind}_path")
    return value


def _parse_manifest(
    raw: bytes,
    request: CheckpointCaptureRequestV2,
    *,
    limits: CheckpointPackageLimitsV2,
) -> tuple[dict[str, object], dict[str, dict[str, object]], set[str]]:
    if len(raw) > limits.max_manifest_bytes:
        raise CheckpointDataPlaneError("verify", "manifest_size_limit")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError("verify", "manifest_invalid") from exc
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise CheckpointDataPlaneError("verify", "manifest_fields_invalid")
    expected = {
        "schema": EXPORT_SCHEMA_V2,
        "protocol_version": 2,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": request.checkpoint_abi,
        "checkpoint_id": request.checkpoint_id,
        "checkpoint_lineage_id": request.checkpoint_lineage_id,
        "snapshot_generation": request.snapshot_generation,
        "capture_id": request.capture_id,
        "identity_fingerprint": request.identity_fingerprint,
        "recovery_capability": request.recovery_capability,
        "native_state_schema": request.native_state_schema,
        "captured_at": request.captured_at,
        "capture_storage": "container_native",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise CheckpointDataPlaneError("verify", "manifest_identity_mismatch")
    raw_directories = value.get("directories")
    raw_files = value.get("files")
    if not isinstance(raw_directories, list) or not isinstance(raw_files, list):
        raise CheckpointDataPlaneError("verify", "manifest_inventory_invalid")
    directories: set[str] = set()
    for raw_directory in raw_directories:
        directory = _safe_relative_path(raw_directory, kind="directory", limits=limits)
        if directory in directories:
            raise CheckpointDataPlaneError("verify", "duplicate_directory")
        directories.add(directory)
        if len(directories) > limits.max_files:
            raise CheckpointDataPlaneError("verify", "entry_count_limit")
    files: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for raw_file in raw_files:
        if not isinstance(raw_file, dict) or set(raw_file) != {"path", "size", "sha256", "mode"}:
            raise CheckpointDataPlaneError("verify", "manifest_file_invalid")
        path = _safe_relative_path(raw_file.get("path"), kind="file", limits=limits)
        size = raw_file.get("size")
        digest = raw_file.get("sha256")
        mode = raw_file.get("mode")
        if (
            path in files
            or path in directories
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > limits.max_file_bytes
            or not isinstance(digest, str)
            or _DIGEST_RE.fullmatch(digest) is None
            or mode != 0o600
        ):
            raise CheckpointDataPlaneError("verify", "manifest_file_invalid")
        files[path] = raw_file
        if len(files) + len(directories) > limits.max_files:
            raise CheckpointDataPlaneError("verify", "entry_count_limit")
        total_bytes += size
        if total_bytes > limits.max_total_bytes:
            raise CheckpointDataPlaneError("verify", "total_size_limit")
    if (
        len(files) != value.get("file_count")
        or total_bytes != value.get("total_bytes")
        or len(files) > limits.max_files
    ):
        raise CheckpointDataPlaneError("verify", "manifest_totals_invalid")
    inventory = directories | set(files)
    for path in inventory:
        parent = PurePosixPath(path).parent
        while parent != PurePosixPath("."):
            if parent.as_posix() not in directories:
                raise CheckpointDataPlaneError("verify", "missing_parent_directory")
            parent = parent.parent
    return value, files, directories


def _extract_verified_archive(
    archive_path: Path,
    destination: Path,
    request: CheckpointCaptureRequestV2,
    export: ContainerSealedExportV2,
    *,
    limits: CheckpointPackageLimitsV2,
    sensitive_values: tuple[bytes, ...] = (),
) -> tuple[str, int, dict[str, object]]:
    archive_sha256, archive_size = _sha256_file(
        archive_path, max_bytes=limits.max_archive_bytes,
    )
    if archive_sha256 != export.archive_sha256 or archive_size != export.archive_size:
        raise CheckpointDataPlaneError("verify", "archive_digest_mismatch")
    if destination.exists() or destination.is_symlink():
        raise CheckpointDataPlaneError("verify", "verification_stage_exists")
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)
    payload_destination = destination / PAYLOAD_ROOT
    files: dict[str, dict[str, object]] | None = None
    directories: set[str] | None = None
    seen: set[str] = set()
    try:
        with archive_path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as zipped:
                with tarfile.open(fileobj=zipped, mode="r|") as archive:
                    for index, member in enumerate(archive):
                        name = member.name.rstrip("/")
                        if index == 0:
                            if (
                                name != MANIFEST_NAME
                                or not member.isfile()
                                or member.issym()
                                or member.islnk()
                                or member.size > limits.max_manifest_bytes
                                or stat.S_IMODE(member.mode) != 0o600
                            ):
                                raise CheckpointDataPlaneError("verify", "manifest_member_invalid")
                            source = archive.extractfile(member)
                            if source is None:
                                raise CheckpointDataPlaneError("verify", "manifest_unreadable")
                            manifest_bytes = source.read(limits.max_manifest_bytes + 1)
                            if len(manifest_bytes) != member.size:
                                raise CheckpointDataPlaneError("verify", "manifest_truncated")
                            if hashlib.sha256(manifest_bytes).hexdigest() != export.manifest_sha256:
                                raise CheckpointDataPlaneError("verify", "manifest_digest_mismatch")
                            manifest, files, directories = _parse_manifest(
                                manifest_bytes, request, limits=limits,
                            )
                            manifest_path = destination / MANIFEST_NAME
                            manifest_path.write_bytes(manifest_bytes)
                            manifest_path.chmod(0o600)
                            continue
                        if files is None or directories is None:
                            raise CheckpointDataPlaneError("verify", "manifest_not_first")
                        if name == PAYLOAD_ROOT:
                            if (
                                not member.isdir()
                                or member.issym()
                                or member.islnk()
                                or stat.S_IMODE(member.mode) != 0o700
                                or name in seen
                            ):
                                raise CheckpointDataPlaneError("verify", "payload_root_invalid")
                            payload_destination.mkdir(mode=0o700)
                            payload_destination.chmod(0o700)
                            seen.add(name)
                            continue
                        prefix = PAYLOAD_ROOT + "/"
                        if not name.startswith(prefix):
                            raise CheckpointDataPlaneError("verify", "unexpected_archive_member")
                        relative = name[len(prefix):]
                        _safe_relative_path(relative, kind="archive", limits=limits)
                        if name in seen:
                            raise CheckpointDataPlaneError("verify", "duplicate_archive_member")
                        seen.add(name)
                        target = payload_destination.joinpath(*PurePosixPath(relative).parts)
                        if relative in directories:
                            if (
                                not member.isdir()
                                or member.issym()
                                or member.islnk()
                                or stat.S_IMODE(member.mode) != 0o700
                            ):
                                raise CheckpointDataPlaneError("verify", "directory_member_invalid")
                            target.mkdir(mode=0o700)
                            target.chmod(0o700)
                            continue
                        expected = files.get(relative)
                        if expected is None:
                            raise CheckpointDataPlaneError("verify", "unexpected_archive_member")
                        if (
                            not member.isfile()
                            or member.issym()
                            or member.islnk()
                            or member.size != expected["size"]
                            or stat.S_IMODE(member.mode) != 0o600
                        ):
                            raise CheckpointDataPlaneError("verify", "file_member_invalid")
                        source = archive.extractfile(member)
                        if source is None:
                            raise CheckpointDataPlaneError("verify", "file_member_unreadable")
                        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                        descriptor = os.open(target, flags, 0o600)
                        try:
                            digest = hashlib.sha256()
                            remaining = int(expected["size"])
                            overlap = b""
                            while remaining:
                                chunk = source.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise CheckpointDataPlaneError("verify", "file_member_truncated")
                                scan = overlap + chunk
                                if _contains_secret(scan, sensitive_values):
                                    raise CheckpointDataPlaneError("verify", "secret_detected")
                                overlap = scan[-512:]
                                digest.update(chunk)
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(descriptor, view)
                                    if written <= 0:
                                        raise CheckpointDataPlaneError("verify", "file_write_failed")
                                    view = view[written:]
                                remaining -= len(chunk)
                            if source.read(1):
                                raise CheckpointDataPlaneError("verify", "file_member_oversized")
                            if digest.hexdigest() != expected["sha256"]:
                                raise CheckpointDataPlaneError("verify", "file_digest_mismatch")
                            os.fchmod(descriptor, 0o600)
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                        if relative == "untracked.tar.gz":
                            _validate_nested_untracked_archive(
                                target,
                                sensitive_values=sensitive_values,
                                limits=limits,
                                stage="verify",
                            )
        if files is None or directories is None:
            raise CheckpointDataPlaneError("verify", "manifest_missing")
        expected_members = {
            PAYLOAD_ROOT,
            *(f"{PAYLOAD_ROOT}/{path}" for path in directories),
            *(f"{PAYLOAD_ROOT}/{path}" for path in files),
        }
        if seen != expected_members:
            raise CheckpointDataPlaneError("verify", "archive_inventory_mismatch")
        _fsync_tree(destination)
        return archive_sha256, archive_size, manifest
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _fsync_tree(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise CheckpointDataPlaneError("verify", "published_tree_unsafe")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise CheckpointDataPlaneError("verify", "published_tree_unsafe")
        _fsync_directory(current_path)


@contextmanager
def _publication_lock(checkpoint_root: Path):
    """Serialize generation publication without a crash-sticky lock file."""

    lock_path = checkpoint_root / "PUBLICATION.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CheckpointDataPlaneError("publish", "publication_lock_failed") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CheckpointDataPlaneError("publish", "publication_lock_unsafe")
        if os.name == "posix" and metadata.st_uid != os.getuid():
            raise CheckpointDataPlaneError("publish", "publication_lock_unsafe")
        os.fchmod(descriptor, 0o600)
        if os.name == "nt":
            import msvcrt
            if metadata.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_current(checkpoint_root: Path) -> dict[str, object] | None:
    path = checkpoint_root / "CURRENT"
    if not path.exists() and not path.is_symlink():
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointDataPlaneError("publish", "current_pointer_unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > 2048
            or (os.name == "posix" and metadata.st_uid != os.getuid())
        ):
            raise CheckpointDataPlaneError("publish", "current_pointer_unsafe")
        raw = os.read(descriptor, 2049)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError("publish", "current_pointer_invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"generation", "directory", "manifest_sha256", "authoritative"}
        or not isinstance(value.get("generation"), int)
        or isinstance(value.get("generation"), bool)
        or int(value["generation"]) < 0
        or value.get("directory") != f"generation-{int(value['generation']):020d}"
        or not isinstance(value.get("manifest_sha256"), str)
        or _DIGEST_RE.fullmatch(str(value["manifest_sha256"])) is None
        or not isinstance(value.get("authoritative"), bool)
    ):
        raise CheckpointDataPlaneError("publish", "current_pointer_invalid")
    return value


def next_shadow_generation_v2(storage_root: Path, checkpoint_id: str) -> int:
    """Return a collision-free local generation without trusting CURRENT.

    A crash can publish a generation just before updating CURRENT.  Scanning
    only strictly named directory entries therefore avoids reusing that
    generation on restart.  Contents are not interpreted here; malformed or
    hostile entries remain preserved for diagnosis and merely advance the
    local counter.
    """

    if _IDENTIFIER_RE.fullmatch(checkpoint_id) is None:
        raise CheckpointDataPlaneError("capture", "invalid_checkpoint_id")
    root = Path(storage_root) / "checkpoints" / checkpoint_id / "generations"
    if not root.exists() and not root.is_symlink():
        return 1
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError("capture", "storage_unavailable") from exc
    if (
        root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or (os.name == "posix" and (
            metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ))
    ):
        raise CheckpointDataPlaneError("capture", "unsafe_storage_permissions")
    maximum = 0
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise CheckpointDataPlaneError("capture", "storage_unavailable") from exc
    for path in entries:
        match = re.fullmatch(r"generation-([0-9]{20})", path.name)
        if match is None:
            continue
        maximum = max(maximum, int(match.group(1)))
    if maximum >= 2**31 - 1:
        raise CheckpointDataPlaneError("capture", "generation_limit")
    return maximum + 1


def publish_checkpoint_export_v2(
    archive_path: Path,
    storage_root: Path,
    request: CheckpointCaptureRequestV2,
    export: ContainerSealedExportV2,
    *,
    authoritative: bool,
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
) -> PublishedCheckpointV2:
    """Verify an untrusted export and atomically publish one generation."""

    request.validate()
    export.validate(request)
    storage_root = Path(storage_root)
    _ensure_private_directory(storage_root, stage="publish")
    checkpoint_root = storage_root / request.checkpoint_id
    generations = checkpoint_root / "generations"
    _ensure_private_directory(checkpoint_root, stage="publish")
    _ensure_private_directory(generations, stage="publish")
    target = generations / f"generation-{request.snapshot_generation:020d}"
    # The capture ID is content identity, not a reusable staging pathname.
    # A hard process crash cannot run our finally block, so a unique private
    # transaction directory lets the same exact capture retry without being
    # permanently blocked by an incomplete predecessor.
    incoming = checkpoint_root / (
        f".incoming-{request.capture_id}-{uuid.uuid4().hex}.part"
    )
    target_existed = False
    if incoming.exists() or incoming.is_symlink():
        raise CheckpointDataPlaneError("publish", "incoming_stage_exists")
    try:
        archive_sha256, archive_size, manifest = _extract_verified_archive(
            Path(archive_path), incoming, request, export, limits=limits,
        )
        retained_archive = incoming / "export.tar.gz"
        _copy_regular_file_snapshot(
            Path(archive_path),
            retained_archive,
            expected_sha256=archive_sha256,
            expected_size=archive_size,
            max_bytes=limits.max_archive_bytes,
            stage="publish",
        )
        receipt = {
            "schema": "dradar-checkpoint-publication-v2",
            "checkpoint_id": request.checkpoint_id,
            "snapshot_generation": request.snapshot_generation,
            "capture_id": request.capture_id,
            "manifest_sha256": export.manifest_sha256,
            "archive_sha256": archive_sha256,
            "authoritative": bool(authoritative),
        }
        receipt_bytes = _canonical_json(receipt)
        receipt_path = incoming / "publication.json"
        receipt_path.write_bytes(receipt_bytes)
        receipt_path.chmod(0o600)
        with receipt_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(incoming)
        selected = False
        with _publication_lock(checkpoint_root):
            if target.exists() or target.is_symlink():
                target_existed = True
                existing = target / "publication.json"
                try:
                    if existing.read_bytes() != receipt_bytes:
                        raise CheckpointDataPlaneError("publish", "generation_conflict")
                finally:
                    shutil.rmtree(incoming, ignore_errors=True)
            else:
                os.replace(incoming, target)
                _fsync_directory(generations)
            current = _read_current(checkpoint_root)
            if (
                current is None
                or int(current["generation"]) <= request.snapshot_generation
            ):
                selected = True
                current_payload = _canonical_json({
                    "generation": request.snapshot_generation,
                    "directory": target.name,
                    "manifest_sha256": export.manifest_sha256,
                    "authoritative": bool(authoritative),
                })
                current_temp = checkpoint_root / (
                    f".CURRENT.{request.capture_id}.{uuid.uuid4().hex}.part"
                )
                try:
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    descriptor = os.open(current_temp, flags, 0o600)
                    try:
                        view = memoryview(current_payload)
                        while view:
                            written = os.write(descriptor, view)
                            if written <= 0:
                                raise CheckpointDataPlaneError(
                                    "publish", "current_write_failed",
                                )
                            view = view[written:]
                        os.fchmod(descriptor, 0o600)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    os.replace(current_temp, checkpoint_root / "CURRENT")
                    _fsync_directory(checkpoint_root)
                finally:
                    try:
                        current_temp.unlink(missing_ok=True)
                    except OSError:
                        pass
        published = PublishedCheckpointV2(
            checkpoint_id=request.checkpoint_id,
            snapshot_generation=request.snapshot_generation,
            capture_id=request.capture_id,
            root=target,
            payload_root=target / PAYLOAD_ROOT,
            archive_path=target / "export.tar.gz",
            manifest_sha256=export.manifest_sha256,
            archive_sha256=archive_sha256,
            archive_bytes=archive_size,
            file_count=int(manifest["file_count"]),
            payload_bytes=int(manifest["total_bytes"]),
            authoritative=bool(authoritative),
            selected=selected,
        )
        if target_existed:
            # A replay must not trust a matching receipt alone.  A previous
            # process may have crashed after publication, or local material
            # may have drifted before the retry.  Re-hash the exact existing
            # generation before reporting the idempotent publication as
            # sealed; never replace it with the retry's incoming bytes.
            revalidate_published_checkpoint_v2(
                published,
                expected_identity_fingerprint=request.identity_fingerprint,
                expected_checkpoint_abi=request.checkpoint_abi,
                limits=limits,
            )
        return published
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise


def _prune_shadow_generations_v2(
    published: PublishedCheckpointV2,
    *,
    keep: int,
) -> int:
    """Keep recent non-authoritative generations under the publication lock.

    The just-published generation is always protected, including an
    out-of-order diagnostic replay.  A later publication naturally removes it
    once it is no longer among the newest retained generations.  Unknown,
    malformed, symlinked, or authoritative entries are preserved rather than
    guessed about.
    """

    if published.authoritative:
        return 0
    checkpoint_root = published.root.parent.parent
    generations = published.root.parent
    removed = 0
    with _publication_lock(checkpoint_root):
        current = _read_current(checkpoint_root)
        protected_names = {published.root.name}
        if current is not None:
            protected_names.add(str(current["directory"]))
        candidates: list[
            tuple[int, Path, tuple[int, int, int, int], bytes]
        ] = []
        try:
            entries = list(generations.iterdir())
        except OSError:
            return 0
        for path in entries:
            match = re.fullmatch(r"generation-([0-9]{20})", path.name)
            if match is None:
                continue
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if (
                path.is_symlink()
                or not stat.S_ISDIR(metadata.st_mode)
                or (os.name == "posix" and (
                    metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o077
                ))
            ):
                continue
            receipt_path = path / "publication.json"
            try:
                receipt_metadata = receipt_path.lstat()
                receipt_bytes = receipt_path.read_bytes()
                receipt = json.loads(receipt_bytes)
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (
                not stat.S_ISREG(receipt_metadata.st_mode)
                or receipt_path.is_symlink()
                or receipt_metadata.st_nlink != 1
                or not isinstance(receipt, dict)
                or receipt.get("schema") != "dradar-checkpoint-publication-v2"
                or receipt.get("checkpoint_id") != published.checkpoint_id
                or receipt.get("snapshot_generation") != int(match.group(1))
                or receipt.get("authoritative") is not False
            ):
                continue
            candidates.append((
                int(match.group(1)),
                path,
                (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime_ns,
                    metadata.st_mode,
                ),
                receipt_bytes,
            ))
        newest_names = {
            path.name
            for _, path, _, _ in sorted(candidates, reverse=True)[:keep]
        }
        protected_names.update(newest_names)
        for _, path, expected_metadata, expected_receipt in sorted(candidates):
            if path.name in protected_names:
                continue
            try:
                observed = path.lstat()
                if (
                    path.is_symlink()
                    or not stat.S_ISDIR(observed.st_mode)
                    or (
                        observed.st_dev,
                        observed.st_ino,
                        observed.st_mtime_ns,
                        observed.st_mode,
                    ) != expected_metadata
                    or (path / "publication.json").read_bytes()
                    != expected_receipt
                ):
                    continue
                quarantine = generations / (
                    f".prune-{path.name}-{uuid.uuid4().hex}"
                )
                os.replace(path, quarantine)
                _fsync_directory(generations)
                shutil.rmtree(quarantine)
            except OSError:
                continue
            removed += 1
        if removed:
            _fsync_directory(generations)
    return removed


def revalidate_published_checkpoint_v2(
    published: PublishedCheckpointV2,
    *,
    expected_identity_fingerprint: str,
    expected_checkpoint_abi: str,
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
) -> dict[str, object]:
    """Re-hash a stored generation immediately before any restore attempt."""

    root = Path(published.root)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError("restore", "published_snapshot_missing") from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root.is_symlink()
        or (os.name == "posix" and (
            root_metadata.st_uid != os.getuid()
            or stat.S_IMODE(root_metadata.st_mode) & 0o077
        ))
    ):
        raise CheckpointDataPlaneError("restore", "published_snapshot_unsafe")
    receipt_path = root / "publication.json"
    manifest_path = root / MANIFEST_NAME
    archive_path = root / "export.tar.gz"
    for path, code, max_bytes in (
        (receipt_path, "publication_receipt_invalid", 4096),
        (manifest_path, "published_manifest_invalid", limits.max_manifest_bytes),
        (archive_path, "published_archive_invalid", limits.max_archive_bytes),
    ):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CheckpointDataPlaneError("restore", code) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_nlink != 1
            or metadata.st_size > max_bytes
            or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
        ):
            raise CheckpointDataPlaneError("restore", code)
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError("restore", "publication_receipt_invalid") from exc
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema", "checkpoint_id", "snapshot_generation", "capture_id",
            "manifest_sha256", "archive_sha256", "authoritative",
        }
        or receipt.get("schema") != "dradar-checkpoint-publication-v2"
        or receipt.get("checkpoint_id") != published.checkpoint_id
        or receipt.get("snapshot_generation") != published.snapshot_generation
        or receipt.get("capture_id") != published.capture_id
        or receipt.get("manifest_sha256") != published.manifest_sha256
        or receipt.get("archive_sha256") != published.archive_sha256
        or receipt.get("authoritative") is not published.authoritative
    ):
        raise CheckpointDataPlaneError("restore", "publication_receipt_invalid")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != published.manifest_sha256:
        raise CheckpointDataPlaneError("restore", "published_manifest_digest_mismatch")
    archive_sha256, archive_size = _sha256_file(
        archive_path, max_bytes=limits.max_archive_bytes,
    )
    if (
        archive_sha256 != published.archive_sha256
        or archive_size != published.archive_bytes
    ):
        raise CheckpointDataPlaneError("restore", "published_archive_digest_mismatch")
    try:
        preliminary = json.loads(manifest_bytes)
        request = CheckpointCaptureRequestV2(
            checkpoint_id=preliminary["checkpoint_id"],
            checkpoint_lineage_id=preliminary["checkpoint_lineage_id"],
            snapshot_generation=preliminary["snapshot_generation"],
            capture_id=preliminary["capture_id"],
            identity_fingerprint=preliminary["identity_fingerprint"],
            checkpoint_abi=preliminary["checkpoint_abi"],
            recovery_capability=preliminary["recovery_capability"],
            native_state_schema=preliminary["native_state_schema"],
            captured_at=preliminary["captured_at"],
        )
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError("restore", "published_manifest_invalid") from exc
    try:
        request.validate()
    except CheckpointDataPlaneError as exc:
        raise CheckpointDataPlaneError("restore", "published_manifest_invalid") from exc
    if (
        request.checkpoint_id != published.checkpoint_id
        or request.snapshot_generation != published.snapshot_generation
        or request.capture_id != published.capture_id
        or request.identity_fingerprint != expected_identity_fingerprint
        or request.checkpoint_abi != expected_checkpoint_abi
    ):
        raise CheckpointDataPlaneError("restore", "restore_identity_mismatch")
    try:
        manifest, files, directories = _parse_manifest(
            manifest_bytes, request, limits=limits,
        )
    except CheckpointDataPlaneError as exc:
        raise CheckpointDataPlaneError("restore", "published_manifest_invalid") from exc
    payload_root = root / PAYLOAD_ROOT
    try:
        payload_metadata = payload_root.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError("restore", "published_payload_missing") from exc
    if (
        not stat.S_ISDIR(payload_metadata.st_mode)
        or payload_root.is_symlink()
        or (os.name == "posix" and stat.S_IMODE(payload_metadata.st_mode) != 0o700)
    ):
        raise CheckpointDataPlaneError("restore", "published_payload_unsafe")
    seen_directories: set[str] = set()
    seen_files: set[str] = set()
    for current, directory_names, file_names in os.walk(
        payload_root, topdown=True, followlinks=False,
    ):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            relative = path.relative_to(payload_root).as_posix()
            metadata = path.lstat()
            if (
                relative not in directories
                or not stat.S_ISDIR(metadata.st_mode)
                or path.is_symlink()
                or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700)
            ):
                raise CheckpointDataPlaneError("restore", "published_payload_unsafe")
            seen_directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(payload_root).as_posix()
            expected = files.get(relative)
            metadata = path.lstat()
            if (
                expected is None
                or not stat.S_ISREG(metadata.st_mode)
                or path.is_symlink()
                or metadata.st_nlink != 1
                or metadata.st_size != expected["size"]
                or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
            ):
                raise CheckpointDataPlaneError("restore", "published_payload_unsafe")
            digest, size = _sha256_file(path, max_bytes=limits.max_file_bytes)
            if digest != expected["sha256"] or size != expected["size"]:
                raise CheckpointDataPlaneError("restore", "published_payload_digest_mismatch")
            if relative == "untracked.tar.gz":
                _validate_nested_untracked_archive(
                    path,
                    sensitive_values=(),
                    limits=limits,
                    stage="restore",
                )
            seen_files.add(relative)
    if seen_directories != directories or seen_files != set(files):
        raise CheckpointDataPlaneError("restore", "published_payload_inventory_mismatch")
    return manifest


def load_exact_published_checkpoint_v2(
    storage_root: Path,
    *,
    checkpoint_id: str,
    checkpoint_lineage_id: str,
    snapshot_generation: int,
    capture_id: str,
    manifest_sha256: str,
    expected_identity_fingerprint: str,
    expected_checkpoint_core_abi: str,
    expected_checkpoint_abi: str,
    expected_recovery_capability: str,
    expected_native_state_schema: str | None,
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
) -> PublishedCheckpointV2:
    """Load and fully revalidate one exact server-selected generation.

    No ``CURRENT`` pointer or neighbouring directory participates.  This is
    the restart-safe counterpart of the in-memory publication receipt: all
    identity supplied by the server is matched against both the publication
    receipt and the sealed manifest before any restore adapter is invoked.
    """

    if (
        _IDENTIFIER_RE.fullmatch(checkpoint_id) is None
        or _IDENTIFIER_RE.fullmatch(checkpoint_lineage_id) is None
        or _IDENTIFIER_RE.fullmatch(capture_id) is None
        or not isinstance(snapshot_generation, int)
        or isinstance(snapshot_generation, bool)
        or snapshot_generation < 0
        or _DIGEST_RE.fullmatch(manifest_sha256) is None
        or _DIGEST_RE.fullmatch(expected_identity_fingerprint) is None
        or expected_checkpoint_core_abi != CHECKPOINT_CORE_ABI_V2
        or _ABI_RE.fullmatch(expected_checkpoint_abi) is None
        or expected_recovery_capability not in {
            "NATIVE_VALID", "WORKSPACE_ONLY",
        }
        or (
            expected_native_state_schema is not None
            and _NATIVE_SCHEMA_RE.fullmatch(expected_native_state_schema) is None
        )
    ):
        raise CheckpointDataPlaneError(
            "restore", "selected_generation_descriptor_invalid",
        )
    storage_root = Path(os.path.abspath(storage_root))
    checkpoints = storage_root / "checkpoints"
    checkpoint_root = checkpoints / checkpoint_id
    generations = checkpoint_root / "generations"
    for directory in (
        storage_root, checkpoints, checkpoint_root, generations,
    ):
        _assert_private_directory(directory, stage="restore")
    target = generations / f"generation-{snapshot_generation:020d}"
    receipt_path = target / "publication.json"
    try:
        receipt_metadata = receipt_path.lstat()
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise CheckpointDataPlaneError(
            "restore", "publication_receipt_invalid",
        ) from exc
    if (
        receipt_path.is_symlink()
        or not stat.S_ISREG(receipt_metadata.st_mode)
        or receipt_metadata.st_nlink != 1
        or len(receipt_bytes) > 4096
        or (
            os.name == "posix"
            and stat.S_IMODE(receipt_metadata.st_mode) != 0o600
        )
    ):
        raise CheckpointDataPlaneError(
            "restore", "publication_receipt_invalid",
        )
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError(
            "restore", "publication_receipt_invalid",
        ) from exc
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema", "checkpoint_id", "snapshot_generation", "capture_id",
            "manifest_sha256", "archive_sha256", "authoritative",
        }
        or receipt.get("schema") != "dradar-checkpoint-publication-v2"
        or receipt.get("checkpoint_id") != checkpoint_id
        or receipt.get("snapshot_generation") != snapshot_generation
        or receipt.get("capture_id") != capture_id
        or receipt.get("manifest_sha256") != manifest_sha256
        or _DIGEST_RE.fullmatch(str(receipt.get("archive_sha256"))) is None
        or receipt.get("authoritative") is not True
    ):
        raise CheckpointDataPlaneError(
            "restore", "publication_receipt_invalid",
        )
    manifest_path = target / MANIFEST_NAME
    archive_path = target / "export.tar.gz"
    try:
        manifest_metadata = manifest_path.lstat()
        manifest_bytes = manifest_path.read_bytes()
        archive_metadata = archive_path.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError(
            "restore", "published_snapshot_missing",
        ) from exc
    if (
        manifest_path.is_symlink()
        or not stat.S_ISREG(manifest_metadata.st_mode)
        or manifest_metadata.st_nlink != 1
        or len(manifest_bytes) > limits.max_manifest_bytes
        or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
        or archive_path.is_symlink()
        or not stat.S_ISREG(archive_metadata.st_mode)
        or archive_metadata.st_nlink != 1
        or archive_metadata.st_size <= 0
        or archive_metadata.st_size > limits.max_archive_bytes
    ):
        raise CheckpointDataPlaneError(
            "restore", "selected_generation_material_invalid",
        )
    try:
        preliminary = json.loads(manifest_bytes)
        file_count = preliminary["file_count"]
        payload_bytes = preliminary["total_bytes"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError(
            "restore", "published_manifest_invalid",
        ) from exc
    published = PublishedCheckpointV2(
        checkpoint_id=checkpoint_id,
        snapshot_generation=snapshot_generation,
        capture_id=capture_id,
        root=target,
        payload_root=target / PAYLOAD_ROOT,
        archive_path=archive_path,
        manifest_sha256=manifest_sha256,
        archive_sha256=str(receipt["archive_sha256"]),
        archive_bytes=int(archive_metadata.st_size),
        file_count=file_count,
        payload_bytes=payload_bytes,
        authoritative=True,
        selected=True,
    )
    manifest = revalidate_published_checkpoint_v2(
        published,
        expected_identity_fingerprint=expected_identity_fingerprint,
        expected_checkpoint_abi=expected_checkpoint_abi,
        limits=limits,
    )
    if (
        manifest.get("checkpoint_lineage_id") != checkpoint_lineage_id
        or manifest.get("checkpoint_core_abi") != expected_checkpoint_core_abi
        or manifest.get("recovery_capability")
        != expected_recovery_capability
        or manifest.get("native_state_schema") != expected_native_state_schema
    ):
        raise CheckpointDataPlaneError(
            "restore", "selected_generation_identity_mismatch",
        )
    return published


_RETENTION_RELEASE_SCHEMA_V2 = "dradar-checkpoint-retention-release-v2"


def _retention_release_marker_bytes_v2(
    acknowledgement: CheckpointRetentionAcknowledgementV2,
    reference: CheckpointGenerationRefV2,
    published: PublishedCheckpointV2,
) -> bytes:
    return _canonical_json({
        "schema": _RETENTION_RELEASE_SCHEMA_V2,
        "assignment_id": acknowledgement.assignment_id,
        "operation_id": acknowledgement.operation_id,
        "checkpoint_id": reference.checkpoint_id,
        "snapshot_generation": reference.snapshot_generation,
        "manifest_sha256": reference.manifest_sha256,
        "capture_id": published.capture_id,
        "archive_sha256": published.archive_sha256,
        "authoritative": True,
    })


def _validate_retention_release_marker_v2(
    marker: Path,
    expected: bytes,
) -> None:
    try:
        metadata = marker.lstat()
        actual = marker.read_bytes()
    except OSError as exc:
        raise CheckpointDataPlaneError(
            "retention", "release_marker_unreadable",
        ) from exc
    if (
        marker.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected)
        or actual != expected
        or (
            os.name == "posix"
            and (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            )
        )
    ):
        raise CheckpointDataPlaneError(
            "retention", "release_marker_mismatch",
        )


def _write_retention_release_marker_v2(
    marker: Path,
    expected: bytes,
) -> None:
    if marker.exists() or marker.is_symlink():
        _validate_retention_release_marker_v2(marker, expected)
        return
    temporary = marker.with_name(
        f".{marker.name}.{uuid.uuid4().hex}.part"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(expected)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CheckpointDataPlaneError(
                    "retention", "release_marker_write_failed",
                )
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, marker)
        _fsync_directory(marker.parent)
    except CheckpointDataPlaneError:
        raise
    except OSError as exc:
        raise CheckpointDataPlaneError(
            "retention", "release_marker_write_failed",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_retention_reference_v2(
    reference: CheckpointGenerationRefV2,
) -> tuple[str, int, str]:
    if not isinstance(reference, CheckpointGenerationRefV2):
        raise CheckpointDataPlaneError(
            "retention", "invalid_generation_reference",
        )
    if (
        _IDENTIFIER_RE.fullmatch(reference.checkpoint_id) is None
        or not isinstance(reference.snapshot_generation, int)
        or isinstance(reference.snapshot_generation, bool)
        or reference.snapshot_generation < 0
        or _DIGEST_RE.fullmatch(reference.manifest_sha256) is None
    ):
        raise CheckpointDataPlaneError(
            "retention", "invalid_generation_reference",
        )
    return reference.key


def _expected_retention_publication_v2(
    storage_root: Path,
    reference: CheckpointGenerationRefV2,
    published: PublishedCheckpointV2,
) -> tuple[Path, Path, Path]:
    checkpoint_root = storage_root / "checkpoints" / reference.checkpoint_id
    generations = checkpoint_root / "generations"
    target = generations / (
        f"generation-{reference.snapshot_generation:020d}"
    )
    if (
        not isinstance(published, PublishedCheckpointV2)
        or published.checkpoint_id != reference.checkpoint_id
        or published.snapshot_generation != reference.snapshot_generation
        or published.manifest_sha256 != reference.manifest_sha256
        or published.authoritative is not True
        or Path(os.path.abspath(published.root))
        != Path(os.path.abspath(target))
        or Path(os.path.abspath(published.payload_root))
        != Path(os.path.abspath(target / PAYLOAD_ROOT))
        or Path(os.path.abspath(published.archive_path))
        != Path(os.path.abspath(target / "export.tar.gz"))
        or _IDENTIFIER_RE.fullmatch(published.capture_id) is None
        or _DIGEST_RE.fullmatch(published.archive_sha256) is None
        or not isinstance(published.archive_bytes, int)
        or isinstance(published.archive_bytes, bool)
        or published.archive_bytes <= 0
    ):
        raise CheckpointDataPlaneError(
            "retention", "local_generation_identity_mismatch",
        )
    return checkpoint_root, generations, target


def _revalidate_retention_quarantine_v2(
    quarantine: Path,
    published: PublishedCheckpointV2,
    reference: CheckpointGenerationRefV2,
    *,
    limits: CheckpointPackageLimitsV2,
) -> None:
    manifest_path = quarantine / MANIFEST_NAME
    try:
        metadata = manifest_path.lstat()
        if (
            manifest_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > limits.max_manifest_bytes
        ):
            raise OSError("unsafe manifest")
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != reference.manifest_sha256:
            raise OSError("manifest digest mismatch")
        manifest = json.loads(manifest_bytes)
        identity_fingerprint = manifest["identity_fingerprint"]
        checkpoint_abi = manifest["checkpoint_abi"]
        if (
            not isinstance(identity_fingerprint, str)
            or _DIGEST_RE.fullmatch(identity_fingerprint) is None
            or not isinstance(checkpoint_abi, str)
            or _ABI_RE.fullmatch(checkpoint_abi) is None
        ):
            raise OSError("invalid manifest identity")
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError(
            "retention", "generation_revalidation_failed",
        ) from exc
    quarantined = replace(
        published,
        root=quarantine,
        payload_root=quarantine / PAYLOAD_ROOT,
        archive_path=quarantine / "export.tar.gz",
    )
    try:
        revalidate_published_checkpoint_v2(
            quarantined,
            expected_identity_fingerprint=identity_fingerprint,
            expected_checkpoint_abi=checkpoint_abi,
            limits=limits,
        )
    except CheckpointDataPlaneError as exc:
        raise CheckpointDataPlaneError(
            "retention", "generation_revalidation_failed",
        ) from exc


def _current_matches_retention_reference_v2(
    current: dict[str, object] | None,
    reference: CheckpointGenerationRefV2,
) -> bool:
    if current is None:
        return False
    directory = f"generation-{reference.snapshot_generation:020d}"
    if current.get("directory") != directory:
        return False
    if (
        current.get("generation") != reference.snapshot_generation
        or current.get("manifest_sha256") != reference.manifest_sha256
        or current.get("authoritative") is not True
    ):
        raise CheckpointDataPlaneError(
            "retention", "current_pointer_identity_mismatch",
        )
    return True


def apply_checkpoint_generation_retention_v2(
    storage_root: Path,
    acknowledgement: CheckpointRetentionAcknowledgementV2,
    published_generations: Iterable[PublishedCheckpointV2],
    *,
    limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
) -> CheckpointRetentionApplyResultV2:
    """Delete only exact generations released by the authoritative server.

    The active generation is first atomically moved to an operation-bound
    quarantine.  A durable content-bound marker is written only after a full
    revalidation, allowing cleanup to resume safely if recursive deletion is
    interrupted.  Unknown, retained, malformed, or drifted evidence is never
    guessed about and remains on disk for diagnosis.
    """

    if not isinstance(acknowledgement, CheckpointRetentionAcknowledgementV2):
        raise CheckpointDataPlaneError("retention", "invalid_acknowledgement")
    if (
        _IDENTIFIER_RE.fullmatch(acknowledgement.assignment_id) is None
        or _IDENTIFIER_RE.fullmatch(acknowledgement.operation_id) is None
        or not isinstance(acknowledgement.owner_epoch_observed, int)
        or isinstance(acknowledgement.owner_epoch_observed, bool)
        or acknowledgement.owner_epoch_observed < 0
        or not isinstance(acknowledgement.current_owner_epoch, int)
        or isinstance(acknowledgement.current_owner_epoch, bool)
        or acknowledgement.current_owner_epoch < 0
        or not isinstance(acknowledgement.result_evidence_release, bool)
        or (
            acknowledgement.result_evidence_release
            and (
                not isinstance(acknowledgement.upload_intent_id, str)
                or _DIGEST_RE.fullmatch(
                    acknowledgement.upload_intent_id,
                ) is None
                or not isinstance(acknowledgement.submission_id, str)
                or _IDENTIFIER_RE.fullmatch(
                    acknowledgement.submission_id,
                ) is None
            )
        )
        or (
            not acknowledgement.result_evidence_release
            and (
                acknowledgement.upload_intent_id is not None
                or acknowledgement.submission_id is not None
            )
        )
    ):
        raise CheckpointDataPlaneError("retention", "invalid_acknowledgement")

    released_keys = [
        _validate_retention_reference_v2(reference)
        for reference in acknowledgement.delete_generations
    ]
    retained_keys = [
        _validate_retention_reference_v2(reference)
        for reference in acknowledgement.retain_generations
    ]
    all_keys = released_keys + retained_keys
    if (
        len(all_keys) > 64
        or len(all_keys) != len(set(all_keys))
        or set(released_keys) & set(retained_keys)
    ):
        raise CheckpointDataPlaneError(
            "retention", "invalid_acknowledgement_inventory",
        )
    materialized = tuple(published_generations)
    by_key: dict[tuple[str, int, str], PublishedCheckpointV2] = {}
    for published in materialized:
        if not isinstance(published, PublishedCheckpointV2):
            raise CheckpointDataPlaneError(
                "retention", "invalid_local_inventory",
            )
        key = (
            published.checkpoint_id,
            published.snapshot_generation,
            published.manifest_sha256,
        )
        if key in by_key:
            raise CheckpointDataPlaneError(
                "retention", "duplicate_local_generation",
            )
        by_key[key] = published
    if set(by_key) != set(all_keys):
        raise CheckpointDataPlaneError(
            "retention", "local_inventory_mismatch",
        )

    storage_root = Path(os.path.abspath(storage_root))
    if all_keys:
        _assert_private_directory(storage_root, stage="retention")
        _assert_private_directory(
            storage_root / "checkpoints", stage="retention",
        )
    expected_paths: dict[tuple[str, int, str], tuple[Path, Path, Path]] = {}
    for reference in (
        *acknowledgement.delete_generations,
        *acknowledgement.retain_generations,
    ):
        expected_paths[reference.key] = _expected_retention_publication_v2(
            storage_root, reference, by_key[reference.key],
        )

    deleted: list[CheckpointGenerationRefV2] = []
    absent: list[CheckpointGenerationRefV2] = []
    for reference in acknowledgement.delete_generations:
        published = by_key[reference.key]
        checkpoint_root, generations, target = expected_paths[reference.key]
        _assert_private_directory(checkpoint_root, stage="retention")
        _assert_private_directory(generations, stage="retention")
        suffix = (
            f"{reference.snapshot_generation:020d}-"
            f"{reference.manifest_sha256[:16]}-"
            f"{acknowledgement.operation_id}"
        )
        quarantine = generations / f".retention-{suffix}"
        marker = checkpoint_root / f".retention-release-{suffix}.json"
        marker_bytes = _retention_release_marker_bytes_v2(
            acknowledgement, reference, published,
        )
        with _publication_lock(checkpoint_root):
            current = _read_current(checkpoint_root)
            remove_current = _current_matches_retention_reference_v2(
                current, reference,
            )
            target_exists = target.exists() or target.is_symlink()
            quarantine_exists = quarantine.exists() or quarantine.is_symlink()
            marker_exists = marker.exists() or marker.is_symlink()
            if target_exists and (quarantine_exists or marker_exists):
                raise CheckpointDataPlaneError(
                    "retention", "release_state_conflict",
                )
            if marker_exists:
                _validate_retention_release_marker_v2(marker, marker_bytes)
            elif target_exists:
                try:
                    os.replace(target, quarantine)
                    _fsync_directory(generations)
                except OSError as exc:
                    raise CheckpointDataPlaneError(
                        "retention", "generation_quarantine_failed",
                    ) from exc
                quarantine_exists = True
            elif not quarantine_exists:
                if remove_current:
                    try:
                        (checkpoint_root / "CURRENT").unlink()
                        _fsync_directory(checkpoint_root)
                    except OSError as exc:
                        raise CheckpointDataPlaneError(
                            "retention", "current_pointer_cleanup_failed",
                        ) from exc
                absent.append(reference)
                continue

            if remove_current:
                try:
                    (checkpoint_root / "CURRENT").unlink()
                    _fsync_directory(checkpoint_root)
                except OSError as exc:
                    raise CheckpointDataPlaneError(
                        "retention", "current_pointer_cleanup_failed",
                    ) from exc
            if not marker_exists:
                _revalidate_retention_quarantine_v2(
                    quarantine, published, reference, limits=limits,
                )
                _write_retention_release_marker_v2(marker, marker_bytes)
                marker_exists = True
            if quarantine.exists() or quarantine.is_symlink():
                try:
                    metadata = quarantine.lstat()
                    if (
                        quarantine.is_symlink()
                        or not stat.S_ISDIR(metadata.st_mode)
                        or (
                            os.name == "posix"
                            and (
                                metadata.st_uid != os.getuid()
                                or stat.S_IMODE(metadata.st_mode) & 0o077
                            )
                        )
                    ):
                        raise OSError("unsafe quarantine")
                    shutil.rmtree(quarantine)
                    _fsync_directory(generations)
                except OSError as exc:
                    raise CheckpointDataPlaneError(
                        "retention", "generation_cleanup_failed",
                    ) from exc
            try:
                marker.unlink()
                _fsync_directory(checkpoint_root)
            except OSError as exc:
                raise CheckpointDataPlaneError(
                    "retention", "release_marker_cleanup_failed",
                ) from exc
            deleted.append(reference)

    return CheckpointRetentionApplyResultV2(
        deleted_generations=tuple(deleted),
        already_absent_generations=tuple(absent),
        retained_generations=acknowledgement.retain_generations,
    )


def _bounded_failure_type(exc: BaseException) -> str:
    value = type(exc).__name__
    return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value) else "Exception"


def _record_local_checkpoint_failure_v2(
    storage_root: Path,
    *,
    exc: BaseException,
    stage: str,
    code: str,
) -> None:
    """Best-effort private diagnostic journal; never affects mainline work.

    Arbitrary exception strings and command output are intentionally omitted.
    A typed transport may attach only bounded scalar facts such as an exit
    status and stdout/stderr digests.  The file is capped rather than rotated
    so diagnostics cannot turn a checkpoint failure into disk pressure.
    """

    try:
        root = Path(storage_root)
        _ensure_private_directory(root, stage="cleanup")
        path = root / "diagnostics.jsonl"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size >= 2 * 1024 * 1024
            ):
                return
            diagnostic = (
                dict(exc.diagnostic)
                if isinstance(exc, CheckpointDataPlaneError)
                else {}
            )
            payload = {
                "schema": "dradar-checkpoint-local-diagnostic-v2",
                "at": datetime.now(timezone.utc).replace(
                    microsecond=0,
                ).isoformat(),
                "stage": stage,
                "code": code,
                "failure_type": _bounded_failure_type(exc),
                "diagnostic": diagnostic,
            }
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii") + b"\n"
            if len(encoded) <= 4096:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        # OBSERVE/RESTORE_TEST diagnostics are strictly subordinate to the
        # normal trial, including when the diagnostic root itself is broken.
        return


def record_local_checkpoint_failure_v2(
    storage_root: Path,
    *,
    exc: BaseException,
    stage: str,
    code: str,
) -> None:
    """Public bounded diagnostic hook for owner/reconciliation failures."""

    _record_local_checkpoint_failure_v2(
        storage_root,
        exc=exc,
        stage=stage,
        code=code,
    )


def checkpoint_observation_failure_family_v2(code: str | None) -> str:
    """Collapse local detail into one stable, low-cardinality server family."""

    if not code:
        return "observer_failed"
    if code in {"observer_failed", "observer_cancel_timeout", "disk_full"}:
        return code
    if code in {"mainline_completed", "mainline_aborted"}:
        return code
    if "secret" in code:
        return "secret_detected"
    if "cleanup" in code or code == "mainline_aborted":
        return "cleanup_failed"
    if "adapter" in code or "abi" in code or "capability" in code:
        return "adapter_incompatible"
    if "identity" in code or code.startswith("invalid_"):
        return "identity_invalid"
    if code.startswith("restore_"):
        return "restore_failed"
    if "limit" in code or "oversized" in code:
        return "resource_limit"
    if code.startswith("source_") or code in {
        "unsafe_directory", "unsafe_file_type", "unsafe_path",
        "unsafe_source_file", "filesystem_boundary",
        "container_storage_cross_device", "container_storage_unavailable",
        "export_outside_container_storage", "unsafe_capture_storage",
    }:
        return "source_unsafe"
    if code.startswith("download_") or code in {
        "copy_failed", "file_write_failed",
    }:
        return "transfer_failed"
    if code in {
        "generation_conflict", "incoming_stage_exists", "current_write_failed",
        "publication_lock_failed", "publication_lock_unsafe",
    }:
        return "publication_conflict"
    if "storage" in code or "current_pointer" in code:
        return "storage_unsafe"
    if any(fragment in code for fragment in (
        "archive", "manifest", "member", "payload", "published",
        "directory", "file_digest", "verification_stage", "missing_parent",
    )):
        return "archive_invalid"
    return "capture_failed"


def checkpoint_observation_payload_v2(
    request: CheckpointCaptureRequestV2,
    observation: CheckpointObservationV2,
    activation: CheckpointActivationV2,
    runtime: CheckpointObservationRuntimeV2,
) -> dict[str, object]:
    """Build the only bounded wire shape accepted by shadow telemetry."""

    request.validate()
    if not activation.capture_enabled or observation.status == "skipped":
        raise CheckpointDataPlaneError("capture", "observation_not_enabled")
    if _IDENTIFIER_RE.fullmatch(runtime.operation_id) is None:
        raise CheckpointDataPlaneError("capture", "invalid_observation_id")
    if _IDENTIFIER_RE.fullmatch(runtime.assignment_id) is None:
        raise CheckpointDataPlaneError("capture", "invalid_assignment_id")
    if (
        not isinstance(runtime.elapsed_ms, int)
        or isinstance(runtime.elapsed_ms, bool)
        or not 0 <= runtime.elapsed_ms <= 86_400_000
    ):
        raise CheckpointDataPlaneError("capture", "invalid_elapsed_time")
    if runtime.platform not in {"macos", "linux", "wsl", "windows", "other"}:
        raise CheckpointDataPlaneError("capture", "invalid_platform")
    if runtime.container_backend not in {
        "docker", "orbstack", "podman", "native", "other",
    }:
        raise CheckpointDataPlaneError("capture", "invalid_container_backend")
    if re.fullmatch(r"[A-Za-z0-9._+-]{1,64}", runtime.client_version) is None:
        raise CheckpointDataPlaneError("capture", "invalid_client_version")
    if re.fullmatch(r"[A-Za-z0-9._/+:-]{1,160}", runtime.adapter_version) is None:
        raise CheckpointDataPlaneError("capture", "invalid_adapter_version")
    if observation.remote_cleanup not in {"not_needed", "discarded", "failed"}:
        raise CheckpointDataPlaneError("cleanup", "invalid_cleanup_result")
    if observation.status not in {"sealed", "failed", "aborted"}:
        raise CheckpointDataPlaneError("capture", "invalid_observation_status")
    published = observation.published
    if observation.status == "sealed" and published is None:
        raise CheckpointDataPlaneError("capture", "sealed_observation_missing")
    if observation.status != "sealed" and published is not None:
        raise CheckpointDataPlaneError("capture", "failed_observation_published")
    payload: dict[str, object] = {
        "observation_kind": "capture",
        "assignment_id": runtime.assignment_id,
        "operation_id": runtime.operation_id,
        "capture_id": request.capture_id,
        "checkpoint_id": request.checkpoint_id,
        "checkpoint_lineage_id": request.checkpoint_lineage_id,
        "snapshot_generation": request.snapshot_generation,
        "rollout_mode": activation.effective_mode.wire_value,
        "status": observation.status,
        "stage": None if published is not None else observation.stage,
        "failure_code": (
            None if published is not None
            else checkpoint_observation_failure_family_v2(observation.code)
        ),
        "failure_type": None if published is not None else observation.failure_type,
        "identity_fingerprint": request.identity_fingerprint,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": request.checkpoint_abi,
        "capture_storage": "container_native",
        "manifest_sha256": published.manifest_sha256 if published else None,
        "archive_sha256": published.archive_sha256 if published else None,
        "archive_bytes": published.archive_bytes if published else None,
        "file_count": published.file_count if published else None,
        "payload_bytes": published.payload_bytes if published else None,
        "elapsed_ms": runtime.elapsed_ms,
        "platform": runtime.platform,
        "container_backend": runtime.container_backend,
        "client_version": runtime.client_version,
        "adapter_version": runtime.adapter_version,
        "remote_cleanup": observation.remote_cleanup,
        "authoritative": published.authoritative if published else False,
        "selected_local": published.selected if published else False,
    }
    return payload


def checkpoint_restore_observation_payload_v2(
    capture_request: CheckpointCaptureRequestV2,
    restore_request: CheckpointRestoreRequestV2,
    observation: CheckpointRestoreObservationV2,
    activation: CheckpointActivationV2,
    runtime: CheckpointObservationRuntimeV2,
) -> dict[str, object]:
    """Build bounded evidence for one disposable, non-paid restore test."""

    capture_request.validate()
    published = restore_request.published
    if (
        not activation.offline_restore_enabled
        or observation.status == "skipped"
    ):
        raise CheckpointDataPlaneError("restore", "observation_not_enabled")
    if (
        restore_request.restore_id != observation.restore_id
        or published.capture_id != capture_request.capture_id
        or published.checkpoint_id != capture_request.checkpoint_id
        or published.snapshot_generation != capture_request.snapshot_generation
        or restore_request.expected_identity_fingerprint
        != capture_request.identity_fingerprint
    ):
        raise CheckpointDataPlaneError("restore", "restore_identity_mismatch")
    if _IDENTIFIER_RE.fullmatch(runtime.operation_id) is None:
        raise CheckpointDataPlaneError("restore", "invalid_observation_id")
    if _IDENTIFIER_RE.fullmatch(runtime.assignment_id) is None:
        raise CheckpointDataPlaneError("restore", "invalid_assignment_id")
    if (
        not isinstance(runtime.elapsed_ms, int)
        or isinstance(runtime.elapsed_ms, bool)
        or not 0 <= runtime.elapsed_ms <= 86_400_000
    ):
        raise CheckpointDataPlaneError("restore", "invalid_elapsed_time")
    if runtime.platform not in {"macos", "linux", "wsl", "windows", "other"}:
        raise CheckpointDataPlaneError("restore", "invalid_platform")
    if runtime.container_backend not in {
        "docker", "orbstack", "podman", "native", "other",
    }:
        raise CheckpointDataPlaneError("restore", "invalid_container_backend")
    if re.fullmatch(r"[A-Za-z0-9._+-]{1,64}", runtime.client_version) is None:
        raise CheckpointDataPlaneError("restore", "invalid_client_version")
    if re.fullmatch(r"[A-Za-z0-9._/+:-]{1,160}", runtime.adapter_version) is None:
        raise CheckpointDataPlaneError("restore", "invalid_adapter_version")
    if observation.status not in {"verified", "failed", "aborted"}:
        raise CheckpointDataPlaneError("restore", "invalid_observation_status")
    evidence = observation.evidence
    if observation.status == "verified":
        if (
            evidence is None
            or observation.stage is not None
            or observation.code is not None
            or observation.failure_type is not None
            or evidence.restore_id != restore_request.restore_id
            or evidence.manifest_sha256 != published.manifest_sha256
            or evidence.identity_fingerprint
            != restore_request.expected_identity_fingerprint
            or evidence.restore_adapter_version != runtime.adapter_version
            or evidence.paid_execution_started
        ):
            raise CheckpointDataPlaneError("restore", "restore_evidence_invalid")
    elif (
        evidence is not None
        or observation.stage is None
        or observation.code is None
    ):
        raise CheckpointDataPlaneError("restore", "restore_failure_invalid")
    return {
        "observation_kind": "restore",
        "assignment_id": runtime.assignment_id,
        "operation_id": runtime.operation_id,
        "restore_id": restore_request.restore_id,
        "source_capture_id": capture_request.capture_id,
        "checkpoint_id": capture_request.checkpoint_id,
        "checkpoint_lineage_id": capture_request.checkpoint_lineage_id,
        "snapshot_generation": capture_request.snapshot_generation,
        "rollout_mode": activation.effective_mode.wire_value,
        "status": observation.status,
        "stage": None if evidence is not None else observation.stage,
        "failure_code": (
            None if evidence is not None
            else checkpoint_observation_failure_family_v2(observation.code)
        ),
        "failure_type": None if evidence is not None else observation.failure_type,
        "identity_fingerprint": capture_request.identity_fingerprint,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": capture_request.checkpoint_abi,
        "manifest_sha256": published.manifest_sha256,
        "elapsed_ms": runtime.elapsed_ms,
        "platform": runtime.platform,
        "container_backend": runtime.container_backend,
        "client_version": runtime.client_version,
        "adapter_version": runtime.adapter_version,
        "paid_execution_started": False,
        "authoritative": False,
    }


class CheckpointDataPlaneV2:
    """Fail-open coordinator for optional capture and offline restore."""

    def __init__(
        self,
        *,
        activation: CheckpointActivationV2,
        storage_root: Path,
        limits: CheckpointPackageLimitsV2 = DEFAULT_PACKAGE_LIMITS_V2,
        retention: CheckpointRetentionPolicyV2 = DEFAULT_RETENTION_POLICY_V2,
    ) -> None:
        self.activation = activation
        self.storage_root = Path(storage_root)
        self.limits = limits
        self.retention = retention

    async def observe_capture(
        self,
        request: CheckpointCaptureRequestV2,
        exporter: HarnessCheckpointExporterV2,
    ) -> CheckpointObservationV2:
        if not self.activation.capture_enabled:
            return CheckpointObservationV2(status="skipped", capture_id=None)
        export: ContainerSealedExportV2 | None = None
        download: Path | None = None
        stage = "capture"
        try:
            request.validate()
            if exporter.checkpoint_abi != request.checkpoint_abi:
                raise CheckpointDataPlaneError("capture", "adapter_abi_mismatch")
            # Validate the operator-selected host boundary before asking a
            # Harness to spend CPU or create container-side state.  Checking
            # only the nested download/publication directories would allow a
            # permissive or substituted ancestor to go unnoticed.
            _ensure_private_directory(self.storage_root, stage="download")
            try:
                free_bytes = shutil.disk_usage(self.storage_root).free
            except OSError as exc:
                raise CheckpointDataPlaneError(
                    "capture", "storage_unavailable",
                ) from exc
            required_free = (
                self.retention.minimum_free_bytes
                + self.limits.max_archive_bytes
            )
            if free_bytes < required_free:
                raise CheckpointDataPlaneError("capture", "disk_full")
            export = await exporter.capture_and_seal(request)
            stage = "seal"
            export.validate(request)
            if export.archive_size > self.limits.max_archive_bytes:
                raise CheckpointDataPlaneError("seal", "archive_size_limit")
            incoming_root = self.storage_root / ".downloads"
            _ensure_private_directory(incoming_root, stage="download")
            # A unique target is essential for kill -9/power-loss recovery:
            # an abandoned partial transfer is never interpreted as input and
            # cannot block an exact capture replay. Stale parts remain inert
            # diagnostics until bounded storage maintenance removes them.
            download = incoming_root / (
                f"{request.capture_id}-{uuid.uuid4().hex}.tar.gz.part"
            )
            stage = "download"
            await exporter.download_export(
                export,
                download,
                max_bytes=self.limits.max_archive_bytes,
            )
            stage = "publish"
            published = publish_checkpoint_export_v2(
                download,
                self.storage_root / "checkpoints",
                request,
                export,
                authoritative=self.activation.authoritative,
                limits=self.limits,
            )
            _prune_shadow_generations_v2(
                published,
                keep=self.retention.shadow_generations,
            )
            cleanup = await self._discard_export(exporter, export)
            return CheckpointObservationV2(
                status="sealed",
                capture_id=request.capture_id,
                published=published,
                remote_cleanup=cleanup,
            )
        except asyncio.CancelledError:
            if export is not None:
                await asyncio.shield(self._discard_export(exporter, export))
            raise
        except Exception as exc:
            if isinstance(exc, CheckpointDataPlaneError):
                stage, code = exc.stage, exc.code
            else:
                code = f"{stage}_failed"
            _record_local_checkpoint_failure_v2(
                self.storage_root,
                exc=exc,
                stage=stage,
                code=code,
            )
            cleanup = (
                await self._discard_export(exporter, export)
                if export is not None else "not_needed"
            )
            return CheckpointObservationV2(
                status="failed",
                capture_id=request.capture_id,
                stage=stage,
                code=code,
                failure_type=_bounded_failure_type(exc),
                remote_cleanup=cleanup,
            )
        finally:
            if download is not None:
                try:
                    if download.is_symlink():
                        download.unlink()
                    elif download.exists():
                        metadata = download.lstat()
                        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                            download.unlink()
                except OSError:
                    # Leftovers are never selected as a generation because
                    # their name remains below the non-authoritative download
                    # staging directory.
                    pass

    @staticmethod
    async def _discard_export(
        exporter: HarnessCheckpointExporterV2,
        export: ContainerSealedExportV2,
    ) -> str:
        try:
            await exporter.discard_export(export)
        except Exception:
            return "failed"
        return "discarded"

    async def observe_offline_restore(
        self,
        request: CheckpointRestoreRequestV2,
        restorer: HarnessCheckpointRestorerV2,
    ) -> CheckpointRestoreObservationV2:
        if not self.activation.offline_restore_enabled:
            return CheckpointRestoreObservationV2(
                status="skipped", restore_id=request.restore_id,
            )
        try:
            if _IDENTIFIER_RE.fullmatch(request.restore_id) is None:
                raise CheckpointDataPlaneError("restore", "invalid_restore_id")
            if _DIGEST_RE.fullmatch(request.expected_identity_fingerprint) is None:
                raise CheckpointDataPlaneError("restore", "invalid_identity")
            revalidate_published_checkpoint_v2(
                request.published,
                expected_identity_fingerprint=request.expected_identity_fingerprint,
                expected_checkpoint_abi=restorer.checkpoint_abi,
                limits=self.limits,
            )
            evidence = await restorer.restore_offline(request)
            if (
                evidence.restore_id != request.restore_id
                or evidence.manifest_sha256 != request.published.manifest_sha256
                or evidence.identity_fingerprint != request.expected_identity_fingerprint
                or evidence.restore_adapter_version != restorer.adapter_version
                or evidence.paid_execution_started
            ):
                raise CheckpointDataPlaneError("restore", "restore_evidence_invalid")
            return CheckpointRestoreObservationV2(
                status="verified", restore_id=request.restore_id, evidence=evidence,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, CheckpointDataPlaneError):
                stage, code = exc.stage, exc.code
            else:
                stage, code = "restore", "restore_failed"
            _record_local_checkpoint_failure_v2(
                self.storage_root,
                exc=exc,
                stage=stage,
                code=code,
            )
            return CheckpointRestoreObservationV2(
                status="failed",
                restore_id=request.restore_id,
                stage=stage,
                code=code,
                failure_type=_bounded_failure_type(exc),
            )


async def run_mainline_with_shadow_checkpoint_v2(
    mainline: Awaitable[object],
    observation: Awaitable[CheckpointObservationV2],
    *,
    on_observation: Callable[[CheckpointObservationV2], None] | None = None,
    observation_join_timeout_sec: float = 0.25,
) -> object:
    """Prove the shadow writer cannot replace the mainline result.

    The mainline task is awaited directly.  The observation is reaped in a
    child task and reduced to a typed failure if it raises unexpectedly.  A
    mainline exception or cancellation is propagated unchanged after the
    shadow task has been cancelled and reaped.
    """

    if not isinstance(observation_join_timeout_sec, (int, float)) or not (
        0 < float(observation_join_timeout_sec) <= 30
    ):
        raise ValueError("checkpoint observation join timeout is invalid")
    shadow = asyncio.create_task(observation, name="dradar-checkpoint-v2-observe")

    def consume_late_result(task: asyncio.Task[CheckpointObservationV2]) -> None:
        try:
            task.exception()
        except BaseException:
            pass

    async def reap_shadow(*, mainline_completed: bool) -> CheckpointObservationV2:
        if not shadow.done():
            shadow.cancel()
        try:
            return await asyncio.wait_for(
                asyncio.shield(shadow), timeout=float(observation_join_timeout_sec),
            )
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return CheckpointObservationV2(
                status="aborted",
                capture_id=None,
                stage="cleanup",
                code=(
                    "mainline_completed" if mainline_completed
                    else "mainline_aborted"
                ),
            )
        except TimeoutError:
            shadow.cancel()
            shadow.add_done_callback(consume_late_result)
            return CheckpointObservationV2(
                status="failed",
                capture_id=None,
                stage="cleanup",
                code="observer_cancel_timeout",
                failure_type="TimeoutError",
            )
        except Exception as exc:
            return CheckpointObservationV2(
                status="failed",
                capture_id=None,
                stage="capture",
                code="observer_failed",
                failure_type=_bounded_failure_type(exc),
            )

    try:
        result = await mainline
    except BaseException:
        await reap_shadow(mainline_completed=False)
        raise
    observed = await reap_shadow(mainline_completed=True)
    if on_observation is not None:
        try:
            on_observation(observed)
        except Exception:
            # Diagnostics are optional too.  A callback cannot veto a valid
            # model result or turn it into an assignment failure.
            pass
    return result


async def run_mainline_with_periodic_shadow_captures_v2(
    mainline: Awaitable[object],
    capture: Callable[[int], Awaitable[CheckpointObservationV2]],
    *,
    on_observation: Callable[[CheckpointObservationV2], None] | None = None,
    first_generation: int = 1,
    initial_delay_sec: float = 300.0,
    interval_sec: float = 300.0,
    maximum_captures: int = 24,
    consecutive_failure_limit: int = 3,
    shutdown_timeout_sec: float = 0.25,
) -> object:
    """Sample restart evidence beside a mainline without becoming its owner.

    Captures are sequential, bounded, and stop locally after repeated writer
    failures.  This circuit has no assignment/refill authority: the paid
    mainline remains the only return value and every callback failure is
    ignored.  Cancellation is delegated to the data plane, which reaps the
    exact remote export before propagating it.
    """

    for value, label, lower, upper in (
        (initial_delay_sec, "initial delay", 0, 86_400),
        (interval_sec, "interval", 0.01, 86_400),
        (shutdown_timeout_sec, "shutdown timeout", 0.01, 30),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not lower <= float(value) <= upper
        ):
            raise ValueError(f"checkpoint shadow {label} is invalid")
    for value, label, lower, upper in (
        (first_generation, "first generation", 0, 2**31 - 1),
        (maximum_captures, "capture limit", 1, 10_000),
        (consecutive_failure_limit, "failure limit", 1, 100),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not lower <= value <= upper
        ):
            raise ValueError(f"checkpoint shadow {label} is invalid")

    def report(observation: CheckpointObservationV2) -> None:
        if on_observation is None:
            return
        try:
            on_observation(observation)
        except Exception:
            pass

    async def sampler() -> CheckpointObservationV2:
        if initial_delay_sec:
            await asyncio.sleep(float(initial_delay_sec))
        generation = first_generation
        failures = 0
        last = CheckpointObservationV2(status="skipped", capture_id=None)
        for index in range(maximum_captures):
            try:
                observation = await capture(generation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                observation = CheckpointObservationV2(
                    status="failed",
                    capture_id=None,
                    stage="capture",
                    code="observer_failed",
                    failure_type=_bounded_failure_type(exc),
                )
            report(observation)
            last = observation
            failures = failures + 1 if observation.status == "failed" else 0
            if failures >= consecutive_failure_limit:
                return last
            generation += 1
            if index + 1 < maximum_captures:
                await asyncio.sleep(float(interval_sec))
        return last

    def report_terminal(observation: CheckpointObservationV2) -> None:
        # A cancellation/cleanup timeout happens outside an individual sample
        # and is useful restart evidence.  A naturally exhausted sampler
        # returns its last sample, which was already emitted above.
        if (
            observation.capture_id is None
            and observation.code in {
                "mainline_completed", "mainline_aborted",
                "observer_cancel_timeout",
            }
        ):
            report(observation)

    # The lower-level joiner preserves the exact mainline exception/result and
    # bounds cancellation cleanup.  Per-sample observations are emitted above;
    # only a distinct terminal cleanup observation is added here.
    return await run_mainline_with_shadow_checkpoint_v2(
        mainline,
        sampler(),
        on_observation=report_terminal,
        observation_join_timeout_sec=shutdown_timeout_sec,
    )


def new_capture_request_v2(
    *,
    checkpoint_id: str,
    checkpoint_lineage_id: str,
    snapshot_generation: int,
    identity_fingerprint: str,
    checkpoint_abi: str,
    recovery_capability: str,
    native_state_schema: str | None,
    captured_at: str | None = None,
) -> CheckpointCaptureRequestV2:
    return CheckpointCaptureRequestV2(
        checkpoint_id=checkpoint_id,
        checkpoint_lineage_id=checkpoint_lineage_id,
        snapshot_generation=snapshot_generation,
        capture_id=uuid.uuid4().hex,
        identity_fingerprint=identity_fingerprint,
        checkpoint_abi=checkpoint_abi,
        recovery_capability=recovery_capability,
        native_state_schema=native_state_schema,
        captured_at=captured_at or datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "CONTAINER_EXPORT_ROOT",
    "DEFAULT_PACKAGE_LIMITS_V2",
    "DEFAULT_RETENTION_POLICY_V2",
    "EXPORT_SCHEMA_V2",
    "CheckpointCaptureRequestV2",
    "CheckpointDataPlaneError",
    "CheckpointDataPlaneV2",
    "CheckpointObservationV2",
    "CheckpointObservationRuntimeV2",
    "CheckpointPackageLimitsV2",
    "CheckpointRetentionPolicyV2",
    "CheckpointRestoreEvidenceV2",
    "CheckpointRestoreObservationV2",
    "CheckpointRestoreRequestV2",
    "CheckpointRetentionApplyResultV2",
    "ContainerSealedExportV2",
    "HarnessCheckpointExporterV2",
    "HarnessCheckpointRestorerV2",
    "PublishedCheckpointV2",
    "new_capture_request_v2",
    "next_shadow_generation_v2",
    "apply_checkpoint_generation_retention_v2",
    "checkpoint_observation_failure_family_v2",
    "checkpoint_observation_payload_v2",
    "checkpoint_restore_observation_payload_v2",
    "publish_checkpoint_export_v2",
    "load_exact_published_checkpoint_v2",
    "revalidate_published_checkpoint_v2",
    "record_local_checkpoint_failure_v2",
    "run_mainline_with_shadow_checkpoint_v2",
    "run_mainline_with_periodic_shadow_captures_v2",
    "seal_checkpoint_export_v2",
]
