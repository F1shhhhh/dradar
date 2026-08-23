"""Build the dependency-light, checksum-pinned Checkpoint V2 container helper."""

from __future__ import annotations

import hashlib
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path


CONTAINER_HELPER_SCHEMA_V2 = "dradar-checkpoint-container-helper-v2"
_BUNDLED_MODULES = (
    "checkpoint_activation_v2.py",
    "checkpoint_adapters_v2.py",
    "checkpoint_runtime_v2.py",
    "checkpoint_adapter_runtime_v2.py",
)


@dataclass(frozen=True)
class CheckpointContainerBundleV2:
    path: Path
    sha256: str
    size: int
    allow_test_root: bool


def _main_source(*, allow_test_root: bool) -> str:
    return f'''\
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

from dradar.checkpoint_adapter_runtime_v2 import (
    create_adapter_capture_root_v2,
    restore_adapter_capture_offline_v2,
)
from dradar.checkpoint_adapters_v2 import checkpoint_adapter_contract_v2
from dradar.checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneError,
    ContainerSealedExportV2,
    publish_checkpoint_export_v2,
    seal_checkpoint_export_v2,
)

SCHEMA = {CONTAINER_HELPER_SCHEMA_V2!r}
ALLOW_TEST_ROOT = {allow_test_root!r}
IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{{8,64}}")
CAPTURE_FIELDS = {{
    "schema", "operation", "filesystem_root", "harness", "provider",
    "worktree_path", "capture_root_path", "export_path", "base_commit",
    "captured_at", "session_id", "sensitive_values", "request",
}}
RESTORE_FIELDS = {{
    "schema", "operation", "filesystem_root", "harness", "provider",
    "worktree_path", "state_root_path", "storage_root_path", "archive_path",
    "base_commit", "expected_identity_fingerprint", "request", "export",
}}
REQUEST_FIELDS = {{
    "checkpoint_id", "checkpoint_lineage_id", "snapshot_generation",
    "capture_id", "identity_fingerprint", "checkpoint_abi",
    "recovery_capability", "native_state_schema", "captured_at",
}}
EXPORT_FIELDS = {{
    "capture_id", "remote_path", "archive_sha256", "archive_size",
    "manifest_sha256", "capture_storage",
}}


def fail(stage, code, exit_code=74):
    value = {{"schema": SCHEMA, "ok": False, "stage": stage, "code": code}}
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    raise SystemExit(exit_code)


def read_spec(path):
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > 128 * 1024
        ):
            fail("capture", "helper_spec_unsafe", 64)
        raw = os.read(descriptor, 128 * 1024 + 1)
        if len(raw) != metadata.st_size:
            fail("capture", "helper_spec_changed", 64)
        value = json.loads(raw)
    except SystemExit:
        raise
    except BaseException:
        fail("capture", "helper_spec_invalid", 64)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        fail("capture", "helper_spec_invalid", 64)
    return value


def filesystem_root(spec):
    raw = spec.get("filesystem_root", "/")
    if not isinstance(raw, str):
        fail("capture", "helper_filesystem_root_invalid", 64)
    root = Path(raw)
    if not root.is_absolute() or (not ALLOW_TEST_ROOT and root != Path("/")):
        fail("capture", "helper_filesystem_root_invalid", 64)
    return root


def resolve(root, raw):
    if not isinstance(raw, str):
        fail("capture", "helper_path_invalid", 64)
    logical = PurePosixPath(raw)
    if not logical.is_absolute() or ".." in logical.parts:
        fail("capture", "helper_path_invalid", 64)
    return root.joinpath(*logical.parts[1:])


def request_from(spec):
    raw = spec.get("request")
    if not isinstance(raw, dict) or set(raw) != REQUEST_FIELDS:
        fail("capture", "helper_request_invalid", 64)
    try:
        value = CheckpointCaptureRequestV2(**raw)
        value.validate()
    except BaseException:
        fail("capture", "helper_request_invalid", 64)
    return value


def export_from(spec):
    raw = spec.get("export")
    if not isinstance(raw, dict) or set(raw) != EXPORT_FIELDS:
        fail("restore", "helper_export_invalid", 64)
    try:
        return ContainerSealedExportV2(**raw)
    except BaseException:
        fail("restore", "helper_export_invalid", 64)


def contract_from(spec):
    harness = spec.get("harness")
    provider = spec.get("provider")
    if not isinstance(harness, str) or not isinstance(provider, str):
        fail("capture", "helper_adapter_invalid", 64)
    try:
        return checkpoint_adapter_contract_v2(harness, provider)
    except BaseException:
        fail("capture", "helper_adapter_invalid", 64)


def capture(spec):
    if set(spec) != CAPTURE_FIELDS or spec.get("operation") != "capture":
        fail("capture", "helper_spec_fields_invalid", 64)
    root = filesystem_root(spec)
    contract = contract_from(spec)
    request = request_from(spec)
    if request.checkpoint_abi != contract.checkpoint_abi:
        fail("capture", "helper_adapter_abi_mismatch", 64)
    sensitive = spec.get("sensitive_values")
    if (
        not isinstance(sensitive, list)
        or len(sensitive) > 32
        or any(not isinstance(value, str) or len(value) > 4096 for value in sensitive)
    ):
        fail("capture", "helper_sensitive_values_invalid", 64)
    capture_root = resolve(root, spec.get("capture_root_path"))
    export_path = resolve(root, spec.get("export_path"))
    try:
        summary = create_adapter_capture_root_v2(
            filesystem_root=root,
            worktree_path=spec["worktree_path"],
            capture_root_path=spec["capture_root_path"],
            contract=contract,
            base_commit=spec["base_commit"],
            captured_at=spec["captured_at"],
            session_id=spec["session_id"],
            sensitive_values=sensitive,
        )
        if (
            summary.recovery_capability != request.recovery_capability
            or request.native_state_schema != contract.native_state_schema
            or request.captured_at != spec["captured_at"]
        ):
            raise CheckpointDataPlaneError("capture", "helper_capability_mismatch")
        exported = seal_checkpoint_export_v2(
            summary.capture_root,
            export_path,
            request,
            sensitive_values=sensitive,
            container_export_root=resolve(root, "/run/dradar-checkpoint-v2"),
        )
        exported = replace(exported, remote_path=spec["export_path"])
        value = {{
            "schema": SCHEMA,
            "ok": True,
            "operation": "capture",
            "capture_id": request.capture_id,
            "recovery_capability": summary.recovery_capability,
            "present_artifacts": sorted(summary.present_artifacts),
            "workspace_patch_bytes": summary.workspace_patch_bytes,
            "untracked_files": summary.untracked_files,
            "untracked_bytes": summary.untracked_bytes,
            "export": {{
                "capture_id": exported.capture_id,
                "remote_path": exported.remote_path,
                "archive_sha256": exported.archive_sha256,
                "archive_size": exported.archive_size,
                "manifest_sha256": exported.manifest_sha256,
                "capture_storage": exported.capture_storage,
            }},
        }}
        print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    finally:
        shutil.rmtree(capture_root, ignore_errors=True)


def restore(spec):
    if set(spec) != RESTORE_FIELDS or spec.get("operation") != "restore":
        fail("restore", "helper_spec_fields_invalid", 64)
    root = filesystem_root(spec)
    contract = contract_from(spec)
    request = request_from(spec)
    exported = export_from(spec)
    exported.validate(request)
    archive_path = resolve(root, spec.get("archive_path"))
    storage_root = resolve(root, spec.get("storage_root_path"))
    published = publish_checkpoint_export_v2(
        archive_path,
        storage_root,
        request,
        exported,
        authoritative=False,
    )
    evidence = restore_adapter_capture_offline_v2(
        published=published,
        contract=contract,
        destination_worktree=resolve(root, spec.get("worktree_path")),
        destination_state_root=resolve(root, spec.get("state_root_path")),
        expected_identity_fingerprint=spec["expected_identity_fingerprint"],
        base_commit=spec["base_commit"],
    )
    value = {{
        "schema": SCHEMA,
        "ok": True,
        "operation": "restore",
        "capture_id": request.capture_id,
        "manifest_sha256": published.manifest_sha256,
        "identity_fingerprint": spec["expected_identity_fingerprint"],
        "session_id": evidence.session_id,
        "recovery_capability": evidence.recovery_capability,
        "restored_untracked_files": evidence.restored_untracked_files,
        "paid_execution_started": False,
    }}
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def main():
    if len(sys.argv) != 2:
        fail("capture", "helper_argv_invalid", 64)
    spec_path = Path(sys.argv[1])
    try:
        spec = read_spec(spec_path)
        operation = spec.get("operation")
        if operation == "capture":
            capture(spec)
        elif operation == "restore":
            restore(spec)
        else:
            fail("capture", "helper_operation_invalid", 64)
    except CheckpointDataPlaneError as exc:
        fail(exc.stage, exc.code)
    except SystemExit:
        raise
    except BaseException:
        fail("capture", "helper_internal_error")
    finally:
        try:
            spec_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
'''


def _zip_info(name: str, *, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def build_checkpoint_container_bundle_v2(
    destination: Path,
    *,
    allow_test_root: bool = False,
) -> CheckpointContainerBundleV2:
    """Build one deterministic zipapp from the reviewed dependency-light core."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        with zipfile.ZipFile(temporary, mode="x") as bundle:
            bundle.writestr(_zip_info("dradar/__init__.py"), b"")
            for name in _BUNDLED_MODULES:
                payload = (source_root / name).read_bytes()
                bundle.writestr(_zip_info(f"dradar/{name}"), payload)
            bundle.writestr(
                _zip_info("__main__.py"),
                _main_source(allow_test_root=allow_test_root).encode("utf-8"),
            )
        temporary.chmod(0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination.chmod(0o600)
    payload = destination.read_bytes()
    return CheckpointContainerBundleV2(
        path=destination,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        allow_test_root=bool(allow_test_root),
    )


__all__ = [
    "CONTAINER_HELPER_SCHEMA_V2",
    "CheckpointContainerBundleV2",
    "build_checkpoint_container_bundle_v2",
]
