"""Pier transport for the optional container-native Checkpoint V2 helper.

The transport uses only Pier's public ``exec``/upload/download surface.  All
maintenance runs as root with an empty environment and fixed paths below the
container-native ``/run/dradar-checkpoint-v2`` tree.  Helper output is treated
as untrusted bounded JSON; stdout/stderr are never forwarded into telemetry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol

from .checkpoint_adapters_v2 import HarnessCheckpointContractV2
from .checkpoint_container_bundle_v2 import (
    CONTAINER_HELPER_SCHEMA_V2,
    build_checkpoint_container_bundle_v2,
)
from .checkpoint_runtime_v2 import (
    CONTAINER_EXPORT_ROOT,
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneError,
    CheckpointRestoreEvidenceV2,
    CheckpointRestoreRequestV2,
    ContainerSealedExportV2,
    revalidate_published_checkpoint_v2,
)


_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]{8,64}")
_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_ROOT_ENV = {
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "BASH_ENV": "/dev/null",
    "ENV": "/dev/null",
    "CDPATH": "",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _root_exec_diagnostic(
    result: object,
    *,
    operation: str,
) -> dict[str, str | int | bool]:
    """Return local-only comparison facts without persisting command output."""

    diagnostic: dict[str, str | int | bool] = {"operation": operation}
    return_code = getattr(result, "return_code", None)
    if isinstance(return_code, int) and not isinstance(return_code, bool):
        diagnostic["exit_code"] = return_code
    for label in ("stdout", "stderr"):
        value = getattr(result, label, None)
        if isinstance(value, str):
            encoded = value.encode("utf-8", errors="replace")
            diagnostic[f"{label}_bytes"] = len(encoded)
            diagnostic[f"{label}_sha256"] = hashlib.sha256(encoded).hexdigest()
    return diagnostic


class PierCheckpointEnvironmentV2(Protocol):
    default_user: object | None

    async def upload_file(self, source: Path, destination: str) -> Any: ...

    async def download_file(self, source: str, destination: Path) -> Any: ...

    async def exec(
        self,
        *,
        command: str,
        user: str,
        env: dict[str, str],
        cwd: str,
        timeout_sec: int,
    ) -> Any: ...


def _logical_child(*parts: str) -> PurePosixPath:
    for value in parts:
        if _PATH_SEGMENT_RE.fullmatch(value) is None:
            raise CheckpointDataPlaneError("capture", "transport_identity_invalid")
    return CONTAINER_EXPORT_ROOT.joinpath(*parts)


def _canonical_spec(value: dict[str, object]) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CheckpointDataPlaneError("capture", "transport_spec_invalid") from exc
    if len(payload) > 128 * 1024:
        raise CheckpointDataPlaneError("capture", "transport_spec_size_limit")
    return payload


def _parse_helper_response(raw: object, *, operation: str) -> dict[str, object]:
    if not isinstance(raw, str) or len(raw.encode("utf-8", errors="ignore")) > 16 * 1024:
        raise CheckpointDataPlaneError(operation, "helper_response_invalid")
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CheckpointDataPlaneError(operation, "helper_response_invalid")
    try:
        value = json.loads(lines[0])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError(operation, "helper_response_invalid") from exc
    if not isinstance(value, dict) or value.get("schema") != CONTAINER_HELPER_SCHEMA_V2:
        raise CheckpointDataPlaneError(operation, "helper_response_invalid")
    if value.get("ok") is not True:
        stage = value.get("stage")
        code = value.get("code")
        if (
            stage not in {"capture", "seal", "download", "verify", "publish", "cleanup", "restore"}
            or not isinstance(code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{2,63}", code) is None
        ):
            raise CheckpointDataPlaneError(operation, "helper_response_invalid")
        raise CheckpointDataPlaneError(stage, code)
    if value.get("operation") != operation:
        raise CheckpointDataPlaneError(operation, "helper_response_invalid")
    return value


class _PierHelperTransportV2:
    def __init__(
        self,
        environment: PierCheckpointEnvironmentV2,
        *,
        _test_filesystem_root: Path | None = None,
    ) -> None:
        self.environment = environment
        self._test_filesystem_root = (
            Path(_test_filesystem_root).absolute()
            if _test_filesystem_root is not None else None
        )

    @property
    def _helper_filesystem_root(self) -> str:
        return (
            os.fspath(self._test_filesystem_root)
            if self._test_filesystem_root is not None else "/"
        )

    async def _exec_root(
        self,
        command: str,
        *,
        stage: str,
        timeout_sec: int = 120,
        failure_code: str = "transport_root_exec_failed",
        diagnostic_operation: str = "root_exec",
    ) -> Any:
        clean = (
            "/usr/bin/env -i "
            "PATH=/usr/sbin:/usr/bin:/sbin:/bin "
            "HOME=/root LANG=C LC_ALL=C BASH_ENV=/dev/null ENV=/dev/null "
            "CDPATH= GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null "
            "/bin/bash --noprofile --norc -c "
            + shlex.quote(command)
        )
        try:
            result = await self.environment.exec(
                command=clean,
                user="root",
                env=dict(_ROOT_ENV),
                cwd="/",
                timeout_sec=timeout_sec,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise CheckpointDataPlaneError(
                stage,
                failure_code,
                diagnostic={
                    "operation": diagnostic_operation,
                    "transport_exception": type(exc).__name__[:64],
                },
            ) from exc
        if getattr(result, "return_code", None) != 0:
            # The helper emits only structured stage/code JSON.  Parse it when
            # possible but never persist arbitrary root stdout/stderr.
            try:
                _parse_helper_response(getattr(result, "stdout", None), operation=stage)
            except CheckpointDataPlaneError as exc:
                if exc.code != "helper_response_invalid":
                    raise
            raise CheckpointDataPlaneError(
                stage,
                failure_code,
                diagnostic=_root_exec_diagnostic(
                    result,
                    operation=diagnostic_operation,
                ),
            )
        return result

    async def _cleanup_noexcept(self, command: str) -> None:
        """Reap one exact cleanup through caller cancellation without leaking."""

        task = asyncio.create_task(
            self._exec_root(
                command,
                stage="cleanup",
                failure_code="transport_cleanup_failed",
                diagnostic_operation="cleanup",
            ),
            name="dradar-checkpoint-v2-cleanup",
        )
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            pass

    async def _install_and_run(
        self,
        *,
        operation_id: str,
        checkpoint_id: str,
        operation: str,
        spec: dict[str, object],
        timeout_sec: int,
    ) -> dict[str, object]:
        helper_root = _logical_child(checkpoint_id, "helpers", operation_id)
        upload_prefix = f"/tmp/.dradar-cpv2-{operation_id}"
        bundle_upload = upload_prefix + ".pyz.upload"
        spec_upload = upload_prefix + ".json.upload"
        bundle_remote = helper_root / "helper.pyz"
        spec_remote = helper_root / "spec.json"
        with tempfile.TemporaryDirectory(prefix="dradar-checkpoint-v2-helper-") as raw:
            local_root = Path(raw)
            bundle = build_checkpoint_container_bundle_v2(
                local_root / "helper.pyz",
                allow_test_root=self._test_filesystem_root is not None,
            )
            spec_path = local_root / "spec.json"
            spec_path.write_bytes(_canonical_spec(spec))
            spec_path.chmod(0o600)
            await self.environment.upload_file(bundle.path, bundle_upload)
            await self.environment.upload_file(spec_path, spec_upload)
        quoted = {name: shlex.quote(str(path)) for name, path in {
            "root": CONTAINER_EXPORT_ROOT,
            "checkpoint": _logical_child(checkpoint_id),
            "helpers": _logical_child(checkpoint_id, "helpers"),
            "helper_root": helper_root,
            "bundle_upload": PurePosixPath(bundle_upload),
            "spec_upload": PurePosixPath(spec_upload),
            "bundle": bundle_remote,
            "spec": spec_remote,
        }.items()}
        install = (
            "set -eu; umask 077; "
            "test -x /usr/bin/python3; test -x /usr/bin/install; "
            f"for path in {quoted['root']} {quoted['checkpoint']} {quoted['helpers']}; do "
            "if [ -e \"$path\" ] || [ -L \"$path\" ]; then "
            "test -d \"$path\" && test ! -L \"$path\"; "
            "else /usr/bin/install -d -o 0 -g 0 -m 0700 \"$path\"; fi; "
            "/usr/bin/chown 0:0 \"$path\"; /usr/bin/chmod 0700 \"$path\"; done; "
            f"test ! -e {quoted['helper_root']} && test ! -L {quoted['helper_root']}; "
            f"/usr/bin/install -d -o 0 -g 0 -m 0700 {quoted['helper_root']}; "
            f"test -f {quoted['bundle_upload']} && test ! -L {quoted['bundle_upload']}; "
            f"test -f {quoted['spec_upload']} && test ! -L {quoted['spec_upload']}; "
            f"test \"$(/usr/bin/sha256sum {quoted['bundle_upload']} | /usr/bin/awk '{{print $1}}')\" "
            f"= {shlex.quote(bundle.sha256)}; "
            f"/usr/bin/install -o 0 -g 0 -m 0555 {quoted['bundle_upload']} {quoted['bundle']}; "
            f"/usr/bin/install -o 0 -g 0 -m 0600 {quoted['spec_upload']} {quoted['spec']}; "
            f"/usr/bin/rm -f -- {quoted['bundle_upload']} {quoted['spec_upload']}"
        )
        try:
            await self._exec_root(
                install,
                stage=operation,
                failure_code="transport_helper_install_failed",
                diagnostic_operation="helper_install",
            )
            result = await self._exec_root(
                f"/usr/bin/python3 {shlex.quote(str(bundle_remote))} "
                f"{shlex.quote(str(spec_remote))}",
                stage=operation,
                timeout_sec=timeout_sec,
                failure_code="transport_helper_exec_failed",
                diagnostic_operation="helper_execute",
            )
            return _parse_helper_response(
                getattr(result, "stdout", None), operation=operation,
            )
        finally:
            cleanup = (
                "set -eu; "
                f"/usr/bin/rm -f -- {shlex.quote(bundle_upload)} {shlex.quote(spec_upload)}; "
                f"/usr/bin/rm -rf -- {shlex.quote(str(helper_root))}"
            )
            await self._cleanup_noexcept(cleanup)


class PierContainerCheckpointExporterV2(_PierHelperTransportV2):
    """Concrete Pier exporter for one reviewed Harness contract."""

    def __init__(
        self,
        environment: PierCheckpointEnvironmentV2,
        *,
        contract: HarnessCheckpointContractV2,
        base_commit: str,
        session_id: str | None,
        sensitive_values: Iterable[str | bytes] = (),
        worktree_path: str = "/app",
        _test_filesystem_root: Path | None = None,
    ) -> None:
        super().__init__(
            environment, _test_filesystem_root=_test_filesystem_root,
        )
        self.contract = contract
        self.base_commit = base_commit
        self.session_id = session_id
        self.sensitive_values = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else value
            for value in sensitive_values
            if isinstance(value, (str, bytes)) and len(value) >= 8
        )
        self.worktree_path = worktree_path
        self.adapter_version = contract.exporter_version
        self.checkpoint_abi = contract.checkpoint_abi
        self._requests: dict[str, CheckpointCaptureRequestV2] = {}

    async def capture_and_seal(
        self, request: CheckpointCaptureRequestV2,
    ) -> ContainerSealedExportV2:
        request.validate()
        if request.checkpoint_abi != self.checkpoint_abi:
            raise CheckpointDataPlaneError("capture", "adapter_abi_mismatch")
        checkpoint_root = _logical_child(request.checkpoint_id)
        capture_root = checkpoint_root / "staging" / request.capture_id
        sealed_root = checkpoint_root / "sealed"
        export_path = sealed_root / f"{request.capture_id}.tar.gz"
        setup = (
            "set -eu; umask 077; "
            f"for path in {shlex.quote(str(checkpoint_root))} "
            f"{shlex.quote(str(checkpoint_root / 'staging'))} "
            f"{shlex.quote(str(sealed_root))}; do "
            "if [ -e \"$path\" ] || [ -L \"$path\" ]; then "
            "test -d \"$path\" && test ! -L \"$path\"; "
            "else /usr/bin/install -d -o 0 -g 0 -m 0700 \"$path\"; fi; "
            "/usr/bin/chown 0:0 \"$path\"; /usr/bin/chmod 0700 \"$path\"; done; "
            f"test ! -e {shlex.quote(str(capture_root))}; "
            f"test ! -e {shlex.quote(str(export_path))}"
        )
        await self._exec_root(
            setup,
            stage="capture",
            failure_code="transport_capture_setup_failed",
            diagnostic_operation="capture_setup",
        )
        spec: dict[str, object] = {
            "schema": CONTAINER_HELPER_SCHEMA_V2,
            "operation": "capture",
            "filesystem_root": self._helper_filesystem_root,
            "harness": self.contract.harness,
            "provider": self.contract.provider,
            "worktree_path": self.worktree_path,
            "capture_root_path": str(capture_root),
            "export_path": str(export_path),
            "base_commit": self.base_commit,
            "captured_at": request.captured_at,
            "session_id": self.session_id,
            "sensitive_values": list(self.sensitive_values),
            "request": asdict(request),
        }
        try:
            response = await self._install_and_run(
                operation_id=request.capture_id,
                checkpoint_id=request.checkpoint_id,
                operation="capture",
                spec=spec,
                timeout_sec=180,
            )
            raw_export = response.get("export")
            if not isinstance(raw_export, dict):
                raise CheckpointDataPlaneError("capture", "helper_response_invalid")
            exported = ContainerSealedExportV2(**raw_export)
            exported.validate(request)
            if response.get("capture_id") != request.capture_id:
                raise CheckpointDataPlaneError("capture", "helper_response_invalid")
            self._requests[request.capture_id] = request
            return exported
        except BaseException:
            cleanup = (
                "set -eu; "
                f"/usr/bin/rm -rf -- {shlex.quote(str(capture_root))}; "
                f"/usr/bin/rm -f -- {shlex.quote(str(export_path))}"
            )
            await self._cleanup_noexcept(cleanup)
            raise

    async def download_export(
        self,
        export: ContainerSealedExportV2,
        destination: Path,
        *,
        max_bytes: int,
    ) -> None:
        request = self._requests.get(export.capture_id)
        if request is None:
            raise CheckpointDataPlaneError("download", "unknown_capture")
        export.validate(request)
        transfer = asyncio.create_task(
            self.environment.download_file(export.remote_path, destination),
            name=f"dradar-checkpoint-v2-download-{export.capture_id}",
        )
        caller_cancellation: asyncio.CancelledError | None = None
        while not transfer.done():
            try:
                await asyncio.shield(transfer)
            except asyncio.CancelledError as exc:
                caller_cancellation = caller_cancellation or exc
                continue
            except BaseException:
                break
        try:
            transfer.result()
        except asyncio.CancelledError as exc:
            raise CheckpointDataPlaneError("download", "transfer_cancelled") from exc
        except BaseException as exc:
            raise CheckpointDataPlaneError("download", "transfer_failed") from exc
        try:
            metadata = Path(destination).lstat()
            if (
                Path(destination).is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != export.archive_size
                or metadata.st_size > max_bytes
            ):
                raise CheckpointDataPlaneError("download", "download_file_unsafe")
            Path(destination).chmod(0o600)
        except OSError as exc:
            raise CheckpointDataPlaneError("download", "download_file_unsafe") from exc
        if caller_cancellation is not None:
            raise caller_cancellation

    async def discard_export(self, export: ContainerSealedExportV2) -> None:
        request = self._requests.pop(export.capture_id, None)
        if request is None:
            return
        export.validate(request)
        await self._exec_root(
            f"set -eu; /usr/bin/rm -f -- {shlex.quote(export.remote_path)}",
            stage="cleanup",
            failure_code="transport_cleanup_failed",
            diagnostic_operation="discard_export",
        )


class PierContainerCheckpointRestorerV2(_PierHelperTransportV2):
    """Offline-only Pier restorer; it cannot invoke a Provider/model process."""

    def __init__(
        self,
        environment: PierCheckpointEnvironmentV2,
        *,
        contract: HarnessCheckpointContractV2,
        base_commit: str,
        worktree_path: str = "/app",
        _test_filesystem_root: Path | None = None,
    ) -> None:
        super().__init__(
            environment, _test_filesystem_root=_test_filesystem_root,
        )
        self.contract = contract
        self.base_commit = base_commit
        self.worktree_path = worktree_path
        self.adapter_version = contract.restorer_version
        self.checkpoint_abi = contract.checkpoint_abi
        self.last_state_root: str | None = None

    async def restore_offline(
        self, request: CheckpointRestoreRequestV2,
    ) -> CheckpointRestoreEvidenceV2:
        if _IDENTIFIER_RE.fullmatch(request.restore_id) is None:
            raise CheckpointDataPlaneError("restore", "invalid_restore_id")
        manifest = revalidate_published_checkpoint_v2(
            request.published,
            expected_identity_fingerprint=request.expected_identity_fingerprint,
            expected_checkpoint_abi=self.checkpoint_abi,
        )
        capture_request = CheckpointCaptureRequestV2(
            checkpoint_id=str(manifest["checkpoint_id"]),
            checkpoint_lineage_id=str(manifest["checkpoint_lineage_id"]),
            snapshot_generation=int(manifest["snapshot_generation"]),
            capture_id=str(manifest["capture_id"]),
            identity_fingerprint=str(manifest["identity_fingerprint"]),
            checkpoint_abi=str(manifest["checkpoint_abi"]),
            recovery_capability=str(manifest["recovery_capability"]),
            native_state_schema=(
                str(manifest["native_state_schema"])
                if manifest.get("native_state_schema") is not None else None
            ),
            captured_at=str(manifest["captured_at"]),
        )
        operation_root = _logical_child(
            capture_request.checkpoint_id, "restore", request.restore_id,
        )
        archive_remote = operation_root / "incoming.tar.gz"
        state_root = operation_root / "state"
        storage_root = operation_root / "storage"
        await self._exec_root(
            "set -eu; umask 077; "
            f"test ! -e {shlex.quote(str(operation_root))}; "
            f"/usr/bin/install -d -o 0 -g 0 -m 0700 {shlex.quote(str(operation_root))}",
            stage="restore",
            failure_code="transport_restore_setup_failed",
            diagnostic_operation="restore_setup",
        )
        upload = f"/tmp/.dradar-cpv2-{request.restore_id}.archive.upload"
        try:
            await self.environment.upload_file(request.published.archive_path, upload)
            await self._exec_root(
                "set -eu; "
                f"test -f {shlex.quote(upload)} && test ! -L {shlex.quote(upload)}; "
                f"test \"$(/usr/bin/sha256sum {shlex.quote(upload)} | /usr/bin/awk '{{print $1}}')\" "
                f"= {shlex.quote(request.published.archive_sha256)}; "
                f"/usr/bin/install -o 0 -g 0 -m 0600 {shlex.quote(upload)} "
                f"{shlex.quote(str(archive_remote))}; /usr/bin/rm -f -- {shlex.quote(upload)}",
                stage="restore",
                failure_code="transport_restore_upload_failed",
                diagnostic_operation="restore_upload",
            )
            exported = ContainerSealedExportV2(
                capture_id=capture_request.capture_id,
                remote_path=str(archive_remote),
                archive_sha256=request.published.archive_sha256,
                archive_size=request.published.archive_bytes,
                manifest_sha256=request.published.manifest_sha256,
                capture_storage="container_native",
            )
            spec: dict[str, object] = {
                "schema": CONTAINER_HELPER_SCHEMA_V2,
                "operation": "restore",
                "filesystem_root": self._helper_filesystem_root,
                "harness": self.contract.harness,
                "provider": self.contract.provider,
                "worktree_path": self.worktree_path,
                "state_root_path": str(state_root),
                "storage_root_path": str(storage_root),
                "archive_path": str(archive_remote),
                "base_commit": self.base_commit,
                "expected_identity_fingerprint": request.expected_identity_fingerprint,
                "request": asdict(capture_request),
                "export": asdict(exported),
            }
            response = await self._install_and_run(
                operation_id=request.restore_id,
                checkpoint_id=capture_request.checkpoint_id,
                operation="restore",
                spec=spec,
                timeout_sec=180,
            )
            if (
                response.get("capture_id") != capture_request.capture_id
                or response.get("manifest_sha256") != request.published.manifest_sha256
                or response.get("identity_fingerprint")
                != request.expected_identity_fingerprint
                or response.get("paid_execution_started") is not False
            ):
                raise CheckpointDataPlaneError("restore", "restore_evidence_invalid")
            self.last_state_root = str(state_root)
            return CheckpointRestoreEvidenceV2(
                restore_id=request.restore_id,
                manifest_sha256=request.published.manifest_sha256,
                identity_fingerprint=request.expected_identity_fingerprint,
                restore_adapter_version=self.adapter_version,
                paid_execution_started=False,
            )
        except BaseException:
            await self._cleanup_noexcept(
                "set -eu; "
                f"/usr/bin/rm -f -- {shlex.quote(upload)}; "
                f"/usr/bin/rm -rf -- {shlex.quote(str(operation_root))}"
            )
            raise


__all__ = [
    "PierCheckpointEnvironmentV2",
    "PierContainerCheckpointExporterV2",
    "PierContainerCheckpointRestorerV2",
]
