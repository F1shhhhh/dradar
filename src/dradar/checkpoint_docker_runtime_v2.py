"""Docker CLI bridge for live, optional Checkpoint V2 shadow sampling.

The ordinary Pier process remains authoritative.  This module discovers one
running task container only by an exact bind mount below the current job root,
pins its full container and image IDs, and exposes the same narrow transport
surface used by the reviewed container helper.  It never starts a Provider or
changes assignment state.

RESTORE_TEST uses a fresh, network-disabled container from the exact captured
image.  The capture container's writable layer is therefore not available to
the restore and cannot create a false-positive restart result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .checkpoint_adapters_v2 import HarnessCheckpointContractV2
from .checkpoint_adapters_v2 import recovery_capability_for_capture_v2
from .checkpoint_pier_transport_v2 import (
    PierContainerCheckpointExporterV2,
    PierContainerCheckpointRestorerV2,
)
from .checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneError,
    CheckpointRestoreRequestV2,
    ContainerSealedExportV2,
)


_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._:-]{8,160}")
_KIMI_SESSION_ID_RE = re.compile(
    r"session_[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


@dataclass(frozen=True)
class DockerCommandResultV2:
    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class DockerContainerIdentityV2:
    container_id: str
    image_id: str


def _run_docker_v2(
    arguments: list[str],
    *,
    timeout_sec: float = 30,
    stage: str = "capture",
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["docker", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckpointDataPlaneError(
            stage,
            "docker_transport_unavailable",
            diagnostic={
                "operation": arguments[0] if arguments else "unknown",
                "transport_exception": "TimeoutExpired",
            },
        ) from exc
    except OSError as exc:
        raise CheckpointDataPlaneError(
            stage,
            "docker_transport_unavailable",
            diagnostic={
                "operation": arguments[0] if arguments else "unknown",
                "transport_exception": type(exc).__name__[:64],
            },
        ) from exc
    if result.returncode != 0:
        stdout = result.stdout.encode("utf-8", errors="replace")
        stderr = result.stderr.encode("utf-8", errors="replace")
        raise CheckpointDataPlaneError(
            stage,
            "docker_transport_failed",
            diagnostic={
                "operation": arguments[0] if arguments else "unknown",
                "exit_code": result.returncode,
                "stdout_bytes": len(stdout),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_bytes": len(stderr),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            },
        )
    if len(result.stdout.encode("utf-8", errors="ignore")) > 2 * 1024 * 1024:
        raise CheckpointDataPlaneError(stage, "docker_response_oversized")
    return result


def _inspect_containers_v2(
    container_ids: list[str],
    *,
    stage: str,
) -> list[dict]:
    if not container_ids or any(_CONTAINER_ID_RE.fullmatch(value) is None for value in container_ids):
        raise CheckpointDataPlaneError(stage, "container_identity_invalid")
    raw = _run_docker_v2(
        ["inspect", *container_ids], timeout_sec=30, stage=stage,
    ).stdout
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointDataPlaneError(stage, "container_inspect_invalid") from exc
    if not isinstance(value, list) or len(value) != len(container_ids):
        raise CheckpointDataPlaneError(stage, "container_inspect_invalid")
    if any(not isinstance(item, dict) for item in value):
        raise CheckpointDataPlaneError(stage, "container_inspect_invalid")
    return value


def _canonical_job_root(job_root: Path) -> Path:
    try:
        value = Path(job_root).resolve(strict=True)
        metadata = value.lstat()
    except OSError as exc:
        raise CheckpointDataPlaneError("capture", "job_root_unavailable") from exc
    if not value.is_dir() or value.is_symlink() or metadata.st_nlink < 1:
        raise CheckpointDataPlaneError("capture", "job_root_unsafe")
    return value


def _mount_belongs_to_job_v2(source: object, job_root: Path) -> bool:
    if not isinstance(source, str) or not source:
        return False
    try:
        path = Path(source).resolve(strict=True)
    except OSError:
        return False
    return path == job_root or path.is_relative_to(job_root)


def discover_pier_container_v2(job_root: Path) -> DockerContainerIdentityV2:
    """Return exactly one running container owned by this Pier job."""

    root = _canonical_job_root(job_root)
    listed = _run_docker_v2(
        ["ps", "--quiet", "--no-trunc"], timeout_sec=30, stage="capture",
    ).stdout.split()
    container_ids = [value for value in listed if _CONTAINER_ID_RE.fullmatch(value)]
    if not container_ids:
        raise CheckpointDataPlaneError("capture", "task_container_not_ready")
    matches: list[DockerContainerIdentityV2] = []
    for item in _inspect_containers_v2(container_ids, stage="capture"):
        container_id = item.get("Id")
        image_id = item.get("Image")
        running = (item.get("State") or {}).get("Running") is True
        mounts = item.get("Mounts")
        if (
            not running
            or not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or not isinstance(image_id, str)
            or _IMAGE_ID_RE.fullmatch(image_id) is None
            or not isinstance(mounts, list)
        ):
            continue
        if any(
            isinstance(mount, dict)
            and mount.get("Type") == "bind"
            and _mount_belongs_to_job_v2(mount.get("Source"), root)
            for mount in mounts
        ):
            matches.append(DockerContainerIdentityV2(container_id, image_id))
    if not matches:
        raise CheckpointDataPlaneError("capture", "task_container_not_ready")
    if len(matches) != 1:
        raise CheckpointDataPlaneError("capture", "task_container_ambiguous")
    return matches[0]


def docker_container_backend_v2() -> str:
    """Classify the active Docker context without weakening a failed probe."""

    context = _run_docker_v2(
        ["context", "show"], timeout_sec=10, stage="capture",
    ).stdout.strip().lower()
    if not context or len(context) > 128:
        raise CheckpointDataPlaneError("capture", "docker_context_invalid")
    return "orbstack" if "orbstack" in context else "docker"


class DockerCliCheckpointEnvironmentV2:
    """Pinned Docker container implementing the public Pier transport shape."""

    default_user = None

    def __init__(
        self,
        identity: DockerContainerIdentityV2,
        *,
        job_root: Path | None,
    ) -> None:
        if _CONTAINER_ID_RE.fullmatch(identity.container_id) is None:
            raise ValueError("checkpoint container id is invalid")
        if _IMAGE_ID_RE.fullmatch(identity.image_id) is None:
            raise ValueError("checkpoint image id is invalid")
        self.identity = identity
        self.job_root = _canonical_job_root(job_root) if job_root is not None else None

    def _validate(self, *, stage: str) -> None:
        item = _inspect_containers_v2(
            [self.identity.container_id], stage=stage,
        )[0]
        if (
            item.get("Id") != self.identity.container_id
            or item.get("Image") != self.identity.image_id
            or (item.get("State") or {}).get("Running") is not True
        ):
            raise CheckpointDataPlaneError(stage, "container_identity_changed")
        if self.job_root is not None:
            mounts = item.get("Mounts")
            if not isinstance(mounts, list) or not any(
                isinstance(mount, dict)
                and mount.get("Type") == "bind"
                and _mount_belongs_to_job_v2(mount.get("Source"), self.job_root)
                for mount in mounts
            ):
                raise CheckpointDataPlaneError(stage, "container_job_boundary_changed")

    async def upload_file(self, source: Path, destination: str):
        await asyncio.to_thread(self._validate, stage="capture")
        await asyncio.to_thread(
            _run_docker_v2,
            ["cp", os.fspath(source), f"{self.identity.container_id}:{destination}"],
            timeout_sec=60,
            stage="capture",
        )

    async def download_file(self, source: str, destination: Path):
        await asyncio.to_thread(self._validate, stage="download")
        await asyncio.to_thread(
            _run_docker_v2,
            ["cp", f"{self.identity.container_id}:{source}", os.fspath(destination)],
            timeout_sec=120,
            stage="download",
        )

    async def exec(self, *, command, user, env, cwd, timeout_sec):
        del env  # The helper command already constructs its own empty environment.
        await asyncio.to_thread(self._validate, stage="capture")
        arguments = ["exec", "-u", str(user), "-w", str(cwd)]
        arguments.extend((self.identity.container_id, "/bin/sh", "-c", str(command)))
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", *arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CheckpointDataPlaneError("capture", "docker_transport_unavailable") from exc
        return DockerCommandResultV2(result.returncode, result.stdout, result.stderr)


@dataclass(frozen=True)
class _CapturedRuntimeV2:
    identity: DockerContainerIdentityV2
    base_commit: str


@dataclass(frozen=True)
class _PreparedRuntimeV2:
    identity: DockerContainerIdentityV2
    base_commit: str
    session_id: str | None
    recovery_capability: str


async def _bounded_container_text_v2(
    environment: DockerCliCheckpointEnvironmentV2,
    command: str,
    *,
    max_bytes: int = 1024 * 1024,
) -> str:
    result = await environment.exec(
        command=command,
        user="root",
        env={},
        cwd="/",
        timeout_sec=15,
    )
    if result.return_code != 0:
        return ""
    if len(result.stdout.encode("utf-8", errors="ignore")) > max_bytes:
        return ""
    return result.stdout


async def _native_session_id_v2(
    environment: DockerCliCheckpointEnvironmentV2,
    contract: HarnessCheckpointContractV2,
) -> str | None:
    if contract.harness == "codex":
        raw = await _bounded_container_text_v2(
            environment,
            "head -c 1048576 /logs/agent/codex.txt 2>/dev/null || true",
        )
        values = []
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                continue
            value = event.get("thread_id") if isinstance(event, dict) else None
            if (
                isinstance(event, dict)
                and event.get("type") == "thread.started"
                and isinstance(value, str)
                and _SESSION_ID_RE.fullmatch(value)
            ):
                values.append(value)
        return values[0] if len(set(values)) == 1 else None
    if contract.harness == "dsh":
        raw = await _bounded_container_text_v2(
            environment,
            "head -c 512 /logs/agent/dsh-session-id 2>/dev/null || true",
            max_bytes=512,
        )
        value = raw.strip()
        return value if _SESSION_ID_RE.fullmatch(value) else None
    if contract.harness == "zcode":
        raw = await _bounded_container_text_v2(
            environment,
            "head -c 512 /logs/agent/zcode-session-id 2>/dev/null || true",
            max_bytes=512,
        )
        value = raw.strip()
        return value if _SESSION_ID_RE.fullmatch(value) else None
    if contract.harness == "kimi-code":
        raw = await _bounded_container_text_v2(
            environment,
            "tail -c 1048576 /tmp/dradar-kimi-home/session_index.jsonl "
            "2>/dev/null || true",
        )
        active: dict[str, tuple[str, str] | None] = {}
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            session_id = event.get("sessionId")
            if (
                not isinstance(session_id, str)
                or _KIMI_SESSION_ID_RE.fullmatch(session_id) is None
            ):
                continue
            if event.get("deleted") is True:
                active[session_id] = None
                continue
            session_dir = event.get("sessionDir")
            work_dir = event.get("workDir")
            expected_dir = os.path.join(
                "/tmp/dradar-kimi-home/sessions", session_id,
            )
            if (
                isinstance(session_dir, str)
                and os.path.normpath(session_dir) == expected_dir
                and work_dir == "/app"
            ):
                active[session_id] = (session_dir, work_dir)
        candidates = [key for key, value in active.items() if value is not None]
        return candidates[0] if len(candidates) == 1 else None
    return None


async def _material_native_artifacts_v2(
    environment: DockerCliCheckpointEnvironmentV2,
    contract: HarnessCheckpointContractV2,
) -> frozenset[str]:
    present: set[str] = set()
    for artifact in contract.artifacts:
        source = shlex.quote(artifact.source_path)
        if artifact.kind == "file":
            command = f"test -f {source} && test ! -L {source} && test -s {source}"
        else:
            command = (
                f"test -d {source} && test ! -L {source} && "
                f"find {source} -xdev -type f -size +0c -print -quit | grep -q ."
            )
        result = await environment.exec(
            command=command,
            user="root",
            env={},
            cwd="/",
            timeout_sec=15,
        )
        if result.return_code == 0:
            present.add(artifact.name)
    return frozenset(present)


class DockerCliLazyCheckpointExporterV2:
    """Bind a capture to the exact live task container at sample time."""

    def __init__(
        self,
        *,
        job_root: Path,
        contract: HarnessCheckpointContractV2,
        ready_timeout_sec: float = 180.0,
    ) -> None:
        # Pier creates the job directory after its subprocess starts.  Keep
        # the absolute intended path without creating it; every actual sample
        # resolves and validates the now-existing directory before discovery.
        self.job_root = Path(job_root).absolute()
        if self.job_root.exists() and self.job_root.is_symlink():
            raise CheckpointDataPlaneError("capture", "job_root_unsafe")
        if (
            not isinstance(ready_timeout_sec, (int, float))
            or isinstance(ready_timeout_sec, bool)
            or not 0 <= float(ready_timeout_sec) <= 600
        ):
            raise ValueError("checkpoint container readiness timeout is invalid")
        self.ready_timeout_sec = float(ready_timeout_sec)
        self.contract = contract
        self.adapter_version = contract.exporter_version
        self.checkpoint_abi = contract.checkpoint_abi
        self._delegates: dict[str, PierContainerCheckpointExporterV2] = {}
        self._runtime: dict[str, _CapturedRuntimeV2] = {}
        self._prepared: _PreparedRuntimeV2 | None = None

    async def _prepare(self) -> _PreparedRuntimeV2:
        deadline = time.monotonic() + self.ready_timeout_sec
        while True:
            try:
                identity = await asyncio.to_thread(
                    discover_pier_container_v2, self.job_root,
                )
                break
            except CheckpointDataPlaneError as exc:
                if (
                    exc.code != "task_container_not_ready"
                    or time.monotonic() >= deadline
                ):
                    raise
                await asyncio.sleep(2.0)
        environment = DockerCliCheckpointEnvironmentV2(
            identity, job_root=self.job_root,
        )
        result = await environment.exec(
            command="git -C /app rev-parse HEAD",
            user="root",
            env={},
            cwd="/",
            timeout_sec=15,
        )
        base_commit = result.stdout.strip() if result.return_code == 0 else ""
        if _GIT_COMMIT_RE.fullmatch(base_commit) is None:
            raise CheckpointDataPlaneError("capture", "task_base_commit_invalid")
        session_id = await _native_session_id_v2(environment, self.contract)
        present = await _material_native_artifacts_v2(environment, self.contract)
        capability = recovery_capability_for_capture_v2(
            self.contract,
            present_artifacts=present,
            has_session_id=session_id is not None,
        )
        return _PreparedRuntimeV2(
            identity=identity,
            base_commit=base_commit,
            session_id=session_id,
            recovery_capability=capability,
        )

    async def recovery_facts(self) -> tuple[str, str]:
        prepared = await self._prepare()
        self._prepared = prepared
        return prepared.recovery_capability, self.contract.native_state_schema

    async def capture_and_seal(
        self, request: CheckpointCaptureRequestV2,
    ) -> ContainerSealedExportV2:
        prepared = self._prepared or await self._prepare()
        self._prepared = None
        identity = prepared.identity
        environment = DockerCliCheckpointEnvironmentV2(
            identity, job_root=self.job_root,
        )
        if request.recovery_capability != prepared.recovery_capability:
            raise CheckpointDataPlaneError("capture", "recovery_capability_changed")
        delegate = PierContainerCheckpointExporterV2(
            environment,
            contract=self.contract,
            base_commit=prepared.base_commit,
            session_id=prepared.session_id,
        )
        exported = await delegate.capture_and_seal(request)
        self._delegates[request.capture_id] = delegate
        self._runtime[request.capture_id] = _CapturedRuntimeV2(
            identity=identity, base_commit=prepared.base_commit,
        )
        return exported

    async def download_export(self, export, destination, *, max_bytes):
        delegate = self._delegates.get(export.capture_id)
        if delegate is None:
            raise CheckpointDataPlaneError("download", "capture_delegate_missing")
        await delegate.download_export(export, destination, max_bytes=max_bytes)

    async def discard_export(self, export):
        delegate = self._delegates.pop(export.capture_id, None)
        if delegate is None:
            raise CheckpointDataPlaneError("cleanup", "capture_delegate_missing")
        await delegate.discard_export(export)

    def restorer(self) -> "DockerCliDisposableCheckpointRestorerV2":
        return DockerCliDisposableCheckpointRestorerV2(self)


class DockerCliDisposableCheckpointRestorerV2:
    """Restore one capture in a fresh exact-image, network-disabled container."""

    def __init__(self, exporter: DockerCliLazyCheckpointExporterV2) -> None:
        self.exporter = exporter
        self.adapter_version = exporter.contract.restorer_version
        self.checkpoint_abi = exporter.contract.checkpoint_abi

    @staticmethod
    async def _remove_container(container_id: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(
            _run_docker_v2,
            ["rm", "-f", container_id],
            timeout_sec=30,
            stage="cleanup",
        ))
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = exc
                continue
            except Exception:
                break
        try:
            task.result()
        except Exception as exc:
            raise CheckpointDataPlaneError(
                "restore", "restore_cleanup_failed",
            ) from exc
        if cancellation is not None:
            raise cancellation

    async def restore_offline(self, request: CheckpointRestoreRequestV2):
        runtime = self.exporter._runtime.get(request.published.capture_id)
        if runtime is None:
            raise CheckpointDataPlaneError("restore", "capture_runtime_missing")
        name = f"dradar-cpv2-restore-{uuid.uuid4().hex[:20]}"
        started = await asyncio.to_thread(
            _run_docker_v2,
            [
                "run", "-d", "--rm", "--network", "none", "--name", name,
                "--entrypoint", "/bin/sh", runtime.identity.image_id,
                "-c", "sleep 300",
            ],
            timeout_sec=60,
            stage="restore",
        )
        container_id = started.stdout.strip()
        if _CONTAINER_ID_RE.fullmatch(container_id) is None:
            await asyncio.to_thread(
                _run_docker_v2,
                ["rm", "-f", name],
                timeout_sec=30,
                stage="cleanup",
            )
            raise CheckpointDataPlaneError("restore", "restore_container_invalid")
        identity = DockerContainerIdentityV2(
            container_id=container_id,
            image_id=runtime.identity.image_id,
        )
        environment = DockerCliCheckpointEnvironmentV2(identity, job_root=None)
        restorer = PierContainerCheckpointRestorerV2(
            environment,
            contract=self.exporter.contract,
            base_commit=runtime.base_commit,
            worktree_path="/app",
        )
        evidence = None
        primary_error: BaseException | None = None
        try:
            evidence = await restorer.restore_offline(request)
        except BaseException as exc:
            primary_error = exc
        finally:
            try:
                await self._remove_container(container_id)
            except BaseException as cleanup_error:
                if isinstance(cleanup_error, asyncio.CancelledError):
                    raise
                raise cleanup_error from primary_error
        if primary_error is not None:
            raise primary_error
        if evidence is None:  # pragma: no cover - defensive protocol fence
            raise CheckpointDataPlaneError("restore", "restore_evidence_missing")
        return evidence


__all__ = [
    "DockerCliCheckpointEnvironmentV2",
    "DockerCliDisposableCheckpointRestorerV2",
    "DockerCliLazyCheckpointExporterV2",
    "DockerContainerIdentityV2",
    "discover_pier_container_v2",
    "docker_container_backend_v2",
]
