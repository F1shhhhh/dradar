from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import queue
import shutil
import stat
import tarfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

import dradar.checkpoint_runtime_v2 as runtime
from dradar.checkpoint_v2 import (
    CHECKPOINT_CORE_ABI_V2,
    CheckpointGenerationRefV2,
    CheckpointRetentionAcknowledgementV2,
    negotiate_checkpoint_activation_v2,
)
from dradar.checkpoint_runtime_v2 import (
    CheckpointCaptureRequestV2,
    CheckpointDataPlaneError,
    CheckpointDataPlaneV2,
    CheckpointObservationRuntimeV2,
    CheckpointRetentionPolicyV2,
    CheckpointRestoreEvidenceV2,
    CheckpointRestoreRequestV2,
    CheckpointShadowStorageBudgetV2,
    ContainerSealedExportV2,
    PublishedCheckpointV2,
    apply_checkpoint_generation_retention_v2,
    publish_checkpoint_export_v2,
    record_local_checkpoint_failure_v2,
    release_shadow_generation_v2,
    checkpoint_observation_payload_v2,
    checkpoint_restore_observation_payload_v2,
    load_exact_published_checkpoint_v2,
    next_shadow_generation_v2,
    run_mainline_with_periodic_shadow_captures_v2,
    run_mainline_with_shadow_checkpoint_v2,
    seal_checkpoint_export_v2,
)


def _publish_checkpoint_process(
    archive_path: str,
    storage_root: str,
    request: CheckpointCaptureRequestV2,
    export: ContainerSealedExportV2,
    start_event,
    result_queue,
    process_index: int,
) -> None:
    if not start_event.wait(timeout=30):
        result_queue.put(("error", process_index, "barrier_timeout"))
        return
    try:
        published = publish_checkpoint_export_v2(
            Path(archive_path),
            Path(storage_root),
            request,
            export,
            authoritative=False,
        )
    except CheckpointDataPlaneError as exc:
        result_queue.put(("error", process_index, exc.code))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        result_queue.put((
            "exception", process_index,
            f"{type(exc).__name__}: {str(exc)[:256]}",
        ))
    else:
        result_queue.put((
            "ok", process_index, published.capture_id,
            published.manifest_sha256, published.selected,
        ))


def _crash_publication_before_current_process(
    archive_path: str,
    storage_root: str,
    request: CheckpointCaptureRequestV2,
    export: ContainerSealedExportV2,
    crash_reached,
) -> None:
    real_replace = runtime.os.replace

    def crash_before_current(source, destination):
        if Path(destination).name == "CURRENT":
            crash_reached.set()
            os._exit(93)
        return real_replace(source, destination)

    runtime.os.replace = crash_before_current
    publish_checkpoint_export_v2(
        Path(archive_path),
        Path(storage_root),
        request,
        export,
        authoritative=True,
    )
    os._exit(94)


def _request(**updates) -> CheckpointCaptureRequestV2:
    values = {
        "checkpoint_id": "checkpoint-0001",
        "checkpoint_lineage_id": "lineage-0001",
        "snapshot_generation": 1,
        "capture_id": "capture-0001",
        "identity_fingerprint": "a" * 64,
        "checkpoint_abi": "dradar-checkpoint-v2/zcode/1",
        "recovery_capability": "NATIVE_VALID",
        "native_state_schema": "zcode-session/1",
        "captured_at": "2026-08-23T12:00:00+00:00",
    }
    values.update(updates)
    return CheckpointCaptureRequestV2(**values)


def _source(root: Path) -> Path:
    source = root / "capture"
    (source / "workspace").mkdir(parents=True)
    (source / "native" / "sessions").mkdir(parents=True)
    (source / "workspace" / "model.patch").write_bytes(b"diff --git a/a b/a\n")
    (source / "native" / "sessions" / "state.json").write_text(
        '{"step":3}\n', encoding="utf-8",
    )
    return source


def _seal(root: Path, request: CheckpointCaptureRequestV2 | None = None):
    request = request or _request()
    container_root = root / "container-native"
    container_root.mkdir(parents=True, mode=0o700)
    container_root.chmod(0o700)
    export_root = container_root / "sealed"
    export_root.mkdir(parents=True, mode=0o700)
    export_root.chmod(0o700)
    archive = export_root / f"{request.capture_id}.tar.gz"
    exported = seal_checkpoint_export_v2(
        _source(root), archive, request,
        container_export_root=container_root,
    )
    return request, archive, exported


class FakeExporter:
    adapter_version = "fake-zcode-exporter/1"
    checkpoint_abi = "dradar-checkpoint-v2/zcode/1"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.local_archive: Path | None = None
        self.discarded = 0
        self.capture_calls = 0
        self.download_calls = 0
        self.capture_error: Exception | None = None
        self.download_error: Exception | None = None
        self.download_started: asyncio.Event | None = None
        self.download_continue: asyncio.Event | None = None

    async def capture_and_seal(
        self, request: CheckpointCaptureRequestV2,
    ) -> ContainerSealedExportV2:
        self.capture_calls += 1
        if self.capture_error is not None:
            raise self.capture_error
        _, archive, exported = _seal(self.root, request)
        self.local_archive = archive
        return replace(
            exported,
            remote_path=(
                f"/run/dradar-checkpoint-v2/{request.checkpoint_id}/sealed/"
                f"{request.capture_id}.tar.gz"
            ),
        )

    async def download_export(
        self,
        export: ContainerSealedExportV2,
        destination: Path,
        *,
        max_bytes: int,
    ) -> None:
        self.download_calls += 1
        assert self.local_archive is not None
        if self.download_started is not None:
            self.download_started.set()
        if self.download_continue is not None:
            await self.download_continue.wait()
        if self.download_error is not None:
            raise self.download_error
        if self.local_archive.stat().st_size > max_bytes:
            raise CheckpointDataPlaneError("download", "archive_size_limit")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(destination, flags, 0o600)
        try:
            with self.local_archive.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    os.write(descriptor, chunk)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    async def discard_export(self, export: ContainerSealedExportV2) -> None:
        self.discarded += 1
        if self.local_archive is not None:
            self.local_archive.unlink(missing_ok=True)


class FakeRestorer:
    adapter_version = "fake-zcode-restorer/1"
    checkpoint_abi = "dradar-checkpoint-v2/zcode/1"

    def __init__(self) -> None:
        self.paid_execution_started = False

    async def restore_offline(
        self, request: CheckpointRestoreRequestV2,
    ) -> CheckpointRestoreEvidenceV2:
        assert (request.published.payload_root / "workspace/model.patch").is_file()
        return CheckpointRestoreEvidenceV2(
            restore_id=request.restore_id,
            manifest_sha256=request.published.manifest_sha256,
            identity_fingerprint=request.expected_identity_fingerprint,
            restore_adapter_version=self.adapter_version,
            paid_execution_started=self.paid_execution_started,
        )


def _activation(local: str, server: str | None = None, *, controlled=False):
    return negotiate_checkpoint_activation_v2(
        local_mode=local,
        server_mode=server or local,
        controlled_account=controlled,
    )


def _retention_ref(
    published: PublishedCheckpointV2,
) -> CheckpointGenerationRefV2:
    return CheckpointGenerationRefV2(
        checkpoint_id=published.checkpoint_id,
        snapshot_generation=published.snapshot_generation,
        manifest_sha256=published.manifest_sha256,
    )


def _retention_ack(
    *,
    delete: tuple[PublishedCheckpointV2, ...],
    retain: tuple[PublishedCheckpointV2, ...] = (),
    operation_id: str = "retention-operation-0001",
) -> CheckpointRetentionAcknowledgementV2:
    return CheckpointRetentionAcknowledgementV2(
        assignment_id="assignment-0001",
        operation_id=operation_id,
        owner_epoch_observed=4,
        current_owner_epoch=5,
        delete_generations=tuple(_retention_ref(item) for item in delete),
        retain_generations=tuple(_retention_ref(item) for item in retain),
        result_evidence_release=False,
        upload_intent_id=None,
        submission_id=None,
    )


def _authoritative_generations(
    root: Path,
    count: int,
) -> tuple[Path, tuple[PublishedCheckpointV2, ...]]:
    storage_root = root / "host"
    plane = CheckpointDataPlaneV2(
        activation=_activation("canary", "canary", controlled=True),
        storage_root=storage_root,
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2,
            minimum_free_bytes=0,
        ),
    )
    published: list[PublishedCheckpointV2] = []
    for generation in range(1, count + 1):
        observation = asyncio.run(plane.observe_capture(
            _request(
                snapshot_generation=generation,
                capture_id=f"capture-auth-{generation:04d}",
            ),
            FakeExporter(root / f"container-authoritative-{generation}"),
        ))
        assert observation.status == "sealed"
        assert observation.published is not None
        published.append(observation.published)
    return storage_root, tuple(published)


def _archive_sha(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _rewrite_archive(
    original: Path,
    destination: Path,
    mutate,
) -> None:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(original, "r:gz") as archive:
        for member in archive:
            source = archive.extractfile(member) if member.isfile() else None
            members.append((member, source.read() if source is not None else None))
    members = mutate(members)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w:", format=tarfile.PAX_FORMAT) as archive:
                for member, content in members:
                    archive.addfile(member, io.BytesIO(content) if content is not None else None)


def _nested_untracked_archive(
    entries: list[tuple[str, bytes | None, str]],
) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w:", format=tarfile.PAX_FORMAT) as archive:
            for name, content, kind in entries:
                info = tarfile.TarInfo(name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.pax_headers = {}
                if kind == "directory":
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o700
                    archive.addfile(info)
                elif kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "target"
                    info.mode = 0o600
                    archive.addfile(info)
                else:
                    assert content is not None
                    info.size = len(content)
                    info.mode = 0o600
                    archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def test_off_mode_is_a_strict_noop(tmp_path: Path) -> None:
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("off", "on"),
        storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "skipped"
    assert observed.mainline_may_continue is True
    assert exporter.capture_calls == exporter.download_calls == 0
    assert not (tmp_path / "host").exists()


def test_observe_mode_seals_verifies_and_publishes_non_authoritative_snapshot(
    tmp_path: Path,
) -> None:
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "on"),
        storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "sealed"
    assert observed.mainline_may_continue is True
    assert observed.remote_cleanup == "discarded"
    assert observed.published is not None
    assert observed.published.authoritative is False
    assert (
        observed.published.payload_root / "workspace/model.patch"
    ).read_bytes() == b"diff --git a/a b/a\n"
    receipt = json.loads(
        (observed.published.root / "publication.json").read_text()
    )
    assert receipt["authoritative"] is False
    assert exporter.discarded == 1
    assert not list((tmp_path / "host" / ".downloads").iterdir())
    assert next_shadow_generation_v2(
        plane.storage_root, "checkpoint-0001",
    ) == 2
    assert observed.published.root == (
        plane.storage_root
        / "checkpoints"
        / "checkpoint-0001"
        / "generations"
        / "generation-00000000000000000001"
    )

    payload = checkpoint_observation_payload_v2(
        _request(),
        observed,
        plane.activation,
        CheckpointObservationRuntimeV2(
            assignment_id="assignment-0001",
            operation_id="operation-observe-0001",
            elapsed_ms=83,
            platform="macos",
            container_backend="orbstack",
            client_version="0.5.98",
            adapter_version=exporter.adapter_version,
        ),
    )
    assert payload["status"] == "sealed"
    assert payload["observation_kind"] == "capture"
    assert payload["rollout_mode"] == "observe"
    assert payload["capture_storage"] == "container_native"
    assert payload["authoritative"] is False
    assert payload["selected_local"] is True
    assert payload["file_count"] == 2
    assert payload["payload_bytes"] > 0
    assert payload["failure_code"] is None


def test_verified_shadow_generation_is_released_after_experiment_use(
    tmp_path: Path,
) -> None:
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "observe"),
        storage_root=tmp_path / "host",
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2,
            minimum_free_bytes=0,
        ),
    )
    request = _request()
    observation = asyncio.run(plane.observe_capture(
        request, FakeExporter(tmp_path / "container"),
    ))
    assert observation.status == "sealed"
    assert observation.published is not None
    published = observation.published

    with pytest.raises(
        CheckpointDataPlaneError, match="shadow_release_identity_invalid",
    ):
        release_shadow_generation_v2(
            tmp_path / "other-shadow-root",
            published,
            expected_identity_fingerprint=request.identity_fingerprint,
            expected_checkpoint_abi=request.checkpoint_abi,
        )
    assert published.root.is_dir()
    assert release_shadow_generation_v2(
        plane.storage_root,
        published,
        expected_identity_fingerprint=request.identity_fingerprint,
        expected_checkpoint_abi=request.checkpoint_abi,
    ) is True
    assert not published.root.exists()
    assert not (published.root.parent.parent / "CURRENT").exists()
    assert release_shadow_generation_v2(
        plane.storage_root,
        published,
        expected_identity_fingerprint=request.identity_fingerprint,
        expected_checkpoint_abi=request.checkpoint_abi,
    ) is False


def test_shadow_release_resumes_after_quarantine_cleanup_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "observe"),
        storage_root=tmp_path / "host",
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2,
            minimum_free_bytes=0,
        ),
    )
    request = _request()
    observation = asyncio.run(plane.observe_capture(
        request, FakeExporter(tmp_path / "container"),
    ))
    assert observation.published is not None
    published = observation.published
    real_rmtree = runtime.shutil.rmtree
    failed = False

    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if Path(path).name.startswith(".shadow-release-") and not failed:
            failed = True
            raise OSError("injected cleanup crash")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(runtime.shutil, "rmtree", fail_once)
    with pytest.raises(
        CheckpointDataPlaneError, match="shadow_release_cleanup_failed",
    ):
        release_shadow_generation_v2(
            plane.storage_root,
            published,
            expected_identity_fingerprint=request.identity_fingerprint,
            expected_checkpoint_abi=request.checkpoint_abi,
        )
    assert not published.root.exists()
    assert len(list(published.root.parent.glob(".shadow-release-*"))) == 1

    monkeypatch.setattr(runtime.shutil, "rmtree", real_rmtree)
    assert release_shadow_generation_v2(
        plane.storage_root,
        published,
        expected_identity_fingerprint=request.identity_fingerprint,
        expected_checkpoint_abi=request.checkpoint_abi,
    ) is True
    assert list(published.root.parent.glob(".shadow-release-*")) == []


def test_shadow_release_refuses_authoritative_generation(tmp_path: Path) -> None:
    storage_root, published = _authoritative_generations(tmp_path, 1)
    generation = published[0]

    with pytest.raises(
        CheckpointDataPlaneError, match="shadow_release_identity_invalid",
    ):
        release_shadow_generation_v2(
            storage_root,
            generation,
            expected_identity_fingerprint="a" * 64,
            expected_checkpoint_abi="dradar-checkpoint-v2/zcode/1",
        )
    assert generation.root.is_dir()
    assert (storage_root / "checkpoints" / generation.checkpoint_id / "CURRENT").is_file()


def test_shadow_capture_disk_pressure_is_fail_open_before_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "on"),
        storage_root=tmp_path / "host",
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2,
            minimum_free_bytes=1024,
        ),
    )
    monkeypatch.setattr(
        runtime.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 1024})(),
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "failed"
    assert observed.stage == "capture"
    assert observed.code == "disk_full"
    assert observed.mainline_may_continue is True
    assert exporter.capture_calls == 0


def test_global_shadow_budget_stops_optional_capture_before_export(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "shadow"
    retained = global_root / "retained-assignment"
    retained.mkdir(parents=True, mode=0o700)
    global_root.chmod(0o700)
    retained.chmod(0o700)
    evidence = retained / "failed-snapshot.bin"
    evidence.touch(mode=0o600)
    evidence.chmod(0o600)
    os.truncate(evidence, 61 * 1024 * 1024)
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "observe"),
        storage_root=global_root / "current-assignment",
        limits=replace(
            runtime.DEFAULT_PACKAGE_LIMITS_V2,
            max_archive_bytes=4 * 1024 * 1024,
        ),
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2, minimum_free_bytes=0,
        ),
        shadow_budget_root=global_root,
        shadow_budget=CheckpointShadowStorageBudgetV2(
            max_bytes=64 * 1024 * 1024, max_entries=100,
        ),
    )

    observed = asyncio.run(plane.observe_capture(_request(), exporter))

    assert observed.status == "failed"
    assert observed.code == "shadow_storage_budget_exhausted"
    assert exporter.capture_calls == 0
    assert evidence.stat().st_size == 61 * 1024 * 1024
    assert not plane.storage_root.exists()


def test_global_shadow_budget_preserves_unsafe_unknown_material(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "shadow"
    global_root.mkdir(mode=0o700)
    unknown = global_root / "unknown-assignment"
    unknown.symlink_to(tmp_path / "outside", target_is_directory=True)
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "observe"),
        storage_root=global_root / "current-assignment",
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2, minimum_free_bytes=0,
        ),
        shadow_budget_root=global_root,
    )

    observed = asyncio.run(plane.observe_capture(_request(), exporter))

    assert observed.status == "failed"
    assert observed.code == "shadow_storage_unsafe"
    assert exporter.capture_calls == 0
    assert unknown.is_symlink()
    assert not plane.storage_root.exists()


def test_global_shadow_budget_guard_is_nonblocking_across_writers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shadow"
    budget = CheckpointShadowStorageBudgetV2(
        max_bytes=64 * 1024 * 1024, max_entries=100,
    )

    with runtime.shadow_storage_budget_guard_v2(
        root, required_bytes=1024, budget=budget,
    ):
        with pytest.raises(CheckpointDataPlaneError) as failure:
            with runtime.shadow_storage_budget_guard_v2(
                root, required_bytes=1024, budget=budget,
            ):
                raise AssertionError("contended shadow writer entered")

    assert failure.value.code == "shadow_storage_busy"


def test_shadow_generation_restarts_after_published_directory_without_current(
    tmp_path: Path,
) -> None:
    generations = (
        tmp_path / "host" / "checkpoints" / "checkpoint-0001" / "generations"
    )
    generations.mkdir(parents=True, mode=0o700)
    generations.chmod(0o700)
    # Simulate a crash after the generation rename and before CURRENT.  Even
    # malformed evidence is preserved and its name cannot be reused.
    (generations / "generation-00000000000000000003").mkdir(mode=0o700)
    assert next_shadow_generation_v2(
        tmp_path / "host", "checkpoint-0001",
    ) == 4


def test_shadow_generation_discovery_rejects_symlinked_storage(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    checkpoint_root = tmp_path / "host" / "checkpoints" / "checkpoint-0001"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "generations").symlink_to(
        target, target_is_directory=True,
    )
    with pytest.raises(CheckpointDataPlaneError, match="unsafe_storage"):
        next_shadow_generation_v2(tmp_path / "host", "checkpoint-0001")


def test_shadow_retention_keeps_two_newest_generations(
    tmp_path: Path,
) -> None:
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe", "on"),
        storage_root=tmp_path / "host",
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2,
            minimum_free_bytes=0,
        ),
    )
    published = []
    for generation in range(1, 5):
        exporter = FakeExporter(tmp_path / f"container-{generation}")
        request = _request(
            snapshot_generation=generation,
            capture_id=f"capture-{generation:04d}",
        )
        observed = asyncio.run(plane.observe_capture(request, exporter))
        assert observed.status == "sealed"
        assert observed.published is not None
        published.append(observed.published)
    generations = published[-1].root.parent
    assert sorted(path.name for path in generations.iterdir()) == [
        "generation-00000000000000000003",
        "generation-00000000000000000004",
    ]
    current = json.loads(
        (published[-1].root.parent.parent / "CURRENT").read_text()
    )
    assert current["generation"] == 4


def test_authoritative_generations_are_never_pruned_locally(
    tmp_path: Path,
) -> None:
    plane = CheckpointDataPlaneV2(
        activation=_activation("canary", "canary", controlled=True),
        storage_root=tmp_path / "host",
        retention=CheckpointRetentionPolicyV2(
            shadow_generations=2,
            minimum_free_bytes=0,
        ),
    )
    roots = []
    for generation in range(1, 4):
        exporter = FakeExporter(tmp_path / f"container-authoritative-{generation}")
        observed = asyncio.run(plane.observe_capture(
            _request(
                snapshot_generation=generation,
                capture_id=f"capture-auth-{generation:04d}",
            ),
            exporter,
        ))
        assert observed.status == "sealed"
        assert observed.published is not None
        assert observed.published.authoritative is True
        roots.append(observed.published.root)
    assert all(path.is_dir() for path in roots)


def test_restart_loads_only_the_exact_server_selected_generation(
    tmp_path: Path,
) -> None:
    storage_root, published = _authoritative_generations(tmp_path, 1)
    expected = published[0]
    loaded = load_exact_published_checkpoint_v2(
        storage_root,
        checkpoint_id="checkpoint-0001",
        checkpoint_lineage_id="lineage-0001",
        snapshot_generation=1,
        capture_id="capture-auth-0001",
        manifest_sha256=expected.manifest_sha256,
        expected_identity_fingerprint="a" * 64,
        expected_checkpoint_core_abi=CHECKPOINT_CORE_ABI_V2,
        expected_checkpoint_abi="dradar-checkpoint-v2/zcode/1",
        expected_recovery_capability="NATIVE_VALID",
        expected_native_state_schema="zcode-session/1",
    )
    assert loaded == expected

    with pytest.raises(
        CheckpointDataPlaneError, match="publication_receipt_invalid",
    ):
        load_exact_published_checkpoint_v2(
            storage_root,
            checkpoint_id="checkpoint-0001",
            checkpoint_lineage_id="lineage-0001",
            snapshot_generation=1,
            capture_id="capture-wrong-0001",
            manifest_sha256=expected.manifest_sha256,
            expected_identity_fingerprint="a" * 64,
            expected_checkpoint_core_abi=CHECKPOINT_CORE_ABI_V2,
            expected_checkpoint_abi="dradar-checkpoint-v2/zcode/1",
            expected_recovery_capability="NATIVE_VALID",
            expected_native_state_schema="zcode-session/1",
        )


def test_server_ack_deletes_only_exact_released_generation_and_replays(
    tmp_path: Path,
) -> None:
    storage_root, published = _authoritative_generations(tmp_path, 3)
    first, second, third = published
    acknowledgement = _retention_ack(
        delete=(first,), retain=(second, third),
    )

    applied = apply_checkpoint_generation_retention_v2(
        storage_root, acknowledgement, published,
    )

    assert applied.deleted_generations == (_retention_ref(first),)
    assert applied.already_absent_generations == ()
    assert applied.retained_generations == (
        _retention_ref(second), _retention_ref(third),
    )
    assert not first.root.exists()
    assert second.root.is_dir()
    assert third.root.is_dir()
    current = json.loads(
        (third.root.parent.parent / "CURRENT").read_text()
    )
    assert current["generation"] == 3

    replayed = apply_checkpoint_generation_retention_v2(
        storage_root, acknowledgement, published,
    )
    assert replayed.deleted_generations == ()
    assert replayed.already_absent_generations == (_retention_ref(first),)
    assert second.root.is_dir()
    assert third.root.is_dir()


def test_terminal_release_removes_exact_current_pointer(
    tmp_path: Path,
) -> None:
    storage_root, (published,) = _authoritative_generations(tmp_path, 1)
    checkpoint_root = published.root.parent.parent
    assert (checkpoint_root / "CURRENT").is_file()

    applied = apply_checkpoint_generation_retention_v2(
        storage_root,
        _retention_ack(delete=(published,), operation_id="retention-current-0001"),
        (published,),
    )

    assert applied.deleted_generations == (_retention_ref(published),)
    assert not published.root.exists()
    assert not (checkpoint_root / "CURRENT").exists()


def test_retention_preserves_tampered_symlinked_generation_in_quarantine(
    tmp_path: Path,
) -> None:
    storage_root, (published,) = _authoritative_generations(tmp_path, 1)
    outside = tmp_path / "outside-receipt.json"
    outside.write_text("{}", encoding="utf-8")
    receipt = published.root / "publication.json"
    receipt.unlink()
    receipt.symlink_to(outside)

    with pytest.raises(
        CheckpointDataPlaneError, match="generation_revalidation_failed",
    ):
        apply_checkpoint_generation_retention_v2(
            storage_root,
            _retention_ack(delete=(published,), operation_id="retention-tamper-0001"),
            (published,),
        )

    assert outside.read_text(encoding="utf-8") == "{}"
    checkpoint_root = published.root.parent.parent
    quarantines = list(published.root.parent.glob(".retention-*"))
    assert len(quarantines) == 1
    assert quarantines[0].is_dir()
    assert not list(checkpoint_root.glob(".retention-release-*.json"))


def test_retention_quarantine_rename_failure_keeps_original_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root, (published,) = _authoritative_generations(tmp_path, 1)
    real_replace = runtime.os.replace

    def fail_target_rename(source, destination):
        if Path(source) == published.root:
            raise OSError("injected rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_target_rename)
    with pytest.raises(
        CheckpointDataPlaneError, match="generation_quarantine_failed",
    ):
        apply_checkpoint_generation_retention_v2(
            storage_root,
            _retention_ack(delete=(published,), operation_id="retention-rename-0001"),
            (published,),
        )
    assert published.root.is_dir()
    assert not list(published.root.parent.glob(".retention-*"))


def test_retention_cleanup_resumes_from_durable_release_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root, (published,) = _authoritative_generations(tmp_path, 1)
    acknowledgement = _retention_ack(
        delete=(published,), operation_id="retention-cleanup-0001",
    )
    real_rmtree = runtime.shutil.rmtree

    def fail_quarantine_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".retention-"):
            raise OSError("injected cleanup interruption")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(runtime.shutil, "rmtree", fail_quarantine_cleanup)
    with pytest.raises(
        CheckpointDataPlaneError, match="generation_cleanup_failed",
    ):
        apply_checkpoint_generation_retention_v2(
            storage_root, acknowledgement, (published,),
        )
    checkpoint_root = published.root.parent.parent
    assert len(list(published.root.parent.glob(".retention-*"))) == 1
    assert len(list(checkpoint_root.glob(".retention-release-*.json"))) == 1
    assert not (checkpoint_root / "CURRENT").exists()

    monkeypatch.setattr(runtime.shutil, "rmtree", real_rmtree)
    resumed = apply_checkpoint_generation_retention_v2(
        storage_root, acknowledgement, (published,),
    )
    assert resumed.deleted_generations == (_retention_ref(published),)
    assert not list(published.root.parent.glob(".retention-*"))
    assert not list(checkpoint_root.glob(".retention-release-*.json"))


def test_retention_rejects_duplicate_ack_without_deleting(
    tmp_path: Path,
) -> None:
    storage_root, (published,) = _authoritative_generations(tmp_path, 1)
    reference = _retention_ref(published)
    acknowledgement = replace(
        _retention_ack(delete=(published,), operation_id="retention-duplicate-0001"),
        delete_generations=(reference, reference),
    )
    with pytest.raises(
        CheckpointDataPlaneError, match="invalid_acknowledgement_inventory",
    ):
        apply_checkpoint_generation_retention_v2(
            storage_root, acknowledgement, (published,),
        )
    assert published.root.is_dir()


def test_container_export_is_deterministic_for_same_capture(tmp_path: Path) -> None:
    request = _request()
    source = _source(tmp_path)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir(mode=0o700)
    second_root.mkdir(mode=0o700)
    first = first_root / "snapshot.tar.gz"
    second = second_root / "snapshot.tar.gz"
    first_export = seal_checkpoint_export_v2(
        source, first, request, container_export_root=first_root,
    )
    second_export = seal_checkpoint_export_v2(
        source, second, request, container_export_root=second_root,
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_export.archive_sha256 == second_export.archive_sha256
    assert first_export.manifest_sha256 == second_export.manifest_sha256


def test_container_seal_recovers_exact_export_after_atomic_rename_crash(
    tmp_path: Path,
) -> None:
    request = _request()
    source = _source(tmp_path)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    archive = export_root / "snapshot.tar.gz"
    first = seal_checkpoint_export_v2(
        source, archive, request, container_export_root=export_root,
    )
    archive_bytes = archive.read_bytes()

    # Simulate a hard crash after the final archive rename but before the
    # process could remove its private copy stage or return the seal receipt.
    abandoned_stage = export_root / f".sealed-{request.capture_id}"
    abandoned_stage.mkdir(mode=0o700)
    (abandoned_stage / "partial").write_bytes(b"incomplete predecessor")

    recovered = seal_checkpoint_export_v2(
        source, archive, request, container_export_root=export_root,
    )

    assert recovered == first
    assert archive.read_bytes() == archive_bytes
    assert abandoned_stage.is_dir()
    assert not list(export_root.glob(f".recover-{request.capture_id}-*.part"))


def test_container_seal_abandoned_stage_does_not_block_restart(
    tmp_path: Path,
) -> None:
    request = _request()
    source = _source(tmp_path)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    abandoned_stage = export_root / (
        f".sealed-{request.capture_id}-{'0' * 32}.part"
    )
    abandoned_stage.mkdir(mode=0o700)
    (abandoned_stage / "partial").write_bytes(b"incomplete predecessor")
    archive = export_root / "snapshot.tar.gz"

    exported = seal_checkpoint_export_v2(
        source, archive, request, container_export_root=export_root,
    )

    assert archive.is_file()
    assert exported.archive_sha256 == hashlib.sha256(
        archive.read_bytes(),
    ).hexdigest()
    assert abandoned_stage.is_dir()
    assert not [
        path for path in export_root.glob(
            f".sealed-{request.capture_id}-*.part",
        )
        if path != abandoned_stage
    ]


def test_container_seal_serializes_duplicate_live_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    source = _source(tmp_path)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    archive = export_root / "snapshot.tar.gz"
    real_copy = runtime._copy_source_tree
    guard = threading.Lock()
    active = 0
    maximum_active = 0

    def observed_copy(*args, **kwargs):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.03)
            return real_copy(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(runtime, "_copy_source_tree", observed_copy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                seal_checkpoint_export_v2,
                source,
                archive,
                request,
                container_export_root=export_root,
            )
            for _index in range(2)
        ]
        exports = [future.result(timeout=5) for future in futures]

    assert exports[0] == exports[1]
    assert maximum_active == 1
    assert not list(export_root.glob(f".sealed-{request.capture_id}-*.part"))


def test_container_seal_never_overwrites_invalid_existing_export(
    tmp_path: Path,
) -> None:
    request = _request()
    source = _source(tmp_path)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    archive = export_root / "snapshot.tar.gz"
    invalid = b"preserve-corrupt-evidence"
    archive.write_bytes(invalid)
    archive.chmod(0o600)

    with pytest.raises(
        CheckpointDataPlaneError, match="existing_export_invalid",
    ):
        seal_checkpoint_export_v2(
            source, archive, request, container_export_root=export_root,
        )

    assert archive.read_bytes() == invalid
    assert not list(export_root.glob(f".recover-{request.capture_id}-*.part"))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_container_seal_rejects_non_regular_state(
    tmp_path: Path, kind: str,
) -> None:
    source = _source(tmp_path)
    unsafe = source / "native" / "unsafe"
    if kind == "symlink":
        unsafe.symlink_to(source / "workspace/model.patch")
    elif kind == "hardlink":
        os.link(source / "workspace/model.patch", unsafe)
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO is unavailable")
        os.mkfifo(unsafe)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    archive = export_root / "snapshot.tar.gz"
    with pytest.raises(CheckpointDataPlaneError, match="unsafe_file_type"):
        seal_checkpoint_export_v2(
            source, archive, _request(), container_export_root=export_root,
        )
    assert not archive.exists()


def test_container_seal_rejects_exact_and_generic_credentials(tmp_path: Path) -> None:
    source = _source(tmp_path)
    exact = "opaque-subscription-value-123456789"
    (source / "native/credential").write_text(exact, encoding="utf-8")
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    archive = export_root / "snapshot.tar.gz"
    with pytest.raises(CheckpointDataPlaneError, match="secret_detected"):
        seal_checkpoint_export_v2(
            source, archive, _request(), sensitive_values=(exact,),
            container_export_root=export_root,
        )
    assert not archive.exists()

    (source / "native/credential").write_text(
        "sk-proj-abcdefghijklmnopqrstuv", encoding="utf-8",
    )
    with pytest.raises(CheckpointDataPlaneError, match="secret_detected"):
        seal_checkpoint_export_v2(
            source, archive, _request(), container_export_root=export_root,
        )
    assert not archive.exists()


def test_container_seal_inspects_secrets_and_links_inside_untracked_archive(
    tmp_path: Path,
) -> None:
    exact = "opaque-subscription-value-123456789"
    source = _source(tmp_path)
    (source / "untracked.tar.gz").write_bytes(_nested_untracked_archive([
        ("generated.txt", exact.encode(), "file"),
    ]))
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    with pytest.raises(CheckpointDataPlaneError, match="secret_detected"):
        seal_checkpoint_export_v2(
            source,
            export_root / "secret.tar.gz",
            _request(),
            sensitive_values=(exact,),
            container_export_root=export_root,
        )

    (source / "untracked.tar.gz").write_bytes(_nested_untracked_archive([
        ("linked", None, "symlink"),
    ]))
    with pytest.raises(
        CheckpointDataPlaneError, match="nested_archive_member_invalid",
    ):
        seal_checkpoint_export_v2(
            source,
            export_root / "link.tar.gz",
            _request(),
            container_export_root=export_root,
        )


def test_host_verifier_reinspects_untracked_archive_after_outer_manifest_rewrite(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    source = _source(fixture)
    source.joinpath("untracked.tar.gz").write_bytes(_nested_untracked_archive([
        ("safe.txt", b"ordinary worktree data", "file"),
    ]))
    request = _request()
    container_root = fixture / "container-native"
    container_root.mkdir(mode=0o700)
    original = container_root / "original.tar.gz"
    exported = seal_checkpoint_export_v2(
        source, original, request, container_export_root=container_root,
    )
    hostile_nested = _nested_untracked_archive([
        ("credential.txt", b"sk-proj-abcdefghijklmnopqrstuv", "file"),
    ])
    hostile = tmp_path / "hostile.tar.gz"

    def rewrite_nested_and_manifest(members):
        rewritten = []
        manifest = None
        for member, content in members:
            if member.name == "payload/untracked.tar.gz":
                member.size = len(hostile_nested)
                content = hostile_nested
            if member.name == "manifest.json":
                manifest = json.loads(content)
                for entry in manifest["files"]:
                    if entry["path"] == "untracked.tar.gz":
                        manifest["total_bytes"] += len(hostile_nested) - entry["size"]
                        entry["size"] = len(hostile_nested)
                        entry["sha256"] = hashlib.sha256(hostile_nested).hexdigest()
                content = runtime._canonical_json(manifest)
                member.size = len(content)
            rewritten.append((member, content))
        assert manifest is not None
        return rewritten

    _rewrite_archive(original, hostile, rewrite_nested_and_manifest)
    with tarfile.open(hostile, "r:gz") as archive:
        manifest_bytes = archive.extractfile("manifest.json").read()
    digest, size = _archive_sha(hostile)
    forged = replace(
        exported,
        archive_sha256=digest,
        archive_size=size,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        remote_path="/run/dradar-checkpoint-v2/x/sealed/hostile.tar.gz",
    )
    with pytest.raises(CheckpointDataPlaneError, match="secret_detected"):
        publish_checkpoint_export_v2(
            hostile,
            tmp_path / "host",
            request,
            forged,
            authoritative=False,
        )


def test_download_failure_is_observed_and_never_published(tmp_path: Path) -> None:
    exporter = FakeExporter(tmp_path / "container")
    exporter.download_error = OSError("simulated truncated copy")
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "failed"
    assert observed.stage == "download"
    assert observed.code == "download_failed"
    assert observed.mainline_may_continue is True
    assert observed.remote_cleanup == "discarded"
    assert not (tmp_path / "host/checkpoints/checkpoint-0001").exists()


def test_abandoned_download_and_publication_stages_do_not_block_exact_replay(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host"
    downloads = host / ".downloads"
    checkpoint_root = host / "checkpoints" / "checkpoint-0001"
    downloads.mkdir(parents=True, mode=0o700)
    checkpoint_root.mkdir(parents=True, mode=0o700)
    for directory in (host, downloads, host / "checkpoints", checkpoint_root):
        directory.chmod(0o700)
    abandoned_download = (
        downloads / ("capture-0001-" + "0" * 32 + ".tar.gz.part")
    )
    abandoned_download.write_bytes(b"truncated")
    abandoned_download.chmod(0o600)
    abandoned_publish = checkpoint_root / (
        ".incoming-capture-0001-" + "0" * 32 + ".part"
    )
    abandoned_publish.mkdir(mode=0o700)

    plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=host,
    )
    observed = asyncio.run(plane.observe_capture(
        _request(), FakeExporter(tmp_path / "container"),
    ))

    assert observed.status == "sealed"
    assert observed.published is not None
    assert observed.published.root.is_dir()
    assert abandoned_download.is_file()
    assert abandoned_publish.is_dir()
    active_parts = [
        path for path in downloads.iterdir()
        if path != abandoned_download
    ]
    assert active_parts == []


def test_typed_failure_writes_only_bounded_local_diagnostics(
    tmp_path: Path,
) -> None:
    exporter = FakeExporter(tmp_path / "container")
    exporter.capture_error = CheckpointDataPlaneError(
        "capture",
        "transport_helper_install_failed",
        diagnostic={
            "operation": "helper_install",
            "exit_code": 1,
            "stderr_bytes": 27,
            "stderr_sha256": "a" * 64,
        },
    )
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))

    assert observed.status == "failed"
    assert observed.code == "transport_helper_install_failed"
    diagnostic_path = tmp_path / "host" / "diagnostics.jsonl"
    assert stat.S_IMODE(diagnostic_path.stat().st_mode) == 0o600
    record = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert record == {
        "at": record["at"],
        "code": "transport_helper_install_failed",
        "diagnostic": {
            "exit_code": 1,
            "operation": "helper_install",
            "stderr_bytes": 27,
            "stderr_sha256": "a" * 64,
        },
        "failure_type": "CheckpointDataPlaneError",
        "schema": "dradar-checkpoint-local-diagnostic-v2",
        "stage": "capture",
    }


def test_owner_reconcile_diagnostic_omits_exception_text(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authoritative" / "assignment-0001"
    record_local_checkpoint_failure_v2(
        root,
        exc=RuntimeError("token=must-never-be-persisted /private/path"),
        stage="reconcile",
        code="orphan_reconcile_blocked",
    )
    path = root / "diagnostics.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = path.read_text(encoding="utf-8")
    assert "must-never" not in raw
    assert "/private/path" not in raw
    assert json.loads(raw) == {
        "at": json.loads(raw)["at"],
        "code": "orphan_reconcile_blocked",
        "diagnostic": {},
        "failure_type": "RuntimeError",
        "schema": "dradar-checkpoint-local-diagnostic-v2",
        "stage": "reconcile",
    }


def test_local_diagnostic_contract_rejects_sensitive_or_unbounded_fields() -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        CheckpointDataPlaneError(
            "capture", "transport_failed", diagnostic={"api_key": "secret"},
        )
    with pytest.raises(ValueError, match="diagnostic"):
        CheckpointDataPlaneError(
            "capture", "transport_failed", diagnostic={"detail": "x" * 161},
        )


def test_corrupt_archive_is_rejected_without_selecting_a_generation(
    tmp_path: Path,
) -> None:
    exporter = FakeExporter(tmp_path / "container")
    original_download = exporter.download_export

    async def corrupt_download(export, destination, *, max_bytes):
        await original_download(export, destination, max_bytes=max_bytes)
        with destination.open("ab") as handle:
            handle.write(b"corruption")

    exporter.download_export = corrupt_download  # type: ignore[method-assign]
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "failed"
    assert observed.stage == "verify"
    assert observed.code == "archive_digest_mismatch"
    assert not list((tmp_path / "host" / ".downloads").iterdir())
    assert not (tmp_path / "host/checkpoints/checkpoint-0001/CURRENT").exists()
    payload = checkpoint_observation_payload_v2(
        _request(),
        observed,
        plane.activation,
        CheckpointObservationRuntimeV2(
            assignment_id="assignment-0001",
            operation_id="operation-observe-0001",
            elapsed_ms=12,
            platform="macos",
            container_backend="orbstack",
            client_version="0.5.98",
            adapter_version=exporter.adapter_version,
        ),
    )
    assert payload["failure_code"] == "archive_invalid"
    assert payload["manifest_sha256"] is None
    assert payload["authoritative"] is False


def test_archive_path_traversal_is_rejected_and_cannot_escape(tmp_path: Path) -> None:
    request, original, exported = _seal(tmp_path / "fixture")
    malicious = tmp_path / "malicious.tar.gz"

    def add_traversal(members):
        info = tarfile.TarInfo("payload/../escape")
        info.size = 4
        info.mode = 0o600
        members.append((info, b"evil"))
        return members

    _rewrite_archive(original, malicious, add_traversal)
    digest, size = _archive_sha(malicious)
    malicious_export = replace(
        exported, archive_sha256=digest, archive_size=size,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/malicious.tar.gz",
    )
    with pytest.raises(CheckpointDataPlaneError):
        publish_checkpoint_export_v2(
            malicious, tmp_path / "host", request, malicious_export,
            authoritative=False,
        )
    assert not (tmp_path / "host/escape").exists()
    assert not (tmp_path / "escape").exists()


def test_payload_corruption_is_caught_by_manifest_digest(tmp_path: Path) -> None:
    request, original, exported = _seal(tmp_path / "fixture")
    corrupt = tmp_path / "corrupt.tar.gz"

    def change_payload(members):
        result = []
        for member, content in members:
            if member.name == "payload/workspace/model.patch":
                content = b"x" * len(content or b"")
            result.append((member, content))
        return result

    _rewrite_archive(original, corrupt, change_payload)
    digest, size = _archive_sha(corrupt)
    corrupt_export = replace(
        exported, archive_sha256=digest, archive_size=size,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/corrupt.tar.gz",
    )
    with pytest.raises(CheckpointDataPlaneError, match="file_digest_mismatch"):
        publish_checkpoint_export_v2(
            corrupt, tmp_path / "host", request, corrupt_export,
            authoritative=False,
        )


def test_same_generation_is_idempotent_but_conflicting_capture_is_rejected(
    tmp_path: Path,
) -> None:
    request, archive, exported = _seal(tmp_path / "first")
    store = tmp_path / "host"
    first = publish_checkpoint_export_v2(
        archive, store, request, replace(
            exported,
            remote_path="/run/dradar-checkpoint-v2/x/sealed/first.tar.gz",
        ), authoritative=False,
    )
    repeated = publish_checkpoint_export_v2(
        archive, store, request, replace(
            exported,
            remote_path="/run/dradar-checkpoint-v2/x/sealed/first.tar.gz",
        ), authoritative=False,
    )
    assert repeated.root == first.root

    second_request = replace(request, capture_id="capture-0002")
    _, second_archive, second_export = _seal(tmp_path / "second", second_request)
    with pytest.raises(CheckpointDataPlaneError, match="generation_conflict"):
        publish_checkpoint_export_v2(
            second_archive,
            store,
            second_request,
            replace(
                second_export,
                remote_path="/run/dradar-checkpoint-v2/x/sealed/second.tar.gz",
            ),
            authoritative=False,
        )
    current = json.loads((store / request.checkpoint_id / "CURRENT").read_text())
    assert current["manifest_sha256"] == first.manifest_sha256


@pytest.mark.parametrize("_attempt", range(8))
def test_spawned_publishers_commit_one_generation_without_corruption(
    tmp_path: Path,
    _attempt: int,
) -> None:
    request, archive, exported = _seal(tmp_path / "first")
    exported = replace(
        exported,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/first.tar.gz",
    )
    conflicting_request = replace(request, capture_id="capture-0002")
    _, conflicting_archive, conflicting_export = _seal(
        tmp_path / "second", conflicting_request,
    )
    conflicting_export = replace(
        conflicting_export,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/second.tar.gz",
    )
    store = tmp_path / "host"
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    inputs = (
        (archive, request, exported),
        (archive, request, exported),
        (conflicting_archive, conflicting_request, conflicting_export),
    )
    processes = [
        context.Process(
            target=_publish_checkpoint_process,
            args=(
                str(item[0]), str(store), item[1], item[2],
                start_event, result_queue, index,
            ),
        )
        for index, item in enumerate(inputs)
    ]
    try:
        for process in processes:
            process.start()
        start_event.set()
        messages = []
        for _ in processes:
            try:
                messages.append(result_queue.get(timeout=30))
            except queue.Empty as exc:
                raise AssertionError("publisher process did not report") from exc
        for process in processes:
            process.join(timeout=30)
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()

    assert not [message for message in messages if message[0] == "exception"]
    successes = [message for message in messages if message[0] == "ok"]
    conflicts = [message for message in messages if message[0] == "error"]
    assert successes
    winner_capture_id = successes[0][2]
    assert {message[2] for message in successes} == {winner_capture_id}
    assert all(
        message[2] == "generation_conflict" for message in conflicts
    ), messages
    if winner_capture_id == request.capture_id:
        assert len(successes) == 2
        assert len(conflicts) == 1
    else:
        assert winner_capture_id == conflicting_request.capture_id
        assert len(successes) == 1
        assert len(conflicts) == 2

    checkpoint_root = store / request.checkpoint_id
    generations = list((checkpoint_root / "generations").iterdir())
    assert [path.name for path in generations] == [
        "generation-00000000000000000001",
    ]
    receipt = json.loads(
        (generations[0] / "publication.json").read_text(encoding="utf-8")
    )
    assert receipt["capture_id"] == winner_capture_id
    current = json.loads(
        (checkpoint_root / "CURRENT").read_text(encoding="utf-8")
    )
    assert current["generation"] == 1
    assert current["manifest_sha256"] == receipt["manifest_sha256"]
    assert not list(checkpoint_root.glob(".incoming-*.part"))


def test_hard_crash_while_publishing_recovers_exact_generation_on_restart(
    tmp_path: Path,
) -> None:
    request, archive, exported = _seal(tmp_path / "fixture")
    exported = replace(
        exported,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/export.tar.gz",
    )
    storage_home = tmp_path / "host"
    storage_home.mkdir(mode=0o700)
    storage_home.chmod(0o700)
    store = storage_home / "checkpoints"
    context = multiprocessing.get_context("spawn")
    crash_reached = context.Event()
    process = context.Process(
        target=_crash_publication_before_current_process,
        args=(str(archive), str(store), request, exported, crash_reached),
    )
    try:
        process.start()
        assert crash_reached.wait(timeout=30)
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 93
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    checkpoint_root = store / request.checkpoint_id
    generation = (
        checkpoint_root / "generations" /
        "generation-00000000000000000001"
    )
    assert generation.is_dir()
    assert not (checkpoint_root / "CURRENT").exists()
    crash_evidence = list(checkpoint_root.glob(".CURRENT.*.part"))
    assert len(crash_evidence) == 1

    recovered = publish_checkpoint_export_v2(
        archive, store, request, exported, authoritative=True,
    )
    assert recovered.root == generation
    assert recovered.capture_id == request.capture_id
    assert recovered.selected is True
    assert load_exact_published_checkpoint_v2(
        storage_home,
        checkpoint_id=request.checkpoint_id,
        checkpoint_lineage_id=request.checkpoint_lineage_id,
        snapshot_generation=request.snapshot_generation,
        capture_id=request.capture_id,
        manifest_sha256=exported.manifest_sha256,
        expected_identity_fingerprint=request.identity_fingerprint,
        expected_checkpoint_core_abi=runtime.CHECKPOINT_CORE_ABI_V2,
        expected_checkpoint_abi=request.checkpoint_abi,
        expected_recovery_capability=request.recovery_capability,
        expected_native_state_schema=request.native_state_schema,
    ).root == generation
    assert json.loads(
        (checkpoint_root / "CURRENT").read_text(encoding="utf-8")
    )["generation"] == 1
    assert crash_evidence[0].is_file()
    assert not list(checkpoint_root.glob(".incoming-*.part"))


def test_publication_retries_after_crash_between_generation_and_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, archive, exported = _seal(tmp_path / "fixture")
    exported = replace(
        exported,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/export.tar.gz",
    )
    store = tmp_path / "host"
    real_replace = runtime.os.replace
    failed_once = False

    def fail_first_current_replace(source, destination):
        nonlocal failed_once
        if Path(destination).name == "CURRENT" and not failed_once:
            failed_once = True
            raise OSError("injected crash before CURRENT replace")
        return real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_first_current_replace)
    with pytest.raises(OSError, match="before CURRENT"):
        publish_checkpoint_export_v2(
            archive, store, request, exported, authoritative=False,
        )

    checkpoint_root = store / request.checkpoint_id
    generation = (
        checkpoint_root / "generations" /
        "generation-00000000000000000001"
    )
    assert generation.is_dir()
    assert not (checkpoint_root / "CURRENT").exists()
    assert not list(checkpoint_root.glob(".CURRENT.*.part"))

    retried = publish_checkpoint_export_v2(
        archive, store, request, exported, authoritative=False,
    )
    assert retried.root == generation
    assert retried.selected is True
    current = json.loads((checkpoint_root / "CURRENT").read_text())
    assert current["generation"] == 1
    assert current["manifest_sha256"] == retried.manifest_sha256


def test_publication_replay_rehashes_existing_generation(
    tmp_path: Path,
) -> None:
    request, archive, exported = _seal(tmp_path / "fixture")
    exported = replace(
        exported,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/export.tar.gz",
    )
    store = tmp_path / "host"
    published = publish_checkpoint_export_v2(
        archive, store, request, exported, authoritative=False,
    )
    tampered = published.payload_root / "workspace/model.patch"
    original = tampered.read_bytes()
    tampered.write_bytes(b"X" + original[1:])
    tampered.chmod(0o600)

    with pytest.raises(
        CheckpointDataPlaneError,
        match="published_payload_digest_mismatch",
    ):
        publish_checkpoint_export_v2(
            archive, store, request, exported, authoritative=False,
        )


def test_publication_retries_after_current_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, archive, exported = _seal(tmp_path / "fixture")
    exported = replace(
        exported,
        remote_path="/run/dradar-checkpoint-v2/x/sealed/export.tar.gz",
    )
    store = tmp_path / "host"
    checkpoint_root = store / request.checkpoint_id
    real_fsync_directory = runtime._fsync_directory
    failed_once = False

    def fail_first_current_directory_fsync(path: Path) -> None:
        nonlocal failed_once
        if (
            Path(path) == checkpoint_root
            and (checkpoint_root / "CURRENT").is_file()
            and not failed_once
        ):
            failed_once = True
            raise OSError("injected CURRENT directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        runtime, "_fsync_directory", fail_first_current_directory_fsync,
    )
    with pytest.raises(OSError, match="directory fsync"):
        publish_checkpoint_export_v2(
            archive, store, request, exported, authoritative=False,
        )

    assert (checkpoint_root / "CURRENT").is_file()
    retried = publish_checkpoint_export_v2(
        archive, store, request, exported, authoritative=False,
    )
    assert retried.selected is True
    assert not list(checkpoint_root.glob(".CURRENT.*.part"))


def test_late_older_generation_is_retained_but_never_rewinds_current(
    tmp_path: Path,
) -> None:
    store = tmp_path / "host"
    newer_request = _request(
        snapshot_generation=2, capture_id="capture-0002",
    )
    _, newer_archive, newer_export = _seal(tmp_path / "newer", newer_request)
    newer = publish_checkpoint_export_v2(
        newer_archive,
        store,
        newer_request,
        replace(
            newer_export,
            remote_path="/run/dradar-checkpoint-v2/x/sealed/newer.tar.gz",
        ),
        authoritative=False,
    )
    assert newer.selected is True

    older_request = _request(snapshot_generation=1, capture_id="capture-0001")
    _, older_archive, older_export = _seal(tmp_path / "older", older_request)
    older = publish_checkpoint_export_v2(
        older_archive,
        store,
        older_request,
        replace(
            older_export,
            remote_path="/run/dradar-checkpoint-v2/x/sealed/older.tar.gz",
        ),
        authoritative=False,
    )
    assert older.root.is_dir()
    assert older.selected is False
    current = json.loads(
        (store / newer_request.checkpoint_id / "CURRENT").read_text()
    )
    assert current["generation"] == 2
    assert current["manifest_sha256"] == newer.manifest_sha256


def test_capture_storage_attestation_rejects_bind_mount_exports(
    tmp_path: Path,
) -> None:
    request, archive, exported = _seal(tmp_path / "fixture")
    with pytest.raises(CheckpointDataPlaneError, match="unsafe_capture_storage"):
        publish_checkpoint_export_v2(
            archive,
            tmp_path / "host",
            request,
            replace(
                exported,
                remote_path="/run/dradar-checkpoint-v2/x/sealed/export.tar.gz",
                capture_storage="host_bind_mount",
            ),
            authoritative=False,
        )


def test_container_reference_writer_rejects_export_outside_native_root(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    native = tmp_path / "native"
    native.mkdir(mode=0o700)
    native.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    outside.chmod(0o700)
    with pytest.raises(
        CheckpointDataPlaneError, match="export_outside_container_storage",
    ):
        seal_checkpoint_export_v2(
            source,
            outside / "snapshot.tar.gz",
            _request(),
            container_export_root=native,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_host_storage_permission_drift_fails_open(tmp_path: Path) -> None:
    storage = tmp_path / "host"
    storage.mkdir(mode=0o700)
    storage.chmod(0o755)
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=storage,
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "failed"
    assert observed.code == "unsafe_storage_permissions"
    assert observed.mainline_may_continue is True


def test_publish_disk_failure_is_fail_open(monkeypatch, tmp_path: Path) -> None:
    exporter = FakeExporter(tmp_path / "container")

    def disk_full(*_args, **_kwargs):
        raise CheckpointDataPlaneError("publish", "disk_full")

    monkeypatch.setattr(runtime, "publish_checkpoint_export_v2", disk_full)
    plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=tmp_path / "host",
    )
    observed = asyncio.run(plane.observe_capture(_request(), exporter))
    assert observed.status == "failed"
    assert observed.stage == "publish"
    assert observed.code == "disk_full"
    assert observed.remote_cleanup == "discarded"


def test_capture_cancellation_reaps_remote_export(tmp_path: Path) -> None:
    async def scenario() -> None:
        exporter = FakeExporter(tmp_path / "container")
        exporter.download_started = asyncio.Event()
        exporter.download_continue = asyncio.Event()
        plane = CheckpointDataPlaneV2(
            activation=_activation("observe"), storage_root=tmp_path / "host",
        )
        running = asyncio.create_task(plane.observe_capture(_request(), exporter))
        await exporter.download_started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert exporter.discarded == 1
        assert not (tmp_path / "host/checkpoints/checkpoint-0001/CURRENT").exists()

    asyncio.run(scenario())


def test_offline_restore_requires_restore_test_mode_and_never_authorizes_paid_run(
    tmp_path: Path,
) -> None:
    exporter = FakeExporter(tmp_path / "container")
    observe_plane = CheckpointDataPlaneV2(
        activation=_activation("observe"), storage_root=tmp_path / "host",
    )
    captured = asyncio.run(observe_plane.observe_capture(_request(), exporter))
    assert captured.published is not None
    restore_request = CheckpointRestoreRequestV2(
        published=captured.published,
        expected_identity_fingerprint="a" * 64,
        restore_id="restore-0001",
    )
    restorer = FakeRestorer()
    skipped = asyncio.run(
        observe_plane.observe_offline_restore(restore_request, restorer)
    )
    assert skipped.status == "skipped"
    assert skipped.paid_execution_authorized is False

    restore_plane = CheckpointDataPlaneV2(
        activation=_activation("restore-test"), storage_root=tmp_path / "host",
    )
    verified = asyncio.run(
        restore_plane.observe_offline_restore(restore_request, restorer)
    )
    assert verified.status == "verified"
    assert verified.evidence is not None
    assert verified.paid_execution_authorized is False
    restore_payload = checkpoint_restore_observation_payload_v2(
        _request(),
        restore_request,
        verified,
        restore_plane.activation,
        CheckpointObservationRuntimeV2(
            assignment_id="assignment-0001",
            operation_id="operation-restore-0001",
            elapsed_ms=91,
            platform="macos",
            container_backend="orbstack",
            client_version="0.5.98",
            adapter_version=restorer.adapter_version,
        ),
    )
    assert restore_payload["observation_kind"] == "restore"
    assert restore_payload["status"] == "verified"
    assert restore_payload["source_capture_id"] == "capture-0001"
    assert restore_payload["manifest_sha256"] == (
        captured.published.manifest_sha256
    )
    assert restore_payload["paid_execution_started"] is False
    assert restore_payload["authoritative"] is False


def test_offline_restorer_cannot_smuggle_a_paid_model_start(tmp_path: Path) -> None:
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("restore-test"), storage_root=tmp_path / "host",
    )
    captured = asyncio.run(plane.observe_capture(_request(), exporter))
    assert captured.published is not None
    request = CheckpointRestoreRequestV2(
        published=captured.published,
        expected_identity_fingerprint="a" * 64,
        restore_id="restore-0001",
    )
    restorer = FakeRestorer()
    restorer.paid_execution_started = True
    observed = asyncio.run(plane.observe_offline_restore(request, restorer))
    assert observed.status == "failed"
    assert observed.code == "restore_evidence_invalid"
    assert observed.paid_execution_authorized is False


def test_offline_restore_rehashes_published_payload_after_storage_tamper(
    tmp_path: Path,
) -> None:
    exporter = FakeExporter(tmp_path / "container")
    plane = CheckpointDataPlaneV2(
        activation=_activation("restore-test"), storage_root=tmp_path / "host",
    )
    captured = asyncio.run(plane.observe_capture(_request(), exporter))
    assert captured.published is not None
    patch = captured.published.payload_root / "workspace/model.patch"
    patch.write_bytes(b"tampered after publication\n")
    patch.chmod(0o600)
    request = CheckpointRestoreRequestV2(
        published=captured.published,
        expected_identity_fingerprint="a" * 64,
        restore_id="restore-0001",
    )
    observed = asyncio.run(
        plane.observe_offline_restore(request, FakeRestorer())
    )
    assert observed.status == "failed"
    assert observed.code in {
        "published_payload_unsafe", "published_payload_digest_mismatch",
    }
    assert observed.paid_execution_authorized is False


def test_shadow_failure_cannot_replace_a_valid_mainline_result() -> None:
    async def mainline():
        await asyncio.sleep(0)
        return {"submission": "valid"}

    async def broken_observer():
        raise RuntimeError("unexpected writer defect")

    observed = []
    result = asyncio.run(
        run_mainline_with_shadow_checkpoint_v2(
            mainline(), broken_observer(), on_observation=observed.append,
        )
    )
    assert result == {"submission": "valid"}
    assert observed[0].status == "failed"
    assert observed[0].code == "observer_failed"


def test_mainline_exception_is_preserved_and_shadow_is_cancelled() -> None:
    shadow_cancelled = False

    async def mainline():
        await asyncio.sleep(0)
        raise ValueError("authoritative model failure")

    async def shadow():
        nonlocal shadow_cancelled
        try:
            await asyncio.Event().wait()
        finally:
            shadow_cancelled = True

    with pytest.raises(ValueError, match="authoritative model failure"):
        asyncio.run(run_mainline_with_shadow_checkpoint_v2(mainline(), shadow()))
    assert shadow_cancelled is True


def test_slow_shadow_cleanup_is_bounded_and_cannot_delay_mainline() -> None:
    async def scenario() -> None:
        async def mainline():
            await asyncio.sleep(0)
            return "submitted"

        async def slow_to_cancel():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)
                return runtime.CheckpointObservationV2(
                    status="aborted", capture_id=None,
                )

        observations = []
        started = asyncio.get_running_loop().time()
        result = await run_mainline_with_shadow_checkpoint_v2(
            mainline(),
            slow_to_cancel(),
            on_observation=observations.append,
            observation_join_timeout_sec=0.005,
        )
        elapsed = asyncio.get_running_loop().time() - started
        assert result == "submitted"
        assert elapsed < 0.04
        assert observations[0].code == "observer_cancel_timeout"
        await asyncio.sleep(0.06)

    asyncio.run(scenario())


def test_periodic_shadow_sampler_collects_multiple_generations_without_owning_result(
    ) -> None:
    async def scenario() -> None:
        generations = []
        observations = []
        fourth = asyncio.Event()

        async def mainline():
            await fourth.wait()
            await asyncio.sleep(0)
            return {"submission": "authoritative"}

        async def capture(generation: int):
            generations.append(generation)
            await asyncio.sleep(0)
            if generation == 4:
                fourth.set()
            return runtime.CheckpointObservationV2(
                status="aborted",
                capture_id=f"capture-{generation:04d}",
            )

        result = await run_mainline_with_periodic_shadow_captures_v2(
            mainline(),
            capture,
            on_observation=observations.append,
            initial_delay_sec=0,
            interval_sec=0.01,
            maximum_captures=10,
        )
        assert result == {"submission": "authoritative"}
        assert generations == [1, 2, 3, 4]
        assert [item.capture_id for item in observations[:-1]] == [
            "capture-0001", "capture-0002", "capture-0003", "capture-0004",
        ]
        assert observations[-1].capture_id is None
        assert observations[-1].code == "mainline_completed"

    asyncio.run(scenario())


def test_periodic_shadow_sampler_locally_stops_after_repeated_failures() -> None:
    async def scenario() -> None:
        attempts = []

        async def mainline():
            await asyncio.sleep(0.05)
            return "completed"

        async def capture(generation: int):
            attempts.append(generation)
            raise RuntimeError("unbounded local detail must not escape")

        observed = []
        result = await run_mainline_with_periodic_shadow_captures_v2(
            mainline(), capture,
            on_observation=lambda value: (
                observed.append(value),
                (_ for _ in ()).throw(RuntimeError("telemetry callback broke")),
            ),
            initial_delay_sec=0,
            interval_sec=0.01,
            maximum_captures=20,
            consecutive_failure_limit=3,
        )
        assert result == "completed"
        assert attempts == [1, 2, 3]
        assert len(observed) == 3
        assert all(item.code == "observer_failed" for item in observed)

    asyncio.run(scenario())


def test_periodic_shadow_sampler_stops_immediately_when_activation_is_disabled(
    ) -> None:
    async def scenario() -> None:
        attempts = []

        async def mainline():
            await asyncio.sleep(0.04)
            return "completed"

        async def capture(generation: int):
            attempts.append(generation)
            return runtime.CheckpointObservationV2(
                status="skipped", capture_id=None,
            )

        result = await run_mainline_with_periodic_shadow_captures_v2(
            mainline(), capture,
            initial_delay_sec=0,
            interval_sec=0.01,
            maximum_captures=20,
        )
        assert result == "completed"
        assert attempts == [1]

    asyncio.run(scenario())
