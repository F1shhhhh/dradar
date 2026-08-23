"""Credential-free runner for the fixed Checkpoint V2 crash-cut matrix.

This module does not accept operator-supplied verdicts or test selectors.  It
runs one reviewed, versioned probe plan from clean client/server revisions,
stores private diagnostic logs outside the review bundle, and writes the exact
mode-0600 result consumed by :mod:`checkpoint_lifecycle_evidence_v2`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .checkpoint_lifecycle_evidence_v2 import (
    LIFECYCLE_CASE_IDS_V2,
    LIFECYCLE_RESULTS_SCHEMA_V2,
    LIFECYCLE_ZERO_METRICS_V2,
    _cohort,
)


MAX_LIFECYCLE_LOG_BYTES_V2 = 1024 * 1024
LIFECYCLE_PROBE_PLAN_VERSION_V2 = "checkpoint-v2-crash-cut-plan-v1"
LIFECYCLE_PROBES_V2: dict[str, tuple[tuple[str, str], ...]] = {
    "owner_checkout_precommit": ((
        "server",
        "tests/test_checkpoint_v2_foundation.py::"
        "test_v2_process_crash_before_owner_checkout_commit_rolls_back_and_restarts",
    ),),
    "owner_start_precommit": ((
        "server",
        "tests/test_checkpoint_v2_foundation.py::"
        "test_v2_process_crash_before_paid_start_commit_rolls_back_and_restarts",
    ),),
    "owner_start_response_lost": (
        (
            "client",
            "tests/test_checkpoint_owner_runtime_v2.py::"
            "test_ambiguous_start_without_server_commit_reconciles_to_fresh_v1",
        ),
        (
            "server",
            "tests/test_checkpoint_v2_foundation.py::"
            "test_v2_cross_process_paid_gate_reconcile_faults_committed_start_without_replay",
        ),
    ),
    "capture_interrupted": ((
        "client",
        "tests/test_checkpoint_runtime_v2.py::"
        "test_capture_cancellation_reaps_remote_export",
    ),),
    "seal_interrupted": ((
        "client",
        "tests/test_checkpoint_runtime_v2.py::"
        "test_container_seal_recovers_exact_export_after_atomic_rename_crash",
    ),),
    "download_interrupted": ((
        "client",
        "tests/test_checkpoint_runtime_v2.py::"
        "test_abandoned_download_and_publication_stages_do_not_block_exact_replay",
    ),),
    "publication_interrupted": ((
        "client",
        "tests/test_checkpoint_runtime_v2.py::"
        "test_hard_crash_while_publishing_recovers_exact_generation_on_restart",
    ),),
    "restore_interrupted": (
        (
            "client",
            "tests/test_checkpoint_adapter_runtime_v2.py::"
            "test_partial_restore_rolls_worktree_back_before_fresh_retry",
        ),
        (
            "client",
            "tests/test_pier_checkpoint.py::"
            "test_checkpoint_v2_partial_restore_is_never_retried_in_place",
        ),
    ),
    "restore_before_paid_commit": (
        (
            "client",
            "tests/test_checkpoint_owner_runtime_v2.py::"
            "test_owner_restart_reserves_restores_then_commits_at_paid_gate",
        ),
        (
            "server",
            "tests/test_checkpoint_v2_foundation.py::"
            "test_v2_reserved_restore_can_fallback_before_paid_commit",
        ),
    ),
    "paid_resume_response_lost": (
        (
            "client",
            "tests/test_checkpoint_v2_journal.py::"
            "test_resume_commit_response_loss_replays_exact_paid_permit",
        ),
        (
            "server",
            "tests/test_checkpoint_v2_foundation.py::"
            "test_v2_resume_segments_are_deduplicated_into_submission_usage",
        ),
    ),
    "completed_result_finalize_failure": ((
        "server",
        "tests/test_checkpoint_v2_foundation.py::"
        "test_v2_completed_result_is_durable_before_finalize_failure",
    ),),
    "submission_response_lost": ((
        "client",
        "tests/test_checkpoint_owner_runtime_v2.py::"
        "test_v2_submit_response_loss_reuses_exact_intent_without_model_rerun",
    ),),
    "retention_response_lost": ((
        "client",
        "tests/test_checkpoint_owner_runtime_v2.py::"
        "test_completed_result_retention_recovers_after_live_owner_process_is_gone",
    ),),
    "process_orphan_reconciliation": (
        (
            "client",
            "tests/test_checkpoint_owner_runtime_v2.py::"
            "test_cross_process_orphan_reaps_exact_pier_before_fresh_fallback",
        ),
        (
            "client",
            "tests/test_checkpoint_owner_runtime_v2.py::"
            "test_cross_process_orphan_preserves_completed_result_and_never_restarts",
        ),
    ),
}

if set(LIFECYCLE_PROBES_V2) != LIFECYCLE_CASE_IDS_V2:  # pragma: no cover
    raise RuntimeError("checkpoint lifecycle probe plan is incomplete")

_PROVIDER_ENV_NAMES = frozenset({
    "ANTHROPIC_API_KEY",
    "CODEX_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "ZHIPUAI_API_KEY",
    "ZCODE_API_KEY",
})


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _private_directory(path: Path, *, create: bool) -> None:
    if create:
        try:
            path.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            pass
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ValueError("lifecycle runner directory is unsafe")
    if hasattr(os, "getuid") and (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("lifecycle runner directory is not private")


def _write_private_once(path: Path, encoded: bytes) -> None:
    _private_directory(path.parent, create=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short lifecycle runner write")
            view = view[written:]
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_private_json(path: Path) -> Mapping[str, Any]:
    before = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= 64 * 1024
    ):
        raise ValueError("lifecycle cohort input is not a private regular file")
    if hasattr(os, "getuid") and (
        before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
    ):
        raise ValueError("lifecycle cohort input is not private")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise ValueError("lifecycle cohort input changed")
        encoded = os.read(descriptor, 64 * 1024 + 1)
        after = os.fstat(descriptor)
        if len(encoded) != after.st_size or after.st_mtime_ns != opened.st_mtime_ns:
            raise ValueError("lifecycle cohort input changed")
    finally:
        os.close(descriptor)
    value = json.loads(encoded)
    if not isinstance(value, Mapping):
        raise ValueError("lifecycle cohort input is not an object")
    return value


def _network_isolated() -> bool:
    """Require a real network namespace containing loopback interfaces only."""

    try:
        names = [name for _index, name in socket.if_nameindex()]
    except OSError:
        return False
    loopback_names = {"lo", "lo0"}
    return bool(names) and all(name in loopback_names for name in names)


def _provider_credentials_present() -> bool:
    return any(os.environ.get(name) for name in _PROVIDER_ENV_NAMES)


def _git_revision(root: Path) -> str:
    root = root.resolve(strict=True)
    status = subprocess.run(
        ["git", "-C", os.fspath(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if status.stdout:
        raise ValueError("lifecycle source has local changes")
    revision = subprocess.run(
        ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    ).stdout.decode("ascii").strip()
    if len(revision) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in revision
    ):
        raise ValueError("lifecycle source revision is invalid")
    return revision


def _materialize_git_revision(root: Path, revision: str, target: Path) -> None:
    """Run probes from an exact archive, never from a mutable worktree."""

    _private_directory(target, create=True)
    archive = subprocess.run(
        ["git", "-C", os.fspath(root), "archive", "--format=tar", revision],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    ).stdout
    if not 0 < len(archive) <= 512 * 1024 * 1024:
        raise ValueError("lifecycle source archive size is invalid")
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            relative = Path(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
            ):
                raise ValueError("lifecycle source archive path is unsafe")
            destination = target.joinpath(*relative.parts)
            if member.isdir():
                _private_directory(destination, create=True)
                continue
            if not member.isfile() or member.size < 0 or member.size > 32 * 1024 * 1024:
                raise ValueError("lifecycle source archive entry is unsafe")
            total += member.size
            if total > 256 * 1024 * 1024:
                raise ValueError("lifecycle source archive is too large")
            _private_directory(destination.parent, create=True)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError("lifecycle source archive entry is unreadable")
            payload = extracted.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError("lifecycle source archive entry changed")
            _write_private_once(destination, payload)
            if member.mode & 0o111:
                os.chmod(destination, 0o700)


def _test_suite_digest(roots: Mapping[str, Path]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({
        "plan_version": LIFECYCLE_PROBE_PLAN_VERSION_V2,
        "probes": LIFECYCLE_PROBES_V2,
    }))
    files: set[tuple[str, str]] = set()
    for probes in LIFECYCLE_PROBES_V2.values():
        for component, node_id in probes:
            files.add((component, node_id.split("::", 1)[0]))
    files.add(("client", "src/dradar/checkpoint_lifecycle_runner_v2.py"))
    files.add(("client", "src/dradar/checkpoint_lifecycle_evidence_v2.py"))
    for component, relative in sorted(files):
        root = roots[component].resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise ValueError("lifecycle probe source is unsafe")
        payload = path.read_bytes()
        digest.update(_canonical_bytes({
            "component": component,
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }))
    return digest.hexdigest()


def _probe_environment(sandbox_home: Path, source_root: Path) -> dict[str, str]:
    allowed = {
        name: os.environ[name]
        for name in ("LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TMPDIR")
        if os.environ.get(name)
    }
    allowed.update({
        "HOME": os.fspath(sandbox_home),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.fspath(source_root / "src"),
    })
    return allowed


def _run_probe(
    *,
    component: str,
    node_id: str,
    source_root: Path,
    python: Path,
    sandbox_home: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            [os.fspath(python), "-m", "pytest", "-q", node_id],
            cwd=source_root,
            env=_probe_environment(sandbox_home, source_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    elapsed_ms = min(int((time.monotonic() - started) * 1000), 86_400_000)
    truncated = (
        len(stdout) > MAX_LIFECYCLE_LOG_BYTES_V2
        or len(stderr) > MAX_LIFECYCLE_LOG_BYTES_V2
    )
    return {
        "component": component,
        "node_id": node_id,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_ms": elapsed_ms,
        "stdout": stdout[-MAX_LIFECYCLE_LOG_BYTES_V2:],
        "stderr": stderr[-MAX_LIFECYCLE_LOG_BYTES_V2:],
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "output_truncated": truncated,
    }


def _case_evidence(case_id: str, probes: Sequence[Mapping[str, Any]]) -> bytes:
    return _canonical_bytes({
        "case_id": case_id,
        "plan_version": LIFECYCLE_PROBE_PLAN_VERSION_V2,
        "probes": [
            {
                "component": probe["component"],
                "node_id": probe["node_id"],
                "returncode": probe["returncode"],
                "timed_out": probe["timed_out"],
                "elapsed_ms": probe["elapsed_ms"],
                "stdout_sha256": probe["stdout_sha256"],
                "stderr_sha256": probe["stderr_sha256"],
                "output_truncated": probe["output_truncated"],
            }
            for probe in probes
        ],
    })


def run_lifecycle_matrix_v2(
    *,
    cohort_path: Path,
    output_path: Path,
    client_source: Path,
    server_source: Path,
    client_python: Path,
    server_python: Path,
    probe_timeout_sec: float = 600.0,
) -> dict[str, Any]:
    if not 1 <= probe_timeout_sec <= 3600:
        raise ValueError("lifecycle probe timeout is invalid")
    if not _network_isolated():
        raise ValueError("lifecycle runner is not network-isolated")
    if _provider_credentials_present():
        raise ValueError("lifecycle runner has Provider credentials")
    cohort = _cohort(_read_private_json(Path(cohort_path)))
    if cohort["client_version"] != __version__:
        raise ValueError("lifecycle cohort client version does not match runner")
    roots = {
        "client": Path(client_source).resolve(strict=True),
        "server": Path(server_source).resolve(strict=True),
    }
    pythons = {
        "client": Path(client_python).resolve(strict=True),
        "server": Path(server_python).resolve(strict=True),
    }
    if any(
        not path.is_file() or not os.access(path, os.X_OK)
        for path in pythons.values()
    ):
        raise ValueError("lifecycle probe Python is not executable")
    revisions = {
        component: _git_revision(root)
        for component, root in roots.items()
    }
    suite_digest = _test_suite_digest(roots)
    output = Path(output_path).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("lifecycle result output already exists")
    _private_directory(output.parent, create=True)
    output_parent = output.parent.resolve(strict=True)
    if any(
        output_parent == root or root in output_parent.parents
        for root in roots.values()
    ):
        raise ValueError("lifecycle result output must be outside source trees")
    logs = output.parent / f".{output.name}.logs-{uuid.uuid4().hex}"
    _private_directory(logs, create=True)
    sandbox_home = logs / "sandbox-home"
    _private_directory(sandbox_home, create=True)
    materialized_root = logs / "exact-sources"
    _private_directory(materialized_root, create=True)
    exact_roots = {
        component: materialized_root / component
        for component in roots
    }
    for component in roots:
        _materialize_git_revision(
            roots[component], revisions[component], exact_roots[component],
        )
    if _test_suite_digest(exact_roots) != suite_digest:
        raise ValueError("lifecycle source archive does not match reviewed tests")
    observed_from = datetime.now(timezone.utc).replace(microsecond=0)
    cases: list[dict[str, Any]] = []
    for case_id in sorted(LIFECYCLE_CASE_IDS_V2):
        probe_results = []
        for index, (component, node_id) in enumerate(
            LIFECYCLE_PROBES_V2[case_id], start=1,
        ):
            probe = _run_probe(
                component=component,
                node_id=node_id,
                source_root=exact_roots[component],
                python=pythons[component],
                sandbox_home=sandbox_home,
                timeout_sec=probe_timeout_sec,
            )
            probe_results.append(probe)
            log_value = {
                **probe,
                "stdout": probe["stdout"].decode("utf-8", errors="replace"),
                "stderr": probe["stderr"].decode("utf-8", errors="replace"),
            }
            _write_private_once(
                logs / f"{case_id}.{index}.json",
                json.dumps(
                    log_value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8"),
            )
        evidence = _case_evidence(case_id, probe_results)
        elapsed_ms = min(
            sum(int(probe["elapsed_ms"]) for probe in probe_results),
            86_400_000,
        )
        passed = all(
            probe["returncode"] == 0
            and not probe["timed_out"]
            and not probe["output_truncated"]
            for probe in probe_results
        )
        cases.append({
            "case_id": case_id,
            "status": "passed" if passed else "failed",
            "elapsed_ms": elapsed_ms,
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            **{field: 0 for field in LIFECYCLE_ZERO_METRICS_V2},
        })
    if {
        component: _git_revision(root)
        for component, root in roots.items()
    } != revisions:
        raise ValueError("lifecycle source changed while probes were running")
    if not _network_isolated() or _provider_credentials_present():
        raise ValueError("lifecycle runner isolation changed while probes were running")
    shutil.rmtree(materialized_root)
    if materialized_root.exists() or materialized_root.is_symlink():
        raise ValueError("lifecycle exact-source cleanup failed")
    observed_until = datetime.now(timezone.utc).replace(microsecond=0)
    if observed_until <= observed_from:
        observed_until = observed_from.replace(microsecond=0)
        observed_until = datetime.fromtimestamp(
            observed_until.timestamp() + 1, timezone.utc,
        )
    results = {
        "schema": LIFECYCLE_RESULTS_SCHEMA_V2,
        "cohort": cohort,
        "cases": cases,
        "observed_from": observed_from.isoformat(),
        "observed_until": observed_until.isoformat(),
        "source_revisions": revisions,
        "test_suite_sha256": suite_digest,
        "runner_instance_digest": hashlib.sha256(os.urandom(32)).hexdigest(),
        "network_isolated": True,
        "provider_credentials_present": False,
    }
    _write_private_once(
        output,
        json.dumps(
            results,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii"),
    )
    return results


def cmd_checkpoint_lifecycle_run(args) -> int:
    try:
        results = run_lifecycle_matrix_v2(
            cohort_path=Path(args.cohort),
            output_path=Path(args.output),
            client_source=Path(args.client_source),
            server_source=Path(args.server_source),
            client_python=Path(args.client_python),
            server_python=Path(args.server_python),
            probe_timeout_sec=float(args.probe_timeout_sec),
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"checkpoint lifecycle runner refused: {exc}", file=sys.stderr)
        return 2
    failed = sum(case["status"] == "failed" for case in results["cases"])
    print(
        f"checkpoint lifecycle matrix recorded: "
        f"{len(results['cases']) - failed} passed, {failed} failed",
    )
    return 1 if failed else 0


__all__ = [
    "LIFECYCLE_PROBE_PLAN_VERSION_V2",
    "LIFECYCLE_PROBES_V2",
    "cmd_checkpoint_lifecycle_run",
    "run_lifecycle_matrix_v2",
]
