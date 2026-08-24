"""Credential-free runner for the fixed Checkpoint V2 crash-cut matrix.

This module does not accept operator-supplied verdicts or test selectors.  It
runs one reviewed, versioned probe plan from clean client/server revisions,
stores private diagnostic logs outside the review bundle, and writes the exact
mode-0600 result consumed by :mod:`checkpoint_lifecycle_evidence_v2`.
"""

from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import uuid
import xml.etree.ElementTree as ET
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
from .checkpoint_docker_runtime_v2 import docker_container_backend_v2
from .telemetry import platform_family


MAX_LIFECYCLE_LOG_BYTES_V2 = 1024 * 1024
LIFECYCLE_PROBE_PLAN_VERSION_V2 = "checkpoint-v2-crash-cut-plan-v2"
_MACOS_SANDBOX_MARKER_ENV = (
    "DRADAR_CHECKPOINT_V2_MACOS_SANDBOX_SHA256"
)
_MACOS_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_MACOS_PS_EXEC = Path("/bin/ps")
_INET_DENIED_ERRNOS = frozenset({1, 13})
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
    "capture_interrupted": (
        (
            "client",
            "tests/test_checkpoint_runtime_v2.py::"
            "test_capture_cancellation_reaps_remote_export",
        ),
        (
            "client",
            "tests/test_checkpoint_shadow_v2.py::"
            "test_kill_switch_stops_future_samples_without_touching_mainline",
        ),
        (
            "server",
            "tests/test_checkpoint_v2_foundation.py::"
            "test_shadow_activation_rechecks_downgrade_and_immediate_kill_switch",
        ),
    ),
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
        (
            "client",
            "tests/test_checkpoint_shadow_v2.py::"
            "test_rollout_is_rechecked_after_capture_before_offline_restore",
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
    "completed_result_finalize_failure": (
        (
            "server",
            "tests/test_checkpoint_v2_foundation.py::"
            "test_v2_completed_result_is_durable_before_finalize_failure",
        ),
        (
            "client",
            "tests/test_checkpoint_observation_v2.py::"
            "test_mainline_impact_pair_is_private_idempotent_and_conflict_safe",
        ),
    ),
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
        (
            "client",
            "tests/test_checkpoint_observation_v2.py::"
            "test_incomplete_or_unregistered_mainline_sample_cannot_look_healthy",
        ),
    ),
}

if set(LIFECYCLE_PROBES_V2) != LIFECYCLE_CASE_IDS_V2:  # pragma: no cover
    raise RuntimeError("checkpoint lifecycle probe plan is incomplete")

_HARNESS_PROBE_IDS_V2 = {
    ("codex", "openai"): "codex-openai",
    ("codex", "deepseek"): "codex-deepseek",
    ("dsh", "deepseek"): "dsh-deepseek",
    ("kimi-code", "kimi-subscription"): "kimi-code-kimi-subscription",
    ("zcode", "bigmodel-coding-plan"): "zcode-bigmodel-coding-plan",
}


def lifecycle_probe_plan_v2(
    cohort: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Return the compiled common + exact-Harness plan for one cohort."""

    key = (cohort.get("harness"), cohort.get("provider"))
    harness_id = _HARNESS_PROBE_IDS_V2.get(key)
    if harness_id is None:
        raise ValueError("lifecycle cohort has no reviewed Harness probe plan")
    plan = dict(LIFECYCLE_PROBES_V2)
    plan["capture_interrupted"] = (
        *plan["capture_interrupted"],
        (
            "client",
            "tests/test_checkpoint_adapter_runtime_v2.py::"
            "test_every_reviewed_harness_contract_builds_material_native_state"
            f"[{harness_id}]",
        ),
    )
    plan["restore_interrupted"] = (
        *plan["restore_interrupted"],
        (
            "client",
            "tests/test_checkpoint_docker_lifecycle_v2.py::"
            "test_real_container_native_capture_seal_download_restore"
            f"[{harness_id}]",
        ),
    )
    return plan

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


def _inet_outbound_denied() -> bool:
    """Return true only when the kernel denies IPv4 and IPv6 outbound use."""

    targets = (
        (socket.AF_INET, ("192.0.2.1", 9)),
        (socket.AF_INET6, ("2001:db8::1", 9)),
    )
    for family, target in targets:
        candidate = None
        try:
            candidate = socket.socket(family, socket.SOCK_STREAM)
            candidate.settimeout(0.05)
            result = candidate.connect_ex(target)
        except OSError as exc:
            result = exc.errno
        finally:
            if candidate is not None:
                candidate.close()
        if result not in _INET_DENIED_ERRNOS:
            return False
    return True


def _subprocess_inet_outbound_denied() -> bool:
    """Prove that the network denial is inherited by runner descendants."""

    probe = (
        "import socket,sys\n"
        "ok=True\n"
        "for family,target in ((socket.AF_INET,('192.0.2.1',9)),"
        "(socket.AF_INET6,('2001:db8::1',9))):\n"
        " s=None\n"
        " try:\n"
        "  s=socket.socket(family,socket.SOCK_STREAM);s.settimeout(.05)\n"
        "  result=s.connect_ex(target)\n"
        " except OSError as exc:\n"
        "  result=exc.errno\n"
        " finally:\n"
        "  s.close() if s is not None else None\n"
        " ok=ok and result in (1,13)\n"
        "sys.exit(0 if ok else 1)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _docker_unix_socket() -> Path:
    try:
        completed = subprocess.run(
            [
                "docker", "context", "inspect", "--format",
                '{{(index .Endpoints "docker").Host}}',
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("macOS lifecycle Docker context is unavailable") from exc
    endpoint = completed.stdout.decode("utf-8", errors="strict").strip()
    if not endpoint.startswith("unix://"):
        raise ValueError("macOS lifecycle Docker endpoint is not a Unix socket")
    raw_path = endpoint.removeprefix("unix://")
    if (
        not raw_path.startswith("/")
        or len(raw_path) > 1024
        or any(character in raw_path for character in ('\x00', '\n', '\r'))
    ):
        raise ValueError("macOS lifecycle Docker socket path is invalid")
    path = Path(raw_path).resolve(strict=True)
    metadata = path.stat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError("macOS lifecycle Docker socket is unsafe")
    return path


def _trusted_root_executable(path: Path, *, label: str) -> Path:
    executable = path.resolve(strict=True)
    metadata = executable.stat()
    if (
        executable != path
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError(f"macOS lifecycle {label} executable is unsafe")
    return executable


def _macos_sandbox_profile() -> tuple[str, str]:
    _trusted_root_executable(_MACOS_SANDBOX_EXEC, label="sandbox")
    ps_executable = _trusted_root_executable(_MACOS_PS_EXEC, label="ps")
    docker_socket = os.fspath(_docker_unix_socket())
    quoted_socket = json.dumps(docker_socket, ensure_ascii=True)
    quoted_ps = json.dumps(os.fspath(ps_executable), ensure_ascii=True)
    profile = " ".join((
        "(version 1)",
        "(allow default)",
        "(allow process-exec* (with no-sandbox)",
        f"(literal {quoted_ps}))",
        "(deny network*)",
        "(allow network-outbound",
        f"(remote unix-socket (path {quoted_socket})))",
    ))
    return profile, hashlib.sha256(profile.encode("utf-8")).hexdigest()


def _network_isolated() -> bool:
    """Verify a loopback-only namespace or an inherited macOS deny sandbox."""

    try:
        names = [name for _index, name in socket.if_nameindex()]
    except OSError:
        return False
    loopback_names = {"lo", "lo0"}
    if bool(names) and all(name in loopback_names for name in names):
        return True
    if platform_family() != "macos":
        return False
    try:
        _profile, expected_digest = _macos_sandbox_profile()
    except (OSError, ValueError):
        return False
    return (
        os.environ.get(_MACOS_SANDBOX_MARKER_ENV) == expected_digest
        and _inet_outbound_denied()
        and _subprocess_inet_outbound_denied()
    )


def _macos_sandbox_environment(profile_digest: str) -> dict[str, str]:
    allowed = {
        name: os.environ[name]
        for name in (
            "DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST",
            "DRADAR_CHECKPOINT_V2_DOCKER_IMAGE", "HOME", "LANG", "LC_ALL",
            "PATH", "PYTHONPATH", "TMPDIR",
        )
        if os.environ.get(name)
    }
    allowed.update({
        _MACOS_SANDBOX_MARKER_ENV: profile_digest,
        "PYTHONNOUSERSITE": "1",
    })
    return allowed


def _run_in_macos_sandbox(args) -> int | None:
    """Re-exec the CLI under the system deny-network sandbox on macOS."""

    if (
        platform_family() != "macos"
        or os.environ.get(_MACOS_SANDBOX_MARKER_ENV)
    ):
        return None
    if _provider_credentials_present():
        raise ValueError("lifecycle runner has Provider credentials")
    profile, profile_digest = _macos_sandbox_profile()
    command = [
        os.fspath(_MACOS_SANDBOX_EXEC), "-p", profile,
        sys.executable, "-m", "dradar.cli", "checkpoint", "lifecycle-run",
        "--cohort", os.fspath(Path(args.cohort).absolute()),
        "--output", os.fspath(Path(args.output).absolute()),
        "--client-source", os.fspath(Path(args.client_source).absolute()),
        "--server-source", os.fspath(Path(args.server_source).absolute()),
        "--client-python", os.fspath(Path(args.client_python).absolute()),
        "--server-python", os.fspath(Path(args.server_python).absolute()),
        "--probe-timeout-sec", str(float(args.probe_timeout_sec)),
    ]
    completed = subprocess.run(
        command,
        check=False,
        env=_macos_sandbox_environment(profile_digest),
    )
    return int(completed.returncode)


def _provider_credentials_present() -> bool:
    return any(os.environ.get(name) for name in _PROVIDER_ENV_NAMES)


def _observed_platform_backend() -> tuple[str, str]:
    try:
        return platform_family(), docker_container_backend_v2()
    except Exception as exc:
        raise ValueError("lifecycle runner cannot verify its Docker backend") from exc


def _runner_environment() -> dict[str, Any]:
    packages = sorted({
        (
            str(distribution.metadata.get("Name") or "").lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    })
    if not packages or len(packages) > 10_000 or any(
        not name or len(name) > 200 or len(version) > 200
        for name, version in packages
    ):
        raise ValueError("lifecycle runner package inventory is invalid")
    try:
        raw = subprocess.run(
            ["docker", "version", "--format", "{{json .}}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout
        docker = json.loads(raw)
        if not isinstance(docker, Mapping):
            raise ValueError("lifecycle runner Docker response is invalid")
        selected_docker = {
            side.lower(): {
                field: value.get(field)
                for field in ("Version", "ApiVersion", "Os", "Arch")
            }
            for side in ("Client", "Server")
            if isinstance((value := docker.get(side)), Mapping)
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError("lifecycle runner Docker environment is unavailable") from exc
    if set(selected_docker) != {"client", "server"} or any(
        not all(isinstance(value, str) and value for value in facts.values())
        for facts in selected_docker.values()
    ):
        raise ValueError("lifecycle runner Docker environment is invalid")
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "abi": getattr(sys.implementation, "cache_tag", None),
            "byteorder": sys.byteorder,
        },
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
        },
        "docker": selected_docker,
        "packages": [
            {"name": name, "version": version}
            for name, version in packages
        ],
    }


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


def _test_suite_digest(
    roots: Mapping[str, Path],
    plan: Mapping[str, Sequence[tuple[str, str]]],
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({
        "plan_version": LIFECYCLE_PROBE_PLAN_VERSION_V2,
        "probes": plan,
    }))
    files: set[tuple[str, str]] = set()
    for probes in plan.values():
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
        for name in (
            "DOCKER_HOST", "DRADAR_CHECKPOINT_V2_DOCKER_IMAGE", "LANG",
            "LC_ALL", "PATH", "SYSTEMROOT", "TMPDIR",
        )
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
    junit = sandbox_home / f"junit-{uuid.uuid4().hex}.xml"
    try:
        completed = subprocess.run(
            [
                os.fspath(python), "-m", "pytest", "-q", node_id,
                f"--junitxml={junit}",
            ],
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
    junit_valid = False
    tests_collected = tests_skipped = tests_failed = tests_errored = -1
    try:
        metadata = junit.lstat()
        if (
            not junit.is_symlink()
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and 0 < metadata.st_size <= MAX_LIFECYCLE_LOG_BYTES_V2
        ):
            root = ET.fromstring(junit.read_bytes())
            suites = [root] if root.tag == "testsuite" else list(
                root.findall("testsuite")
            )
            if suites:
                tests_collected = sum(int(item.attrib.get("tests", "0")) for item in suites)
                tests_skipped = sum(int(item.attrib.get("skipped", "0")) for item in suites)
                tests_failed = sum(int(item.attrib.get("failures", "0")) for item in suites)
                tests_errored = sum(int(item.attrib.get("errors", "0")) for item in suites)
                junit_valid = all(
                    value >= 0 for value in (
                        tests_collected, tests_skipped, tests_failed, tests_errored,
                    )
                )
    except (OSError, ET.ParseError, TypeError, ValueError):
        pass
    finally:
        try:
            junit.unlink()
        except FileNotFoundError:
            pass
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
        "junit_valid": junit_valid,
        "tests_collected": tests_collected,
        "tests_skipped": tests_skipped,
        "tests_failed": tests_failed,
        "tests_errored": tests_errored,
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
                "junit_valid": probe["junit_valid"],
                "tests_collected": probe["tests_collected"],
                "tests_skipped": probe["tests_skipped"],
                "tests_failed": probe["tests_failed"],
                "tests_errored": probe["tests_errored"],
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
    if _observed_platform_backend() != (
        cohort["platform"], cohort["container_backend"],
    ):
        raise ValueError("lifecycle cohort platform/backend does not match runner")
    roots = {
        "client": Path(client_source).resolve(strict=True),
        "server": Path(server_source).resolve(strict=True),
    }
    pythons = {
        "client": Path(client_python).absolute(),
        "server": Path(server_python).absolute(),
    }
    if any(
        not path.resolve(strict=True).is_file() or not os.access(path, os.X_OK)
        for path in pythons.values()
    ):
        raise ValueError("lifecycle probe Python is not executable")
    revisions = {
        component: _git_revision(root)
        for component, root in roots.items()
    }
    plan = lifecycle_probe_plan_v2(cohort)
    suite_digest = _test_suite_digest(roots, plan)
    environment = _runner_environment()
    environment_digest = hashlib.sha256(
        _canonical_bytes(environment),
    ).hexdigest()
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
    _write_private_once(
        logs / "runner-environment.json",
        json.dumps(
            environment, ensure_ascii=True, indent=2, sort_keys=True,
        ).encode("ascii"),
    )
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
    if _test_suite_digest(exact_roots, plan) != suite_digest:
        raise ValueError("lifecycle source archive does not match reviewed tests")
    observed_from = datetime.now(timezone.utc).replace(microsecond=0)
    cases: list[dict[str, Any]] = []
    for case_id in sorted(LIFECYCLE_CASE_IDS_V2):
        probe_results = []
        for index, (component, node_id) in enumerate(
            plan[case_id], start=1,
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
            and probe["junit_valid"]
            and probe["tests_collected"] == 1
            and probe["tests_skipped"] == 0
            and probe["tests_failed"] == 0
            and probe["tests_errored"] == 0
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
    if _observed_platform_backend() != (
        cohort["platform"], cohort["container_backend"],
    ):
        raise ValueError("lifecycle runner platform/backend changed while probes were running")
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
        "runner_environment_sha256": environment_digest,
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
        sandboxed = _run_in_macos_sandbox(args)
        if sandboxed is not None:
            return sandboxed
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
    "lifecycle_probe_plan_v2",
    "run_lifecycle_matrix_v2",
]
