"""Host-private process fencing for Checkpoint V2 paid-gate recovery.

The outer DRadar process can disappear after it starts Pier because Pier runs
in its own session.  A replacement process must never infer that the old
Provider is gone from a stale assignment alone.  This module records and
revalidates the exact Pier leader/PGID without persisting its command line.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROCESS_EVIDENCE_SCHEMA_V2 = "dradar-checkpoint-pier-process-v2"
PROCESS_EXITED_SCHEMA_V2 = "dradar-checkpoint-pier-process-exited-v2"
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_HEX_32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


class CheckpointV2ProcessGuardError(RuntimeError):
    """A former Pier process cannot be attributed or reaped safely."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _linux_process_identity(pid: int) -> tuple[int, str, str, bool] | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        stat_bytes = stat_path.read_bytes()
        cmdline = cmdline_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process identity is unreadable"
        ) from exc
    close = stat_bytes.rfind(b")")
    if close < 0:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process stat is malformed"
        )
    fields = stat_bytes[close + 2 :].split()
    # fields[0] is state, fields[2] is pgrp, fields[19] is starttime.
    if len(fields) <= 19:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process stat is incomplete"
        )
    try:
        pgid = int(fields[2])
        start_signature = f"linux-proc-ticks:{int(fields[19])}"
    except ValueError as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process stat is invalid"
        ) from exc
    return pgid, start_signature, _sha256(cmdline), bool(cmdline)


def _posix_process_identity(
    pid: int,
    *,
    marker: str,
) -> tuple[int, str, str, bool] | None:
    try:
        result = subprocess.run(
            [
                "ps", "-ww", "-o", "pgid=", "-o", "lstart=",
                "-o", "command=", "-p", str(pid),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process identity could not be queried"
        ) from exc
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip("\n")
    match = re.fullmatch(
        r"\s*(\d+)\s+"
        r"(\S+\s+\S+\s+\S+\s+\S+\s+\S+)\s+(.+)",
        line,
    )
    if match is None:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process identity response is malformed"
        )
    command = match.group(3)
    return (
        int(match.group(1)),
        f"posix-lstart:{match.group(2)}",
        _sha256(command.encode("utf-8", "surrogateescape")),
        marker in command,
    )


def _process_identity(
    pid: int,
    *,
    marker: str,
) -> tuple[int, str, str, bool] | None:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process id is invalid"
        )
    if sys.platform.startswith("linux"):
        identity = _linux_process_identity(pid)
        if identity is None:
            return None
        pgid, started, command_sha256, command_present = identity
        if not command_present:
            raise CheckpointV2ProcessGuardError(
                "checkpoint Pier process command is unavailable"
            )
        try:
            command_bytes = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError as exc:
            raise CheckpointV2ProcessGuardError(
                "checkpoint Pier process command changed during inspection"
            ) from exc
        return pgid, started, command_sha256, (
            marker.encode("utf-8") in command_bytes
        )
    if os.name == "posix":
        return _posix_process_identity(pid, marker=marker)
    raise CheckpointV2ProcessGuardError(
        "checkpoint Pier process fencing is unsupported on this platform"
    )


def _safe_job_root(job_root: Path, home: Path) -> Path:
    try:
        canonical = job_root.resolve(strict=False)
        jobs_root = (home / "work" / "jobs").resolve(strict=False)
        canonical.relative_to(jobs_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier job root is outside the private work tree"
        ) from exc
    if canonical == jobs_root:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier job root is not assignment-scoped"
        )
    return canonical


def capture_pier_process_evidence_v2(
    process: Any,
    command: Sequence[str],
    *,
    assignment_id: str,
    gate_nonce: str,
    gate_dir: Path,
    job_root: Path,
    home: Path,
) -> dict[str, Any]:
    """Capture exact, non-secret evidence before paid authorization is tried."""

    pid = getattr(process, "pid", None)
    if (
        _ID_RE.fullmatch(str(assignment_id)) is None
        or _HEX_32_RE.fullmatch(str(gate_nonce)) is None
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier launch evidence is invalid"
        )
    marker = os.fspath(gate_dir.absolute())
    if not any(marker in item for item in command):
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier command is not bound to its paid gate"
        )
    identity = _process_identity(pid, marker=marker)
    if identity is None:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier exited before process evidence was durable"
        )
    pgid, start_signature, command_sha256, marker_present = identity
    if pgid != pid or not marker_present:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier is not an isolated, gate-bound process group"
        )
    canonical_job_root = _safe_job_root(Path(job_root), Path(home))
    return {
        "schema": PROCESS_EVIDENCE_SCHEMA_V2,
        "assignment_id": assignment_id,
        "gate_nonce": gate_nonce,
        "pid": pid,
        "pgid": pgid,
        "process_start_signature": start_signature,
        "command_sha256": command_sha256,
        "gate_marker_sha256": _sha256(marker.encode("utf-8")),
        "job_root": os.fspath(canonical_job_root),
    }


def validate_pier_process_evidence_v2(
    value: Mapping[str, Any],
    *,
    assignment_id: str,
    gate_nonce: str,
    gate_dir: Path,
    home: Path,
) -> dict[str, Any]:
    expected_fields = {
        "schema", "assignment_id", "gate_nonce", "pid", "pgid",
        "process_start_signature", "command_sha256", "gate_marker_sha256",
        "job_root",
    }
    marker = os.fspath(gate_dir.absolute())
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != PROCESS_EVIDENCE_SCHEMA_V2
        or value.get("assignment_id") != assignment_id
        or value.get("gate_nonce") != gate_nonce
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("pid"), bool)
        or value["pid"] <= 1
        or value.get("pgid") != value.get("pid")
        or not isinstance(value.get("process_start_signature"), str)
        or not 1 <= len(value["process_start_signature"]) <= 160
        or _HEX_64_RE.fullmatch(str(value.get("command_sha256"))) is None
        or value.get("gate_marker_sha256")
        != _sha256(marker.encode("utf-8"))
        or not isinstance(value.get("job_root"), str)
        or not Path(value["job_root"]).is_absolute()
    ):
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process evidence is inconsistent"
        )
    canonical_job_root = _safe_job_root(Path(value["job_root"]), Path(home))
    materialized = dict(value)
    materialized["job_root"] = os.fspath(canonical_job_root)
    return materialized


def validate_process_exited_receipt_v2(
    value: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "schema", "assignment_id", "gate_nonce", "pid", "pgid",
            "process_start_signature", "returncode",
        }
        or value.get("schema") != PROCESS_EXITED_SCHEMA_V2
        or any(
            value.get(field) != evidence.get(field)
            for field in (
                "assignment_id", "gate_nonce", "pid", "pgid",
                "process_start_signature",
            )
        )
        or not isinstance(value.get("returncode"), int)
        or isinstance(value.get("returncode"), bool)
    ):
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier exit receipt is inconsistent"
        )


def process_exited_receipt_v2(
    evidence: Mapping[str, Any],
    *,
    returncode: int,
) -> dict[str, Any]:
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier return code is unavailable"
        )
    return {
        "schema": PROCESS_EXITED_SCHEMA_V2,
        "assignment_id": evidence["assignment_id"],
        "gate_nonce": evidence["gate_nonce"],
        "pid": evidence["pid"],
        "pgid": evidence["pgid"],
        "process_start_signature": evidence["process_start_signature"],
        "returncode": returncode,
    }


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        # macOS may report EPERM for a just-reaped session leader instead of
        # ESRCH.  Confirm whether any process still advertises the PGID before
        # treating that as an ownership failure.
        try:
            listed = subprocess.run(
                ["ps", "-axo", "pgid=", "-o", "state="],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as audit_exc:
            raise CheckpointV2ProcessGuardError(
                "checkpoint Pier process group ownership is unavailable"
            ) from audit_exc
        if listed.returncode == 0:
            states = []
            for line in listed.stdout.splitlines():
                parts = line.split()
                if (
                    len(parts) >= 2
                    and parts[0].isdigit()
                    and int(parts[0]) == pgid
                ):
                    states.append(parts[1])
            if not states or all(state.startswith("Z") for state in states):
                return False
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process group ownership is unavailable"
        ) from exc
    except OSError as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process group could not be audited"
        ) from exc
    return True


def _reap_if_child(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass


def terminate_exact_orphaned_pier_v2(
    evidence: Mapping[str, Any],
    *,
    gate_dir: Path,
    grace_sec: float = 15.0,
) -> bool:
    """Reap only the exact recorded Pier process group.

    If the leader identity cannot still be proven, the function fails closed
    instead of signalling a possibly reused PID/PGID.
    """

    if os.name != "posix":
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier orphan reaping is unsupported on this platform"
        )
    pid = int(evidence["pid"])
    pgid = int(evidence["pgid"])
    marker = os.fspath(gate_dir.absolute())
    current = _process_identity(pid, marker=marker)
    if current is None:
        if _process_group_exists(pgid):
            raise CheckpointV2ProcessGuardError(
                "checkpoint Pier leader vanished while its process group survives"
            )
        return False
    current_pgid, started, command_sha256, marker_present = current
    if (
        current_pgid != pgid
        or started != evidence.get("process_start_signature")
        or command_sha256 != evidence.get("command_sha256")
        or not marker_present
    ):
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier PID was reused or its identity changed"
        )
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process group could not be terminated"
        ) from exc
    deadline = time.monotonic() + max(0.1, min(float(grace_sec), 30.0))
    while time.monotonic() < deadline:
        _reap_if_child(pid)
        if not _process_group_exists(pgid):
            return False
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint Pier process group could not be killed"
        ) from exc
    kill_deadline = time.monotonic() + 5.0
    while time.monotonic() < kill_deadline:
        _reap_if_child(pid)
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.05)
    raise CheckpointV2ProcessGuardError(
        "checkpoint Pier process group survived SIGKILL"
    )


def read_private_json_v2(path: Path, *, maximum: int = 8192) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        data = path.read_bytes()
    except OSError as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint process evidence is unreadable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or len(data) > maximum
        or (
            os.name == "posix"
            and (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            )
        )
    ):
        raise CheckpointV2ProcessGuardError(
            "checkpoint process evidence is unsafe"
        )
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointV2ProcessGuardError(
            "checkpoint process evidence is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointV2ProcessGuardError(
            "checkpoint process evidence is invalid"
        )
    return value


__all__ = [
    "CheckpointV2ProcessGuardError",
    "PROCESS_EVIDENCE_SCHEMA_V2",
    "PROCESS_EXITED_SCHEMA_V2",
    "capture_pier_process_evidence_v2",
    "process_exited_receipt_v2",
    "read_private_json_v2",
    "terminate_exact_orphaned_pier_v2",
    "validate_pier_process_evidence_v2",
    "validate_process_exited_receipt_v2",
]
