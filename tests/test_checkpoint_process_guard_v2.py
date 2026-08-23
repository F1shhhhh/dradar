from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dradar.checkpoint_process_guard_v2 import (
    CheckpointV2ProcessGuardError,
    capture_pier_process_evidence_v2,
    process_exited_receipt_v2,
    terminate_exact_orphaned_pier_v2,
    validate_pier_process_evidence_v2,
    validate_process_exited_receipt_v2,
)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    home = tmp_path / "home"
    gate = (
        home / "checkpoint-v2" / "paid-gates"
        / "assignment-0001" / ("1" * 32)
    )
    job = home / "work" / "jobs" / "aassignment-0001"
    gate.mkdir(parents=True, mode=0o700)
    job.mkdir(parents=True)
    return home, gate, job


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_exact_gate_bound_process_is_reaped_without_storing_argv(
    tmp_path: Path,
) -> None:
    home, gate, job = _paths(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(120)",
        str(gate),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        evidence = capture_pier_process_evidence_v2(
            process,
            command,
            assignment_id="assignment-0001",
            gate_nonce="1" * 32,
            gate_dir=gate,
            job_root=job,
            home=home,
        )
        encoded = json.dumps(evidence, sort_keys=True)
        assert "time.sleep" not in encoded
        assert str(gate) not in encoded
        assert evidence["pid"] == evidence["pgid"] == process.pid
        assert terminate_exact_orphaned_pier_v2(
            evidence,
            gate_dir=gate,
            grace_sec=0.2,
        ) in {True, False}
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


def test_process_evidence_rejects_job_root_outside_private_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home, gate, _job = _paths(tmp_path)
    monkeypatch.setattr(
        "dradar.checkpoint_process_guard_v2._process_identity",
        lambda _pid, marker: (41, "start", "a" * 64, True),
    )
    process = type("Process", (), {"pid": 41})()
    with pytest.raises(CheckpointV2ProcessGuardError, match="outside"):
        capture_pier_process_evidence_v2(
            process,
            ["pier", str(gate)],
            assignment_id="assignment-0001",
            gate_nonce="1" * 32,
            gate_dir=gate,
            job_root=tmp_path / "somewhere-else",
            home=home,
        )


def test_process_evidence_and_exit_receipt_are_exactly_bound(
    tmp_path: Path,
) -> None:
    home, gate, job = _paths(tmp_path)
    evidence = {
        "schema": "dradar-checkpoint-pier-process-v2",
        "assignment_id": "assignment-0001",
        "gate_nonce": "1" * 32,
        "pid": 99,
        "pgid": 99,
        "process_start_signature": "posix-lstart:Sat Aug 23 12:00:00 2026",
        "command_sha256": "a" * 64,
        "gate_marker_sha256": __import__("hashlib").sha256(
            str(gate.absolute()).encode()
        ).hexdigest(),
        "job_root": str(job.resolve()),
    }
    validated = validate_pier_process_evidence_v2(
        evidence,
        assignment_id="assignment-0001",
        gate_nonce="1" * 32,
        gate_dir=gate,
        home=home,
    )
    receipt = process_exited_receipt_v2(validated, returncode=0)
    validate_process_exited_receipt_v2(receipt, evidence=validated)
    receipt["process_start_signature"] = "reused"
    with pytest.raises(CheckpointV2ProcessGuardError, match="inconsistent"):
        validate_process_exited_receipt_v2(receipt, evidence=validated)

