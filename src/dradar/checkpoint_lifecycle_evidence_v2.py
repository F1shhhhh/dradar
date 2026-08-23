"""Strict, privacy-bounded Checkpoint V2 lifecycle evidence bundles.

The bundle is an operator-review input, never an activation credential.  It
contains no paths, logs, commands, hostnames, prompts, or Provider material.
Every required crash cut must be present exactly once so a runner cannot turn
an incomplete matrix into a green attestation by omitting failed cases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint_observation_v2 import (
    CHECKPOINT_COHORT_FIELDS_V2,
    EVIDENCE_ATTESTATION_SCHEMA_V2,
)


LIFECYCLE_ARTIFACT_SCHEMA_V2 = (
    "dradar-checkpoint-v2-lifecycle-matrix-artifact-v2"
)
LIFECYCLE_BUNDLE_SCHEMA_V2 = (
    "dradar-checkpoint-v2-reviewed-evidence-bundle-v2"
)
LIFECYCLE_RESULTS_SCHEMA_V2 = (
    "dradar-checkpoint-v2-lifecycle-matrix-results-v2"
)
LIFECYCLE_SOURCE_COMPONENTS_V2 = ("client", "server")
LIFECYCLE_CASE_IDS_V2 = frozenset({
    "owner_checkout_precommit",
    "owner_start_precommit",
    "owner_start_response_lost",
    "capture_interrupted",
    "seal_interrupted",
    "download_interrupted",
    "publication_interrupted",
    "restore_interrupted",
    "restore_before_paid_commit",
    "paid_resume_response_lost",
    "completed_result_finalize_failure",
    "submission_response_lost",
    "retention_response_lost",
    "process_orphan_reconciliation",
})
LIFECYCLE_ZERO_METRICS_V2 = (
    "unsupported_state_transitions",
    "paid_execution_in_restore_test",
    "duplicate_paid_segments",
    "result_losses",
    "cleanup_residue",
)

_HEX_RE = re.compile(r"[0-9a-f]+")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("lifecycle evidence timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _digest(value: object, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or _HEX_RE.fullmatch(value) is None
    ):
        raise ValueError(f"lifecycle evidence {field} is invalid")
    return value


def _cohort(value: Mapping[str, Any]) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(CHECKPOINT_COHORT_FIELDS_V2)
    ):
        raise ValueError("lifecycle evidence cohort fields are invalid")
    result: dict[str, str] = {}
    for field in CHECKPOINT_COHORT_FIELDS_V2:
        item = value[field]
        if not isinstance(item, str) or not item or len(item) > 200:
            raise ValueError("lifecycle evidence cohort value is invalid")
        result[field] = item
    _digest(
        result["runtime_compatibility_digest"],
        field="runtime compatibility digest",
    )
    return result


def _case(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "case_id", "status", "elapsed_ms", "evidence_sha256",
        *LIFECYCLE_ZERO_METRICS_V2,
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("lifecycle evidence case fields are invalid")
    case_id = value["case_id"]
    if case_id not in LIFECYCLE_CASE_IDS_V2:
        raise ValueError("lifecycle evidence case id is invalid")
    status = value["status"]
    if status not in {"passed", "failed"}:
        raise ValueError("lifecycle evidence case status is invalid")
    elapsed_ms = value["elapsed_ms"]
    if (
        not isinstance(elapsed_ms, int)
        or isinstance(elapsed_ms, bool)
        or not 0 <= elapsed_ms <= 24 * 60 * 60 * 1000
    ):
        raise ValueError("lifecycle evidence elapsed time is invalid")
    result: dict[str, Any] = {
        "case_id": case_id,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "evidence_sha256": _digest(
            value["evidence_sha256"], field="case digest",
        ),
    }
    for field in LIFECYCLE_ZERO_METRICS_V2:
        metric = value[field]
        if (
            not isinstance(metric, int)
            or isinstance(metric, bool)
            or not 0 <= metric <= 1_000_000
        ):
            raise ValueError("lifecycle evidence case metric is invalid")
        result[field] = metric
    return result


def _source_revisions(value: Mapping[str, Any]) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(LIFECYCLE_SOURCE_COMPONENTS_V2)
    ):
        raise ValueError("lifecycle evidence source revisions are invalid")
    return {
        component: _digest(
            value[component],
            field=f"{component} source revision",
            lengths=(40, 64),
        )
        for component in LIFECYCLE_SOURCE_COMPONENTS_V2
    }


def _metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "crash_cuts_passed": sum(case["status"] == "passed" for case in cases),
        "failed_crash_cuts": sum(case["status"] == "failed" for case in cases),
        **{
            field: sum(int(case[field]) for case in cases)
            for field in LIFECYCLE_ZERO_METRICS_V2
        },
    }


def build_lifecycle_matrix_bundle_v2(
    *,
    cohort: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    observed_from: datetime,
    observed_until: datetime,
    source_revisions: Mapping[str, Any],
    test_suite_sha256: str,
    runner_environment_sha256: str,
    runner_instance_digest: str,
    network_isolated: bool,
    provider_credentials_present: bool,
) -> dict[str, Any]:
    """Build one exact-cohort lifecycle artifact and bound attestation.

    Failed cases remain representable and therefore block promotion. Missing,
    duplicate, skipped, or unstructured cases are refused instead of silently
    disappearing from the denominator.
    """

    normalized_cohort = _cohort(cohort)
    start = _utc(observed_from)
    end = _utc(observed_until)
    if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
        raise ValueError("lifecycle evidence window is invalid")
    if not isinstance(network_isolated, bool):
        raise ValueError("lifecycle evidence network isolation is invalid")
    if not isinstance(provider_credentials_present, bool):
        raise ValueError("lifecycle evidence credential fact is invalid")
    if not network_isolated or provider_credentials_present:
        raise ValueError(
            "lifecycle evidence requires a credential-free isolated runner"
        )
    normalized_cases = [_case(item) for item in cases]
    case_ids = [item["case_id"] for item in normalized_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("lifecycle evidence contains duplicate cases")
    if set(case_ids) != LIFECYCLE_CASE_IDS_V2:
        raise ValueError("lifecycle evidence matrix is incomplete")
    normalized_cases.sort(key=lambda item: item["case_id"])
    runner = {
        "platform": normalized_cohort["platform"],
        "container_backend": normalized_cohort["container_backend"],
        "client_version": normalized_cohort["client_version"],
        "source_revisions": _source_revisions(source_revisions),
        "test_suite_sha256": _digest(
            test_suite_sha256, field="test suite digest",
        ),
        "runner_environment_sha256": _digest(
            runner_environment_sha256, field="runner environment digest",
        ),
        "runner_instance_digest": _digest(
            runner_instance_digest, field="runner instance digest",
        ),
        "network_isolated": network_isolated,
        "provider_credentials_present": provider_credentials_present,
    }
    core = {
        "schema": LIFECYCLE_ARTIFACT_SCHEMA_V2,
        "kind": "lifecycle_matrix",
        "cohort": normalized_cohort,
        "observed_from": start,
        "observed_until": end,
        "runner": runner,
        "cases": normalized_cases,
    }
    identity = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    artifact = {
        **core,
        "artifact_id": f"lifecycle-matrix-{identity[:40]}",
    }
    artifact_sha256 = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
    attestation = {
        "schema": EVIDENCE_ATTESTATION_SCHEMA_V2,
        "attestation_id": f"lifecycle-matrix-{artifact_sha256[:40]}",
        "kind": "lifecycle_matrix",
        "cohort": normalized_cohort,
        "observed_from": start,
        "observed_until": end,
        "artifact_sha256": artifact_sha256,
        "metrics": _metrics(normalized_cases),
    }
    return {
        "schema": LIFECYCLE_BUNDLE_SCHEMA_V2,
        "artifact": artifact,
        "attestation": attestation,
    }


def _read_private_results(path: Path) -> Mapping[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("lifecycle result input is not a private regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_dev != before.st_dev
            or metadata.st_ino != before.st_ino
        ):
            raise ValueError("lifecycle result input is not a private regular file")
        if os.name == "posix" and metadata.st_uid != os.getuid():
            raise ValueError("lifecycle result input has another owner")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("lifecycle result input is not private")
        if not 0 < metadata.st_size <= 64 * 1024:
            raise ValueError("lifecycle result input size is invalid")
        chunks: list[bytes] = []
        remaining = 64 * 1024 + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) != metadata.st_size:
            raise ValueError("lifecycle result input changed during read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("lifecycle result input is not JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("lifecycle result input is not an object")
    return value


def load_lifecycle_matrix_results_v2(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Convert one strict offline runner result into a reviewed-input bundle."""

    value = _read_private_results(Path(path))
    expected = {
        "schema", "cohort", "cases", "observed_from", "observed_until",
        "source_revisions", "test_suite_sha256", "runner_environment_sha256",
        "runner_instance_digest",
        "network_isolated", "provider_credentials_present",
    }
    if set(value) != expected or value["schema"] != LIFECYCLE_RESULTS_SCHEMA_V2:
        raise ValueError("lifecycle result input fields are invalid")
    try:
        observed_from = datetime.fromisoformat(value["observed_from"])
        observed_until = datetime.fromisoformat(value["observed_until"])
    except (TypeError, ValueError) as exc:
        raise ValueError("lifecycle result input window is invalid") from exc
    return build_lifecycle_matrix_bundle_v2(
        cohort=value["cohort"],
        cases=value["cases"],
        observed_from=observed_from,
        observed_until=observed_until,
        source_revisions=value["source_revisions"],
        test_suite_sha256=value["test_suite_sha256"],
        runner_environment_sha256=value["runner_environment_sha256"],
        runner_instance_digest=value["runner_instance_digest"],
        network_isolated=value["network_isolated"],
        provider_credentials_present=value["provider_credentials_present"],
    )


def cmd_checkpoint_lifecycle_bundle(args) -> int:
    """Emit a digest-bound bundle without contacting DRadar or a Provider."""

    try:
        bundle = load_lifecycle_matrix_results_v2(args.results)
    except (OSError, ValueError) as exc:
        print(f"checkpoint lifecycle evidence refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "LIFECYCLE_ARTIFACT_SCHEMA_V2",
    "LIFECYCLE_BUNDLE_SCHEMA_V2",
    "LIFECYCLE_CASE_IDS_V2",
    "LIFECYCLE_RESULTS_SCHEMA_V2",
    "LIFECYCLE_SOURCE_COMPONENTS_V2",
    "LIFECYCLE_ZERO_METRICS_V2",
    "build_lifecycle_matrix_bundle_v2",
    "cmd_checkpoint_lifecycle_bundle",
    "load_lifecycle_matrix_results_v2",
]
