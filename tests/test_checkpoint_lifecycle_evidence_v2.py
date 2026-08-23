from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from dradar.checkpoint_lifecycle_evidence_v2 import (
    LIFECYCLE_BUNDLE_SCHEMA_V2,
    LIFECYCLE_CASE_IDS_V2,
    LIFECYCLE_RESULTS_SCHEMA_V2,
    LIFECYCLE_ZERO_METRICS_V2,
    build_lifecycle_matrix_bundle_v2,
    load_lifecycle_matrix_results_v2,
)
from dradar.cli import main as cli_main


NOW = datetime(2026, 8, 23, 8, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=3)
PROTOCOL_VECTOR_ARTIFACT_SHA256 = (
    "8feecf8009a23e91b7b0a482926b4d9ad0e629e2d88a8942789cc58e6d32a509"
)
PROTOCOL_VECTOR_ARTIFACT_ID = (
    "lifecycle-matrix-be8b227e5e90b0dafb6bdc197989fc8915c9c4d5"
)


def _cohort() -> dict[str, str]:
    return {
        "platform": "macos",
        "container_backend": "orbstack",
        "harness": "zcode",
        "provider": "bigmodel-coding-plan",
        "client_version": "0.5.99",
        "agent_version": "0.16.3",
        "runtime_profile": "pier-zcode-glm-v1",
        "model_config_version": "zcode-glm-v1",
        "runtime_compatibility_digest": "a" * 64,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": "dradar-checkpoint-v2/zcode/1",
    }


def _cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "status": "passed",
            "elapsed_ms": index + 1,
            "evidence_sha256": hashlib.sha256(
                case_id.encode("ascii"),
            ).hexdigest(),
            **{field: 0 for field in LIFECYCLE_ZERO_METRICS_V2},
        }
        for index, case_id in enumerate(sorted(LIFECYCLE_CASE_IDS_V2))
    ]


def _bundle(**updates):
    values = {
        "cohort": _cohort(),
        "cases": _cases(),
        "observed_from": SINCE - timedelta(minutes=1),
        "observed_until": NOW + timedelta(minutes=1),
        "source_revisions": {
            "client": "a" * 40,
            "server": "d" * 40,
        },
        "test_suite_sha256": "b" * 64,
        "runner_environment_sha256": "e" * 64,
        "runner_instance_digest": "c" * 64,
        "network_isolated": True,
        "provider_credentials_present": False,
    }
    values.update(updates)
    return build_lifecycle_matrix_bundle_v2(**values)


def _results() -> dict[str, object]:
    return {
        "schema": LIFECYCLE_RESULTS_SCHEMA_V2,
        "cohort": _cohort(),
        "cases": _cases(),
        "observed_from": (SINCE - timedelta(minutes=1)).isoformat(),
        "observed_until": (NOW + timedelta(minutes=1)).isoformat(),
        "source_revisions": {
            "client": "a" * 40,
            "server": "d" * 40,
        },
        "test_suite_sha256": "b" * 64,
        "runner_environment_sha256": "e" * 64,
        "runner_instance_digest": "c" * 64,
        "network_isolated": True,
        "provider_credentials_present": False,
    }


def test_lifecycle_bundle_is_deterministic_complete_and_digest_bound() -> None:
    first = _bundle()
    second = _bundle(cases=list(reversed(_cases())))
    assert first == second
    assert first["schema"] == LIFECYCLE_BUNDLE_SCHEMA_V2
    artifact = first["artifact"]
    canonical = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    attestation = first["attestation"]
    assert attestation["artifact_sha256"] == hashlib.sha256(
        canonical,
    ).hexdigest()
    assert attestation["artifact_sha256"] == PROTOCOL_VECTOR_ARTIFACT_SHA256
    assert artifact["artifact_id"] == PROTOCOL_VECTOR_ARTIFACT_ID
    assert attestation["cohort"] == artifact["cohort"]
    assert attestation["metrics"] == {
        "crash_cuts_passed": len(LIFECYCLE_CASE_IDS_V2),
        "failed_crash_cuts": 0,
        **{field: 0 for field in LIFECYCLE_ZERO_METRICS_V2},
    }


def test_lifecycle_bundle_preserves_failures_in_the_denominator() -> None:
    cases = _cases()
    cases[0]["status"] = "failed"
    cases[0]["cleanup_residue"] = 1
    metrics = _bundle(cases=cases)["attestation"]["metrics"]
    assert metrics["crash_cuts_passed"] == len(LIFECYCLE_CASE_IDS_V2) - 1
    assert metrics["failed_crash_cuts"] == 1
    assert metrics["cleanup_residue"] == 1


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_lifecycle_bundle_refuses_incomplete_or_duplicate_matrix(mode: str) -> None:
    cases = _cases()
    if mode == "missing":
        cases.pop()
    else:
        cases[-1] = dict(cases[0])
    with pytest.raises(ValueError, match="incomplete|duplicate"):
        _bundle(cases=cases)


@pytest.mark.parametrize(
    ("network_isolated", "provider_credentials_present"),
    [(False, False), (True, True)],
)
def test_lifecycle_bundle_requires_credential_free_network_isolation(
    network_isolated: bool,
    provider_credentials_present: bool,
) -> None:
    with pytest.raises(ValueError, match="credential-free isolated"):
        _bundle(
            network_isolated=network_isolated,
            provider_credentials_present=provider_credentials_present,
        )


def test_lifecycle_bundle_rejects_cohort_or_case_drift() -> None:
    cohort = _cohort()
    cohort["runtime_compatibility_digest"] = "not-a-digest"
    with pytest.raises(ValueError, match="runtime compatibility digest"):
        _bundle(cohort=cohort)

    cases = _cases()
    cases[0]["case_id"] = "unreviewed-cut"
    with pytest.raises(ValueError, match="case id"):
        _bundle(cases=cases)


@pytest.mark.parametrize(
    "source_revisions",
    [
        {"client": "a" * 40},
        {"client": "a" * 40, "server": "not-a-revision"},
        {"client": "a" * 40, "server": "d" * 40, "worker": "e" * 40},
    ],
)
def test_lifecycle_bundle_requires_exact_client_and_server_revisions(
    source_revisions,
) -> None:
    with pytest.raises(ValueError, match="source revision"):
        _bundle(source_revisions=source_revisions)


def test_private_runner_results_round_trip_through_the_cli(tmp_path, capsys) -> None:
    results = tmp_path / "lifecycle-results.json"
    results.write_text(json.dumps(_results()), encoding="utf-8")
    results.chmod(0o600)

    assert load_lifecycle_matrix_results_v2(results) == _bundle()
    assert cli_main([
        "checkpoint", "lifecycle-bundle", "--results", os.fspath(results),
    ]) == 0
    assert json.loads(capsys.readouterr().out) == _bundle()


def test_runner_results_refuse_public_or_symlinked_input(tmp_path, capsys) -> None:
    results = tmp_path / "lifecycle-results.json"
    results.write_text(json.dumps(_results()), encoding="utf-8")
    results.chmod(0o644)
    assert cli_main([
        "checkpoint", "lifecycle-bundle", "--results", os.fspath(results),
    ]) == 1
    assert "not private" in capsys.readouterr().err

    results.chmod(0o600)
    linked = tmp_path / "linked-results.json"
    linked.symlink_to(results)
    with pytest.raises(ValueError, match="private regular file"):
        load_lifecycle_matrix_results_v2(linked)
