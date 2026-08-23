from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from dradar import __version__
from dradar.checkpoint_lifecycle_evidence_v2 import (
    LIFECYCLE_CASE_IDS_V2,
    load_lifecycle_matrix_results_v2,
)
from dradar import checkpoint_lifecycle_runner_v2 as runner


@pytest.fixture(autouse=True)
def _verified_runner_platform(monkeypatch):
    monkeypatch.setattr(
        runner, "_observed_platform_backend", lambda: ("linux", "docker"),
    )
    monkeypatch.setattr(
        runner,
        "_runner_environment",
        lambda: {
            "python": {"implementation": "cpython", "version": "3.12.0"},
            "platform": {"system": "linux", "machine": "x86_64"},
            "docker": {"client": {"version": "1"}, "server": {"version": "1"}},
            "packages": [{"name": "pytest", "version": "9.1.1"}],
        },
    )


def _cohort() -> dict[str, str]:
    return {
        "platform": "linux",
        "container_backend": "docker",
        "harness": "codex",
        "provider": "openai",
        "client_version": __version__,
        "agent_version": "1.2.3",
        "runtime_profile": "codex-runtime-v1",
        "model_config_version": "codex-config-v1",
        "runtime_compatibility_digest": "a" * 64,
        "checkpoint_core_abi": "dradar-checkpoint-core-v2/1",
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
    }


def _commit_fake_source(root: Path, component: str) -> str:
    plan = runner.lifecycle_probe_plan_v2(_cohort())
    paths = {
        node_id.split("::", 1)[0]
        for probes in plan.values()
        for probe_component, node_id in probes
        if probe_component == component
    }
    if component == "client":
        paths.update({
            "src/dradar/checkpoint_lifecycle_runner_v2.py",
            "src/dradar/checkpoint_lifecycle_evidence_v2.py",
        })
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {component}:{relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(
        ["git", "-C", root, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", root, "config", "user.name", "Checkpoint Test"],
        check=True,
    )
    subprocess.run(["git", "-C", root, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", root, "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _inputs(tmp_path: Path):
    client = tmp_path / "client"
    server = tmp_path / "server"
    revisions = {
        "client": _commit_fake_source(client, "client"),
        "server": _commit_fake_source(server, "server"),
    }
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cohort = private / "cohort.json"
    cohort.write_text(json.dumps(_cohort()), encoding="utf-8")
    cohort.chmod(0o600)
    return client, server, revisions, cohort, private / "results.json"


def _successful_probe(**kwargs):
    return {
        "component": kwargs["component"],
        "node_id": kwargs["node_id"],
        "returncode": 0,
        "timed_out": False,
        "elapsed_ms": 7,
        "stdout": b"one passed\n",
        "stderr": b"",
        "stdout_sha256": "b" * 64,
        "stderr_sha256": "c" * 64,
        "output_truncated": False,
        "junit_valid": True,
        "tests_collected": 1,
        "tests_skipped": 0,
        "tests_failed": 0,
        "tests_errored": 0,
    }


def test_runner_executes_fixed_complete_plan_and_writes_private_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, server, revisions, cohort, output = _inputs(tmp_path)
    monkeypatch.setattr(runner, "_network_isolated", lambda: True)
    monkeypatch.setattr(runner, "_provider_credentials_present", lambda: False)
    monkeypatch.setattr(runner, "_run_probe", _successful_probe)

    result = runner.run_lifecycle_matrix_v2(
        cohort_path=cohort,
        output_path=output,
        client_source=client,
        server_source=server,
        client_python=Path(sys.executable),
        server_python=Path(sys.executable),
    )
    assert result["source_revisions"] == revisions
    assert {case["case_id"] for case in result["cases"]} == (
        LIFECYCLE_CASE_IDS_V2
    )
    assert all(case["status"] == "passed" for case in result["cases"])
    assert result["network_isolated"] is True
    assert result["provider_credentials_present"] is False
    assert len(result["runner_environment_sha256"]) == 64
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    logs = list(output.parent.glob(f".{output.name}.logs-*"))
    assert len(logs) == 1
    assert stat.S_IMODE(logs[0].stat().st_mode) == 0o700
    assert not (logs[0] / "exact-sources").exists()
    environment_log = logs[0] / "runner-environment.json"
    assert stat.S_IMODE(environment_log.stat().st_mode) == 0o600
    log_files = [
        path for path in logs[0].glob("*.json")
        if path != environment_log
    ]
    assert len(log_files) == sum(
        len(probes)
        for probes in runner.lifecycle_probe_plan_v2(_cohort()).values()
    )
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in log_files)
    bundle = load_lifecycle_matrix_results_v2(output)
    assert bundle["artifact"]["runner"]["source_revisions"] == revisions


def test_runner_records_failure_without_omitting_later_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, server, _revisions, cohort, output = _inputs(tmp_path)
    monkeypatch.setattr(runner, "_network_isolated", lambda: True)
    monkeypatch.setattr(runner, "_provider_credentials_present", lambda: False)

    def one_failure(**kwargs):
        probe = _successful_probe(**kwargs)
        if kwargs["node_id"] == runner.LIFECYCLE_PROBES_V2[
            "capture_interrupted"
        ][0][1]:
            probe["returncode"] = 1
            probe["stderr"] = b"private failure detail"
            probe["stderr_sha256"] = "d" * 64
        return probe

    monkeypatch.setattr(runner, "_run_probe", one_failure)
    result = runner.run_lifecycle_matrix_v2(
        cohort_path=cohort,
        output_path=output,
        client_source=client,
        server_source=server,
        client_python=Path(sys.executable),
        server_python=Path(sys.executable),
    )
    assert len(result["cases"]) == len(LIFECYCLE_CASE_IDS_V2)
    failed = [case for case in result["cases"] if case["status"] == "failed"]
    assert [case["case_id"] for case in failed] == ["capture_interrupted"]
    assert "private failure detail" not in output.read_text(encoding="ascii")


def test_probe_requires_one_collected_non_skipped_test(tmp_path: Path) -> None:
    source = tmp_path / "source"
    tests = source / "tests"
    tests.mkdir(parents=True)
    (tests / "test_probe.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "def test_skip(): pytest.skip('not evidence')\n",
        encoding="utf-8",
    )
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(mode=0o700)
    passed = runner._run_probe(
        component="client",
        node_id="tests/test_probe.py::test_pass",
        source_root=source,
        python=Path(sys.executable),
        sandbox_home=sandbox,
        timeout_sec=30,
    )
    assert passed["returncode"] == 0
    assert passed["junit_valid"] is True
    assert passed["tests_collected"] == 1
    assert passed["tests_skipped"] == 0

    skipped = runner._run_probe(
        component="client",
        node_id="tests/test_probe.py::test_skip",
        source_root=source,
        python=Path(sys.executable),
        sandbox_home=sandbox,
        timeout_sec=30,
    )
    assert skipped["returncode"] == 0
    assert skipped["junit_valid"] is True
    assert skipped["tests_collected"] == 1
    assert skipped["tests_skipped"] == 1


@pytest.mark.parametrize(
    ("network_isolated", "credentials", "message"),
    [
        (False, False, "not network-isolated"),
        (True, True, "Provider credentials"),
    ],
)
def test_runner_refuses_without_isolation_or_with_credentials(
    tmp_path: Path,
    monkeypatch,
    network_isolated: bool,
    credentials: bool,
    message: str,
) -> None:
    client, server, _revisions, cohort, output = _inputs(tmp_path)
    monkeypatch.setattr(runner, "_network_isolated", lambda: network_isolated)
    monkeypatch.setattr(
        runner, "_provider_credentials_present", lambda: credentials,
    )
    with pytest.raises(ValueError, match=message):
        runner.run_lifecycle_matrix_v2(
            cohort_path=cohort,
            output_path=output,
            client_source=client,
            server_source=server,
            client_python=Path(sys.executable),
            server_python=Path(sys.executable),
        )
    assert not output.exists()


def test_runner_refuses_dirty_source_and_existing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, server, _revisions, cohort, output = _inputs(tmp_path)
    monkeypatch.setattr(runner, "_network_isolated", lambda: True)
    monkeypatch.setattr(runner, "_provider_credentials_present", lambda: False)
    monkeypatch.setattr(runner, "_run_probe", _successful_probe)
    tracked = client / "src/dradar/checkpoint_lifecycle_runner_v2.py"
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local changes"):
        runner.run_lifecycle_matrix_v2(
            cohort_path=cohort,
            output_path=output,
            client_source=client,
            server_source=server,
            client_python=Path(sys.executable),
            server_python=Path(sys.executable),
        )
    assert not output.exists()

    subprocess.run(["git", "-C", client, "checkout", "--", "."], check=True)
    output.write_text("do not overwrite", encoding="utf-8")
    output.chmod(0o600)
    with pytest.raises(ValueError, match="already exists"):
        runner.run_lifecycle_matrix_v2(
            cohort_path=cohort,
            output_path=output,
            client_source=client,
            server_source=server,
            client_python=Path(sys.executable),
            server_python=Path(sys.executable),
        )
    assert output.read_text() == "do not overwrite"


def test_probe_plan_has_no_operator_selectors_or_missing_cases() -> None:
    assert set(runner.LIFECYCLE_PROBES_V2) == LIFECYCLE_CASE_IDS_V2
    assert all(runner.LIFECYCLE_PROBES_V2.values())
    for harness, provider in (
        ("codex", "openai"),
        ("codex", "deepseek"),
        ("dsh", "deepseek"),
        ("kimi-code", "kimi-subscription"),
        ("zcode", "bigmodel-coding-plan"),
    ):
        cohort = _cohort() | {"harness": harness, "provider": provider}
        cohort["checkpoint_abi"] = f"dradar-checkpoint-v2/{harness}/1"
        plan = runner.lifecycle_probe_plan_v2(cohort)
        assert set(plan) == LIFECYCLE_CASE_IDS_V2
        assert all(plan.values())
        assert {
            component
            for probes in plan.values()
            for component, _node_id in probes
        } == {"client", "server"}
        selected = "\n".join(
            node_id for probes in plan.values() for _component, node_id in probes
        )
        assert f"[{harness}-{provider}]" in selected
