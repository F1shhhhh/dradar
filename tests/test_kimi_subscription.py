"""Kimi Code integration is K3 subscription OAuth with native concurrency."""

from __future__ import annotations

import json
import os
import ast
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import dradar.providers as providers
import dradar.runner as runner
from dradar.providers import (
    KIMI_AGENT,
    KIMI_API_KEY_ENVS,
    KIMI_CAPABILITY,
    KIMI_CLI_VERSION,
    KIMI_MODEL,
    KIMI_PROVIDER,
    advertised_capabilities,
    kimi_auth_error,
    kimi_auth_path,
    kimi_live_error,
    kimi_subscription_session,
    parse_kimi_cli_version,
)
from dradar.runner import RunnerError


def _oauth(access: str = "access", refresh: str = "refresh") -> dict:
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_at": 4_102_444_800,
    }


def _write_auth(path: Path, payload: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _oauth()), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)
    return path


def _assignment(**overrides: object) -> dict:
    value = {
        "assignment_id": "a-kimi-1",
        "task_id": "task-1",
        "agent": KIMI_AGENT,
        "provider": KIMI_PROVIDER,
        "model": KIMI_MODEL,
        "effort": "high",
        "agent_version": KIMI_CLI_VERSION,
        "est_minutes": 5,
    }
    value.update(overrides)
    return value


def test_official_kimi_version_banner_is_parsed() -> None:
    assert parse_kimi_cli_version("0.36.1\n") == KIMI_CLI_VERSION
    assert parse_kimi_cli_version("kimi version 0.36.1\n") == KIMI_CLI_VERSION
    assert parse_kimi_cli_version("unexpected") is None


def test_kimi_oauth_validator_rejects_api_key_shaped_auth(tmp_path: Path) -> None:
    path = _write_auth(tmp_path / "auth.json", {"api_key": "secret"})
    assert "not a refreshable subscription OAuth" in (kimi_auth_error(path) or "")


def test_kimi_oauth_validator_rejects_symlink_and_broad_mode(
    tmp_path: Path,
) -> None:
    target = _write_auth(tmp_path / "target.json")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert "not a symlink" in (kimi_auth_error(link) or "")
    if os.name != "nt":
        target.chmod(0o644)
        assert "too broadly readable" in (kimi_auth_error(target) or "")


def test_kimi_subscription_session_uses_canonical_native_lock_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("DRADAR_HOME", str(home))
    canonical = _write_auth(kimi_auth_path(), _oauth("old", "old-refresh"))

    with kimi_subscription_session(tmp_path / "work") as shared:
        assert shared == canonical
        assert (home / "providers" / "kimi" / "oauth" / "kimi-code").is_file()
        assert (home / "providers" / "kimi" / "credentials").is_dir()

    assert canonical.is_file()


def test_kimi_live_probe_uses_proxy_and_writes_back_rotated_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _write_auth(tmp_path / "auth.json", _oauth("old", "old-refresh"))
    seen = {}
    monkeypatch.setattr(
        providers,
        "provider_subprocess_env",
        lambda: {
            "HTTPS_PROXY": "http://127.0.0.1:18080",
            "KIMI_API_KEY": "must-not-leak",
        },
    )

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, env=kwargs["env"])
        native = (
            Path(kwargs["env"]["KIMI_CODE_HOME"])
            / "credentials" / "kimi-code.json"
        )
        _write_auth(native, _oauth("new", "new-refresh"))
        (Path(kwargs["env"]["KIMI_CODE_HOME"]) / "config.toml").write_text(
            '[models."kimi-code/k3"]\nmodel = "k3"\n'
        )
        return providers.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(providers.subprocess, "run", fake_run)

    assert kimi_live_error("/managed/kimi", auth) is None
    assert json.loads(auth.read_text())["refresh_token"] == "new-refresh"
    assert seen["env"]["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert "KIMI_API_KEY" not in seen["env"]
    assert "old-refresh" not in " ".join(seen["cmd"])
    assert "new-refresh" not in " ".join(seen["cmd"])


def test_kimi_live_probe_rejects_missing_k3_without_losing_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _write_auth(tmp_path / "auth.json", _oauth("old", "old-refresh"))
    def fake_run(cmd, **kwargs):
        native = (
            Path(kwargs["env"]["KIMI_CODE_HOME"])
            / "credentials" / "kimi-code.json"
        )
        _write_auth(native, _oauth("new", "new-refresh"))
        (Path(kwargs["env"]["KIMI_CODE_HOME"]) / "config.toml").write_text(
            '[models."kimi-code/k2"]\nmodel = "k2"\n'
        )
        return providers.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(providers.subprocess, "run", fake_run)

    assert "cannot access k3" in (kimi_live_error("/managed/kimi", auth) or "")
    assert json.loads(auth.read_text())["refresh_token"] == "new-refresh"


def test_kimi_live_probe_distinguishes_revoked_oauth_from_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _write_auth(tmp_path / "auth.json")
    monkeypatch.setattr(
        providers.subprocess,
        "run",
        lambda cmd, **kwargs: providers.subprocess.CompletedProcess(
            cmd, 1, "", "invalid_grant",
        ),
    )

    issue = kimi_live_error("/managed/kimi", auth) or ""
    assert "OAuth session was rejected" in issue
    assert "provider setup kimi" in issue


def test_kimi_capability_requires_cli_and_safe_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRADAR_HOME", str(tmp_path / "home"))
    assert KIMI_CAPABILITY not in advertised_capabilities({})
    _write_auth(kimi_auth_path())
    assert KIMI_CAPABILITY in advertised_capabilities({"KIMI_CLI_PATH": "/kimi"})


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_pier_command_uses_private_kimi_adapter_without_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/pier")
    for name in KIMI_API_KEY_ENVS:
        monkeypatch.setenv(name, "must-not-leak")
    tasks = tmp_path / "tasks"
    (tasks / "task-1").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    auth = _write_auth(
        tmp_path / "providers" / "kimi" / "credentials" / "kimi-code.json"
    )
    cli = tmp_path / "kimi"
    cli.write_text("binary", encoding="utf-8")

    assignment = _assignment(effort=effort)
    cmd = runner.build_pier_command(
        assignment,
        tasks,
        tmp_path / "jobs",
        "job",
        home,
        provider_auth_path=auth,
        provider_cli_path=cli,
    )

    joined = " ".join(cmd)
    assert runner.KIMI_AGENT_IMPORT_PATH in cmd
    assert f"reasoning_effort={effort}" in cmd
    assert f"auth_json_file={auth}" in cmd
    assert "shared_oauth=true" in cmd
    assert runner.SHARED_OAUTH_ENV_IMPORT_PATH in cmd
    assert f"kimi_cli_file={cli}" in cmd
    assert f"version={KIMI_CLI_VERSION}" in cmd
    assert "must-not-leak" not in joined
    adapter = home / runner.KIMI_AGENT_MODULE_FILENAME
    assert adapter.read_bytes() == Path(runner.__file__).with_name("pier_kimi.py").read_bytes()
    recovery = home / runner.KIMI_RECOVERY_MODULE_FILENAME
    assert recovery.read_bytes() == (
        Path(runner.__file__).with_name("kimi_recovery.py").read_bytes()
    )

    env = runner._pier_process_env(assignment, kimi_module_dir=home)
    assert all(name not in env for name in KIMI_API_KEY_ENVS)
    assert env["PYTHONPATH"] == str(home)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"provider": "kimi-api"}, "explicitly use provider"),
        ({"model": "k3-256k"}, "unsupported Kimi subscription model"),
        ({"effort": "medium"}, "effort must be low, high, or max"),
        ({"agent_version": "9.9.9"}, "pinned to CLI"),
    ],
)
def test_unverified_kimi_assignments_fail_before_paid_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
    message: str,
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/pier")
    assignment = _assignment(**overrides)
    tasks = tmp_path / "tasks"
    (tasks / assignment["task_id"]).mkdir(parents=True)
    auth = _write_auth(tmp_path / "auth.json")
    cli = tmp_path / "kimi"
    cli.write_text("binary", encoding="utf-8")
    with pytest.raises(RunnerError, match=message):
        runner.build_pier_command(
            assignment,
            tasks,
            tmp_path / "jobs",
            "job",
            tmp_path,
            provider_auth_path=auth,
            provider_cli_path=cli,
        )


def test_kimi_checkpoint_resume_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/pier")
    tasks = tmp_path / "tasks"
    (tasks / "task-1").mkdir(parents=True)
    auth = _write_auth(tmp_path / "auth.json")
    cli = tmp_path / "kimi"
    cli.write_text("binary", encoding="utf-8")
    with pytest.raises(RunnerError, match="checkpoints are not supported"):
        runner.build_pier_command(
            _assignment(),
            tasks,
            tmp_path / "jobs",
            "job",
            tmp_path,
            resume_checkpoint=tmp_path / "checkpoint",
            provider_auth_path=auth,
            provider_cli_path=cli,
        )


def test_kimi_adapter_source_has_fixed_security_contract() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    assert 'return NetworkAllowlist(domains=["auth.kimi.com", "api.kimi.com"])' in source
    assert 'enabled = ["Read", "ReadMediaFile", "Glob", "Grep", "Write", "Edit", "Bash"]' in source
    assert '"WebSearch"' not in source
    assert '"FetchURL"' not in source
    assert 'event = "PreToolUse"' in source
    assert '"KIMI_CODE_HOME": remote_home' in source
    assert "run_with_kimi_resume" in source
    assert '"/logs/agent/kimi-code.stderr.log"' in source
    assert "tail -n 1" in source
    assert "classify_retryable_error=classify_retryable_error" in source
    assert '"--session", session_id' in source
    assert 'tee = "tee -a" if append else "tee"' in source
    assert '"--config-file"' not in source
    assert '"--agent-file"' not in source
    assert "KIMI_MODEL_THINKING_EFFORT" in source
    assert "kimi-code-linux-${kimi_arch}" in source
    assert "KIMI_DISABLE_TELEMETRY" in source
    assert "KIMI_DISABLE_CRON" in source
    assert "[REDACTED_KIMI_CREDENTIAL]" in source


def test_kimi_wire_usage_sums_request_records_without_cache_overlap() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    module = ast.parse(source)
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_usage_instant", "_kimi_usage_facts"}
    ]
    namespace = {
        "Any": Any, "datetime": datetime, "timezone": timezone,
        "deque": deque,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "pier_kimi.py", "exec"),
         namespace)
    status = lambda usage, at: {
        "time": at,
        "type": "usage.record",
        "usageScope": "turn",
        "model": "kimi-code/k3",
        "usage": usage,
    }
    facts = namespace["_kimi_usage_facts"]([
        {"type": "metadata", "protocol_version": "1"},
        {"type": "turn.prompt", "time": "2026-08-18T00:59:59Z"},
        {"type": "llm.request", "model": "k3",
         "turnStep": "1.1", "time": "2026-08-18T00:59:59.500Z"},
        status({
            "inputOther": 1_964,
            "inputCacheCreation": 101,
            "inputCacheRead": 19_200,
            "output": 27,
        }, "2026-08-18T01:00:00Z"),
        {"type": "llm.request", "model": "k3",
         "turnStep": "1.2", "time": "2026-08-18T01:00:01Z"},
        status({
            "inputOther": 10,
            "inputCacheCreation": 20,
            "inputCacheRead": 30,
            "output": 4,
        }, "2026-08-18T01:00:02Z"),
        {"type": "llm.request", "model": "k3",
         "turnStep": "1.3", "time": "2026-08-18T01:00:03Z"},
        status({
            "inputOther": 0,
            "inputCacheCreation": 0,
            "inputCacheRead": 0,
            "output": 0,
        }, "2026-08-18T01:00:04Z"),
        {"type": "turn.ended", "turnId": 1, "reason": "completed",
         "time": "2026-08-18T01:00:05Z"},
        {"message": {"type": "Unrelated"}},
    ])
    assert facts["complete"] is True
    assert facts["n_input_tokens"] == 21_325
    assert facts["n_cache_tokens"] == 19_230
    assert facts["n_output_tokens"] == 31
    assert facts["cache_creation_tokens"] == 121
    assert facts["request_count"] == 3
    assert facts["session_usage_model_request_count"] == 3
    assert facts["completed_turn_count"] == 1
    assert facts["turn_prompt_count"] == 1
    assert facts["request_ledger_duplicate_count"] == 0
    assert facts["request_usage_complete"] is True
    assert sum(e["n_input_tokens"] for e in facts["token_usage_events"]) == 21_325

    incomplete = namespace["_kimi_usage_facts"]([
        {"type": "metadata", "protocol_version": "1"},
        {"type": "turn.prompt", "time": "2026-08-18T00:59:59Z"},
        {"type": "llm.request", "model": "k3",
         "turnStep": "1.1", "time": "2026-08-18T00:59:59.500Z"},
        status({
            "inputOther": 1_964,
            "inputCacheCreation": 0,
            "inputCacheRead": 19_200,
            "output": 27,
        }, "2026-08-18T01:00:00Z"),
    ])
    assert incomplete["complete"] is False
    assert incomplete["request_usage_complete"] is False
    assert incomplete["request_usage_observed"] is True
    assert incomplete["usage_evidence_tier"] == "observed_unreconciled"
    assert incomplete["n_input_tokens"] == 21_164
    assert incomplete["n_cache_tokens"] == 19_200
    assert incomplete["n_output_tokens"] == 27
    assert len(incomplete["token_usage_events"]) == 1


def test_kimi_wire_usage_fails_closed_on_replay_conflict_and_bad_terminal() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    module = ast.parse(source)
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_usage_instant", "_kimi_usage_facts"}
    ]
    namespace = {
        "Any": Any, "datetime": datetime, "timezone": timezone,
        "deque": deque,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "pier_kimi.py", "exec"),
         namespace)
    usage = {
        "inputOther": 10, "inputCacheCreation": 1,
        "inputCacheRead": 20, "output": 3,
    }

    def one_turn(*, reason: str = "completed") -> list[dict]:
        return [
            {"type": "metadata", "protocol_version": "1"},
            {"type": "turn.prompt", "time": 1_787_000_000_000},
            {"type": "llm.request", "model": "k3",
             "turnStep": "1.1", "time": 1_787_000_000_100},
            {"type": "usage.record", "usageScope": "turn",
             "model": "kimi-code/k3", "time": 1_787_000_000_200,
             "usage": dict(usage)},
            {"type": "turn.ended", "turnId": 1, "reason": reason,
             "time": 1_787_000_000_300},
        ]

    facts = namespace["_kimi_usage_facts"](one_turn())
    assert facts["complete"] is True

    replayed = one_turn()
    replayed.insert(4, dict(replayed[3]))
    replayed_facts = namespace["_kimi_usage_facts"](replayed)
    assert replayed_facts["complete"] is False
    assert replayed_facts["request_usage_observed"] is False
    assert replayed_facts["request_ledger_duplicate_count"] == 1
    assert replayed_facts["token_usage_events"] == []

    conflicting = one_turn()
    conflict = json.loads(json.dumps(conflicting[3]))
    conflict["usage"]["output"] += 1
    conflicting.insert(4, conflict)
    conflict_facts = namespace["_kimi_usage_facts"](conflicting)
    assert conflict_facts["complete"] is False
    assert conflict_facts["request_usage_observed"] is False

    limited = namespace["_kimi_usage_facts"](one_turn(reason="resource_limit"))
    assert limited["complete"] is False
    assert limited["request_usage_observed"] is True
    assert limited["usage_incomplete_reason"] == "turn_completion_ledger_mismatch"

    cross_session = one_turn()
    cross_session.insert(1, {"type": "metadata", "protocol_version": "1"})
    cross_session_facts = namespace["_kimi_usage_facts"](cross_session)
    assert cross_session_facts["complete"] is False
    assert cross_session_facts["request_usage_observed"] is False
    assert cross_session_facts["usage_incomplete_reason"] == (
        "request_ledger_unavailable_or_invalid"
    )


def test_kimi_wire_groups_retries_and_accepts_multiturn_equal_millis() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    module = ast.parse(source)
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_usage_instant", "_kimi_usage_facts"}
    ]
    namespace = {
        "Any": Any, "datetime": datetime, "timezone": timezone,
        "deque": deque,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "pier_kimi.py", "exec"),
         namespace)

    def request(step: str, at: int, **extra) -> dict:
        return {"type": "llm.request", "model": "k3", "turnStep": step,
                "time": at, **extra}

    def usage(at: int, output: int) -> dict:
        return {
            "type": "usage.record", "usageScope": "turn",
            "model": "kimi-code/k3", "time": at,
            "usage": {"inputOther": 10, "inputCacheCreation": 0,
                      "inputCacheRead": 20, "output": output},
        }

    def retry(attempt: int, *, error_name: str = "APIProviderRateLimitError",
              status_code: int | None = 429) -> dict:
        return {
            "role": "meta", "type": "turn.step.retrying",
            "failed_attempt": attempt, "next_attempt": attempt + 1,
            "max_attempts": 10, "delay_ms": 500.0,
            "error_name": error_name, "error_message": "retryable",
            "status_code": status_code,
        }

    records = [
        {"type": "metadata", "protocol_version": "1"},
        {"type": "turn.prompt", "time": 100},
        request("0.1", 110),
        request("0.2", 120),
        usage(200, 3),
        request("0.3", 200),
        usage(200, 4),
        {"type": "turn.ended", "turnId": 0, "reason": "completed", "time": 210},
        {"type": "turn.prompt", "time": 300},
        request("1.1", 310),
        usage(400, 5),
        {"type": "turn.ended", "turnId": 1, "reason": "completed", "time": 410},
    ]
    facts = namespace["_kimi_usage_facts"](records, [retry(1)])
    assert facts["complete"] is True
    assert facts["request_count"] == 3
    assert facts["session_usage_model_request_count"] == 3
    assert facts["session_usage_request_attempt_count"] == 4
    assert facts["session_usage_request_retry_count"] == 1
    assert facts["completed_turn_count"] == 2
    assert facts["turn_prompt_count"] == 2

    replay = list(records)
    replay.insert(5, request("0.2", 201))
    replayed = namespace["_kimi_usage_facts"](replay, [retry(1)])
    assert replayed["complete"] is False
    assert replayed["request_usage_observed"] is False
    assert replayed["request_ledger_duplicate_count"] == 1


def test_kimi_replays_real_cliffy_429_retry_fixture() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    module = ast.parse(source)
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_usage_instant", "_kimi_usage_facts"}
    ]
    namespace = {
        "Any": Any, "datetime": datetime, "timezone": timezone,
        "deque": deque,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "pier_kimi.py", "exec"),
         namespace)
    fixture = json.loads(
        (Path(__file__).with_name("fixtures")
         / "kimi_cliffy_429_reconciliation.json").read_text()
    )
    expected = fixture["expected"]
    facts = namespace["_kimi_usage_facts"](
        fixture["wire_records"], fixture["retry_records"],
    )
    assert facts["complete"] is True
    assert facts["usage_evidence_tier"] == "complete_reconciled"
    assert facts["request_count"] == expected["request_count"]
    assert facts["session_usage_model_request_count"] == expected["request_count"]
    assert facts["session_usage_request_attempt_count"] == expected["attempt_count"]
    assert facts["session_usage_request_retry_count"] == expected["retry_count"]
    assert facts["n_input_tokens"] == expected["n_input_tokens"]
    assert facts["n_cache_tokens"] == expected["n_cache_tokens"]
    assert facts["n_output_tokens"] == expected["n_output_tokens"]
    assert facts["request_ledger_source"] == (
        "kimi-code-0.36.1-main-wire-retry-v2"
    )


def test_kimi_retry_reconciliation_whitelists_connection_errors() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    module = ast.parse(source)
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_usage_instant", "_kimi_usage_facts"}
    ]
    namespace = {
        "Any": Any, "datetime": datetime, "timezone": timezone,
        "deque": deque,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "pier_kimi.py", "exec"),
         namespace)
    wire = [
        {"type": "metadata", "protocol_version": "1.5"},
        {"type": "turn.prompt", "time": 100},
        {"type": "llm.request", "model": "k3", "turnStep": "0.1",
         "time": 110},
        {"type": "llm.request", "model": "k3", "turnStep": "0.2",
         "time": 120},
        {"type": "usage.record", "usageScope": "turn",
         "model": "kimi-code/k3", "time": 130,
         "usage": {"inputOther": 10, "inputCacheCreation": 0,
                   "inputCacheRead": 20, "output": 3}},
        {"type": "turn.ended", "turnId": 0, "reason": "completed",
         "time": 140},
    ]
    connection_retry = [{
        "role": "meta", "type": "turn.step.retrying",
        "failed_attempt": 1, "next_attempt": 2, "max_attempts": 10,
        "delay_ms": 500, "error_name": "APIConnectionError",
        "error_message": "connection reset",
    }]
    facts = namespace["_kimi_usage_facts"](wire, connection_retry)
    assert facts["complete"] is True
    assert facts["request_count"] == 1
    assert facts["session_usage_request_attempt_count"] == 2
    assert facts["session_usage_request_retry_count"] == 1


def test_kimi_retry_reconciliation_rejects_count_coincidence_and_forgery() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    module = ast.parse(source)
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_usage_instant", "_kimi_usage_facts"}
    ]
    namespace = {
        "Any": Any, "datetime": datetime, "timezone": timezone,
        "deque": deque,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "pier_kimi.py", "exec"),
         namespace)

    def retry(attempt: int, *, error_name: str = "APIProviderRateLimitError",
              role: str = "meta", extra: dict | None = None) -> dict:
        value = {
            "role": role, "type": "turn.step.retrying",
            "failed_attempt": attempt, "next_attempt": attempt + 1,
            "max_attempts": 10, "delay_ms": 500,
            "error_name": error_name, "error_message": "retryable",
            "status_code": 429,
        }
        value.update(extra or {})
        return value

    wire = [
        {"type": "metadata", "protocol_version": "1.5"},
        {"type": "turn.prompt", "time": 100},
        {"type": "llm.request", "model": "k3", "turnStep": "0.1",
         "time": 110},
        {"type": "llm.request", "model": "k3", "turnStep": "0.2",
         "time": 120},
        {"type": "llm.request", "model": "k3", "turnStep": "0.3",
         "time": 130},
        {"type": "usage.record", "usageScope": "turn",
         "model": "kimi-code/k3", "time": 140,
         "usage": {"inputOther": 10, "inputCacheCreation": 0,
                   "inputCacheRead": 20, "output": 3}},
        {"type": "turn.ended", "turnId": 0, "reason": "completed",
         "time": 150},
    ]
    facts = namespace["_kimi_usage_facts"]

    # Two retry events are not enough: they must form the one 1->2->3 group
    # implied by the three consecutive wire attempts.
    count_coincidence = facts(wire, [retry(1), retry(1)])
    assert count_coincidence["complete"] is False
    assert count_coincidence["usage_evidence_tier"] == "observed_unreconciled"

    assert facts(wire, [retry(1)])["complete"] is False
    assert facts(wire, [retry(1), retry(2, error_name="UnknownError")])[
        "complete"] is False
    assert facts(wire, [retry(1), retry(2, role="assistant")])[
        "complete"] is False
    assert facts(wire, [retry(1), retry(2, extra={"forged": True})])[
        "complete"] is False

    unfinished = wire[:-1]
    assert facts(unfinished, [retry(1), retry(2)])["complete"] is False

    successful_wire = [wire[0], wire[1], wire[4], wire[5], wire[6]]
    successful_wire[2] = dict(successful_wire[2], turnStep="0.1")
    assert facts(successful_wire, [retry(1)])["complete"] is False


def test_kimi_copies_only_the_main_agent_durable_wire() -> None:
    source = Path(providers.__file__).with_name("pier_kimi.py").read_text()
    assert "-path '*/agents/main/wire.jsonl'" in source


def _write_kimi_session(
    trial: Path, *, include_result: bool = True, malformed: bool = False,
) -> None:
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (agent / "trajectory.json").write_text(json.dumps({
        "session_id": "kimi-session-1",
        "agent": {"model_name": KIMI_MODEL},
    }))
    records = [
        {"role": "meta", "type": "system.version", "version": KIMI_CLI_VERSION},
        {"role": "meta", "type": "session.resume_hint",
         "session_id": "kimi-session-1"},
        {"role": "assistant", "tool_calls": [{
            "type": "function", "id": "call-1",
            "function": {
                "name": "Bash",
                "arguments": json.dumps({
                    "command": "curl https://example.com/data",
                }),
            },
        }]},
    ]
    if include_result:
        records.append({
            "role": "tool", "tool_call_id": "call-1",
            "content": "proxy denied",
        })
    text = "\n".join(json.dumps(record) for record in records) + "\n"
    if malformed:
        text += "{not-json}\n"
    (agent / "kimi-code.jsonl").write_text(text)


def test_kimi_tool_bundle_retains_calls_results_and_pairing(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    _write_kimi_session(trial)

    bundle = runner.build_kimi_trajectory_bundle(trial)

    assert bundle is not None
    assert bundle["schema_version"] == "dradar-kimi-trajectory-bundle-v1"
    assert bundle["complete"] is True
    session = bundle["sessions"][0]
    assert session["tool_call_count"] == 1
    assert session["tool_result_count"] == 1
    call = next(event for event in session["events"]
                if event["type"] == "tool_call")
    result = next(event for event in session["events"]
                  if event["type"] == "tool_result")
    assert call["payload"]["call_id"] == result["payload"]["call_id"] == "call-1"
    assert "https://example.com/data" in call["payload"]["arguments"]


@pytest.mark.parametrize(
    ("include_result", "malformed"),
    [(False, False), (True, True)],
)
def test_kimi_tool_bundle_keeps_partial_evidence_without_blocking_upload(
    tmp_path: Path, include_result: bool, malformed: bool,
) -> None:
    trial = tmp_path / "trial"
    _write_kimi_session(
        trial, include_result=include_result, malformed=malformed,
    )

    bundle = runner.build_kimi_trajectory_bundle(trial)

    assert bundle is not None
    assert bundle["complete"] is False
    assert bundle["sessions"][0]["events"]


def test_missing_kimi_session_log_has_no_bundle(tmp_path: Path) -> None:
    assert runner.build_kimi_trajectory_bundle(tmp_path) is None
