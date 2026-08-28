"""Google Antigravity integration is isolated, pinned, and token-audited."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
from pathlib import Path

import pytest

import dradar.providers as providers
import dradar.runner as runner
from dradar.providers import (
    ANTIGRAVITY_AGENT,
    ANTIGRAVITY_CAPABILITY,
    ANTIGRAVITY_CLI_VERSION,
    ANTIGRAVITY_MODEL,
    ANTIGRAVITY_PROVIDER,
    ANTIGRAVITY_RUNTIME_MODELS,
    antigravity_auth_error,
    antigravity_auth_path,
    antigravity_settings_payload,
    antigravity_subscription_session,
    mark_antigravity_ready,
    privatize_antigravity_home,
    write_antigravity_settings,
)
from dradar.runner import RunnerError


def _assignment(**overrides) -> dict:
    value = {
        "assignment_id": "agy-1",
        "task_id": "task-1",
        "agent": ANTIGRAVITY_AGENT,
        "provider": ANTIGRAVITY_PROVIDER,
        "model": ANTIGRAVITY_MODEL,
        "effort": "low",
        "agent_version": ANTIGRAVITY_CLI_VERSION,
        "est_minutes": 5,
    }
    value.update(overrides)
    return value


def _ready_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "dradar"
    monkeypatch.setenv("DRADAR_HOME", str(home))
    auth = antigravity_auth_path()
    auth.mkdir(parents=True)
    token = auth / "config" / "oauth-state.json"
    token.parent.mkdir(parents=True)
    token.write_text('{"refresh":"hidden"}', encoding="utf-8")
    write_antigravity_settings()
    mark_antigravity_ready()
    privatize_antigravity_home()
    return auth.resolve()


def _usage_helper():
    source = Path(providers.__file__).with_name("pier_antigravity.py").read_text()
    module = ast.parse(source)
    names = {"_nonnegative_int", "_usage_values", "_antigravity_usage_facts"}
    helpers = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"ANTIGRAVITY_MODEL": ANTIGRAVITY_MODEL}
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), "pier_antigravity.py", "exec"),
        namespace,
    )
    return namespace["_antigravity_usage_facts"]


def _model_line_pattern_helper():
    source = Path(providers.__file__).with_name("pier_antigravity.py").read_text()
    module = ast.parse(source)
    helper = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_model_line_pattern"
    )
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "pier_antigravity.py", "exec"),
        namespace,
    )
    return namespace["_model_line_pattern"]


def _usage(input_tokens: int, output_tokens: int, cache: int, thinking: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking,
        "cache_read_tokens": cache,
        "total_tokens": input_tokens + output_tokens,
    }


def test_isolated_oauth_home_requires_exact_sandbox_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _ready_home(tmp_path, monkeypatch)
    assert antigravity_auth_error() is None
    assert antigravity_settings_payload()["permissions"]["allow"] == ["command(*)"]
    deny = set(antigravity_settings_payload()["permissions"]["deny"])
    assert "read_file(/tmp/dradar-antigravity-user/.gemini)" in deny
    assert "unsandboxed(*)" in deny
    assert "read_url(*)" in deny

    settings = auth / "antigravity-cli" / "settings.json"
    payload = json.loads(settings.read_text())
    payload["enableTerminalSandbox"] = False
    settings.write_text(json.dumps(payload), encoding="utf-8")
    if os.name != "nt":
        settings.chmod(0o600)
    assert "safe policy" in (antigravity_auth_error() or "")


def test_oauth_home_rejects_links_and_broad_secret_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _ready_home(tmp_path, monkeypatch)
    secret = auth / "config" / "oauth-state.json"
    if os.name != "nt":
        secret.chmod(0o644)
        assert "broadly accessible" in (antigravity_auth_error() or "")
        secret.chmod(0o600)
    link = auth / "config" / "linked-token"
    link.symlink_to(secret)
    assert "must not be a symlink" in (antigravity_auth_error() or "")


def test_home_hardening_preserves_only_managed_runtime_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "dradar"
    monkeypatch.setenv("DRADAR_HOME", str(home))
    runtime = (
        home / "providers" / "antigravity" / "runtime"
        / ANTIGRAVITY_CLI_VERSION / "aarch64"
    )
    runtime.mkdir(parents=True)
    executable = runtime / "antigravity"
    executable.write_bytes(b"reviewed-binary")
    proof = runtime / ".archive.sha512"
    proof.write_text("reviewed-proof", encoding="utf-8")
    auth_named_file = antigravity_auth_path() / "antigravity"
    auth_named_file.parent.mkdir(parents=True)
    auth_named_file.write_text("secret", encoding="utf-8")
    log_dir = antigravity_auth_path() / "antigravity-cli" / "log"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "cli-20260828_000000.log"
    log_file.write_text("official log", encoding="utf-8")
    cli_log = log_dir.parent / "cli.log"
    cli_log.symlink_to(Path("log") / log_file.name)

    privatize_antigravity_home()

    if os.name != "nt":
        assert executable.stat().st_mode & 0o777 == 0o700
        assert proof.stat().st_mode & 0o777 == 0o600
        assert auth_named_file.stat().st_mode & 0o777 == 0o600
        assert log_file.stat().st_mode & 0o777 == 0o600
    assert not cli_log.exists()


def test_subscription_session_exposes_only_canonical_gemini_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _ready_home(tmp_path, monkeypatch)
    with antigravity_subscription_session(tmp_path / "work") as shared:
        assert shared.resolve() == auth
        assert shared.name == ".gemini"


@pytest.mark.parametrize("raises", [False, True])
def test_subscription_session_restores_policy_after_every_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raises: bool,
) -> None:
    auth = _ready_home(tmp_path, monkeypatch)
    settings = auth / "antigravity-cli" / "settings.json"

    def run() -> None:
        with antigravity_subscription_session(tmp_path / "work"):
            payload = json.loads(settings.read_text(encoding="utf-8"))
            payload.pop("allowNonWorkspaceAccess")
            settings.write_text(json.dumps(payload), encoding="utf-8")
            if raises:
                raise RuntimeError("trial interrupted")

    if raises:
        with pytest.raises(RuntimeError, match="trial interrupted"):
            run()
    else:
        run()

    assert json.loads(settings.read_text(encoding="utf-8")) == (
        antigravity_settings_payload()
    )
    assert antigravity_auth_error() is None


def test_subscription_session_never_follows_a_runtime_created_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _ready_home(tmp_path, monkeypatch)
    settings_parent = auth / "antigravity-cli"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "settings.json"
    sentinel.write_text("outside must stay untouched", encoding="utf-8")

    with pytest.raises(ValueError, match="contains a symlink"):
        with antigravity_subscription_session(tmp_path / "work"):
            shutil.rmtree(settings_parent)
            settings_parent.symlink_to(outside, target_is_directory=True)

    assert sentinel.read_text(encoding="utf-8") == "outside must stay untouched"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_all_three_efforts_build_the_same_public_card_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: str,
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/pier")
    tasks = tmp_path / "tasks"
    (tasks / "task-1").mkdir(parents=True)
    auth = tmp_path / "providers" / "antigravity" / ".gemini"
    auth.mkdir(parents=True)
    if os.name != "nt":
        auth.chmod(0o700)
    cmd = runner.build_pier_command(
        _assignment(effort=effort), tasks, tmp_path / "jobs", "job", tmp_path,
        provider_auth_path=auth.resolve(),
    )
    assert runner.ANTIGRAVITY_AGENT_IMPORT_PATH in cmd
    assert runner.SHARED_OAUTH_ENV_IMPORT_PATH in cmd
    assert cmd[cmd.index("--model") + 1] == ANTIGRAVITY_MODEL
    assert f"reasoning_effort={effort}" in cmd
    assert f"auth_home_dir={auth.resolve()}" in cmd
    assert "shared_oauth=true" in cmd
    assert f"version={ANTIGRAVITY_CLI_VERSION}" in cmd
    mounts = cmd[cmd.index("--ek") + 1]
    assert "/tmp/dradar-antigravity-user/.gemini" in mounts
    assert ANTIGRAVITY_RUNTIME_MODELS[effort] in Path(
        providers.__file__
    ).with_name("pier_antigravity.py").read_text()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"provider": "google-api"}, "explicitly use provider"),
        ({"model": "gemini-other"}, "unsupported Antigravity model"),
        ({"effort": "max"}, "effort must be low, medium, or high"),
        ({"agent_version": "9.9.9"}, "pinned to CLI"),
    ],
)
def test_unverified_assignments_fail_before_a_paid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    overrides: dict, message: str,
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/pier")
    assignment = _assignment(**overrides)
    tasks = tmp_path / "tasks"
    (tasks / assignment["task_id"]).mkdir(parents=True)
    auth = tmp_path / "providers" / "antigravity" / ".gemini"
    auth.mkdir(parents=True)
    with pytest.raises(RunnerError, match=message):
        runner.build_pier_command(
            assignment, tasks, tmp_path / "jobs", "job", tmp_path,
            provider_auth_path=auth.resolve(),
        )


def test_adapter_never_uses_dangerous_permissions_or_scratch_workspace() -> None:
    source = Path(providers.__file__).with_name("pier_antigravity.py").read_text()
    assert '"--new-project"' in source
    assert '"--sandbox"' in source
    assert "--dangerously-skip-permissions" not in source
    assert 'init.get("cwd") == "/app"' in source
    assert 'init.get("permission_mode") != "always-proceed"' in source
    assert "sha512sum --check --strict" in source
    assert "storage.googleapis.com" in source
    assert '"www.googleapis.com"' in source
    assert '"lh3.googleusercontent.com"' in source
    assert "*.googleapis.com" not in source
    assert "*.googleusercontent.com" not in source


def test_runtime_model_preflight_accepts_the_official_tabular_output() -> None:
    helper = _model_line_pattern_helper()
    slug = ANTIGRAVITY_RUNTIME_MODELS["low"]
    assert helper(slug) == "^" + re.escape(slug) + r"([[:space:]]|$)"
    source = Path(providers.__file__).with_name("pier_antigravity.py").read_text()
    assert "grep -Eq" in source
    assert "grep -Fqx {shlex.quote(slug)}" not in source


def test_official_step_ledger_reconciles_without_double_counting_thinking() -> None:
    helper = _usage_helper()
    runtime = ANTIGRAVITY_RUNTIME_MODELS["low"]
    first = _usage(100, 20, 60, 15)
    checkpoint = _usage(10, 2, 0, 0)
    terminal = _usage(110, 22, 60, 15)
    events = [
        {"event": "init", "init": {
            "model": runtime, "cwd": "/app", "permission_mode": "request-review",
        }},
        {"event": "step_update", "step_update": {
            "step_index": 1, "step_type": "agent_response", "state": "DONE",
            "usage": first,
        }},
        {"event": "step_update", "step_update": {
            "step_index": 2, "step_type": "checkpoint", "state": "DONE",
            "usage": checkpoint,
        }},
        {"event": "result", "result": {
            "status": "SUCCESS", "num_turns": 1, "usage": terminal,
        }},
    ]
    facts = helper(events, expected_runtime_model=runtime)
    assert facts["complete"] is True
    assert facts["request_count"] == 2
    assert facts["n_input_tokens"] == 110
    assert facts["n_cache_tokens"] == 60
    assert facts["n_output_tokens"] == 22
    assert facts["thinking_tokens"] == 15
    assert sum(item["n_output_tokens"] for item in facts["token_usage_events"]) == 22


def test_conflicting_duplicate_or_wrong_runtime_never_becomes_complete() -> None:
    helper = _usage_helper()
    runtime = ANTIGRAVITY_RUNTIME_MODELS["medium"]
    events = [
        {"event": "init", "init": {
            "model": runtime, "cwd": "/app", "permission_mode": "request-review",
        }},
        {"event": "step_update", "step_update": {
            "step_index": 1, "step_type": "agent_response", "state": "DONE",
            "usage": _usage(10, 2, 0, 1),
        }},
        {"event": "step_update", "step_update": {
            "step_index": 1, "step_type": "agent_response", "state": "DONE",
            "usage": _usage(11, 2, 0, 1),
        }},
        {"event": "result", "result": {
            "status": "SUCCESS", "num_turns": 1, "usage": _usage(10, 2, 0, 1),
        }},
    ]
    assert helper(events, expected_runtime_model=runtime)["complete"] is False
    events[0]["init"]["model"] = "gemini-3.7-flash-low"
    assert helper(events[:2] + events[3:], expected_runtime_model=runtime)["complete"] is False


def test_provider_failure_is_additive_and_does_not_strip_other_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "direct-google-key-must-not-enter-agy")
    monkeypatch.setenv("XAI_API_KEY", "unrelated-grok-key")
    agy_env = runner._pier_process_env(_assignment())
    assert "GEMINI_API_KEY" not in agy_env
    assert agy_env["XAI_API_KEY"] == "unrelated-grok-key"

    grok_env = runner._pier_process_env({"agent": providers.GROK_AGENT})
    assert "XAI_API_KEY" not in grok_env
    assert grok_env["GEMINI_API_KEY"] == "direct-google-key-must-not-enter-agy"


def test_capability_name_is_additive_and_refill_alias_is_canonical() -> None:
    assert ANTIGRAVITY_CAPABILITY.startswith("antigravity-gemini-3.7-flash-")
    assert providers.normalize_refill_harness("agy") == ANTIGRAVITY_AGENT
    assert providers.validate_refill_scope(
        "antigravity", ANTIGRAVITY_MODEL, "high",
    ) == (ANTIGRAVITY_AGENT, ANTIGRAVITY_MODEL, "high")
