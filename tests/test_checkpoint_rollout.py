from pathlib import Path

import pytest

from dradar import runner


def _private_file(path: Path, payload: str = "ready") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return path


def _assignment(agent: str, **overrides: object) -> dict:
    value = {
        "assignment_id": f"assignment-{agent}-fresh-1",
        "task_id": "task-1",
        "agent": agent,
        "model": "gpt-5.4",
        "effort": "high",
        "agent_version": "0.145.0",
        "_durable_checkpoint_enabled": False,
    }
    value.update(overrides)
    return value


def _agent_values(command: list[str]) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--ak"
    ]


def test_public_durable_checkpoint_rollout_is_hard_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DRADAR_ENABLE_CHECKPOINT", "1")
    assert runner.DURABLE_CHECKPOINT_ROLLOUT_ENABLED is False
    assert runner.durable_checkpoint_rollout_enabled() is False


def test_disabled_rollout_uses_stock_codex_without_private_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "_resolve_user_tool", lambda *_args, **_kwargs: "/usr/bin/pier",
    )
    tasks = tmp_path / "tasks"
    (tasks / "task-1").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    auth = _private_file(tmp_path / "codex-auth.json", "{}")
    monkeypatch.setattr(runner, "codex_auth_path", lambda: auth)

    command = runner.build_pier_command(
        _assignment("codex"), tasks, tmp_path / "jobs", "job", home,
    )

    assert command[command.index("--agent") + 1] == "codex"
    assert "--agent-import-path" not in command
    assert not (home / runner.CODEX_AGENT_MODULE_FILENAME).exists()
    assert not (home / runner.CHECKPOINT_MODULE_FILENAME).exists()


@pytest.mark.parametrize(
    ("assignment", "credential_kind"),
    [
        (_assignment("codex"), "codex"),
        (
            _assignment(
                "codex",
                provider=runner.DEEPSEEK_PROVIDER,
                model="deepseek-v4-pro",
                effort="max",
                agent_version=runner.DEEPSEEK_MIN_CODEX_VERSION,
            ),
            "deepseek",
        ),
        (
            _assignment(
                runner.DSH_AGENT,
                provider=runner.DEEPSEEK_PROVIDER,
                model="dsh-deepseek-v4-pro",
                agent_version=runner.DSH_VERSION,
            ),
            "dsh",
        ),
        (
            _assignment(
                runner.KIMI_AGENT,
                provider=runner.KIMI_PROVIDER,
                model=runner.KIMI_MODEL,
                agent_version=runner.KIMI_CLI_VERSION,
            ),
            "kimi",
        ),
        (
            _assignment(
                runner.ZCODE_AGENT,
                provider=runner.ZCODE_PROVIDER,
                model=runner.ZCODE_MODEL,
                agent_version=runner.ZCODE_CLI_VERSION,
            ),
            "zcode",
        ),
    ],
)
def test_fresh_harness_commands_omit_all_checkpoint_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assignment: dict,
    credential_kind: str,
) -> None:
    monkeypatch.setattr(
        runner, "_resolve_user_tool",
        lambda name, **_kwargs: "/usr/bin/uvx" if name == "uvx" else "/usr/bin/pier",
    )
    tasks = tmp_path / "tasks"
    (tasks / "task-1").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    credential = _private_file(tmp_path / "provider-credential")
    if credential_kind == "kimi":
        monkeypatch.setenv(
            "DRADAR_KIMI_HOME", str(tmp_path / "providers" / "kimi")
        )
        credential = _private_file(
            tmp_path / "providers" / "kimi" / "credentials" / "kimi-code.json",
            '{"access_token":"access","refresh_token":"refresh"}',
        )
    cli = _private_file(tmp_path / "provider-cli")
    codex_auth = _private_file(tmp_path / "codex-auth.json", "{}")
    monkeypatch.setattr(runner, "codex_auth_path", lambda: codex_auth)
    provider_kwargs = {}
    if credential_kind != "codex":
        provider_kwargs["provider_auth_path"] = credential
    if credential_kind in {"kimi", "zcode"}:
        provider_kwargs["provider_cli_path"] = cli

    command = runner.build_pier_command(
        assignment,
        tasks,
        tmp_path / "jobs",
        "job",
        home,
        **provider_kwargs,
    )

    assert not any(
        value.startswith("checkpoint_") for value in _agent_values(command)
    )


def test_disabled_rollout_refuses_checkpoint_resume_before_command_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_resolve_user_tool", lambda *_a, **_k: "/usr/bin/pier")
    tasks = tmp_path / "tasks"
    (tasks / "task-1").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    auth = _private_file(tmp_path / "codex-auth.json", "{}")
    monkeypatch.setattr(runner, "codex_auth_path", lambda: auth)

    with pytest.raises(runner.RunnerError, match="temporarily unavailable"):
        runner.build_pier_command(
            _assignment("codex"),
            tasks,
            tmp_path / "jobs",
            "job",
            home,
            resume_checkpoint=tmp_path / "saved-checkpoint",
        )


def test_run_trial_wires_public_rollout_gate_into_fresh_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class CommandObserved(RuntimeError):
        pass

    def observe_build(assignment, *_args, **_kwargs):
        captured.update(assignment)
        raise CommandObserved

    monkeypatch.setattr(
        runner, "resolve_latest_codex_cli_version", lambda *_a, **_k: "0.145.0",
    )
    monkeypatch.setattr(
        runner.egress, "prepare_egress_proxy_runtime", lambda **_kwargs: {},
    )
    monkeypatch.setattr(runner, "build_pier_command", observe_build)

    with pytest.raises(CommandObserved):
        runner.run_trial(
            _assignment("codex"), tmp_path / "tasks", tmp_path / "work",
        )

    assert captured["_durable_checkpoint_enabled"] is False


def test_disabled_rollout_does_not_inject_codex_adapter_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class EnvironmentObserved(RuntimeError):
        pass

    def observe_environment(_assignment, **kwargs):
        captured.update(kwargs)
        raise EnvironmentObserved

    monkeypatch.setattr(
        runner, "resolve_latest_codex_cli_version", lambda *_a, **_k: "0.145.0",
    )
    monkeypatch.setattr(
        runner.egress, "prepare_egress_proxy_runtime", lambda **_kwargs: {},
    )
    monkeypatch.setattr(runner, "build_pier_command", lambda *_a, **_k: ["pier"])
    monkeypatch.setattr(runner, "_pier_process_env", observe_environment)

    with pytest.raises(EnvironmentObserved):
        runner.run_trial(
            _assignment("codex"), tmp_path / "tasks", tmp_path / "work",
        )

    assert captured["codex_module_dir"] is None
