from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pier")

from dradar import kimi_recovery, pier_checkpoint

# The packaged Kimi adapter is materialized as a top-level Pier module in
# production. Reproduce that import boundary in source-tree tests.
sys.modules.setdefault("_dradar_pier_checkpoint", pier_checkpoint)
sys.modules.setdefault("_dradar_kimi_recovery", kimi_recovery)

from dradar import pier_kimi, pier_zcode


class Environment:
    default_user = "benchmark-agent"

    def __init__(self) -> None:
        self.uploads: list[tuple[Path, str]] = []

    async def upload_file(self, source: Path, target: str) -> None:
        self.uploads.append((Path(source), target))


def _result() -> SimpleNamespace:
    return SimpleNamespace(return_code=0, stdout="", stderr="")


def test_kimi_disabled_checkpoint_uses_normal_root_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True, mode=0o700)
    auth = tmp_path / "kimi-code.json"
    auth.write_text(json.dumps({
        "access_token": "access-token-value",
        "refresh_token": "refresh-token-value",
    }), encoding="utf-8")
    auth.chmod(0o600)
    cli = tmp_path / "kimi"
    cli.write_text("binary", encoding="utf-8")
    agent = pier_kimi.KimiCode(
        logs_dir=logs,
        model_name="k3",
        version=pier_kimi.KIMI_CLI_VERSION,
        auth_json_file=str(auth),
        kimi_cli_file=str(cli),
        reasoning_effort="high",
        checkpoint_enabled=False,
    )
    root_commands: list[str] = []

    async def forbidden_checkpoint_root(*_args, **_kwargs):
        raise AssertionError("disabled Kimi used checkpoint root helper")

    async def exec_as_root(_environment, command, **_kwargs):
        root_commands.append(command)
        return _result()

    async def exec_as_agent(_environment, command, **_kwargs):
        del command
        return _result()

    async def no_model_run(**_kwargs):
        return 0, None

    monkeypatch.setattr(
        agent._checkpoint, "exec_root_maintenance", forbidden_checkpoint_root,
    )
    monkeypatch.setattr(agent, "exec_as_root", exec_as_root)
    monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
    monkeypatch.setattr(pier_kimi, "run_with_kimi_resume", no_model_run)

    asyncio.run(agent.run("unchanged", Environment(), object()))

    assert len(root_commands) == 1
    assert "chown benchmark-agent" in root_commands[0]
    assert "/tmp/dradar-kimi-home/credentials/kimi-code.json" in root_commands[0]


def test_zcode_disabled_checkpoint_uses_normal_root_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True, mode=0o700)
    key = tmp_path / "zcode-key"
    key.write_text("zcode-secret-value\n", encoding="utf-8")
    key.chmod(0o600)
    cli = tmp_path / "zcode.cjs"
    cli.write_text("pinned-zcode", encoding="utf-8")
    monkeypatch.setattr(
        pier_zcode,
        "ZCODE_CLI_SHA256",
        hashlib.sha256(cli.read_bytes()).hexdigest(),
    )
    agent = pier_zcode.ZCodeBigModel(
        logs_dir=logs,
        model_name="glm-5.3",
        version=pier_zcode.ZCODE_CLI_VERSION,
        api_key_file=str(key),
        zcode_cli_file=str(cli),
        reasoning_effort="high",
        session_timeout_sec=300,
        checkpoint_enabled=False,
    )
    root_commands: list[str] = []

    async def forbidden_checkpoint_root(*_args, **_kwargs):
        raise AssertionError("disabled ZCode used checkpoint root helper")

    async def exec_as_root(_environment, command, **_kwargs):
        root_commands.append(command)
        return _result()

    async def exec_as_agent(_environment, command, **_kwargs):
        del command
        return _result()

    monkeypatch.setattr(
        agent._checkpoint, "exec_root_maintenance", forbidden_checkpoint_root,
    )
    monkeypatch.setattr(agent, "exec_as_root", exec_as_root)
    monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)

    asyncio.run(agent.run("unchanged", Environment(), object()))

    assert len(root_commands) == 1
    assert "chown benchmark-agent" in root_commands[0]
    assert "/tmp/dradar-zcode-bin/zcode.cjs" in root_commands[0]
