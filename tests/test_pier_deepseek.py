from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip("pier")

from pier.agents.installed.codex import Codex

from dradar import pier_codex, pier_deepseek
from dradar.pier_codex import DurableCodex
from dradar.pier_deepseek import DeepSeekCodex
from dradar.providers import deepseek_catalog_path


DEEPSEEK_SECRET = "sk-deepseek-secret-that-must-never-enter-a-checkpoint"
DEEPSEEK_ACCOUNT = "deepseek-account-identifier-that-is-private"


class RecordingCheckpoint:
    instances: list["RecordingCheckpoint"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.enabled = bool(kwargs["enabled"])
        self.sensitive_values = tuple(kwargs.get("sensitive_values", ()))
        logs_dir = Path(kwargs["logs_dir"])
        self.host_dir = logs_dir.parent / "checkpoint"
        checkpoint_path = kwargs.get("checkpoint_path")
        self.previous_dir = Path(checkpoint_path) if checkpoint_path else None
        self.previous: dict[str, object] | None = None
        self.manifest_path: Path | None = None
        self.root_thread_id: str | None = None
        self.finish_calls: list[dict[str, object]] = []
        self.finish_started: asyncio.Event | None = None
        self.finish_release: asyncio.Event | None = None
        self.finish_done: asyncio.Event | None = None
        type(self).instances.append(self)

    async def prepare_agent_environment(self, agent, environment, env):
        del agent, environment, env
        return None

    async def exec_root_maintenance(
        self, environment, command, *, timeout_sec=120,
    ):
        return await environment.exec(
            command=command,
            user="root",
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "BASH_ENV": "/dev/null"},
            cwd="/",
            timeout_sec=timeout_sec,
        )

    async def start(self, agent, environment, env):
        del agent, environment, env
        self.manifest_path = self.host_dir / "checkpoint.json"
        return self.root_thread_id

    async def finish(
        self,
        agent,
        environment,
        env,
        *,
        completed,
        failure,
        session_id=None,
    ) -> None:
        del agent, environment, env
        self.finish_calls.append(
            {
                "completed": completed,
                "failure": failure,
                "session_id": session_id,
            }
        )
        if self.finish_started is not None:
            self.finish_started.set()
        try:
            if self.finish_release is not None:
                await self.finish_release.wait()
        finally:
            if self.finish_done is not None:
                self.finish_done.set()

    def _event(self, event: str, **detail: object) -> None:
        del event, detail


@pytest.fixture
def recording_checkpoint(monkeypatch: pytest.MonkeyPatch):
    RecordingCheckpoint.instances = []
    monkeypatch.setattr(pier_codex, "DurableCheckpoint", RecordingCheckpoint)
    yield RecordingCheckpoint
    RecordingCheckpoint.instances = []


def _auth_file(tmp_path: Path) -> Path:
    auth = tmp_path / "deepseek-auth.json"
    auth.write_text(
        json.dumps(
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": DEEPSEEK_SECRET,
                "account": {"id": DEEPSEEK_ACCOUNT},
            }
        ),
        encoding="utf-8",
    )
    auth.chmod(0o600)
    return auth


def _agent(
    tmp_path: Path,
    recording_checkpoint,
    *,
    checkpoint_enabled: bool = True,
) -> DeepSeekCodex:
    del recording_checkpoint
    logs_dir = tmp_path / "trial" / "agent"
    logs_dir.mkdir(parents=True, mode=0o700)
    return DeepSeekCodex(
        logs_dir=logs_dir,
        model_name="deepseek-v4-flash",
        version="0.149.0",
        model_catalog_json_file=str(deepseek_catalog_path()),
        checkpoint_enabled=checkpoint_enabled,
        checkpoint_assignment_id="assignment-deepseek-checkpoint-1",
        checkpoint_task_id="pompeii-task-1",
        checkpoint_effort="max",
        checkpoint_resume_generation=0,
        extra_env={"CODEX_AUTH_JSON_PATH": str(_auth_file(tmp_path))},
    )


def test_constructor_uses_host_private_sibling_and_scans_deepseek_auth(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    agent = _agent(tmp_path, recording_checkpoint)
    checkpoint = recording_checkpoint.instances[-1]

    assert isinstance(agent, DurableCodex)
    assert checkpoint.host_dir == tmp_path / "trial" / "checkpoint"
    assert agent._checkpoint_host_dir == tmp_path / "trial" / "checkpoint"
    assert checkpoint.kwargs["provider"] == "deepseek"
    assert checkpoint.kwargs["harness"] == "codex"
    assert checkpoint.kwargs["agent_version"] == "0.149.0"
    assert {DEEPSEEK_SECRET, DEEPSEEK_ACCOUNT} <= set(
        checkpoint.kwargs["sensitive_values"]
    )
    state_paths = checkpoint.kwargs["state_paths"]
    assert [(item.name, item.remote_path) for item in state_paths] == [
        ("sessions", "/tmp/codex-home/sessions"),
    ]
    assert agent.network_allowlist().domains == ["api.deepseek.com"]
    assert "/logs/agent/checkpoint" not in inspect.getsource(pier_deepseek)


def test_catalog_is_verified_before_unchanged_benchmark_instruction(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Environment:
        default_user = "benchmark-agent"

        def __init__(self) -> None:
            self.uploads: list[tuple[Path, str]] = []
            self.root_commands: list[str] = []

        async def upload_file(self, source: Path, target: str) -> None:
            self.uploads.append((Path(source), target))

        async def exec(self, **kwargs):
            self.root_commands.append(str(kwargs["command"]))
            return type("Result", (), {"return_code": 0})()

    async def scenario() -> None:
        agent = _agent(tmp_path, recording_checkpoint)
        environment = Environment()
        agent_commands: list[str] = []
        delegated: list[str] = []

        async def exec_as_agent(_environment, command, **_kwargs):
            agent_commands.append(command)

        async def durable_run(self, instruction, _environment, _context):
            del self
            delegated.append(instruction)

        monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
        monkeypatch.setattr(DurableCodex, "run", durable_run)
        instruction = "BENCHMARK-PROMPT::keep this byte-for-byte unchanged"

        await agent.run(instruction, environment, object())

        assert delegated == [instruction]
        assert environment.uploads == [
            (deepseek_catalog_path(), "/tmp/codex-home/models.json")
        ]
        assert agent_commands[0] == "mkdir -p /tmp/codex-home"
        assert "sha256sum /tmp/codex-home/models.json" in agent_commands[1]
        assert environment.root_commands == [
            "chown benchmark-agent /tmp/codex-home/models.json"
        ]

    asyncio.run(scenario())


def test_disabled_checkpoint_uses_upstream_root_for_catalog_handoff(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Environment:
        default_user = "benchmark-agent"

        async def upload_file(self, _source: Path, _target: str) -> None:
            return None

    async def scenario() -> None:
        agent = _agent(
            tmp_path, recording_checkpoint, checkpoint_enabled=False,
        )
        checkpoint = recording_checkpoint.instances[-1]
        root_commands: list[str] = []

        async def forbidden_checkpoint_root(*_args, **_kwargs):
            raise AssertionError("disabled rollout used checkpoint root helper")

        async def exec_as_root(_environment, command, **_kwargs):
            root_commands.append(command)

        async def exec_as_agent(_environment, command, **_kwargs):
            del command
            return None

        async def durable_run(self, _instruction, _environment, _context):
            del self

        monkeypatch.setattr(
            checkpoint, "exec_root_maintenance", forbidden_checkpoint_root,
        )
        monkeypatch.setattr(agent, "exec_as_root", exec_as_root)
        monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
        monkeypatch.setattr(DurableCodex, "run", durable_run)

        await agent.run("unchanged", Environment(), object())

        assert root_commands == [
            "chown benchmark-agent /tmp/codex-home/models.json"
        ]

    asyncio.run(scenario())


def test_cancellation_during_finalizer_reaps_deepseek_checkpoint_before_cleanup(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Environment:
        default_user = None

        async def upload_file(self, _source: Path, _target: str) -> None:
            return None

    async def scenario() -> None:
        agent = _agent(tmp_path, recording_checkpoint)
        checkpoint = recording_checkpoint.instances[-1]
        checkpoint.finish_started = asyncio.Event()
        checkpoint.finish_release = asyncio.Event()
        checkpoint.finish_done = asyncio.Event()
        cleanup_started = asyncio.Event()

        async def exec_as_agent(_environment, command, **_kwargs):
            del command
            return None

        async def upstream_with_shielded_finalizer(
            self, instruction, environment, context,
        ) -> None:
            del instruction, context
            self._upstream_phase = "post-model"
            try:
                await asyncio.shield(
                    self._finish_checkpoint(
                        environment, {}, completed=True, failure=None,
                    )
                )
            except BaseException:
                pass
            cleanup_started.set()

        monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
        monkeypatch.setattr(Codex, "run", upstream_with_shielded_finalizer)
        running = asyncio.create_task(
            agent.run("unchanged", Environment(), object())
        )
        await asyncio.wait_for(checkpoint.finish_started.wait(), timeout=2)
        running.cancel()
        await asyncio.sleep(0)

        assert not running.done()
        assert not cleanup_started.is_set()
        assert not checkpoint.finish_done.is_set()
        checkpoint.finish_release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert checkpoint.finish_done.is_set()
        assert cleanup_started.is_set()
        assert agent._checkpoint_finalizer_task is not None
        assert agent._checkpoint_finalizer_task.done()

    asyncio.run(scenario())
