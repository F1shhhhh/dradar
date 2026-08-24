from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pier")

from pier.agents.installed.codex import Codex
from pier.agents.installed.base import NonZeroAgentExitCodeError

from dradar import pier_codex
from dradar.pier_codex import DurableCodex


ROOT_THREAD_ID = "12345678-1234-4abc-8def-123456789abc"
CHILD_THREAD_ID = "ffffffff-ffff-4fff-8fff-ffffffffffff"
GENERATION = "a" * 32


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
        self.events: list[tuple[str, dict[str, object]]] = []
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
    ):
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
        self.events.append((event, detail))


@pytest.fixture
def recording_checkpoint(monkeypatch: pytest.MonkeyPatch):
    RecordingCheckpoint.instances = []
    monkeypatch.setattr(pier_codex, "DurableCheckpoint", RecordingCheckpoint)
    yield RecordingCheckpoint
    RecordingCheckpoint.instances = []


def _auth_file(tmp_path: Path) -> tuple[Path, set[str]]:
    values = {
        "access-token-that-must-never-enter-a-checkpoint",
        "refresh-token-that-must-never-enter-a-checkpoint",
        "account-identifier-that-is-also-private",
        "2026-08-23T00:00:00Z",
    }
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": next(value for value in values if value.startswith("access")),
                    "refresh_token": next(value for value in values if value.startswith("refresh")),
                    "account": next(value for value in values if value.startswith("account")),
                },
                "last_refresh": next(value for value in values if value.startswith("2026")),
            }
        ),
        encoding="utf-8",
    )
    auth.chmod(0o600)
    return auth, values


def _agent(
    tmp_path: Path,
    recording_checkpoint,
    *,
    checkpoint_path: Path | None = None,
) -> DurableCodex:
    del recording_checkpoint
    logs_dir = tmp_path / "trial" / "agent"
    logs_dir.mkdir(parents=True, mode=0o700)
    auth, _values = _auth_file(tmp_path)
    return DurableCodex(
        logs_dir=logs_dir,
        model_name="openai/gpt-5.4",
        version="0.118.0",
        checkpoint_enabled=True,
        checkpoint_assignment_id="assignment-codex-checkpoint-1",
        checkpoint_task_id="pompeii-task-1",
        checkpoint_effort="high",
        checkpoint_resume_generation=2,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        extra_env={"CODEX_AUTH_JSON_PATH": str(auth)},
    )


def _published_checkpoint(tmp_path: Path, *, with_sessions: bool = True) -> Path:
    root = tmp_path / "previous-checkpoint"
    payload = root / "snapshots" / GENERATION
    payload.mkdir(parents=True)
    (root / "current-generation").write_text(GENERATION + "\n", encoding="ascii")
    (payload / "session-id").write_text(ROOT_THREAD_ID + "\n", encoding="utf-8")
    if with_sessions:
        sessions = payload / "provider-state" / "sessions" / "2026" / "08" / "23"
        sessions.mkdir(parents=True)
        (sessions / f"rollout-2026-08-23T00-00-00-{ROOT_THREAD_ID}.jsonl").write_text(
            "{}\n", encoding="utf-8",
        )
    return root


def test_constructor_uses_sibling_publication_and_all_auth_strings(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    agent = _agent(tmp_path, recording_checkpoint)
    checkpoint = recording_checkpoint.instances[-1]
    _auth, expected_values = _auth_file(tmp_path)

    assert checkpoint.host_dir == tmp_path / "trial" / "checkpoint"
    assert agent._checkpoint_host_dir == tmp_path / "trial" / "checkpoint"
    assert checkpoint.kwargs["provider"] == "openai"
    assert checkpoint.kwargs["harness"] == "codex"
    assert checkpoint.kwargs["agent_version"] == "0.118.0"
    assert set(checkpoint.kwargs["sensitive_values"]) >= expected_values
    state_paths = checkpoint.kwargs["state_paths"]
    assert [(item.name, item.remote_path) for item in state_paths] == [
        ("sessions", "/tmp/codex-home/sessions"),
    ]
    assert "/tmp/codex-home/sessions" in checkpoint.kwargs["session_probe"]
    assert "/logs/agent/codex.txt" in checkpoint.kwargs["session_probe"]
    assert "/logs/agent/checkpoint" not in inspect.getsource(pier_codex)


def test_session_probe_prefers_output_root_and_rejects_newer_child_rollout(
    tmp_path: Path,
) -> None:
    output = tmp_path / "codex.txt"
    sessions = tmp_path / "sessions" / "2026" / "08" / "23"
    sessions.mkdir(parents=True)
    output.write_text(
        json.dumps({"type": "thread.started", "thread_id": ROOT_THREAD_ID})
        + "\n"
        + json.dumps({"type": "thread.started", "thread_id": CHILD_THREAD_ID})
        + "\n",
        encoding="utf-8",
    )
    (sessions / f"rollout-2026-08-23T00-00-00-{ROOT_THREAD_ID}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": ROOT_THREAD_ID, "source": "exec"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sessions / f"rollout-2026-08-23T00-01-00-{CHILD_THREAD_ID}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": CHILD_THREAD_ID,
                    "source": {
                        "subagent": {
                            "thread_spawn": {"parent_thread_id": ROOT_THREAD_ID}
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    command = pier_codex._codex_session_probe_command(
        str(output), str(sessions.parent.parent.parent),
    )

    completed = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ROOT_THREAD_ID

    # A periodic/final snapshot can race the first CLI JSON event. In that
    # window the sole source=exec rollout remains authoritative; the newer
    # child filename must not replace it.
    output.write_text("", encoding="utf-8")
    completed = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ROOT_THREAD_ID


@pytest.mark.parametrize(
    ("reader_name", "expected"),
    [
        ("_root_thread_id_from_output", None),
        ("_codex_session_unavailable", False),
    ],
)
def test_codex_output_readers_reject_regular_to_fifo_open_race_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_name: str,
    expected: object,
) -> None:
    output = tmp_path / "codex.txt"
    output.write_text(
        json.dumps({"type": "thread.started", "thread_id": ROOT_THREAD_ID})
        + "\nError: session not found\n",
        encoding="utf-8",
    )
    real_open = os.open
    observed_flags: list[int] = []

    def replace_with_fifo_before_open(
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(filename) == output:
            observed_flags.append(flags)
            output.unlink()
            os.mkfifo(output)
        return real_open(filename, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(pier_codex.os, "open", replace_with_fifo_before_open)

    assert getattr(pier_codex, reader_name)(output) == expected
    assert observed_flags
    assert observed_flags[0] & os.O_NONBLOCK
    assert observed_flags[0] & os.O_CLOEXEC


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "The phrase session not found is part of the answer.",
                },
            },
            False,
        ),
        (
            {"type": "error", "message": "failed to load session state"},
            True,
        ),
        (
            {
                "type": "turn.failed",
                "error": {"message": "conversation not found"},
            },
            True,
        ),
    ],
)
def test_session_unavailable_uses_only_structured_error_records(
    tmp_path: Path,
    record: dict[str, object],
    expected: bool,
) -> None:
    output = tmp_path / "codex.txt"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert pier_codex._codex_session_unavailable(output) is expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "Error: thread/resume failed: no rollout found for thread id "
            + ROOT_THREAD_ID,
            True,
        ),
        ("Fatal: session not found", True),
        ("The model says session not found as ordinary prose", False),
        ('{"type":"error","message":"session not found"', False),
    ],
)
def test_session_unavailable_accepts_only_explicit_non_json_error_lines(
    tmp_path: Path,
    line: str,
    expected: bool,
) -> None:
    output = tmp_path / "codex.txt"
    output.write_text(line + "\n", encoding="utf-8")

    assert pier_codex._codex_session_unavailable(output) is expected


def test_checkpoint_hook_classifier_does_not_fallback_for_agent_prose(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, recording_checkpoint)
    output = agent.logs_dir / agent._OUTPUT_FILENAME
    output.write_text(
        json.dumps({
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "Explain why session not found can occur.",
            },
        }) + "\n",
        encoding="utf-8",
    )

    # post3 selects its upstream checkpoint-hook path via this flag and calls
    # the virtual method.  The override must remain strict even when the
    # public-Pier bridge is not involved.
    monkeypatch.setattr(pier_codex, "_UPSTREAM_HAS_CHECKPOINT_HOOKS", True)
    assert "_session_unavailable" in DurableCodex.__dict__
    assert agent._session_unavailable() is False

    output.write_text(
        json.dumps({"type": "error", "message": "session not found"}) + "\n",
        encoding="utf-8",
    )
    assert agent._session_unavailable() is True


def test_run_delegates_to_upstream_so_benchmark_prompt_is_not_rewritten() -> None:
    assert DurableCodex.run is not Codex.run
    source = inspect.getsource(DurableCodex.run)
    assert "unwrapped_parent_run" in source
    assert "@with_prompt_template" in source
    assert "Continue from the interrupted turn" not in source


def test_public_pier_bridge_uses_native_resume_and_defers_cleanup(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_root = _published_checkpoint(tmp_path)
    agent = _agent(
        tmp_path,
        recording_checkpoint,
        checkpoint_path=previous_root,
    )
    checkpoint = recording_checkpoint.instances[-1]
    checkpoint.previous = {
        "assignment_id": "assignment-codex-checkpoint-1",
        "phase": "paused",
    }
    checkpoint.root_thread_id = ROOT_THREAD_ID
    executed: list[tuple[str, int]] = []
    cleanup = (
        f"rm -rf {agent._REMOTE_CODEX_SECRETS_DIR.as_posix()} "
        '"$CODEX_HOME"'
    )
    fresh_command = (
        "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
        "codex exec --dangerously-bypass-approvals-and-sandbox "
        "--skip-git-repo-check --model gpt-5.4 --json "
        "--enable unified_exec -- 'ORIGINAL BENCHMARK PROMPT' "
        "2>&1 </dev/null | tee /logs/agent/codex.txt"
    )

    async def record_exec(
        self, environment, command, env=None, cwd=None, timeout_sec=None,
    ):
        del self, environment, env, cwd, timeout_sec
        executed.append((command, len(checkpoint.finish_calls)))
        return object()

    async def public_parent(self, instruction, environment, context) -> None:
        del instruction, context
        await self.exec_as_agent(environment, fresh_command, env={})
        await self.exec_as_agent(environment, cleanup, env={})

    monkeypatch.setattr(pier_codex, "_UPSTREAM_HAS_CHECKPOINT_HOOKS", False)
    monkeypatch.setattr(Codex, "exec_as_agent", record_exec)
    monkeypatch.setattr(Codex, "run", public_parent)

    asyncio.run(agent.run("ORIGINAL BENCHMARK PROMPT", object(), object()))

    model_command = executed[0][0]
    assert "codex exec resume " in model_command
    assert ROOT_THREAD_ID in model_command
    assert "'ORIGINAL BENCHMARK PROMPT'" in model_command
    assert "Continue from the interrupted turn" not in model_command
    assert "| tee /logs/agent/codex.txt" in model_command
    assert str(tmp_path) not in model_command
    assert executed[1] == ("true", 0)
    assert executed[-1] == (cleanup, 1)
    assert checkpoint.finish_calls == [
        {
            "completed": True,
            "failure": None,
            "session_id": None,
        }
    ]


def test_public_bridge_falls_back_only_for_explicit_missing_native_session(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_root = _published_checkpoint(tmp_path)
    agent = _agent(
        tmp_path, recording_checkpoint, checkpoint_path=previous_root,
    )
    checkpoint = recording_checkpoint.instances[-1]
    checkpoint.previous = {
        "assignment_id": "assignment-codex-checkpoint-1",
        "phase": "paused",
    }
    checkpoint.root_thread_id = ROOT_THREAD_ID
    executed: list[str] = []
    fresh_command = (
        "codex exec --model gpt-5.4 --json -- 'ORIGINAL BENCHMARK PROMPT' "
        "2>&1 </dev/null | tee /logs/agent/codex.txt"
    )

    async def record_exec(
        self, environment, command, env=None, cwd=None, timeout_sec=None,
    ):
        del self, environment, env, cwd, timeout_sec
        executed.append(command)
        if "codex exec resume " in command:
            (agent.logs_dir / agent._OUTPUT_FILENAME).write_text(
                "Error: session not found\n", encoding="utf-8",
            )
            raise NonZeroAgentExitCodeError("resume failed")
        return object()

    async def public_parent(self, instruction, environment, context) -> None:
        del instruction, context
        await self.exec_as_agent(environment, fresh_command, env={})

    monkeypatch.setattr(pier_codex, "_UPSTREAM_HAS_CHECKPOINT_HOOKS", False)
    monkeypatch.setattr(Codex, "exec_as_agent", record_exec)
    monkeypatch.setattr(Codex, "run", public_parent)

    asyncio.run(agent.run("ORIGINAL BENCHMARK PROMPT", object(), object()))

    model_commands = [command for command in executed if "codex exec " in command]
    assert len(model_commands) == 2
    assert "codex exec resume " in model_commands[0]
    assert "codex exec resume " not in model_commands[1]
    assert all("'ORIGINAL BENCHMARK PROMPT'" in command for command in model_commands)
    assert 'rm -rf "$CODEX_HOME/sessions"' in executed
    assert checkpoint.events == [(
        "session_resume_degraded",
        {
            "reason": "session_unavailable",
            "progress_summary_available": False,
        },
    )]


def test_public_bridge_does_not_fallback_for_agent_message_text(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_root = _published_checkpoint(tmp_path)
    agent = _agent(
        tmp_path, recording_checkpoint, checkpoint_path=previous_root,
    )
    checkpoint = recording_checkpoint.instances[-1]
    checkpoint.previous = {
        "assignment_id": "assignment-codex-checkpoint-1",
        "phase": "paused",
    }
    checkpoint.root_thread_id = ROOT_THREAD_ID
    executed: list[str] = []
    fresh_command = (
        "codex exec --model gpt-5.4 --json -- 'ORIGINAL BENCHMARK PROMPT' "
        "2>&1 </dev/null | tee /logs/agent/codex.txt"
    )

    async def record_exec(
        self, environment, command, env=None, cwd=None, timeout_sec=None,
    ):
        del self, environment, env, cwd, timeout_sec
        executed.append(command)
        if "codex exec resume " in command:
            (agent.logs_dir / agent._OUTPUT_FILENAME).write_text(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Explain why session not found can occur.",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raise NonZeroAgentExitCodeError("resume failed")
        return object()

    async def public_parent(self, instruction, environment, context) -> None:
        del instruction, context
        await self.exec_as_agent(environment, fresh_command, env={})

    monkeypatch.setattr(pier_codex, "_UPSTREAM_HAS_CHECKPOINT_HOOKS", False)
    monkeypatch.setattr(Codex, "exec_as_agent", record_exec)
    monkeypatch.setattr(Codex, "run", public_parent)

    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(agent.run("ORIGINAL BENCHMARK PROMPT", object(), object()))

    model_commands = [command for command in executed if "codex exec " in command]
    assert len(model_commands) == 1
    assert "codex exec resume " in model_commands[0]
    assert 'rm -rf "$CODEX_HOME/sessions"' not in executed
    assert checkpoint.events == []


@pytest.mark.parametrize("resume", [False, True])
def test_model_command_rewrite_preserves_exact_rendered_instruction(
    resume: bool,
) -> None:
    instruction = "Original benchmark's exact prompt; keep $variables and spaces"
    prefix = "codex exec resume --model gpt-5.4 " if resume else (
        "codex exec --model gpt-5.4 -- "
    )
    root = f"{ROOT_THREAD_ID} " if resume else ""
    command = (
        f"{prefix}{root}{shlex.quote('changed continuation')} "
        "2>&1 </dev/null | tee /logs/agent/codex.txt"
    )

    rewritten = pier_codex._rewrite_codex_model_prompt(command, instruction)

    assert rewritten.endswith("| tee /logs/agent/codex.txt")
    assert shlex.quote(instruction) in rewritten
    assert "changed continuation" not in rewritten


def test_public_pier_bridge_leaves_fresh_model_command_unchanged(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, recording_checkpoint)
    executed: list[str] = []
    fresh_command = (
        "codex exec --model gpt-5.4 --json -- 'ORIGINAL BENCHMARK PROMPT' "
        "2>&1 </dev/null | tee /logs/agent/codex.txt"
    )

    async def record_exec(
        self, environment, command, env=None, cwd=None, timeout_sec=None,
    ):
        del self, environment, env, cwd, timeout_sec
        executed.append(command)
        return object()

    async def public_parent(self, instruction, environment, context) -> None:
        del instruction, context
        await self.exec_as_agent(environment, fresh_command, env={})

    monkeypatch.setattr(pier_codex, "_UPSTREAM_HAS_CHECKPOINT_HOOKS", False)
    monkeypatch.setattr(Codex, "exec_as_agent", record_exec)
    monkeypatch.setattr(Codex, "run", public_parent)

    asyncio.run(agent.run("ORIGINAL BENCHMARK PROMPT", object(), object()))

    assert executed[0] == fresh_command


def test_start_maps_published_sessions_and_native_root_thread(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    previous_root = _published_checkpoint(tmp_path)
    agent = _agent(
        tmp_path,
        recording_checkpoint,
        checkpoint_path=previous_root,
    )
    checkpoint = recording_checkpoint.instances[-1]
    checkpoint.previous = {
        "assignment_id": "assignment-codex-checkpoint-1",
        "phase": "paused",
    }
    checkpoint.root_thread_id = ROOT_THREAD_ID

    previous, root = asyncio.run(agent._start_checkpoint(object(), {}))

    payload = previous_root / "snapshots" / GENERATION
    assert root == ROOT_THREAD_ID
    assert previous is not None
    assert previous["sessions_dir"] == "provider-state/sessions"
    assert previous["root_thread_id"] == ROOT_THREAD_ID
    assert agent._resume_checkpoint_path == payload
    assert agent._checkpoint_manifest_path == checkpoint.manifest_path
    assert agent._checkpoint_root_thread_id(previous_root) == ROOT_THREAD_ID


def test_pre_session_checkpoint_restores_workspace_as_fresh_original_run(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    previous_root = _published_checkpoint(tmp_path, with_sessions=False)
    agent = _agent(
        tmp_path,
        recording_checkpoint,
        checkpoint_path=previous_root,
    )
    checkpoint = recording_checkpoint.instances[-1]
    checkpoint.previous = {"phase": "paused"}
    checkpoint.root_thread_id = ROOT_THREAD_ID

    assert asyncio.run(agent._start_checkpoint(object(), {})) == (None, None)


def test_finish_passes_native_output_thread_and_events_to_durable_manager(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    agent = _agent(tmp_path, recording_checkpoint)
    checkpoint = recording_checkpoint.instances[-1]
    output = agent.logs_dir / agent._OUTPUT_FILENAME
    output.write_text(
        json.dumps({"type": "thread.started", "thread_id": ROOT_THREAD_ID})
        + "\n"
        + json.dumps({"type": "turn.completed"})
        + "\n",
        encoding="utf-8",
    )
    failure = RuntimeError("paid run interrupted")

    async def finalize() -> None:
        await agent._finish_checkpoint(
            object(), {}, completed=False, failure=failure,
        )

    asyncio.run(finalize())
    agent._checkpoint_event("session_resume_started", attempt=1)

    assert checkpoint.finish_calls == [
        {
            "completed": False,
            "failure": failure,
            "session_id": ROOT_THREAD_ID,
        }
    ]
    assert checkpoint.events == [("session_resume_started", {"attempt": 1})]


def test_run_cancellation_waits_for_shielded_finalizer(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        agent = _agent(tmp_path, recording_checkpoint)
        checkpoint = recording_checkpoint.instances[-1]
        model_started = asyncio.Event()
        model_wait = asyncio.Event()
        checkpoint.finish_started = asyncio.Event()
        checkpoint.finish_release = asyncio.Event()
        checkpoint.finish_done = asyncio.Event()

        async def upstream_with_the_current_shield_bug(
            self, instruction, environment, context,
        ) -> None:
            del instruction, context
            failure = None
            try:
                self._upstream_phase = "model"
                model_started.set()
                await model_wait.wait()
            except BaseException as exc:
                failure = exc
                raise
            finally:
                try:
                    await asyncio.shield(
                        self._finish_checkpoint(
                            environment, {}, completed=False, failure=failure,
                        )
                    )
                except BaseException:
                    pass

        monkeypatch.setattr(Codex, "run", upstream_with_the_current_shield_bug)
        running = asyncio.create_task(agent.run("unchanged", object(), object()))
        await model_started.wait()
        running.cancel()
        await checkpoint.finish_started.wait()
        await asyncio.sleep(0)

        assert not running.done()
        assert not checkpoint.finish_done.is_set()
        checkpoint.finish_release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert checkpoint.finish_done.is_set()
        assert agent._checkpoint_finalizer_task is not None
        assert agent._checkpoint_finalizer_task.done()

    asyncio.run(scenario())


def test_run_cancellation_in_post_start_window_cancels_paid_child_and_finalizes(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        agent = _agent(tmp_path, recording_checkpoint)
        checkpoint = recording_checkpoint.instances[-1]
        checkpoint.finish_done = asyncio.Event()
        post_start = asyncio.Event()
        paid_child_reaped = asyncio.Event()
        never = asyncio.Event()
        cleanup_commands: list[str] = []
        cleanup_after_finish: list[bool] = []

        async def record_agent_command(
            self, environment, command, env=None, cwd=None, timeout_sec=None,
        ):
            del self, environment, env, cwd, timeout_sec
            cleanup_commands.append(command)
            cleanup_after_finish.append(checkpoint.finish_done.is_set())
            return None

        async def upstream_stuck_after_checkpoint_start(
            self, instruction, environment, context,
        ) -> None:
            del instruction, context
            await self._start_checkpoint(environment, {})
            post_start.set()
            try:
                await never.wait()
            finally:
                paid_child_reaped.set()

        monkeypatch.setattr(
            Codex, "run", upstream_stuck_after_checkpoint_start,
        )
        monkeypatch.setattr(Codex, "exec_as_agent", record_agent_command)
        running = asyncio.create_task(
            agent.run("unchanged", object(), object()),
        )
        await post_start.wait()
        running.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=2)
        assert paid_child_reaped.is_set()
        assert len(checkpoint.finish_calls) == 1
        assert checkpoint.finish_calls[0]["completed"] is False
        assert isinstance(
            checkpoint.finish_calls[0]["failure"], asyncio.CancelledError,
        )
        assert agent._checkpoint_finalizer_task is not None
        assert agent._checkpoint_finalizer_task.done()
        assert cleanup_commands
        assert set(cleanup_commands) == {
            'rm -rf /tmp/codex-secrets "$CODEX_HOME"',
        }
        assert all(cleanup_after_finish)

    asyncio.run(scenario())


def test_run_cancellation_during_finalizer_does_not_race_upstream_cleanup(
    tmp_path: Path,
    recording_checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        agent = _agent(tmp_path, recording_checkpoint)
        checkpoint = recording_checkpoint.instances[-1]
        checkpoint.finish_started = asyncio.Event()
        checkpoint.finish_release = asyncio.Event()
        checkpoint.finish_done = asyncio.Event()
        cleanup_started = asyncio.Event()

        async def upstream_with_the_current_shield_bug(
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

        monkeypatch.setattr(Codex, "run", upstream_with_the_current_shield_bug)
        running = asyncio.create_task(agent.run("unchanged", object(), object()))
        await checkpoint.finish_started.wait()
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

    asyncio.run(scenario())
