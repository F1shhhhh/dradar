from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pier")

from dradar import pier_dsh
from dradar.pier_checkpoint import CheckpointError, CheckpointIncompatibleError
from dradar.pier_dsh import (
    DSH_VERSION,
    NODE_SHA256,
    NODE_VERSION,
    RUNTIME_MODELS,
    SUPPORTED_MODELS,
    SUPPORTED_REASONING_EFFORTS,
    DshMinimal,
)
from dradar.providers import (
    DSH_MODELS as MAIN_FLOW_MODELS,
    DSH_SUPPORTED_EFFORTS as MAIN_FLOW_EFFORTS,
    DSH_VERSION as MAIN_FLOW_VERSION,
)


class FakeEnvironment:
    default_user = "agent"

    def __init__(self, *, fail_dsh: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.uploads: list[tuple[Path, str, bytes]] = []
        self.fail_dsh = fail_dsh

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        command = str(kwargs.get("command", ""))
        if self.fail_dsh and "dsh --profile headless" in command:
            return SimpleNamespace(return_code=7, stdout="", stderr="quota")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        self.uploads.append((source, target_path, source.read_bytes()))


class RecordingCheckpoint:
    instances: list["RecordingCheckpoint"] = []
    resume_session_id: str | None = None
    start_error: BaseException | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.finished: tuple[bool, BaseException | None] | None = None
        type(self).instances.append(self)

    async def start(self, agent, environment, env):
        del agent, environment, env
        self.started = True
        if type(self).start_error is not None:
            raise type(self).start_error
        return type(self).resume_session_id

    async def finish(
        self, agent, environment, env, *, completed, failure, session_id=None,
    ):
        del agent, environment, env, session_id
        self.finished = (completed, failure)


@pytest.fixture
def recording_checkpoint(monkeypatch: pytest.MonkeyPatch):
    RecordingCheckpoint.instances = []
    RecordingCheckpoint.resume_session_id = None
    RecordingCheckpoint.start_error = None
    monkeypatch.setattr(pier_dsh, "DurableCheckpoint", RecordingCheckpoint)
    yield RecordingCheckpoint
    RecordingCheckpoint.instances = []
    RecordingCheckpoint.resume_session_id = None
    RecordingCheckpoint.start_error = None


def test_standalone_adapter_matches_main_flow_contract() -> None:
    assert DSH_VERSION == MAIN_FLOW_VERSION
    assert SUPPORTED_MODELS == frozenset(MAIN_FLOW_MODELS)
    assert SUPPORTED_REASONING_EFFORTS == MAIN_FLOW_EFFORTS


def make_key(tmp_path: Path, value: str = "test-secret-never-log") -> Path:
    key = tmp_path / "deepseek.key"
    key.write_text(value, encoding="utf-8")
    if os.name != "nt":
        key.chmod(0o600)
    return key


def make_agent(tmp_path: Path, **kwargs: object) -> DshMinimal:
    return DshMinimal(
        logs_dir=tmp_path / "logs",
        api_key_file=str(make_key(tmp_path)),
        **kwargs,
    )


def artifact_binding() -> dict[str, str]:
    return {
        "artifact_assignment_id": "a" * 32,
        "artifact_run_id": "b" * 32,
        "artifact_task_id": "httpx-streaming-json-iteration",
    }


def test_install_spec_is_fully_pinned(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    spec = agent.install_spec()
    command = " ".join(step.run for step in spec.steps)

    assert spec.agent_name == "dsh-minimal"
    assert spec.version == DSH_VERSION
    assert f"@deepseek-ai/dsh@{DSH_VERSION}" in command
    assert f"node-v{NODE_VERSION}-linux-${{node_arch}}.tar.xz" in command
    assert NODE_SHA256["x64"] in command
    assert NODE_SHA256["arm64"] in command
    assert "@latest" not in command
    assert "requires a glibc task image" in command
    assert "g++ make python3" in command
    assert "inotify-tools" not in command
    assert "--fetch-retries=5" in command
    assert "--fetch-retry-maxtimeout=120000" in command
    assert spec.verification_command is not None
    assert f"v{NODE_VERSION}" in spec.verification_command
    assert DSH_VERSION in spec.verification_command
    assert spec.cache_key == (
        f"dradar-dsh-minimal-{DSH_VERSION}-node-{NODE_VERSION}-patch-v4"
    )
    bash = shutil.which("bash")
    if bash is not None:
        syntax = subprocess.run(
            [bash, "-n", "-c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_only_deepseek_api_is_allowed_at_runtime(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    assert agent.network_allowlist().domains == ["api.deepseek.com"]


@pytest.mark.parametrize("effort", ["low", "medium", "invalid"])
def test_rejects_unsupported_reasoning_effort(tmp_path: Path, effort: str) -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        make_agent(tmp_path, reasoning_effort=effort)


def test_normalizes_supported_model_prefix(tmp_path: Path) -> None:
    agent = make_agent(
        tmp_path,
        model_name="deepseek-official/dsh-deepseek-v4-pro",
        version=None,
    )
    assert agent.model_name == "deepseek-v4-pro"
    assert agent.version() == DSH_VERSION


def test_rejects_unsupported_model_or_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DSH model"):
        make_agent(tmp_path, model_name="deepseek-chat")
    with pytest.raises(ValueError, match="exact version"):
        make_agent(tmp_path, version="0.1.0-rc.5")


def test_rejects_checkpoint_effort_identity_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint effort"):
        make_agent(
            tmp_path,
            reasoning_effort="high",
            checkpoint_effort="max",
        )


@pytest.mark.parametrize(
    "extra_env",
    [
        {"DEEPSEEK_API_KEY": "wrong-channel"},
        {"DSH_MODEL": "unexpected"},
        {"DEEPSEEK_BASE_URL": "https://unexpected.invalid"},
        {"NODE_OPTIONS": "--require=/tmp/untrusted.js"},
        {"NODE_USE_ENV_PROXY": "0"},
    ],
)
def test_rejects_reserved_extra_env(tmp_path: Path, extra_env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="api_key_file"):
        make_agent(tmp_path, extra_env=extra_env)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission check")
def test_rejects_group_readable_key_file(tmp_path: Path) -> None:
    key = make_key(tmp_path)
    key.chmod(0o640)
    with pytest.raises(ValueError, match="permissions"):
        DshMinimal(logs_dir=tmp_path / "logs", api_key_file=str(key))


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        (model, effort)
        for model in (
            "dsh-deepseek-v4-flash",
            "dsh-deepseek-v4-pro",
            "dsh-deepseek-v4-flash-vision-exp",
        )
        for effort in ("off", "high", "max")
    ],
)
def test_run_supports_model_effort_matrix_without_logging_secret(
    tmp_path: Path,
    model: str,
    effort: str,
) -> None:
    secret = "test-secret-never-log"
    agent = make_agent(
        tmp_path,
        model_name=model,
        reasoning_effort=effort,
        **artifact_binding(),
    )
    environment = FakeEnvironment()

    asyncio.run(
        agent.run(
            "Fix the quoted 'edge' and do not expand $HOME",
            environment,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )
    )

    assert len(environment.uploads) == 3
    uploaded = {target: data for _, target, data in environment.uploads}
    patch = uploaded["/tmp/dsh-config/headless-minimal.patch.yml"].decode()
    runner = uploaded["/tmp/dsh-config/minimal-headless-runner.mjs"].decode()
    key_target = next(
        target for target in uploaded if target.endswith("/deepseek-api-key")
    )
    assert secret.encode() == uploaded[key_target]
    assert "id: agent-presets" in patch
    assert "default: minimal" in patch
    assert "includeUserRoot: false" in patch
    assert "config/agent-presets" in patch
    assert "id: tool-web\n  disabled: true" in patch
    assert "id: tool-subagent\n  disabled: true" in patch
    assert "id: system-prompt\n  disabled: true" not in patch
    assert "id: headless-runner\n  disabled: true" in patch
    assert "id: minimal-headless-runner" in patch
    assert "name: /tmp/dsh-config/minimal-headless-runner.mjs" in patch
    assert "path: !!js process.env.DSH_CREDENTIALS_FILE" in patch
    assert "watch: false" in patch
    assert 'presets.resolve("minimal")' in runner
    assert "await presets.mount(agentCtx, preset.id)" in runner
    assert "await agents.resume({" in runner
    assert "resumeSessionId: SessionId(resumeSessionId)" in runner
    assert 'writeFileSync(process.env.DSH_SESSION_ID_FILE' in runner
    assert 'String(agent.session.id) + "\\n"' in runner
    assert "const outcome = summarize(agent.session.events, 0)" in runner
    assert "continue the previous" not in runner.lower()
    assert 'const attachments = ctx.get("attachments")' in runner
    assert "unlinkSync(process.env.DSH_CREDENTIALS_FILE)" in runner
    assert 'readFileSync("/app/question.png")' in runner
    assert "await attachments.saveImage" in runner
    assert '{ type: "image", attachment: imageRef }' in runner
    assert "visionInputAttached: imageRef !== null" in runner
    assert "agentPreset: preset.id" in runner
    assert 'event.type === "assistant/chunk"' in runner
    assert 'usageByStep.set(`${event.data.turn}:${event.data.step}`' in runner
    assert 'schema: "dsh-provider-usage-v2"' in runner
    assert "occurredAt: usageTimestamp(event)" in runner
    assert "writeFileSync(process.env.DSH_USAGE_FILE" in runner
    assert 'schema: "dradar-dsh-outcome-v1"' in runner
    assert "assignmentId: process.env.DRADAR_ASSIGNMENT_ID" in runner
    assert "artifactRunId: process.env.DRADAR_ARTIFACT_RUN_ID" in runner
    assert "writeFileSync(process.env.DSH_OUTCOME_FILE" in runner

    serialized_calls = repr(environment.calls)
    assert secret not in serialized_calls
    setup_call = next(
        call
        for call in environment.calls
        if "mkdir -p /logs/agent/dsh-home" in str(call["command"])
    )
    assert setup_call["user"] == "root"
    assert "chown agent" in str(setup_call["command"])
    dsh_call = next(
        call
        for call in environment.calls
        if "dsh --profile headless" in str(call["command"])
    )
    command = str(dsh_call["command"])
    assert "set -euo pipefail; " in command
    assert command.index(f"rm -f {key_target}") < command.index(
        "dsh --profile headless"
    )
    assert "inotifywait" not in command
    assert "DEEPSEEK_API_KEY" not in (dsh_call.get("env") or {})
    assert dsh_call["cwd"] == "/app"
    assert (dsh_call.get("env") or {})["DSH_MODEL"] == RUNTIME_MODELS[model]
    assert (dsh_call.get("env") or {})["DSH_REASONING_EFFORT"] == effort
    assert (dsh_call.get("env") or {})["DSH_USAGE_FILE"] == (
        "/logs/agent/dsh-home/dsh-usage.json"
    )
    assert (dsh_call.get("env") or {})["DSH_OUTCOME_FILE"] == (
        "/logs/agent/dsh-home/dsh-outcome.json"
    )
    assert (dsh_call.get("env") or {})["DSH_SESSION_ID_FILE"] == (
        "/logs/agent/dsh-session-id"
    )
    assert (dsh_call.get("env") or {})["DRADAR_ASSIGNMENT_ID"] == "a" * 32
    assert (dsh_call.get("env") or {})["DRADAR_ARTIFACT_RUN_ID"] == "b" * 32
    assert (dsh_call.get("env") or {})["DRADAR_TASK_ID"] == (
        "httpx-streaming-json-iteration"
    )
    assert (dsh_call.get("env") or {})["NODE_USE_ENV_PROXY"] == "1"
    assert agent.SUPPORTS_ATIF is False


def test_checkpoint_resume_uses_native_session_and_exact_same_instruction(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    recording_checkpoint.resume_session_id = "session-resume-12345678"
    instruction = "Fix the quoted 'edge' and do not expand $HOME"
    agent = make_agent(
        tmp_path,
        checkpoint_enabled="true",
        checkpoint_assignment_id="assignment-checkpoint-1",
        checkpoint_task_id="httpx-streaming-json-iteration",
        checkpoint_effort="max",
        checkpoint_resume_generation="2",
        checkpoint_path=str(tmp_path / "previous"),
        model_name="dsh-deepseek-v4-pro",
        reasoning_effort="max",
        **artifact_binding(),
    )
    environment = FakeEnvironment()

    asyncio.run(
        agent.run(
            instruction,
            environment,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )
    )

    checkpoint = recording_checkpoint.instances[-1]
    assert checkpoint.started is True
    assert checkpoint.finished == (True, None)
    assert checkpoint.kwargs["assignment_id"] == "assignment-checkpoint-1"
    assert checkpoint.kwargs["task_id"] == "httpx-streaming-json-iteration"
    assert checkpoint.kwargs["model"] == "dsh-deepseek-v4-pro"
    assert checkpoint.kwargs["effort"] == "max"
    assert checkpoint.kwargs["resume_generation"] == "2"
    assert checkpoint.kwargs["harness"] == "dsh-minimal"
    assert checkpoint.kwargs["provider"] == "deepseek"
    paths = {
        (item.name, item.remote_path)
        for item in checkpoint.kwargs["state_paths"]
    }
    assert paths == {
        ("dsh-sessions", "/logs/agent/dsh-home/sessions"),
        ("dsh-attachments", "/logs/agent/dsh-home/attachments"),
    }
    assert all("credential" not in path for _, path in paths)

    dsh_call = next(
        call
        for call in environment.calls
        if "dsh --profile headless" in str(call["command"])
    )
    assert (dsh_call.get("env") or {})["DSH_RESUME_SESSION_ID"] == (
        "session-resume-12345678"
    )
    command = str(dsh_call["command"])
    assert shlex.quote(instruction) in command
    assert "continue the previous" not in command.lower()


def test_checkpoint_marks_failed_paid_run_paused(
    tmp_path: Path,
    recording_checkpoint,
) -> None:
    agent = make_agent(
        tmp_path,
        checkpoint_enabled=True,
        checkpoint_assignment_id="assignment-checkpoint-2",
        checkpoint_task_id="task-checkpoint-2",
    )
    environment = FakeEnvironment(fail_dsh=True)

    with pytest.raises(Exception, match="exit 7"):
        asyncio.run(
            agent.run(
                "Fix it",
                environment,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )
        )

    checkpoint = recording_checkpoint.instances[-1]
    assert checkpoint.finished is not None
    completed, failure = checkpoint.finished
    assert completed is False
    assert failure is not None


@pytest.mark.parametrize(
    "start_error",
    [
        CheckpointError("checkpoint manifest is unreadable"),
        CheckpointIncompatibleError("checkpoint runtime identity mismatch"),
    ],
)
def test_checkpoint_rejects_corrupt_or_mismatched_state_before_paid_run(
    tmp_path: Path,
    recording_checkpoint,
    start_error: BaseException,
) -> None:
    recording_checkpoint.start_error = start_error
    agent = make_agent(
        tmp_path,
        checkpoint_enabled=True,
        checkpoint_assignment_id="assignment-checkpoint-3",
        checkpoint_task_id="task-checkpoint-3",
    )
    environment = FakeEnvironment()

    with pytest.raises(type(start_error), match=str(start_error)):
        asyncio.run(
            agent.run(
                "Fix it",
                environment,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )
        )

    assert not any(
        "dsh --profile headless" in str(call["command"])
        for call in environment.calls
    )
    assert not any(
        target.endswith("/deepseek-api-key")
        for _, target, _ in environment.uploads
    )
