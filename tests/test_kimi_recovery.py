from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from dradar.kimi_recovery import (
    KIMI_RESUME_PROMPT,
    kimi_provider_connection_stderr_is_retryable,
    native_session_probe_command,
    pier_exit_code,
    run_with_kimi_resume,
    unique_session_probe_command,
    validated_session_id,
)

SESSION = "832d7f94-ab9a-4f83-b630-37a3dab65025"
KIMI_SESSION = f"session_{SESSION}"


class CommandFailed(RuntimeError):
    def __init__(self, code: int, *, stdout: str = "", stderr: str = ""):
        super().__init__(
            f"Command failed (exit {code}): kimi --print\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )


def test_pier_exit_code_is_fail_closed() -> None:
    assert pier_exit_code(CommandFailed(75)) == 75
    assert pier_exit_code(RuntimeError("model said exit 75")) is None


def test_exact_kimi_connection_error_on_stderr_is_retryable() -> None:
    assert kimi_provider_connection_stderr_is_retryable(
        (
            "error: failed to run prompt: provider.connection_error: "
            "Connection error.\n"
        )
    ) is True


@pytest.mark.parametrize(
    "stderr_line",
    [
        "",
        (
            "warning: provider request failed after 10 retries\n"
            "error: failed to run prompt: provider.connection_error: "
            "Connection error."
        ),
        (
            "error: failed to run prompt: provider.connection_error: "
            "Connection error.\nunrelated terminal failure"
        ),
        (
            "error: failed to run prompt: provider.connection_error: "
            "Connection error. "
        ),
    ],
)
def test_kimi_connection_fallback_is_fail_closed(stderr_line: str) -> None:
    assert kimi_provider_connection_stderr_is_retryable(stderr_line) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (SESSION, KIMI_SESSION),
        (f" {SESSION}\n", KIMI_SESSION),
        (KIMI_SESSION, KIMI_SESSION),
        (f" {KIMI_SESSION}\n", KIMI_SESSION),
        ("not-a-session", None),
        (f"{SESSION}; touch /tmp/owned", None),
        (f"session_{SESSION}; touch /tmp/owned", None),
        (f"other_{SESSION}", None),
    ],
)
def test_session_id_requires_one_canonical_kimi_id(
    value: str, expected: str | None,
) -> None:
    assert validated_session_id(value) == expected


def test_native_session_probe_finds_one_nested_session(tmp_path) -> None:
    wire = (
        tmp_path / "2026-08-23" / "project" / KIMI_SESSION
        / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    wire.write_text("{}\n", encoding="utf-8")
    (wire.parents[2] / "state.json").write_text("{}\n", encoding="utf-8")
    index = tmp_path / "session_index.jsonl"
    index.write_text(json.dumps({
        "sessionId": KIMI_SESSION,
        "sessionDir": str(wire.parents[2]),
        "workDir": "/app",
    }) + "\n", encoding="utf-8")

    command = native_session_probe_command(str(tmp_path), str(index), SESSION)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == KIMI_SESSION + "\n"


def test_native_session_probe_rejects_missing_or_duplicate_session(tmp_path) -> None:
    index = tmp_path / "session_index.jsonl"
    index.write_text("", encoding="utf-8")
    command = native_session_probe_command(
        str(tmp_path), str(index), KIMI_SESSION,
    )
    missing = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert missing.stdout == ""

    for parent in ("first", "second"):
        wire = tmp_path / parent / KIMI_SESSION / "agents" / "main" / "wire.jsonl"
        wire.parent.mkdir(parents=True)
        wire.write_text("{}\n", encoding="utf-8")
        (wire.parents[2] / "state.json").write_text("{}\n", encoding="utf-8")
        with index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "sessionId": KIMI_SESSION,
                "sessionDir": str(wire.parents[2]),
                "workDir": "/app",
            }) + "\n")
    duplicate = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert duplicate.stdout == ""


def test_native_session_probe_rejects_unsafe_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="session id is invalid"):
        native_session_probe_command(
            str(tmp_path), str(tmp_path / "session_index.jsonl"),
            f"{KIMI_SESSION}; touch owned",
        )


def test_native_session_probe_rejects_index_outside_session_root(tmp_path) -> None:
    root = tmp_path / "sessions"
    wire = root / "wd" / KIMI_SESSION / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text("{}\n", encoding="utf-8")
    (wire.parents[2] / "state.json").write_text("{}\n", encoding="utf-8")
    index = tmp_path / "session_index.jsonl"
    index.write_text(json.dumps({
        "sessionId": KIMI_SESSION,
        "sessionDir": str(tmp_path / "outside" / KIMI_SESSION),
        "workDir": "/app",
    }) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            "/bin/sh", "-c",
            native_session_probe_command(str(root), str(index), KIMI_SESSION),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_native_session_probe_uses_latest_append_only_index_entry(
    tmp_path,
) -> None:
    root = tmp_path / "sessions"
    session_dir = root / "wd" / KIMI_SESSION
    wire = session_dir / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text("{}\n", encoding="utf-8")
    (session_dir / "state.json").write_text("{}\n", encoding="utf-8")
    entry = {
        "sessionId": KIMI_SESSION,
        "sessionDir": str(session_dir),
        "workDir": "/app",
    }
    index = tmp_path / "session_index.jsonl"
    index.write_text(
        json.dumps(entry) + "\n" + json.dumps(entry) + "\n",
        encoding="utf-8",
    )

    command = native_session_probe_command(str(root), str(index), KIMI_SESSION)
    duplicate_live = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert duplicate_live.stdout == KIMI_SESSION + "\n"

    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "sessionId": KIMI_SESSION,
            "deleted": True,
        }) + "\n")
    deleted = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert deleted.stdout == ""


def test_native_session_probe_rejects_index_wire_directory_mismatch(
    tmp_path,
) -> None:
    root = tmp_path / "sessions"
    indexed_dir = root / "indexed" / KIMI_SESSION
    wire_dir = root / "actual" / KIMI_SESSION
    wire = wire_dir / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text("{}\n", encoding="utf-8")
    (wire_dir / "state.json").write_text("{}\n", encoding="utf-8")
    index = tmp_path / "session_index.jsonl"
    index.write_text(json.dumps({
        "sessionId": KIMI_SESSION,
        "sessionDir": str(indexed_dir),
        "workDir": "/app",
    }) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            "/bin/sh", "-c",
            native_session_probe_command(str(root), str(index), KIMI_SESSION),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_native_session_probe_prefers_state_workdir_over_stale_index(
    tmp_path,
) -> None:
    root = tmp_path / "sessions"
    session_dir = root / "wd" / KIMI_SESSION
    wire = session_dir / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text("{}\n", encoding="utf-8")
    (session_dir / "state.json").write_text(
        json.dumps({"workDir": "/app"}) + "\n", encoding="utf-8",
    )
    index = tmp_path / "session_index.jsonl"
    index.write_text(json.dumps({
        "sessionId": KIMI_SESSION,
        "sessionDir": str(session_dir),
        "workDir": "/stale/original/path",
    }) + "\n", encoding="utf-8")
    command = native_session_probe_command(str(root), str(index), KIMI_SESSION)

    accepted = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert accepted.stdout == KIMI_SESSION + "\n"

    (session_dir / "state.json").write_text(
        json.dumps({"workDir": "/another/worktree"}) + "\n",
        encoding="utf-8",
    )
    rejected = subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert rejected.stdout == ""


def test_unique_session_probe_finds_and_copies_one_nested_session(tmp_path) -> None:
    wire = (
        tmp_path / "sessions" / "workdir-key" / KIMI_SESSION
        / "agents" / "main" / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    wire.write_text("wire\n", encoding="utf-8")
    copied = tmp_path / "copied-wire.jsonl"

    result = subprocess.run(
        [
            "/bin/sh", "-c",
            unique_session_probe_command(
                str(tmp_path / "sessions"), copy_to=str(copied),
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == KIMI_SESSION + "\n"
    assert copied.read_text(encoding="utf-8") == "wire\n"


def test_unique_session_probe_rejects_ambiguous_sessions(tmp_path) -> None:
    root = tmp_path / "sessions"
    for index in (1, 2):
        wire = (
            root / f"workdir-{index}" / f"session_{index:08d}-0000-4000-8000-000000000000"
            / "agents" / "main" / "wire.jsonl"
        )
        wire.parent.mkdir(parents=True)
        wire.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        ["/bin/sh", "-c", unique_session_probe_command(str(root))],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_success_does_not_probe_or_resume() -> None:
    calls: list[object] = []

    async def scenario() -> tuple[int, str | None]:
        async def initial() -> None:
            calls.append("initial")

        async def find() -> str | None:
            calls.append("find")
            return SESSION

        async def resume(session_id: str, prompt: str) -> None:
            calls.append((session_id, prompt))

        return await run_with_kimi_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            delays=(0,),
        )

    assert asyncio.run(scenario()) == (0, None)
    assert calls == ["initial"]


def test_exit_75_resumes_same_session_and_workspace() -> None:
    calls: list[object] = []

    async def scenario() -> tuple[int, str | None]:
        async def initial() -> None:
            calls.append("initial")
            raise CommandFailed(75)

        async def find() -> str | None:
            calls.append("find")
            return SESSION

        async def resume(session_id: str, prompt: str) -> None:
            calls.append(("resume", session_id, prompt))

        async def no_wait(delay: float) -> None:
            calls.append(("sleep", delay))

        return await run_with_kimi_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            delays=(10, 30),
            sleep=no_wait,
        )

    assert asyncio.run(scenario()) == (1, KIMI_SESSION)
    assert calls == [
        "initial",
        "find",
        ("sleep", 10),
        ("resume", KIMI_SESSION, KIMI_RESUME_PROMPT),
    ]


def test_exit_1_with_exact_provider_signal_resumes_same_session() -> None:
    calls: list[object] = []

    async def scenario() -> tuple[int, str | None]:
        async def initial() -> None:
            calls.append("initial")
            raise CommandFailed(
                1,
                stderr=(
                    "error: failed to run prompt: provider.connection_error: "
                    "Connection error."
                ),
            )

        async def find() -> str | None:
            calls.append("find")
            return SESSION

        async def resume(session_id: str, prompt: str) -> None:
            calls.append(("resume", session_id, prompt))

        async def no_wait(delay: float) -> None:
            calls.append(("sleep", delay))

        async def classify(error: BaseException) -> bool:
            calls.append(("classify", pier_exit_code(error)))
            return (
                pier_exit_code(error) == 1
                and kimi_provider_connection_stderr_is_retryable(
                    "error: failed to run prompt: provider.connection_error: "
                    "Connection error.\n"
                )
            )

        return await run_with_kimi_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            delays=(10, 30),
            sleep=no_wait,
            classify_retryable_error=classify,
        )

    assert asyncio.run(scenario()) == (1, KIMI_SESSION)
    assert calls == [
        "initial",
        ("classify", 1),
        "find",
        ("sleep", 10),
        ("resume", KIMI_SESSION, KIMI_RESUME_PROMPT),
    ]


def test_retry_budget_is_bounded_and_reraises_last_failure() -> None:
    attempts: list[str] = []

    async def scenario() -> None:
        async def initial() -> None:
            raise CommandFailed(75)

        async def find() -> str | None:
            return SESSION

        async def resume(session_id: str, prompt: str) -> None:
            del prompt
            attempts.append(session_id)
            raise CommandFailed(75)

        async def no_wait(_delay: float) -> None:
            return None

        await run_with_kimi_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            delays=(10, 30),
            sleep=no_wait,
        )

    with pytest.raises(CommandFailed, match=r"exit 75"):
        asyncio.run(scenario())
    assert attempts == [KIMI_SESSION, KIMI_SESSION]


def test_retry_never_drifts_to_a_newly_discovered_session() -> None:
    other_session = "d9428888-122b-4543-9bda-fcb60bf132d1"
    resumed_sessions: list[str] = []
    find_calls = 0

    async def scenario() -> tuple[int, str | None]:
        async def initial() -> None:
            raise CommandFailed(75)

        async def find() -> str | None:
            nonlocal find_calls
            find_calls += 1
            return SESSION if find_calls == 1 else other_session

        async def resume(session_id: str, prompt: str) -> None:
            del prompt
            resumed_sessions.append(session_id)
            if len(resumed_sessions) == 1:
                raise CommandFailed(75)

        async def no_wait(_delay: float) -> None:
            return None

        return await run_with_kimi_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            delays=(10, 30),
            sleep=no_wait,
        )

    assert asyncio.run(scenario()) == (2, KIMI_SESSION)
    assert resumed_sessions == [KIMI_SESSION, KIMI_SESSION]
    assert find_calls == 1


def test_nonretryable_exit_and_missing_session_never_restart() -> None:
    resume_calls = 0

    async def run(code: int, session: str | None) -> None:
        nonlocal resume_calls

        async def initial() -> None:
            raise CommandFailed(code)

        async def find() -> str | None:
            return session

        async def resume(_session_id: str, _prompt: str) -> None:
            nonlocal resume_calls
            resume_calls += 1

        await run_with_kimi_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            delays=(0,),
        )

    with pytest.raises(CommandFailed, match=r"exit 1"):
        asyncio.run(run(1, SESSION))
    with pytest.raises(CommandFailed, match=r"exit 75"):
        asyncio.run(run(75, "invalid"))
    assert resume_calls == 0
