from __future__ import annotations

import asyncio

import pytest

from dradar.grok_recovery import (
    GROK_RESUME_PROMPT,
    grok_provider_stream_is_retryable,
    pier_exit_code,
    run_with_grok_resume,
    validated_session_id,
)


SESSION = "01a01e4f-040a-71e3-a6c4-fdf6083ae20a"
STREAM_ERROR = (
    "reqwest error stream: error sending request for url "
    "(https://cli-chat-proxy.grok.com/v1/responses)"
)


class CommandFailed(RuntimeError):
    def __init__(self, code: int, output: str = STREAM_ERROR):
        self.output = output
        super().__init__(f"Command failed (exit {code}): grok\n{output}")


def test_stream_classifier_is_specific_to_official_response_host() -> None:
    assert grok_provider_stream_is_retryable(STREAM_ERROR)
    assert not grok_provider_stream_is_retryable("reqwest error stream")
    assert not grok_provider_stream_is_retryable(
        "error sending request for url (https://example.com/v1/responses)"
    )


def test_exit_code_and_session_id_are_fail_closed() -> None:
    assert pier_exit_code(CommandFailed(1)) == 1
    assert pier_exit_code(RuntimeError("exit 1")) is None
    assert validated_session_id(SESSION) == SESSION
    assert validated_session_id(f"{SESSION}; touch /tmp/owned") is None


def test_retry_resumes_same_session_with_bounded_backoff() -> None:
    calls: list[object] = []
    failures = iter([CommandFailed(1), None])

    async def scenario() -> tuple[int, str | None]:
        async def initial() -> None:
            calls.append("initial")
            raise CommandFailed(1)

        async def find() -> str | None:
            calls.append("find")
            return SESSION

        async def resume(session_id: str, prompt: str) -> None:
            calls.append(("resume", session_id, prompt))
            failure = next(failures)
            if failure is not None:
                raise failure

        async def classify(error: BaseException) -> bool:
            return pier_exit_code(error) == 1 and grok_provider_stream_is_retryable(
                str(error)
            )

        async def no_wait(delay: float) -> None:
            calls.append(("sleep", delay))

        return await run_with_grok_resume(
            run_initial=initial,
            find_session_id=find,
            run_resume=resume,
            classify_retryable_error=classify,
            delays=(2, 8),
            sleep=no_wait,
        )

    assert asyncio.run(scenario()) == (2, SESSION)
    assert calls[0:3] == ["initial", "find", ("sleep", 2)]
    resumes = [call for call in calls if isinstance(call, tuple) and call[0] == "resume"]
    assert len(resumes) == 2
    assert all(call[1:] == (SESSION, GROK_RESUME_PROMPT) for call in resumes)


def test_non_transport_failure_is_not_retried() -> None:
    async def scenario() -> tuple[int, str | None]:
        async def initial() -> None:
            raise CommandFailed(1, "authentication failed")

        async def classify(error: BaseException) -> bool:
            return grok_provider_stream_is_retryable(str(error))

        return await run_with_grok_resume(
            run_initial=initial,
            find_session_id=lambda: asyncio.sleep(0, result=SESSION),
            run_resume=lambda _session, _prompt: asyncio.sleep(0),
            classify_retryable_error=classify,
            delays=(0,),
        )

    with pytest.raises(CommandFailed, match="authentication failed"):
        asyncio.run(scenario())
