"""Bounded same-session recovery for official Grok Build transport failures.

This module deliberately has no Pier dependency.  The public runner copies it
beside the isolated Grok adapter, while unit tests exercise the orchestration
without starting Docker or spending subscription quota.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence

GROK_RESUME_DELAYS_SECONDS = (2, 8)
GROK_RESUME_PROMPT = (
    "Continue the unfinished task from the interrupted provider turn. "
    "Inspect the current working tree first, preserve completed work, finish "
    "the remaining analysis, persist the best valid answer, and commit it."
)

_PIER_EXIT_CODE_RE = re.compile(r"^Command failed \(exit ([0-9]+)\):")
_GROK_RESPONSE_HOST = "cli-chat-proxy.grok.com/v1/responses"
_GROK_STREAM_ERRORS = (
    "reqwest error stream: error sending request for url",
    "stream disconnected before completion",
    "connection reset by peer",
)


def pier_exit_code(error: BaseException) -> int | None:
    """Read Pier's stable non-zero command prefix without importing Pier."""

    match = _PIER_EXIT_CODE_RE.match(str(error))
    return int(match.group(1)) if match else None


def grok_provider_stream_is_retryable(output_tail: str) -> bool:
    """Accept only a Grok response-stream failure from the official host."""

    low = output_tail.lower()
    return (
        _GROK_RESPONSE_HOST in low
        and any(marker in low for marker in _GROK_STREAM_ERRORS)
    )


def validated_session_id(value: str | None) -> str | None:
    """Accept only one canonical UUID emitted by Grok's init event."""

    candidate = (value or "").strip()
    try:
        parsed = uuid.UUID(candidate)
    except (ValueError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if candidate.lower() == canonical else None


async def run_with_grok_resume(
    *,
    run_initial: Callable[[], Awaitable[None]],
    find_session_id: Callable[[], Awaitable[str | None]],
    run_resume: Callable[[str, str], Awaitable[None]],
    classify_retryable_error: Callable[[BaseException], Awaitable[bool]],
    delays: Sequence[float] = GROK_RESUME_DELAYS_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> tuple[int, str | None]:
    """Run once, then resume only an allowlisted transient stream failure."""

    try:
        await run_initial()
        return 0, None
    except Exception as error:
        if not await classify_retryable_error(error):
            raise
        last_error = error

    try:
        session_id = validated_session_id(await find_session_id())
    except Exception:  # noqa: BLE001 - recovery must preserve the run error
        session_id = None
    if session_id is None:
        raise last_error

    for attempt, delay in enumerate(delays, start=1):
        if on_retry is not None:
            on_retry(attempt, delay, session_id)
        await sleep(delay)
        try:
            await run_resume(session_id, GROK_RESUME_PROMPT)
            return attempt, session_id
        except Exception as error:
            if not await classify_retryable_error(error):
                raise
            last_error = error

    raise last_error


__all__ = [
    "GROK_RESUME_DELAYS_SECONDS",
    "GROK_RESUME_PROMPT",
    "grok_provider_stream_is_retryable",
    "pier_exit_code",
    "run_with_grok_resume",
    "validated_session_id",
]
