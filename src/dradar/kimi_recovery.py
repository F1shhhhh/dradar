"""Bounded, provider-declared recovery for an interrupted Kimi CLI turn.

This module deliberately has no Pier dependency.  DRadar copies it beside the
private Pier adapter, while unit tests exercise the exact same orchestration
without starting Docker or spending subscription quota.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable, Sequence

KIMI_RETRYABLE_EXIT_CODE = 75
KIMI_PROVIDER_CONNECTION_EXIT_CODE = 1
KIMI_RESUME_DELAYS_SECONDS = (2, 8)
KIMI_RESUME_PROMPT = (
    "Continue the unfinished task from where the previous turn stopped. "
    "Inspect the current working tree first, preserve completed work, finish "
    "the remaining implementation and tests, and commit the final result."
)

_PIER_EXIT_CODE_RE = re.compile(r"^Command failed \(exit ([0-9]+)\):")
_KIMI_PROVIDER_CONNECTION_ERROR = (
    "error: failed to run prompt: provider.connection_error: Connection error."
)


def pier_exit_code(error: BaseException) -> int | None:
    """Read Pier's stable non-zero command prefix without importing Pier."""

    match = _PIER_EXIT_CODE_RE.match(str(error))
    return int(match.group(1)) if match else None


def kimi_provider_connection_stderr_is_retryable(stderr_line: str) -> bool:
    """Accept only the pinned Kimi CLI's exact terminal error line."""

    return stderr_line.rstrip("\r\n") == _KIMI_PROVIDER_CONNECTION_ERROR


async def _failure_is_retryable(
    error: BaseException,
    classify_retryable_error: (
        Callable[[BaseException], Awaitable[bool]] | None
    ),
) -> bool:
    if pier_exit_code(error) == KIMI_RETRYABLE_EXIT_CODE:
        return True
    if classify_retryable_error is None:
        return False
    try:
        return bool(await classify_retryable_error(error))
    except Exception:
        return False


def validated_session_id(value: str | None) -> str | None:
    """Return Kimi's canonical ``session_<UUID>`` identifier.

    Kimi Code 0.39.x stores sessions in directories named
    ``session_<UUID>`` and its ``--session`` flag requires that full basename.
    Accepting a bare UUID keeps older same-turn retry logs readable, but all
    callers receive the current CLI-owned spelling.
    """

    candidate = (value or "").strip()
    raw_uuid = candidate.removeprefix("session_")
    try:
        parsed = uuid.UUID(raw_uuid)
    except (ValueError, AttributeError):
        return None
    canonical = str(parsed)
    if raw_uuid.lower() != canonical:
        return None
    return f"session_{canonical}"


def native_session_probe_command(
    session_root: str,
    session_index: str,
    session_id: str,
    *,
    expected_workdir: str = "/app",
) -> str:
    """Probe one restored session with a matching official index entry."""

    canonical = validated_session_id(session_id)
    if canonical is None:
        raise ValueError("Kimi session id is invalid")
    index_probe = """\
import json
import os
import sys

index_path, session_id, session_root, expected_workdir = sys.argv[1:]
selected_dir = None
selected_index_workdir = None
try:
    with open(index_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(entry, dict) or entry.get("sessionId") != session_id:
                continue
            if entry.get("deleted") is True:
                selected_dir = None
                selected_index_workdir = None
                continue
            session_dir = entry.get("sessionDir")
            work_dir = entry.get("workDir")
            if (
                not isinstance(session_dir, str)
                or not os.path.isabs(session_dir)
                or not isinstance(work_dir, str)
            ):
                selected_dir = None
                selected_index_workdir = None
                continue
            root = os.path.normpath(session_root)
            directory = os.path.normpath(session_dir)
            try:
                inside_root = os.path.commonpath((root, directory)) == root
            except ValueError:
                inside_root = False
            if inside_root and os.path.basename(directory) == session_id:
                selected_dir = directory
                selected_index_workdir = work_dir
            else:
                selected_dir = None
                selected_index_workdir = None
except (OSError, UnicodeError):
    selected_dir = None
    selected_index_workdir = None

wire_dirs = []
try:
    for directory, _children, _files in os.walk(session_root, followlinks=False):
        normalized = os.path.normpath(directory)
        if os.path.basename(normalized) != session_id:
            continue
        wire = os.path.join(normalized, "agents", "main", "wire.jsonl")
        if os.path.isfile(wire) and not os.path.islink(wire):
            wire_dirs.append(normalized)
except OSError:
    wire_dirs = []

state_path = (
    os.path.join(selected_dir, "state.json")
    if selected_dir is not None else ""
)
state = None
if state_path and os.path.isfile(state_path) and not os.path.islink(state_path):
    try:
        with open(state_path, encoding="utf-8") as handle:
            candidate_state = json.load(handle)
        if isinstance(candidate_state, dict):
            state = candidate_state
    except (OSError, UnicodeError, json.JSONDecodeError):
        state = None
state_workdir = state.get("workDir") if state is not None else None
effective_workdir = (
    state_workdir if isinstance(state_workdir, str) else selected_index_workdir
)
valid = (
    selected_dir is not None
    and not os.path.islink(selected_dir)
    and state is not None
    and isinstance(effective_workdir, str)
    and os.path.isabs(effective_workdir)
    and os.path.normpath(effective_workdir) == os.path.normpath(expected_workdir)
    and wire_dirs == [selected_dir]
)
if valid:
    print(session_id)
"""
    return (
        "python3 -c "
        + shlex.quote(index_probe)
        + " " + shlex.quote(session_index)
        + " " + shlex.quote(canonical)
        + " " + shlex.quote(session_root)
        + " " + shlex.quote(expected_workdir)
    )


def unique_session_probe_command(
    session_root: str, *, copy_to: str | None = None,
) -> str:
    """Return one protected session basename, refusing ambiguous state."""

    find = (
        "find " + shlex.quote(session_root)
        + " -type f -path '*/agents/main/wire.jsonl' -print 2>/dev/null"
    )
    copy = (
        "cp \"$candidate\" " + shlex.quote(copy_to) + "; "
        if copy_to is not None else ""
    )
    return (
        f"session_count=$({find} | wc -l); "
        "if [ \"$session_count\" -eq 1 ]; then "
        f"candidate=$({find}); {copy}"
        "session_dir=${candidate%/agents/main/wire.jsonl}; "
        "basename \"$session_dir\"; fi"
    )


async def run_with_kimi_resume(
    *,
    run_initial: Callable[[], Awaitable[None]],
    find_session_id: Callable[[], Awaitable[str | None]],
    run_resume: Callable[[str, str], Awaitable[None]],
    delays: Sequence[float] = KIMI_RESUME_DELAYS_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, float, str], None] | None = None,
    classify_retryable_error: (
        Callable[[BaseException], Awaitable[bool]] | None
    ) = None,
) -> tuple[int, str | None]:
    """Run once, then resume only Kimi-declared temporary failures.

    Returns ``(resume_attempts, session_id)``.  Non-retryable failures and an
    exhausted retry budget re-raise the original Pier exception.
    """

    try:
        await run_initial()
        return 0, None
    except Exception as error:
        if not await _failure_is_retryable(error, classify_retryable_error):
            raise
        last_error = error

    try:
        session_id = validated_session_id(await find_session_id())
    except Exception:  # noqa: BLE001 - recovery must not replace the run error
        session_id = None
    if session_id is None:
        raise last_error

    for attempt, delay in enumerate(delays, start=1):
        if on_retry is not None:
            on_retry(attempt, delay, session_id)
        await sleep(delay)
        try:
            await run_resume(session_id, KIMI_RESUME_PROMPT)
            return attempt, session_id
        except Exception as error:
            if not await _failure_is_retryable(error, classify_retryable_error):
                raise
            last_error = error

    raise last_error


__all__ = [
    "KIMI_RESUME_DELAYS_SECONDS",
    "KIMI_RESUME_PROMPT",
    "KIMI_PROVIDER_CONNECTION_EXIT_CODE",
    "KIMI_RETRYABLE_EXIT_CODE",
    "kimi_provider_connection_stderr_is_retryable",
    "native_session_probe_command",
    "pier_exit_code",
    "run_with_kimi_resume",
    "unique_session_probe_command",
    "validated_session_id",
]
