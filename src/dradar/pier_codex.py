"""Durable checkpoint adapter for the stock OpenAI Codex Pier agent.

The upstream Codex implementation publishes its checkpoint directly below
``/logs/agent``.  That directory is writable by the task container, so it is
not a trustworthy publication boundary.  This subclass keeps the upstream
model command, prompt handling, retry behaviour, and trajectory conversion,
but delegates checkpoint capture and restore to DRadar's host-published
``DurableCheckpoint``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Any, Iterable

from pier.agents.installed.codex import Codex
from pier.agents.installed.base import NonZeroAgentExitCodeError, with_prompt_template
from pier.environments.base import BaseEnvironment
from pier.models.trial.paths import EnvironmentPaths

try:
    from _dradar_pier_checkpoint import (
        CheckpointError,
        DurableCheckpoint,
        StatePath,
        _snapshot_payload_dir,
    )
except ModuleNotFoundError as exc:  # Source-tree unit tests.
    if exc.name != "_dradar_pier_checkpoint":
        raise
    from dradar.pier_checkpoint import (
        CheckpointError,
        DurableCheckpoint,
        StatePath,
        _snapshot_payload_dir,
    )


_ROOT_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
_UPSTREAM_HAS_CHECKPOINT_HOOKS = all(
    hasattr(Codex, name)
    for name in ("_start_checkpoint", "_finish_checkpoint", "_run_with_capacity_resume")
)
_MAX_AUTH_JSON_BYTES = 4 * 1024 * 1024
_MAX_CODEX_THREAD_PREFIX_BYTES = 1024 * 1024
_SESSION_UNAVAILABLE_MARKERS = (
    b"session not found",
    b"conversation not found",
    b"no rollout found",
    b"failed to load session",
    b"could not find session",
)
_SESSION_ERROR_EVENT_TYPES = frozenset({"error", "turn.failed"})
_NON_JSON_ERROR_PREFIX_RE = re.compile(
    rb"^(?:error|fatal|codex(?:\s+cli)?\s+error)\s*:",
    re.IGNORECASE,
)
_CODEX_SESSION_PROBE_PROGRAM = r"""
const fs = require("fs");
const path = require("path");

const outputPath = process.argv[1];
const sessionsRoot = process.argv[2];
const idPattern = /^[A-Za-z0-9._:-]{8,160}$/;
const maxPrefixBytes = 1024 * 1024;
const maxRollouts = 256;
const maxDepth = 8;

function readPrefix(filename) {
  let descriptor;
  try {
    descriptor = fs.openSync(
      filename,
      fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0),
    );
    const metadata = fs.fstatSync(descriptor);
    if (!metadata.isFile() || metadata.nlink !== 1) return null;
    const buffer = Buffer.alloc(maxPrefixBytes);
    const length = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
    return buffer.subarray(0, length).toString("utf8");
  } catch (_error) {
    return null;
  } finally {
    if (descriptor !== undefined) {
      try { fs.closeSync(descriptor); } catch (_error) {}
    }
  }
}

function jsonlEvents(filename) {
  const prefix = readPrefix(filename);
  if (prefix === null) return [];
  const events = [];
  for (const line of prefix.split(/\r?\n/)) {
    try {
      const event = JSON.parse(line);
      if (event && typeof event === "object" && !Array.isArray(event)) {
        events.push(event);
      }
    } catch (_error) {}
  }
  return events;
}

for (const event of jsonlEvents(outputPath)) {
  if (
    event.type === "thread.started"
    && typeof event.thread_id === "string"
    && idPattern.test(event.thread_id)
  ) {
    process.stdout.write(`${event.thread_id}\n`);
    process.exit(0);
  }
}

const pending = [{directory: sessionsRoot, depth: 0}];
const rollouts = [];
while (pending.length > 0 && rollouts.length <= maxRollouts) {
  const {directory, depth} = pending.pop();
  let entries;
  try {
    entries = fs.readdirSync(directory, {withFileTypes: true});
  } catch (_error) {
    continue;
  }
  for (const entry of entries) {
    if (entry.isSymbolicLink()) continue;
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (depth < maxDepth) pending.push({directory: candidate, depth: depth + 1});
    } else if (entry.isFile() && /^rollout-.*\.jsonl$/.test(entry.name)) {
      rollouts.push(candidate);
      if (rollouts.length > maxRollouts) break;
    }
  }
}

if (rollouts.length <= maxRollouts) {
  const roots = new Set();
  for (const rollout of rollouts) {
    for (const event of jsonlEvents(rollout)) {
      if (event.type !== "session_meta") continue;
      const payload = event.payload;
      const id = payload && payload.id;
      if (
        payload
        && payload.source === "exec"
        && typeof id === "string"
        && idPattern.test(id)
        && path.basename(rollout).endsWith(`-${id}.jsonl`)
      ) {
        roots.add(id);
      }
      break;
    }
  }
  if (roots.size === 1) process.stdout.write(`${roots.values().next().value}\n`);
}
""".strip()


def _codex_session_probe_command(
    output_path: str = "/logs/agent/codex.txt",
    sessions_root: str = "/tmp/codex-home/sessions",
) -> str:
    """Return a fail-closed probe for the native root Codex thread.

    ``codex.txt`` is authoritative because its first ``thread.started`` event
    belongs to the root ``codex exec`` invocation.  Before that event is
    flushed (or after a very early crash), session metadata is a safe fallback
    only when it identifies exactly one ``source=exec`` rollout.  Subagent
    rollouts have an object-valued source and can never win by filename order.
    """

    return " ".join(
        (
            "node",
            "-e",
            shlex.quote(_CODEX_SESSION_PROBE_PROGRAM),
            shlex.quote(output_path),
            shlex.quote(sessions_root),
        )
    )


_CODEX_SESSION_PROBE = _codex_session_probe_command()


def _iter_string_values(value: Any) -> Iterable[str]:
    """Yield every JSON string leaf without retaining field names."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_string_values(child)


def _auth_sensitive_values(path: Path) -> tuple[str, ...]:
    """Read one bounded, plain auth document and return its string leaves.

    The values are passed only to the checkpoint scanner.  They are never
    included in an exception, manifest, event, command, or log message.
    """

    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_AUTH_JSON_BYTES
        ):
            raise ValueError
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino)
                != (metadata.st_dev, metadata.st_ino)
            ):
                raise ValueError
            raw = b""
            while len(raw) <= _MAX_AUTH_JSON_BYTES:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, _MAX_AUTH_JSON_BYTES + 1 - len(raw)),
                )
                if not chunk:
                    break
                raw += chunk
            if len(raw) > _MAX_AUTH_JSON_BYTES:
                raise ValueError
            after = os.fstat(descriptor)
            if (
                after.st_size != opened.st_size
                or after.st_mtime_ns != opened.st_mtime_ns
                or after.st_ctime_ns != opened.st_ctime_ns
            ):
                raise ValueError
        finally:
            os.close(descriptor)
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Codex auth JSON is invalid or unreadable") from exc
    # Preserve deterministic order while removing duplicates. DurableCheckpoint
    # applies its own minimum-length threshold before scanning artifacts.
    return tuple(dict.fromkeys(_iter_string_values(payload)))


def _same_stable_regular_file(
    candidate: os.stat_result,
    reference: os.stat_result,
) -> bool:
    """Require one unchanged, singly-linked regular-file identity."""

    return (
        stat.S_ISREG(candidate.st_mode)
        and candidate.st_nlink == 1
        and (candidate.st_dev, candidate.st_ino)
        == (reference.st_dev, reference.st_ino)
        and candidate.st_size == reference.st_size
        and candidate.st_mtime_ns == reference.st_mtime_ns
        and candidate.st_ctime_ns == reference.st_ctime_ns
    )


def _read_stable_regular_file_edges(
    path: Path,
    *,
    prefix_bytes: int,
    tail_bytes: int = 0,
) -> tuple[bytes, bytes, bool] | None:
    """Read bounded file edges without blocking on a raced special file.

    The task container can replace ``codex.txt`` between ``lstat`` and
    ``open``.  ``O_NONBLOCK`` keeps a regular-to-FIFO swap from hanging the
    host finalizer, while the descriptor and path checks reject every changed
    identity or size.  The boolean result records an omitted middle section.
    """

    if prefix_bytes < 0 or tail_bytes < 0:
        raise ValueError("bounded Codex output sizes must be non-negative")
    try:
        observed = path.lstat()
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            return None
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not _same_stable_regular_file(opened, observed):
                return None

            total_edge_bytes = prefix_bytes + tail_bytes
            if tail_bytes and opened.st_size <= total_edge_bytes:
                prefix_length = opened.st_size
                tail_length = 0
            else:
                prefix_length = min(opened.st_size, prefix_bytes)
                tail_length = min(
                    max(0, opened.st_size - prefix_length), tail_bytes,
                )

            prefix = b""
            while len(prefix) < prefix_length:
                chunk = os.read(descriptor, prefix_length - len(prefix))
                if not chunk:
                    return None
                prefix += chunk

            tail = b""
            if tail_length:
                os.lseek(descriptor, opened.st_size - tail_length, os.SEEK_SET)
                while len(tail) < tail_length:
                    chunk = os.read(descriptor, tail_length - len(tail))
                    if not chunk:
                        return None
                    tail += chunk

            after = os.fstat(descriptor)
            if not _same_stable_regular_file(after, opened):
                return None
            path_after = path.lstat()
            if not _same_stable_regular_file(path_after, opened):
                return None
        finally:
            os.close(descriptor)
    except OSError:
        return None
    return prefix, tail, opened.st_size > len(prefix) + len(tail)


def _root_thread_id_from_output(path: Path) -> str | None:
    """Read Codex's initial JSONL prefix without following a model-made link."""

    captured = _read_stable_regular_file_edges(
        path, prefix_bytes=_MAX_CODEX_THREAD_PREFIX_BYTES,
    )
    if captured is None:
        return None
    raw, _tail, _omitted_middle = captured
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and _ROOT_THREAD_ID_RE.fullmatch(thread_id):
            return thread_id
    return None


def _line_reports_codex_session_unavailable(line: bytes) -> bool:
    """Classify one complete Codex JSONL/stderr line without reading prose."""

    stripped = line.strip()
    if not stripped:
        return False
    try:
        event = json.loads(stripped)
    except (UnicodeDecodeError, json.JSONDecodeError):
        lowered = stripped.lower()
        return bool(_NON_JSON_ERROR_PREFIX_RE.match(stripped)) and any(
            marker in lowered for marker in _SESSION_UNAVAILABLE_MARKERS
        )
    if (
        not isinstance(event, dict)
        or event.get("type") not in _SESSION_ERROR_EVENT_TYPES
    ):
        return False
    lowered = "\n".join(_iter_string_values(event)).encode(
        "utf-8", errors="ignore",
    ).lower()
    return any(marker in lowered for marker in _SESSION_UNAVAILABLE_MARKERS)


def _codex_session_unavailable(path: Path) -> bool:
    """Recognize only explicit native-session loss from a bounded plain file."""

    captured = _read_stable_regular_file_edges(
        path,
        prefix_bytes=_MAX_CODEX_THREAD_PREFIX_BYTES,
        tail_bytes=_MAX_CODEX_THREAD_PREFIX_BYTES,
    )
    if captured is None:
        return False
    prefix, tail, omitted_middle = captured
    if omitted_middle:
        # Neither edge may turn a truncated JSON object into a synthetic raw
        # stderr record. Only complete lines remain eligible for matching.
        prefix = prefix.rsplit(b"\n", 1)[0] if b"\n" in prefix else b""
        tail = tail.split(b"\n", 1)[1] if b"\n" in tail else b""
    return any(
        _line_reports_codex_session_unavailable(line)
        for section in (prefix, tail)
        for line in section.splitlines()
    )


def _rewrite_codex_model_prompt(command: str, instruction: str) -> str:
    """Keep every fresh/resume/fallback model call on the exact same prompt."""

    marker = " 2>&1 </dev/null | tee "
    if "codex exec " not in command or marker not in command:
        return command
    model_command, tee_target = command.rsplit(marker, 1)
    try:
        tokens = shlex.split(model_command)
    except ValueError:
        return command
    if not tokens:
        return command
    previous_prompt = shlex.quote(tokens[-1])
    if not model_command.endswith(previous_prompt):
        return command
    return (
        model_command[:-len(previous_prompt)]
        + shlex.quote(instruction)
        + marker
        + tee_target
    )


class DurableCodex(Codex):
    """Stock OpenAI Codex with DRadar's host-private checkpoint lifecycle."""

    _CHECKPOINT_PROVIDER = "openai"
    _CHECKPOINT_HARNESS = "codex"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Public Pier 0.3.0 accepts arbitrary agent kwargs in BaseAgent but
        # does not retain the checkpoint extension fields.  ds0's pinned
        # post3 build does retain them.  Sample the public CLI kwargs before
        # delegation, then fill only attributes the upstream build omitted.
        checkpoint_enabled = kwargs.get("checkpoint_enabled", False)
        checkpoint_assignment_id = kwargs.get("checkpoint_assignment_id")
        checkpoint_task_id = kwargs.get("checkpoint_task_id")
        checkpoint_effort = kwargs.get("checkpoint_effort")
        checkpoint_resume_generation = kwargs.get(
            "checkpoint_resume_generation", 0,
        )
        checkpoint_path = kwargs.get("checkpoint_path")
        checkpoint_interval_sec = kwargs.get("checkpoint_interval_sec", 30)
        checkpoint_workdir = kwargs.get("checkpoint_workdir", "/app")
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_checkpoint_enabled"):
            self._checkpoint_enabled = checkpoint_enabled
        if not hasattr(self, "_checkpoint_assignment_id"):
            self._checkpoint_assignment_id = checkpoint_assignment_id
        if not hasattr(self, "_checkpoint_task_id"):
            self._checkpoint_task_id = checkpoint_task_id
        if not hasattr(self, "_checkpoint_effort"):
            self._checkpoint_effort = checkpoint_effort
        if not hasattr(self, "_checkpoint_resume_generation"):
            self._checkpoint_resume_generation = checkpoint_resume_generation
        if not hasattr(self, "_resume_checkpoint_path"):
            self._resume_checkpoint_path = (
                Path(str(checkpoint_path)) if checkpoint_path else None
            )
        if not hasattr(self, "_checkpoint_interval_sec"):
            self._checkpoint_interval_sec = checkpoint_interval_sec
        if not hasattr(self, "_checkpoint_workdir"):
            self._checkpoint_workdir = checkpoint_workdir
        if not hasattr(self, "_checkpoint_manifest_path"):
            self._checkpoint_manifest_path = None

        sensitive_values: list[str] = []
        auth_json_path = self._resolve_auth_json_path()
        if auth_json_path is not None:
            sensitive_values.extend(_auth_sensitive_values(auth_json_path))
        api_key = self._get_env("OPENAI_API_KEY")
        if isinstance(api_key, str) and api_key:
            sensitive_values.append(api_key)

        self._durable_checkpoint = DurableCheckpoint(
            logs_dir=self.logs_dir,
            enabled=self._checkpoint_enabled,
            assignment_id=self._checkpoint_assignment_id,
            task_id=self._checkpoint_task_id,
            model=self.model_name,
            effort=self._checkpoint_effort,
            resume_generation=self._checkpoint_resume_generation,
            checkpoint_path=(
                str(self._resume_checkpoint_path)
                if self._resume_checkpoint_path is not None
                else None
            ),
            harness=self._CHECKPOINT_HARNESS,
            provider=self._CHECKPOINT_PROVIDER,
            agent_version=self._version or "unknown",
            interval_sec=self._checkpoint_interval_sec,
            workdir=self._checkpoint_workdir,
            state_paths=(
                StatePath(
                    "sessions",
                    (self._REMOTE_CODEX_HOME / "sessions").as_posix(),
                ),
            ),
            sensitive_values=sensitive_values,
            session_probe=_CODEX_SESSION_PROBE,
        )
        self._checkpoint_finalizer_task: asyncio.Task[None] | None = None
        self._upstream_phase = "idle"
        self._public_checkpoint_run_active = False
        self._public_resume_root: str | None = None
        self._public_resume_pending = False
        self._checkpoint_runtime_environment: BaseEnvironment | None = None
        self._checkpoint_runtime_env: dict[str, str] | None = None
        self._rendered_benchmark_instruction: str | None = None
        if self._checkpoint_enabled:
            self._extra_env.setdefault("HOME", "/tmp/dradar-agent-home")

    def _session_unavailable(self) -> bool:
        """Use DRadar's structured classifier on checkpoint-hook builds too.

        The post3 upstream hook calls this virtual method after a failed
        native resume.  Its prose-wide implementation can mistake an ordinary
        agent answer containing words such as ``session not found`` for loss
        of the native thread, delete the restored sessions, and start a second
        paid fresh turn.  Public Pier uses the explicit bridge below, but the
        override keeps both execution paths on the same strict contract.
        """

        return _codex_session_unavailable(
            self.logs_dir / self._OUTPUT_FILENAME,
        )

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        """Bridge public Pier's stock run without changing its fresh prompt.

        Public Pier 0.3.0 has no checkpoint hooks.  During that one code path,
        replace only the already-rendered model command with native ``resume``
        when a root thread exists, and defer CODEX_HOME deletion until the
        durable final snapshot has captured its sessions.  All setup, flags,
        prompt templating, logging and trajectory behaviour remain upstream.
        """

        fresh_fallback_command: str | None = None
        if self._public_checkpoint_run_active:
            remote_secrets = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
            cleanup = f'rm -rf {shlex.quote(remote_secrets)} "$CODEX_HOME"'
            if command.strip() == cleanup:
                # The public parent ignores this result. Execute a harmless
                # command through the same environment instead of fabricating
                # an upstream result object.
                return await super().exec_as_agent(
                    environment,
                    "true",
                    env=env,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                )

            root = self._public_resume_root
            if (
                self._public_resume_pending
                and root is not None
                and "codex exec " in command
                and "-- " in command
            ):
                fresh_fallback_command = command
                prefix, rendered = command.split("codex exec ", 1)
                options, _fresh_instruction = rendered.split("-- ", 1)
                instruction = self._rendered_benchmark_instruction
                if instruction is None:
                    raise CheckpointError(
                        "rendered benchmark instruction is unavailable for resume",
                    )
                output_path = (
                    EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME
                ).as_posix()
                command = (
                    f"{prefix}codex exec resume {options}"
                    f"{shlex.quote(root)} {shlex.quote(instruction)} "
                    f"2>&1 </dev/null | tee {shlex.quote(output_path)}"
                )
                self._public_resume_pending = False

        if self._rendered_benchmark_instruction is not None:
            command = _rewrite_codex_model_prompt(
                command, self._rendered_benchmark_instruction,
            )
        try:
            return await super().exec_as_agent(
                environment,
                command,
                env=env,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
        except NonZeroAgentExitCodeError:
            if (
                fresh_fallback_command is None
                or not _codex_session_unavailable(
                    self.logs_dir / self._OUTPUT_FILENAME,
                )
            ):
                raise
            self._checkpoint_event(
                "session_resume_degraded",
                reason="session_unavailable",
                progress_summary_available=False,
            )
            await super().exec_as_agent(
                environment,
                'rm -rf "$CODEX_HOME/sessions"',
                env=env,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
            if self._rendered_benchmark_instruction is not None:
                fresh_fallback_command = _rewrite_codex_model_prompt(
                    fresh_fallback_command,
                    self._rendered_benchmark_instruction,
                )
            return await super().exec_as_agent(
                environment,
                fresh_fallback_command,
                env=env,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )

    async def exec_as_root(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        """Keep checkpoint-enabled root maintenance free of task env hooks."""

        if not self._durable_checkpoint.enabled:
            return await super().exec_as_root(
                environment,
                command,
                env=env,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
        if cwd not in (None, "/"):
            raise CheckpointError("checkpoint root maintenance cwd is unsafe")
        return await self._durable_checkpoint.exec_root_maintenance(
            environment,
            command,
            timeout_sec=timeout_sec or 120,
        )

    @staticmethod
    async def _reap_cancelled_child(
        task: asyncio.Task[Any],
    ) -> BaseException | None:
        """Wait through repeated caller cancellation until ``task`` is done."""

        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.cancelled():
            return None
        try:
            return task.exception()
        except asyncio.CancelledError:
            return None

    async def _cleanup_checkpoint_runtime_state(
        self,
        environment: BaseEnvironment,
        env: dict[str, str],
    ) -> None:
        """Remove credential staging and native state after final capture."""

        cleanup = (
            f"rm -rf {shlex.quote(self._REMOTE_CODEX_SECRETS_DIR.as_posix())} "
            '"$CODEX_HOME"'
        )
        await super().exec_as_agent(environment, cleanup, env=env)

    @with_prompt_template
    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        """Delegate the unchanged upstream turn while making cancellation durable.

        Pier's current Codex ``finally`` shields checkpoint finalization but
        catches the resulting ``CancelledError``.  A cancellation that lands
        during the final snapshot can therefore make the upstream coroutine
        return while its shield-created writer is still running.  Isolating
        the upstream coroutine lets us cancel the model turn, then explicitly
        reap both it and the tracked finalizer before propagating cancellation.
        Prompt expansion and every model command remain upstream-owned.
        """

        self._rendered_benchmark_instruction = instruction
        if self._durable_checkpoint.enabled:
            boundary_env = self.build_process_env({
                "CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix(),
                "HOME": "/tmp/dradar-agent-home",
            })
            await self._durable_checkpoint.prepare_agent_environment(
                self, environment, boundary_env,
            )

        self._checkpoint_finalizer_task = None
        self._upstream_phase = "setup"
        parent_run = super().run
        unwrapped_parent_run = getattr(parent_run, "__wrapped__", None)
        upstream_run = (
            (
                unwrapped_parent_run(self, instruction, environment, context)
                if unwrapped_parent_run is not None
                else parent_run(instruction, environment, context)
            )
            if _UPSTREAM_HAS_CHECKPOINT_HOOKS
            else self._run_public_pier(instruction, environment, context)
        )
        upstream = asyncio.create_task(
            upstream_run,
            name="dradar-codex-upstream-run",
        )
        try:
            await asyncio.shield(upstream)
        except asyncio.CancelledError as cancellation:
            if self._upstream_phase == "post-start":
                # There is no await between upstream checkpoint start and its
                # model helper. Give it a scheduling boundary to declare the
                # model phase; cancelling the subsequent post-model finally
                # would otherwise skip or race checkpoint finalization.
                for _attempt in range(3):
                    sleeper = asyncio.create_task(asyncio.sleep(0))
                    await self._reap_cancelled_child(sleeper)
                    if self._upstream_phase != "post-start" or upstream.done():
                        break
                if self._upstream_phase == "post-start" and not upstream.done():
                    # start() has already launched the periodic writer, but
                    # upstream has not yet entered its model/finally region.
                    # Leaving this child alive would ignore user cancellation
                    # and can continue consuming paid quota indefinitely.
                    upstream.cancel()
            if self._upstream_phase in {"setup", "checkpoint-start", "model"}:
                upstream.cancel()
            upstream_error = await self._reap_cancelled_child(upstream)
            finalizer = self._checkpoint_finalizer_task
            if (
                finalizer is None
                and self._durable_checkpoint.manifest_path is not None
                and self._checkpoint_runtime_environment is not None
                and self._checkpoint_runtime_env is not None
            ):
                # Cancellation can land after DurableCheckpoint.start() but
                # before the upstream harness installs its own finally block.
                # Close that exact checkpoint explicitly after the upstream
                # child has been reaped, so there is no concurrent model
                # process left to mutate staging.
                finalizer = self._finish_checkpoint(
                    self._checkpoint_runtime_environment,
                    self._checkpoint_runtime_env,
                    completed=False,
                    failure=cancellation,
                )
            finalizer_error: BaseException | None = None
            if finalizer is not None and finalizer is not asyncio.current_task():
                finalizer_error = await self._reap_cancelled_child(finalizer)
            if upstream_error is not None:
                self.logger.warning(
                    "Codex upstream failed while cancellation was being reaped",
                    exc_info=(
                        type(upstream_error), upstream_error,
                        upstream_error.__traceback__,
                    ),
                )
            if finalizer_error is not None:
                self.logger.warning(
                    "Codex checkpoint finalizer failed during cancellation",
                    exc_info=(
                        type(finalizer_error), finalizer_error,
                        finalizer_error.__traceback__,
                    ),
                )
            cleanup_error: BaseException | None = None
            runtime_environment = self._checkpoint_runtime_environment
            runtime_env = self._checkpoint_runtime_env
            if runtime_environment is not None and runtime_env is not None:
                # The upstream cleanup may not have been installed when the
                # post-start window is cancelled. Run the same idempotent
                # cleanup only after final capture, and reap it through any
                # repeated caller cancellation so credentials/CODEX_HOME are
                # not stranded even if checkpoint finalization itself failed.
                cleanup_task = asyncio.create_task(
                    self._cleanup_checkpoint_runtime_state(
                        runtime_environment, runtime_env,
                    ),
                    name="dradar-codex-cancellation-cleanup",
                )
                cleanup_error = await self._reap_cancelled_child(cleanup_task)
            if cleanup_error is not None:
                self.logger.warning(
                    "Codex runtime cleanup failed during cancellation",
                    exc_info=(
                        type(cleanup_error), cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
            self._checkpoint_runtime_environment = None
            self._checkpoint_runtime_env = None
            self._rendered_benchmark_instruction = None
            raise cancellation
        self._checkpoint_runtime_environment = None
        self._checkpoint_runtime_env = None
        self._rendered_benchmark_instruction = None

    async def _run_public_pier(
        self, instruction: str, environment: BaseEnvironment, context: Any,
    ) -> None:
        """Add the durable lifecycle around unmodified public Pier 0.3.0."""

        env = self.build_process_env(
            {"CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix()},
        )
        completed = False
        failure: BaseException | None = None
        self._public_checkpoint_run_active = True
        try:
            previous, root = await self._start_checkpoint(environment, env)
            del previous
            self._public_resume_root = root
            self._public_resume_pending = root is not None
            self._upstream_phase = "model"
            try:
                # This remains the public decorated method, so the benchmark
                # instruction and any configured prompt template are rendered
                # exactly once by Pier.
                parent_run = super().run
                unwrapped_parent_run = getattr(parent_run, "__wrapped__", None)
                if unwrapped_parent_run is not None:
                    await unwrapped_parent_run(
                        self, instruction, environment, context,
                    )
                else:
                    await parent_run(instruction, environment, context)
                completed = True
            except BaseException as exc:
                failure = exc
                raise
            finally:
                self._upstream_phase = "post-model"
                finalizer = self._finish_checkpoint(
                    environment,
                    env,
                    completed=completed,
                    failure=failure,
                )
                await asyncio.shield(finalizer)
        finally:
            self._public_resume_root = None
            self._public_resume_pending = False
            self._public_checkpoint_run_active = False
            cleanup = (
                f"rm -rf {shlex.quote(self._REMOTE_CODEX_SECRETS_DIR.as_posix())} "
                '"$CODEX_HOME"'
            )
            try:
                await super().exec_as_agent(
                    environment, cleanup, env=env,
                )
            except Exception:
                self.logger.warning(
                    "Could not clean up deferred public Codex state",
                    exc_info=True,
                )

    async def _run_with_capacity_resume(self, *args: Any, **kwargs: Any) -> None:
        """Track only the upstream model-execution await; do not alter it."""

        self._upstream_phase = "model"
        try:
            await super()._run_with_capacity_resume(*args, **kwargs)
        finally:
            self._upstream_phase = "post-model"

    @property
    def _checkpoint_host_dir(self) -> Path:
        checkpoint = getattr(self, "_durable_checkpoint", None)
        if checkpoint is not None:
            return checkpoint.host_dir
        # ``Codex.__init__`` does not currently access this property, but the
        # fallback keeps construction safe across compatible upstream builds.
        return self.logs_dir.parent / "checkpoint"

    def _checkpoint_event(self, event: str, **detail: Any) -> None:
        checkpoint = getattr(self, "_durable_checkpoint", None)
        if checkpoint is not None:
            checkpoint._event(event, **detail)

    def _checkpoint_root_thread_id(self, checkpoint_dir: Path) -> str | None:
        """Map a published generation's native session id to Codex's root id."""

        try:
            payload = _snapshot_payload_dir(checkpoint_dir)
            sidecar = payload / "session-id"
            metadata = sidecar.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > 512
            ):
                return None
            value = sidecar.read_text(
                encoding="utf-8", errors="strict",
            ).strip()
        except (CheckpointError, OSError, UnicodeDecodeError, ValueError):
            return None
        return value if _ROOT_THREAD_ID_RE.fullmatch(value) else None

    async def _start_checkpoint(
        self, environment: BaseEnvironment, env: dict[str, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        self._upstream_phase = "checkpoint-start"
        self._checkpoint_runtime_environment = environment
        self._checkpoint_runtime_env = dict(env)
        root_thread_id = await self._durable_checkpoint.start(
            self, environment, env,
        )
        self._upstream_phase = "post-start"
        self._checkpoint_manifest_path = self._durable_checkpoint.manifest_path
        previous = self._durable_checkpoint.previous
        previous_dir = self._durable_checkpoint.previous_dir
        if previous is None or previous_dir is None:
            return None, None

        payload = _snapshot_payload_dir(previous_dir)
        sessions_dir = payload / "provider-state" / "sessions"
        # A checkpoint taken before Codex created a native session still has a
        # useful restored workspace. Returning it as a fresh run preserves the
        # exact benchmark instruction and avoids upstream's recovery prompt.
        if root_thread_id is None or not sessions_dir.is_dir():
            return None, None

        # Upstream Codex verifies this path immediately before invoking
        # ``codex exec resume``. Point it at the immutable published payload,
        # never at the container-visible staging tree.
        self._resume_checkpoint_path = payload
        compatible = dict(previous)
        compatible["sessions_dir"] = "provider-state/sessions"
        compatible["root_thread_id"] = root_thread_id
        return compatible, root_thread_id

    def _finish_checkpoint(
        self,
        environment: BaseEnvironment,
        env: dict[str, str],
        *,
        completed: bool,
        failure: BaseException | None,
    ) -> asyncio.Task[None]:
        self._upstream_phase = "finalizer"

        async def finalize() -> None:
            try:
                root_thread_id = _root_thread_id_from_output(
                    self.logs_dir / self._OUTPUT_FILENAME,
                )
                await self._durable_checkpoint.finish(
                    self,
                    environment,
                    env,
                    completed=completed,
                    failure=failure,
                    session_id=root_thread_id,
                )
                self._checkpoint_manifest_path = (
                    self._durable_checkpoint.manifest_path
                )
            finally:
                self._upstream_phase = "post-finalizer"

        task = asyncio.create_task(
            finalize(), name="dradar-codex-checkpoint-finalizer",
        )
        self._checkpoint_finalizer_task = task
        return task


__all__ = ["DurableCodex"]
