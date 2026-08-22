"""Provider-neutral, credential-free checkpoints for paid Pier harnesses.

The checkpoint is written below ``/logs/agent`` (Pier's host bind mount).  It
contains repository progress plus only the provider session paths explicitly
allowlisted by an adapter.  Credentials are injected again by DRadar and are
never copied into the checkpoint or named in its manifest.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
_SENSITIVE_KEY_PARTS = (
    "token", "secret", "password", "credential", "api_key", "auth",
)
_FIXED_ARTIFACTS = {
    "workspace_patch": "workspace.patch",
    "untracked_archive": "untracked.tar.gz",
    "state_dir": "provider-state",
    "events_file": "events.jsonl",
}


class CheckpointError(RuntimeError):
    """A checkpoint is corrupt or cannot be safely restored."""


class CheckpointIncompatibleError(CheckpointError):
    """A valid checkpoint belongs to a different runtime identity."""


@dataclass(frozen=True)
class StatePath:
    """One credential-free provider path copied into the checkpoint."""

    name: str
    remote_path: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", self.name):
            raise ValueError("checkpoint state name is invalid")
        if not self.remote_path.startswith("/"):
            raise ValueError("checkpoint state path must be absolute")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(child) for child in value)
    return False


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    if _contains_sensitive_key(data):
        raise CheckpointError("checkpoint manifest contains a sensitive field name")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("checkpoint manifest is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointIncompatibleError("checkpoint schema is unsupported")
    if _contains_sensitive_key(value):
        raise CheckpointError("checkpoint manifest contains a sensitive field name")
    checkpoint_id = value.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or _ID_RE.fullmatch(checkpoint_id) is None:
        raise CheckpointError("checkpoint id is invalid")
    for name in ("assignment_id", "phase", "created_at"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise CheckpointError(f"checkpoint is missing {name}")
    generation = value.get("resume_generation", 0)
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise CheckpointError("checkpoint generation is invalid")
    for name, expected in _FIXED_ARTIFACTS.items():
        if value.get(name, expected) != expected:
            raise CheckpointError(f"checkpoint {name} path is unsafe")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CheckpointError("checkpoint artifact path is invalid")
    if root.is_symlink():
        raise CheckpointError("checkpoint root is a symlink")
    canonical_root = root.resolve()
    path = root / relative
    if path.is_symlink():
        raise CheckpointError("checkpoint artifact is a symlink")
    canonical = path.resolve()
    if canonical_root not in canonical.parents:
        raise CheckpointError("checkpoint artifact escaped its directory")
    if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
        raise CheckpointError("checkpoint state contains a symlink")
    return path


def _stream_contains_any(handle: Any, needles: tuple[bytes, ...]) -> bool:
    if not needles:
        return False
    overlap = max(len(value) for value in needles) - 1
    previous = b""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            return False
        combined = previous + chunk
        if any(value in combined for value in needles):
            return True
        previous = combined[-overlap:] if overlap else b""


def _validate_archive(path: Path, needles: tuple[bytes, ...]) -> bool:
    """Validate a generated untracked archive and scan its regular files."""

    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    raise CheckpointError("checkpoint archive contains an unsafe member")
                if member.isfile():
                    source = archive.extractfile(member)
                    if source is None:
                        raise CheckpointError("checkpoint archive member is unreadable")
                    with source:
                        if _stream_contains_any(source, needles):
                            return True
    except (OSError, tarfile.TarError) as exc:
        raise CheckpointError("checkpoint archive is unreadable") from exc
    return False


def _path_contains_any(path: Path, needles: tuple[bytes, ...]) -> bool:
    candidates = path.rglob("*") if path.is_dir() else (path,)
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            with candidate.open("rb") as handle:
                if _stream_contains_any(handle, needles):
                    return True
    return False


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("checkpoint_enabled must be a boolean")


def _snapshot_script(
    *, checkpoint_dir: str, workdir: str, interval_sec: int,
    state_paths: Iterable[StatePath], session_probe: str | None,
) -> str:
    """Build the periodic in-container snapshotter.

    Provider state is copied only from adapter-declared paths.  A broad secret
    signature invalidates the provider-state copy while retaining the workspace
    snapshot, allowing a safe same-prompt fallback without persisting secrets.
    """

    interval = max(10, min(int(interval_sec), 300))
    copy_lines: list[str] = []
    for item in state_paths:
        source = shlex.quote(item.remote_path)
        target = shlex.quote(item.name)
        copy_lines.extend([
            f"  if [ -d {source} ] && [ ! -L {source} ]; then "
            f"cp -R {source} \"$state_tmp\"/{target};",
            f"  elif [ -f {source} ] && [ ! -L {source} ]; then cp {source} \"$state_tmp\"/{target}; fi",
        ])
    probe = ""
    if session_probe:
        probe = f"""
  session_id=$({session_probe} 2>/dev/null || true)
  case "$session_id" in
    ''|*[!A-Za-z0-9._:-]*) ;;
    *)
      if [ "${{#session_id}}" -ge 8 ] && [ "${{#session_id}}" -le 160 ]; then
        printf '%s\\n' "$session_id" > "$checkpoint_dir/session-id.tmp"
        mv "$checkpoint_dir/session-id.tmp" "$checkpoint_dir/session-id"
      fi ;;
  esac
"""
    copies = "\n".join(copy_lines)
    return f"""#!/bin/sh
set -eu
umask 077
workdir={shlex.quote(workdir)}
checkpoint_dir={shlex.quote(checkpoint_dir)}
interval={interval}
snapshot_lock="$checkpoint_dir/snapshot.lock"
stop_file="$checkpoint_dir/stop"
secret_re='(sk-(ant-|proj-)?[A-Za-z0-9_-]{{16,}}|ghp_[A-Za-z0-9]{{20,}}|github_pat_[A-Za-z0-9_]{{20,}}|gAAAAA[A-Za-z0-9_-]{{40,}}|eyJ[A-Za-z0-9_-]{{10,}}[.][A-Za-z0-9_-]{{10,}}[.][A-Za-z0-9_-]{{10,}})'
snapshot_once() {{
  if ! mkdir "$snapshot_lock" 2>/dev/null; then return 75; fi
  base=$(cat "$checkpoint_dir/base_commit")
  git -C "$workdir" diff --binary "$base" -- > "$checkpoint_dir/workspace.patch.tmp"
  if LC_ALL=C grep -aEq "$secret_re" "$checkpoint_dir/workspace.patch.tmp"; then
    rm -f "$checkpoint_dir/workspace.patch.tmp" "$checkpoint_dir/workspace.patch"
    printf 'credential-shaped content detected in workspace patch\\n' > "$checkpoint_dir/invalid-secret"
  else
    mv "$checkpoint_dir/workspace.patch.tmp" "$checkpoint_dir/workspace.patch"
  fi
  git -C "$workdir" status --short > "$checkpoint_dir/progress-summary.txt.tmp"
  printf '\\nChanged files:\\n' >> "$checkpoint_dir/progress-summary.txt.tmp"
  git -C "$workdir" diff --stat "$base" -- >> "$checkpoint_dir/progress-summary.txt.tmp"
  mv "$checkpoint_dir/progress-summary.txt.tmp" "$checkpoint_dir/progress-summary.txt"
  git -C "$workdir" ls-files --others --exclude-standard -z | \
    tar -C "$workdir" --null -czf "$checkpoint_dir/untracked.tar.gz.tmp" \
      --exclude='.env' --exclude='.env.*' --exclude='auth.json' \
      --exclude='*.pem' --exclude='*.key' --exclude='credentials*' \
      --exclude='token*' --exclude='secret*' --exclude='password*' \
      --exclude='*.p12' --exclude='*.pfx' --files-from=-
  if tar -xOzf "$checkpoint_dir/untracked.tar.gz.tmp" 2>/dev/null | \
      LC_ALL=C grep -aEq "$secret_re"; then
    rm -f "$checkpoint_dir/untracked.tar.gz.tmp" "$checkpoint_dir/untracked.tar.gz"
    printf 'credential-shaped content detected in untracked files\\n' > "$checkpoint_dir/invalid-secret"
  else
    mv "$checkpoint_dir/untracked.tar.gz.tmp" "$checkpoint_dir/untracked.tar.gz"
  fi
  state_tmp="$checkpoint_dir/provider-state.tmp"
  rm -rf "$state_tmp"
  mkdir -p "$state_tmp"
{copies}
  if LC_ALL=C grep -aErq "$secret_re" "$state_tmp" 2>/dev/null; then
    rm -rf "$state_tmp" "$checkpoint_dir/provider-state"
    printf 'provider state omitted because it contained credential-shaped content\\n' \
      > "$checkpoint_dir/session-omitted-sensitive"
  else
    rm -rf "$checkpoint_dir/provider-state"
    mv "$state_tmp" "$checkpoint_dir/provider-state"
    rm -f "$checkpoint_dir/session-omitted-sensitive"
  fi
{probe}
  date -u +%Y-%m-%dT%H:%M:%SZ > "$checkpoint_dir/last_heartbeat.tmp"
  mv "$checkpoint_dir/last_heartbeat.tmp" "$checkpoint_dir/last_heartbeat"
  rmdir "$snapshot_lock" 2>/dev/null || true
}}
if [ "${{1:-}}" = "--once" ]; then snapshot_once; exit 0; fi
while [ ! -f "$stop_file" ]; do
  snapshot_once || true
  waited=0
  while [ "$waited" -lt "$interval" ] && [ ! -f "$stop_file" ]; do
    sleep 1
    waited=$((waited + 1))
  done
done
"""


class DurableCheckpoint:
    """Lifecycle manager shared by custom paid harness adapters."""

    REMOTE_DIR = PurePosixPath("/logs/agent/checkpoint")

    def __init__(
        self,
        *,
        logs_dir: Path,
        enabled: str | bool,
        assignment_id: str | None,
        task_id: str | None,
        model: str | None,
        effort: str | None,
        resume_generation: str | int = 0,
        checkpoint_path: str | None = None,
        harness: str,
        provider: str,
        agent_version: str,
        interval_sec: str | int = 30,
        state_paths: Iterable[StatePath] = (),
        sensitive_values: Iterable[str | bytes] = (),
        session_probe: str | None = None,
        workdir: str = "/app",
    ) -> None:
        self.enabled = _parse_bool(enabled)
        self.assignment_id = assignment_id
        self.task_id = task_id
        self.model = model
        self.effort = effort
        self.resume_generation = int(resume_generation)
        if self.resume_generation < 0:
            raise ValueError("checkpoint generation must be non-negative")
        self.previous_dir = Path(checkpoint_path) if checkpoint_path else None
        self.harness = harness
        self.provider = provider
        self.agent_version = agent_version
        self.interval_sec = max(10, min(int(interval_sec), 300))
        self.state_paths = tuple(state_paths)
        self.sensitive_values = tuple(
            raw if isinstance(raw, bytes) else raw.encode("utf-8")
            for raw in sensitive_values
            if isinstance(raw, (str, bytes)) and len(raw) >= 8
        )
        self.session_probe = session_probe
        self.workdir = workdir
        self.host_dir = logs_dir / "checkpoint"
        self.manifest_path: Path | None = None
        self.previous: dict[str, Any] | None = None
        self.session_id: str | None = None
        self.snapshot_launch_attempted = False
        self.snapshot_token: str | None = None

    def _event(self, event: str, **detail: Any) -> None:
        if self.manifest_path is None:
            return
        payload = {"at": _utc_now(), "event": event, "detail": detail}
        if _contains_sensitive_key(payload):
            raise CheckpointError("checkpoint event contains a sensitive field name")
        path = self.host_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _update(self, **changes: Any) -> None:
        if self.manifest_path is None:
            return
        value = _load_manifest(self.manifest_path)
        value.update(changes)
        value["updated_at"] = _utc_now()
        _write_manifest(self.manifest_path, value)

    async def _base_commit(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> str:
        result = await agent.exec_as_agent(
            environment,
            command=f"git -C {shlex.quote(self.workdir)} rev-parse HEAD",
            env=env,
        )
        value = (result.stdout or "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise CheckpointIncompatibleError("task worktree has no stable base commit")
        return value

    def _validate_previous(self, value: dict[str, Any], base_commit: str) -> None:
        expected = {
            "assignment_id": self.assignment_id,
            "task_id": self.task_id,
            "model": self.model,
            "effort": self.effort,
            "harness": self.harness,
            "provider": self.provider,
            "agent_version": self.agent_version,
            "base_commit": base_commit,
        }
        mismatched = [
            name for name, wanted in expected.items() if value.get(name) != wanted
        ]
        if mismatched:
            raise CheckpointIncompatibleError(
                "checkpoint runtime identity mismatch: " + ", ".join(mismatched)
            )

    async def _restore_workspace(
        self, agent: Any, environment: Any, env: dict[str, str], previous_dir: Path,
    ) -> None:
        remote_restore = "/tmp/dradar-checkpoint-restore"
        await agent.exec_as_agent(
            environment,
            command=f"rm -rf {remote_restore} && mkdir -p {remote_restore}",
            env=env,
        )
        patch = _safe_path(previous_dir, "workspace.patch")
        if patch.is_file() and patch.stat().st_size:
            remote_patch = f"{remote_restore}/workspace.patch"
            await environment.upload_file(patch, remote_patch)
            await agent.exec_as_agent(
                environment,
                command=(
                    f"git -C {shlex.quote(self.workdir)} apply --binary {remote_patch}"
                ),
                env=env,
            )
        archive = _safe_path(previous_dir, "untracked.tar.gz")
        if archive.is_file() and archive.stat().st_size:
            if _validate_archive(archive, self.sensitive_values):
                raise CheckpointError("checkpoint archive contains a credential")
            remote_archive = f"{remote_restore}/untracked.tar.gz"
            await environment.upload_file(archive, remote_archive)
            await agent.exec_as_agent(
                environment,
                command=(
                    f"tar -xzf {remote_archive} -C {shlex.quote(self.workdir)}"
                ),
                env=env,
            )

    async def _restore_state(
        self, agent: Any, environment: Any, env: dict[str, str], previous_dir: Path,
    ) -> None:
        state_root = _safe_path(previous_dir, "provider-state")
        if not state_root.is_dir():
            return
        for item in self.state_paths:
            source = _safe_path(state_root, item.name)
            if not source.exists():
                continue
            if _path_contains_any(source, self.sensitive_values):
                raise CheckpointError("checkpoint provider state contains a credential")
            remote = item.remote_path
            await agent.exec_as_agent(
                environment,
                command=(
                    f"rm -rf {shlex.quote(remote)} && mkdir -p "
                    f"{shlex.quote(str(PurePosixPath(remote).parent))}"
                ),
                env=env,
            )
            if source.is_dir():
                await environment.upload_dir(source, remote)
            elif source.is_file():
                await environment.upload_file(source, remote)

    def _previous_session_id(self, previous_dir: Path) -> str | None:
        candidates = [self.previous.get("session_id") if self.previous else None]
        path = _safe_path(previous_dir, "session-id")
        if path.is_file() and path.stat().st_size <= 512:
            candidates.insert(0, path.read_text(encoding="utf-8", errors="replace").strip())
        for candidate in candidates:
            if isinstance(candidate, str) and _SESSION_ID_RE.fullmatch(candidate):
                return candidate
        return None

    def _remove_sensitive_artifacts(self) -> bool:
        """Fail closed if an injected credential leaked into checkpoint data."""

        if not self.sensitive_values:
            return False
        leaked: list[Path] = []
        patch = self.host_dir / "workspace.patch"
        if patch.is_file() and _path_contains_any(patch, self.sensitive_values):
            leaked.append(patch)
        archive = self.host_dir / "untracked.tar.gz"
        if archive.is_file() and _validate_archive(archive, self.sensitive_values):
            leaked.append(archive)
        state_root = self.host_dir / "provider-state"
        if state_root.is_dir():
            for candidate in state_root.rglob("*"):
                if candidate.is_symlink():
                    leaked.append(state_root)
                    break
                if candidate.is_file() and _path_contains_any(
                    candidate, self.sensitive_values,
                ):
                    leaked.append(state_root)
                    break
        for candidate in set(leaked):
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink(missing_ok=True)
        if leaked:
            (self.host_dir / "invalid-secret").write_text(
                "injected credential detected in checkpoint artifact\n",
                encoding="utf-8",
            )
        return bool(leaked)

    def _discard_untrusted_artifacts(self) -> None:
        """Remove all payload-bearing files after snapshot synchronization fails."""

        for name in (
            "workspace.patch",
            "workspace.patch.tmp",
            "untracked.tar.gz",
            "untracked.tar.gz.tmp",
            "provider-state",
            "provider-state.tmp",
        ):
            candidate = self.host_dir / name
            try:
                if candidate.is_symlink() or candidate.is_file():
                    candidate.unlink(missing_ok=True)
                elif candidate.is_dir():
                    shutil.rmtree(candidate)
            except OSError:
                pass

    async def start(
        self, agent: Any, environment: Any, env: dict[str, str],
    ) -> str | None:
        if not self.enabled:
            return None
        if not isinstance(self.assignment_id, str) or not self.assignment_id:
            raise CheckpointError("checkpoint assignment id is required")
        base_commit = await self._base_commit(agent, environment, env)
        previous = None
        if self.previous_dir is not None:
            if self.previous_dir.is_symlink():
                raise CheckpointError("checkpoint root is a symlink")
            if (self.previous_dir / "invalid-secret").is_file():
                raise CheckpointError("checkpoint contains rejected credential data")
            if (self.previous_dir / "invalid-snapshot").is_file():
                raise CheckpointError("checkpoint snapshot did not finish safely")
            previous_path = _safe_path(self.previous_dir, "checkpoint.json")
            previous = _load_manifest(previous_path)
            try:
                self._validate_previous(previous, base_commit)
            except CheckpointIncompatibleError as exc:
                if self.host_dir.is_symlink():
                    raise CheckpointError("checkpoint output root is a symlink")
                if self.host_dir.exists():
                    shutil.rmtree(self.host_dir)
                self.host_dir.mkdir(parents=True, mode=0o700)
                self.manifest_path = self.host_dir / "checkpoint.json"
                now = _utc_now()
                incompatible = {
                    "schema_version": SCHEMA_VERSION,
                    "checkpoint_id": previous["checkpoint_id"],
                    "assignment_id": self.assignment_id,
                    "phase": "incompatible",
                    "created_at": previous.get("created_at", now),
                    "updated_at": now,
                    "last_heartbeat": now,
                    "task_id": self.task_id,
                    "model": self.model,
                    "effort": self.effort,
                    "harness": self.harness,
                    "provider": self.provider,
                    "agent_version": self.agent_version,
                    "base_commit": base_commit,
                    "resume_generation": self.resume_generation,
                    "resume_count": int(previous.get("resume_count", 0)),
                    "failure_type": type(exc).__name__,
                    **_FIXED_ARTIFACTS,
                }
                _write_manifest(self.manifest_path, incompatible)
                self._event(
                    "restore_rejected", failure_type=type(exc).__name__,
                )
                raise

        if self.host_dir.is_symlink():
            raise CheckpointError("checkpoint output root is a symlink")
        if self.host_dir.exists():
            shutil.rmtree(self.host_dir)
        self.host_dir.mkdir(parents=True, mode=0o700)
        self.manifest_path = self.host_dir / "checkpoint.json"
        if self.previous_dir is not None:
            old_events = self.previous_dir / "events.jsonl"
            if old_events.is_file() and not old_events.is_symlink():
                shutil.copy2(old_events, self.host_dir / "events.jsonl")
        now = _utc_now()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": (
                previous["checkpoint_id"] if previous is not None else uuid.uuid4().hex
            ),
            "assignment_id": self.assignment_id,
            "phase": "running",
            "created_at": previous.get("created_at", now) if previous else now,
            "updated_at": now,
            "last_heartbeat": now,
            "task_id": self.task_id,
            "model": self.model,
            "effort": self.effort,
            "harness": self.harness,
            "provider": self.provider,
            "agent_version": self.agent_version,
            "base_commit": base_commit,
            "resume_generation": self.resume_generation,
            "resume_count": int(previous.get("resume_count", 0)) + 1 if previous else 0,
            **_FIXED_ARTIFACTS,
        }
        _write_manifest(self.manifest_path, manifest)
        self.snapshot_token = manifest["checkpoint_id"]
        (self.host_dir / "base_commit").write_text(base_commit + "\n", encoding="utf-8")
        script = self.host_dir / "snapshot.sh"
        script.write_text(
            _snapshot_script(
                checkpoint_dir=str(self.REMOTE_DIR),
                workdir=self.workdir,
                interval_sec=self.interval_sec,
                state_paths=self.state_paths,
                session_probe=self.session_probe,
            ),
            encoding="utf-8",
        )
        script.chmod(0o700)
        self.previous = previous
        self._event("checkpoint_started", resumed=previous is not None)
        if previous is not None and self.previous_dir is not None:
            try:
                await self._restore_workspace(
                    agent, environment, env, self.previous_dir,
                )
                await self._restore_state(agent, environment, env, self.previous_dir)
            except BaseException as exc:
                self._update(phase="incompatible", failure_type=type(exc).__name__)
                self._event("restore_rejected", failure_type=type(exc).__name__)
                raise
            self.session_id = self._previous_session_id(self.previous_dir)
            self._event(
                "checkpoint_restored", native_session_available=bool(self.session_id),
            )
        remote_script = self.REMOTE_DIR / "snapshot.sh"
        self.snapshot_launch_attempted = True
        await agent.exec_as_agent(
            environment,
            command=(
                "command -v setsid >/dev/null 2>&1 || exit 75; "
                "DRADAR_SNAPSHOT_TOKEN="
                f"{shlex.quote(self.snapshot_token)} "
                f"nohup setsid sh {shlex.quote(str(remote_script))} >"
                f"{shlex.quote(str(self.REMOTE_DIR / 'snapshot.log'))} 2>&1 & "
                "snapshot_pid=$!; "
                f"echo \"$snapshot_pid\" >"
                f"{shlex.quote(str(self.REMOTE_DIR / 'snapshot.pid'))}"
            ),
            env=env,
        )
        return self.session_id

    async def finish(
        self,
        agent: Any,
        environment: Any,
        env: dict[str, str],
        *,
        completed: bool,
        failure: BaseException | None,
        session_id: str | None = None,
    ) -> None:
        if not self.enabled or self.manifest_path is None:
            return
        remote = self.REMOTE_DIR
        snapshot_stopped = True
        if self.snapshot_launch_attempted:
            try:
                snapshot_token = shlex.quote(self.snapshot_token or "")
                await agent.exec_as_agent(
                    environment,
                    command=(
                    f"touch {shlex.quote(str(remote / 'stop'))}; "
                    f"snapshot_pid=''; if [ -f "
                    f"{shlex.quote(str(remote / 'snapshot.pid'))} ]; then "
                    f"snapshot_pid=$(cat {shlex.quote(str(remote / 'snapshot.pid'))}); "
                    "fi; case \"$snapshot_pid\" in ''|*[!0-9]*) snapshot_pid='';; esac; "
                    f"snapshot_token={snapshot_token}; "
                    "snapshot_group_running() { "
                    "[ -n \"$snapshot_pid\" ] || return 1; "
                    "for snapshot_proc in /proc/[0-9]*; do "
                    "[ -r \"$snapshot_proc/stat\" ] "
                    "&& [ -r \"$snapshot_proc/environ\" ] || continue; "
                    "snapshot_state=$(awk '{print $3}' \"$snapshot_proc/stat\" "
                    "2>/dev/null || true); "
                    "snapshot_pgrp=$(awk '{print $5}' \"$snapshot_proc/stat\" "
                    "2>/dev/null || true); "
                    "[ \"$snapshot_state\" != Z ] "
                    "&& [ \"$snapshot_pgrp\" = \"$snapshot_pid\" ] || continue; "
                    "if tr '\\000' '\\n' < \"$snapshot_proc/environ\" "
                    "2>/dev/null | grep -Fqx "
                    "\"DRADAR_SNAPSHOT_TOKEN=$snapshot_token\"; then "
                    "return 0; fi; done; return 1; }; "
                    "waited=0; while snapshot_group_running "
                    "&& [ \"$waited\" -lt 300 ]; do "
                    "sleep 0.1; waited=$((waited + 1)); done; "
                    "if snapshot_group_running; then "
                    "kill -TERM -- \"-$snapshot_pid\" 2>/dev/null || true; "
                    "waited=0; while snapshot_group_running "
                    "&& [ \"$waited\" -lt 50 ]; do "
                    "sleep 0.1; waited=$((waited + 1)); done; fi; "
                    "if snapshot_group_running; then "
                    "kill -KILL -- \"-$snapshot_pid\" 2>/dev/null || true; "
                    "waited=0; while snapshot_group_running "
                    "&& [ \"$waited\" -lt 50 ]; do "
                    "sleep 0.1; waited=$((waited + 1)); done; fi; "
                    "if snapshot_group_running; then exit 75; fi; "
                    f"sh {shlex.quote(str(remote / 'snapshot.sh'))} --once"
                    ),
                    env=env,
                )
            except BaseException:
                snapshot_stopped = False
                self._discard_untrusted_artifacts()
                (self.host_dir / "invalid-snapshot").write_text(
                    "checkpoint snapshot did not stop cleanly\n",
                    encoding="utf-8",
                )
        try:
            if snapshot_stopped:
                self._remove_sensitive_artifacts()
        except BaseException:
            self._discard_untrusted_artifacts()
            (self.host_dir / "invalid-snapshot").write_text(
                "checkpoint artifact validation failed\n", encoding="utf-8",
            )
        selected_session = session_id or self.session_id
        if not (
            isinstance(selected_session, str)
            and _SESSION_ID_RE.fullmatch(selected_session)
        ):
            sidecar = self.host_dir / "session-id"
            if (
                sidecar.is_file()
                and not sidecar.is_symlink()
                and sidecar.stat().st_size <= 512
            ):
                candidate = sidecar.read_text(
                    encoding="utf-8", errors="replace",
                ).strip()
                if _SESSION_ID_RE.fullmatch(candidate):
                    selected_session = candidate
        if isinstance(selected_session, str) and _SESSION_ID_RE.fullmatch(selected_session):
            (self.host_dir / "session-id").write_text(
                selected_session + "\n", encoding="utf-8",
            )
            try:
                (self.host_dir / "session-id").chmod(0o600)
            except OSError:
                pass
        invalid_secret = (self.host_dir / "invalid-secret").is_file()
        invalid_snapshot = (self.host_dir / "invalid-snapshot").is_file()
        invalid = invalid_secret or invalid_snapshot
        phase = (
            "invalid"
            if invalid
            else (
                "incompatible"
                if isinstance(failure, CheckpointIncompatibleError)
                else ("agent_completed" if completed else "paused")
            )
        )
        changes: dict[str, Any] = {"phase": phase}
        if isinstance(selected_session, str) and _SESSION_ID_RE.fullmatch(selected_session):
            changes["session_id"] = selected_session
        if invalid_secret:
            changes["failure_type"] = "CheckpointSecretDetected"
        elif invalid_snapshot:
            changes["failure_type"] = "CheckpointSnapshotInvalid"
        elif failure is not None:
            changes["failure_type"] = type(failure).__name__
        self._update(**changes)
        self._event(
            phase,
            native_session_available=bool(changes.get("session_id")),
            failure_type=type(failure).__name__ if failure else None,
        )


__all__ = [
    "CheckpointError",
    "CheckpointIncompatibleError",
    "DurableCheckpoint",
    "StatePath",
]
