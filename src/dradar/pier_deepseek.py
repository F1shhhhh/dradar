"""Public Pier adapter that installs DeepSeek's official Codex model catalog.

The stock datacurve-pier 0.3.0 Codex agent can inject ``config.toml`` text and
an auth file, but it has no generic extra-file upload hook.  Codex runs with an
isolated ``/tmp/codex-home`` inside the task container, so a catalog path on
the host is otherwise unusable.  This narrow subclass uploads exactly one
integrity-pinned public metadata file before delegating normal execution and
trajectory collection to Pier.
"""

from __future__ import annotations

import hashlib
import shlex
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from pier.agents.network import allowlist_from_urls
from pier.agents.installed.codex import Codex
from pier.agents.installed.codex_checkpoint import (
    CheckpointIncompatibleError,
    load_manifest,
    new_manifest,
    update_manifest,
    write_manifest,
)
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.network import NetworkAllowlist

_CATALOG_SHA256 = (
    "8cfa8ab037573ae9914478e6dcd544c43d93c1b126cab5ad58252230dcbe071d"
)


class DeepSeekCodex(Codex):
    """Stock Codex plus a fail-closed, container-local model catalog."""

    _REMOTE_MODEL_CATALOG = PurePosixPath("/tmp/codex-home/models.json")
    _CHECKPOINT_PROVIDER = "deepseek"
    _CHECKPOINT_HARNESS = "codex"

    def network_allowlist(self) -> NetworkAllowlist:
        """Allow only the paid provider endpoint during agent execution.

        Stock Pier's Codex allowlist always includes api.openai.com as a
        default. DeepSeek does not need that endpoint, and apps, remote plugin,
        and web search are intentionally disabled for benchmark isolation.
        """

        return allowlist_from_urls(
            ["https://api.deepseek.com/"],
            default_domains=[],
        )

    def __init__(
        self,
        *args: Any,
        model_catalog_json_file: str,
        **kwargs: Any,
    ):
        catalog = Path(model_catalog_json_file)
        try:
            digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(
                f"DeepSeek model catalog is unreadable: {catalog}"
            ) from exc
        if digest != _CATALOG_SHA256:
            raise ValueError(
                "DeepSeek model catalog integrity check failed; reinstall or "
                "upgrade dradar before running a paid task"
            )
        self._model_catalog_json_file = catalog
        super().__init__(*args, **kwargs)

    async def _start_checkpoint(
        self, environment: BaseEnvironment, env: dict[str, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Fence DeepSeek state from any other Codex provider/version."""

        previous_dir = self._resume_checkpoint_path
        if self._checkpoint_enabled and previous_dir is not None:
            if previous_dir.is_symlink():
                raise CheckpointIncompatibleError(
                    "DeepSeek checkpoint root is a symlink"
                )
            previous = load_manifest(previous_dir / "checkpoint.json")
            expected = {
                "assignment_id": self._checkpoint_assignment_id,
                "task_id": self._checkpoint_task_id,
                "model": self.model_name,
                "effort": self._checkpoint_effort,
                "provider": self._CHECKPOINT_PROVIDER,
                "harness": self._CHECKPOINT_HARNESS,
                "agent_version": self._version,
            }
            mismatched = [
                key for key, value in expected.items()
                if previous.get(key) != value
            ]
            if mismatched:
                base_commit = await self._current_base_commit(environment, env)
                checkpoint_dir = self._checkpoint_host_dir
                if checkpoint_dir.is_symlink():
                    raise CheckpointIncompatibleError(
                        "DeepSeek checkpoint output root is a symlink"
                    )
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, mode=0o700)
                self._checkpoint_manifest_path = checkpoint_dir / "checkpoint.json"
                incompatible = new_manifest(
                    assignment_id=self._checkpoint_assignment_id,
                    model=self.model_name,
                    task_id=self._checkpoint_task_id,
                    effort=self._checkpoint_effort,
                    base_commit=base_commit,
                    resume_generation=self._checkpoint_resume_generation,
                    checkpoint_id=previous.get("checkpoint_id"),
                    resume_count=int(previous.get("resume_count", 0)),
                )
                incompatible.update({
                    "phase": "incompatible",
                    "failure_type": "CheckpointIncompatibleError",
                    "provider": self._CHECKPOINT_PROVIDER,
                    "harness": self._CHECKPOINT_HARNESS,
                    "agent_version": self._version,
                })
                write_manifest(self._checkpoint_manifest_path, incompatible)
                self._checkpoint_event(
                    "restore_rejected",
                    failure_type="CheckpointIncompatibleError",
                )
                raise CheckpointIncompatibleError(
                    "DeepSeek checkpoint runtime identity mismatch: "
                    + ", ".join(mismatched)
                )
        previous, root = await super()._start_checkpoint(environment, env)
        if self._checkpoint_manifest_path is not None:
            update_manifest(
                self._checkpoint_manifest_path,
                provider=self._CHECKPOINT_PROVIDER,
                harness=self._CHECKPOINT_HARNESS,
                agent_version=self._version,
            )
        return previous, root

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        remote_home = self._REMOTE_CODEX_HOME.as_posix()
        env = self.build_process_env({"CODEX_HOME": remote_home})
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {shlex.quote(remote_home)}",
            env=env,
        )
        await environment.upload_file(
            self._model_catalog_json_file,
            self._REMOTE_MODEL_CATALOG.as_posix(),
        )
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                command=(
                    f"chown {shlex.quote(str(environment.default_user))} "
                    f"{shlex.quote(self._REMOTE_MODEL_CATALOG.as_posix())}"
                ),
                env=env,
            )
        # Re-check inside the real task container, as the same user that will
        # launch Codex. This catches a failed/truncated upload or unreadable
        # ownership before any paid model request can be made.
        await self.exec_as_agent(
            environment,
            command=(
                f"sha256sum {shlex.quote(self._REMOTE_MODEL_CATALOG.as_posix())} "
                f"| grep -Fq {shlex.quote(_CATALOG_SHA256)}"
            ),
            env=env,
        )
        self.logger.info(
            "DeepSeek Codex model catalog verified in task container: %s",
            _CATALOG_SHA256,
        )
        await super().run(instruction, environment, context)


__all__ = ["DeepSeekCodex"]
