"""Durable Pier adapter for DeepSeek's official Codex model catalog.

The stock datacurve-pier 0.3.0 Codex agent can inject ``config.toml`` text and
an auth file, but it has no generic extra-file upload hook.  Codex runs with an
isolated ``/tmp/codex-home`` inside the task container, so a catalog path on
the host is otherwise unusable.  This narrow subclass uploads exactly one
integrity-pinned public metadata file before delegating the unchanged model
turn and trajectory collection to DRadar's host-private Codex adapter.
"""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.network import NetworkAllowlist

try:
    from _dradar_pier_codex import DurableCodex
except ModuleNotFoundError as exc:  # Source-tree unit tests.
    if exc.name != "_dradar_pier_codex":
        raise
    from dradar.pier_codex import DurableCodex

_CATALOG_SHA256 = (
    "8cfa8ab037573ae9914478e6dcd544c43d93c1b126cab5ad58252230dcbe071d"
)


class DeepSeekCodex(DurableCodex):
    """Host-private durable Codex plus a fail-closed DeepSeek catalog."""

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

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        remote_home = self._REMOTE_CODEX_HOME.as_posix()
        env = self.build_process_env({
            "CODEX_HOME": remote_home,
            "HOME": "/tmp/dradar-agent-home",
        })
        if self._durable_checkpoint.enabled:
            await self._durable_checkpoint.prepare_agent_environment(
                self, environment, env,
            )
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
            command = (
                f"chown {shlex.quote(str(environment.default_user))} "
                f"{shlex.quote(self._REMOTE_MODEL_CATALOG.as_posix())}"
            )
            if self._durable_checkpoint.enabled:
                await self._durable_checkpoint.exec_root_maintenance(
                    environment, command=command,
                )
            else:
                await self.exec_as_root(environment, command=command, env=env)
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
