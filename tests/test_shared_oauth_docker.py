"""Security contract for the narrow shared OAuth Docker environment."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest


class _DockerEnvironmentStub:
    pass


# Pier is intentionally installed only inside the ephemeral task runtime, not
# as a DRadar package dependency.  Supply the single class this copied module
# subclasses so its validation contract remains unit-testable here.
_pier_modules = {
    name: types.ModuleType(name)
    for name in (
        "pier",
        "pier.environments",
        "pier.environments.docker",
        "pier.environments.docker.docker",
    )
}
_pier_modules["pier.environments.docker.docker"].DockerEnvironment = (
    _DockerEnvironmentStub
)
_previous_modules = {name: sys.modules.get(name) for name in _pier_modules}
sys.modules.update(_pier_modules)
try:
    from dradar.pier_shared_oauth_docker import (
        SharedOAuthDockerEnvironment,
        _validated_shared_mounts,
    )
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    if os.name != "nt":
        path.chmod(0o700)
    return path.resolve()


def test_validated_shared_mounts_accept_only_private_managed_targets(
    tmp_path: Path,
) -> None:
    source = _private_dir(tmp_path / "oauth")
    assert _validated_shared_mounts([
        {
            "type": "bind",
            "source": str(source),
            "target": "/tmp/dradar-kimi-home/oauth",
        }
    ]) == [
        {
            "type": "bind",
            "source": str(source),
            "target": "/tmp/dradar-kimi-home/oauth",
        }
    ]

    with pytest.raises(ValueError, match="target is not allowed"):
        _validated_shared_mounts([
            {"type": "bind", "source": str(source), "target": "/app"}
        ])

    antigravity = _private_dir(tmp_path / "antigravity" / ".gemini")
    assert _validated_shared_mounts([{
        "type": "bind",
        "source": str(antigravity),
        "target": "/tmp/dradar-antigravity-user/.gemini",
    }])[0]["target"] == "/tmp/dradar-antigravity-user/.gemini"


def test_validated_shared_mounts_reject_symlink_and_broad_permissions(
    tmp_path: Path,
) -> None:
    source = _private_dir(tmp_path / "oauth")
    symlink = tmp_path / "oauth-link"
    symlink.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="existing directory"):
        _validated_shared_mounts([
            {
                "type": "bind",
                "source": str(symlink),
                "target": "/tmp/dradar-kimi-home/oauth",
            }
        ])

    if os.name != "nt":
        source.chmod(0o755)
        with pytest.raises(ValueError, match="too broadly accessible"):
            _validated_shared_mounts([
                {
                    "type": "bind",
                    "source": str(source),
                    "target": "/tmp/dradar-kimi-home/oauth",
                }
            ])


def test_environment_preserves_pier_default_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _private_dir(tmp_path / "grok")

    def fake_init(self, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        self._mounts_json = [
            {"type": "bind", "source": "/host/logs", "target": "/logs/agent"}
        ]

    monkeypatch.setattr(
        "dradar.pier_shared_oauth_docker.DockerEnvironment.__init__", fake_init
    )
    environment = SharedOAuthDockerEnvironment(
        shared_oauth_mounts_json=[
            {
                "type": "bind",
                "source": str(source),
                "target": "/tmp/dradar-grok-user/.grok",
            }
        ]
    )
    assert [mount["target"] for mount in environment._mounts_json] == [
        "/logs/agent",
        "/tmp/dradar-grok-user/.grok",
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership reconciliation")
def test_antigravity_exec_reconciles_only_exact_host_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _private_dir(tmp_path / "antigravity" / ".gemini")
    calls: list[dict[str, object]] = []

    def fake_init(self, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        self._mounts_json = [
            {"type": "bind", "source": "/host/logs", "target": "/logs/agent"}
        ]

    async def fake_exec(self, **kwargs):  # noqa: ANN001, ANN202, ARG001
        calls.append(dict(kwargs))
        return types.SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(
        "dradar.pier_shared_oauth_docker.DockerEnvironment.__init__", fake_init,
    )
    monkeypatch.setattr(
        "dradar.pier_shared_oauth_docker.DockerEnvironment.exec", fake_exec,
        raising=False,
    )
    environment = SharedOAuthDockerEnvironment(
        shared_oauth_mounts_json=[{
            "type": "bind",
            "source": str(source),
            "target": "/tmp/dradar-antigravity-user/.gemini",
        }],
    )

    asyncio.run(environment.exec("antigravity models", env={"HOME": "/tmp"}))

    assert calls[0]["command"] == "antigravity models"
    assert calls[0]["user"] is None
    maintenance = str(calls[1]["command"])
    assert calls[1]["user"] == "root"
    assert "find -P /logs/agent -xdev" in maintenance
    assert "find -P /tmp/dradar-antigravity-user/.gemini -xdev" in maintenance
    assert f"chown -h -- {os.getuid()}:{os.getgid()}" in maintenance
    assert "/app" not in maintenance


def test_non_antigravity_shared_oauth_exec_does_not_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _private_dir(tmp_path / "kimi" / "oauth")
    calls: list[dict[str, object]] = []

    def fake_init(self, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        self._mounts_json = []

    async def fake_exec(self, **kwargs):  # noqa: ANN001, ANN202, ARG001
        calls.append(dict(kwargs))
        return types.SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(
        "dradar.pier_shared_oauth_docker.DockerEnvironment.__init__", fake_init,
    )
    monkeypatch.setattr(
        "dradar.pier_shared_oauth_docker.DockerEnvironment.exec", fake_exec,
        raising=False,
    )
    environment = SharedOAuthDockerEnvironment(
        shared_oauth_mounts_json=[{
            "type": "bind",
            "source": str(source),
            "target": "/tmp/dradar-kimi-home/oauth",
        }],
    )

    asyncio.run(environment.exec("kimi --auto"))

    assert len(calls) == 1
    assert calls[0]["command"] == "kimi --auto"
