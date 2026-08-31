"""Standalone Pier startup shim copied as ``sitecustomize.py`` per trial.

This module must use only the standard library: it executes inside Pier's own
isolated Python environment, where the ``dradar`` package is not installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

_IMAGE_ENV = "DRADAR_EGRESS_PROXY_IMAGE"
_CODEBUDDY_SOURCE_IMAGE_ENV = "DRADAR_CODEBUDDY_SOURCE_IMAGE"
_PATCH_MARKER = "_dradar_prebuilt_egress_codebuddy_v2"
_LOCAL_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CODEBUDDY_SOURCE_IMAGE_RE = re.compile(
    r"dradar-codebuddy:(?P<version>[0-9]+\.[0-9]+\.[0-9]+)\Z"
)
_OFFICIAL_DIGEST_PREFIX = (
    "ghcr.io/codex-radar/dradar-egress-proxy@sha256:"
)
_CODEBUDDY_BUNDLE_COMMAND = (
    "set -euo pipefail; runtime=/opt/dradar-codebuddy-runtime; "
    "mkdir -p \"$runtime/lib\"; "
    "cp -L /opt/codebuddy/bin/codebuddy \"$runtime/codebuddy\"; "
    "loader=$(ldd /opt/codebuddy/bin/codebuddy | "
    "awk '/ld-linux/{print $1; exit}'); "
    "test -n \"$loader\"; cp -L \"$loader\" \"$runtime/loader\"; "
    "ldd /opt/codebuddy/bin/codebuddy | "
    "awk '$2 == \"=>\" && $3 ~ /^\\// {print $3}' | "
    "while IFS= read -r library; do cp -L \"$library\" \"$runtime/lib/\"; done"
)


def _image_is_immutable(image: str) -> bool:
    return bool(
        _LOCAL_IMAGE_ID_RE.fullmatch(image)
        or (
            image.startswith(_OFFICIAL_DIGEST_PREFIX)
            and _LOCAL_IMAGE_ID_RE.fullmatch(image.split("@", 1)[1])
        )
    )


def _proxy_policy_env(allowlist, token: str) -> dict[str, str]:
    environment = {
        "PROXY_TOKEN": token,
        "ALLOWLIST_DOMAINS": ",".join(allowlist.domains),
    }
    mappings = {
        "DRADAR_EGRESS_UPSTREAM_HOST": "UPSTREAM_PROXY_HOST",
        "DRADAR_EGRESS_UPSTREAM_PORT": "UPSTREAM_PROXY_PORT",
        "DRADAR_EGRESS_UPSTREAM_USERNAME": "UPSTREAM_PROXY_USERNAME",
        "DRADAR_EGRESS_UPSTREAM_PASSWORD": "UPSTREAM_PROXY_PASSWORD",
    }
    for source, target in mappings.items():
        if value := os.environ.get(source):
            environment[target] = value
    return environment


def _write_docker_proxy_compose(
    *, path: Path, proxy_dir: Path, allowlist, token: str,
) -> Path:
    del proxy_dir
    image = os.environ[_IMAGE_ENV]
    proxy_service = {
        "image": image,
        "pull_policy": "never",
        "environment": _proxy_policy_env(allowlist, token),
        "healthcheck": {
            "test": ["CMD-SHELL", "bash -lc '</dev/tcp/127.0.0.1/8080'"],
            "interval": "1s",
            "timeout": "1s",
            "retries": 30,
        },
        "networks": ["pier-egress-internal", "default"],
    }
    if os.environ.get("DRADAR_EGRESS_UPSTREAM_HOST") == "host.docker.internal":
        proxy_service["extra_hosts"] = ["host.docker.internal:host-gateway"]
    compose = {
        "services": {
            "main": {
                "networks": ["pier-egress-internal"],
                "depends_on": {
                    "pier-egress-proxy": {"condition": "service_healthy"},
                },
            },
            "pier-egress-proxy": proxy_service,
        },
        "networks": {"pier-egress-internal": {"internal": True}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _build_proxy_override() -> dict[str, object] | None:
    proxy = os.environ.get("DRADAR_EGRESS_BUILD_PROXY")
    if not proxy:
        return None
    arguments = {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
    }
    if no_proxy := os.environ.get("DRADAR_EGRESS_BUILD_NO_PROXY"):
        arguments.update({"NO_PROXY": no_proxy, "no_proxy": no_proxy})
    build: dict[str, object] = {"args": arguments}
    if os.environ.get("DRADAR_EGRESS_UPSTREAM_HOST") == "host.docker.internal":
        build["extra_hosts"] = ["host.docker.internal=host-gateway"]
    return build


def _finalize_docker_proxy_compose(
    path: Path,
    runtime_environment: dict[str, str],
    build_override: dict[str, object] | None,
) -> None:
    """Move the short-lived proxy token out of `docker compose exec` argv."""

    compose = json.loads(path.read_text(encoding="utf-8"))
    main = compose["services"]["main"]
    environment = dict(main.get("environment") or {})
    environment.update(runtime_environment)
    main["environment"] = environment
    if build_override is not None:
        main["build"] = build_override
    path.write_text(json.dumps(compose, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _remove_codebuddy_helper(name: str) -> bool:
    try:
        removed = subprocess.run(
            ["docker", "rm", "-f", name], capture_output=True,
            text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return removed.returncode == 0


def _materialize_codebuddy_runtime(source_image: str, build_dir: Path) -> Path:
    """Export the reviewed runtime into Pier's assignment-local build context.

    Per-assignment BuildKit builders deliberately cannot see Docker Engine's
    local image store.  Referencing the local source tag in a Dockerfile makes
    BuildKit try Docker Hub instead.  Run the already validated image with
    pulling disabled, copy only the bundled executable/runtime libraries into
    the build context, then remove the exact stopped helper container.
    """

    destination = build_dir / "dradar-codebuddy-runtime"
    if destination.exists():
        raise RuntimeError("CodeBuddy runtime build context already exists")
    helper = f"dradar-codebuddy-source-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    try:
        run = subprocess.run(
            [
                "docker", "run", "--name", helper, "--pull", "never",
                source_image, "/bin/bash", "-c", _CODEBUDDY_BUNDLE_COMMAND,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _remove_codebuddy_helper(helper)
        raise RuntimeError(
            "could not export the validated CodeBuddy runtime"
        ) from exc
    if run.returncode != 0:
        _remove_codebuddy_helper(helper)
        raise RuntimeError("could not export the validated CodeBuddy runtime")
    destination.mkdir(mode=0o700)
    try:
        copy = subprocess.run(
            [
                "docker", "cp",
                f"{helper}:/opt/dradar-codebuddy-runtime/.", str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        copy = None
    removed = _remove_codebuddy_helper(helper)
    if copy is None or copy.returncode != 0 or not removed:
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError("could not stage the validated CodeBuddy runtime")
    if not (
        (destination / "codebuddy").is_file()
        and (destination / "loader").is_file()
        and (destination / "lib").is_dir()
    ):
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError("validated CodeBuddy runtime bundle is incomplete")
    return destination


def _rewrite_codebuddy_agent_dockerfile(environment) -> None:
    """Copy the reviewed CLI and glibc bundle from a validated local image."""

    install = environment.agent_install_spec
    if install is None or install.agent_name != "codebuddy":
        return
    source_image = os.environ.get(_CODEBUDDY_SOURCE_IMAGE_ENV, "")
    match = _CODEBUDDY_SOURCE_IMAGE_RE.fullmatch(source_image)
    if match is None or match.group("version") != install.version:
        raise RuntimeError("CodeBuddy source image is missing or version-mismatched")
    if len(install.steps) != 1 or install.steps[0].user != "root":
        raise RuntimeError("CodeBuddy install spec shape changed unexpectedly")
    build_dir = environment._agent_build_context_dir
    if build_dir is None:
        raise RuntimeError("CodeBuddy agent build context was not prepared")
    dockerfile_path = Path(build_dir) / "Dockerfile"
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    install_run = "RUN " + json.dumps(
        ["/bin/bash", "-c", install.steps[0].run]
    )
    suffix = f"USER root\n{install_run}\n"
    if not dockerfile.endswith(suffix):
        raise RuntimeError("CodeBuddy generated Dockerfile shape changed unexpectedly")
    verify_run = "RUN " + json.dumps(
        ["/bin/bash", "-c", install.verification_command]
    )
    wrapper_command = (
        "set -euo pipefail; mkdir -p /opt/codebuddy/bin; "
        "printf '%s\\n' '#!/bin/sh' "
        "'exec /opt/codebuddy/runtime/loader --library-path "
        "/opt/codebuddy/runtime/lib /opt/codebuddy/runtime/codebuddy \"$@\"' "
        "> /opt/codebuddy/bin/codebuddy; chmod 0755 /opt/codebuddy/bin/codebuddy"
    )
    wrapper_run = "RUN " + json.dumps(["/bin/bash", "-c", wrapper_command])
    _materialize_codebuddy_runtime(source_image, build_dir)
    replacement = (
        "USER root\n"
        "COPY dradar-codebuddy-runtime/ "
        "/opt/codebuddy/runtime/\n"
        f"{wrapper_run}\n"
        f"{verify_run}\n"
    )
    dockerfile_path.write_text(
        dockerfile[: -len(suffix)]
        + replacement,
        encoding="utf-8",
    )


def _patch_pier() -> None:
    image = os.environ.get(_IMAGE_ENV)
    codebuddy_source = os.environ.get(_CODEBUDDY_SOURCE_IMAGE_ENV)
    if not image and not codebuddy_source:
        return
    if image and not _image_is_immutable(image):
        raise RuntimeError("DRadar egress image is not pinned by digest")

    from pier.environments import agent_setup
    from pier.environments.docker import docker as docker_environment

    if getattr(docker_environment, _PATCH_MARKER, False):
        return
    original_prepare = None
    if image:
        agent_setup.write_docker_proxy_compose = _write_docker_proxy_compose
        docker_environment.write_docker_proxy_compose = _write_docker_proxy_compose
        original_prepare = (
            docker_environment.DockerEnvironment._prepare_egress_proxy_compose
        )
    original_agent_prepare = (
        docker_environment.DockerEnvironment._prepare_agent_build_context
    )

    def prepare_agent_with_local_codebuddy(self) -> None:
        original_agent_prepare(self)
        _rewrite_codebuddy_agent_dockerfile(self)

    def prepare_with_build_proxy(self) -> None:
        assert original_prepare is not None
        original_prepare(self)
        if self._egress_proxy_compose_path is None:
            return
        path = self._egress_proxy_compose_path
        runtime_environment = dict(self._egress_proxy_env)
        build_override = (
            _build_proxy_override()
            if self.agent_install_spec is not None else None
        )
        _finalize_docker_proxy_compose(
            path, runtime_environment, build_override,
        )
        # The main service already carries these values. Clearing the Pier
        # injection map prevents the short-lived Basic token from appearing in
        # `docker compose exec -e HTTP_PROXY=...` process arguments.
        self._egress_proxy_env = {}

    if image:
        docker_environment.DockerEnvironment._prepare_egress_proxy_compose = (
            prepare_with_build_proxy
        )
    if codebuddy_source:
        docker_environment.DockerEnvironment._prepare_agent_build_context = (
            prepare_agent_with_local_codebuddy
        )
    setattr(docker_environment, _PATCH_MARKER, True)


if __name__ == "sitecustomize":
    try:
        _patch_pier()
    except Exception as exc:  # pragma: no cover - exercised in the Pier subprocess
        # Python normally ignores sitecustomize failures and continues. That would
        # silently fall back to Pier's dynamic apt build, reopening the exact cold
        # machine failure this shim prevents, so fail closed before any task starts.
        sys.stderr.write(
            "DRadar Pier egress bootstrap failed before task start: "
            f"{type(exc).__name__}\n"
        )
        os._exit(78)
