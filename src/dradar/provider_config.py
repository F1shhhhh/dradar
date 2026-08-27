"""Interactive, local-only model-provider credential setup."""

from __future__ import annotations

import getpass
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .providers import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_MODELS,
    GROK_API_KEY_ENV,
    GROK_CLI_VERSION,
    GROK_MODEL,
    KIMI_API_KEY_ENVS,
    KIMI_BINARY_BASE_URL,
    KIMI_BINARY_SHA256,
    KIMI_CLI_VERSION,
    ZCODE_CLI_VERSION,
    ZCODE_MODELS,
    ZCODE_OFFICIAL_DOWNLOAD_PAGE,
    deepseek_api_key,
    deepseek_credential_source,
    deepseek_secret_error,
    deepseek_secret_path,
    grok_auth_error,
    grok_auth_path,
    grok_cli_path,
    grok_home,
    grok_live_error,
    kimi_auth_error,
    kimi_auth_path,
    kimi_cli_path,
    kimi_home,
    kimi_live_error,
    managed_grok_cli_path,
    managed_kimi_cli_path,
    parse_kimi_cli_version,
    parse_grok_cli_version,
    parse_zcode_cli_version,
    provider_subprocess_env,
    store_grok_auth,
    store_deepseek_api_key,
    store_zcode_cli,
    store_zcode_api_key,
    zcode_api_key,
    zcode_cli_error,
    zcode_cli_path,
    zcode_credential_source,
    zcode_secret_error,
    zcode_secret_path,
)

_DEEPSEEK_MODELS_URL = "https://api.deepseek.com/models"
_ZCODE_MODELS_URL = "https://open.bigmodel.cn/api/coding/paas/v4/models"
_GROK_INSTALLER_URL = "https://x.ai/cli/install.sh"
_GROK_INSTALLER_SHA256 = (
    "43d0943123edade1383a476a4f778674877acee7c1f98a00f094c4a0f7349321"
)
def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(path, 0o700)


def _provider_proxy_for_url(url: str, env: dict[str, str]) -> str | None:
    """Resolve shell/macOS proxy settings while respecting NO_PROXY."""

    parsed = urlsplit(url)
    host = parsed.hostname or ""
    no_proxy = env.get("NO_PROXY") or env.get("no_proxy")
    if no_proxy and urllib.request.proxy_bypass_environment(
        host, {"no": no_proxy},
    ):
        return None
    prefix = parsed.scheme.upper()
    value = (
        env.get(f"{prefix}_PROXY")
        or env.get(f"{prefix.lower()}_proxy")
        or env.get("ALL_PROXY")
        or env.get("all_proxy")
    )
    if isinstance(value, str) and value.lower().startswith("socks://"):
        value = "socks5://" + value[len("socks://"):]
    return value or None


def _provider_httpx_get(url: str, **kwargs):
    """GET through the same explicit/OS proxy contract as provider CLIs."""

    env = provider_subprocess_env()
    proxy = _provider_proxy_for_url(url, env)
    timeout = kwargs.pop("timeout", 10.0)
    follow_redirects = kwargs.pop("follow_redirects", False)
    with httpx.Client(
        proxy=proxy,
        trust_env=False,
        timeout=timeout,
        follow_redirects=follow_redirects,
    ) as client:
        return client.get(url, **kwargs)


def _grok_cli_version(executable: str | Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return parse_grok_cli_version(result.stdout + "\n" + result.stderr)


def _kimi_cli_version(executable: str | Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return parse_kimi_cli_version(result.stdout + "\n" + result.stderr)


def _install_managed_grok_cli() -> str | None:
    """Install the current stable Grok release into DRadar's private slot."""

    bash = shutil.which("bash")
    if not bash:
        print(
            "Could not auto-install Grok: bash is unavailable. Install Git Bash "
            "on Windows, or bash on Linux/macOS, then retry."
        )
        return None
    target = managed_grok_cli_path()
    runtime = target.parent.parent
    _private_directory(runtime)
    _private_directory(target.parent)
    try:
        response = _provider_httpx_get(
            _GROK_INSTALLER_URL, timeout=30.0, follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        print(f"Could not download the official Grok installer: {type(exc).__name__}.")
        return None
    if response.status_code != 200 or not response.content.startswith(b"#!"):
        print(
            "Could not download the official Grok installer "
            f"(HTTP {response.status_code})."
        )
        return None
    if hashlib.sha256(response.content).hexdigest() != _GROK_INSTALLER_SHA256:
        print(
            "The official Grok installer changed since this DRadar release; "
            "refusing to execute it until the new script is reviewed."
        )
        return None
    try:
        with tempfile.TemporaryDirectory(prefix=".install-", dir=runtime) as name:
            script = Path(name) / "install.sh"
            script.write_bytes(response.content)
            if os.name != "nt":
                script.chmod(0o700)
            installer_home = runtime / "installer-home"
            _private_directory(installer_home)
            env = provider_subprocess_env()
            env["HOME"] = str(installer_home)
            env["GROK_BIN_DIR"] = str(target.parent)
            env["GROK_CHANNEL"] = "stable"
            for key in ("GROK_DEPLOYMENT_KEY", GROK_API_KEY_ENV):
                env.pop(key, None)
            print(f"Installing official Grok CLI {GROK_CLI_VERSION} for DRadar...")
            result = subprocess.run(
                [bash, str(script), GROK_CLI_VERSION], env=env, check=False,
            )
    except OSError as exc:
        print(f"Could not install Grok CLI: {exc}")
        return None
    if result.returncode != 0 or _grok_cli_version(target) != GROK_CLI_VERSION:
        print("The official Grok installer completed without a usable DRadar runtime.")
        return None
    return str(target)


def _install_managed_kimi_cli() -> str | None:
    """Install the reviewed official Kimi Code native bundle privately."""

    system = platform.system().lower()
    system = {
        "windows": "win32", "darwin": "darwin", "linux": "linux",
    }.get(system, system)
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        arch = "arm64"
    elif machine in {"x86_64", "amd64"}:
        arch = "x64"
    else:
        arch = ""
    artifact = f"{system}-{arch}" if arch else ""
    expected = KIMI_BINARY_SHA256.get(artifact)
    if expected is None:
        print(f"Could not auto-install Kimi Code on {system}/{machine}.")
        return None
    target = managed_kimi_cli_path()
    runtime = target.parent.parent
    _private_directory(runtime)
    _private_directory(target.parent)
    print(f"Installing official Kimi Code CLI {KIMI_CLI_VERSION} for DRadar...")
    try:
        suffix = ".exe" if system == "win32" else ""
        response = _provider_httpx_get(
            f"{KIMI_BINARY_BASE_URL}/kimi-code-{artifact}{suffix}",
            timeout=60.0,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        print(f"Could not download official Kimi Code CLI: {type(exc).__name__}.")
        return None
    if response.status_code != 200:
        print(f"Could not download official Kimi Code CLI (HTTP {response.status_code}).")
        return None
    if hashlib.sha256(response.content).hexdigest() != expected:
        print("Official Kimi Code binary checksum mismatch; refusing to install it.")
        return None
    temp_name: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".kimi-", dir=target.parent)
        with os.fdopen(fd, "wb") as handle:
            handle.write(response.content)
            handle.flush()
            os.fsync(handle.fileno())
        temp = Path(temp_name)
        if os.name != "nt":
            temp.chmod(0o700)
        os.replace(temp, target)
    except OSError as exc:
        print(f"Could not install Kimi Code CLI: {exc}")
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except OSError:
                pass
        return None
    if _kimi_cli_version(target) != KIMI_CLI_VERSION:
        print("The official Kimi installer completed without a usable DRadar runtime.")
        return None
    return str(target)


def _ensure_grok_cli() -> str | None:
    executable = grok_cli_path()
    found = _grok_cli_version(executable) if executable else None
    if found == GROK_CLI_VERSION:
        return executable
    if executable:
        print(
            f"Found Grok CLI {found or 'unknown'}; preparing current DRadar "
            f"runtime {GROK_CLI_VERSION} without changing the global install."
        )
    return _install_managed_grok_cli()


def _ensure_kimi_cli() -> str | None:
    executable = kimi_cli_path()
    found = _kimi_cli_version(executable) if executable else None
    if found == KIMI_CLI_VERSION:
        return executable
    if executable:
        print(
            f"Found Kimi Code CLI {found or 'unknown'}; preparing current DRadar "
            f"runtime {KIMI_CLI_VERSION} without changing the global install."
        )
    return _install_managed_kimi_cli()


def cmd_provider_setup(args) -> int:
    """Read a DeepSeek key without echoing it or placing it in argv/history."""

    if args.provider == "grok":
        return _setup_grok_subscription()
    if args.provider == "kimi":
        return _setup_kimi_subscription()
    if args.provider == "zcode":
        return _setup_zcode()
    if args.provider != "deepseek":
        raise ValueError(f"unsupported provider: {args.provider}")
    if not sys.stdin.isatty():
        print(
            "DeepSeek setup needs an interactive terminal so the key can be "
            "entered with echo disabled. Open your own Terminal and run:\n"
            "  dradar provider setup deepseek\n"
            "Never paste the API key into Codex/chat or pass it as a command argument."
        )
        return 2
    key = getpass.getpass("DeepSeek API key (input hidden): ")
    try:
        path = store_deepseek_api_key(key)
    except (OSError, ValueError) as exc:
        print(f"could not save DeepSeek API key: {exc}")
        return 1
    print(
        f"DeepSeek API key saved locally at {path} (value hidden).\n"
        "It is not stored in config.json and is never sent to the DRadar server."
    )
    if _live_deepseek_status(key) != 0:
        print(
            "The credential remains saved, but it is not ready for a task yet. "
            "Fix the reported account/network issue, then run: "
            "dradar provider status deepseek --live"
        )
        return 1
    return 0


def cmd_provider_status(args) -> int:
    """Report credential readiness without printing secret material."""

    live = bool(getattr(args, "live", False))
    if args.provider == "grok":
        return _status_grok_subscription()
    if args.provider == "kimi":
        return _status_kimi_subscription(live=live)
    if args.provider == "zcode":
        return _status_zcode(live=live)
    if args.provider != "deepseek":
        raise ValueError(f"unsupported provider: {args.provider}")
    path = deepseek_secret_path()
    error = deepseek_secret_error(path)
    if error is not None:
        print(f"DeepSeek provider not ready: {error}")
        return 1
    source = deepseek_credential_source()
    key = deepseek_api_key()
    if source == "environment" and key:
        print(f"DeepSeek provider configured via {DEEPSEEK_API_KEY_ENV} (value hidden).")
        return _live_deepseek_status(key) if live else 0
    if source == "file" and key:
        print(f"DeepSeek provider configured via {path} (value hidden).")
        return _live_deepseek_status(key) if live else 0
    print(
        "DeepSeek provider not configured. In your own interactive Terminal run:\n"
        "  dradar provider setup deepseek"
    )
    return 1


def _live_deepseek_status(key: str) -> int:
    """Verify auth and reachability without making a billable model request."""

    try:
        response = _provider_httpx_get(
            _DEEPSEEK_MODELS_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        print(
            "DeepSeek live check failed before authentication completed: "
            f"{type(exc).__name__}. Check this machine's network/proxy, then retry."
        )
        return 1
    if response.status_code == 401:
        print(
            "DeepSeek live check rejected this API key (HTTP 401). Run "
            "`dradar provider setup deepseek` to replace it."
        )
        return 1
    if response.status_code != 200:
        print(
            "DeepSeek live check failed "
            f"(HTTP {response.status_code}); the saved key was not displayed."
        )
        return 1
    try:
        payload = response.json()
    except ValueError:
        print("DeepSeek live check returned an invalid models response.")
        return 1
    available = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    } if isinstance(payload, dict) else set()
    missing = [model for model in DEEPSEEK_MODELS if model not in available]
    if missing:
        print(
            "DeepSeek authentication succeeded, but the required V4 models are "
            "not available to this account: " + ", ".join(missing)
        )
        return 1
    print("DeepSeek API authentication and V4 model availability verified live.")
    return 0


def _setup_zcode() -> int:
    if not sys.stdin.isatty():
        print(
            "ZCode setup needs an interactive terminal so the Coding Plan key "
            "can be entered with echo disabled. Run:\n"
            "  dradar provider setup zcode\n"
            "The key stays in DRadar's owner-only secret directory."
        )
        return 2
    cli = zcode_cli_path()
    issue = zcode_cli_error(cli)
    if issue is not None:
        print(
            "ZCode setup could not find a verified compatible official "
            f"desktop runtime: {issue}\n"
            f"Install it from {ZCODE_OFFICIAL_DOWNLOAD_PAGE}, then retry. "
            "Advanced users may point ZCODE_CLI_PATH at "
            "Resources/glm/zcode.cjs; DRadar verifies its SHA-256 before use."
        )
        return 1
    try:
        imported_cli = store_zcode_cli(cli)
    except (OSError, ValueError) as exc:
        print(f"could not import the verified ZCode runtime: {exc}")
        return 1
    key = getpass.getpass("BigModel Coding Plan API key (input hidden): ")
    try:
        path = store_zcode_api_key(key)
    except (OSError, ValueError) as exc:
        print(f"could not save ZCode Coding Plan API key: {exc}")
        return 1
    print(
        f"ZCode Coding Plan API key saved locally at {path} (value hidden).\n"
        f"Verified ZCode runtime imported to {imported_cli}.\n"
        "It is never sent to the DRadar server."
    )
    if _live_zcode_status(key) != 0:
        print(
            "The credential remains saved, but it is not ready for a task yet. "
            "Fix the reported account/network issue, then run: "
            "dradar provider status zcode --live"
        )
        return 1
    return 0


def _status_zcode(*, live: bool) -> int:
    path = zcode_secret_path()
    issue = zcode_secret_error(path)
    key = zcode_api_key()
    if issue is not None:
        print(f"ZCode provider not ready: {issue}")
        return 1
    if key is None:
        print(
            "ZCode provider not configured. In your own interactive Terminal run:\n"
            "  dradar provider setup zcode"
        )
        return 1
    cli = zcode_cli_path()
    issue = zcode_cli_error(cli)
    if issue is not None:
        print(f"ZCode provider not ready: {issue}")
        return 1
    node = shutil.which("node")
    if not node:
        print("ZCode provider not ready: Node.js is not available on PATH.")
        return 1
    try:
        proc = subprocess.run(
            [node, str(cli), "version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ZCode provider not ready: could not verify the CLI: {exc}")
        return 1
    found = parse_zcode_cli_version(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0 or found != ZCODE_CLI_VERSION:
        print(
            f"ZCode provider not ready: CLI {ZCODE_CLI_VERSION} required, "
            f"found {found or 'unknown'}."
        )
        return 1
    source = zcode_credential_source()
    print(
        f"ZCode provider ready via {source or 'local credential'} "
        f"(value hidden, CLI {ZCODE_CLI_VERSION}, models "
        f"{', '.join(sorted(ZCODE_MODELS))})."
    )
    return _live_zcode_status(key) if live else 0


def _live_zcode_status(key: str) -> int:
    """Check the domestic Coding Plan catalog without starting a paid turn."""

    try:
        response = _provider_httpx_get(
            _ZCODE_MODELS_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        print(
            "ZCode live check failed before authentication completed: "
            f"{type(exc).__name__}. Check this machine's network/proxy, then retry."
        )
        return 1
    if response.status_code in {401, 403}:
        print(
            f"ZCode live check rejected this Coding Plan key (HTTP "
            f"{response.status_code}). Run `dradar provider setup zcode` to replace it."
        )
        return 1
    if response.status_code != 200:
        print(
            f"ZCode live check failed (HTTP {response.status_code}); the saved "
            "key was not displayed."
        )
        return 1
    try:
        payload = response.json()
    except ValueError:
        print("ZCode live check returned an invalid models response.")
        return 1
    available = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    } if isinstance(payload, dict) else set()
    missing = sorted(ZCODE_MODELS - available)
    if missing:
        print(
            "ZCode authentication succeeded, but the following models are not "
            f"available: {', '.join(missing)} "
            "to this Coding Plan account."
        )
        return 1
    print(
        "ZCode Coding Plan authentication and model availability verified live: "
        + ", ".join(sorted(ZCODE_MODELS)) + "."
    )
    return 0


def _setup_grok_subscription() -> int:
    """Launch official device OAuth in a DRadar-owned GROK_HOME."""

    executable = _ensure_grok_cli()
    if not executable:
        return 1
    if grok_auth_error() is None:
        live_issue = grok_live_error(executable)
        if live_issue is None:
            print(
                f"Grok subscription provider is already ready (CLI "
                f"{GROK_CLI_VERSION}, {GROK_MODEL} verified)."
            )
            return 0
    if not sys.stdin.isatty():
        print(
            f"Grok CLI {GROK_CLI_VERSION} is ready. OAuth setup needs an "
            "interactive terminal. Run:\n"
            "  dradar provider setup grok\n"
            "This opens the official xAI device OAuth flow; no API key is accepted."
        )
        return 2
    home = grok_home()
    home.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".grok-login-", dir=home.parent,
        ) as name:
            native_home = Path(name)
            # Grok is a Rust binary and does not read macOS System
            # Configuration proxies itself.  Preserve explicit shell proxy
            # variables and fill otherwise-missing values from the OS proxy
            # settings, just as the installer and live probe do.
            env = provider_subprocess_env()
            env["HOME"] = str(native_home)
            env.pop("GROK_HOME", None)
            env.pop(GROK_API_KEY_ENV, None)
            print(
                "Starting official Grok device OAuth for the dedicated DRadar slot. "
                "Complete the browser/device prompt shown by Grok."
            )
            proc = subprocess.run(
                [executable, "login", "--device-auth"], env=env,
            )
            if proc.returncode != 0:
                print("Grok OAuth login did not complete successfully.")
                return proc.returncode or 1
            native_auth = native_home / ".grok" / "auth.json"
            try:
                store_grok_auth(native_auth)
            except (OSError, ValueError) as exc:
                print(f"Grok login returned but the credential is not ready: {exc}")
                return 1
    except OSError as exc:
        print(f"could not start Grok login: {exc}")
        return 1
    live_issue = grok_live_error(executable)
    if live_issue is not None:
        print(
            "Grok login completed, but the live subscription check failed: "
            f"{live_issue}. Run `dradar provider setup grok` again after fixing "
            "the account/network issue."
        )
        return 1
    print(
        f"Grok subscription OAuth is ready at {grok_auth_path()} (tokens hidden).\n"
        "The credential stays local and API-key authentication is disabled."
    )
    return 0


def _status_grok_subscription() -> int:
    executable = grok_cli_path()
    if not executable:
        print("Grok subscription provider not ready: official Grok CLI not found.")
        return 1
    found_version = _grok_cli_version(executable)
    if found_version != GROK_CLI_VERSION:
        print(
            f"Grok subscription provider not ready: CLI {GROK_CLI_VERSION} "
            f"required, found {found_version or 'unknown'}. Run "
            "`dradar provider setup grok` to prepare it automatically."
        )
        return 1
    issue = grok_auth_error()
    if issue is not None:
        print(f"Grok subscription provider not ready: {issue}")
        return 1
    live_issue = grok_live_error(executable)
    if live_issue is not None:
        print(f"Grok subscription provider not ready: {live_issue}.")
        return 1
    print(
        f"Grok subscription provider ready via {grok_auth_path()} "
        f"(OAuth tokens hidden, CLI {GROK_CLI_VERSION}, {GROK_MODEL} verified, "
        "API keys disabled)."
    )
    return 0


def _setup_kimi_subscription() -> int:
    """Launch Kimi's device OAuth in a dedicated DRadar data root."""

    executable = _ensure_kimi_cli()
    if not executable:
        return 1
    if kimi_auth_error() is None:
        live_issue = kimi_live_error(executable)
        if live_issue is None:
            print(
                f"Kimi subscription provider is already ready "
                f"(CLI {KIMI_CLI_VERSION}, K3 verified)."
            )
            return 0
    if not sys.stdin.isatty():
        print(
            f"Kimi Code CLI {KIMI_CLI_VERSION} is ready. OAuth setup needs an "
            "interactive terminal. Run:\n"
            "  dradar provider setup kimi\n"
            "This opens the official Kimi device OAuth flow; no API key is accepted."
        )
        return 2
    home = kimi_home()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(home, 0o700)
    try:
        env = provider_subprocess_env()
        env["KIMI_CODE_HOME"] = str(home)
        env["KIMI_DISABLE_TELEMETRY"] = "1"
        env["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
        for name in KIMI_API_KEY_ENVS:
            env.pop(name, None)
        print(
            "Starting official Kimi device OAuth for the dedicated DRadar slot. "
            "Complete the browser/device prompt shown by Kimi."
        )
        proc = subprocess.run([executable, "login"], env=env)
    except OSError as exc:
        print(f"could not start Kimi login: {exc}")
        return 1
    if proc.returncode != 0:
        print("Kimi OAuth login did not complete successfully.")
        return proc.returncode or 1
    path = kimi_auth_path()
    if os.name != "nt" and path.is_file():
        os.chmod(path, 0o600)
    issue = kimi_auth_error(path)
    if issue is not None:
        print(f"Kimi login returned but the credential is not ready: {issue}")
        return 1
    live_issue = kimi_live_error(executable)
    if live_issue is not None:
        print(
            "Kimi login completed, but the live K3 subscription check failed: "
            f"{live_issue}. Run `dradar provider setup kimi` again after fixing "
            "the account/network issue."
        )
        return 1
    print(
        f"Kimi subscription OAuth is ready at {path} (tokens hidden).\n"
        "The credential stays local and API-key authentication is disabled."
    )
    return 0


def _status_kimi_subscription(*, live: bool = True) -> int:
    executable = kimi_cli_path()
    if not executable:
        print("Kimi subscription provider not ready: official Kimi CLI not found.")
        return 1
    found_version = _kimi_cli_version(executable)
    if found_version != KIMI_CLI_VERSION:
        print(
            f"Kimi subscription provider not ready: CLI {KIMI_CLI_VERSION} "
            f"required, found {found_version or 'unknown'}. Run "
            "`dradar provider setup kimi` to prepare it automatically."
        )
        return 1
    issue = kimi_auth_error()
    if issue is not None:
        print(f"Kimi subscription provider not ready: {issue}")
        return 1
    # Keep the default status strict, matching Grok: a structurally valid but
    # revoked refresh token must not be reported as ready. `live` is accepted
    # for CLI symmetry and future callers; both paths are intentionally live.
    del live
    live_issue = kimi_live_error(executable)
    if live_issue is not None:
        print(f"Kimi subscription provider not ready: {live_issue}.")
        return 1
    print(
        f"Kimi subscription provider ready via {kimi_auth_path()} "
        f"(OAuth tokens hidden, CLI {KIMI_CLI_VERSION}, K3 verified, "
        "API keys disabled)."
    )
    return 0


__all__ = ["cmd_provider_setup", "cmd_provider_status"]
