from pathlib import Path

from dradar import providers


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return Path(providers.__file__).with_name(name).read_text(encoding="utf-8")


def test_honey_contract_is_linked_and_names_every_supported_honey() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    contract = (ROOT / "docs" / "HONEY_EXECUTION_SECURITY.md").read_text(
        encoding="utf-8",
    )
    assert "docs/HONEY_EXECUTION_SECURITY.md" in readme
    for name in ("Codex", "DSH", "ZCode", "Kimi Code", "Antigravity"):
        assert name in contract
    assert "新 Honey 接入门禁" in contract
    assert "子代理" in contract
    assert "Docker socket" in contract
    assert "egress" in contract
    for field in (
        "honey_execution_security_profile",
        "honey_inner_permission_mode",
        "honey_child_agent_access",
        "honey_outer_isolation",
    ):
        assert field in contract


def test_all_non_codex_honeys_use_full_container_permissions() -> None:
    dsh = _source("pier_dsh.py")
    assert '"DSH_PERMISSION_MODE": "danger-full-access"' in dsh
    for row in (
        "tool-subagent-control",
        "tool-subagent-list-agents",
        "tool-subagent",
        "tool-subagent-fork",
        "tool-subagent-report",
    ):
        assert f"- id: {row}\n  disabled: true" not in dsh

    zcode = _source("pier_zcode.py")
    assert '"mode": "yolo"' in zcode
    assert "tool_allowlist" not in zcode
    assert "tool_denylist" not in zcode
    assert 'required_tools = {"Read", "Write", "Edit", "Bash", "Agent"}' in zcode

    kimi = _source("pier_kimi.py")
    assert 'default_permission_mode = "auto"' in kimi
    # Kimi 0.39.1 prompt mode rejects the redundant CLI ``--auto`` flag.  The
    # isolated config remains the single source of truth for full permissions.
    assert '"--auto"' not in kimi
    assert "[tools]" not in kimi
    assert "KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY" not in kimi

    antigravity = _source("pier_antigravity.py")
    assert '"--dangerously-skip-permissions"' in antigravity
    assert '"--sandbox"' not in antigravity
    assert 'init.get("permission_mode") == "always-proceed"' in antigravity

    assert providers.HONEY_SECURITY_AGENTS == {
        "codex", "dsh-minimal", "zcode", "kimi-code", "antigravity",
    }


def test_outer_network_boundary_remains_provider_scoped() -> None:
    assert 'return NetworkAllowlist(domains=["api.deepseek.com"])' in _source(
        "pier_dsh.py",
    )
    assert 'return NetworkAllowlist(domains=["auth.kimi.com", "api.kimi.com"])' in _source(
        "pier_kimi.py",
    )
    assert 'return NetworkAllowlist(domains=["open.bigmodel.cn", "zcode.z.ai"])' in _source(
        "pier_zcode.py",
    )
    antigravity = _source("pier_antigravity.py")
    assert "*.googleapis.com" not in antigravity
    assert "*.googleusercontent.com" not in antigravity
