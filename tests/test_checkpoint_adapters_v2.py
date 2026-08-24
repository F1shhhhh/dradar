from __future__ import annotations

import os
from pathlib import Path

import pytest

from dradar.checkpoint_adapters_v2 import (
    CheckpointAdapterContractError,
    checkpoint_adapter_contract_v2,
    recovery_capability_for_capture_v2,
    validate_adapter_capture_root_v2,
)
from dradar.providers import (
    DEEPSEEK_PROVIDER,
    GROK_AGENT,
    GROK_PROVIDER,
    KIMI_AGENT,
    KIMI_PROVIDER,
    ZCODE_AGENT,
    ZCODE_PROVIDER,
)


@pytest.mark.parametrize(
    ("harness", "provider", "artifacts", "native_required"),
    [
        ("codex", "openai", {"sessions"}, False),
        ("codex", DEEPSEEK_PROVIDER, {"sessions"}, False),
        (
            "dsh", DEEPSEEK_PROVIDER,
            {"dsh-sessions", "dsh-attachments"}, True,
        ),
        (
            KIMI_AGENT, KIMI_PROVIDER,
            {"sessions", "session-index"}, True,
        ),
        (
            ZCODE_AGENT, ZCODE_PROVIDER,
            {"xdg-data", "zcode-rollout"}, False,
        ),
    ],
)
def test_reviewed_harness_contracts_are_exact_and_credential_free(
    harness: str,
    provider: str,
    artifacts: set[str],
    native_required: bool,
) -> None:
    contract = checkpoint_adapter_contract_v2(harness, provider)
    assert contract.harness == harness
    assert contract.provider == provider
    assert contract.checkpoint_abi == f"dradar-checkpoint-v2/{harness}/1"
    assert contract.artifact_names == artifacts
    assert contract.native_resume_required is native_required
    for artifact in contract.artifacts:
        assert "credential" not in artifact.name
        assert "secret" not in artifact.name
        assert all(
            not artifact.source_path.startswith(excluded.rstrip("/") + "/")
            and artifact.source_path != excluded
            for excluded in contract.credential_exclusion_paths
        )


@pytest.mark.parametrize(
    ("harness", "provider"),
    [
        (GROK_AGENT, GROK_PROVIDER),
        ("claude", "anthropic"),
        ("dsh", ZCODE_PROVIDER),
        (ZCODE_AGENT, DEEPSEEK_PROVIDER),
    ],
)
def test_unreviewed_or_unsupported_harness_pair_is_rejected(
    harness: str, provider: str,
) -> None:
    with pytest.raises(CheckpointAdapterContractError):
        checkpoint_adapter_contract_v2(harness, provider)


def _capture_root(root: Path) -> Path:
    root.mkdir()
    (root / "workspace.patch").write_bytes(b"")
    (root / "untracked.tar.gz").write_bytes(b"")
    (root / "progress.json").write_text('{"step":1}\n', encoding="utf-8")
    return root


def test_adapter_capture_root_allows_only_reviewed_artifacts(tmp_path: Path) -> None:
    root = _capture_root(tmp_path / "capture")
    state = root / "provider-state"
    (state / "sessions").mkdir(parents=True)
    (state / "sessions" / "state.json").write_text("{}\n", encoding="utf-8")
    (state / "session-index").write_text("{}\n", encoding="utf-8")
    (root / "session-id").write_text("session-0001\n", encoding="utf-8")
    contract = checkpoint_adapter_contract_v2(KIMI_AGENT, KIMI_PROVIDER)
    present = validate_adapter_capture_root_v2(root, contract)
    assert present == {"sessions", "session-index"}
    assert recovery_capability_for_capture_v2(
        contract,
        present_artifacts=present,
        has_session_id=True,
    ) == "NATIVE_VALID"


def test_native_required_harness_degrades_to_non_resumable_diagnostic(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path / "capture")
    contract = checkpoint_adapter_contract_v2(KIMI_AGENT, KIMI_PROVIDER)
    present = validate_adapter_capture_root_v2(root, contract)
    assert present == frozenset()
    assert recovery_capability_for_capture_v2(
        contract,
        present_artifacts=present,
        has_session_id=False,
    ) == "NONE"


def test_workspace_fallback_is_explicit_for_codex_and_zcode(tmp_path: Path) -> None:
    for harness, provider in (
        ("codex", "openai"),
        (ZCODE_AGENT, ZCODE_PROVIDER),
    ):
        root = _capture_root(tmp_path / harness)
        contract = checkpoint_adapter_contract_v2(harness, provider)
        present = validate_adapter_capture_root_v2(root, contract)
        assert recovery_capability_for_capture_v2(
            contract,
            present_artifacts=present,
            has_session_id=False,
        ) == "WORKSPACE_ONLY"


@pytest.mark.parametrize("unsafe_name", ["credentials", "unexpected", "auth.json"])
def test_adapter_capture_root_rejects_extra_top_level_entries(
    tmp_path: Path, unsafe_name: str,
) -> None:
    root = _capture_root(tmp_path / "capture")
    (root / unsafe_name).write_text("do not export\n", encoding="utf-8")
    with pytest.raises(
        CheckpointAdapterContractError, match="unexpected entries",
    ):
        validate_adapter_capture_root_v2(
            root, checkpoint_adapter_contract_v2("codex", "openai"),
        )


def test_adapter_capture_root_rejects_symlinked_provider_state(tmp_path: Path) -> None:
    root = _capture_root(tmp_path / "capture")
    external = tmp_path / "external"
    external.mkdir()
    (root / "provider-state").symlink_to(external, target_is_directory=True)
    with pytest.raises(CheckpointAdapterContractError, match="unsafe"):
        validate_adapter_capture_root_v2(
            root, checkpoint_adapter_contract_v2("codex", "openai"),
        )


def test_adapter_capture_root_rejects_hardlinked_common_artifact(tmp_path: Path) -> None:
    root = _capture_root(tmp_path / "capture")
    os.link(root / "workspace.patch", tmp_path / "second-link")
    with pytest.raises(CheckpointAdapterContractError, match="unsafe"):
        validate_adapter_capture_root_v2(
            root, checkpoint_adapter_contract_v2("codex", "openai"),
        )
