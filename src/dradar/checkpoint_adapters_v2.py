"""Reviewed Harness contracts for optional Checkpoint V2 snapshots.

The shared data plane never guesses a CLI's home directory or session shape.
Each supported Harness exposes an immutable adapter contract describing only
the state it may copy into a container-native capture root.  Credential,
configuration, cache, and arbitrary log trees are deliberately absent.

This module contains no model invocation and no assignment-state mutation.
Grok remains explicitly unsupported and Claude is outside the current scope.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEEPSEEK_PROVIDER = "deepseek"
GROK_AGENT = "grok-build"
GROK_PROVIDER = "xai-subscription"
KIMI_AGENT = "kimi-code"
KIMI_PROVIDER = "kimi-subscription"
ZCODE_AGENT = "zcode"
ZCODE_PROVIDER = "bigmodel-coding-plan"


COMMON_CAPTURE_FILES_V2 = {
    "workspace.patch": 16 * 1024 * 1024,
    "untracked.tar.gz": 512 * 1024 * 1024,
    "progress.json": 256 * 1024,
}
PROVIDER_STATE_DIR_V2 = "provider-state"
SESSION_ID_FILE_V2 = "session-id"


class CheckpointAdapterContractError(ValueError):
    pass


@dataclass(frozen=True)
class NativeStateArtifactV2:
    name: str
    source_path: str
    kind: str
    restore: bool
    max_bytes: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or "/" in self.name
            or self.name in {".", ".."}
            or self.kind not in {"file", "directory"}
            or not isinstance(self.restore, bool)
            or not isinstance(self.max_bytes, int)
            or isinstance(self.max_bytes, bool)
            or self.max_bytes <= 0
        ):
            raise ValueError("checkpoint native-state artifact is invalid")
        source = PurePosixPath(self.source_path)
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("checkpoint native-state source is unsafe")


@dataclass(frozen=True)
class HarnessCheckpointContractV2:
    harness: str
    provider: str
    checkpoint_abi: str
    exporter_version: str
    restorer_version: str
    native_state_schema: str
    native_resume_required: bool
    usage_ledger_scope: str
    artifacts: tuple[NativeStateArtifactV2, ...]
    credential_exclusion_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.checkpoint_abi != f"dradar-checkpoint-v2/{self.harness}/1":
            raise ValueError("checkpoint adapter ABI does not match Harness")
        if self.usage_ledger_scope not in {
            "assignment_cumulative", "segment_delta",
        }:
            raise ValueError("checkpoint adapter usage ledger scope is invalid")
        names = [item.name for item in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint adapter has duplicate artifact names")
        exclusions = tuple(
            PurePosixPath(value) for value in self.credential_exclusion_paths
        )
        if any(
            not value.is_absolute() or ".." in value.parts
            for value in exclusions
        ):
            raise ValueError("checkpoint credential exclusion path is unsafe")
        for artifact in self.artifacts:
            source = PurePosixPath(artifact.source_path)
            if any(
                source == exclusion or source.is_relative_to(exclusion)
                for exclusion in exclusions
            ):
                raise ValueError(
                    "checkpoint artifact overlaps credential exclusion path"
                )

    @property
    def artifact_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.artifacts)

    @property
    def restorable_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.artifacts if item.restore)


_MIB = 1024 * 1024
_CONTRACTS = {
    ("codex", "openai"): HarnessCheckpointContractV2(
        harness="codex",
        provider="openai",
        checkpoint_abi="dradar-checkpoint-v2/codex/1",
        exporter_version="dradar-codex-checkpoint-export-v2/1",
        restorer_version="dradar-codex-checkpoint-restore-v2/1",
        native_state_schema="codex-sessions/1",
        native_resume_required=False,
        usage_ledger_scope="assignment_cumulative",
        artifacts=(
            NativeStateArtifactV2(
                "sessions", "/tmp/codex-home/sessions", "directory", True,
                384 * _MIB,
            ),
        ),
        credential_exclusion_paths=(
            "/tmp/codex-secrets", "/tmp/codex-home/auth.json",
        ),
    ),
    ("codex", DEEPSEEK_PROVIDER): HarnessCheckpointContractV2(
        harness="codex",
        provider=DEEPSEEK_PROVIDER,
        checkpoint_abi="dradar-checkpoint-v2/codex/1",
        exporter_version="dradar-deepseek-codex-checkpoint-export-v2/1",
        restorer_version="dradar-deepseek-codex-checkpoint-restore-v2/1",
        native_state_schema="codex-sessions/1",
        native_resume_required=False,
        usage_ledger_scope="assignment_cumulative",
        artifacts=(
            NativeStateArtifactV2(
                "sessions", "/tmp/codex-home/sessions", "directory", True,
                384 * _MIB,
            ),
        ),
        credential_exclusion_paths=(
            "/tmp/codex-secrets", "/tmp/codex-home/auth.json",
        ),
    ),
    ("dsh", DEEPSEEK_PROVIDER): HarnessCheckpointContractV2(
        harness="dsh",
        provider=DEEPSEEK_PROVIDER,
        checkpoint_abi="dradar-checkpoint-v2/dsh/1",
        exporter_version="dradar-dsh-checkpoint-export-v2/1",
        restorer_version="dradar-dsh-checkpoint-restore-v2/1",
        native_state_schema="dsh-sessions/1",
        native_resume_required=True,
        usage_ledger_scope="assignment_cumulative",
        artifacts=(
            NativeStateArtifactV2(
                "dsh-sessions", "/logs/agent/dsh-home/sessions",
                "directory", True, 384 * _MIB,
            ),
            NativeStateArtifactV2(
                "dsh-attachments", "/logs/agent/dsh-home/attachments",
                "directory", True, 96 * _MIB,
            ),
        ),
        credential_exclusion_paths=("/run/secrets/deepseek-api-key",),
    ),
    (KIMI_AGENT, KIMI_PROVIDER): HarnessCheckpointContractV2(
        harness=KIMI_AGENT,
        provider=KIMI_PROVIDER,
        checkpoint_abi=f"dradar-checkpoint-v2/{KIMI_AGENT}/1",
        exporter_version="dradar-kimi-checkpoint-export-v2/1",
        restorer_version="dradar-kimi-checkpoint-restore-v2/1",
        native_state_schema="kimi-k3-sessions/1",
        native_resume_required=True,
        usage_ledger_scope="segment_delta",
        artifacts=(
            NativeStateArtifactV2(
                "sessions", "/tmp/dradar-kimi-home/sessions",
                "directory", True, 384 * _MIB,
            ),
            NativeStateArtifactV2(
                "session-index", "/tmp/dradar-kimi-home/session_index.jsonl",
                "file", True, 16 * _MIB,
            ),
        ),
        credential_exclusion_paths=(
            "/tmp/dradar-kimi-home/credentials",
            "/tmp/dradar-kimi-home/oauth",
            "/tmp/dradar-kimi-home/config.toml",
        ),
    ),
    (ZCODE_AGENT, ZCODE_PROVIDER): HarnessCheckpointContractV2(
        harness=ZCODE_AGENT,
        provider=ZCODE_PROVIDER,
        checkpoint_abi=f"dradar-checkpoint-v2/{ZCODE_AGENT}/1",
        exporter_version="dradar-zcode-checkpoint-export-v2/1",
        restorer_version="dradar-zcode-checkpoint-restore-v2/1",
        native_state_schema="zcode-rollout/1",
        native_resume_required=False,
        usage_ledger_scope="segment_delta",
        artifacts=(
            NativeStateArtifactV2(
                "xdg-data", "/tmp/dradar-zcode-home/data",
                "directory", True, 256 * _MIB,
            ),
            NativeStateArtifactV2(
                "zcode-rollout", "/tmp/dradar-zcode-user/.zcode/cli/rollout",
                "directory", True, 256 * _MIB,
            ),
        ),
        credential_exclusion_paths=("/tmp/dradar-zcode-secrets",),
    ),
}


def checkpoint_adapter_contract_v2(
    harness: str,
    provider: str,
) -> HarnessCheckpointContractV2:
    if (harness, provider) == (GROK_AGENT, GROK_PROVIDER):
        raise CheckpointAdapterContractError(
            "Grok does not support checkpoint v2"
        )
    try:
        return _CONTRACTS[(harness, provider)]
    except KeyError as exc:
        raise CheckpointAdapterContractError(
            "Harness/provider has no reviewed checkpoint v2 adapter"
        ) from exc


def _bounded_regular_file(path: Path, max_bytes: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointAdapterContractError(
            f"required checkpoint artifact is unavailable: {path.name}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_nlink != 1
        or metadata.st_size > max_bytes
    ):
        raise CheckpointAdapterContractError(
            f"checkpoint artifact is unsafe: {path.name}"
        )


def _validate_native_artifact_payload_v2(
    path: Path,
    artifact: NativeStateArtifactV2,
) -> bool:
    """Return whether an artifact has material state, enforcing its own cap."""

    if artifact.kind == "file":
        _bounded_regular_file(path, artifact.max_bytes)
        return path.stat().st_size > 0
    total = 0
    material = False
    entries = 0
    try:
        for current, directory_names, file_names in os.walk(
            path, topdown=True, followlinks=False,
        ):
            current_path = Path(current)
            for name in directory_names:
                candidate = current_path / name
                metadata = candidate.lstat()
                entries += 1
                if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                    raise CheckpointAdapterContractError(
                        f"checkpoint native-state artifact is unsafe: {artifact.name}"
                    )
            for name in file_names:
                candidate = current_path / name
                metadata = candidate.lstat()
                entries += 1
                if (
                    candidate.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise CheckpointAdapterContractError(
                        f"checkpoint native-state artifact is unsafe: {artifact.name}"
                    )
                total += metadata.st_size
                material = material or metadata.st_size > 0
                if total > artifact.max_bytes or entries > 20_000:
                    raise CheckpointAdapterContractError(
                        f"checkpoint native-state artifact is oversized: {artifact.name}"
                    )
    except OSError as exc:
        raise CheckpointAdapterContractError(
            f"checkpoint native-state artifact is unreadable: {artifact.name}"
        ) from exc
    return material


def validate_adapter_capture_root_v2(
    root: Path,
    contract: HarnessCheckpointContractV2,
) -> frozenset[str]:
    """Validate the exact adapter-owned layout before the shared seal step."""

    root = Path(root)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise CheckpointAdapterContractError(
            "checkpoint adapter capture root is unavailable"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise CheckpointAdapterContractError(
            "checkpoint adapter capture root is unsafe"
        )
    allowed = set(COMMON_CAPTURE_FILES_V2) | {
        PROVIDER_STATE_DIR_V2, SESSION_ID_FILE_V2,
    }
    entries = {item.name for item in os.scandir(root)}
    if not set(COMMON_CAPTURE_FILES_V2) <= entries or not entries <= allowed:
        raise CheckpointAdapterContractError(
            "checkpoint adapter capture root has missing or unexpected entries"
        )
    for name, max_bytes in COMMON_CAPTURE_FILES_V2.items():
        _bounded_regular_file(root / name, max_bytes)
    if SESSION_ID_FILE_V2 in entries:
        _bounded_regular_file(root / SESSION_ID_FILE_V2, 512)
    provider_state = root / PROVIDER_STATE_DIR_V2
    present: set[str] = set()
    if provider_state.exists() or provider_state.is_symlink():
        state_metadata = provider_state.lstat()
        if not stat.S_ISDIR(state_metadata.st_mode) or provider_state.is_symlink():
            raise CheckpointAdapterContractError(
                "checkpoint provider-state root is unsafe"
            )
        artifact_by_name = {item.name: item for item in contract.artifacts}
        state_entries = {item.name for item in os.scandir(provider_state)}
        if not state_entries <= set(artifact_by_name):
            raise CheckpointAdapterContractError(
                "checkpoint provider-state has an unexpected artifact"
            )
        for name in state_entries:
            artifact = artifact_by_name[name]
            path = provider_state / name
            artifact_metadata = path.lstat()
            if artifact.kind == "file":
                _bounded_regular_file(path, artifact.max_bytes)
            elif (
                not stat.S_ISDIR(artifact_metadata.st_mode)
                or path.is_symlink()
            ):
                raise CheckpointAdapterContractError(
                    f"checkpoint native-state artifact is unsafe: {name}"
                )
            if _validate_native_artifact_payload_v2(path, artifact):
                present.add(name)
    elif contract.native_resume_required:
        # A workspace-only diagnostic capture is still valid.  Its recovery
        # capability is NONE for native-required Harnesses, determined below.
        present.clear()
    return frozenset(present)


def recovery_capability_for_capture_v2(
    contract: HarnessCheckpointContractV2,
    *,
    present_artifacts: frozenset[str],
    has_session_id: bool,
) -> str:
    if not present_artifacts <= contract.artifact_names:
        raise CheckpointAdapterContractError(
            "checkpoint capability references an unknown native artifact"
        )
    has_native = contract.restorable_names <= present_artifacts and has_session_id
    if has_native:
        return "NATIVE_VALID"
    if contract.native_resume_required:
        return "NONE"
    return "WORKSPACE_ONLY"


__all__ = [
    "COMMON_CAPTURE_FILES_V2",
    "PROVIDER_STATE_DIR_V2",
    "SESSION_ID_FILE_V2",
    "CheckpointAdapterContractError",
    "HarnessCheckpointContractV2",
    "NativeStateArtifactV2",
    "checkpoint_adapter_contract_v2",
    "recovery_capability_for_capture_v2",
    "validate_adapter_capture_root_v2",
]
