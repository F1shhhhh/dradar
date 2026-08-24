"""Dependency-free value types shared by Checkpoint V2 host and helper.

The container helper must not import the HTTP client, Provider registry or
assignment command journal merely to validate a retention reference. Keeping
these immutable wire-neutral values here preserves the dependency boundary of
the checksum-pinned zipapp.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointGenerationRefV2:
    """Exact, content-bound identity of one retained local generation."""

    checkpoint_id: str
    snapshot_generation: int
    manifest_sha256: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (
            self.checkpoint_id,
            self.snapshot_generation,
            self.manifest_sha256,
        )


@dataclass(frozen=True)
class CheckpointRetentionAcknowledgementV2:
    """Server-owned evidence release decision safe for a local consumer."""

    assignment_id: str
    operation_id: str
    owner_epoch_observed: int
    current_owner_epoch: int
    delete_generations: tuple[CheckpointGenerationRefV2, ...]
    retain_generations: tuple[CheckpointGenerationRefV2, ...]
    result_evidence_release: bool
    upload_intent_id: str | None
    submission_id: str | None


__all__ = [
    "CheckpointGenerationRefV2",
    "CheckpointRetentionAcknowledgementV2",
]
