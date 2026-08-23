"""Dependency-light Checkpoint V2 rollout negotiation.

This module intentionally imports only the Python standard library so the
container-native capture helper can use the same activation ABI without
loading HTTP, credentials, legacy checkpoint discovery, or Provider modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping


CHECKPOINT_CORE_ABI_V2 = "dradar-checkpoint-core-v2/1"


class CheckpointV2ProtocolError(RuntimeError):
    pass


class CheckpointRolloutModeV2(IntEnum):
    """Ordered permission level; negotiation always selects the lower side."""

    OFF = 0
    OBSERVE = 1
    RESTORE_TEST = 2
    CANARY = 3
    ON = 4

    @classmethod
    def parse(cls, value: object, *, source: str) -> "CheckpointRolloutModeV2":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise CheckpointV2ProtocolError(
                f"checkpoint v2 {source} rollout mode is invalid"
            )
        normalized = value.strip().upper().replace("-", "_")
        try:
            return cls[normalized]
        except KeyError as exc:
            raise CheckpointV2ProtocolError(
                f"checkpoint v2 {source} rollout mode is invalid"
            ) from exc

    @property
    def wire_value(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class CheckpointActivationV2:
    """Effective optional capability after two-sided negotiation."""

    local_mode: CheckpointRolloutModeV2
    server_mode: CheckpointRolloutModeV2
    effective_mode: CheckpointRolloutModeV2
    controlled_account: bool

    @property
    def capture_enabled(self) -> bool:
        return self.effective_mode >= CheckpointRolloutModeV2.OBSERVE

    @property
    def offline_restore_enabled(self) -> bool:
        return self.effective_mode >= CheckpointRolloutModeV2.RESTORE_TEST

    @property
    def paid_resume_enabled(self) -> bool:
        return self.effective_mode in {
            CheckpointRolloutModeV2.CANARY,
            CheckpointRolloutModeV2.ON,
        }

    @property
    def authoritative(self) -> bool:
        return self.paid_resume_enabled

    @property
    def writer_failure_changes_assignment(self) -> bool:
        return False

    @property
    def failure_disposition(self) -> str:
        return "continue_without_checkpoint"


def negotiate_checkpoint_activation_v2(
    *,
    local_mode: object = "off",
    server_mode: object = "off",
    controlled_account: bool = False,
) -> CheckpointActivationV2:
    local = CheckpointRolloutModeV2.parse(local_mode, source="local")
    server = CheckpointRolloutModeV2.parse(server_mode, source="server")
    effective = min(local, server)
    if effective == CheckpointRolloutModeV2.CANARY and not controlled_account:
        effective = CheckpointRolloutModeV2.RESTORE_TEST
    return CheckpointActivationV2(
        local_mode=local,
        server_mode=server,
        effective_mode=effective,
        controlled_account=bool(controlled_account),
    )


def checkpoint_activation_from_assignment_v2(
    assignment: Mapping[str, Any],
    *,
    local_mode: object = "off",
) -> CheckpointActivationV2:
    server_mode = assignment.get("checkpoint_v2_rollout_mode", "off")
    controlled = assignment.get("checkpoint_v2_controlled_account", False)
    if not isinstance(controlled, bool):
        raise CheckpointV2ProtocolError(
            "checkpoint v2 controlled-account marker is invalid"
        )
    return negotiate_checkpoint_activation_v2(
        local_mode=local_mode,
        server_mode=server_mode,
        controlled_account=controlled,
    )


__all__ = [
    "CHECKPOINT_CORE_ABI_V2",
    "CheckpointActivationV2",
    "CheckpointRolloutModeV2",
    "CheckpointV2ProtocolError",
    "checkpoint_activation_from_assignment_v2",
    "negotiate_checkpoint_activation_v2",
]
