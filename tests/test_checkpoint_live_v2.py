from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from dradar.checkpoint_activation_v2 import CheckpointV2ProtocolError
from dradar.checkpoint_live_v2 import (
    LiveCheckpointShadowControllerV2,
    build_live_checkpoint_shadow_v2,
    checkpoint_live_activation_v2,
)
from dradar.checkpoint_v2 import CHECKPOINT_CORE_ABI_V2


def _assignment(mode: str = "off", *, controlled: bool = False) -> dict:
    return {
        "assignment_id": "assignment-0001",
        "checkpoint_protocol_version": 1,
        "checkpoint_v2_identity_protocol_version": 2,
        "checkpoint_v2_rollout_mode": mode,
        "checkpoint_v2_controlled_account": controlled,
        "benchmark_id": "deep-swe",
        "task_content_hash": "a" * 64,
        "agent": "codex",
        "provider": "openai",
        "model": "gpt-test",
        "effort": "high",
        "execution_identity": {
            "benchmark_id": "deep-swe",
            "task_content_sha256": "a" * 64,
            "harness": "codex",
            "provider": "openai",
            "model": "gpt-test",
            "effort": "high",
            "agent_version": None,
            "runtime_profile": None,
            "model_config_version": None,
            "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
            "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
            "identity_state": "PROVISIONAL",
            "identity_source": "claim_snapshot",
            "runtime_compatibility_digest": None,
        },
    }


def test_server_off_is_a_strict_lazy_noop(tmp_path: Path) -> None:
    home = tmp_path / "never-created"
    result = build_live_checkpoint_shadow_v2(
        assignment=_assignment("off"),
        effective_assignment={"agent_version": "1.2.3"},
        local_mode="restore-test",
        api=object(),
        telemetry=object(),
        home=home,
        job_root=tmp_path / "missing-job",
    )
    assert result is None
    assert not home.exists()


def test_authoritative_mode_cannot_fall_back_to_shadow_mainline() -> None:
    with pytest.raises(
        CheckpointV2ProtocolError,
        match="requires the V2 owner runner",
    ):
        checkpoint_live_activation_v2(
            _assignment("canary", controlled=True),
            local_mode="canary",
        )


def test_uncontrolled_canary_is_capped_at_restore_test() -> None:
    activation = checkpoint_live_activation_v2(
        _assignment("canary", controlled=False),
        local_mode="canary",
    )
    assert activation.effective_mode.wire_value == "restore_test"
    assert activation.authoritative is False


class _Coordinator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.released = threading.Event()
        self.impact_sample_id = "impact-" + "1" * 48

    async def run(self, mainline, **_kwargs):
        self.started.set()
        try:
            return await mainline
        finally:
            self.stopped.set()

    async def release_local_snapshots(self):
        self.released.set()


def test_live_controller_has_a_bounded_non_authoritative_lifetime() -> None:
    coordinator = _Coordinator()
    controller = LiveCheckpointShadowControllerV2(
        coordinator, initial_delay_sec=0, interval_sec=0.01,
    )
    controller.start()
    assert coordinator.started.wait(1)
    controller.close(timeout=1)
    assert coordinator.stopped.wait(1)
    assert coordinator.released.wait(1)
    assert controller.impact_sample_id == coordinator.impact_sample_id
    assert controller.checkpoint_sync_elapsed_ms >= 0
    assert controller.mainline_elapsed_ms >= 1


def test_live_controller_swallows_shadow_exception() -> None:
    class Broken:
        async def run(self, mainline, **_kwargs):
            close = getattr(mainline, "close", None)
            if callable(close):
                close()
            raise RuntimeError("shadow-only failure")

        async def release_local_snapshots(self):
            raise RuntimeError("shadow-only cleanup failure")

    controller = LiveCheckpointShadowControllerV2(
        Broken(), initial_delay_sec=0, interval_sec=0.01,
    )
    controller.start()
    controller.close(timeout=1)
    assert controller._thread is not None
    assert not controller._thread.is_alive()
