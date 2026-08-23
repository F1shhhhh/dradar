from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

import dradar.checkpoint_shadow_v2 as shadow_runtime
from dradar.api_client import ApiError
from dradar.checkpoint_runtime_v2 import (
    CheckpointObservationV2,
    CheckpointRestoreEvidenceV2,
    CheckpointRestoreObservationV2,
    PublishedCheckpointV2,
)
from dradar.checkpoint_shadow_v2 import (
    CheckpointShadowCoordinatorV2,
    CheckpointShadowRuntimeFactsV2,
    run_mainline_with_optional_checkpoint_shadow_v2,
)
from dradar.checkpoint_v2 import (
    CHECKPOINT_CORE_ABI_V2,
    CheckpointV2Journal,
    CheckpointV2ProtocolError,
    ExecutionIdentityV2,
    negotiate_checkpoint_activation_v2,
)


def _assignment(mode: str) -> dict:
    wire_mode = mode.replace("-", "_")
    return {
        "assignment_id": "assignment-0001",
        "checkpoint_protocol_version": 1,
        "checkpoint_v2_identity_protocol_version": 2,
        "checkpoint_v2_rollout_mode": wire_mode,
        "benchmark_id": "deep-swe",
        "task_content_hash": "a" * 64,
        "agent": "codex",
        "provider": "openai",
        "model": "model-v2",
        "effort": "high",
        "agent_version": "1.2.3",
        "execution_identity": {
            "benchmark_id": "deep-swe",
            "task_content_sha256": "a" * 64,
            "harness": "codex",
            "provider": "openai",
            "model": "model-v2",
            "effort": "high",
            "agent_version": "1.2.3",
            "runtime_profile": None,
            "model_config_version": None,
            "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
            "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
            "identity_state": "PROVISIONAL",
            "identity_source": "claim_snapshot",
            "runtime_compatibility_digest": None,
        },
    }


class IdentityApi:
    def __init__(
        self,
        assignment: dict,
        *,
        error: Exception | None = None,
        current_mode: str | None = None,
        kill_switch: bool = False,
    ):
        self.assignment = assignment
        self.error = error
        self.calls = []
        self.activation_calls = []
        self.current_mode = current_mode or assignment[
            "checkpoint_v2_rollout_mode"
        ]
        self.kill_switch = kill_switch

    def checkpoint_v2_command(self, command, payload):
        self.calls.append((command, dict(payload)))
        if self.error is not None:
            raise self.error
        finalized = dict(self.assignment)
        finalized["execution_identity"] = {
            **self.assignment["execution_identity"],
            "harness": payload["harness"],
            "provider": payload["provider"],
            "model": payload["model"],
            "effort": payload["effort"],
            "agent_version": payload["agent_version"],
            "runtime_profile": payload["runtime_profile"],
            "model_config_version": payload["model_config_version"],
            "checkpoint_core_abi": payload["checkpoint_core_abi"],
            "checkpoint_abi": payload["checkpoint_abi"],
            "runtime_compatibility_digest": (
                payload["runtime_compatibility_digest"]
            ),
            "identity_state": "FINAL",
        }
        identity = ExecutionIdentityV2.from_assignment(finalized)
        return {
            "ok": True,
            "assignment_id": payload["assignment_id"],
            "identity_state": "FINAL",
            "checkpoint_protocol_version": 1,
            "checkpoint_v2_identity_protocol_version": 2,
            "checkpoint_core_abi": payload["checkpoint_core_abi"],
            "checkpoint_abi": payload["checkpoint_abi"],
            "runtime_compatibility_digest": (
                payload["runtime_compatibility_digest"]
            ),
            "identity_fingerprint": identity.fingerprint,
            "assignment_ownership_unchanged": True,
            "paid_execution_authorized": False,
        }

    def checkpoint_v2_activation(self, payload):
        self.activation_calls.append(dict(payload))
        if self.error is not None:
            raise self.error
        ranks = {
            "off": 0, "observe": 1, "restore_test": 2,
            "canary": 3, "on": 4,
        }
        snapshot = self.assignment["checkpoint_v2_rollout_mode"]
        current = "off" if self.kill_switch else self.current_mode
        effective = min((snapshot, current), key=ranks.__getitem__)
        capture = ranks[effective] >= ranks["observe"]
        restore = ranks[effective] >= ranks["restore_test"]
        reason = (
            "kill_switch" if self.kill_switch
            else "enabled" if capture else "rollout_lowered"
        )
        return {
            "ok": True,
            "assignment_id": payload["assignment_id"],
            "identity_fingerprint": payload["identity_fingerprint"],
            "requested_rollout_mode": payload["requested_rollout_mode"],
            "assignment_snapshot_mode": snapshot,
            "current_server_mode": current,
            "effective_rollout_mode": effective,
            "capture_authorized": capture,
            "restore_test_authorized": restore,
            "kill_switch_active": self.kill_switch,
            "reason": reason,
            "assignment_unchanged": True,
            "paid_execution_authorized": False,
        }


class Exporter:
    adapter_version = "capture-adapter/1"
    checkpoint_abi = "dradar-checkpoint-v2/codex/1"


class Restorer:
    adapter_version = "restore-adapter/1"
    checkpoint_abi = "dradar-checkpoint-v2/codex/1"


class DataPlane:
    def __init__(self, activation, storage_root: Path):
        self.activation = activation
        self.storage_root = storage_root
        self.captures = 0
        self.restores = 0

    async def observe_capture(self, request, _exporter):
        self.captures += 1
        root = self.storage_root / "synthetic" / request.capture_id
        published = PublishedCheckpointV2(
            checkpoint_id=request.checkpoint_id,
            snapshot_generation=request.snapshot_generation,
            capture_id=request.capture_id,
            root=root,
            payload_root=root / "payload",
            archive_path=root / "export.tar.gz",
            manifest_sha256="b" * 64,
            archive_sha256="c" * 64,
            archive_bytes=4096,
            file_count=3,
            payload_bytes=2048,
            authoritative=False,
            selected=True,
        )
        return CheckpointObservationV2(
            status="sealed",
            capture_id=request.capture_id,
            published=published,
            remote_cleanup="discarded",
        )

    async def observe_offline_restore(self, request, restorer):
        self.restores += 1
        return CheckpointRestoreObservationV2(
            status="verified",
            restore_id=request.restore_id,
            evidence=CheckpointRestoreEvidenceV2(
                restore_id=request.restore_id,
                manifest_sha256=request.published.manifest_sha256,
                identity_fingerprint=request.expected_identity_fingerprint,
                restore_adapter_version=restorer.adapter_version,
                paid_execution_started=False,
            ),
        )


class Sink:
    def __init__(self):
        self.payloads = []
        self.cohorts = []

    def record_checkpoint_observation(self, payload):
        self.payloads.append(payload)
        return True

    def register_checkpoint_cohort(self, payload):
        self.cohorts.append(payload)
        return True


def _runtime() -> CheckpointShadowRuntimeFactsV2:
    return CheckpointShadowRuntimeFactsV2(
        harness="codex",
        provider="openai",
        agent_version="1.2.3",
        runtime_profile="runtime-profile-v2",
        model_config_version="model-config-v2",
        checkpoint_abi="dradar-checkpoint-v2/codex/1",
        runtime_compatibility_digest="d" * 64,
        recovery_capability="NATIVE_VALID",
        native_state_schema="codex-sessions/1",
        machine_fingerprint="e" * 64,
        platform="macos",
        container_backend="orbstack",
        capture_adapter_version="capture-adapter/1",
        client_version="0.5.98",
    )


def _coordinator(tmp_path: Path, mode: str, *, api=None):
    assignment = _assignment(mode)
    activation = negotiate_checkpoint_activation_v2(
        local_mode=mode, server_mode=mode,
    )
    plane = DataPlane(activation, tmp_path / "storage")
    sink = Sink()
    coordinator = CheckpointShadowCoordinatorV2(
        assignment=assignment,
        activation=activation,
        api=api or IdentityApi(assignment),
        journal=CheckpointV2Journal(tmp_path / "home"),
        data_plane=plane,
        exporter=Exporter(),
        observation_sink=sink,
        runtime=_runtime(),
        restorer_factory=(Restorer if mode == "restore-test" else None),
    )
    return coordinator, plane, sink


def test_observe_coordinator_keeps_mainline_authoritative_and_reports_capture(
    tmp_path: Path,
) -> None:
    coordinator, plane, sink = _coordinator(tmp_path, "observe")

    async def mainline():
        await asyncio.sleep(0.02)
        return {"submission": "authoritative"}

    result = asyncio.run(coordinator.run(
        mainline(), initial_delay_sec=0, interval_sec=0.01,
        maximum_captures=1,
    ))
    assert result == {"submission": "authoritative"}
    assert plane.captures == 1
    assert plane.restores == 0
    assert [item["observation_kind"] for item in sink.payloads] == ["capture"]
    assert sink.payloads[0]["authoritative"] is False
    assert len(sink.cohorts) == 1
    assert sink.cohorts[0]["cohort"] == {
        "platform": "macos",
        "container_backend": "orbstack",
        "harness": "codex",
        "provider": "openai",
        "client_version": "0.5.98",
        "agent_version": "1.2.3",
        "runtime_profile": "runtime-profile-v2",
        "model_config_version": "model-config-v2",
        "runtime_compatibility_digest": "d" * 64,
        "checkpoint_core_abi": CHECKPOINT_CORE_ABI_V2,
        "checkpoint_abi": "dradar-checkpoint-v2/codex/1",
    }


def test_restore_test_reports_capture_before_nonpaid_restore(tmp_path: Path) -> None:
    coordinator, plane, sink = _coordinator(tmp_path, "restore-test")

    async def mainline():
        await asyncio.sleep(0.02)
        return "done"

    assert asyncio.run(coordinator.run(
        mainline(), initial_delay_sec=0, interval_sec=0.01,
        maximum_captures=1,
    )) == "done"
    assert plane.captures == plane.restores == 1
    assert [item["observation_kind"] for item in sink.payloads] == [
        "capture", "restore",
    ]
    assert sink.payloads[1]["paid_execution_started"] is False
    assert sink.payloads[1]["authoritative"] is False
    assert sink.payloads[1]["source_capture_id"] == sink.payloads[0]["capture_id"]


@pytest.mark.parametrize("mode", ["observe", "restore-test"])
def test_clean_consumed_shadow_snapshot_is_released_off_mainline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    coordinator, _plane, _sink = _coordinator(tmp_path, mode)
    released = []

    def release(storage_root, published, **identity):
        released.append((storage_root, published, identity))
        return True

    monkeypatch.setattr(
        shadow_runtime, "release_shadow_generation_v2", release,
    )

    async def exercise():
        async def mainline():
            await asyncio.sleep(0.02)
            return "done"

        assert await coordinator.run(
            mainline(), initial_delay_sec=0, interval_sec=0.01,
            maximum_captures=1,
        ) == "done"
        await coordinator.release_local_snapshots()

    asyncio.run(exercise())
    assert len(released) == 1
    assert released[0][0] == coordinator.data_plane.storage_root
    assert released[0][2] == {
        "expected_identity_fingerprint": coordinator._identity.fingerprint,
        "expected_checkpoint_abi": "dradar-checkpoint-v2/codex/1",
    }


def test_restore_skipped_by_dynamic_downgrade_preserves_local_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignment = _assignment("restore-test")
    api = IdentityApi(assignment)
    original_activation = api.checkpoint_v2_activation

    def changing_activation(payload):
        response = original_activation(payload)
        api.current_mode = "observe"
        return response

    api.checkpoint_v2_activation = changing_activation
    coordinator, plane, _sink = _coordinator(
        tmp_path, "restore-test", api=api,
    )
    released = []
    monkeypatch.setattr(
        shadow_runtime,
        "release_shadow_generation_v2",
        lambda *args, **kwargs: released.append((args, kwargs)),
    )

    async def exercise():
        async def mainline():
            await asyncio.sleep(0.02)
            return "done"

        assert await coordinator.run(
            mainline(), initial_delay_sec=0, interval_sec=0.01,
            maximum_captures=1,
        ) == "done"
        await coordinator.release_local_snapshots()

    asyncio.run(exercise())
    assert plane.captures == 1
    assert plane.restores == 0
    assert released == []


def test_unrecorded_capture_is_preserved_for_local_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, plane, sink = _coordinator(tmp_path, "observe")
    sink.record_checkpoint_observation = lambda _payload: False
    released = []
    monkeypatch.setattr(
        shadow_runtime,
        "release_shadow_generation_v2",
        lambda *args, **kwargs: released.append((args, kwargs)),
    )

    async def exercise():
        async def mainline():
            await asyncio.sleep(0.02)
            return "done"

        assert await coordinator.run(
            mainline(), initial_delay_sec=0, interval_sec=0.01,
            maximum_captures=1,
        ) == "done"
        await coordinator.release_local_snapshots()

    asyncio.run(exercise())
    assert plane.captures == 1
    assert released == []


def test_server_can_downgrade_restore_test_to_observe_per_sample(
    tmp_path: Path,
) -> None:
    assignment = _assignment("restore-test")
    api = IdentityApi(assignment, current_mode="observe")
    coordinator, plane, sink = _coordinator(
        tmp_path, "restore-test", api=api,
    )

    async def mainline():
        await asyncio.sleep(0.02)
        return "done"

    assert asyncio.run(coordinator.run(
        mainline(), initial_delay_sec=0, interval_sec=0.01,
        maximum_captures=1,
    )) == "done"
    assert plane.captures == 1
    assert plane.restores == 0
    assert sink.payloads[0]["rollout_mode"] == "observe"


def test_rollout_is_rechecked_after_capture_before_offline_restore(
    tmp_path: Path,
) -> None:
    assignment = _assignment("restore-test")
    api = IdentityApi(assignment)
    original_activation = api.checkpoint_v2_activation
    restore_decision_seen = threading.Event()

    def changing_activation(payload):
        response = original_activation(payload)
        api.current_mode = "observe"
        if len(api.activation_calls) == 2:
            restore_decision_seen.set()
        return response

    api.checkpoint_v2_activation = changing_activation
    coordinator, plane, sink = _coordinator(
        tmp_path, "restore-test", api=api,
    )

    async def mainline():
        for _attempt in range(1000):
            if restore_decision_seen.is_set():
                return "done"
            await asyncio.sleep(0.001)
        raise AssertionError("restore activation decision was not observed")

    assert asyncio.run(coordinator.run(
        mainline(), initial_delay_sec=0, interval_sec=0.01,
        maximum_captures=1,
    )) == "done"
    assert plane.captures == 1
    assert plane.restores == 0
    assert len(api.activation_calls) == 2
    assert [item["observation_kind"] for item in sink.payloads] == ["capture"]


def test_kill_switch_stops_future_samples_without_touching_mainline(
    tmp_path: Path,
) -> None:
    assignment = _assignment("restore-test")
    api = IdentityApi(assignment, kill_switch=True)
    activation_seen = threading.Event()
    original_activation = api.checkpoint_v2_activation

    def observed_activation(payload):
        response = original_activation(payload)
        activation_seen.set()
        return response

    api.checkpoint_v2_activation = observed_activation
    coordinator, plane, sink = _coordinator(
        tmp_path, "restore-test", api=api,
    )

    async def mainline():
        for _attempt in range(1000):
            if activation_seen.is_set():
                return "authoritative-result"
            await asyncio.sleep(0.001)
        raise AssertionError("kill-switch activation decision was not observed")

    assert asyncio.run(coordinator.run(
        mainline(), initial_delay_sec=0, interval_sec=0.01,
        maximum_captures=10,
    )) == "authoritative-result"
    assert plane.captures == plane.restores == 0
    assert sink.payloads == []
    assert len(api.activation_calls) == 1


def test_identity_failure_never_replaces_mainline_result_or_calls_exporter(
    tmp_path: Path,
) -> None:
    assignment = _assignment("observe")
    api = IdentityApi(assignment, error=ApiError("identity server unavailable"))
    coordinator, plane, sink = _coordinator(tmp_path, "observe", api=api)

    async def mainline():
        await asyncio.sleep(0.01)
        return "still-valid"

    assert asyncio.run(coordinator.run(
        mainline(), initial_delay_sec=0, interval_sec=0.01,
        maximum_captures=1,
    )) == "still-valid"
    assert plane.captures == 0
    assert sink.payloads == []


def test_off_mode_never_constructs_coordinator_or_creates_checkpoint_state(
    tmp_path: Path,
) -> None:
    activation = negotiate_checkpoint_activation_v2(
        local_mode="off", server_mode="on",
    )
    constructed = []

    async def mainline():
        return "normal"

    def factory():
        constructed.append(True)
        raise AssertionError("OFF must not construct checkpoint machinery")

    result = asyncio.run(run_mainline_with_optional_checkpoint_shadow_v2(
        mainline(), activation=activation, coordinator_factory=factory,
        initial_delay_sec=0, interval_sec=0.01, maximum_captures=1,
    ))
    assert result == "normal"
    assert constructed == []
    assert list(tmp_path.iterdir()) == []


def test_authoritative_mode_cannot_fall_back_into_shadow_or_normal_mainline() -> None:
    activation = negotiate_checkpoint_activation_v2(
        local_mode="canary", server_mode="canary", controlled_account=True,
    )
    mainline_started = []

    async def mainline():
        mainline_started.append(True)
        return "must-not-run"

    async def scenario():
        await run_mainline_with_optional_checkpoint_shadow_v2(
            mainline(),
            activation=activation,
            coordinator_factory=lambda: (_ for _ in ()).throw(
                AssertionError("authoritative mode must be fenced first")
            ),
        )

    with pytest.raises(CheckpointV2ProtocolError, match="owner state machine"):
        asyncio.run(scenario())
    assert mainline_started == []
