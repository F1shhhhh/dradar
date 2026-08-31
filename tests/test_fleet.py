import argparse
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from dradar import cli, fleet, runloop
from dradar.capacity import CapacityReport


BATCH_A = "550e8400e29b41d4a716446655440000"
BATCH_B = "6ba7b8109dad11d180b400c04fd430c8"


@pytest.fixture(autouse=True)
def _release_process_locks():
    yield
    fleet.release_pool_locks_for_tests()


def test_cli_parses_agent_facing_fleet_add(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_fleet_add", lambda args: seen.append(args) or 0)

    assert cli.main([
        "fleet", "add",
        "--batch-id", "550E8400-E29B-41D4-A716-446655440000",
        "--workers", "2",
    ]) == 0

    assert seen[0].batch_id == BATCH_A
    assert seen[0].workers == 2


def test_fleet_help_teaches_public_commands_without_internal_serve(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["fleet", "--help"])

    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "{add,status,watch,stop}" in output
    assert "idempotently add one exact claimed batch" in output
    assert "serve" not in output
    assert "SUPPRESS" not in output


def test_cli_parses_exact_post_seed_refill_campaign(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_fleet_add", lambda args: seen.append(args) or 0)

    assert cli.main([
        "fleet", "add", "--batch-id", BATCH_A, "--workers", "2",
        "--refill", "--max-tasks", "10",
        "--refill-harness", "kimi-code",
        "--refill-model", "kimi-k2.5",
        "--refill-effort", "high",
    ]) == 0

    assert seen[0].refill is True
    assert seen[0].max_tasks == 10
    assert seen[0].refill_harness == "kimi-code"
    assert seen[0].refill_model == "kimi-k2.5"
    assert seen[0].refill_effort == "high"


def test_fleet_add_is_idempotent_for_one_batch(tmp_path, monkeypatch):
    fleet._prepare_dirs(tmp_path)
    state = fleet._initial_state("controller-1", None)
    state["status"] = "active"
    processes = {}
    logs = {}
    spawned = []

    class Process:
        pid = 1234

        def poll(self):
            return None

        def send_signal(self, _signal):
            pass

    monkeypatch.setattr(
        fleet, "_resolve_workers",
        lambda *_args: (2, [], {"account_limit": 5, "held_tasks": 2}),
    )

    def spawn(*_args, **_kwargs):
        spawned.append(True)
        return Process(), io.StringIO()

    monkeypatch.setattr(fleet, "_spawn_pool", spawn)
    first = {
        "request_id": "request-1", "controller_id": "controller-1",
        "command": "add", "batch_id": BATCH_A, "workers": 2,
    }
    second = dict(first, request_id="request-2")

    fleet._handle_request(tmp_path, state, processes, logs, first)
    fleet._handle_request(tmp_path, state, processes, logs, second)

    assert spawned == [True]
    assert set(processes) == {BATCH_A}
    response = json.loads(
        (fleet._root(tmp_path) / fleet.RESPONSE_DIR / "request-2.json").read_text()
    )
    assert response["ok"] is True
    assert response["already_active"] is True
    assert response["batch"]["workers"] == 2


def test_fleet_tracks_separate_honeypot_batches_and_total_workers(
    tmp_path, monkeypatch,
):
    fleet._prepare_dirs(tmp_path)
    state = fleet._initial_state("controller-1", None)
    state["status"] = "active"
    processes = {}
    logs = {}

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

        def send_signal(self, _signal):
            pass

    worker_targets = iter((2, 3))
    monkeypatch.setattr(
        fleet, "_resolve_workers",
        lambda *_args: (next(worker_targets), [], {"account_limit": 5}),
    )
    pids = iter((111, 222))
    monkeypatch.setattr(
        fleet, "_spawn_pool",
        lambda *_args, **_kwargs: (Process(next(pids)), io.StringIO()),
    )

    for request_id, batch_id in (("a", BATCH_A), ("b", BATCH_B)):
        fleet._handle_request(tmp_path, state, processes, logs, {
            "request_id": request_id,
            "controller_id": "controller-1",
            "command": "add",
            "batch_id": batch_id,
            "workers": "auto",
        })

    persisted = json.loads(fleet._state_path(tmp_path).read_text())
    assert persisted["total_workers"] == 5
    assert set(persisted["batches"]) == {BATCH_A, BATCH_B}
    assert persisted["batches"][BATCH_A]["workers"] == 2
    assert persisted["batches"][BATCH_B]["workers"] == 3


def test_controller_liveness_requires_process_lifetime_lease_not_reused_pid(
    tmp_path, monkeypatch,
):
    state = fleet._initial_state("controller-reused-pid", None)
    state["status"] = "active"
    fleet._write_state(tmp_path, state)
    monkeypatch.setattr(fleet, "_pid_alive", lambda _pid: True)

    # A live-looking/reused PID and fresh heartbeat are insufficient without
    # the exact controller lease held by the controller process.
    assert fleet.controller_is_active(tmp_path) is False

    with fleet._controller_lease(tmp_path, "controller-reused-pid"):
        assert fleet.controller_is_active(tmp_path) is True


def test_dead_controller_without_pool_lock_exposes_interrupted_zero_reservation(
    tmp_path,
):
    state = fleet._initial_state("dead-controller", None)
    state["status"] = "active"
    state["batches"][BATCH_A] = {
        "batch_id": BATCH_A,
        "status": "running",
        "workers": 3,
        "plan_id": "plan-dead",
    }
    fleet._write_state(tmp_path, state)

    public = fleet._public_state(tmp_path)

    assert public["active"] is False
    assert public["batches"][BATCH_A]["status"] == "interrupted"
    assert public["total_workers"] == 0
    assert fleet.batch_status(BATCH_A, home=tmp_path)["status"] == "interrupted"
    assert fleet.reserved_workers(tmp_path) == 0


def test_dead_controller_with_live_pool_lock_keeps_orphan_reservation_and_credential(
    tmp_path,
):
    credentials = tmp_path / "run-plans" / "plan-orphan.json"
    credentials.parent.mkdir(parents=True)
    credentials.write_text("{}")
    state = fleet._initial_state("dead-controller", None)
    state["status"] = "active"
    state["batches"][BATCH_A] = {
        "batch_id": BATCH_A,
        "status": "running",
        "workers": 2,
        "plan_id": "plan-orphan",
        "credentials_file": str(credentials),
    }
    fleet._write_state(tmp_path, state)
    fleet.acquire_pool_lock(tmp_path, BATCH_A, "dead-controller")

    public = fleet._public_state(tmp_path)

    assert public["active"] is False
    assert public["batches"][BATCH_A]["status"] == "orphaned"
    assert public["total_workers"] == 2
    assert fleet.reserved_workers(tmp_path) == 2
    assert fleet.credentials_file_in_use(credentials, home=tmp_path) is True


def test_auto_workers_subtract_existing_machine_reservations(monkeypatch):
    class Client:
        def set_batch_id(self, value):
            self.batch_id = value

    report = CapacityReport(
        recommended_workers=3,
        docker_cpus=12,
        docker_memory_gib=32,
        disk_free_gib=100,
        account_limit=5,
        held_tasks=3,
        task_limit=3,
        cpu_limit=6,
        memory_limit=5,
        disk_limit=7,
    )
    monkeypatch.setattr(fleet, "_load_config", lambda: {})
    monkeypatch.setattr(fleet, "_client", lambda _cfg: Client())
    monkeypatch.setattr(fleet, "inspect_capacity", lambda _client: report)
    monkeypatch.setattr(fleet, "docker_resources", lambda: (12, 32, ()))
    state = {
        "batches": {
            BATCH_A: {"status": "running", "workers": 2},
        }
    }

    workers, warnings, metadata = fleet._resolve_workers("auto", BATCH_B, state)

    assert workers == 2  # global automatic cap 4 minus the existing two
    assert warnings == []
    assert metadata["reserved_before"] == 2


def test_pool_command_is_exact_batch_resume_without_claim_or_refill(
    tmp_path, monkeypatch,
):
    fleet._prepare_dirs(tmp_path)
    captured = {}

    class Process:
        pid = 123

    def popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(fleet.subprocess, "Popen", popen)
    monkeypatch.setattr("dradar.machine._lock_handle", None)
    state = {"controller_id": "controller-1"}

    _process, log = fleet._spawn_pool(tmp_path, state, BATCH_A, 2)
    log.close()

    assert captured["command"][3:5] == ["resume", "-y"]
    assert captured["command"][captured["command"].index("--batch-id") + 1] == BATCH_A
    assert captured["command"][captured["command"].index("--workers") + 1] == "2"
    assert "--refill" not in captured["command"]
    assert "--auto" not in captured["command"]
    assert captured["env"][fleet.POOL_BATCH_ENV] == BATCH_A


def test_refill_pool_command_keeps_exact_batch_and_total_cap(tmp_path, monkeypatch):
    fleet._prepare_dirs(tmp_path)
    captured = {}

    class Process:
        pid = 124

    def popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(fleet.subprocess, "Popen", popen)
    monkeypatch.setattr("dradar.machine._lock_handle", None)
    state = {"controller_id": "controller-1"}

    _process, log = fleet._spawn_pool(
        tmp_path, state, BATCH_A, 2,
        refill=True,
        max_tasks=10,
        refill_harness="kimi-code",
        refill_model="kimi-k2.5",
        refill_effort="high",
    )
    log.close()

    command = captured["command"]
    assert command[command.index("--batch-id") + 1] == BATCH_A
    assert command[command.index("--refill-to") + 1] == "2"
    assert command[command.index("--max-tasks") + 1] == "10"
    assert command[command.index("--refill-harness") + 1] == "kimi-code"
    assert command[command.index("--refill-model") + 1] == "kimi-k2.5"
    assert command[command.index("--refill-effort") + 1] == "high"
    assert captured["env"]["DRADAR_REFILL_PLAN_SCOPE"] == BATCH_A


def test_plan_token_stays_in_private_file_not_fleet_argv_env_or_state(
    tmp_path, monkeypatch,
):
    fleet._prepare_dirs(tmp_path)
    credentials = tmp_path / "run-plans" / "plan-example.json"
    credentials.parent.mkdir(mode=0o700)
    token = "drp_extremely_private_plan_token"
    credentials.write_text(json.dumps({"token": token}))
    credentials.chmod(0o600)
    captured = {}

    class Process:
        pid = 321

        def poll(self):
            return None

    def popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(fleet.subprocess, "Popen", popen)
    monkeypatch.setattr("dradar.machine._lock_handle", None)
    process, log = fleet._spawn_pool(
        tmp_path,
        {"controller_id": "controller-1"},
        BATCH_A,
        2,
        credentials_file=str(credentials),
    )
    log.close()
    assert process.pid == 321
    assert captured["command"][captured["command"].index("--credentials-file") + 1] == str(credentials)
    assert token not in captured["command"]
    assert all(token not in str(value) for value in captured["env"].values())

    state = fleet._initial_state("controller-1", None)
    state["status"] = "active"
    monkeypatch.setattr(
        fleet, "_resolve_workers",
        lambda *_args: (2, [], {"account_limit": 4}),
    )
    monkeypatch.setattr(
        fleet,
        "_spawn_pool",
        lambda *_args, **_kwargs: (Process(), io.StringIO()),
    )
    fleet._handle_request(
        tmp_path,
        state,
        {},
        {},
        {
            "request_id": "plan-request",
            "controller_id": "controller-1",
            "command": "add",
            "batch_id": BATCH_A,
            "workers": 2,
            "credentials_file": str(credentials),
            "plan_id": "plan-example",
        },
    )
    persisted = fleet._state_path(tmp_path).read_text()
    assert token not in persisted
    assert str(credentials) in persisted


def test_pool_lock_rejects_duplicate_parent_and_dies_with_process(tmp_path):
    source = Path(__file__).parent.parent / "src"
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(source)!r})
            from pathlib import Path
            from dradar.fleet import acquire_pool_lock
            acquire_pool_lock(Path({str(tmp_path)!r}), {BATCH_A!r}, "controller-a")
            print("locked", flush=True)
            time.sleep(30)
        """)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(fleet.FleetError, match="already has a local Fleet pool owner"):
            fleet.acquire_pool_lock(tmp_path, BATCH_A, "controller-b")
    finally:
        holder.kill()
        holder.wait()

    fleet.acquire_pool_lock(tmp_path, BATCH_A, "controller-b")


def test_internal_fleet_pool_requires_controller_contract(monkeypatch):
    args = argparse.Namespace(
        workers=2, yes=True, keep=False, allow_task_drift=False,
        dev_agent=None, refill=False, refill_to=None, max_tasks=None,
        max_estimated_quota_pct=None, quota_tier="plus", auto=None, pick=None,
        parallel=False, worker_child=False, resume=True,
        worker_target_file=None, archive_session=False, batch_id=BATCH_A,
        fleet_pool=True, expect_assignment=None,
        forget_assignment_boundary=False,
    )
    monkeypatch.delenv(fleet.CONTROLLER_ID_ENV, raising=False)
    monkeypatch.delenv(fleet.POOL_BATCH_ENV, raising=False)

    with pytest.raises(SystemExit, match="invalid internal Fleet pool"):
        runloop.cmd_go(args)


def test_internal_fleet_pool_rejects_stale_controller_identity(monkeypatch):
    args = argparse.Namespace(
        workers=2, yes=True, keep=False, allow_task_drift=False,
        dev_agent=None, refill=False, refill_to=None, max_tasks=None,
        max_estimated_quota_pct=None, quota_tier="plus", auto=None, pick=None,
        parallel=False, worker_child=False, resume=True,
        worker_target_file=None, archive_session=False, batch_id=BATCH_A,
        fleet_pool=True, expect_assignment=None,
        forget_assignment_boundary=False,
    )
    monkeypatch.setenv(fleet.CONTROLLER_ID_ENV, "stale-controller")
    monkeypatch.setenv(fleet.POOL_BATCH_ENV, BATCH_A)
    monkeypatch.setattr(fleet, "controller_matches", lambda *_args: False)

    with pytest.raises(SystemExit, match="invalid internal Fleet pool"):
        runloop.cmd_go(args)


def test_explicit_stop_is_reported_as_stopped_and_halts_refill(
    tmp_path, monkeypatch,
):
    fleet._prepare_dirs(tmp_path)
    state = fleet._initial_state("controller-1", None)
    state["status"] = "active"
    state["batches"][BATCH_A] = {
        "batch_id": BATCH_A,
        "workers": 2,
        "status": "stopping",
        "refill": True,
    }

    class Process:
        def poll(self):
            return 130

    process = Process()
    log = io.StringIO()
    processes = {BATCH_A: process}
    logs = {BATCH_A: log}
    stopped = []
    monkeypatch.setattr(
        fleet, "_stop_remote_refill",
        lambda batch_id, reason: stopped.append((batch_id, reason)),
    )

    fleet._settle_pool(
        tmp_path, state, processes, logs, BATCH_A, 130,
    )

    item = state["batches"][BATCH_A]
    assert item["status"] == "stopped"
    assert item["returncode"] == 130
    assert processes == {}
    assert logs == {}
    assert log.closed
    assert stopped == [(BATCH_A, "stopped by the machine-local Fleet")]


def test_failed_refill_pool_stops_server_campaign(tmp_path, monkeypatch):
    fleet._prepare_dirs(tmp_path)
    state = fleet._initial_state("controller-1", None)
    state["status"] = "active"
    state["batches"][BATCH_A] = {
        "batch_id": BATCH_A,
        "workers": 1,
        "status": "running",
        "refill": True,
    }
    stopped = []
    monkeypatch.setattr(
        fleet, "_stop_remote_refill",
        lambda batch_id, reason: stopped.append((batch_id, reason)),
    )

    fleet._settle_pool(
        tmp_path, state, {BATCH_A: object()}, {}, BATCH_A, 7,
    )

    assert state["batches"][BATCH_A]["status"] == "failed"
    assert stopped == [(BATCH_A, "local Fleet pool exited with code 7")]
