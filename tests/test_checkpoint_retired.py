from argparse import Namespace

import pytest

from dradar import cli, pending, runloop


def test_checkpoint_commands_are_not_exposed():
    with pytest.raises(SystemExit) as exc:
        cli.main(["checkpoints"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["checkpoint", "discard", "cp-1"])
    assert exc.value.code == 2


def test_acquire_filters_pending_for_go_resume_auto_and_pool_start(tmp_path, monkeypatch):
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    pending.record(tmp_path, {"assignment_id": "paid", "trial_dir": "kept"})

    class Client:
        def get_assignment(self):
            return {"active": [
                {"assignment_id": "paid", "task_id": "done"},
                {"assignment_id": "fresh", "task_id": "new"},
            ], "free_pick": True}

    # _acquire_batch is shared by ordinary go/resume, auto-refill preparation,
    # and the multi-worker supervisor before it determines its process count.
    active, _ = runloop._acquire_batch(Client(), True)
    assert [item["assignment_id"] for item in active] == ["fresh"]


def test_checkout_loop_excludes_every_pending_assignment(tmp_path, monkeypatch):
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    pending.record(tmp_path, {"assignment_id": "paid", "trial_dir": "kept"})
    monkeypatch.setattr(runloop, "_check_version_pin", lambda *_a, **_k: None)
    monkeypatch.setattr(runloop, "_worker_slot_is_enabled", lambda: True)
    monkeypatch.setattr(runloop, "_pool_abort_reason", lambda: None)
    monkeypatch.setattr(runloop, "_pool_degraded_exclusions", lambda _client: set())
    seen = []

    class Client:
        def checkout(self, exclude_assignment_ids=None, session_id=None):
            seen.append(set(exclude_assignment_ids or ()))
            return {"assignment": None, "held": 1, "unstarted": 0}

    args = Namespace(
        allow_task_drift=False, refill=False, dev_agent=None, keep=True,
        yes=True, parallel=True, worker_child=True,
    )
    assert runloop._run_checkout_loop(
        args, Client(), tmp_path,
        [{"assignment_id": "paid", "deep_swe_commit": None}], telemetry=None,
    ) == 0
    assert seen == [{"paid"}]


def test_final_model_start_guard_refuses_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    pending.record(tmp_path, {"assignment_id": "paid", "trial_dir": "kept"})
    monkeypatch.setattr(runloop, "check_task_content_hash", lambda *_a: True)
    called = []
    monkeypatch.setattr(runloop, "run_trial", lambda *_a, **_k: called.append(True))
    args = Namespace(
        allow_task_drift=False, dev_agent=None, keep=True, yes=True,
        parallel=True, archive_session=False,
    )
    assignment = {"assignment_id": "paid", "task_id": "t", "nonce": "n"}
    assert runloop._run_and_submit(
        object(), assignment, tmp_path, args, None,
        _assignment_lock_held=True,
    ) == "pending-upload"
    assert called == []
