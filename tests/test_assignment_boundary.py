import json
from types import SimpleNamespace

import pytest

from dradar import assignment_boundary
from dradar import runloop


def _assignment(assignment_id: str, task_id: str | None = None) -> dict:
    return {
        "assignment_id": assignment_id,
        "task_id": task_id or f"task-{assignment_id}",
        "model": "glm-5.3-flash",
        "effort": "low",
    }


def test_boundary_detects_unresolved_assignment_disappearing(tmp_path):
    active = [_assignment("a1"), _assignment("a2")]
    path = assignment_boundary.prepare(tmp_path, "bench", active)

    with pytest.raises(
        assignment_boundary.BoundaryError,
        match="disappeared.*a2",
    ):
        assignment_boundary.prepare(tmp_path, "bench", [active[0]])

    assert path is not None and path.is_file()


def test_submitted_assignment_may_leave_active_leases(tmp_path):
    active = [_assignment("a1"), _assignment("a2")]
    path = assignment_boundary.prepare(tmp_path, "bench", active)
    assignment_boundary.record_outcome(path, active[0], "submitted")

    report = assignment_boundary.reconcile(path, [active[1]])

    assert report is not None
    assert report.missing_ids == frozenset()
    assert report.settled_ids == frozenset({"a1"})
    assert not report.complete


def test_complete_boundary_is_removed(tmp_path):
    active = [_assignment("a1"), _assignment("a2")]
    path = assignment_boundary.prepare(tmp_path, "bench", active)
    assignment_boundary.record_outcome(path, active[0], "submitted")
    assignment_boundary.record_outcome(path, active[1], "interrupted")
    report = assignment_boundary.reconcile(path, [])

    assert report is not None and report.complete
    assignment_boundary.finish_if_complete(path, report)
    assert path is not None and not path.exists()


def test_explicit_boundary_requires_every_expected_id(tmp_path):
    with pytest.raises(
        assignment_boundary.BoundaryError,
        match="not active.*a2",
    ):
        assignment_boundary.prepare(
            tmp_path,
            "bench",
            [_assignment("a1")],
            expected_ids=["a1", "a2"],
        )


def test_explicit_boundary_rejects_unlisted_active_assignment(tmp_path):
    with pytest.raises(
        assignment_boundary.BoundaryError,
        match="outside the explicit boundary.*a2",
    ):
        assignment_boundary.prepare(
            tmp_path,
            "bench",
            [_assignment("a1"), _assignment("a2")],
            expected_ids=["a1"],
        )


def test_strict_boundary_cannot_be_extended_after_creation(tmp_path):
    path = assignment_boundary.prepare(
        tmp_path,
        "bench",
        [_assignment("a1")],
        expected_ids=["a1"],
    )

    with pytest.raises(
        assignment_boundary.BoundaryError,
        match="outside the explicit boundary.*a2",
    ):
        assignment_boundary.add_expected(path, [_assignment("a2")])


def test_forget_replaces_only_the_named_benchmark_boundary(tmp_path):
    old = [_assignment("old")]
    path = assignment_boundary.prepare(tmp_path, "bench", old)
    replacement = [_assignment("new")]

    new_path = assignment_boundary.prepare(
        tmp_path,
        "bench",
        replacement,
        expected_ids=["new"],
        forget_existing=True,
    )

    assert new_path == path
    payload = json.loads(path.read_text())
    assert set(payload["expected"]) == {"new"}
    assert "nonce" not in path.read_text()


def test_corrupt_boundary_fails_closed_without_explicit_forget(tmp_path):
    path = assignment_boundary.state_path(tmp_path, "bench")
    path.parent.mkdir(parents=True)
    path.write_text("not-json")

    with pytest.raises(assignment_boundary.BoundaryError, match="unreadable"):
        assignment_boundary.prepare(tmp_path, "bench", [_assignment("a1")])


def test_runloop_reconciliation_reports_a_lost_assignment(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runloop, "HOME", tmp_path)
    active = [_assignment("a1"), _assignment("a2")]
    args = SimpleNamespace(
        refill=False,
        expect_assignment=None,
        forget_assignment_boundary=False,
    )

    class Client:
        def get_assignment(self):
            return {"active": [active[0]]}

    path = runloop._prepare_assignment_boundary(
        args, Client(), "bench", active,
    )

    assert runloop._finish_assignment_boundary(Client(), path) is False
    assert "a2" in capsys.readouterr().out
    assert path is not None and path.is_file()


def test_worker_child_does_not_reconcile_parent_owned_boundary(
        tmp_path, monkeypatch):
    path = tmp_path / "shared-boundary.json"
    monkeypatch.setenv(runloop._ASSIGNMENT_BOUNDARY_ENV, str(path))
    reconciled = []
    monkeypatch.setattr(
        runloop, "_finish_assignment_boundary",
        lambda _client, _path: reconciled.append(_path) or False,
    )
    args = SimpleNamespace(worker_child=True)

    assert runloop._finish_invocation_assignment_boundary(
        args, object(), path,
    ) is True
    assert reconciled == []


def test_standalone_invocation_still_reconciles_its_boundary(
        tmp_path, monkeypatch):
    path = tmp_path / "owned-boundary.json"
    reconciled = []
    monkeypatch.delenv(runloop._ASSIGNMENT_BOUNDARY_ENV, raising=False)
    monkeypatch.setattr(
        runloop, "_finish_assignment_boundary",
        lambda _client, _path: reconciled.append(_path) or True,
    )
    args = SimpleNamespace(worker_child=True)

    assert runloop._finish_invocation_assignment_boundary(
        args, object(), path,
    ) is True
    assert reconciled == [path]
