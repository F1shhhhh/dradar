from argparse import Namespace
import json

import pytest

from dradar import leases


def _cell(aid, *, started=False):
    return {
        "assignment_id": aid,
        "task_id": f"task-{aid}",
        "model": "gpt-5.6-sol",
        "effort": "low",
        "expires_at": "2099-01-01T00:00:00+00:00",
        "started_at": "2098-12-31T23:00:00+00:00" if started else None,
    }


class FakeClient:
    def __init__(self, active, *, recent_inactive=None, benchmark_id="deep-swe"):
        self.active = active
        self.recent_inactive = recent_inactive or []
        self.benchmark_id = benchmark_id
        self.release_calls = []

    def benchmarks(self):
        return {"benchmarks": [{"id": self.benchmark_id}]}

    def get_assignment(self):
        return {
            "active": self.active,
            "recent_inactive": self.recent_inactive,
            "free_pick": True,
        }

    def release_assignments(self, assignment_ids=None, *, release_all=False, force=False):
        self.release_calls.append((assignment_ids, release_all, force))
        targets = self.active if release_all else [
            x for x in self.active if x["assignment_id"] in (assignment_ids or [])]
        released, skipped = [], []
        for item in targets:
            basic = {key: item[key] for key in
                     ("assignment_id", "task_id", "model", "effort")}
            if item.get("started_at") and not force:
                skipped.append({**basic, "reason": "running"})
            else:
                released.append({**basic, "was_running": bool(item.get("started_at"))})
        return {"released": released, "skipped": skipped,
                "already_released": [], "held": len(skipped)}


def _wire(monkeypatch, client):
    monkeypatch.setattr(leases, "_load_config", lambda: {})
    monkeypatch.setattr(leases, "_client", lambda cfg: client)


def test_leases_lists_waiting_and_running_with_recovery_hint(monkeypatch, capsys):
    client = FakeClient([_cell("a1"), _cell("a2", started=True)])
    _wire(monkeypatch, client)

    assert leases.cmd_leases(Namespace()) == 0

    out = capsys.readouterr().out
    assert "1 running, 1 waiting" in out
    assert "a1" in out and "a2" in out
    assert "dradar release --all" in out
    assert "--force" in out


def test_leases_json_preserves_exact_batch_evidence(monkeypatch, capsys):
    waiting = _cell("a1")
    waiting["batch_id"] = "550e8400e29b41d4a716446655440000"
    running = _cell("a2", started=True)
    running.update({
        "batch_id": "6ba7b8109dad11d180b400c04fd430c8",
        "heartbeat_running": True,
        "execution_state": "running",
    })
    client = FakeClient([waiting, running])
    _wire(monkeypatch, client)

    assert leases.cmd_leases(Namespace(json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "ok"
    assert payload["summary"] == {
        "paused": 0,
        "running": 1,
        "stale": 0,
        "total": 2,
        "waiting": 1,
    }
    assert [item["batch_id"] for item in payload["active"]] == [
        "550e8400e29b41d4a716446655440000",
        "6ba7b8109dad11d180b400c04fd430c8",
    ]


def test_leases_keeps_recent_expired_unsubmitted_assignment_visible(
    monkeypatch, capsys,
):
    expired = {
        "assignment_id": "expired-a1",
        "task_id": "pompeii-adjacency-rp-089",
        "model": "glm-5.3-flash",
        "effort": "high",
        "status": "expired",
        "reason": "lease_deadline_elapsed",
        "inactive_at": "2026-08-27T12:13:03+00:00",
    }
    client = FakeClient([], recent_inactive=[expired],
                        benchmark_id="pompeii-adjacency")
    _wire(monkeypatch, client)

    assert leases.cmd_leases(Namespace()) == 0

    out = capsys.readouterr().out
    assert "no active leases" in out
    assert "recent unsubmitted leases" in out
    assert "expired-a1" in out
    assert "pompeii-adjacency-rp-089" in out
    assert "lease_deadline_elapsed" in out


def test_started_history_without_healthy_runner_is_resumable_not_running(
    monkeypatch, capsys,
):
    stale = _cell("a1", started=True)
    stale.update({"heartbeat_running": False, "execution_state": "running"})
    live = _cell("a2", started=True)
    live.update({"heartbeat_running": True, "execution_state": "running"})
    paused = _cell("a3", started=True)
    paused.update({"heartbeat_running": False, "execution_state": "paused"})
    client = FakeClient([stale, live, paused])
    _wire(monkeypatch, client)

    assert leases.cmd_leases(Namespace()) == 0

    out = capsys.readouterr().out
    assert "1 running, 1 paused, 1 stale" in out
    assert "a1" in out and "stale" in out
    assert "not automatically resumable" in out


@pytest.mark.parametrize("server_state", ("resumable", "checkpoint_retired"))
def test_legacy_checkpoint_tombstone_is_stale_not_resumable(
    monkeypatch, capsys, server_state,
):
    legacy = _cell("legacy-checkpoint", started=True)
    legacy.update({
        "heartbeat_running": False,
        "execution_state": "running",
        "runner_state": server_state,
        "checkpoint_id": "retired-checkpoint-tombstone",
    })
    client = FakeClient([legacy])
    _wire(monkeypatch, client)

    assert leases.cmd_leases(Namespace()) == 0

    out = capsys.readouterr().out
    assert "1 stale" in out
    assert "resumable" not in out.splitlines()[0]


def test_leases_lists_and_labels_assignments_across_benchmarks(
    monkeypatch, capsys,
):
    class MultiBenchmarkClient(FakeClient):
        def __init__(self):
            super().__init__([], benchmark_id="deep-swe")
            self.by_benchmark = {
                "deep-swe": [_cell("deep")],
                "pompeii-adjacency": [{
                    **_cell("pompeii"),
                    "task_id": "pompeii-adjacency-rp-044",
                    "model": "glm-5.3-flash",
                    "effort": "high",
                }],
            }

        def benchmarks(self):
            return {"benchmarks": [
                {"id": "deep-swe"}, {"id": "pompeii-adjacency"},
            ]}

        def get_assignment(self):
            return {"active": self.by_benchmark[self.benchmark_id]}

    client = MultiBenchmarkClient()
    _wire(monkeypatch, client)

    assert leases.cmd_leases(Namespace()) == 0

    out = capsys.readouterr().out
    assert "holding 2 cell(s)" in out
    assert "task-deep" in out and "[deep-swe]" in out
    assert "pompeii-adjacency-rp-044" in out
    assert "glm-5.3-flash@high" in out
    assert "[pompeii-adjacency]" in out
    assert client.benchmark_id == "deep-swe"


def test_cross_benchmark_inventory_deduplicates_assignment_ids(monkeypatch):
    class DuplicateClient(FakeClient):
        def benchmarks(self):
            return {"benchmarks": [{"id": "deep-swe"}, {"id": "mirror"}]}

        def get_assignment(self):
            return {"active": [_cell("same")]}

    client = DuplicateClient([_cell("same")])
    _wire(monkeypatch, client)

    assert len(leases._all_active(client)) == 1
    assert client.benchmark_id == "deep-swe"


def test_cross_benchmark_inventory_skips_inaccessible_channel(monkeypatch):
    class RestrictedClient(FakeClient):
        def benchmarks(self):
            return {"benchmarks": [{"id": "deep-swe"}, {"id": "private"}]}

        def get_assignment(self):
            if self.benchmark_id == "private":
                raise leases.ApiError("forbidden", status_code=403)
            return {"active": [_cell("deep")]}

    client = RestrictedClient([_cell("deep")])
    _wire(monkeypatch, client)

    assert [item["assignment_id"] for item in leases._all_active(client)] == ["deep"]
    assert client.benchmark_id == "deep-swe"


def test_cross_benchmark_inventory_falls_back_for_legacy_server(monkeypatch):
    class LegacyClient(FakeClient):
        def benchmarks(self):
            raise leases.ApiError("not found", status_code=404)

    client = LegacyClient([_cell("legacy")])
    _wire(monkeypatch, client)

    assert [item["assignment_id"] for item in leases._all_active(client)] == ["legacy"]
    assert client.benchmark_id == "deep-swe"


def test_release_all_protects_running_without_force(monkeypatch, capsys):
    client = FakeClient([_cell("a1"), _cell("a2", started=True)])
    _wire(monkeypatch, client)
    args = Namespace(assignment_ids=[], all=True, force=False, yes=True)

    assert leases.cmd_release(args) == 0

    assert client.release_calls == [(None, True, False)]
    out = capsys.readouterr().out
    assert "released 1" in out and "kept 1" in out
    assert "--force" in out


def test_release_interactive_selection(monkeypatch):
    client = FakeClient([_cell("a1"), _cell("a2")])
    _wire(monkeypatch, client)
    answers = iter(["2", "y"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    args = Namespace(assignment_ids=[], all=False, force=False, yes=False)

    assert leases.cmd_release(args) == 0
    assert client.release_calls == [(["a2"], False, False)]


def test_release_rejects_ids_plus_all(monkeypatch):
    client = FakeClient([])
    _wire(monkeypatch, client)
    args = Namespace(assignment_ids=["a1"], all=True, force=False, yes=True)
    with pytest.raises(SystemExit, match="not both"):
        leases.cmd_release(args)
