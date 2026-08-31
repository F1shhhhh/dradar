"""config.json safety: atomic saves (a kill mid-write must never truncate
the volunteer's only token) and an actionable message — not a traceback —
when the file on disk is corrupt."""

import json
import os
import stat

import pytest

from dradar import local_config


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setattr(local_config, "HOME", tmp_path)
    monkeypatch.setattr(local_config, "CONFIG_PATH", tmp_path / "config.json")
    return tmp_path


def test_valid_config_still_loads(home):
    # negative control for the corrupt-config path
    local_config._save_config({"server": "https://api.example.com", "token": "drt_x"})
    assert local_config._load_config() == {
        "server": "https://api.example.com", "token": "drt_x"}


def test_missing_config_loads_empty(home):
    assert local_config._load_config() == {}


def test_corrupt_config_exits_with_recovery_guidance(home):
    (home / "config.json").write_text("{not json")
    with pytest.raises(SystemExit) as ei:
        local_config._load_config()
    msg = str(ei.value)
    assert "corrupt" in msg
    assert str(home / "config.json") in msg
    assert "login --github" in msg


def test_save_failure_leaves_prior_config_intact(home, monkeypatch):
    # atomicity invariant: os.replace is the commit point — a failure
    # anywhere before it must leave the previous config loadable.
    local_config._save_config({"token": "drt_old"})
    monkeypatch.setattr(local_config.os, "replace",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        local_config._save_config({"token": "drt_new"})
    assert local_config._load_config() == {"token": "drt_old"}


def test_saved_config_is_owner_only(home):
    local_config._save_config({"token": "drt_secret"})
    mode = stat.S_IMODE(os.stat(home / "config.json").st_mode)
    assert mode == 0o600


def test_fresh_on_corrupt_returns_empty_and_warns(tmp_path, monkeypatch, capsys):
    """`dradar login` must be able to run over a corrupt config — it IS the
    recovery command the corrupt-config error recommends."""
    monkeypatch.setattr(local_config, "HOME", tmp_path)
    monkeypatch.setattr(local_config, "CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / "config.json").write_text('{"serv')  # truncated by a crash
    cfg = local_config._load_config(fresh_on_corrupt=True)
    assert cfg == {}
    assert "starting fresh" in capsys.readouterr().out


def test_benchmark_task_roots_are_isolated_from_legacy_deep_swe(home):
    cfg = {
        "tasks_root": "/legacy/deep/tasks",
        "tasks_roots": {"pompeii-adjacency": "/visual/pompeii/tasks"},
    }
    assert local_config.tasks_root_from_config(cfg) == local_config.Path(
        "/legacy/deep/tasks")
    assert local_config.tasks_root_from_config(
        cfg, "pompeii-adjacency") == local_config.Path("/visual/pompeii/tasks")
    assert local_config.default_tasks_root(
        "pompeii-adjacency") == home / "benchmarks" / "pompeii-adjacency" / "tasks"


def test_private_run_plan_credentials_overlay_without_replacing_user_config(home):
    local_config._save_config({
        "server": "https://ordinary.example",
        "token": "drt_ordinary",
        "tasks_root": "/kept/tasks",
    })
    credentials = home / "run-plans" / "plan-example.json"
    credentials.parent.mkdir(mode=0o700)
    credentials.write_text(json.dumps({
        "credential_kind": "run_plan_v1",
        "server": "https://api.codexradar.com",
        "token": "drp_scoped",
        "benchmark": "deep-swe",
        "batch_id": "12345678123456781234567812345678",
        "plan_id": "plan-example",
        "logical_session_id": "drl_logical_session_example",
        "plan": {"points_tier": "pro-20x"},
    }))
    credentials.chmod(0o600)

    runtime = local_config.runtime_config(credentials)

    assert runtime["server"] == "https://api.codexradar.com"
    assert runtime["token"] == "drp_scoped"
    assert runtime["run_plan_batch_id"] == "12345678123456781234567812345678"
    assert runtime["run_plan_logical_session_id"] == "drl_logical_session_example"
    assert runtime["tasks_root"] == "/kept/tasks"
    assert local_config._load_config()["token"] == "drt_ordinary"


def test_private_run_plan_credentials_reject_permissive_mode(home):
    credentials = home / "plan.json"
    credentials.write_text(json.dumps({
        "credential_kind": "run_plan_v1",
        "server": "https://api.codexradar.com",
        "token": "drp_scoped",
        "benchmark": "deep-swe",
        "batch_id": "12345678123456781234567812345678",
    }))
    if os.name == "nt":
        pytest.skip("POSIX permission contract")
    credentials.chmod(0o644)

    with pytest.raises(ValueError, match="invalid private run-plan"):
        local_config.runtime_config(credentials)


def test_private_run_plan_does_not_require_a_valid_long_term_login_config(home):
    (home / "config.json").write_text("{broken")
    credentials = home / "plan.json"
    credentials.write_text(json.dumps({
        "credential_kind": "run_plan_v1",
        "server": "https://api.claudecoderadar.com",
        "token": "drp_scoped",
        "benchmark": "deep-swe",
        "batch_id": "12345678123456781234567812345678",
        "plan_id": "plan-example",
        "logical_session_id": "drl_logical_session_example",
        "plan": {"points_tier": "pro-5x"},
    }))
    credentials.chmod(0o600)

    runtime = local_config.runtime_config(credentials)

    assert runtime["server"] == "https://api.claudecoderadar.com"
    assert runtime["token"] == "drp_scoped"
