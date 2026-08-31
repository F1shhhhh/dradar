"""deep-swe version pin: client-side commit detection (compared against the
server's advertised grading commit — see the server repo's test suite for
the server-side half of this pin)."""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from dradar import runloop, runner
from dradar.runner import local_deep_swe_commit


def test_local_deep_swe_commit_non_repo(tmp_path):
    assert local_deep_swe_commit(tmp_path) is None


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)
def test_local_deep_swe_commit_real_repo(tmp_path):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(tmp_path),  # ignore user gitconfig (signing hooks etc.)
    }
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, env=env, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "f").write_text("x")
    git("add", "f")
    git("commit", "-q", "-m", "c1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # tasks_root is typically a subdir of the repo; both must resolve.
    sub = tmp_path / "tasks"
    sub.mkdir()
    assert local_deep_swe_commit(tmp_path) == head
    assert local_deep_swe_commit(sub) == head


def test_sync_fetches_pin_from_dradar_public_task_repo(monkeypatch, tmp_path):
    pinned = "d" * 40
    calls = []

    class Completed:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "local_deep_swe_commit", lambda _root: pinned)

    assert runner.sync_deep_swe_commit(tmp_path / "tasks", pinned) is True
    assert calls == [
        [
            "git", "-C", str(tmp_path / "tasks"), "fetch", "--depth", "1",
            runner.DEEP_SWE_REPO, pinned,
        ],
        ["git", "-C", str(tmp_path / "tasks"), "checkout", pinned],
    ]
    assert runner.DEEP_SWE_REPO == "https://github.com/SecurityMind/deep-swe"


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)
def test_pinned_snapshot_preserves_dirty_user_checkout_and_is_reused(
    tmp_path, monkeypatch,
):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    source = tmp_path / "source"
    source.mkdir()

    def git(cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=env, check=True,
            capture_output=True, text=True,
        )

    git(source, "init", "-q")
    tasks = source / "tasks"
    tasks.mkdir()
    task_file = tasks / "task.toml"
    task_file.write_text("version = 1\n")
    git(source, "add", ".")
    git(source, "commit", "-q", "-m", "v1")
    first = git(source, "rev-parse", "HEAD").stdout.strip()
    task_file.write_text("version = 2\n")
    git(source, "commit", "-qam", "v2")
    second = git(source, "rev-parse", "HEAD").stdout.strip()

    configured = tmp_path / "configured"
    git(tmp_path, "clone", "-q", str(source), str(configured))
    git(configured, "checkout", "-q", first)
    configured_task = configured / "tasks" / "task.toml"
    configured_task.write_text("my private edit\n")
    monkeypatch.setattr(runner, "DEEP_SWE_REPO", str(source))

    managed = runner.prepare_pinned_deep_swe_tasks(
        tmp_path / "dradar-home", second,
    )
    reused = runner.prepare_pinned_deep_swe_tasks(
        tmp_path / "dradar-home", second,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(
            lambda commit: runner.prepare_pinned_deep_swe_tasks(
                tmp_path / "dradar-home", commit,
            ),
            (first, second),
        ))

    assert reused == managed
    assert runner.local_deep_swe_commit(parallel[0]) == first
    assert runner.local_deep_swe_commit(parallel[1]) == second
    assert (parallel[0] / "task.toml").read_text() == "version = 1\n"
    assert runner.local_deep_swe_commit(managed) == second
    assert (managed / "task.toml").read_text() == "version = 2\n"
    assert runner.local_deep_swe_commit(configured / "tasks") == first
    assert configured_task.read_text() == "my private edit\n"
    assert git(configured, "status", "--porcelain").stdout.strip()


# --- _check_version_pin: drift handling (self-heal / hard-stop / --allow-task-drift)

LOCAL = "a" * 40
PINNED = "b" * 40


def _pin(monkeypatch, tmp_path, *, sync_ok, allow_drift, pinned=PINNED):
    monkeypatch.setattr(runloop, "local_deep_swe_commit", lambda root: LOCAL)
    synced = []

    def fake_snapshot(_home, commit):
        synced.append(commit)
        if not sync_ok:
            raise runner.RunnerError("snapshot unavailable")
        return tmp_path / "managed" / commit / "tasks"

    monkeypatch.setattr(runloop, "prepare_pinned_deep_swe_tasks", fake_snapshot)
    return runloop._check_version_pin(
        pinned, tmp_path, allow_drift, return_tasks_root=True,
    ), synced


def test_version_pin_match_is_silent(monkeypatch, tmp_path, capsys):
    got, synced = _pin(monkeypatch, tmp_path, sync_ok=False, allow_drift=False,
                       pinned=LOCAL)
    assert got == (tmp_path, LOCAL)
    assert synced == []  # no drift -> no sync attempt
    assert capsys.readouterr().out == ""


def test_version_pin_drift_self_heals_via_sync(monkeypatch, tmp_path, capsys):
    got, synced = _pin(monkeypatch, tmp_path, sync_ok=True, allow_drift=False)
    assert got == (tmp_path / "managed" / PINNED / "tasks", PINNED)
    assert synced == [PINNED]
    assert "isolated task snapshot ready" in capsys.readouterr().out


def test_version_pin_drift_sync_failure_hard_stops_with_fix(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as ei:
        _pin(monkeypatch, tmp_path, sync_ok=False, allow_drift=False)
    msg = str(ei.value)
    assert "configured task files were left unchanged" in msg
    assert "network" in msg and "disk" in msg


def test_version_pin_drift_allowed_warns_and_proceeds(monkeypatch, tmp_path, capsys):
    got, synced = _pin(monkeypatch, tmp_path, sync_ok=False, allow_drift=True)
    assert got == (tmp_path, LOCAL)
    assert "warning" in capsys.readouterr().out
