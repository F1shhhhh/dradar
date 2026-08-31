import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from dradar import cli, doctor, fleet, run_plans
from dradar.api_client import ApiClient, ApiError


BATCH_ID = "12345678123456781234567812345678"
RUN_CODE = "run_very_high_entropy_example_123456789"
PLAN_TOKEN = "drp_plan_scoped_secret"


def _envelope(
    status="started",
    *,
    interaction="notify",
    decision_required=False,
    user_message="已在这台设备开始运行。无需操作。",
    agent_action="monitor",
    error_code=None,
    retryable=False,
    choices=None,
    **extra,
):
    return {
        "status": status,
        "interaction": interaction,
        "decision_required": decision_required,
        "user_message": user_message,
        "agent_action": agent_action,
        "error_code": error_code,
        "retryable": retryable,
        "choices": list(choices or []),
        **extra,
    }


def _plan(
    *,
    mode="auto",
    concurrency=None,
    task_count=2,
    refill=False,
    refill_to=None,
    max_tasks=None,
    harness="codex",
    provider=None,
):
    assignments = []
    for index in range(task_count):
        item = {
            "assignment_id": f"assignment-{index}",
            "task_id": f"task-{index}",
            "model": "gpt-5.4",
            "effort": "high",
        }
        if provider:
            item["provider"] = provider
        assignments.append(item)
    return {
        "schema_version": 1,
        "plan_id": "plan_test_123456",
        "plan_version": 1,
        "batch_id": BATCH_ID,
        "benchmark_id": "deep-swe",
        "harness": harness,
        "assignments": assignments,
        "concurrency": {"mode": mode, "value": concurrency},
        "refill": {
            "enabled": refill,
            "refill_to": refill_to,
            "max_tasks": max_tasks,
        },
        "locale": "zh-CN",
        "expires_at": "2099-01-01T00:00:00Z",
    }


def _server_response(plan, envelope=None, **extra):
    return {
        "schema_version": 1,
        "plan": plan,
        "state": {"devices": []},
        "envelope": envelope or _envelope(),
        **extra,
    }


def _state(tmp_path, plan):
    path = tmp_path / f"plan-{plan['plan_id']}.json"
    state = {
        "schema_version": 1,
        "credential_kind": "run_plan_v1",
        "server": "https://api.codexradar.com",
        "token": PLAN_TOKEN,
        "run_code_hash": run_plans._run_code_digest(RUN_CODE),
        "plan": plan,
        "plan_id": plan["plan_id"],
        "benchmark": plan["benchmark_id"],
        "batch_id": plan["batch_id"],
        "logical_session_id": "drl_logical_session_123456",
        "identity": {"nickname": "测试用户", "concurrent_limit": 8},
        "limits": {
            "account_concurrency": 8,
            "account_claim_limit": 8,
            "plan_task_limit": max(
                len(plan["assignments"]), int(plan["refill"].get("max_tasks") or 0),
            ),
        },
        "pending_decision": None,
        "pending_local_capacity": None,
        "authorized_concurrency": None,
    }
    run_plans._atomic_json(path, state)
    return path, state


def _args(*, concurrency=None, decision_token=None, scope=None):
    return SimpleNamespace(
        plan=RUN_CODE,
        server="https://api.codexradar.com",
        concurrency=concurrency,
        decision_token=decision_token,
        scope=scope,
        json=True,
    )


def _snapshot(*, available=4, auto_workers=4, account_limit=8):
    facts = {
        "safe_total": available,
        "reserved_by_other_runs": 0,
        "available": available,
        "auto_workers": auto_workers,
        "docker_cpus": 8,
        "docker_memory_gib": 32.0,
        "disk_limit": 8,
        "account_limit": account_limit,
        "held_tasks": 2,
        "automatic_cap": 4,
    }
    facts["digest"] = "capacity-snapshot-1"
    return facts


def _capacity_error(*, requested, available, original_mode):
    if available:
        envelope = _envelope(
            status=("capacity_changed" if original_mode == "auto" else "decision_required"),
            interaction=("notify" if original_mode == "auto" else "confirm"),
            decision_required=original_mode == "fixed",
            user_message="可用数量刚刚发生变化。",
            agent_action=(
                "retry_with_available_concurrency"
                if original_mode == "auto" else "ask_user"
            ),
            error_code="concurrency_capacity_reserved",
            retryable=True,
            choices=(
                [{"id": "lower_concurrency", "label": f"改为 {available} 个"},
                 {"id": "cancel", "label": "取消"}]
                if original_mode == "fixed" else []
            ),
        )
    else:
        envelope = _envelope(
            status="waiting",
            user_message="当前没有空余位置；我会等待后重试。",
            agent_action="wait_and_retry",
            error_code="concurrency_capacity_reserved",
            retryable=True,
        )
    payload = {
        "detail": "safe capacity detail",
        "code": "concurrency_capacity_reserved",
        "requested_concurrency": requested,
        "available_concurrency": available,
        "original_concurrency_mode": original_mode,
        "limiting_scope": "account",
        "account_concurrency": 8,
        "account_concurrency_in_use": 8 - available,
        "plan_concurrency": 8,
        "plan_concurrency_in_use": 8 - available,
        "envelope": envelope,
    }
    return ApiError(
        "unsafe transport message",
        status_code=409,
        code="concurrency_capacity_reserved",
        payload=payload,
    )


class FakeClient:
    def __init__(self, starts=None, progress=None, stops=None):
        self.starts = list(starts or [])
        self.progress_results = list(progress or [])
        self.stop_results = list(stops or [])
        self.start_calls = []
        self.progress_calls = []
        self.stop_calls = []

    @staticmethod
    def _next(values):
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def start_run_plan(self, **kwargs):
        self.start_calls.append(kwargs)
        return self._next(self.starts)

    def run_plan_progress(self, plan_id):
        self.progress_calls.append(plan_id)
        return self._next(self.progress_results)

    def stop_run_plan(self, **kwargs):
        self.stop_calls.append(kwargs)
        return self._next(self.stop_results)


def _prepare_run(
    monkeypatch,
    tmp_path,
    *,
    plan,
    client,
    snapshot=None,
    environment_issue=None,
):
    path, state = _state(tmp_path, plan)
    monkeypatch.setattr(
        run_plans,
        "_state_and_client",
        lambda _args: (RUN_CODE, path, state, client),
    )
    monkeypatch.setattr(
        doctor, "plan_environment_issue", lambda _plan: environment_issue,
    )
    if snapshot is not None:
        monkeypatch.setattr(
            run_plans,
            "_capacity_snapshot",
            lambda _client, _plan, _limits=None: snapshot,
        )
    monkeypatch.setattr(fleet, "batch_status", lambda _batch_id: None)
    return path, state


def test_cli_parses_user_intent_run_progress_and_stop_commands(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_run_plan", lambda args: seen.append(("run", args)) or 0)
    monkeypatch.setattr(
        cli, "cmd_progress_plan", lambda args: seen.append(("progress", args)) or 0,
    )
    monkeypatch.setattr(cli, "cmd_stop_plan", lambda args: seen.append(("stop", args)) or 0)

    assert cli.main([
        "run", "--plan", RUN_CODE,
        "--server", "https://api.claudecoderadar.com",
        "--concurrency", "auto", "--json",
    ]) == 0
    assert cli.main([
        "progress", "--plan", RUN_CODE,
        "--server", "https://api.claudecoderadar.com", "--json",
    ]) == 0
    assert cli.main([
        "stop", "--plan", RUN_CODE,
        "--server", "https://api.claudecoderadar.com",
        "--scope", "all-devices", "--decision-token", "drd_once", "--json",
    ]) == 0

    assert seen[0][1].concurrency == "auto"
    assert seen[0][1].server == "https://api.claudecoderadar.com"
    assert seen[1][1].plan == RUN_CODE
    assert seen[2][1].scope == "all-devices"
    assert seen[2][1].decision_token == "drd_once"


def test_exchange_keeps_run_code_out_of_state_and_uses_private_files(
    tmp_path, monkeypatch,
):
    plan = _plan()
    captured = {}

    class ExchangeClient:
        def __init__(self, server, token):
            captured["server"] = server
            captured["constructor_token"] = token

        def exchange_run_plan(self, **kwargs):
            captured["exchange"] = kwargs
            return {
                "schema_version": 1,
                "plan_access_token": PLAN_TOKEN,
                "access_expires_at": "2099-01-01T00:00:00Z",
                "plan": plan,
                "identity": {"nickname": "测试用户", "concurrent_limit": 8},
                "limits": {"account_concurrency": 8, "account_claim_limit": 8},
                "envelope": _envelope(status="resolved", agent_action="check_environment"),
            }

    monkeypatch.setattr(run_plans, "ApiClient", ExchangeClient)

    path, state = run_plans._exchange(
        RUN_CODE, "https://api.claudecoderadar.com", home=tmp_path,
    )

    assert captured["server"] == "https://api.claudecoderadar.com"
    assert captured["constructor_token"] == ""
    assert captured["exchange"]["run_code"] == RUN_CODE
    assert captured["exchange"]["device_id"].startswith("drv_")
    assert state["token"] == PLAN_TOKEN
    contents = path.read_text()
    assert RUN_CODE not in contents
    assert state["run_code_hash"] in contents
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    assert run_plans._saved_state(RUN_CODE, home=tmp_path)[0] == path
    assert run_plans.stable_device(tmp_path) == (
        captured["exchange"]["device_id"], captured["exchange"]["device_name"],
    )


def test_stable_device_is_single_identity_under_concurrent_first_use(tmp_path):
    with ThreadPoolExecutor(max_workers=12) as pool:
        identities = list(pool.map(lambda _index: run_plans.stable_device(tmp_path), range(30)))

    assert len(set(identities)) == 1
    device_path = tmp_path / run_plans.PLAN_DIR / run_plans.DEVICE_FILE
    assert json.loads(device_path.read_text())["device_id"] == identities[0][0]
    if os.name != "nt":
        assert device_path.stat().st_mode & 0o777 == 0o600


def test_stable_device_first_use_is_consistent_across_processes(tmp_path):
    source = Path(__file__).parent.parent / "src"
    barrier = tmp_path / "start"
    program = (
        "import sys,time;"
        f"sys.path.insert(0,{str(source)!r});"
        "from pathlib import Path;"
        "from dradar.run_plans import stable_device;"
        f"barrier=Path({str(barrier)!r});"
        "\nwhile not barrier.exists(): time.sleep(0.005)\n"
        f"print(stable_device(Path({str(tmp_path)!r}))[0],flush=True)"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _index in range(8)
    ]
    barrier.touch()
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        outputs.append(stdout.strip())

    assert len(set(outputs)) == 1
    assert outputs[0].startswith("drv_")


def test_concurrent_first_exchange_mints_one_logical_session(
    tmp_path, monkeypatch,
):
    plan = _plan()
    exchanges = []

    class ExchangeClient:
        def __init__(self, _server, token, **_kwargs):
            self.token = token

        def exchange_run_plan(self, **kwargs):
            exchanges.append(kwargs)
            time.sleep(0.05)
            return {
                "schema_version": 1,
                "plan_access_token": PLAN_TOKEN,
                "access_expires_at": "2099-01-01T00:00:00Z",
                "plan": plan,
                "identity": {"nickname": "测试用户", "concurrent_limit": 8},
                "limits": {"account_concurrency": 8, "plan_task_limit": 2},
                "envelope": _envelope(status="resolved"),
            }

    monkeypatch.setattr(run_plans, "HOME", tmp_path)
    monkeypatch.setattr(run_plans, "ApiClient", ExchangeClient)
    args = _args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: run_plans._state_and_client(args), range(2)))

    assert len(exchanges) == 1
    assert results[0][2]["logical_session_id"] == results[1][2]["logical_session_id"]
    assert results[0][2]["token"] == PLAN_TOKEN


def test_exchange_http_contract_uses_json_body_without_authorization():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = request.url.query
        captured["authorization"] = request.headers.get("Authorization")
        captured["json"] = json.loads(request.read())
        return httpx.Response(200, json={"schema_version": 1})

    client = ApiClient(
        "https://api.codexradar.com",
        "",
        transport=httpx.MockTransport(handler),
        capabilities=(),
    )
    client.exchange_run_plan(run_code=RUN_CODE, device_id="drv_device")

    assert captured == {
        "method": "POST",
        "path": "/api/v1/run-plans/exchange",
        "query": b"",
        "authorization": None,
        "json": {
            "schema_version": 1,
            "run_code": RUN_CODE,
            "device_id": "drv_device",
        },
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://api.codexradar.com", "https://api.codexradar.com"),
        ("https://api.claudecoderadar.com/", "https://api.claudecoderadar.com"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    ],
)
def test_server_url_accepts_two_public_sites_and_loopback(value, expected):
    assert run_plans.validate_server_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "http://api.codexradar.com",
        "https://user:pass@api.codexradar.com",
        "https://api.codexradar.com/path",
        "https://api.codexradar.com/?token=secret",
    ],
)
def test_server_url_fails_closed_for_unsafe_values(value):
    with pytest.raises(run_plans.RunPlanClientError) as raised:
        run_plans.validate_server_url(value)
    assert raised.value.code == "server_url_invalid"


def test_user_error_messages_do_not_expose_agent_or_scheduler_terms():
    errors = [
        run_plans.RunPlanClientError(
            "run_code_invalid", "网页复制的运行信息无效，请回网页重新复制。",
        ),
        run_plans.RunPlanClientError(
            "local_capacity_unavailable",
            "这台设备当前没有空余运行位置；请等待其他运行结束后重试。",
        ),
    ]
    forbidden = (
        "batch_id", "Fleet", "provider", "refill", "worker",
        "assignment", "leases", "运行码",
    )
    for error in errors:
        message = run_plans._local_error_response(error)["user_message"]
        assert all(term not in message for term in forbidden)


def _write_credential_state(home, *, index, expires_at, token=None):
    plan_id = f"plan_cleanup_{index:04d}"
    path = run_plans._state_path(plan_id, home)
    run_plans._atomic_json(path, {
        "schema_version": 1,
        "credential_kind": "run_plan_v1",
        "server": "https://api.codexradar.com",
        "token": token or f"drp_cleanup_secret_{index}",
        "access_expires_at": expires_at,
        "run_code_hash": run_plans._run_code_digest(f"run_cleanup_{index:04d}"),
        "plan_id": plan_id,
        "plan": {"expires_at": expires_at},
    })
    return path


def test_expired_state_is_not_reused_and_token_is_scrubbed(tmp_path):
    code = "run_cleanup_0001"
    path = _write_credential_state(
        tmp_path, index=1, expires_at="2000-01-01T00:00:00Z",
        token="drp_expired_secret",
    )

    assert run_plans._saved_state(code, home=tmp_path) is None
    payload = json.loads(path.read_text())
    assert payload["credential_kind"] == "run_plan_expired_summary_v1"
    assert "token" not in payload
    assert "drp_expired_secret" not in path.read_text()


def test_expired_credentials_used_by_active_fleet_are_preserved_but_not_reused(
    tmp_path, monkeypatch,
):
    path = _write_credential_state(
        tmp_path, index=2, expires_at="2000-01-01T00:00:00Z",
        token="drp_active_closing_secret",
    )
    monkeypatch.setattr(
        fleet, "credentials_file_in_use", lambda candidate, **_kwargs: Path(candidate) == path,
    )

    run_plans._cleanup_states(tmp_path)

    assert "drp_active_closing_secret" in path.read_text()
    assert run_plans._saved_state("run_cleanup_0002", home=tmp_path) is None


def test_inactive_plan_state_files_have_bounded_credential_and_audit_counts(tmp_path):
    for index in range(140):
        _write_credential_state(
            tmp_path, index=index, expires_at="2099-01-01T00:00:00Z",
        )

    run_plans._cleanup_states(tmp_path)

    payloads = [
        json.loads(path.read_text())
        for path in (tmp_path / run_plans.PLAN_DIR).glob("plan-*.json")
    ]
    credentials = [
        item for item in payloads if item["credential_kind"] == "run_plan_v1"
    ]
    summaries = [
        item for item in payloads
        if item["credential_kind"] == "run_plan_expired_summary_v1"
    ]
    assert len(credentials) <= run_plans.MAX_CREDENTIAL_STATES
    assert len(summaries) <= run_plans.MAX_AUDIT_SUMMARIES
    assert len(payloads) <= (
        run_plans.MAX_CREDENTIAL_STATES + run_plans.MAX_AUDIT_SUMMARIES
    )
    assert all("token" not in item for item in summaries)


def test_auto_refill_uses_safe_effective_concurrency_not_seed_count(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(refill=True, max_tasks=20, task_count=2)
    client = FakeClient(starts=[_server_response(
        plan,
        _envelope(agent_action="start_runner"),
    )])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=4, auto_workers=4),
    )
    added = []

    def add_batch(**kwargs):
        added.append(kwargs)
        return {"batch": {"workers": kwargs["workers"]}}

    monkeypatch.setattr(fleet, "add_batch", add_batch)

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "started"
    assert payload["agent_action"] == "monitor"
    assert payload["agent"]["server_status"]["agent_action"] == "start_runner"
    assert client.start_calls[0]["concurrency_mode"] == "fixed"
    assert client.start_calls[0]["concurrency"] == 4
    assert added[0]["workers"] == 4
    assert added[0]["refill"] is True
    assert added[0]["max_tasks"] == 20
    assert added[0]["batch_id"] == BATCH_ID


def test_auto_resource_downgrade_returns_one_top_level_warn_envelope(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(refill=True, max_tasks=20, task_count=2)
    server = _server_response(plan)
    client = FakeClient(starts=[server])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=2, auto_workers=2),
    )
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["interaction"] == "warn"
    assert payload["decision_required"] is False
    assert payload["agent_action"] == "monitor"
    assert payload["agent"]["selected_concurrency"] == 2
    assert payload["agent"]["server_status"]["status"] == "started"
    assert "envelope" not in payload


def test_auto_capacity_reservation_race_retries_lower_and_warns(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(refill=True, max_tasks=20, task_count=4)
    client = FakeClient(starts=[
        _capacity_error(requested=4, available=2, original_mode="auto"),
        _server_response(plan),
    ])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=4, auto_workers=4),
    )
    added = []
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: added.append(kwargs) or {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)

    assert [call["concurrency"] for call in client.start_calls] == [4, 2]
    assert added[0]["workers"] == 2
    assert payload["interaction"] == "warn"
    assert payload["decision_required"] is False
    assert payload["agent"]["selected_concurrency"] == 2


def test_fixed_capacity_reservation_race_requires_one_use_lower_decision(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(mode="fixed", concurrency=4, task_count=4)
    client = FakeClient(starts=[
        _capacity_error(requested=4, available=2, original_mode="fixed"),
        _server_response(plan),
    ])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=4, auto_workers=4),
    )
    added = []
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: added.append(kwargs) or {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    confirm = json.loads(capsys.readouterr().out)
    token = confirm["decision_token"]
    assert confirm["decision"] == "server_capacity"
    assert confirm["decision_required"] is True
    assert confirm["choices"] == [
        {"id": "use_recommended", "label": "按建议数量运行（2 道）"},
        {"id": "cancel", "label": "取消"},
    ]
    assert confirm["agent"]["choice_actions"]["use_recommended"]["args"] == [
        "--concurrency", "2", "--decision-token", token,
    ]
    assert all("arguments" not in choice for choice in confirm["choices"])
    assert added == []

    assert run_plans.cmd_run_plan(
        _args(concurrency=2, decision_token=token),
    ) == 0
    started = json.loads(capsys.readouterr().out)
    assert started["status"] == "started"
    assert [call["concurrency"] for call in client.start_calls] == [4, 2]
    assert added[0]["workers"] == 2


def test_capacity_reservation_with_zero_available_waits_without_phantom_pool(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(refill=True, max_tasks=20, task_count=4)
    client = FakeClient(starts=[
        _capacity_error(requested=4, available=0, original_mode="auto"),
    ])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=4, auto_workers=4),
    )
    monkeypatch.setattr(
        fleet, "add_batch", lambda **_kwargs: pytest.fail("waiting cannot start Fleet"),
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "waiting"
    assert payload["agent_action"] == "wait_and_retry"
    assert len(client.start_calls) == 1


def test_server_stopped_start_response_never_launches_a_local_pool(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    stopped = _server_response(plan, _envelope(
        status="stopped",
        user_message="这次运行已经停止，不会再开始新题。",
        agent_action="stop_runner",
    ))
    client = FakeClient(starts=[stopped])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=2, auto_workers=2),
    )
    stopped_batches = []
    monkeypatch.setattr(
        fleet, "add_batch", lambda **_kwargs: pytest.fail("stopped plan cannot start"),
    )
    monkeypatch.setattr(fleet, "stop_batch", stopped_batches.append)

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stopped"
    assert payload["agent_action"] == "stop_runner"
    assert stopped_batches == [BATCH_ID]


def test_fixed_capacity_decision_precedes_server_start_and_is_one_use(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(mode="fixed", concurrency=4, task_count=4)
    client = FakeClient(starts=[_server_response(plan)])
    path, state = _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=2, auto_workers=2),
    )
    added = []
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: added.append(kwargs) or {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    decision = json.loads(capsys.readouterr().out)
    token = decision["decision_token"]
    assert decision["decision"] == "local_capacity"
    assert decision["decision_required"] is True
    assert client.start_calls == []
    assert added == []
    assert token not in path.read_text()
    assert state["pending_local_capacity"]["token_hash"]

    assert run_plans.cmd_run_plan(
        _args(concurrency=2, decision_token=token),
    ) == 0
    started = json.loads(capsys.readouterr().out)
    assert started["status"] == "started"
    assert client.start_calls[0]["concurrency"] == 2
    assert added[0]["workers"] == 2

    assert run_plans.cmd_run_plan(
        _args(concurrency=2, decision_token=token),
    ) == 1
    reused = json.loads(capsys.readouterr().out)
    assert reused["error_code"] == "decision_invalid_or_capacity_changed"
    assert len(client.start_calls) == 1


def test_explicit_safe_lower_fixed_count_needs_no_redundant_confirmation(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(mode="fixed", concurrency=4, task_count=4)
    client = FakeClient(starts=[_server_response(plan)])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=4, auto_workers=4),
    )
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args(concurrency=2)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision_required"] is False
    assert client.start_calls[0]["concurrency"] == 2


def test_explicit_concurrency_cannot_launch_empty_workers_beyond_plan_supply(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(task_count=2, refill=False)
    client = FakeClient(starts=[_server_response(plan)])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=40, auto_workers=2, account_limit=40),
    )
    monkeypatch.setattr(
        fleet, "add_batch", lambda **_kwargs: pytest.fail("must not start Fleet"),
    )

    assert run_plans.cmd_run_plan(_args(concurrency=40)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "concurrency_not_allowed"
    assert client.start_calls == []


def test_other_device_confirmation_is_server_authoritative_and_starts_nothing(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    confirm = _envelope(
        status="decision_required",
        interaction="confirm",
        decision_required=True,
        user_message="另一台设备正在运行这次领取，是否让这台设备也一起处理？",
        agent_action="ask_user",
        decision="join_existing",
        decision_token="drd_join_once",
        choices=[
            {"id": "join_existing", "label": "一起运行"},
            {"id": "cancel", "label": "取消"},
        ],
    )
    client = FakeClient(starts=[_server_response(plan, confirm)])
    _path, state = _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=2, auto_workers=2),
    )
    monkeypatch.setattr(
        fleet, "add_batch", lambda **_kwargs: pytest.fail("confirmation cannot start Fleet"),
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["decision"] == "join_existing"
    assert payload["decision_token"] == "drd_join_once"
    assert payload["interaction"] == "confirm"
    assert payload["agent"]["state"] == {"devices": []}
    assert payload["agent"]["choice_actions"] == {
        "join_existing": {
            "mode": "replay_current_command_with_args",
            "args": ["--decision-token", "drd_join_once"],
        },
        "cancel": {"mode": "no_command", "args": []},
    }
    assert state["pending_decision"] == {
        "command": "run", "decision": "join_existing",
    }


def test_join_confirmation_then_fixed_capacity_lowering_does_not_reask_join(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(mode="fixed", concurrency=4, task_count=4)
    join = _server_response(plan, _envelope(
        status="decision_required",
        interaction="confirm",
        decision_required=True,
        user_message="另一台设备正在运行，是否一起处理？",
        agent_action="ask_user",
        decision="join_existing",
        decision_token="drd_join_once",
        choices=[
            {"id": "join_existing", "label": "一起运行"},
            {"id": "cancel", "label": "取消"},
        ],
    ))
    client = FakeClient(starts=[
        join,
        _capacity_error(requested=4, available=2, original_mode="fixed"),
        _server_response(plan),
    ])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=4, auto_workers=4),
    )
    added = []
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: added.append(kwargs) or {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["decision"] == "join_existing"

    assert run_plans.cmd_run_plan(
        _args(decision_token=first["decision_token"]),
    ) == 0
    lower = json.loads(capsys.readouterr().out)
    composite = lower["decision_token"]
    assert lower["decision"] == "server_capacity"
    assert composite.startswith("drlc_")
    # The private state binds only a digest; it never stores the server token.
    state_text = next(tmp_path.glob("plan-*.json")).read_text()
    assert "drd_join_once" not in state_text

    assert run_plans.cmd_run_plan(
        _args(concurrency=2, decision_token=composite),
    ) == 0
    final = json.loads(capsys.readouterr().out)
    assert final["decision_required"] is False
    assert client.start_calls[1]["decision"] == "join_existing"
    assert client.start_calls[2]["decision"] == "join_existing"
    assert client.start_calls[2]["decision_token"] == "drd_join_once"
    assert added[0]["workers"] == 2


def test_lost_start_response_retry_ensures_missing_local_fleet_pool(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    already_running = _server_response(
        plan,
        _envelope(
            status="already_running",
            user_message="这台设备已经在运行，正在继续监控。无需操作。",
        ),
    )
    client = FakeClient(starts=[ApiError("response lost"), already_running])
    _prepare_run(
        monkeypatch, tmp_path, plan=plan, client=client,
        snapshot=_snapshot(available=2, auto_workers=2),
    )
    added = []
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: added.append(kwargs) or {"batch": {"workers": kwargs["workers"]}},
    )

    assert run_plans.cmd_run_plan(_args()) == 1
    first = json.loads(capsys.readouterr().out)
    assert first["error_code"] == "service_unavailable"
    assert added == []

    assert run_plans.cmd_run_plan(_args()) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["status"] == "already_running"
    assert second["agent_action"] == "monitor"
    assert second["agent"]["server_status"]["status"] == "already_running"
    assert len(added) == 1
    assert added[0]["credentials_file"].name.startswith("plan-")


def test_same_device_with_live_local_pool_is_idempotently_ensured(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    client = FakeClient(starts=[_server_response(
        plan, _envelope(status="already_running"),
    )])
    path, state = _state(tmp_path, plan)
    monkeypatch.setattr(
        run_plans,
        "_state_and_client",
        lambda _args: (RUN_CODE, path, state, client),
    )
    monkeypatch.setattr(
        fleet,
        "batch_status",
        lambda _batch: {
            "status": "running", "plan_id": plan["plan_id"], "workers": 2,
        },
    )
    monkeypatch.setattr(
        run_plans,
        "_capacity_snapshot",
        lambda *_args, **_kwargs: pytest.fail("idempotent monitor needs no capacity check"),
    )
    ensured = []
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: ensured.append(kwargs) or {
            "already_active": True,
            "batch": {"workers": kwargs["workers"]},
        },
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "already_running"
    assert client.start_calls[0]["concurrency"] == 2
    assert len(ensured) == 1
    assert ensured[0]["workers"] == 2


@pytest.mark.parametrize(
    "status,expected_code",
    [
        ("running", "local_concurrency_change_requires_restart"),
        ("stopping", "local_run_stopping"),
    ],
)
def test_active_local_pool_is_not_silently_resized_or_reactivated(
    status, expected_code, tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    client = FakeClient(starts=[_server_response(plan)])
    path, state = _state(tmp_path, plan)
    monkeypatch.setattr(
        run_plans, "_state_and_client",
        lambda _args: (RUN_CODE, path, state, client),
    )
    monkeypatch.setattr(
        fleet, "batch_status",
        lambda _batch: {
            "status": status, "plan_id": plan["plan_id"], "workers": 2,
        },
    )
    monkeypatch.setattr(
        fleet, "add_batch", lambda **_kwargs: pytest.fail("must not resize or restart"),
    )

    assert run_plans.cmd_run_plan(_args(concurrency=1)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == expected_code
    assert client.start_calls == []


def test_orphaned_live_pool_is_counted_and_never_spawned_twice(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    client = FakeClient(starts=[_server_response(
        plan, _envelope(status="already_running"),
    )])
    path, state = _state(tmp_path, plan)
    monkeypatch.setattr(
        run_plans, "_state_and_client",
        lambda _args: (RUN_CODE, path, state, client),
    )
    monkeypatch.setattr(
        fleet, "batch_status",
        lambda _batch: {
            "status": "orphaned", "plan_id": plan["plan_id"], "workers": 2,
        },
    )
    monkeypatch.setattr(
        fleet, "add_batch", lambda **_kwargs: pytest.fail("orphan lock forbids duplicate"),
    )

    assert run_plans.cmd_run_plan(_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "already_running"
    assert client.start_calls[0]["concurrency"] == 2


def test_concurrent_auto_plans_share_one_atomic_local_admission_budget(
    tmp_path, monkeypatch,
):
    plan_a = _plan(task_count=4)
    plan_b = json.loads(json.dumps(_plan(task_count=4)))
    plan_b["plan_id"] = "plan_test_second_123456"
    plan_b["batch_id"] = "87654321876543218765432187654321"
    path_a, state_a = _state(tmp_path, plan_a)
    path_b, state_b = _state(tmp_path, plan_b)
    client_a = FakeClient(starts=[_server_response(plan_a)])
    client_b = FakeClient(starts=[_server_response(plan_b)])
    contexts = {
        "run_concurrent_plan_a": (path_a, state_a, client_a),
        "run_concurrent_plan_b": (path_b, state_b, client_b),
    }
    reservations = {}
    first_snapshot = threading.Event()
    outputs = {}

    monkeypatch.setattr(run_plans, "HOME", tmp_path)
    monkeypatch.setattr(
        run_plans,
        "_state_and_client",
        lambda args: (args.plan, *contexts[args.plan]),
    )
    monkeypatch.setattr(doctor, "plan_environment_issue", lambda _plan: None)
    monkeypatch.setattr(
        fleet, "batch_status", lambda batch_id: reservations.get(batch_id),
    )

    def snapshot(_client, _plan, _limits=None):
        used = sum(item["workers"] for item in reservations.values())
        if not first_snapshot.is_set():
            first_snapshot.set()
            time.sleep(0.1)
        available = max(0, 4 - used)
        result = _snapshot(
            available=available, auto_workers=min(4, available), account_limit=8,
        )
        result["digest"] = f"used-{used}"
        return result

    def add_batch(**kwargs):
        reservations[kwargs["batch_id"]] = {
            "status": "running",
            "plan_id": kwargs["plan_id"],
            "workers": kwargs["workers"],
        }
        return {"batch": reservations[kwargs["batch_id"]]}

    monkeypatch.setattr(run_plans, "_capacity_snapshot", snapshot)
    monkeypatch.setattr(fleet, "add_batch", add_batch)
    monkeypatch.setattr(
        run_plans,
        "_output",
        lambda args, response: outputs.setdefault(args.plan, response) is None and 0 or 0,
    )

    def args(code):
        return SimpleNamespace(
            plan=code, server="https://api.codexradar.com", concurrency=None,
            decision_token=None, scope=None, json=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_plans.cmd_run_plan, args("run_concurrent_plan_a"))
        assert first_snapshot.wait(timeout=2)
        second = pool.submit(run_plans.cmd_run_plan, args("run_concurrent_plan_b"))
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert sorted(results) == [0, 1]
    assert sum(item["workers"] for item in reservations.values()) == 4
    assert len(client_a.start_calls) + len(client_b.start_calls) == 1
    assert outputs["run_concurrent_plan_b"]["error_code"] == "local_capacity_unavailable"


def test_fixed_plan_rechecks_capacity_after_concurrent_plan_reservation(
    tmp_path, monkeypatch,
):
    plan_a = _plan(task_count=3)
    plan_b = _plan(mode="fixed", concurrency=2, task_count=2)
    plan_b["plan_id"] = "plan_test_fixed_second_123456"
    plan_b["batch_id"] = "abcdefabcdefabcdefabcdefabcdefab"
    path_a, state_a = _state(tmp_path, plan_a)
    path_b, state_b = _state(tmp_path, plan_b)
    client_a = FakeClient(starts=[_server_response(plan_a)])
    client_b = FakeClient(starts=[_server_response(plan_b)])
    contexts = {
        "run_capacity_plan_a": (path_a, state_a, client_a),
        "run_capacity_plan_b": (path_b, state_b, client_b),
    }
    reservations = {}
    first_snapshot = threading.Event()
    outputs = {}
    monkeypatch.setattr(run_plans, "HOME", tmp_path)
    monkeypatch.setattr(
        run_plans, "_state_and_client",
        lambda args: (args.plan, *contexts[args.plan]),
    )
    monkeypatch.setattr(doctor, "plan_environment_issue", lambda _plan: None)
    monkeypatch.setattr(fleet, "batch_status", lambda batch: reservations.get(batch))

    def snapshot(_client, plan, _limits=None):
        used = sum(item["workers"] for item in reservations.values())
        if plan["plan_id"] == plan_a["plan_id"]:
            first_snapshot.set()
            time.sleep(0.1)
        available = max(0, 4 - used)
        result = _snapshot(
            available=available,
            auto_workers=min(len(plan["assignments"]), available),
            account_limit=8,
        )
        result["digest"] = f"used-{used}"
        return result

    monkeypatch.setattr(run_plans, "_capacity_snapshot", snapshot)
    monkeypatch.setattr(
        fleet,
        "add_batch",
        lambda **kwargs: reservations.setdefault(kwargs["batch_id"], {
            "status": "running", "plan_id": kwargs["plan_id"],
            "workers": kwargs["workers"],
        }) and {"batch": reservations[kwargs["batch_id"]]},
    )
    monkeypatch.setattr(
        run_plans, "_output",
        lambda args, response: outputs.__setitem__(args.plan, response) or 0,
    )

    def args(code):
        return SimpleNamespace(
            plan=code, server="https://api.codexradar.com", concurrency=None,
            decision_token=None, scope=None, json=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_plans.cmd_run_plan, args("run_capacity_plan_a"))
        assert first_snapshot.wait(timeout=2)
        second = pool.submit(run_plans.cmd_run_plan, args("run_capacity_plan_b"))
        assert first.result(timeout=5) == 0
        assert second.result(timeout=5) == 0

    assert sum(item["workers"] for item in reservations.values()) == 3
    assert client_b.start_calls == []
    assert outputs["run_capacity_plan_b"]["decision_required"] is True
    assert outputs["run_capacity_plan_b"]["agent"]["recommended_concurrency"] == 1


def test_stop_linearizes_after_inflight_start_and_stops_the_new_pool(
    tmp_path, monkeypatch,
):
    plan = _plan(task_count=2)
    path, state = _state(tmp_path, plan)
    stopped = _server_response(plan, _envelope(
        status="stopped",
        user_message="已停止这台设备。",
        agent_action="stop_runner",
    ))
    client = FakeClient(starts=[_server_response(plan)], stops=[stopped])
    monkeypatch.setattr(run_plans, "HOME", tmp_path)
    monkeypatch.setattr(
        run_plans, "_state_and_client",
        lambda _args: (RUN_CODE, path, state, client),
    )
    monkeypatch.setattr(doctor, "plan_environment_issue", lambda _plan: None)
    monkeypatch.setattr(
        run_plans, "_capacity_snapshot",
        lambda *_args, **_kwargs: _snapshot(available=2, auto_workers=2),
    )
    local = {}
    events = []
    add_entered = threading.Event()
    allow_add = threading.Event()
    monkeypatch.setattr(fleet, "batch_status", lambda batch: local.get(batch))

    def add_batch(**kwargs):
        add_entered.set()
        assert allow_add.wait(timeout=3)
        local[kwargs["batch_id"]] = {
            "status": "running", "plan_id": kwargs["plan_id"],
            "workers": kwargs["workers"],
        }
        events.append("add")
        return {"batch": local[kwargs["batch_id"]]}

    def stop_batch(batch_id):
        events.append("stop")
        local[batch_id]["status"] = "stopping"

    monkeypatch.setattr(fleet, "add_batch", add_batch)
    monkeypatch.setattr(fleet, "stop_batch", stop_batch)
    monkeypatch.setattr(run_plans, "_output", lambda _args, _response: 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        starting = pool.submit(run_plans.cmd_run_plan, _args())
        assert add_entered.wait(timeout=3)
        stopping = pool.submit(
            run_plans.cmd_stop_plan, _args(scope="this-device"),
        )
        time.sleep(0.05)
        assert client.stop_calls == []
        allow_add.set()
        assert starting.result(timeout=5) == 0
        assert stopping.result(timeout=5) == 0

    assert events == ["add", "stop"]
    assert client.stop_calls[0]["scope"] == "this_device"


def test_current_plan_environment_failure_happens_before_server_start(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan(harness="grok-build")
    client = FakeClient(starts=[_server_response(plan)])
    _prepare_run(
        monkeypatch,
        tmp_path,
        plan=plan,
        client=client,
        environment_issue={
            "error_code": "current_tool_not_ready",
            "user_message": "这次运行需要 Grok；请完成 Grok 的安装和登录后重试。",
            "agent_action": "setup_current_tool",
        },
    )
    monkeypatch.setattr(
        run_plans,
        "_capacity_snapshot",
        lambda *_args: pytest.fail("environment failure must precede capacity"),
    )

    assert run_plans.cmd_run_plan(_args()) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "current_tool_not_ready"
    assert payload["agent_action"] == "setup_current_tool"
    assert client.start_calls == []


def test_codex_plan_does_not_probe_unrelated_grok_or_kimi_credentials(
    tmp_path, monkeypatch,
):
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(doctor, "_probe", lambda _command: True)
    monkeypatch.setattr(doctor.runner, "ensure_pier", lambda: None)
    monkeypatch.setattr(doctor.runner, "_resolve_user_tool", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(doctor.runner, "codex_auth_path", lambda: auth)
    monkeypatch.setattr(
        doctor, "grok_auth_error", lambda: pytest.fail("Grok is unrelated"),
    )
    monkeypatch.setattr(
        doctor, "kimi_auth_error", lambda: pytest.fail("Kimi is unrelated"),
    )

    assert doctor.plan_environment_issue(_plan(harness="codex")) is None


def test_server_wire_dsh_harness_maps_to_local_dsh_minimal_preflight(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        doctor.shutil, "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr(doctor, "_probe", lambda _command: True)
    monkeypatch.setattr(doctor.runner, "ensure_pier", lambda: None)
    monkeypatch.setattr(
        doctor.runner, "_resolve_user_tool",
        lambda name: "/usr/bin/uvx" if name == "uvx" else None,
    )
    monkeypatch.setattr(doctor, "deepseek_api_key", lambda: "configured")

    assert doctor.plan_environment_issue(_plan(harness="dsh")) is None
    recovery = doctor._plan_agent_recovery("dsh", setup_provider="deepseek")
    assert recovery["next_commands"][-1]["argv"] == [
        "dradar", "doctor", "--agent", "dsh-minimal",
    ]


def test_missing_current_tool_on_second_machine_has_actionable_issue(
    monkeypatch,
):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(doctor, "_probe", lambda _command: True)
    monkeypatch.setattr(doctor.runner, "ensure_pier", lambda: None)
    monkeypatch.setattr(doctor, "grok_cli_path", lambda: None)

    issue = doctor.plan_environment_issue(_plan(harness="grok-build"))

    assert issue["error_code"] == "current_tool_not_ready"
    assert issue["user_message"] == "这次运行需要 Grok；请完成 Grok 的安装和登录后重试。"
    assert issue["agent_action"] == "setup_current_tool"
    assert issue["agent"]["requires_user_action"] is True
    assert [item["argv"] for item in issue["agent"]["next_commands"]] == [
        ["dradar", "provider", "setup", "grok"],
        ["dradar", "provider", "status", "grok", "--live"],
        ["dradar", "doctor", "--agent", "grok-build"],
    ]


@pytest.mark.parametrize(
    "harness,provider",
    [
        ("dsh-minimal", "deepseek"),
        ("grok-build", "grok"),
        ("kimi-code", "kimi"),
        ("zcode", "zcode"),
        ("antigravity", "antigravity"),
        ("codebuddy", "codebuddy"),
    ],
)
def test_every_optional_tool_has_versioned_nonsecret_agent_commands(
    harness, provider,
):
    recovery = doctor._plan_agent_recovery(harness, setup_provider=provider)

    assert recovery["schema_version"] == 1
    assert recovery["requires_user_action"] is True
    commands = [item["argv"] for item in recovery["next_commands"]]
    assert commands == [
        ["dradar", "provider", "setup", provider],
        ["dradar", "provider", "status", provider, "--live"],
        ["dradar", "doctor", "--agent", harness],
    ]
    serialized = json.dumps(recovery)
    assert PLAN_TOKEN not in serialized
    assert RUN_CODE not in serialized


def test_codex_login_and_docker_recovery_require_user_action(monkeypatch):
    codex = doctor._plan_agent_recovery("codex", codex_login=True)
    assert [item["argv"] for item in codex["next_commands"]] == [
        ["codex", "login"],
        ["dradar", "doctor", "--agent", "codex"],
    ]
    assert codex["next_commands"][0]["interactive"] is True

    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    issue = doctor.plan_environment_issue(_plan(harness="codex"))
    assert issue["error_code"] == "docker_not_installed"
    assert issue["agent"]["requires_user_action"] is True
    assert issue["agent"]["next_commands"] == [{
        "argv": ["dradar", "doctor", "--agent", "codex"],
        "interactive": False,
        "purpose": "verify_current_environment",
    }]


@pytest.mark.parametrize("status_code", [409, 410, 429])
def test_structured_server_error_envelope_is_preserved(status_code):
    envelope = _envelope(
        status="decision_required" if status_code == 409 else "error",
        interaction="confirm" if status_code == 409 else "notify",
        decision_required=status_code == 409,
        user_message=f"safe server message {status_code}",
        agent_action="ask_user" if status_code == 409 else "stop",
        error_code=f"server_error_{status_code}",
        choices=[{"id": "cancel", "label": "取消"}] if status_code == 409 else [],
        decision="recover_stale" if status_code == 409 else None,
        decision_token="drd_server_once" if status_code == 409 else None,
    )
    exc = ApiError(
        "unsafe transport prose",
        status_code=status_code,
        code=f"server_error_{status_code}",
        payload={
            "detail": "safe detail",
            "code": f"server_error_{status_code}",
            "envelope": envelope,
        },
    )

    result = run_plans._api_error_response(exc)

    assert result["user_message"] == f"safe server message {status_code}"
    assert result["error_code"] == f"server_error_{status_code}"
    assert result["decision_required"] is (status_code == 409)
    assert "unsafe transport prose" not in json.dumps(result)


def test_progress_and_stop_reuse_saved_plan_access_without_exchange(
    tmp_path, monkeypatch, capsys,
):
    plan = _plan()
    path, state = _state(tmp_path, plan)
    progress = _server_response(
        plan,
        _envelope(status="running", user_message="正在运行 1 道题。无需操作。"),
        progress={"running": 1, "waiting": 1, "submitted": 0},
    )
    stopped = _server_response(
        plan,
        _envelope(
            status="stopped",
            user_message="已停止这台设备，其他设备不受影响。",
            agent_action="stop_runner",
        ),
    )
    client = FakeClient(progress=[progress], stops=[stopped])
    monkeypatch.setattr(
        run_plans, "_saved_state", lambda _code, **_kwargs: (path, state),
    )
    monkeypatch.setattr(
        run_plans,
        "_exchange",
        lambda *_args, **_kwargs: pytest.fail("saved commands must not exchange again"),
    )
    monkeypatch.setattr(run_plans, "ApiClient", lambda *_args, **_kwargs: client)
    stopped_batches = []
    monkeypatch.setattr(fleet, "stop_batch", stopped_batches.append)

    assert run_plans.cmd_progress_plan(_args()) == 0
    progress_payload = json.loads(capsys.readouterr().out)
    assert progress_payload["status"] == "running"
    assert progress_payload["agent"]["progress"]["running"] == 1

    assert run_plans.cmd_stop_plan(_args(scope="this-device")) == 0
    stop_payload = json.loads(capsys.readouterr().out)
    assert stop_payload["status"] == "stopped"
    assert client.stop_calls[0]["scope"] == "this_device"
    assert stopped_batches == [BATCH_ID]


def test_plan_scoped_capacity_uses_identity_and_exact_inventory_not_whoami(
    monkeypatch,
):
    paths = []

    def handler(request):
        paths.append(request.url.path + (f"?{request.url.query.decode()}" if request.url.query else ""))
        if request.url.path == "/api/v1/run-plans/identity":
            return httpx.Response(200, json={
                "nickname": "测试用户", "concurrent_limit": 8, "claim_limit": 6,
            })
        if request.url.path == "/api/v1/assignment":
            active = [
                {"assignment_id": f"a-{index}", "batch_id": BATCH_ID}
                for index in range(2)
            ]
            return httpx.Response(200, json={"active": active})
        return httpx.Response(404, json={"detail": "unexpected"})

    client = ApiClient(
        "https://api.codexradar.com",
        PLAN_TOKEN,
        transport=httpx.MockTransport(handler),
        capabilities=(),
        benchmark_id="deep-swe",
        batch_id=BATCH_ID,
    )
    monkeypatch.setattr("dradar.capacity.docker_resources", lambda: (16, 64.0, ()))
    monkeypatch.setattr(
        "dradar.capacity.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=200 * 1024 ** 3),
    )
    monkeypatch.setattr(fleet, "reserved_workers", lambda **_kwargs: 1)

    snapshot = run_plans._capacity_snapshot(
        client,
        _plan(refill=True, max_tasks=20),
        {
            "account_concurrency": 8,
            "account_claim_limit": 6,
            "plan_task_limit": 20,
        },
    )

    assert "/api/v1/run-plans/identity" in paths
    assert not any(path == "/api/v1/whoami" for path in paths)
    assert any(path.startswith("/api/v1/assignment?") and "batch_id=" in path for path in paths)
    assert snapshot["account_limit"] == 6
    assert snapshot["reserved_by_other_runs"] == 1
    assert snapshot["auto_workers"] == 4


def test_empty_exact_inventory_is_left_for_server_no_remaining_decision(
    monkeypatch,
):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/v1/run-plans/identity":
            return httpx.Response(200, json={
                "nickname": "测试用户", "concurrent_limit": 4, "claim_limit": 4,
            })
        if request.url.path == "/api/v1/assignment":
            return httpx.Response(404, json={
                "detail": "not found", "code": "claim_batch_not_found",
            })
        return httpx.Response(404)

    client = ApiClient(
        "https://api.codexradar.com",
        PLAN_TOKEN,
        transport=httpx.MockTransport(handler),
        capabilities=(),
        benchmark_id="deep-swe",
        batch_id=BATCH_ID,
    )
    monkeypatch.setattr("dradar.capacity.docker_resources", lambda: (8, 32.0, ()))
    monkeypatch.setattr(
        "dradar.capacity.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024 ** 3),
    )
    monkeypatch.setattr(fleet, "reserved_workers", lambda **_kwargs: 0)

    snapshot = run_plans._capacity_snapshot(
        client,
        _plan(task_count=2),
        {"account_concurrency": 4, "plan_task_limit": 2},
    )

    assert snapshot["held_tasks"] == 0
    assert snapshot["auto_workers"] == 2
    assert "/api/v1/run-plans/identity" in paths
    assert "/api/v1/assignment" in paths


def test_invalid_plan_or_nested_schema_version_fails_closed():
    invalid = _plan()
    invalid["assignments"] = ["not-an-assignment"]
    with pytest.raises(run_plans.RunPlanClientError) as raised:
        run_plans._validate_plan(invalid)
    assert raised.value.code == "plan_response_invalid"

    unsupported_plan = _plan()
    unsupported_plan["schema_version"] = 99
    with pytest.raises(run_plans.RunPlanClientError) as raised:
        run_plans._validate_plan(unsupported_plan)
    assert raised.value.code == "plan_response_invalid"

    response = _server_response(_plan())
    response["envelope"]["schema_version"] = 99
    with pytest.raises(run_plans.RunPlanClientError) as raised:
        run_plans._agent_response_from_server(response)
    assert raised.value.code == "schema_version_unsupported"
