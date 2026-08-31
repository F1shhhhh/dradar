"""Versioned command semantics for Agents operating DRadar for a user.

The ordinary ``--help`` output stays short.  This module is the precise,
machine-readable contract an Agent uses to decide whether a command matches
the user's intent without parsing terminal prose.
"""

from __future__ import annotations

import json


SCHEMA_VERSION = 1


def _argument(
    name: str,
    *,
    user_intent: str,
    allowed_when: str,
    default: object,
    state_change: str,
    decision_required: bool,
    conflicts_with: list[str] | None = None,
    idempotency: str,
    failure_codes: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "user_intent": user_intent,
        "allowed_when": allowed_when,
        "default": default,
        "state_change": state_change,
        "decision_required": decision_required,
        "conflicts_with": conflicts_with or [],
        "idempotency": idempotency,
        "failure_codes": failure_codes or [],
        "user_message": "以命令结果中的版本化 user_message 为准，不解析终端自由文本",
    }


COMMAND_SCHEMAS = {
    "run": {
        "summary": "在当前设备执行网页已经确定的这次领取",
        "environment_contract": {
            "wire_harness_aliases": {"dsh": "dsh-minimal"},
            "scope": "only_the_current_plan_tool",
            "recovery_commands": "agent.next_commands",
        },
        "interaction_rules": {
            "first_device": "notify_and_start",
            "same_device": "notify_and_resume_idempotently",
            "other_healthy_device": "confirm_before_join",
            "other_stale_device": "confirm_before_continue",
            "different_plan": "notify_and_start_independently",
            "auto_capacity_reduction": "warn_and_start_with_safe_count",
            "fixed_capacity_shortfall": "confirm_before_server_start",
            "capacity_changed_during_start_auto": "retry_lower_then_warn",
            "capacity_changed_during_start_fixed": "confirm_lower_or_cancel",
            "missing_current_tool": "notify_before_server_start",
        },
        "arguments": [
            _argument(
                "--plan",
                user_intent="运行网页交给 Agent 的这次领取",
                allowed_when="运行码仍可交换，或本机已保存该计划的短期权限",
                default=None,
                state_change="解析精确题目边界；在允许时登记当前设备",
                decision_required=False,
                idempotency="同一设备重复运行不会创建重复设备或重复题目",
                failure_codes=[
                    "run_code_invalid", "plan_expired", "plan_access_denied",
                ],
            ),
            _argument(
                "--concurrency",
                user_intent="指定这台设备同时处理的题目数量，或让系统自动安排",
                allowed_when="不超过运行计划和服务端允许的范围",
                default="plan",
                state_change="改变当前设备的本地并发目标，不扩大题目范围",
                decision_required=False,
                idempotency="相同值重复提交不改变运行边界",
                failure_codes=[
                    "concurrency_not_allowed", "local_capacity_unavailable",
                    "decision_invalid_or_capacity_changed",
                    "concurrency_capacity_reserved",
                    "local_concurrency_change_requires_restart",
                ],
            ),
            _argument(
                "--server",
                user_intent="使用网页所属站点的 API 交换并执行这次领取",
                allowed_when="HTTPS；本地开发仅允许 localhost/loopback HTTP",
                default="saved_plan_then_config_then_public_default",
                state_change="首次交换后把站点地址绑定到本机计划状态",
                decision_required=False,
                idempotency="同一计划只能继续使用首次绑定的站点",
                failure_codes=["server_url_invalid", "server_scope_mismatch"],
            ),
            _argument(
                "--decision-token",
                user_intent="执行用户刚刚明确同意的跨设备动作或本机固定数量选择",
                allowed_when="前一次响应 decision_required=true，且用户已选择对应选项",
                default=None,
                state_change="消费一次性凭证并允许当前设备加入或继续",
                decision_required=True,
                conflicts_with=["未获得用户同意"],
                idempotency="只能成功消费一次；状态变化后失败关闭并重新询问",
                failure_codes=[
                    "decision_context_missing", "decision_invalid_or_state_changed",
                    "decision_invalid_or_capacity_changed",
                ],
            ),
            _argument(
                "--json",
                user_intent="让 Agent 读取稳定结构，而不是解析自然语言输出",
                allowed_when="任何状态",
                default=False,
                state_change="none",
                decision_required=False,
                idempotency="read_format_only",
            ),
        ],
    },
    "progress": {
        "summary": "读取这次领取在所有设备上的进度",
        "arguments": [
            _argument(
                "--plan",
                user_intent="查看指定这次领取的进度",
                allowed_when="本机可以交换或已有该计划的短期权限",
                default=None,
                state_change="none",
                decision_required=False,
                idempotency="read_only",
                failure_codes=["run_code_invalid", "plan_expired", "plan_access_denied"],
            ),
            _argument(
                "--server",
                user_intent="从这次领取所属的站点读取进度",
                allowed_when="与本机已保存计划绑定的站点一致",
                default="saved_plan_then_config_then_public_default",
                state_change="none",
                decision_required=False,
                idempotency="read_only",
                failure_codes=["server_url_invalid", "server_scope_mismatch"],
            ),
            _argument(
                "--json",
                user_intent="读取稳定的进度结构",
                allowed_when="任何状态",
                default=False,
                state_change="none",
                decision_required=False,
                idempotency="read_format_only",
            ),
        ],
    },
    "stop": {
        "summary": "停止当前设备，或在用户明确同意后停止所有设备",
        "arguments": [
            _argument(
                "--plan",
                user_intent="停止指定这次领取的运行",
                allowed_when="本机可以交换或已有该计划的短期权限",
                default=None,
                state_change="按 scope 请求停止",
                decision_required=False,
                idempotency="重复停止返回当前状态",
                failure_codes=["run_code_invalid", "plan_access_denied"],
            ),
            _argument(
                "--server",
                user_intent="在这次领取所属的站点请求停止",
                allowed_when="与本机已保存计划绑定的站点一致",
                default="saved_plan_then_config_then_public_default",
                state_change="none",
                decision_required=False,
                idempotency="same_server_only",
                failure_codes=["server_url_invalid", "server_scope_mismatch"],
            ),
            _argument(
                "--scope",
                user_intent="选择只停止当前设备或停止所有设备",
                allowed_when="this-device 始终可用；all-devices 需要用户明确同意",
                default=None,
                state_change="停止一个设备或整个运行计划；不直接释放已领取题目",
                decision_required=True,
                idempotency="相同范围重复停止不扩大影响",
                failure_codes=["decision_required", "decision_invalid_or_state_changed"],
            ),
            _argument(
                "--decision-token",
                user_intent="执行用户已同意的停止所有设备动作",
                allowed_when="服务端刚返回对应的一次性确认凭证",
                default=None,
                state_change="消费凭证并停止所有设备后续运行",
                decision_required=True,
                idempotency="single_use",
                failure_codes=["decision_context_missing", "decision_invalid_or_state_changed"],
            ),
            _argument(
                "--json",
                user_intent="读取稳定的停止结果结构",
                allowed_when="任何状态",
                default=False,
                state_change="none",
                decision_required=False,
                idempotency="read_format_only",
            ),
        ],
    },
}


def command_schema(command: str) -> dict:
    body = COMMAND_SCHEMAS[command]
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "result_contract": {
            "stable_fields": [
                "schema_version", "status", "interaction",
                "decision_required", "user_message", "agent_action",
                "error_code", "retryable", "choices", "decision_token",
                "poll_after_seconds", "user_message_policy",
            ],
            "unknown_fields": "ignore",
            "unknown_schema_version": "fail_closed",
            "agent_details": "精确计划和服务端状态位于可选 agent 对象；决策只读取顶层字段",
            "choice_actions": (
                "decision_required=true 时，agent.choice_actions 按 choice id 给出"
                "机器可执行映射；replay_current_command_with_args 的 args 仅追加/替换"
                "当前命令参数，no_command 表示不执行命令"
            ),
            "environment_recovery": (
                "环境错误可在 agent.next_commands 给出非秘密 argv 数组；"
                "requires_user_action=true 时只能提示用户完成交互"
            ),
        },
        **body,
    }


def cmd_schema(args) -> int:
    payload = command_schema(args.schema_command)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    print(f"dradar {args.schema_command}: {payload['summary']}")
    print("Agent 请使用 --json 读取完整、版本化的参数语义。")
    return 0


__all__ = ["SCHEMA_VERSION", "command_schema", "cmd_schema"]
