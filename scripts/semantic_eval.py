# -*- coding: utf-8 -*-
"""真实 LLM 语义矩阵验收：断言 Planner 的 tool_calls 参数，而不是回复文字。

用法（需要真实 Key）：
    $env:DEEPSEEK_API_KEY = "sk-..."
    .venv/Scripts/python.exe scripts/semantic_eval.py

这些用例直接验证“AI 是否通过原生 function calling 正确表达意图”；
任何失败都必须通过改提示词/示例修复，禁止在服务端加关键词路由。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.agent import Agent  # noqa: E402
from backend.config import LLM_BASE_URL, LLM_MODEL  # noqa: E402
from backend.llm import OpenAICompatClient  # noqa: E402
from backend.schemas import profile_payload  # noqa: E402


def _state(*, confirmed: bool = True, pending: dict[str, Any] | None = None,
           pending_candidates: list[dict[str, Any]] | None = None,
           versions: list[dict[str, Any]] | None = None,
           last_selection: dict[str, Any] | None = None) -> dict[str, Any]:
    slots = {
        "goal": "子女教育",
        "horizon": "5 年",
        "drawdown_tolerance": "10%",
        "loss_reaction": "会有点慌，但先不卖",
        "risk_level": "balanced",
    }
    return {
        "market": {"as_of": "2026-11-05", "universe": 29},
        "profile": profile_payload(slots, confirmed),
        "model": {"version_id": None, "name": "均衡多因子", "kind": "builtin",
                  "description": "", "metrics": {}},
        "model_scope": "session",
        "versions": versions or [],
        "pending": pending,
        "pending_candidates": pending_candidates or [],
        "pending_intent": None,
        "last_selection": last_selection,
    }


CASES: list[dict[str, Any]] = [
    {
        "name": "更换数学模型 → code",
        "message": "更换数学模型",
        "check": lambda action, reply: action is not None and action["args"]["engine"] == "code",
    },
    {
        "name": "构建新的投资数学模型 → code",
        "message": "帮我构建一个新的投资数学模型",
        "check": lambda action, reply: action is not None and action["args"]["engine"] == "code",
    },
    {
        "name": "写一个新的打分引擎 → code",
        "message": "帮我写一个新的打分引擎",
        "check": lambda action, reply: action is not None and action["args"]["engine"] == "code",
    },
    {
        "name": "换其他模型 → model_type 自动选择",
        "message": "换成其他模型",
        "check": lambda action, reply: (
            action is not None
            and action["args"]["engine"] == "model_type"
            and action["args"]["model_type"] == ""
            and not action["args"]["escalate"]
        ),
    },
    {
        "name": "换成低波质量 → model_type",
        "message": "换成低波质量模型",
        "check": lambda action, reply: (
            action is not None
            and action["args"]["engine"] == "model_type"
            and action["args"]["model_type"] == "low_vol_quality"
        ),
    },
    {
        "name": "用随机森林 → code",
        "message": "用随机森林模型帮我选股",
        "check": lambda action, reply: (
            action is not None and action["args"]["engine"] == "code"
        ),
        "state": _state(confirmed=True),
    },
    {
        "name": "继续优化 → escalate",
        "message": "继续优化模型，上一版未达到我的预期",
        "check": lambda action, reply: action is not None and bool(action["args"]["escalate"]),
        "state": _state(pending={"summary": "上一版方案"}),
    },
    {
        "name": "普通优化 → 直接调用 auto",
        "message": "帮我优化一下模型",
        "check": lambda action, reply: (
            action is not None and action["args"]["engine"] == "auto"
        ),
    },
    {
        "name": "回撤目标 → objective=mdd",
        "message": "优化模型，回撤再小一点",
        "check": lambda action, reply: (
            action is not None and action["args"]["objective"] == "mdd"
        ),
    },
    {
        "name": "概念问题 → 不触发模型动作",
        "message": "什么是回撤？",
        "check": lambda action, reply: action is None,
        "state": _state(confirmed=False),
    },
    {
        "name": "回滚 → rollback_version",
        "message": "回滚到上一个版本",
        "check": lambda action, reply: action is not None and action["name"] == "rollback_version",
        "state": _state(versions=[
            {"id": "v1", "name": "旧模型", "status": "archived"},
            {"id": "v2", "name": "当前模型", "status": "active"},
        ]),
    },
    {
        "name": "多个候选方案中选第一个 → candidate=1",
        "message": "我觉得还是第一个好，用第一个",
        "check": lambda action, reply: (
            action is not None
            and action["name"] == "apply_pending_change"
            and int(action["args"].get("candidate") or 0) == 1
        ),
        "state": _state(pending={"summary": "第二版方案"}, pending_candidates=[
            {"index": 1, "name": "低波质量（网格寻优）", "summary": "第一版",
             "status": "pending", "available": True, "metrics": {"sharpe": 0.69}},
            {"index": 2, "name": "AI 生成模型", "summary": "第二版",
             "status": "pending", "available": True, "metrics": {"sharpe": 0.73}},
        ]),
    },
    {
        "name": "同意启用（未指定编号）→ 默认最新候选",
        "message": "同意，启用这个改进",
        "check": lambda action, reply: (
            action is not None
            and action["name"] == "apply_pending_change"
            and int(action["args"].get("candidate") or 0) == 0
        ),
        "state": _state(pending={"summary": "第二版方案"}, pending_candidates=[
            {"index": 1, "name": "低波质量（网格寻优）", "summary": "第一版",
             "status": "pending", "available": True},
            {"index": 2, "name": "AI 生成模型", "summary": "第二版",
             "status": "pending", "available": True},
        ]),
    },
    {
        "name": "没有已启用版本时不得回滚",
        "message": "回滚到上一个版本",
        "check": lambda action, reply: (
            action is None or action["name"] != "rollback_version"
        ),
        "state": _state(pending_candidates=[
            {"index": 1, "name": "低波质量（网格寻优）", "summary": "第一版",
             "status": "pending", "available": True},
        ]),
    },
]


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY，无法执行真实语义验收")
        return 2
    agent = Agent(OpenAICompatClient(
        api_key=api_key, base_url=LLM_BASE_URL, model=LLM_MODEL, timeout=90))
    failures = 0
    for index, case in enumerate(CASES, start=1):
        state = case.get("state") or _state()
        history = [{"role": "assistant", "content": "在为您提供建议前，需要确认您的风险承受情况。"}]
        try:
            plan = agent.plan(state, history, case["message"])
        except Exception as exc:  # noqa: BLE001
            print(f"[{index:02d}] FAIL {case['name']}：调用失败 {type(exc).__name__}: {exc}")
            failures += 1
            continue
        action = plan.action("analyze_model") or (
            plan.actions[0] if plan.actions else None)
        try:
            ok = bool(case["check"](action, plan.reply))
        except Exception as exc:  # noqa: BLE001
            print(f"[{index:02d}] FAIL {case['name']}：断言异常 {exc}")
            failures += 1
            continue
        mark = "PASS" if ok else "FAIL"
        payload = json.dumps(action, ensure_ascii=False) if action else "无动作"
        print(f"[{index:02d}] {mark} {case['name']} | tool={payload} | reply={plan.reply[:60]}")
        if not ok:
            failures += 1
    total = len(CASES)
    print(f"\n语义矩阵：{total - failures}/{total} 通过")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
