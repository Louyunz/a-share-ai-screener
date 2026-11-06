# -*- coding: utf-8 -*-
"""动作参数 → 执行指令：确定性策略层（纯函数，可单测）。

这里只做“AI 已经给出意图之后的确定性规则”，不做任何自然语言理解：

- 客户明确要 code / 点名内置模型 / 指定目标：原样采纳；
- 客户点击「继续优化」：按上一版结构做结构性升级（业务策略）；
- 回撤上限：来自画像里的结构化槽位值，不再从用户原话正则推断。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "EngineDirective",
    "resolve_directive",
    "drawdown_tolerance_ratio",
    "build_expectation",
]


@dataclass(frozen=True)
class EngineDirective:
    engine: str          # auto | weights | model_type | code
    model_type: str      # 仅 model_type 时有意义
    objective: str       # sharpe | cagr | mdd
    escalate: bool
    reason: str


def resolve_directive(args: dict[str, Any], *, previous_approach: str = "") -> EngineDirective:
    """把已校验的 analyze_model 参数解析成执行指令。

    ``previous_approach`` 来自上一版 pending 模型（weights/model_type/code），
    仅用于 ``escalate=true`` 时选择结构性升级方向，不读取用户原话。
    """
    engine = str(args.get("engine") or "auto").strip().lower()
    model_type = str(args.get("model_type") or "").strip()
    objective = str(args.get("objective") or "").strip().lower() or "sharpe"
    escalate = bool(args.get("escalate"))
    reason = str(args.get("reason") or "").strip()[:120]

    if engine == "code":
        return EngineDirective("code", "", objective, escalate, reason)
    if engine == "model_type":
        return EngineDirective("model_type", model_type, objective, escalate, reason)
    if engine == "weights":
        return EngineDirective("weights", "", objective, escalate, reason)

    # engine=auto：由服务端的确定性策略决定
    if escalate:
        if previous_approach == "weights":
            return EngineDirective("model_type", "", objective, True, reason)
        return EngineDirective("code", "", objective, True, reason)
    return EngineDirective("auto", "", objective, False, reason)


_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")


def drawdown_tolerance_ratio(slots: dict[str, str] | None) -> float | None:
    """从画像槽位 ``drawdown_tolerance`` 解析数值（如 "15%" → 0.15）。

    这是**数据规范化**，不是意图识别：只读取已由 AI 抽取进槽位的结构化值。
    """
    raw = str((slots or {}).get("drawdown_tolerance") or "").strip()
    if not raw:
        return None
    match = _PERCENT_RE.search(raw)
    if match:
        try:
            value = abs(float(match.group(1))) / 100.0
        except ValueError:
            return None
        return value if value > 0 else None
    try:
        value = abs(float(raw))
    except ValueError:
        return None
    if value > 1:
        value = value / 100.0
    return value if value > 0 else None


def build_expectation(directive: EngineDirective,
                      slots: dict[str, str] | None) -> dict[str, Any]:
    """回测候选的核对标准：目标指标来自 AI 参数，回撤上限来自画像。"""
    return {
        "objective": directive.objective,
        "tolerance": drawdown_tolerance_ratio(slots),
    }
