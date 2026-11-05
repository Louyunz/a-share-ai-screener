# -*- coding: utf-8 -*-
"""画像、选股与调优请求的数据结构与校验。

不依赖 pydantic：标准库 dataclass + 简单校验，方便 dev_server（无 fastapi）复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SLOT_KEYS = [
    "risk_level",
    "horizon",
    "drawdown_tolerance",
    "goal",
    "experience",
    "excluded_industries",
    "capital",
]
REQUIRED_SLOTS = ["risk_level", "horizon", "drawdown_tolerance"]

RISK_LEVELS = {
    "conservative": "保守",
    "balanced": "平衡",
    "aggressive": "积极",
}
_RISK_ALIASES = {
    "保守": "conservative",
    "保守型": "conservative",
    "低风险": "conservative",
    "平衡": "balanced",
    "稳健": "balanced",
    "稳健型": "balanced",
    "平衡型": "balanced",
    "中等风险": "balanced",
    "积极": "aggressive",
    "积极型": "aggressive",
    "激进": "aggressive",
    "激进型": "aggressive",
    "高风险": "aggressive",
}


class ValidationError(ValueError):
    """请求参数非法。"""


def normalize_slots(raw: dict[str, Any] | None) -> dict[str, str]:
    """把任意来源的槽位清洗成统一字符串字典，只保留已知字段。"""
    raw = raw or {}
    slots: dict[str, str] = {}
    for key in SLOT_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = "、".join(str(v) for v in value if str(v).strip())
        slots[key] = str(value).strip()
    risk = slots.get("risk_level", "")
    if risk:
        mapped = _RISK_ALIASES.get(risk) or _RISK_ALIASES.get(risk.lower()) or risk.lower()
        slots["risk_level"] = mapped
    return slots


def risk_level(slots: dict[str, str]) -> str:
    level = slots.get("risk_level", "balanced")
    return level if level in RISK_LEVELS else "balanced"


def profile_complete(slots: dict[str, str]) -> bool:
    return all(bool(slots.get(k)) for k in REQUIRED_SLOTS)


def profile_payload(slots: dict[str, str], confirmed: bool = False) -> dict[str, Any]:
    slots = normalize_slots(slots)
    return {
        "slots": slots,
        "risk_level": risk_level(slots),
        "confirmed": bool(confirmed),
        "complete": profile_complete(slots),
    }


def excluded_industries(slots: dict[str, str]) -> list[str]:
    raw = slots.get("excluded_industries", "")
    for sep in ("、", "，", ",", ";", "；", "|"):
        raw = raw.replace(sep, "|")
    return [item.strip() for item in raw.split("|") if item.strip()]


@dataclass
class SelectionRequest:
    top_n: int = 10

    def validate(self) -> "SelectionRequest":
        if not isinstance(self.top_n, int) or not 1 <= self.top_n <= 50:
            raise ValidationError("top_n 必须是 1-50 的整数")
        return self


@dataclass
class TuneRequest:
    trials: int = 24

    def validate(self) -> "TuneRequest":
        if not isinstance(self.trials, int) or not 3 <= self.trials <= 120:
            raise ValidationError("trials 必须是 3-120 的整数")
        return self


@dataclass
class MessageRequest:
    content: str = ""

    def validate(self) -> "MessageRequest":
        content = (self.content or "").strip()
        if not content:
            raise ValidationError("消息内容不能为空")
        if len(content) > 4000:
            raise ValidationError("消息过长（>4000 字符）")
        self.content = content
        return self


@dataclass
class VersionActionRequest:
    reason: str = "user_action"
