# -*- coding: utf-8 -*-
"""画像槽位、对话计划与展示文案（面向普通用户，不暴露专业术语）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 槽位顺序即画像卡展示顺序；“label”用生活化语言，避免风控术语吓到用户
SLOT_DEFS: list[dict[str, Any]] = [
    {"key": "goal", "label": "投资目标", "required": True, "hint": "这笔钱主要想用来做什么"},
    {"key": "horizon", "label": "投资期限", "required": True, "hint": "大概多久不用这笔钱"},
    {"key": "drawdown_tolerance", "label": "能接受的最大亏损", "required": True, "hint": "最多能接受亏多少"},
    {"key": "loss_reaction", "label": "亏钱时的反应", "required": True, "hint": "真亏了会怎么做"},
    {"key": "capital", "label": "大致投入", "required": False, "hint": "计划投入的资金范围"},
    {"key": "liquidity_need", "label": "用钱计划", "required": False, "hint": "近期是否有大额支出"},
    {"key": "income_stability", "label": "收入稳定性", "required": False, "hint": "收入是否稳定"},
    {"key": "experience", "label": "投资经验", "required": False, "hint": "以前买过基金/股票吗"},
    {"key": "preferred_industries", "label": "偏好行业", "required": False, "hint": "更看好哪些行业"},
    {"key": "excluded_industries", "label": "不碰行业", "required": False, "hint": "哪些行业坚决不买"},
    {"key": "trade_frequency", "label": "调仓频率", "required": False, "hint": "希望多久调整一次"},
    {"key": "concerns", "label": "最担心的事", "required": False, "hint": "最担心什么风险"},
]

SLOT_KEYS = [item["key"] for item in SLOT_DEFS]
REQUIRED_SLOTS = [item["key"] for item in SLOT_DEFS if item["required"]]
SLOT_LABELS = {item["key"]: item["label"] for item in SLOT_DEFS}
SLOT_HINTS = {item["key"]: item["hint"] for item in SLOT_DEFS}

# 外部输入里常见的同义键
_ALIASES = {
    "risk_level": "risk_level",
    "risk": "risk_level",
    "drawdown": "drawdown_tolerance",
    "tolerance": "drawdown_tolerance",
    "loss_feeling": "loss_reaction",
    "loss_reaction": "loss_reaction",
    "industry_preference": "preferred_industries",
    "preferred_industry": "preferred_industries",
    "excluded_industry": "excluded_industries",
    "exclude_industries": "excluded_industries",
    "liquidity": "liquidity_need",
    "money_need": "liquidity_need",
    "income": "income_stability",
    "frequency": "trade_frequency",
    "worry": "concerns",
}

FACTOR_PLAIN = {
    "ep": "价格相对盈利更便宜",
    "bp": "价格相对净资产更便宜",
    "roe": "公司赚钱能力强",
    "ocf_to_profit": "现金流扎实",
    "revenue_growth": "收入增长较快",
    "momentum_60": "近几个月走势偏强",
    "reversal_5": "短期跌得有点多",
    "volatility_60": "波动相对小",
    "liquidity_20": "买卖比较方便",
    "size": "公司规模偏小",
}

RISK_LEVEL_LABELS = {"conservative": "求稳型", "balanced": "稳中求进型", "aggressive": "进取型"}


class ValidationError(ValueError):
    """请求参数不合法。"""


def _risk_level(slots: dict[str, str]) -> str:
    text = " ".join(str(slots.get(key, "")) for key in
                    ("drawdown_tolerance", "loss_reaction", "concerns", "goal"))
    for level in ("conservative", "balanced", "aggressive"):
        if level in slots.get("risk_level", ""):
            return level
    if any(word in text for word in ("不能亏", "亏不起", "保守", "稳定", "怕跌", "保本")):
        return "conservative"
    if any(word in text for word in ("进取", "激进", "想赚", "高风险", "刺激", "翻倍")):
        return "aggressive"
    return "balanced"


def normalize_slots(raw: dict[str, Any] | None) -> dict[str, str]:
    """把外部输入收敛成 12 个标准槽位（值为字符串，空值删除）。"""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError("画像必须是对象")
    slots: dict[str, str] = {}
    for key, value in raw.items():
        mapped = _ALIASES.get(str(key), str(key))
        if mapped not in SLOT_KEYS:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            slots[mapped] = text[:200]
    if slots and "risk_level" not in slots:
        slots["risk_level"] = _risk_level(slots)
    return slots


def merge_slots(base: dict[str, str], delta: dict[str, Any] | None) -> dict[str, str]:
    merged = dict(normalize_slots(base))
    merged.update(normalize_slots(delta))
    if merged:
        merged["risk_level"] = _risk_level(merged)
    return merged


def profile_complete(slots: dict[str, str]) -> bool:
    return all(bool(slots.get(key)) for key in REQUIRED_SLOTS)


def missing_slots(slots: dict[str, str]) -> list[str]:
    return [key for key in REQUIRED_SLOTS if not slots.get(key)]


def _format_value(key: str, value: str) -> str:
    if key == "risk_level":
        return RISK_LEVEL_LABELS.get(value, value)
    return value


def profile_summary(slots: dict[str, str]) -> str:
    """用一句普通人能读懂的话概括画像（不引入新事实）。"""
    parts: list[str] = []
    risk = slots.get("risk_level")
    if risk:
        parts.append(RISK_LEVEL_LABELS.get(risk, risk))
    if slots.get("goal"):
        parts.append(f"目标是{slots['goal']}")
    if slots.get("horizon"):
        parts.append(f"大概 {slots['horizon']} 内用不到这笔钱")
    if slots.get("drawdown_tolerance"):
        parts.append(f"最多能接受亏 {slots['drawdown_tolerance']}")
    if slots.get("loss_reaction"):
        parts.append(f"真亏了会{slots['loss_reaction']}")
    if not parts:
        return "还在了解你的投资情况。"
    return "；".join(parts) + "。"


def profile_payload(slots: dict[str, str] | None, confirmed: bool = False) -> dict[str, Any]:
    normalized = normalize_slots(slots or {})
    items = []
    for item in SLOT_DEFS:
        raw = normalized.get(item["key"], "")
        items.append({
            "key": item["key"],
            "label": item["label"],
            "value": _format_value(item["key"], raw) if raw else item["hint"],
            "filled": bool(raw),
            "required": item["required"],
        })
    filled_required = sum(1 for key in REQUIRED_SLOTS if normalized.get(key))
    return {
        "slots": normalized,
        "confirmed": bool(confirmed),
        "complete": profile_complete(normalized),
        "missing": [SLOT_LABELS[key] for key in missing_slots(normalized)],
        "completeness": round(filled_required / len(REQUIRED_SLOTS), 2),
        "summary": profile_summary(normalized),
        "items": items,
    }


def top_reasons(contributions: dict[str, float], limit: int = 2) -> list[str]:
    """把因子贡献翻译成普通人能懂的选股理由。"""
    positives = [(name, value) for name, value in contributions.items() if value > 0]
    positives.sort(key=lambda pair: pair[1], reverse=True)
    return [FACTOR_PLAIN.get(name, name) for name, _ in positives[:limit]]


@dataclass
class Plan:
    reply: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    profile_delta: dict[str, Any] = field(default_factory=dict)

    def action_names(self) -> list[str]:
        return [str(action.get("name", "")) for action in self.actions]

    def has_action(self, name: str) -> bool:
        return name in self.action_names()

    def action(self, name: str) -> dict[str, Any] | None:
        for action in self.actions:
            if action.get("name") == name:
                return action
        return None
