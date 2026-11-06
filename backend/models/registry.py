# -*- coding: utf-8 -*-
"""内置选股模型类型注册表。

一个「模型类型」就是一组 10 因子权重。因子的方向对齐已在
``backend.engine.factors`` 中完成，因此权重越大的因子代表模型越看重它。

本模块只提供纯数据与纯校验，不跑回测、不依赖 LLM。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

from ..engine.factors import FACTOR_NAMES, MAX_WEIGHT, WEIGHT_TOL

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_LABELS",
    "MAX_WEIGHT",
    "DEFAULT_MODEL_ID",
    "ModelType",
    "MODEL_TYPES",
    "validate_weights",
    "list_models",
    "get_model",
]

# 因子的中文解释（给普通用户看的说法，也用于 LLM 提示词）
FACTOR_LABELS: dict[str, str] = {
    "ep": "估值便宜",
    "bp": "账面价值便宜",
    "roe": "赚钱能力",
    "ocf_to_profit": "利润的现金含量",
    "revenue_growth": "收入成长速度",
    "momentum_60": "中期上涨趋势",
    "reversal_5": "短期超跌反转",
    "volatility_60": "价格波动稳定性",
    "liquidity_20": "成交活跃度",
    "size": "市值规模",
}

# 默认模型（无明确风格时的均衡起点）
DEFAULT_MODEL_ID = "multi_factor"


class ModelTypeError(ValueError):
    """模型类型不合法（未知 id、权重越界等）。"""


@dataclass(frozen=True)
class ModelType:
    """一个内置模型类型（不可变）。"""

    id: str
    name: str
    description: str
    weights: Mapping[str, float] = field(repr=False)

    def to_dict(self, include_weights: bool = False) -> dict[str, Any]:
        """转成普通 dict；``include_weights=True`` 时附带权重副本。"""
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }
        if include_weights:
            payload["weights"] = dict(self.weights)
        return payload


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """校验并归一化 10 因子权重，返回新 dict。

    规则：
    - 未出现的因子按 0 补齐，只允许 ``FACTOR_NAMES`` 中的因子名；
    - 每项必须是有限非负数值，且不超过单因子上限 ``MAX_WEIGHT``（0.4）；
    - 归一化后总和为 1，且每项仍不超过 0.4；
    - 任何不合法情况抛 ``ValueError``，错误信息说明具体原因。
    """
    if not isinstance(weights, Mapping):
        raise ModelTypeError("weights 必须是 dict（因子名 -> 权重）")

    unknown = sorted(str(key) for key in weights if key not in FACTOR_NAMES)
    if unknown:
        raise ModelTypeError(
            f"weights 含未知因子: {', '.join(unknown)}；只允许: {', '.join(FACTOR_NAMES)}"
        )

    values: dict[str, float] = {}
    for name in FACTOR_NAMES:
        if name not in weights:
            values[name] = 0.0
            continue
        raw = weights[name]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ModelTypeError(f"因子 {name} 的权重不是数值: {raw!r}")
        value = float(raw)
        if not math.isfinite(value):
            raise ModelTypeError(f"因子 {name} 的权重不是有限数值: {raw!r}")
        if value < 0:
            raise ModelTypeError(f"因子 {name} 的权重不能为负: {value}")
        if value > MAX_WEIGHT + WEIGHT_TOL:
            raise ModelTypeError(
                f"因子 {name} 的权重 {value} 超过单因子上限 {MAX_WEIGHT}"
            )
        values[name] = value

    total = math.fsum(values.values())
    if total <= WEIGHT_TOL:
        raise ModelTypeError("因子权重之和必须为正数（当前所有因子权重都是 0）")

    normalized = {name: values[name] / total for name in FACTOR_NAMES}
    for name, value in normalized.items():
        if value > MAX_WEIGHT + WEIGHT_TOL:
            raise ModelTypeError(
                f"归一化后因子 {name} 的权重为 {value:.4f}，超过单因子上限 {MAX_WEIGHT}；"
                "请把权重分散到更多因子"
            )
    return normalized


# ---------------------------------------------------------------------------
# 内置模型定义（weights 必须 10 项、和为 1、每项 ∈ [0, 0.4]）
# ---------------------------------------------------------------------------

_BUILTIN_DEFINITIONS: dict[str, dict[str, Any]] = {
    "multi_factor": {
        "name": "均衡多因子",
        "description": "把价值、盈利、成长、趋势和波动这些信号都照顾到，不押注单一风格，"
        "适合大多数人的稳健起点。",
        "weights": {
            "ep": 0.12,
            "bp": 0.10,
            "roe": 0.12,
            "ocf_to_profit": 0.10,
            "revenue_growth": 0.10,
            "momentum_60": 0.10,
            "reversal_5": 0.08,
            "volatility_60": 0.10,
            "liquidity_20": 0.10,
            "size": 0.08,
        },
    },
    "low_vol_quality": {
        "name": "低波质量",
        "description": "优先挑价格波动小、赚钱能力强的公司，追求拿得住、回撤小的稳健体验。",
        "weights": {
            "ep": 0.06,
            "bp": 0.06,
            "roe": 0.20,
            "ocf_to_profit": 0.18,
            "revenue_growth": 0.08,
            "momentum_60": 0.06,
            "reversal_5": 0.06,
            "volatility_60": 0.20,
            "liquidity_20": 0.06,
            "size": 0.04,
        },
    },
    "momentum_growth": {
        "name": "趋势成长",
        "description": "顺着已经走强的趋势，挑收入和业绩增长快的公司，追求更高的上涨弹性。",
        "weights": {
            "ep": 0.04,
            "bp": 0.03,
            "roe": 0.12,
            "ocf_to_profit": 0.06,
            "revenue_growth": 0.22,
            "momentum_60": 0.24,
            "reversal_5": 0.08,
            "volatility_60": 0.07,
            "liquidity_20": 0.06,
            "size": 0.08,
        },
    },
    "value_dividend": {
        "name": "价值红利",
        "description": "重点看估值便宜、现金流扎实的公司，偏向能持续分红、价格相对便宜的稳健标的。",
        "weights": {
            "ep": 0.24,
            "bp": 0.22,
            "roe": 0.12,
            "ocf_to_profit": 0.12,
            "revenue_growth": 0.05,
            "momentum_60": 0.05,
            "reversal_5": 0.06,
            "volatility_60": 0.08,
            "liquidity_20": 0.04,
            "size": 0.02,
        },
    },
    "contrarian": {
        "name": "反转",
        "description": "专门找短期跌过头、但基本面还过得去的公司，博取跌深之后的修复机会。",
        "weights": {
            "ep": 0.12,
            "bp": 0.12,
            "roe": 0.08,
            "ocf_to_profit": 0.06,
            "revenue_growth": 0.04,
            "momentum_60": 0.02,
            "reversal_5": 0.30,
            "volatility_60": 0.14,
            "liquidity_20": 0.06,
            "size": 0.06,
        },
    },
}


def _build_registry() -> dict[str, ModelType]:
    """校验内置定义并冻结成 ``MODEL_TYPES``。"""
    registry: dict[str, ModelType] = {}
    for model_id, definition in _BUILTIN_DEFINITIONS.items():
        name = str(definition["name"]).strip()
        description = str(definition["description"]).strip()
        if not name or not description:
            raise ModelTypeError(f"模型 {model_id} 的 name/description 不能为空")
        weights = validate_weights(definition["weights"])
        registry[model_id] = ModelType(
            id=model_id,
            name=name,
            description=description,
            weights=MappingProxyType(weights),
        )
    return registry


MODEL_TYPES: dict[str, ModelType] = _build_registry()


def list_models() -> list[dict[str, Any]]:
    """返回内置模型清单（``id/name/description``，按注册顺序）。

    需要权重时用 ``get_model(model_id)``。
    """
    return [model.to_dict() for model in MODEL_TYPES.values()]


def get_model(model_id: str) -> dict[str, Any]:
    """按 id 取模型详情，``weights`` 是可直接修改的副本。

    未知 id 抛 ``ValueError``。
    """
    key = str(model_id).strip() if model_id is not None else ""
    model = MODEL_TYPES.get(key)
    if model is None:
        raise ModelTypeError(
            f"未知模型类型 {model_id!r}；可选: {', '.join(MODEL_TYPES)}"
        )
    return model.to_dict(include_weights=True)
