"""A 股 AI 选股引擎公共入口：因子、筛选/组合/回测。"""

from __future__ import annotations

from .factors import (
    DEFAULT_WEIGHTS,
    FACTOR_DIRECTION,
    FACTOR_NAMES,
    MAX_WEIGHT,
    RISK_WEIGHTS,
    TRADING_DAYS,
    WEIGHT_PRESETS,
    compute_factor_scores,
    register_preset,
    validate_weights,
    weights_for,
)
from .selection import build_portfolio, run_backtest, screen

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_DIRECTION",
    "RISK_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "WEIGHT_PRESETS",
    "MAX_WEIGHT",
    "TRADING_DAYS",
    "weights_for",
    "validate_weights",
    "register_preset",
    "compute_factor_scores",
    "screen",
    "build_portfolio",
    "run_backtest",
]
