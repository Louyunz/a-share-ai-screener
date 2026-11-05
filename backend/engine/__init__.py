"""A 股 AI 选股引擎公共入口。

对外暴露三大模块的公共函数与常量：
- 因子：``compute_factor_scores`` 与权重注册表；
- 选股/组合/回测：``screen``、``build_portfolio``、``run_backtest``；
- 调参：``run_tuning``。
"""

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
from .tuner import run_tuning

__all__ = [
    # factors
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
    # selection
    "screen",
    "build_portfolio",
    "run_backtest",
    # tuner
    "run_tuning",
]
