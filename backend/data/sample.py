# -*- coding: utf-8 -*-
"""合成 A 股示例数据：用于离线演示与自动化测试。

为了让演示中的"因子选股"有可观察的效果，合成数据做了两点设计：
1. 价值/质量/成长/小市值/低波因子与个股长期漂移正相关；
2. 个股收益带有弱动量延续（自相关），使 60 日动量具备预测性。
注意：合成数据仅用于跑通流程与演示界面，不具备任何投资参考价值。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import DataBundle

INDUSTRIES = [
    "食品饮料", "医药生物", "电子", "计算机", "电力设备",
    "银行", "非银金融", "机械设备", "基础化工", "汽车",
]


def _codes(n_stocks: int) -> list[str]:
    codes: list[str] = []
    for i in range(n_stocks):
        if i % 2 == 0:
            codes.append(f"{1 + i // 2:06d}.SZ")
        else:
            codes.append(f"{600000 + i // 2:06d}.SH")
    return codes


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def _fundamentals(rng: np.random.Generator, codes: list[str]) -> pd.DataFrame:
    n = len(codes)
    fundamentals = pd.DataFrame(index=codes)
    fundamentals["industry"] = [INDUSTRIES[i % len(INDUSTRIES)] for i in range(n)]
    fundamentals["ep"] = np.clip(rng.normal(0.05, 0.03, size=n), 0.005, 0.25)
    fundamentals["bp"] = np.clip(rng.normal(0.55, 0.30, size=n), 0.05, 3.0)
    fundamentals["pe"] = 1.0 / fundamentals["ep"]
    fundamentals["pb"] = 1.0 / fundamentals["bp"]
    fundamentals["roe"] = np.clip(rng.normal(0.12, 0.07, size=n), -0.05, 0.45)
    fundamentals["ocf_to_profit"] = np.clip(rng.normal(1.0, 0.4, size=n), 0.1, 3.0)
    fundamentals["revenue_growth"] = np.clip(rng.normal(0.12, 0.22, size=n), -0.4, 0.9)
    fundamentals["total_mv"] = np.exp(rng.normal(np.log(3e10), 1.1, size=n))
    fundamentals["float_mv"] = fundamentals["total_mv"] * rng.uniform(0.5, 0.95, size=n)
    fundamentals["report_date"] = "2026-06-30"
    fundamentals["announce_date"] = "2026-08-30"
    return fundamentals[[
        "industry", "ep", "bp", "pe", "pb", "roe", "ocf_to_profit",
        "revenue_growth", "total_mv", "float_mv", "report_date", "announce_date"]]


def generate_bundle(
    n_stocks: int = 150,
    n_days: int = 750,
    seed: int = 42,
    start: str = "2022-01-04",
) -> DataBundle:
    """生成确定性的合成 A 股数据包（默认 150 只 × 750 交易日）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    codes = _codes(n_stocks)
    fundamentals = _fundamentals(rng, codes)

    # 因子隐含的长期漂移（让演示中的因子选股有可观察的超额收益）
    value_z = _zscore(_zscore(fundamentals["ep"].to_numpy()) +
                      _zscore(fundamentals["bp"].to_numpy()))
    quality_z = _zscore(_zscore(fundamentals["roe"].to_numpy()) +
                        _zscore(fundamentals["ocf_to_profit"].to_numpy()))
    growth_z = _zscore(fundamentals["revenue_growth"].to_numpy())
    size_z = _zscore(np.log(fundamentals["float_mv"].to_numpy()))
    drift = (0.00045 * value_z + 0.00035 * quality_z + 0.00025 * growth_z
             - 0.00020 * size_z)

    # 波动率与低波偏好
    idio_vol = rng.uniform(0.010, 0.026, size=n_stocks)
    lowvol_z = _zscore(-idio_vol)
    drift = drift + 0.00020 * lowvol_z

    betas = rng.uniform(0.6, 1.4, size=n_stocks)
    market_ret = rng.normal(0.00035, 0.009, size=n_days)

    # 逐日生成带弱动量延续的收益
    rets = np.zeros((n_days, n_stocks))
    prev_idio = np.zeros(n_stocks)
    for t in range(n_days):
        idio = rng.normal(0, 1, size=n_stocks) * idio_vol + drift + 0.10 * prev_idio
        rets[t] = betas * market_ret[t] + idio
        prev_idio = idio
    rets = np.clip(rets, -0.098, 0.098)
    # 少量涨跌停样本，用于回测规则验证
    limit_mask = rng.random((n_days, n_stocks)) < 0.004
    limit_sign = np.where(rng.random((n_days, n_stocks)) > 0.5, 1.0, -1.0)
    rets = np.where(limit_mask, limit_sign * 0.0995, rets)

    # 停牌：当日成交额为 0、价格保持上一日
    suspend = rng.random((n_days, n_stocks)) < 0.004
    suspend[0, :] = False

    base_price = np.exp(rng.uniform(np.log(4), np.log(60), size=n_stocks))
    prices = np.empty_like(rets)
    prices[0, :] = base_price
    for t in range(1, n_days):
        prev = prices[t - 1, :]
        moved = prev * (1.0 + rets[t, :])
        prices[t, :] = np.where(suspend[t, :], prev, moved)
    prices = np.round(np.clip(prices, 0.5, None), 2)

    amount_base = np.exp(rng.normal(np.log(6e7), 0.8, size=n_stocks))
    amount_noise = np.exp(rng.normal(0, 0.35, size=(n_days, n_stocks)))
    amounts = np.where(suspend, 0.0, amount_base[None, :] * amount_noise).round(0)

    prices_df = pd.DataFrame(prices, index=dates, columns=codes)
    close_raw_df = prices_df.copy()  # 示例数据不复权
    amount_df = pd.DataFrame(amounts, index=dates, columns=codes)
    benchmark = pd.Series(
        3000.0 * np.cumprod(1.0 + market_ret), index=dates, name="benchmark")

    meta: dict[str, Any] = {
        "source": "sample",
        "as_of": dates[-1].strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_stocks": n_stocks,
        "n_days": n_days,
        "note": "合成演示数据，仅用于流程演示与测试，不具投资参考价值",
    }
    return DataBundle(
        prices=prices_df,
        close_raw=close_raw_df,
        amount=amount_df,
        fundamentals=fundamentals,
        industry=fundamentals[["industry"]].copy(),
        benchmark=benchmark,
        universe=list(codes),
        meta=meta,
    )
