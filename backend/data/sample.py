# -*- coding: utf-8 -*-
"""合成 A 股示例数据：用于离线演示与自动化测试。

说明：示例数据仅用于跑通流程与演示界面，不具备任何投资参考价值。
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


def generate_bundle(
    n_stocks: int = 150,
    n_days: int = 750,
    seed: int = 42,
    start: str = "2022-01-04",
) -> DataBundle:
    """生成确定性的合成 A 股数据包（约 150 只 × 750 交易日）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    codes = _codes(n_stocks)

    # ---- 市场因子与个股收益 ----
    market_ret = rng.normal(0.0002, 0.011, size=n_days)
    betas = rng.uniform(0.6, 1.4, size=n_stocks)
    idio_vol = rng.uniform(0.012, 0.032, size=n_stocks)

    # 让部分股票带上"风格倾向"：价值/质量/动量型股票漂移更高，模拟因子有效性
    style_value = rng.normal(0, 1, size=n_stocks)
    style_quality = rng.normal(0, 1, size=n_stocks)
    style_momentum = rng.normal(0, 1, size=n_stocks)
    drift = (0.00008 * style_value + 0.00006 * style_quality +
             0.00005 * style_momentum) / 3.0

    rets = (betas[None, :] * market_ret[:, None] +
            idio_vol[None, :] * rng.normal(0, 1, size=(n_days, n_stocks)) +
            drift[None, :])

    # A 股涨跌停：收益截断在 ±9.8%；个别交易日制造涨跌停样本
    rets = np.clip(rets, -0.098, 0.098)
    limit_mask = rng.random((n_days, n_stocks)) < 0.004
    limit_sign = np.where(rng.random((n_days, n_stocks)) > 0.5, 1.0, -1.0)
    rets = np.where(limit_mask, limit_sign * 0.0995, rets)

    # ---- 停牌：当日成交额为 0、价格保持上一日 ----
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
    noise = np.exp(rng.normal(0, 0.35, size=(n_days, n_stocks)))
    amounts = amount_base[None, :] * noise
    amounts = np.where(suspend, 0.0, amounts).round(0)

    code_labels = codes
    prices_df = pd.DataFrame(prices, index=dates, columns=code_labels)
    close_raw_df = prices_df.copy()  # 示例数据不复权
    amount_df = pd.DataFrame(amounts, index=dates, columns=code_labels)

    # ---- 财务指标 ----
    fundamentals = pd.DataFrame(index=code_labels)
    fundamentals["industry"] = [INDUSTRIES[i % len(INDUSTRIES)] for i in range(n_stocks)]
    fundamentals["ep"] = np.clip(rng.normal(0.05, 0.03, size=n_stocks), 0.005, 0.25)
    fundamentals["bp"] = np.clip(rng.normal(0.55, 0.30, size=n_stocks), 0.05, 3.0)
    fundamentals["pe"] = 1.0 / fundamentals["ep"]
    fundamentals["pb"] = 1.0 / fundamentals["bp"]
    fundamentals["roe"] = np.clip(rng.normal(0.12, 0.07, size=n_stocks), -0.05, 0.45)
    fundamentals["ocf_to_profit"] = np.clip(rng.normal(1.0, 0.4, size=n_stocks), 0.1, 3.0)
    fundamentals["revenue_growth"] = np.clip(rng.normal(0.12, 0.22, size=n_stocks), -0.4, 0.9)
    fundamentals["total_mv"] = np.exp(rng.normal(np.log(3e10), 1.1, size=n_stocks))
    fundamentals["float_mv"] = fundamentals["total_mv"] * rng.uniform(0.5, 0.95, size=n_stocks)
    fundamentals["report_date"] = "2026-06-30"
    fundamentals["announce_date"] = "2026-08-30"
    fundamentals = fundamentals[["industry", "ep", "bp", "pe", "pb", "roe",
                                 "ocf_to_profit", "revenue_growth",
                                 "total_mv", "float_mv", "report_date", "announce_date"]]

    industry_df = fundamentals[["industry"]].copy()

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
        industry=industry_df,
        benchmark=benchmark,
        universe=list(code_labels),
        meta=meta,
    )
