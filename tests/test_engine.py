# -*- coding: utf-8 -*-
"""引擎单元测试：覆盖 docs/CONTRACT.md 第 2 节与 2.4 的验收项。

测试数据为固定 seed 的合成 A 股：60 只股票 × 400 个交易日，
包含少量停牌（价格 NaN / 成交额 0）与涨跌停样本；全程不联网、不依赖 fastapi。
"""

from __future__ import annotations

import math
import types
import unittest

import numpy as np
import pandas as pd

from backend.engine.factors import (
    DEFAULT_WEIGHTS,
    FACTOR_DIRECTION,
    FACTOR_NAMES,
    MAX_WEIGHT,
    RISK_WEIGHTS,
    WEIGHT_PRESETS,
    compute_factor_scores,
    register_preset,
    validate_weights,
    weights_for,
)
from backend.engine.selection import build_portfolio, run_backtest, screen

N_STOCKS = 60
N_DAYS = 400
SEED = 20240601


def _build_synthetic_data() -> dict:
    """构造 60 只股票 × 400 交易日的合成数据（含停牌与涨跌停样本）。"""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2023-01-02", periods=N_DAYS)
    codes = [f"{600000 + i:06d}" for i in range(N_STOCKS)]

    main_industries = ["银行", "医药生物", "电子", "计算机", "电力设备", "机械设备", "汽车"]
    # 最后 2 只归入“小样本行业”，用于验证行业样本 < 3 时回退全市场 z
    industry_labels = [main_industries[i % len(main_industries)] for i in range(N_STOCKS - 2)]
    industry_labels += ["小样本行业", "小样本行业"]

    industry_effect = rng.normal(0.0, 0.002, size=len(main_industries))
    effect_map = {name: industry_effect[i] for i, name in enumerate(main_industries)}
    effects = np.array([effect_map.get(name, 0.0) for name in industry_labels])

    market_return = rng.normal(0.0004, 0.008, size=(N_DAYS, 1))
    returns = (
        market_return
        + effects[None, :]
        + rng.normal(0.0, 0.014, size=(N_DAYS, N_STOCKS))
    )

    # 少量涨跌停样本（+9.9% / -10%），用于回测规则
    returns[150, 3] = 0.100
    returns[170, 8] = -0.100
    returns[250, 12] = 0.099
    returns[330, 5] = -0.099

    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=codes
    )
    close_raw = pd.DataFrame(
        10.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=codes
    )
    amount = pd.DataFrame(
        rng.lognormal(17.5, 0.35, size=(N_DAYS, N_STOCKS)), index=dates, columns=codes
    )

    # 停牌样本：价格 NaN + 成交额 0
    suspensions = {4: (60, 61, 62), 9: (200, 201)}
    for stock_idx, positions in suspensions.items():
        for pos in positions:
            prices.iat[pos, stock_idx] = np.nan
            close_raw.iat[pos, stock_idx] = np.nan
            amount.iat[pos, stock_idx] = 0.0

    ep = np.clip(rng.normal(0.05, 0.02, size=N_STOCKS), 0.005, 0.25)
    bp = np.clip(rng.normal(0.55, 0.25, size=N_STOCKS), 0.05, 2.0)
    total_mv = rng.lognormal(np.log(4.0e10), 0.9, size=N_STOCKS)
    fundamentals = pd.DataFrame(
        {
            "ep": ep,
            "bp": bp,
            "pe": 1.0 / ep,
            "pb": 1.0 / bp,
            "roe": np.clip(rng.normal(0.12, 0.05, size=N_STOCKS), -0.05, 0.4),
            "ocf_to_profit": np.clip(rng.normal(1.0, 0.3, size=N_STOCKS), 0.1, 3.0),
            "revenue_growth": rng.normal(0.10, 0.08, size=N_STOCKS),
            "total_mv": total_mv,
            "float_mv": total_mv * rng.uniform(0.5, 0.95, size=N_STOCKS),
        },
        index=codes,
    )
    industry = pd.DataFrame({"industry": industry_labels}, index=codes)
    benchmark = pd.Series(
        3000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, size=N_DAYS))),
        index=dates,
        name="benchmark",
    )

    return {
        "prices": prices,
        "close_raw": close_raw,
        "amount": amount,
        "fundamentals": fundamentals,
        "industry": industry,
        "benchmark": benchmark,
        "universe": list(codes),
        "meta": {"source": "synthetic", "as_of": dates[-1].strftime("%Y-%m-%d")},
    }


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _build_synthetic_data()
        cls.bundle = types.SimpleNamespace(**cls.data)
        cls.dates = cls.data["prices"].index
        cls.codes = list(cls.data["prices"].columns)

    # ------------------------------------------------------------------
    # 因子
    # ------------------------------------------------------------------
    def _scores(self, **overrides):
        data = dict(self.data)
        data.update(overrides)
        return compute_factor_scores(
            data["prices"],
            data["close_raw"],
            data["amount"],
            data["fundamentals"],
            data["industry"],
            DEFAULT_WEIGHTS,
        )

    def test_factors_columns_sorted_and_asof(self) -> None:
        scores = self._scores()
        self.assertIsInstance(scores, pd.DataFrame)
        self.assertEqual(len(scores), N_STOCKS)
        self.assertEqual(scores.index.name, "code")
        self.assertFalse(scores.index.duplicated().any())

        expected = {f"raw_{name}" for name in FACTOR_NAMES}
        expected |= {f"z_{name}" for name in FACTOR_NAMES}
        expected |= {"factor_score"}
        self.assertTrue(expected.issubset(set(scores.columns)))

        z_cols = [f"z_{name}" for name in FACTOR_NAMES]
        self.assertTrue(np.isfinite(scores[z_cols].to_numpy()).all())
        self.assertTrue(np.isfinite(scores["factor_score"].to_numpy()).all())
        diffs = np.diff(scores["factor_score"].to_numpy())
        self.assertTrue((diffs <= 1e-12).all(), "factor_score 必须按降序排列")

        # asof 切片：用较早日期计算仍可得到完整横截面
        scores_asof = compute_factor_scores(
            self.data["prices"],
            self.data["close_raw"],
            self.data["amount"],
            self.data["fundamentals"],
            self.data["industry"],
            DEFAULT_WEIGHTS,
            asof=self.dates[250],
        )
        self.assertEqual(len(scores_asof), N_STOCKS)
        self.assertTrue(np.isfinite(scores_asof["factor_score"].to_numpy()).all())

        # 数据为空 / 长度不足必须抛明确 ValueError
        with self.assertRaises(ValueError):
            compute_factor_scores(
                pd.DataFrame(), self.data["close_raw"], self.data["amount"],
                self.data["fundamentals"], self.data["industry"],
            )
        short = 110
        with self.assertRaises(ValueError):
            compute_factor_scores(
                self.data["prices"].iloc[:short],
                self.data["close_raw"].iloc[:short],
                self.data["amount"].iloc[:short],
                self.data["fundamentals"],
                self.data["industry"],
            )
        with self.assertRaises(ValueError):
            compute_factor_scores(
                self.data["prices"],
                self.data["close_raw"],
                self.data["amount"],
                self.data["fundamentals"],
                self.data["industry"],
                DEFAULT_WEIGHTS,
                min_history=1000,
            )

    def test_factor_missing_values_and_industry_fallback(self) -> None:
        # 缺失值：z 记 0，raw 保留 NaN，整体得分仍有限
        fundamentals = self.data["fundamentals"].copy()
        industry = self.data["industry"].copy()
        code_ep, code_size, code_ind = self.codes[0], self.codes[1], self.codes[2]
        fundamentals.loc[code_ep, "ep"] = np.nan
        fundamentals.loc[code_size, "float_mv"] = np.nan
        industry.loc[code_ind, "industry"] = None

        scores = self._scores(fundamentals=fundamentals, industry=industry)
        self.assertTrue(math.isnan(float(scores.loc[code_ep, "raw_ep"])))
        self.assertEqual(float(scores.loc[code_ep, "z_ep"]), 0.0)
        self.assertTrue(math.isnan(float(scores.loc[code_size, "raw_size"])))
        self.assertEqual(float(scores.loc[code_size, "z_size"]), 0.0)
        self.assertTrue(
            np.isfinite(scores[[f"z_{name}" for name in FACTOR_NAMES]].to_numpy()).all()
        )
        self.assertTrue(np.isfinite(scores["factor_score"].to_numpy()).all())

        # 行业中性化：样本 >=3 的行业组均值应为 0
        scores_full = self._scores()
        labels = self.data["industry"]["industry"]
        for factor in FACTOR_NAMES:
            column = f"z_{factor}"
            for ind in labels.unique():
                members = labels.index[labels == ind]
                if len(members) < 3:
                    continue
                group_mean = float(scores_full.loc[members, column].mean())
                self.assertAlmostEqual(
                    group_mean, 0.0, places=8,
                    msg=f"{column} 在行业 {ind} 内均值不为 0: {group_mean}",
                )

        # 行业样本 < 3：整组退化为全市场 z（用 ep 手工复算）
        small = labels.index[labels == "小样本行业"]
        self.assertEqual(len(small), 2)
        raw_ep = scores_full["raw_ep"].astype(float)
        low, high = np.nanpercentile(raw_ep.to_numpy(), [1.0, 99.0])
        clipped = raw_ep.clip(lower=low, upper=high)
        market_z = (clipped - clipped.mean()) / clipped.std(ddof=0)
        np.testing.assert_allclose(
            scores_full.loc[small, "z_ep"].to_numpy(),
            (market_z * FACTOR_DIRECTION["ep"]).loc[small].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    # ------------------------------------------------------------------
    # 权重
    # ------------------------------------------------------------------
    def test_weight_registry_and_validation(self) -> None:
        validate_weights(DEFAULT_WEIGHTS)
        self.assertEqual(set(DEFAULT_WEIGHTS), set(FACTOR_NAMES))
        for name, weights in RISK_WEIGHTS.items():
            validate_weights(weights)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=12, msg=name)
            self.assertLessEqual(max(weights.values()), MAX_WEIGHT + 1e-12, msg=name)
        for name, weights in WEIGHT_PRESETS.items():
            validate_weights(weights)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=12, msg=name)

        self.assertEqual(weights_for("balanced"), DEFAULT_WEIGHTS)
        self.assertEqual(weights_for("未知风险"), DEFAULT_WEIGHTS)
        copy = weights_for("balanced")
        copy["ep"] = 0.0
        self.assertNotEqual(copy, DEFAULT_WEIGHTS, "weights_for 必须返回副本")

        missing = dict(DEFAULT_WEIGHTS)
        missing.pop("ep")
        bad_sum = dict(DEFAULT_WEIGHTS)
        bad_sum["ep"] += 0.01
        bad_high = dict(DEFAULT_WEIGHTS)
        bad_high["ep"] = 0.41
        bad_high["bp"] = DEFAULT_WEIGHTS["bp"] - 0.01
        bad_negative = dict(DEFAULT_WEIGHTS)
        bad_negative["ep"] = -0.001
        bad_negative["bp"] = DEFAULT_WEIGHTS["bp"] + 0.001
        bad_nan = dict(DEFAULT_WEIGHTS)
        bad_nan["ep"] = float("nan")
        for weights in (missing, bad_sum, bad_high, bad_negative, bad_nan):
            with self.assertRaises(ValueError):
                validate_weights(weights)
        with self.assertRaises(ValueError):
            validate_weights(None)

        preset_name = "unit_test_quality"
        register_preset(preset_name, WEIGHT_PRESETS["quality"])
        self.assertIn(preset_name, WEIGHT_PRESETS)
        validate_weights(WEIGHT_PRESETS[preset_name])
        with self.assertRaises(ValueError):
            register_preset("", WEIGHT_PRESETS["quality"])
        with self.assertRaises(ValueError):
            register_preset("unit_test_bad", {})

    # ------------------------------------------------------------------
    # 筛选
    # ------------------------------------------------------------------
    def test_screen_top_and_exclude(self) -> None:
        top, full = screen(self.bundle, DEFAULT_WEIGHTS, top_n=7)
        expected_columns = ["code", "score", "industry"] + [
            f"z_{name}" for name in FACTOR_NAMES
        ]
        self.assertEqual(list(top.columns), expected_columns)
        self.assertEqual(len(top), 7)
        self.assertEqual(len(full), N_STOCKS)
        self.assertTrue((np.diff(top["score"].to_numpy()) <= 1e-12).all())
        self.assertTrue(top["industry"].notna().all())
        for _, row in top.iterrows():
            self.assertAlmostEqual(
                float(row["score"]), float(full.loc[row["code"], "factor_score"]), places=10
            )

        excluded = top["code"].iloc[:2].tolist()
        top_list, _ = screen(self.data, DEFAULT_WEIGHTS, top_n=7, exclude_codes=excluded)
        self.assertFalse(set(top_list["code"]) & set(excluded))
        self.assertEqual(len(top_list), 7)

        top_str, _ = screen(
            self.data, DEFAULT_WEIGHTS, top_n=7, exclude_codes=",".join(excluded)
        )
        self.assertFalse(set(top_str["code"]) & set(excluded))

        top_all, _ = screen(self.data, DEFAULT_WEIGHTS, top_n=999)
        self.assertEqual(len(top_all), N_STOCKS)
        top_asof, _ = screen(
            self.data, DEFAULT_WEIGHTS, top_n=5, asof=self.dates[250]
        )
        self.assertEqual(len(top_asof), 5)

        with self.assertRaises(ValueError):
            screen(self.data, DEFAULT_WEIGHTS, top_n=0)
        with self.assertRaises(ValueError):
            screen(self.data, {}, top_n=5)
        with self.assertRaises(ValueError):
            screen(self.data, DEFAULT_WEIGHTS, exclude_codes=self.codes)

    # ------------------------------------------------------------------
    # 组合
    # ------------------------------------------------------------------
    def test_build_portfolio_equal_and_risk_parity(self) -> None:
        top, _ = screen(self.data, DEFAULT_WEIGHTS, top_n=10)
        capital = 2_000_000.0

        equal = build_portfolio(top, self.data["prices"], "equal_weight", capital)
        self.assertEqual(list(equal.columns), ["code", "weight_pct", "amount", "method"])
        self.assertEqual(len(equal), 10)
        self.assertAlmostEqual(float(equal["weight_pct"].sum()), 100.0, places=9)
        self.assertAlmostEqual(float(equal["amount"].sum()), capital, places=4)
        self.assertTrue((equal["method"] == "equal_weight").all())
        np.testing.assert_allclose(equal["weight_pct"].to_numpy(), np.full(10, 10.0))

        risk = build_portfolio(top, self.data["prices"], "risk_parity", capital)
        self.assertEqual(list(risk.columns), ["code", "weight_pct", "amount", "method"])
        self.assertAlmostEqual(float(risk["weight_pct"].sum()), 100.0, places=9)
        self.assertGreater(float(risk["weight_pct"].min()), 0.0)
        self.assertTrue((risk["method"] == "risk_parity").all())
        self.assertFalse(np.allclose(risk["weight_pct"], equal["weight_pct"]))
        self.assertGreater(float(risk["weight_pct"].std()), 1e-6)

        # 风险平价的权重方向应与波动率负相关（低波高权）
        prices = self.data["prices"][top["code"].tolist()]
        returns = prices.iloc[-61:].pct_change(fill_method=None).iloc[1:]
        vol = returns.std(ddof=1) * math.sqrt(252)
        corr = np.corrcoef(vol.to_numpy(), risk["weight_pct"].to_numpy())[0, 1]
        self.assertLess(corr, -0.5)

        with self.assertRaises(ValueError):
            build_portfolio(top, self.data["prices"], "unknown", capital)
        with self.assertRaises(ValueError):
            build_portfolio(top, self.data["prices"], "equal_weight", 0)
        with self.assertRaises(ValueError):
            build_portfolio(pd.DataFrame(), self.data["prices"])
        with self.assertRaises(ValueError):
            build_portfolio(top.drop(columns=["code"]), self.data["prices"])
        duplicates = pd.concat([top.head(2), top.head(1)], ignore_index=True)
        with self.assertRaises(ValueError):
            build_portfolio(duplicates, self.data["prices"])
        with self.assertRaises(ValueError):
            build_portfolio(top, self.data["prices"].iloc[:40], "risk_parity")

    # ------------------------------------------------------------------
    # 回测
    # ------------------------------------------------------------------
    def test_backtest_metrics_and_costs(self) -> None:
        start, end = self.dates[130], self.dates[-1]
        result = run_backtest(
            self.bundle, weights_for("balanced"), start=start, end=end, top_n=8
        )
        nav = result["nav"]
        benchmark_nav = result["benchmark_nav"]
        metrics = result["metrics"]

        self.assertIsInstance(nav, pd.Series)
        self.assertIsInstance(benchmark_nav, pd.Series)
        self.assertTrue(nav.index.equals(benchmark_nav.index))
        self.assertEqual(len(nav), int((self.dates >= start).sum()))
        self.assertAlmostEqual(float(nav.iloc[0]), 1.0, places=12)
        self.assertAlmostEqual(float(benchmark_nav.iloc[0]), 1.0, places=12)
        self.assertTrue(np.isfinite(nav.to_numpy()).all())
        self.assertTrue((nav.to_numpy() > 0).all())
        self.assertTrue(np.isfinite(benchmark_nav.to_numpy()).all())
        self.assertGreater(float(nav.std()), 0.0)

        self.assertEqual(set(metrics), {"cagr", "sharpe", "mdd", "ir", "win_rate"})
        for key, value in metrics.items():
            self.assertTrue(math.isfinite(float(value)), msg=key)
        self.assertLessEqual(metrics["mdd"], 0.0)
        self.assertGreater(metrics["cagr"], -1.0)
        self.assertGreaterEqual(metrics["win_rate"], 0.0)
        self.assertLessEqual(metrics["win_rate"], 1.0)
        self.assertGreater(result["rebalances"], 0)
        self.assertIsInstance(result["rebalances"], int)

        zero_cost = run_backtest(
            self.bundle, DEFAULT_WEIGHTS, start=start, end=end, top_n=8,
            cost_buy=0.0, cost_sell=0.0,
        )
        high_cost = run_backtest(
            self.bundle, DEFAULT_WEIGHTS, start=start, end=end, top_n=8,
            cost_buy=0.003, cost_sell=0.006,
        )
        self.assertGreaterEqual(
            float(zero_cost["nav"].iloc[-1]), float(high_cost["nav"].iloc[-1]) - 1e-12,
            "零成本组合净值不应低于高成本组合",
        )

        with self.assertRaises(ValueError):
            run_backtest(self.data, DEFAULT_WEIGHTS, start=self.dates[-1], end=self.dates[0])
        with self.assertRaises(ValueError):
            run_backtest(
                self.data, DEFAULT_WEIGHTS, start=self.dates[100], end=self.dates[110]
            )
        with self.assertRaises(ValueError):
            run_backtest(self.data, DEFAULT_WEIGHTS, start=start, end=end, rebalance_days=0)
        with self.assertRaises(ValueError):
            run_backtest(self.data, DEFAULT_WEIGHTS, start=start, end=end, top_n=0)
        with self.assertRaises(ValueError):
            run_backtest(
                self.data, DEFAULT_WEIGHTS, start=start, end=end, cost_buy=-0.1
            )

    def test_backtest_limit_up_and_no_volume_rules(self) -> None:
        prev_pos, exec_pos = 200, 201
        start, end = self.dates[prev_pos], self.dates[-1]

        # 1) 执行日全部标的涨停：买单应全部跳过（无持仓时表现为零成交）
        limit_data = dict(self.data)
        prices = self.data["prices"].copy()
        close_raw = self.data["close_raw"].copy()
        amount = self.data["amount"].copy()
        prices.iloc[exec_pos] = prices.iloc[prev_pos].to_numpy() * 1.10
        close_raw.iloc[exec_pos] = close_raw.iloc[prev_pos].to_numpy() * 1.10
        amount.iloc[exec_pos] = 1e8
        amount.iloc[exec_pos + 1 :] = 0.0  # 防止顺延成交，隔离涨停规则
        limit_data.update({"prices": prices, "close_raw": close_raw, "amount": amount})
        result = run_backtest(
            limit_data, DEFAULT_WEIGHTS, start=start, end=end,
            top_n=8, rebalance_days=400,
        )
        self.assertEqual(result["rebalances"], 0)
        np.testing.assert_allclose(result["nav"].to_numpy(), 1.0)
        self.assertEqual(result["metrics"]["mdd"], 0.0)

        # 2) 执行日全部无成交（成交额 0）：订单顺延但始终无法成交
        suspend_data = dict(self.data)
        amount = self.data["amount"].copy()
        amount.iloc[exec_pos:] = 0.0
        suspend_data["amount"] = amount
        result = run_backtest(
            suspend_data, DEFAULT_WEIGHTS, start=start, end=end,
            top_n=8, rebalance_days=400,
        )
        self.assertEqual(result["rebalances"], 0)
        np.testing.assert_allclose(result["nav"].to_numpy(), 1.0)

    # ------------------------------------------------------------------
    # 包导出
    # ------------------------------------------------------------------
    def test_package_exports(self) -> None:
        import backend.engine as engine

        for name in (
            "FACTOR_NAMES", "FACTOR_DIRECTION", "RISK_WEIGHTS", "DEFAULT_WEIGHTS",
            "WEIGHT_PRESETS", "weights_for", "validate_weights", "register_preset",
            "compute_factor_scores", "screen", "build_portfolio", "run_backtest",
        ):
            self.assertTrue(hasattr(engine, name), msg=name)


if __name__ == "__main__":
    unittest.main()
