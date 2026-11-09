# -*- coding: utf-8 -*-
"""服务端网格寻优引擎单测：范围校验、全组合网格/预算采样、目标指标与回撤约束。"""
from __future__ import annotations

import math
import os
import unittest
from unittest import mock

from backend.engine.factors import FACTOR_NAMES
from backend.engine.tuning import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_TUNING_TRIALS,
    MAX_TRIALS,
    MIN_TRIALS,
    OBJECTIVE_LABELS,
    OBJECTIVES,
    TuningError,
    max_trials_limit,
    normalize_param_space,
    normalize_weight_space,
    run_search,
    score_metrics,
)


def _equal_weights() -> dict[str, float]:
    return {name: 1.0 / len(FACTOR_NAMES) for name in FACTOR_NAMES}


def _flat_evaluator(params: dict) -> dict:
    """不看参数、恒定返回一组合法指标（只用来验证编排逻辑）。"""
    return {"metrics": {"sharpe": 0.5, "cagr": 0.1, "mdd": -0.1,
                        "ir": 0.2, "win_rate": 0.55}}


def _weight_space(center: dict[str, float], span: float = 0.2, step: float = 0.05,
                  factors: list[str] | None = None) -> dict[str, dict[str, float]]:
    names = factors or list(FACTOR_NAMES)
    space: dict[str, dict[str, float]] = {}
    for name in names:
        low = max(0.0, round(center.get(name, 0.1) - span, 6))
        high = min(0.4, round(center.get(name, 0.1) + span, 6))
        space[name] = {"min": low, "max": high, "step": step}
    return space


def _grid(grid_space: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    """测试侧复算每轴候选值（min/max/step 含端点）。"""
    result: dict[str, list[float]] = {}
    for key, spec in grid_space.items():
        minimum, maximum, step = spec["min"], spec["max"], spec["step"]
        count = int(math.floor((maximum - minimum) / step + 1e-9)) + 1
        values = [round(minimum + index * step, 10) for index in range(count)]
        if values[-1] < maximum - 1e-9:
            values.append(round(maximum, 10))
        result[key] = values
    return result


class WeightSpaceTest(unittest.TestCase):
    def test_normalizes_and_keeps_order(self) -> None:
        space = normalize_weight_space({
            "roe": {"min": 0.05, "max": 0.25, "step": 0.05},
            "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
            "size": {"min": 0.02, "max": 0.1, "step": 0.02},
        })
        self.assertEqual(list(space), ["roe", "ep", "size"])
        self.assertAlmostEqual(space["roe"]["min"], 0.05)

    def test_allows_single_factor(self) -> None:
        # 至少 1 个因子即可（不再强制 3 个）
        space = normalize_weight_space({
            "roe": {"min": 0.05, "max": 0.25, "step": 0.05},
        })
        self.assertEqual(list(space), ["roe"])
        # min_axes 仍可按调用方需要收紧
        with self.assertRaises(TuningError):
            normalize_weight_space(
                {"roe": {"min": 0.05, "max": 0.25, "step": 0.05}}, min_axes=2)

    def test_rejects_empty_space(self) -> None:
        with self.assertRaises(TuningError):
            normalize_weight_space({})

    def test_rejects_unknown_factor(self) -> None:
        with self.assertRaises(TuningError):
            normalize_weight_space({
                "roe": {"min": 0.05, "max": 0.25, "step": 0.05},
                "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
                "unknown": {"min": 0.0, "max": 0.2, "step": 0.1},
            })

    def test_rejects_range_beyond_single_factor_cap(self) -> None:
        with self.assertRaises(TuningError):
            normalize_weight_space({
                "roe": {"min": 0.05, "max": 0.5, "step": 0.05},
                "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
                "size": {"min": 0.0, "max": 0.2, "step": 0.1},
            })

    def test_rejects_negative_or_reversed_range(self) -> None:
        with self.assertRaises(TuningError):
            normalize_weight_space({
                "roe": {"min": -0.1, "max": 0.2, "step": 0.05},
                "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
                "size": {"min": 0.0, "max": 0.2, "step": 0.1},
            })
        with self.assertRaises(TuningError):
            normalize_weight_space({
                "roe": {"min": 0.2, "max": 0.1, "step": 0.05},
                "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
                "size": {"min": 0.0, "max": 0.2, "step": 0.1},
            })

    def test_rejects_step_larger_than_range(self) -> None:
        with self.assertRaises(TuningError):
            normalize_weight_space({
                "roe": {"min": 0.1, "max": 0.12, "step": 0.05},
                "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
                "size": {"min": 0.0, "max": 0.2, "step": 0.1},
            })


class ParamSpaceTest(unittest.TestCase):
    def test_float_and_int_params(self) -> None:
        space = normalize_param_space({
            "lookback": {"min": 10, "max": 60, "step": 10, "type": "int", "default": 20},
            "tilt": {"min": 0.0, "max": 0.8, "step": 0.2, "type": "float"},
        })
        self.assertEqual(space["lookback"]["type"], "int")
        self.assertEqual(space["lookback"]["default"], 20.0)
        self.assertAlmostEqual(space["tilt"]["default"], 0.4)

    def test_allows_more_than_four_params(self) -> None:
        # 参数个数不再有 4 个硬上限
        space = normalize_param_space({
            f"p{index}": {"min": 0.0, "max": 1.0, "step": 0.5} for index in range(6)
        })
        self.assertEqual(len(space), 6)
        self.assertEqual(list(space), [f"p{index}" for index in range(6)])

    def test_rejects_bad_params(self) -> None:
        with self.assertRaises(TuningError):
            normalize_param_space({})
        with self.assertRaises(TuningError):
            normalize_param_space({"bad-name": {"min": 0, "max": 1, "step": 0.1}})
        with self.assertRaises(TuningError):
            normalize_param_space({"x": {"min": 0, "max": 1, "step": 0.1, "type": "cat"}})
        with self.assertRaises(TuningError):
            normalize_param_space({"x": {"min": 0, "max": 0.05, "step": 0.1}})
        with self.assertRaises(TuningError):
            normalize_param_space({"n": {"min": 0.5, "max": 3.5, "step": 1, "type": "int"}})


class ObjectivesTest(unittest.TestCase):
    def test_objectives_contract(self) -> None:
        self.assertEqual(
            set(OBJECTIVES),
            {"sharpe", "cagr", "mdd", "calmar", "sortino", "ir", "win_rate"},
        )
        for objective in OBJECTIVES:
            self.assertIn(objective, OBJECTIVE_LABELS)
            self.assertTrue(OBJECTIVE_LABELS[objective].strip())
        with self.assertRaises(TuningError):
            score_metrics({"sharpe": 1.0}, "magic")
        with self.assertRaises(TuningError):
            list(run_search(_flat_evaluator, space=_weight_space(_equal_weights()),
                            base_params=_equal_weights(), objective="magic"))

    def test_calmar_uses_metrics_value_or_derives_from_cagr_mdd(self) -> None:
        # metrics 已有 calmar 时直接用
        score, ok = score_metrics({"calmar": 3.5, "cagr": 0.2, "mdd": -0.1}, "calmar")
        self.assertAlmostEqual(score, 3.5)
        self.assertTrue(ok)
        # 缺失时用 cagr / |mdd| 派生
        score, ok = score_metrics({"cagr": 0.2, "mdd": -0.1}, "calmar")
        self.assertAlmostEqual(score, 2.0)
        self.assertTrue(ok)
        # 回撤上限约束对非 mdd 目标仍然生效
        score, ok = score_metrics({"cagr": 0.2, "mdd": -0.3}, "calmar", tolerance=0.2)
        self.assertFalse(ok)
        self.assertLess(score, 0.667)

    def test_sortino_uses_metrics_value_or_derives_from_nav(self) -> None:
        self.assertAlmostEqual(
            score_metrics({"sortino": 1.8, "cagr": 0.1, "mdd": -0.1}, "sortino")[0],
            1.8,
        )
        nav = [1.0, 1.01, 0.99, 1.02, 1.0, 1.03]
        returns = [nav[index] / nav[index - 1] - 1.0 for index in range(1, len(nav))]
        mean = sum(returns) / len(returns)
        downside = [min(value, 0.0) for value in returns]
        deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
        expected = mean / deviation * math.sqrt(252)
        score, ok = score_metrics({"cagr": 0.1, "mdd": -0.1}, "sortino", nav=nav)
        self.assertAlmostEqual(score, expected, places=9)
        self.assertTrue(ok)
        # 缺失 nav 时退化为 0（不抛错）
        self.assertEqual(score_metrics({"cagr": 0.1}, "sortino")[0], 0.0)

    def test_ir_and_win_rate_are_maximized(self) -> None:
        self.assertAlmostEqual(
            score_metrics({"ir": 0.9, "cagr": 0.1, "mdd": -0.1}, "ir")[0], 0.9)
        self.assertAlmostEqual(
            score_metrics({"win_rate": 0.61, "cagr": 0.1, "mdd": -0.1}, "win_rate")[0],
            0.61,
        )


class RunSearchTest(unittest.TestCase):
    def test_limits_and_defaults(self) -> None:
        self.assertEqual(MIN_TRIALS, 1)
        self.assertEqual(DEFAULT_MAX_TRIALS, 1000)
        self.assertGreaterEqual(MAX_TRIALS, 1)
        self.assertGreaterEqual(max_trials_limit(), 1)
        # 预算常量与调参方案默认值对齐
        self.assertEqual(DEFAULT_TUNING_TRIALS, 32)

    def test_accepts_more_than_four_params(self) -> None:
        space = {
            f"p{index}": {"min": 0.0, "max": 1.0, "step": 1.0}
            for index in range(6)
        }
        done = list(run_search(_flat_evaluator, space=space, base_params={},
                               objective="sharpe", max_trials=100,
                               kind="param"))[-1]
        self.assertFalse(done["sampled"])
        self.assertEqual(done["total_combinations"], 64)
        self.assertEqual(done["evaluated"], done["planned_trials"])
        self.assertEqual(len(done["space"]), 6)

    def test_full_grid_when_total_within_budget(self) -> None:
        base = _equal_weights()
        target_raw = {"roe": 0.2, "ep": 0.1, "bp": 0.1, "size": 0.1,
                      "volatility_60": 0.1}
        total = sum(target_raw.values())
        target = {name: value / total for name, value in target_raw.items()}

        def evaluate(params: dict) -> dict:
            raw = {name: float(params.get(name, 0.0)) for name in target_raw}
            raw_total = sum(raw.values()) or 1.0
            weights = {name: value / raw_total for name, value in raw.items()}
            error = sum(abs(weights[name] - target[name]) for name in target_raw)
            return {"metrics": {"sharpe": round(1.0 - error, 6), "cagr": 0.1,
                                "mdd": -0.1, "ir": 0.2, "win_rate": 0.55},
                    "params": weights}

        space = {
            "roe": {"min": 0.1, "max": 0.3, "step": 0.2},
            "ep": {"min": 0.05, "max": 0.15, "step": 0.1},
            "bp": {"min": 0.05, "max": 0.15, "step": 0.1},
            "size": {"min": 0.05, "max": 0.15, "step": 0.1},
            "volatility_60": {"min": 0.05, "max": 0.15, "step": 0.1},
        }
        events = list(run_search(evaluate, space=space, base_params=base,
                                 objective="sharpe", max_trials=40))
        start, done = events[0], events[-1]
        self.assertFalse(start["sampled"])
        self.assertFalse(done["sampled"])
        self.assertEqual(done["total_combinations"], 2 ** 5)
        self.assertEqual(done["planned_trials"], done["evaluated"])
        self.assertLessEqual(done["planned_trials"], 40)
        self.assertEqual(start["total"], done["planned_trials"])
        # 全量网格里必然包含与目标比例一致的组合
        self.assertGreater(done["best"]["score"], 0.9)
        self.assertAlmostEqual(sum(done["best"]["params"].values()), 1.0, places=6)
        self.assertGreaterEqual(len(done["trials"]), done["total_combinations"])

    def test_budget_shortfall_samples_without_dropping_axes(self) -> None:
        base = _equal_weights()
        factors = ["roe", "ep", "bp", "size"]
        space = _weight_space(base, span=0.2, step=0.05, factors=factors)
        expected = _grid(space)
        events = list(run_search(_flat_evaluator, space=space, base_params=base,
                                 objective="sharpe", max_trials=12))
        start, done = events[0], events[-1]
        self.assertTrue(start["sampled"])
        self.assertTrue(done["sampled"])
        self.assertEqual(start["planned_trials"], 12)
        self.assertEqual(done["planned_trials"], 12)
        self.assertLessEqual(done["evaluated"], done["planned_trials"])
        total_combinations = 1
        for values in expected.values():
            total_combinations *= len(values)
        self.assertEqual(done["total_combinations"], total_combinations)
        self.assertGreater(done["total_combinations"], done["planned_trials"])
        # 没有丢轴：所有因子轴都还在，且每个轴都被真正搜索到（出现多个取值）
        self.assertEqual([item["key"] for item in done["space"]], factors)
        trial_params = [event["params"] for event in events
                        if event["event"] == "trial" and "params" in event]
        for axis in factors:
            values = {round(float(params[axis]), 6) for params in trial_params}
            self.assertEqual(values, {round(value, 6) for value in expected[axis]},
                             f"轴 {axis} 未被覆盖")
        # 包含基线组合
        self.assertTrue(any(
            all(abs(float(params[axis]) - base[axis]) < 1e-9 for axis in factors)
            for params in trial_params
        ))
        # 进度单调可预期
        trials = [event for event in events if event["event"] == "trial"]
        self.assertEqual([event["trial"] for event in trials],
                         list(range(1, len(trials) + 1)))
        for event in trials:
            self.assertGreaterEqual(event["progress"], 0)
            self.assertLessEqual(event["progress"], 1)
        self.assertAlmostEqual(trials[-1]["progress"], 1.0, places=6)

    def test_max_trials_none_uses_default_budget(self) -> None:
        events = list(run_search(_flat_evaluator,
                                 space=_weight_space(_equal_weights(), step=0.02),
                                 base_params=_equal_weights(), max_trials=None))
        start, done = events[0], events[-1]
        self.assertEqual(start["planned_trials"], DEFAULT_TUNING_TRIALS)
        self.assertTrue(start["sampled"])
        self.assertLessEqual(done["evaluated"], DEFAULT_TUNING_TRIALS)

    def test_max_trials_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"TUNING_MAX_TRIALS": "7"}):
            self.assertEqual(max_trials_limit(), 7)
            events = list(run_search(
                _flat_evaluator,
                space=_weight_space(_equal_weights(), step=0.02),
                base_params=_equal_weights(), max_trials=100))
            start, done = events[0], events[-1]
            self.assertEqual(start["planned_trials"], 7)
            self.assertEqual(done["planned_trials"], 7)
            self.assertTrue(done["sampled"])
            self.assertLessEqual(done["evaluated"], 7)
        # 环境变量移除后回到默认上限
        self.assertGreaterEqual(max_trials_limit(), 1)

    def test_single_axis_search_space(self) -> None:
        space = {"roe": {"min": 0.1, "max": 0.3, "step": 0.1}}
        events = list(run_search(_flat_evaluator, space=space,
                                 base_params=_equal_weights(), max_trials=10))
        start, done = events[0], events[-1]
        self.assertEqual([item["key"] for item in start["space"]], ["roe"])
        self.assertEqual(done["total_combinations"], 3)
        self.assertFalse(done["sampled"])

    def test_mdd_objective_prefers_smaller_drawdown(self) -> None:
        base = {"roe": 0.1, "ep": 0.15, "size": 0.1, "volatility_60": 0.1}
        space = {
            "roe": {"min": 0.05, "max": 0.25, "step": 0.1},
            "ep": {"min": 0.05, "max": 0.25, "step": 0.1},
            "volatility_60": {"min": 0.05, "max": 0.25, "step": 0.1},
        }

        def evaluate(params):
            # 越看重低波因子，回撤越小
            mdd = -0.25 + 0.6 * float(params.get("volatility_60", 0.0))
            return {"metrics": {"sharpe": 0.1, "cagr": 0.05, "mdd": mdd,
                                "ir": 0.1, "win_rate": 0.4}}

        done = list(run_search(evaluate, space=space, base_params=base,
                               objective="mdd", max_trials=40))[-1]
        self.assertGreater(done["best"]["metrics"]["mdd"], -0.25)
        self.assertAlmostEqual(done["best"]["metric_value"],
                               done["best"]["metrics"]["mdd"])

    def test_calmar_objective_derives_and_ranks(self) -> None:
        base = _equal_weights()
        space = {
            "roe": {"min": 0.1, "max": 0.3, "step": 0.1},
            "ep": {"min": 0.05, "max": 0.15, "step": 0.05},
        }

        def evaluate(params):
            # 不直接给 calmar，让寻优引擎用 cagr / |mdd| 派生
            return {"metrics": {"cagr": 0.05 + float(params.get("roe", 0.1)),
                                "mdd": -0.2, "sharpe": 0.1,
                                "ir": 0.1, "win_rate": 0.5}}

        done = list(run_search(evaluate, space=space, base_params=base,
                               objective="calmar", max_trials=40))[-1]
        self.assertAlmostEqual(done["best"]["metric_value"], (0.05 + 0.3) / 0.2,
                               places=6)

    def test_ir_objective_uses_metric(self) -> None:
        space = {"roe": {"min": 0.1, "max": 0.3, "step": 0.1}}

        def evaluate(params):
            return {"metrics": {"sharpe": 0.1, "cagr": 0.1, "mdd": -0.1,
                                "ir": 0.1 + float(params["roe"]),
                                "win_rate": 0.5}}

        done = list(run_search(evaluate, space=space,
                               base_params=_equal_weights(),
                               objective="ir", max_trials=10))[-1]
        self.assertAlmostEqual(done["best"]["metric_value"], 0.4, places=6)

    def test_drawdown_tolerance_penalizes_risky_candidates(self) -> None:
        base = {"roe": 0.1, "ep": 0.15, "volatility_60": 0.1}
        space = {
            "roe": {"min": 0.1, "max": 0.3, "step": 0.1},
            "ep": {"min": 0.1, "max": 0.3, "step": 0.1},
            "volatility_60": {"min": 0.05, "max": 0.3, "step": 0.1},
        }

        def evaluate(params):
            # 提高 roe 收益高但回撤突破 20%，应被约束扣分
            roe = float(params.get("roe", 0.1))
            sharpe = 0.5 + roe * 2.0
            mdd = -0.12 - roe * 0.8
            return {"metrics": {"sharpe": sharpe, "cagr": 0.1, "mdd": mdd,
                                "ir": 0.2, "win_rate": 0.5}}

        done = list(run_search(evaluate, space=space, base_params=base,
                               objective="sharpe", max_trials=40,
                               tolerance=0.2))[-1]
        self.assertLessEqual(abs(done["best"]["metrics"]["mdd"]), 0.2 + 1e-9)
        self.assertTrue(done["best"]["constraint_ok"])

    def test_param_search_and_int_axis(self) -> None:
        space = {
            "tilt": {"min": 0.0, "max": 0.8, "step": 0.2, "type": "float", "default": 0.4},
            "lookback": {"min": 10, "max": 60, "step": 10, "type": "int", "default": 20},
        }

        def evaluate(params):
            score = (1.0 - abs(float(params["tilt"]) - 0.6)
                     - abs(int(params["lookback"]) - 60) / 100.0)
            return {"metrics": {"sharpe": score, "cagr": 0.1, "mdd": -0.1,
                                "ir": 0.1, "win_rate": 0.5}}

        events = list(run_search(evaluate, space=space, base_params={},
                                 objective="sharpe", max_trials=40, kind="param"))
        done = events[-1]
        self.assertFalse(done["sampled"])
        self.assertAlmostEqual(done["best"]["params"]["tilt"], 0.6, places=6)
        self.assertEqual(done["best"]["params"]["lookback"], 60)
        self.assertIsInstance(done["best"]["params"]["lookback"], int)

    def test_failures_are_skipped_and_reported(self) -> None:
        space = {
            "roe": {"min": 0.1, "max": 0.3, "step": 0.1},
            "ep": {"min": 0.1, "max": 0.3, "step": 0.1},
            "size": {"min": 0.1, "max": 0.3, "step": 0.1},
        }

        def evaluate(params):
            if float(params.get("ep", 0)) > 0.15:
                raise ValueError("bad combo")
            return {"metrics": {"sharpe": 0.2, "cagr": 0.1, "mdd": -0.1,
                                "ir": 0.1, "win_rate": 0.5}}

        done = list(run_search(evaluate, space=space,
                               base_params={"roe": 0.1, "ep": 0.1, "size": 0.1},
                               objective="sharpe", max_trials=40))[-1]
        self.assertGreaterEqual(done["failures"], 1)
        self.assertIsNotNone(done["best"])
        self.assertLessEqual(done["evaluated"], done["planned_trials"])

    def test_all_failures_raise(self) -> None:
        space = {
            "roe": {"min": 0.1, "max": 0.3, "step": 0.1},
            "ep": {"min": 0.1, "max": 0.3, "step": 0.1},
            "size": {"min": 0.1, "max": 0.3, "step": 0.1},
        }

        def evaluate(params):
            raise RuntimeError("backtest exploded")

        with self.assertRaises(TuningError):
            list(run_search(evaluate, space=space,
                            base_params={"roe": 0.1, "ep": 0.1, "size": 0.1}))

    def test_int_params_stay_int_when_baseline_wins(self) -> None:
        space = {
            "lookback": {"min": 10, "max": 60, "step": 10, "type": "int", "default": 20},
            "tilt": {"min": 0.0, "max": 0.8, "step": 0.2, "type": "float", "default": 0.4},
        }

        def evaluate(params):
            # 所有组合同分：应保留基线参数，整数参数仍必须是 int
            return {"metrics": {"sharpe": 0.3, "cagr": 0.1, "mdd": -0.1,
                                "ir": 0.1, "win_rate": 0.5}}

        done = list(run_search(evaluate, space=space, base_params={},
                               objective="sharpe", max_trials=8, kind="param"))[-1]
        self.assertIsInstance(done["best"]["params"]["lookback"], int)
        self.assertEqual(done["best"]["params"]["lookback"], 20)

    def test_sampling_is_deterministic(self) -> None:
        space = _weight_space(_equal_weights(), step=0.02)
        base = _equal_weights()
        first = list(run_search(_flat_evaluator, space=space, base_params=base,
                                max_trials=9))
        second = list(run_search(_flat_evaluator, space=space, base_params=base,
                                 max_trials=9))
        first_params = [event.get("params") for event in first
                        if event["event"] == "trial"]
        second_params = [event.get("params") for event in second
                         if event["event"] == "trial"]
        self.assertEqual(first_params, second_params)


if __name__ == "__main__":
    unittest.main()
