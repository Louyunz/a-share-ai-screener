# -*- coding: utf-8 -*-
"""服务端网格寻优引擎单测：范围校验、逐轴搜索、预算、目标与回撤约束。"""
from __future__ import annotations

import unittest

from backend.engine.factors import FACTOR_NAMES
from backend.engine.tuning import (
    OBJECTIVES,
    TuningError,
    normalize_param_space,
    normalize_weight_space,
    run_search,
)


def _equal_weights() -> dict[str, float]:
    return {name: 1.0 / len(FACTOR_NAMES) for name in FACTOR_NAMES}


def _weight_space(center: dict[str, float], span: float = 0.2, step: float = 0.05,
                  factors: list[str] | None = None) -> dict[str, dict[str, float]]:
    names = factors or list(FACTOR_NAMES)
    space: dict[str, dict[str, float]] = {}
    for name in names:
        low = max(0.0, round(center.get(name, 0.1) - span, 6))
        high = min(0.4, round(center.get(name, 0.1) + span, 6))
        space[name] = {"min": low, "max": high, "step": step}
    return space


class WeightSpaceTest(unittest.TestCase):
    def test_normalizes_and_keeps_order(self) -> None:
        space = normalize_weight_space({
            "roe": {"min": 0.05, "max": 0.25, "step": 0.05},
            "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
            "size": {"min": 0.02, "max": 0.1, "step": 0.02},
        })
        self.assertEqual(list(space), ["roe", "ep", "size"])
        self.assertAlmostEqual(space["roe"]["min"], 0.05)

    def test_requires_three_axes(self) -> None:
        with self.assertRaises(TuningError):
            normalize_weight_space({
                "roe": {"min": 0.05, "max": 0.25, "step": 0.05},
                "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
            })

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

    def test_rejects_bad_params(self) -> None:
        with self.assertRaises(TuningError):
            normalize_param_space({"bad-name": {"min": 0, "max": 1, "step": 0.1}})
        with self.assertRaises(TuningError):
            normalize_param_space({"x": {"min": 0, "max": 1, "step": 0.1, "type": "cat"}})
        with self.assertRaises(TuningError):
            normalize_param_space({"x": {"min": 0, "max": 0.05, "step": 0.1}})
        with self.assertRaises(TuningError):
            normalize_param_space({
                f"p{index}": {"min": 0, "max": 1, "step": 0.5} for index in range(5)
            })
        with self.assertRaises(TuningError):
            normalize_param_space({"n": {"min": 0.5, "max": 3.5, "step": 1, "type": "int"}})


class RunSearchTest(unittest.TestCase):
    def _weight_evaluator(self, target: dict[str, float]):
        def evaluate(params):
            total = sum(float(value) for value in params.values())
            weights = {name: float(params.get(name, 0.0)) / total for name in FACTOR_NAMES}
            error = sum(abs(weights[name] - target[name]) for name in FACTOR_NAMES)
            metrics = {"sharpe": round(1.0 - error, 6), "cagr": 0.1,
                       "mdd": -0.15, "ir": 0.4, "win_rate": 0.5}
            return {"metrics": metrics, "params": weights}
        return evaluate

    def test_finds_better_weights_than_baseline(self) -> None:
        base = _equal_weights()
        target = {name: 0.0889 for name in FACTOR_NAMES}
        target["roe"] = 0.2
        # 搜索范围围绕当前权重与目标设置，验证逐轴搜索能向目标靠近
        space = {
            "roe": {"min": 0.1, "max": 0.3, "step": 0.05},
            "ep": {"min": 0.05, "max": 0.15, "step": 0.05},
            "bp": {"min": 0.05, "max": 0.15, "step": 0.05},
            "volatility_60": {"min": 0.05, "max": 0.15, "step": 0.05},
            "size": {"min": 0.05, "max": 0.15, "step": 0.05},
        }
        events = list(run_search(self._weight_evaluator(target), space=space,
                                 base_params=base, objective="sharpe", max_trials=24))
        kinds = [event["event"] for event in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        self.assertGreater(kinds.count("trial"), 2)
        done = events[-1]
        self.assertGreater(done["best"]["score"], 0.9)
        self.assertLessEqual(done["evaluated"], done["total"])
        self.assertLessEqual(done["total"], 24)
        self.assertIn("roe", done["best"]["params"])
        self.assertAlmostEqual(sum(done["best"]["params"].values()), 1.0, places=6)

    def test_budget_and_progress_are_stable(self) -> None:
        base = _equal_weights()
        space = _weight_space(base, span=0.2, step=0.02)  # 每轴 11 档，必须被裁剪
        events = list(run_search(self._weight_evaluator(dict(base)), space=space,
                                 base_params=base, max_trials=12))
        start = events[0]
        trials = [event for event in events if event["event"] == "trial"]
        self.assertLessEqual(start["total"], 12)
        self.assertLessEqual(len(trials), start["total"])
        total = start["total"]
        for event in trials:
            self.assertGreaterEqual(event["trial"], 1)
            self.assertLessEqual(event["trial"], total)
            self.assertGreaterEqual(event["progress"], 0)
            self.assertLessEqual(event["progress"], 1)

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
                               objective="mdd", max_trials=16))[-1]
        self.assertGreater(done["best"]["metrics"]["mdd"], -0.25)

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
                               objective="sharpe", max_trials=16, tolerance=0.2))[-1]
        self.assertLessEqual(abs(done["best"]["metrics"]["mdd"]), 0.2 + 1e-9)
        self.assertTrue(done["best"]["constraint_ok"])

    def test_param_search_and_int_axis(self) -> None:
        space = {
            "tilt": {"min": 0.0, "max": 0.8, "step": 0.2, "type": "float", "default": 0.4},
            "lookback": {"min": 10, "max": 60, "step": 10, "type": "int", "default": 20},
        }

        def evaluate(params):
            score = 1.0 - abs(float(params["tilt"]) - 0.6) - abs(int(params["lookback"]) - 60) / 100.0
            return {"metrics": {"sharpe": score, "cagr": 0.1, "mdd": -0.1,
                                "ir": 0.1, "win_rate": 0.5}}

        events = list(run_search(evaluate, space=space, base_params={},
                                 objective="sharpe", max_trials=20, kind="param"))
        done = events[-1]
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
                               objective="sharpe", max_trials=16))[-1]
        self.assertGreaterEqual(done["failures"], 1)
        self.assertIsNotNone(done["best"])

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

    def test_objectives_contract(self) -> None:
        self.assertEqual(set(OBJECTIVES), {"sharpe", "cagr", "mdd"})
        with self.assertRaises(TuningError):
            list(run_search(lambda params: {"metrics": {}},
                            space=_weight_space(_equal_weights()),
                            base_params=_equal_weights(), objective="magic"))


if __name__ == "__main__":
    unittest.main()
