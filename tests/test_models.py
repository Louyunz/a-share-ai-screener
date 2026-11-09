# -*- coding: utf-8 -*-
"""AI 建模流水线单元测试：registry / custom 代码安全 / pipeline。

全部使用 unittest，不依赖 backend.agent、backend.service 或真实 LLM。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests.support.scripted_llm import ScriptedLLM
from backend.models import (
    DEFAULT_MODEL_ID,
    MODEL_TYPES,
    ModelCodeError,
    ModelType,
    ModelValidationError,
    compile_code,
    get_model,
    list_models,
    validate_code,
    validate_weights,
)
from backend.models.pipeline import (
    DEFAULT_TUNING_MAX_TRIALS,
    MAX_STATE_CHARS,
    MAX_TUNING_PARAMS,
    MAX_TUNING_REPAIR_ATTEMPTS,
    MIN_SEARCH_FACTORS,
    OBJECTIVES,
    TUNING_REPAIR_SYSTEM,
    TUNING_SYSTEM,
    build_tuning_plan,
    normalize_tuning_plan,
)
from backend.models.registry import FACTOR_NAMES, MAX_WEIGHT

# 合法自定义源码：批量加权 10 个 z 因子
VALID_CODE = '''import math

import numpy as np
import pandas as pd


def build_transform():
    def transform(z, weights):
        score = pd.Series(0.0, index=z.index, dtype=float)
        for factor, weight in weights.items():
            column = "z_" + factor
            if column in z.columns:
                values = pd.to_numeric(z[column], errors="coerce").fillna(0.0)
                score = score + float(weight) * values
        return score

    return transform
'''

# 返回 NaN 的源码：能过 AST，但冒烟应失败
NAN_CODE = '''import pandas as pd


def build_transform():
    def transform(z, weights):
        return pd.Series([float("nan")] * len(z), index=z.index)

    return transform
'''

# 返回 numpy 数组而不是 Series：冒烟应失败
ARRAY_CODE = '''import numpy as np


def build_transform():
    def transform(z, weights):
        return np.zeros(len(z))

    return transform
'''

# 参数化引擎：build_transform(params=None)，tilt 控制低波动因子的额外加分
PARAM_CODE = '''import numpy as np
import pandas as pd


def build_transform(params=None):
    options = params or {}
    tilt = float(options.get("tilt", 0.5))

    def transform(z, weights):
        score = pd.Series(0.0, index=z.index, dtype=float)
        for factor, weight in weights.items():
            column = "z_" + factor
            if column in z.columns:
                values = pd.to_numeric(z[column], errors="coerce").fillna(0.0)
                score = score + float(weight) * values
        extra = pd.to_numeric(z["z_volatility_60"], errors="coerce").fillna(0.0)
        return score + tilt * extra

    return transform
'''

# 必须拿到 params 才能运行的参数化引擎（用于验证冒烟确实传入 params）
REQUIRE_PARAMS_CODE = '''import pandas as pd


def build_transform(params=None):
    options = dict(params or {})
    boost = float(options["boost"])

    def transform(z, weights):
        return pd.Series(0.0, index=z.index) + boost

    return transform
'''

REQUIRED_MODEL_IDS = (
    "multi_factor",
    "low_vol_quality",
    "momentum_growth",
    "value_dividend",
    "contrarian",
)


def _z_frame(rows: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(20240601)
    data = {f"z_{name}": rng.normal(size=rows) for name in FACTOR_NAMES}
    return pd.DataFrame(data, index=[f"6000{i:02d}" for i in range(rows)])


def _equal_weights() -> dict[str, float]:
    return {name: 1.0 / len(FACTOR_NAMES) for name in FACTOR_NAMES}


def _weights_total(weights) -> float:
    return float(sum(float(value) for value in weights.values()))


def _assert_weights_close(case: unittest.TestCase, actual, expected) -> None:
    case.assertEqual(set(actual), set(expected))
    for factor in FACTOR_NAMES:
        case.assertAlmostEqual(actual[factor], expected[factor], places=12)


class FakeProposalLLM:
    """只实现 chat_json 的假 LLM，记录调用参数。"""

    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict:
        self.calls.append((system, messages))
        if self.error is not None:
            raise self.error
        return self.response


class SequenceLLM:
    """按调用顺序返回预设响应的假 LLM（用于自动修复测试）。"""

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict:
        self.calls.append((system, messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class RegistryTests(unittest.TestCase):
    """内置模型注册表。"""

    def test_builtin_model_ids_complete(self) -> None:
        self.assertGreaterEqual(len(MODEL_TYPES), 5)
        for model_id in REQUIRED_MODEL_IDS:
            self.assertIn(model_id, MODEL_TYPES)
            self.assertEqual(MODEL_TYPES[model_id].id, model_id)

    def test_each_model_weights_valid(self) -> None:
        for model_id, model in MODEL_TYPES.items():
            self.assertIsInstance(model, ModelType)
            with self.subTest(model=model_id):
                self.assertEqual(set(model.weights), set(FACTOR_NAMES))
                self.assertEqual(len(model.weights), 10)
                self.assertAlmostEqual(_weights_total(model.weights), 1.0, places=9)
                for factor, weight in model.weights.items():
                    self.assertGreaterEqual(weight, 0.0)
                    self.assertLessEqual(weight, MAX_WEIGHT + 1e-9)

    def test_model_name_and_description_are_plain_chinese(self) -> None:
        for model_id, model in MODEL_TYPES.items():
            with self.subTest(model=model_id):
                self.assertTrue(model.name.strip())
                self.assertTrue(model.description.strip())
                self.assertGreaterEqual(len(model.description), 10)

    def test_list_models(self) -> None:
        models = list_models()
        self.assertIsInstance(models, list)
        self.assertEqual(len(models), len(MODEL_TYPES))
        self.assertEqual([item["id"] for item in models], list(MODEL_TYPES))
        for item in models:
            self.assertEqual(set(item), {"id", "name", "description"})
            self.assertIsInstance(item["name"], str)

    def test_list_models_returns_fresh_copies(self) -> None:
        list_models()[0]["name"] = "被篡改"
        self.assertNotEqual(list_models()[0]["name"], "被篡改")

    def test_get_model_returns_weights_copy(self) -> None:
        model = get_model("multi_factor")
        self.assertEqual(model["id"], "multi_factor")
        self.assertEqual(set(model["weights"]), set(FACTOR_NAMES))
        model["weights"]["ep"] = 999.0
        self.assertNotEqual(get_model("multi_factor")["weights"]["ep"], 999.0)

    def test_get_model_unknown(self) -> None:
        with self.assertRaises(ValueError):
            get_model("does_not_exist")

    def test_validate_weights_fills_missing_and_normalizes(self) -> None:
        result = validate_weights({"ep": 0.2, "bp": 0.2, "roe": 0.2})
        self.assertEqual(set(result), set(FACTOR_NAMES))
        self.assertAlmostEqual(_weights_total(result), 1.0, places=12)
        self.assertAlmostEqual(result["ep"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(result["bp"], 1.0 / 3.0, places=12)
        self.assertEqual(result["momentum_60"], 0.0)
        self.assertEqual(result["size"], 0.0)

    def test_validate_weights_accepts_registry_weights(self) -> None:
        original = get_model(DEFAULT_MODEL_ID)["weights"]
        result = validate_weights(original)
        for factor in FACTOR_NAMES:
            self.assertAlmostEqual(result[factor], original[factor], places=12)

    def test_validate_weights_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights({"ep": -0.1, "bp": 0.5})

    def test_validate_weights_rejects_over_single_limit(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights({"ep": 0.5, "bp": 0.5})

    def test_validate_weights_rejects_normalized_overflow(self) -> None:
        # 归一化后 ep/bp 各 0.5，超过单因子上限 0.4
        with self.assertRaises(ValueError):
            validate_weights({"ep": 0.2, "bp": 0.2})

    def test_validate_weights_rejects_zero_total(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights({})
        with self.assertRaises(ValueError):
            validate_weights({"ep": 0})

    def test_validate_weights_rejects_unknown_factor(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights({"ep": 0.2, "unknown_factor": 0.2})

    def test_validate_weights_rejects_non_numeric(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights({"ep": "0.2"})

    def test_validate_weights_rejects_non_mapping(self) -> None:
        with self.assertRaises(ValueError):
            validate_weights([0.1] * 10)


class CustomCodeTests(unittest.TestCase):
    """AI 生成代码的静态校验与受限编译。"""

    def test_sklearn_custom_model_is_allowed(self) -> None:
        code = """import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_transform():
    def transform(z, weights):
        matrix = z.fillna(0.0).to_numpy(dtype=float)
        scaled = StandardScaler().fit_transform(matrix)
        score = scaled.mean(axis=1)
        return pd.Series(score, index=z.index)
    return transform"""
        info = validate_code(code)
        self.assertTrue(any("sklearn.preprocessing" in item for item in info["imports"]))

    def test_valid_code_passes(self) -> None:
        info = validate_code(VALID_CODE)
        self.assertTrue(info["ok"])
        check_names = [item["name"] for item in info["checks"]]
        self.assertEqual(
            check_names,
            ["source_size", "syntax", "imports", "ast_safety", "entrypoint", "smoke"],
        )
        for item in info["checks"]:
            self.assertEqual(item["status"], "passed")
            self.assertIsInstance(item["detail"], str)
        self.assertIn("numpy as np", info["imports"])
        self.assertIn("pandas as pd", info["imports"])
        self.assertIn("math", info["imports"])

    def test_compile_code_returns_working_transform(self) -> None:
        transform = compile_code(VALID_CODE)
        self.assertTrue(callable(transform))
        frame = _z_frame()
        weights = _equal_weights()
        output = transform(frame, weights)
        self.assertIsInstance(output, pd.Series)
        self.assertEqual(len(output), len(frame))
        values = output.to_numpy(dtype=float)
        self.assertTrue(np.all(np.isfinite(values)))
        expected = sum(
            weights[factor] * frame[f"z_{factor}"].fillna(0.0) for factor in FACTOR_NAMES
        )
        np.testing.assert_allclose(values, expected.to_numpy(dtype=float), rtol=1e-12)

    def test_import_os_rejected(self) -> None:
        code = "import os\n\n\ndef build_transform():\n    return None\n"
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(code)
        self.assertEqual(ctx.exception.checks[-1]["name"], "imports")
        self.assertIn("os", str(ctx.exception))

    def test_open_rejected(self) -> None:
        code = (
            "def build_transform():\n"
            "    def transform(z, weights):\n"
            "        return open('secret.txt')\n"
            "    return transform\n"
        )
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(code)
        self.assertEqual(ctx.exception.checks[-1]["name"], "ast_safety")
        self.assertIn("open", str(ctx.exception))

    def test_eval_rejected(self) -> None:
        code = (
            "def build_transform():\n"
            "    def transform(z, weights):\n"
            "        return eval('1 + 1')\n"
            "    return transform\n"
        )
        with self.assertRaises(ModelCodeError):
            validate_code(code)

    def test_dunder_attribute_rejected(self) -> None:
        code = (
            "def build_transform():\n"
            "    def transform(z, weights):\n"
            "        return z.__class__\n"
            "    return transform\n"
        )
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(code)
        # 非 self/cls 根的 dunder 属性仍然拦截（错误文案由 custom 决定，断言行为）
        self.assertEqual(ctx.exception.checks[-1]["name"], "ast_safety")
        self.assertIn("__class__", str(ctx.exception))

    def test_missing_build_transform_rejected(self) -> None:
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code("import math\n")
        self.assertEqual(ctx.exception.checks[-1]["name"], "entrypoint")
        self.assertIn("build_transform", str(ctx.exception))

    def test_build_transform_with_params_rejected(self) -> None:
        # 参数名必须是 params；其他名字（如 config）被拒绝
        code = "def build_transform(config):\n    return config\n"
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(code)
        self.assertEqual(ctx.exception.checks[-1]["name"], "entrypoint")
        self.assertIn("params", str(ctx.exception))

    def test_build_transform_with_two_params_rejected(self) -> None:
        code = "def build_transform(params, extra=None):\n    return params\n"
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(code)
        self.assertEqual(ctx.exception.checks[-1]["name"], "entrypoint")

    def test_parameterized_build_transform_allowed(self) -> None:
        info = validate_code(PARAM_CODE, params={"tilt": 0.2})
        self.assertTrue(info["ok"])
        self.assertTrue(info["takes_params"])
        self.assertEqual(
            [item["name"] for item in info["checks"]],
            ["source_size", "syntax", "imports", "ast_safety", "entrypoint", "smoke"],
        )
        # 无参数的旧写法继续通过
        self.assertFalse(validate_code(VALID_CODE)["takes_params"])

    def test_parameterized_engine_respects_params(self) -> None:
        frame = _z_frame()
        weights = _equal_weights()
        low = compile_code(PARAM_CODE, params={"tilt": 0.0})(frame, weights)
        high = compile_code(PARAM_CODE, params={"tilt": 2.0})(frame, weights)
        self.assertIsInstance(low, pd.Series)
        diff = (high - low).to_numpy(dtype=float)
        expected = 2.0 * frame["z_volatility_60"].fillna(0.0).to_numpy(dtype=float)
        np.testing.assert_allclose(diff, expected, rtol=1e-12)

    def test_compile_code_validate_false_skips_validation(self) -> None:
        with mock.patch(
            "backend.models.custom.validate_code",
            side_effect=AssertionError("validate=False 时不应重复校验"),
        ):
            transform = compile_code(
                PARAM_CODE, params={"tilt": 0.25}, validate=False
            )
            output = transform(_z_frame(), _equal_weights())
        self.assertIsInstance(output, pd.Series)
        self.assertTrue(np.all(np.isfinite(output.to_numpy(dtype=float))))

    def test_compile_code_validate_false_still_checks_syntax(self) -> None:
        with self.assertRaises(ModelCodeError):
            compile_code("def build_transform(:\n", validate=False)

    def test_params_are_passed_to_smoke(self) -> None:
        # 不传 params 时冒烟应失败；传了才通过 —— 证明参数真的进了 build_transform
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(REQUIRE_PARAMS_CODE)
        self.assertEqual(ctx.exception.checks[-1]["name"], "smoke")
        self.assertTrue(validate_code(REQUIRE_PARAMS_CODE, params={"boost": 0.5})["ok"])

    def test_validate_code_rejects_non_mapping_params(self) -> None:
        with self.assertRaises(ModelCodeError):
            validate_code(PARAM_CODE, params=[("tilt", 0.5)])  # type: ignore[arg-type]

    def test_build_transform_without_return_rejected(self) -> None:
        code = "def build_transform():\n    pass\n"
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(code)
        self.assertEqual(ctx.exception.checks[-1]["name"], "entrypoint")

    def test_syntax_error_rejected(self) -> None:
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code("def build_transform(:\n    pass\n")
        self.assertEqual(ctx.exception.checks[-1]["name"], "syntax")

    def test_source_too_long_rejected(self) -> None:
        # 源码上限默认已放宽，这里用环境变量把上限调小来验证 source_size 检查仍生效
        code = "def build_transform():\n    return None\n" + "# filler\n" * 1000
        with mock.patch.dict(os.environ, {"AI_MAX_SOURCE_CHARS": "1000"}):
            with self.assertRaises(ModelCodeError) as ctx:
                validate_code(code)
        self.assertEqual(ctx.exception.checks[-1]["name"], "source_size")

    def test_non_string_code_rejected(self) -> None:
        with self.assertRaises(ModelCodeError):
            validate_code(123)  # type: ignore[arg-type]

    def test_nan_return_rejected_by_smoke(self) -> None:
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(NAN_CODE)
        self.assertEqual(ctx.exception.checks[-1]["name"], "smoke")

    def test_non_series_return_rejected_by_smoke(self) -> None:
        with self.assertRaises(ModelCodeError) as ctx:
            validate_code(ARRAY_CODE)
        self.assertEqual(ctx.exception.checks[-1]["name"], "smoke")
        self.assertIn("pandas.Series", str(ctx.exception))

    def test_compile_code_rejects_invalid_code(self) -> None:
        with self.assertRaises(ModelCodeError):
            compile_code("import subprocess\n")

    def test_model_code_error_is_validation_error(self) -> None:
        self.assertTrue(issubclass(ModelCodeError, ModelValidationError))
        self.assertTrue(issubclass(ModelCodeError, ValueError))


class TuningPlanTests(unittest.TestCase):
    """调参方案：AI 只产出引擎 + 搜索范围，最终参数由服务端网格寻优。"""

    @staticmethod
    def _state() -> dict:
        model = get_model(DEFAULT_MODEL_ID)
        return {
            "current_model": model,
            "current_metrics": {
                "cagr": 0.11,
                "sharpe": 1.02,
                "mdd": -0.16,
                "ir": 2.1,
                "win_rate": 0.55,
            },
            "weights": model["weights"],
            "factor_stats": {"ep": {"mean": 0.0, "std": 1.0}, "nan_demo": float("nan")},
            "recent_metrics": [{"as_of": "2024-10-31", "sharpe": 0.9, "cagr": 0.08}],
        }

    @staticmethod
    def _search_space() -> dict:
        return {
            "volatility_60": {"min": 0.05, "max": 0.25, "step": 0.1},
            "roe": {"min": 0.1, "max": 0.3, "step": 0.1},
            "ep": {"min": 0.0, "max": 0.2, "step": 0.1},
        }

    @staticmethod
    def _param_space() -> dict:
        return {
            "tilt": {
                "min": 0,
                "max": 0.6,
                "step": 0.2,
                "type": "float",
                "default": 0.5,
            }
        }

    def _valid_raw(self, approach: str = "weights") -> dict:
        raw: dict = {
            "approach": approach,
            "summary": "在几个关键因子上做小范围搜索",
            "why": "最近回撤偏大，低波动因子值得多给一点权重",
            "expected_effect": "回撤和波动有望下降",
            "risks": ["收益可能略降"],
            "objective": "sharpe",
            "max_trials": 20,
        }
        if approach == "weights":
            raw["search_space"] = self._search_space()
        elif approach == "model_type":
            raw["model_type"] = "low_vol_quality"
            raw["search_space"] = self._search_space()
        else:
            raw["code"] = PARAM_CODE
            raw["param_space"] = self._param_space()
        return raw

    # ------------------------------------------------------------------
    # normalize_tuning_plan
    # ------------------------------------------------------------------
    def test_normalize_tuning_plan_weights_approach(self) -> None:
        base = {"ep": 0.2, "bp": 0.2, "roe": 0.2}
        plan = normalize_tuning_plan(self._valid_raw("weights"), base_weights=base)
        self.assertEqual(plan["approach"], "weights")
        self.assertIsNone(plan["model_type"])
        self.assertIsNone(plan["code"])
        self.assertEqual(plan["param_space"], {})
        self.assertEqual(
            list(plan["search_space"]), ["volatility_60", "roe", "ep"]
        )
        self.assertEqual(
            plan["search_space"]["ep"],
            {"min": 0.0, "max": 0.2, "step": 0.1},
        )
        self.assertEqual(plan["objective"], "sharpe")
        self.assertEqual(plan["max_trials"], 20)
        self.assertAlmostEqual(plan["weights"]["ep"], 1.0 / 3.0, places=12)
        self.assertEqual(plan["risks"], ["收益可能略降"])

    def test_normalize_tuning_plan_model_type_approach(self) -> None:
        plan = normalize_tuning_plan(self._valid_raw("model_type"))
        self.assertEqual(plan["model_type"], "low_vol_quality")
        _assert_weights_close(
            self, plan["weights"], get_model("low_vol_quality")["weights"]
        )
        self.assertEqual(list(plan["search_space"]), ["volatility_60", "roe", "ep"])
        self.assertEqual(plan["param_space"], {})

    def test_normalize_tuning_plan_code_approach(self) -> None:
        base = {"ep": 0.2, "bp": 0.2, "roe": 0.2}
        plan = normalize_tuning_plan(self._valid_raw("code"), base_weights=base)
        self.assertEqual(plan["approach"], "code")
        self.assertIsNone(plan["model_type"])
        self.assertEqual(plan["search_space"], {})
        self.assertEqual(plan["code"], PARAM_CODE.strip())
        self.assertEqual(
            plan["param_space"],
            {
                "tilt": {
                    "min": 0.0,
                    "max": 0.6,
                    "step": 0.2,
                    "type": "float",
                    "default": 0.5,
                }
            },
        )
        self.assertAlmostEqual(plan["weights"]["ep"], 1.0 / 3.0, places=12)

    def test_normalize_tuning_plan_int_param_normalized(self) -> None:
        raw = self._valid_raw("code")
        raw["param_space"] = {
            "tilt": self._param_space()["tilt"],
            "rounds": {"min": 1, "max": 5, "step": 1, "type": "int"},
        }
        plan = normalize_tuning_plan(raw)
        self.assertEqual(
            plan["param_space"]["rounds"],
            {"min": 1, "max": 5, "step": 1, "type": "int", "default": 1},
        )

    def test_normalize_tuning_plan_uses_current_model_weights(self) -> None:
        raw = self._valid_raw("weights")
        plan = normalize_tuning_plan(raw, current_model="low_vol_quality")
        _assert_weights_close(
            self, plan["weights"], get_model("low_vol_quality")["weights"]
        )
        plan = normalize_tuning_plan(
            raw, current_model=get_model("value_dividend")
        )
        _assert_weights_close(
            self, plan["weights"], get_model("value_dividend")["weights"]
        )

    def test_normalize_tuning_plan_result_keys(self) -> None:
        plan = normalize_tuning_plan(self._valid_raw("weights"))
        self.assertEqual(
            set(plan),
            {
                "approach",
                "summary",
                "model_type",
                "weights",
                "code",
                "search_space",
                "param_space",
                "objective",
                "max_trials",
                "why",
                "expected_effect",
                "risks",
            },
        )

    def test_normalize_tuning_plan_objective_and_max_trials(self) -> None:
        raw = self._valid_raw("weights")
        raw["objective"] = "CAGR"
        raw["max_trials"] = 12.0
        plan = normalize_tuning_plan(raw)
        self.assertEqual(plan["objective"], "cagr")
        self.assertEqual(plan["max_trials"], 12)

        raw = self._valid_raw("weights")
        del raw["objective"]
        del raw["max_trials"]
        plan = normalize_tuning_plan(raw)
        self.assertEqual(plan["objective"], "sharpe")
        self.assertEqual(plan["max_trials"], DEFAULT_TUNING_MAX_TRIALS)
        self.assertEqual(plan["max_trials"], 32)

        # 归一化不 clamp 上限：大预算留给服务端按上限限制；
        # 下限只要 >=1 即可（不再要求 8 起）
        raw = self._valid_raw("weights")
        raw["max_trials"] = 100
        self.assertEqual(normalize_tuning_plan(raw)["max_trials"], 100)
        for value in (1, 7):
            raw = self._valid_raw("weights")
            raw["max_trials"] = value
            self.assertEqual(normalize_tuning_plan(raw)["max_trials"], value)

        for bad in (
            {"objective": "profit"},
            {"objective": 1},
            {"max_trials": "12"},
            {"max_trials": 0},
            {"max_trials": -3},
            {"max_trials": 12.5},
        ):
            raw = self._valid_raw("weights")
            raw.update(bad)
            with self.subTest(bad=bad), self.assertRaises(ModelValidationError):
                normalize_tuning_plan(raw)

    def test_normalize_tuning_plan_accepts_new_objectives(self) -> None:
        for objective in OBJECTIVES:
            raw = self._valid_raw("weights")
            raw["objective"] = objective.upper()
            with self.subTest(objective=objective):
                self.assertEqual(
                    normalize_tuning_plan(raw)["objective"], objective)

    def test_normalize_tuning_plan_accepts_single_factor(self) -> None:
        # 至少 1 个因子（不再强制 3 个）
        raw = self._valid_raw("weights")
        raw["search_space"] = {
            "volatility_60": {"min": 0.05, "max": 0.25, "step": 0.1},
        }
        plan = normalize_tuning_plan(raw)
        self.assertEqual(list(plan["search_space"]), ["volatility_60"])
        self.assertEqual(MIN_SEARCH_FACTORS, 1)

    def test_normalize_tuning_plan_accepts_more_than_four_params(self) -> None:
        # 参数个数不再有 4 个硬上限
        raw = self._valid_raw("code")
        raw["param_space"] = {
            f"p{index}": {"min": 0.0, "max": 0.4, "step": 0.2}
            for index in range(6)
        }
        plan = normalize_tuning_plan(raw)
        self.assertEqual(len(plan["param_space"]), 6)
        self.assertEqual(list(plan["param_space"]),
                         [f"p{index}" for index in range(6)])

    def test_normalize_tuning_plan_rejects_bad_search_space(self) -> None:
        valid = self._search_space()
        cases = {
            "missing": None,
            "empty": {},
            "unknown_factor": {
                **valid,
                "unknown_factor": {"min": 0.0, "max": 0.1, "step": 0.1},
            },
            "min_gt_max": {
                "volatility_60": {"min": 0.3, "max": 0.1, "step": 0.1},
                "roe": valid["roe"],
                "ep": valid["ep"],
            },
            "step_zero": {
                "volatility_60": {"min": 0.0, "max": 0.2, "step": 0},
                "roe": valid["roe"],
                "ep": valid["ep"],
            },
            "over_max": {
                "volatility_60": {"min": 0.1, "max": 0.5, "step": 0.1},
                "roe": valid["roe"],
                "ep": valid["ep"],
            },
            "negative_min": {
                "volatility_60": {"min": -0.05, "max": 0.2, "step": 0.1},
                "roe": valid["roe"],
                "ep": valid["ep"],
            },
            "single_level": {
                "volatility_60": {"min": 0.1, "max": 0.1, "step": 0.1},
                "roe": valid["roe"],
                "ep": valid["ep"],
            },
            "not_mapping": "oops",
        }
        for label, search_space in cases.items():
            raw = self._valid_raw("weights")
            if search_space is None:
                del raw["search_space"]
            else:
                raw["search_space"] = search_space
            with self.subTest(case=label), self.assertRaises(ModelValidationError):
                normalize_tuning_plan(raw)

    def test_normalize_tuning_plan_rejects_bad_param_space(self) -> None:
        tilt = self._param_space()["tilt"]
        cases = {
            "missing": None,
            "empty": {},
            "bad_name": {"1bad": tilt},
            "keyword": {"class": tilt},
            "min_gt_max": {"tilt": {"min": 0.5, "max": 0.1, "step": 0.1}},
            "step_zero": {"tilt": {"min": 0.0, "max": 0.5, "step": 0}},
            "single_level": {"tilt": {"min": 0.2, "max": 0.2, "step": 0.1}},
            "bad_type": {"tilt": {**tilt, "type": "string"}},
            "default_out_of_range": {"tilt": {**tilt, "default": 0.9}},
            "int_not_integer": {
                "count": {"min": 1, "max": 5, "step": 0.5, "type": "int"}
            },
        }
        for label, param_space in cases.items():
            raw = self._valid_raw("code")
            if param_space is None:
                del raw["param_space"]
            else:
                raw["param_space"] = param_space
            with self.subTest(case=label), self.assertRaises(ModelValidationError):
                normalize_tuning_plan(raw)

    def test_normalize_tuning_plan_rejects_bad_code(self) -> None:
        raw = self._valid_raw("code")
        raw["code"] = "import os\n\n\ndef build_transform(params=None):\n    return None\n"
        with self.assertRaises(ModelCodeError):
            normalize_tuning_plan(raw)

        raw = self._valid_raw("code")
        raw["code"] = VALID_CODE  # 旧写法：不接收 params
        with self.assertRaises(ModelValidationError) as ctx:
            normalize_tuning_plan(raw)
        self.assertIn("params", str(ctx.exception))

    def test_normalize_tuning_plan_rejects_mismatched_fields(self) -> None:
        raw = self._valid_raw("code")
        raw["search_space"] = self._search_space()
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan(raw)

        raw = self._valid_raw("weights")
        raw["param_space"] = self._param_space()
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan(raw)

    def test_normalize_tuning_plan_required_fields(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan({"approach": "weights"})  # 缺 summary/search_space
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan({"approach": "magic", "summary": "x"})
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan(["not", "a", "dict"])  # type: ignore[arg-type]

        raw = self._valid_raw("model_type")
        del raw["model_type"]
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan(raw)

        raw = self._valid_raw("model_type")
        raw["model_type"] = "not_a_model"
        with self.assertRaises(ModelValidationError):
            normalize_tuning_plan(raw)

    # ------------------------------------------------------------------
    # build_tuning_plan
    # ------------------------------------------------------------------
    def test_fake_llm_tuning_plan_matches_contract(self) -> None:
        raw = ScriptedLLM().chat_json(
            TUNING_SYSTEM,
            [{"role": "user", "content": json.dumps({"tuning_request": True})}],
        )
        # ScriptedLLM 的 auto 分支可能产出 weights / model_type（code 仅兼容旧路径）
        self.assertIn(raw["approach"], {"weights", "model_type", "code"})
        self.assertIn(raw["objective"], OBJECTIVES)
        self.assertIsInstance(raw["max_trials"], int)
        self.assertGreaterEqual(raw["max_trials"], 1)
        for key in ("summary", "why", "risks"):
            self.assertTrue(raw.get(key))
        plan = normalize_tuning_plan(raw)
        self.assertEqual(plan["approach"], raw["approach"])
        self.assertIn(plan["objective"], OBJECTIVES)
        self.assertGreaterEqual(plan["max_trials"], 1)

    def test_build_tuning_plan_with_fake_llm(self) -> None:
        llm = FakeProposalLLM(self._valid_raw("weights"))
        plan = build_tuning_plan(llm, self._state())
        self.assertEqual(plan["approach"], "weights")
        self.assertEqual(plan["max_trials"], 20)
        self.assertEqual(plan["objective"], "sharpe")
        self.assertEqual(plan["repair_attempts"], 0)
        self.assertEqual(len(llm.calls), 1)

    def test_build_tuning_plan_prompt_and_state(self) -> None:
        llm = FakeProposalLLM(self._valid_raw("weights"))
        plan = build_tuning_plan(llm, self._state())
        self.assertEqual(plan["approach"], "weights")
        self.assertEqual(plan["repair_attempts"], 0)
        self.assertEqual(len(llm.calls), 1)
        system, messages = llm.calls[0]
        self.assertEqual(system, TUNING_SYSTEM)
        self.assertIn("网格", TUNING_SYSTEM)
        self.assertEqual(len(messages), 1)
        payload = json.loads(messages[0]["content"])
        self.assertTrue(payload["tuning_request"])
        self.assertIn("current_metrics", payload)

    def test_tuning_prompt_has_no_demo_limits(self) -> None:
        # 演示型硬限制已删除：参数个数、因子数、trials 区间都不再写死
        self.assertNotIn("1~4", TUNING_SYSTEM)
        self.assertNotIn("1~4", TUNING_REPAIR_SYSTEM)
        self.assertNotIn("8~48", TUNING_SYSTEM)
        self.assertNotIn("至少 3 个因子", TUNING_SYSTEM)
        self.assertNotIn("至少 3 个已知因子", TUNING_REPAIR_SYSTEM)
        self.assertIn("至少 1 个因子", TUNING_SYSTEM)
        for objective in OBJECTIVES:
            self.assertIn(objective, TUNING_SYSTEM)
        self.assertIn("write_model_program", TUNING_SYSTEM)
        self.assertIn("write_model_program", TUNING_REPAIR_SYSTEM)

    def test_tuning_constants_and_env_defaults(self) -> None:
        self.assertEqual(DEFAULT_TUNING_MAX_TRIALS, 32)
        self.assertEqual(MIN_SEARCH_FACTORS, 1)
        self.assertIsNone(MAX_TUNING_PARAMS)
        if "TUNING_STATE_MAX_CHARS" not in os.environ:
            self.assertEqual(MAX_STATE_CHARS, 60000)
        if "TUNING_MAX_REPAIR_ATTEMPTS" not in os.environ:
            self.assertEqual(MAX_TUNING_REPAIR_ATTEMPTS, 3)

    def test_build_tuning_plan_auto_repair(self) -> None:
        bad = self._valid_raw("weights")
        del bad["search_space"]  # 缺必填字段，必须触发一次自动修复
        llm = SequenceLLM([bad, self._valid_raw("weights")])
        plan = build_tuning_plan(llm, self._state())
        self.assertEqual(plan["repair_attempts"], 1)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1][0], TUNING_REPAIR_SYSTEM)
        repair_payload = json.loads(llm.calls[1][1][0]["content"])
        self.assertIn("error", repair_payload)
        self.assertIn("search_space", repair_payload["error"])

    def test_build_tuning_plan_repair_exhausted(self) -> None:
        bad = self._valid_raw("weights")
        del bad["search_space"]
        llm = SequenceLLM([bad, bad, bad, bad])
        with self.assertRaises(ModelValidationError) as ctx:
            build_tuning_plan(llm, self._state())
        self.assertIn(f"自动修复 {MAX_TUNING_REPAIR_ATTEMPTS} 次", str(ctx.exception))
        self.assertEqual(len(llm.calls), MAX_TUNING_REPAIR_ATTEMPTS + 1)

    def test_build_tuning_plan_repair_attempts_env_override(self) -> None:
        bad = self._valid_raw("weights")
        del bad["search_space"]
        llm = SequenceLLM([bad, bad, bad])
        with mock.patch.dict(os.environ, {"TUNING_MAX_REPAIR_ATTEMPTS": "1"}):
            with self.assertRaises(ModelValidationError) as ctx:
                build_tuning_plan(llm, self._state())
        self.assertIn("自动修复 1 次", str(ctx.exception))
        self.assertEqual(len(llm.calls), 2)

    def test_build_tuning_plan_state_chars_env_override(self) -> None:
        llm = FakeProposalLLM(self._valid_raw("weights"))
        with mock.patch.dict(os.environ, {"TUNING_STATE_MAX_CHARS": "60"}):
            build_tuning_plan(llm, self._state())
        content = llm.calls[0][1][0]["content"]
        self.assertTrue(content.endswith("内容过长已截断）"))
        self.assertLessEqual(len(content), 80)

    def test_build_tuning_plan_rejects_code_directive(self) -> None:
        # engine=code 已迁移到 write_model_program：调参方案直接报错，不调用 LLM
        state = self._state()
        state["directive"] = {"engine": "code"}
        llm = FakeProposalLLM(self._valid_raw("weights"))
        with self.assertRaises(ModelValidationError) as ctx:
            build_tuning_plan(llm, state)
        self.assertIn("write_model_program", str(ctx.exception))
        self.assertEqual(llm.calls, [])

    def test_build_tuning_plan_rejects_structural_escalate(self) -> None:
        # 结构性升级（重写打分引擎）同样交给 write_model_program
        state = self._state()
        state["directive"] = {"engine": "auto", "escalate": True,
                              "previous_approach": "model_type"}
        llm = FakeProposalLLM(self._valid_raw("weights"))
        with self.assertRaises(ModelValidationError) as ctx:
            build_tuning_plan(llm, state)
        self.assertIn("write_model_program", str(ctx.exception))
        self.assertEqual(llm.calls, [])

    def test_build_tuning_plan_auto_rejects_code_plan(self) -> None:
        # auto 路径若 LLM 仍给 code 方案：拒绝并提示 write_model_program，修复为 weights 后成功
        code_raw = self._valid_raw("code")
        llm = SequenceLLM([code_raw, self._valid_raw("weights")])
        plan = build_tuning_plan(llm, self._state())
        self.assertEqual(plan["approach"], "weights")
        self.assertEqual(plan["repair_attempts"], 1)
        self.assertEqual(len(llm.calls), 2)
        repair_payload = json.loads(llm.calls[1][1][0]["content"])
        self.assertIn("write_model_program", repair_payload["error"])

    def test_build_tuning_plan_escalate_from_weights_requires_model_type(self) -> None:
        state = self._state()
        state["directive"] = {"engine": "auto", "escalate": True,
                              "previous_approach": "weights"}
        llm = SequenceLLM([self._valid_raw("weights")])
        with self.assertRaises(ModelValidationError) as ctx:
            build_tuning_plan(llm, state)
        self.assertIn("model_type", str(ctx.exception))
        self.assertEqual(len(llm.calls), MAX_TUNING_REPAIR_ATTEMPTS + 1)

    def test_build_tuning_plan_rejects_non_mapping_state(self) -> None:
        with self.assertRaises(ModelValidationError):
            build_tuning_plan(ScriptedLLM(), ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_build_tuning_plan_invalid_state_weights(self) -> None:
        state = self._state()
        state["weights"] = {"ep": -1}
        llm = FakeProposalLLM(self._valid_raw("weights"))
        with self.assertRaises(ModelValidationError):
            build_tuning_plan(llm, state)
        self.assertEqual(llm.calls, [])

    def test_fake_llm_planner_behavior_unchanged(self) -> None:
        plan = ScriptedLLM().chat_json(
            "planner",
            [
                {
                    "role": "user",
                    "content": json.dumps({"user_message": "你好", "state": {}}),
                }
            ],
        )
        self.assertIn("reply", plan)
        self.assertIn("actions", plan)


if __name__ == "__main__":
    unittest.main()
