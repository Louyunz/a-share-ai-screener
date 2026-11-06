# -*- coding: utf-8 -*-
"""AI 建模流水线单元测试：registry / custom 代码安全 / pipeline。

全部使用 unittest，不依赖 backend.agent、backend.service 或真实 LLM。
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from backend.llm import FakeLLM
from backend.models import (
    DEFAULT_MODEL_ID,
    MODEL_TYPES,
    ModelCodeError,
    ModelType,
    ModelValidationError,
    build_candidate_proposal,
    compile_code,
    describe_model,
    get_model,
    list_models,
    normalize_proposal,
    validate_code,
    validate_weights,
)
from backend.models.pipeline import (
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
        self.assertIn("下划线属性", str(ctx.exception))

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
        code = "def build_transform():\n    return None\n" + "# filler\n" * 1000
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


class PipelineTests(unittest.TestCase):
    """提案校验与 LLM 提案生成。"""

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

    def test_code_repair_retries_after_validation_error(self) -> None:
        class _RepairLLM:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, messages):
                self.calls += 1
                if self.calls == 1:
                    return {"approach": "code", "summary": "无效代码",
                            "code": "def build_transform():\n    return 1\n"}
                return {"approach": "code", "summary": "修复后的模型",
                        "code": VALID_CODE, "risks": ["历史回测不代表未来表现"]}

        llm = _RepairLLM()
        proposal = build_candidate_proposal(llm, self._state())
        self.assertEqual(llm.calls, 2)
        self.assertEqual(proposal.get("repair_attempts"), 1)
        self.assertEqual(proposal["approach"], "code")
        self.assertTrue(proposal.get("code"))

    def test_normalize_model_type_approach(self) -> None:
        proposal = normalize_proposal(
            {
                "approach": "model_type",
                "summary": "换成更看重低波和质量的模型",
                "model_type": "low_vol_quality",
                "why": "最近回撤偏大",
                "expected_effect": "回撤变小",
                "risks": ["收益可能变低"],
            }
        )
        self.assertEqual(proposal["approach"], "model_type")
        self.assertEqual(proposal["model_type"], "low_vol_quality")
        _assert_weights_close(
            self, proposal["weights"], get_model("low_vol_quality")["weights"]
        )
        self.assertIsNone(proposal["code"])
        self.assertEqual(proposal["risks"], ["收益可能变低"])
        self.assertAlmostEqual(_weights_total(proposal["weights"]), 1.0, places=12)

    def test_normalize_weights_approach(self) -> None:
        proposal = normalize_proposal(
            {
                "approach": "weights",
                "summary": "把权重多分给赚钱能力",
                "weights": {"roe": 0.2, "ocf_to_profit": 0.2, "ep": 0.1},
            }
        )
        self.assertEqual(proposal["approach"], "weights")
        self.assertIsNone(proposal["model_type"])
        self.assertIsNone(proposal["code"])
        self.assertAlmostEqual(_weights_total(proposal["weights"]), 1.0, places=12)
        self.assertAlmostEqual(proposal["weights"]["roe"], 0.4, places=12)
        self.assertAlmostEqual(proposal["weights"]["ocf_to_profit"], 0.4, places=12)
        self.assertAlmostEqual(proposal["weights"]["ep"], 0.2, places=12)

    def test_normalize_code_approach_uses_base_weights(self) -> None:
        base = {"ep": 0.2, "bp": 0.2, "roe": 0.2}
        proposal = normalize_proposal(
            {"approach": "code", "summary": "自定义打分", "code": VALID_CODE},
            base_weights=base,
        )
        self.assertEqual(proposal["approach"], "code")
        self.assertIsNone(proposal["model_type"])
        self.assertEqual(proposal["code"], VALID_CODE.strip())
        self.assertAlmostEqual(proposal["weights"]["ep"], 1.0 / 3.0, places=12)
        self.assertEqual(proposal["weights"]["momentum_60"], 0.0)

    def test_normalize_code_approach_uses_named_model_weights(self) -> None:
        proposal = normalize_proposal(
            {
                "approach": "code",
                "summary": "自定义打分",
                "model_type": "value_dividend",
                "code": VALID_CODE,
            }
        )
        _assert_weights_close(
            self, proposal["weights"], get_model("value_dividend")["weights"]
        )
        self.assertIsNone(proposal["model_type"])

    def test_normalize_code_approach_defaults_to_current_model(self) -> None:
        proposal = normalize_proposal(
            {"approach": "code", "summary": "自定义打分", "code": VALID_CODE}
        )
        _assert_weights_close(
            self, proposal["weights"], get_model(DEFAULT_MODEL_ID)["weights"]
        )

    def test_normalize_code_approach_rejects_bad_code(self) -> None:
        code = VALID_CODE.replace("import pandas as pd", "import os")
        with self.assertRaises(ModelCodeError):
            normalize_proposal(
                {"approach": "code", "summary": "自定义打分", "code": code}
            )

    def test_normalize_rejects_bad_approach(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal({"approach": "magic", "summary": "x"})

    def test_normalize_requires_summary(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal({"approach": "model_type", "model_type": "multi_factor"})
        with self.assertRaises(ModelValidationError):
            normalize_proposal({"approach": "model_type", "model_type": "multi_factor", "summary": "  "})

    def test_normalize_rejects_unknown_model_type(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal(
                {"approach": "model_type", "model_type": "not_a_model", "summary": "x"}
            )

    def test_normalize_model_type_requires_model_type(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal({"approach": "model_type", "summary": "x"})

    def test_normalize_weights_requires_weights(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal({"approach": "weights", "summary": "x"})

    def test_normalize_rejects_invalid_weights(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal(
                {"approach": "weights", "summary": "x", "weights": {"ep": -1}}
            )

    def test_normalize_rejects_code_on_other_approaches(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal(
                {
                    "approach": "weights",
                    "summary": "x",
                    "weights": {"ep": 0.2, "bp": 0.2, "roe": 0.2},
                    "code": "print(1)",
                }
            )

    def test_normalize_rejects_non_mapping(self) -> None:
        with self.assertRaises(ModelValidationError):
            normalize_proposal(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_normalize_risks_accepts_string(self) -> None:
        proposal = normalize_proposal(
            {"approach": "model_type", "model_type": "contrarian", "summary": "x", "risks": "波动可能变大"}
        )
        self.assertEqual(proposal["risks"], ["波动可能变大"])

    def test_normalize_result_keys(self) -> None:
        proposal = normalize_proposal(
            {"approach": "model_type", "model_type": "momentum_growth", "summary": "x"}
        )
        self.assertEqual(
            set(proposal),
            {
                "approach",
                "summary",
                "model_type",
                "weights",
                "code",
                "why",
                "expected_effect",
                "risks",
            },
        )

    def test_build_candidate_proposal_success(self) -> None:
        response = {
            "approach": "model_type",
            "summary": "换成更稳的低波质量模型",
            "model_type": "low_vol_quality",
            "why": "最近回撤变大，夏普从 1.02 降到 0.90",
            "expected_effect": "回撤变小",
            "risks": ["上涨时可能跟不上"],
        }
        llm = FakeProposalLLM(response)
        proposal = build_candidate_proposal(llm, self._state())

        self.assertEqual(proposal["model_type"], "low_vol_quality")
        self.assertEqual(proposal["summary"], response["summary"])
        self.assertEqual(len(llm.calls), 1)
        system, messages = llm.calls[0]
        self.assertIn("量化建模助手", system)
        self.assertIn("build_transform", system)
        self.assertIn("multi_factor", system)
        for approach in ("model_type", "weights", "code"):
            self.assertIn(approach, system)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("current_metrics", messages[0]["content"])
        # NaN 指标应被序列化为 null，提示词仍是合法 JSON 文本
        self.assertIn("null", messages[0]["content"])

    def test_build_candidate_proposal_code_approach(self) -> None:
        response = {
            "approach": "code",
            "summary": "用自定义打分照顾多个信号",
            "code": VALID_CODE,
            "why": "当前因子贡献不均衡",
        }
        llm = FakeProposalLLM(response)
        state = self._state()
        proposal = build_candidate_proposal(llm, state)
        self.assertEqual(proposal["approach"], "code")
        self.assertEqual(proposal["code"], VALID_CODE.strip())
        _assert_weights_close(self, proposal["weights"], state["weights"])

    def test_build_candidate_proposal_bad_json_raises(self) -> None:
        llm = FakeProposalLLM(error=ValueError("模型未返回 JSON：<html>oops</html>"))
        with self.assertRaises(ModelValidationError) as ctx:
            build_candidate_proposal(llm, self._state())
        self.assertIn("oops", str(ctx.exception))

    def test_build_candidate_proposal_non_mapping_raises(self) -> None:
        llm = FakeProposalLLM(response=["not", "an", "object"])
        with self.assertRaises(ModelValidationError) as ctx:
            build_candidate_proposal(llm, self._state())
        self.assertIn("不是 JSON 对象", str(ctx.exception))
        self.assertIn("not", str(ctx.exception))

    def test_build_candidate_proposal_invalid_proposal_raises(self) -> None:
        llm = FakeProposalLLM(
            {"approach": "model_type", "model_type": "not_a_model", "summary": "x"}
        )
        with self.assertRaises(ModelValidationError) as ctx:
            build_candidate_proposal(llm, self._state())
        message = str(ctx.exception)
        self.assertIn("not_a_model", message)
        self.assertIn("原始返回", message)

    def test_build_candidate_proposal_missing_state_key(self) -> None:
        state = self._state()
        del state["factor_stats"]
        llm = FakeProposalLLM({"approach": "model_type", "model_type": "contrarian", "summary": "x"})
        with self.assertRaises(ModelValidationError) as ctx:
            build_candidate_proposal(llm, state)
        self.assertIn("factor_stats", str(ctx.exception))
        self.assertEqual(llm.calls, [])

    def test_build_candidate_proposal_invalid_state_weights(self) -> None:
        state = self._state()
        state["weights"] = {"ep": -1}
        llm = FakeProposalLLM({"approach": "model_type", "model_type": "contrarian", "summary": "x"})
        with self.assertRaises(ModelValidationError):
            build_candidate_proposal(llm, state)

    def test_describe_model(self) -> None:
        text = describe_model(get_model("low_vol_quality"))
        self.assertIn("低波质量", text)
        self.assertIn("重点看", text)
        self.assertEqual(text, describe_model("low_vol_quality"))

    def test_describe_model_unknown(self) -> None:
        with self.assertRaises(ValueError):
            describe_model("not_a_model")

    def test_describe_model_invalid_type(self) -> None:
        with self.assertRaises(ModelValidationError):
            describe_model(123)  # type: ignore[arg-type]


class TuningPlanTests(unittest.TestCase):
    """调参方案：AI 只产出引擎 + 搜索范围，最终参数由服务端网格寻优。"""

    @staticmethod
    def _state() -> dict:
        return PipelineTests._state()

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
        self.assertEqual(plan["max_trials"], 12)

        # 归一化不 clamp：跨出 8~48 的值留给服务端限制
        raw = self._valid_raw("weights")
        raw["max_trials"] = 100
        self.assertEqual(normalize_tuning_plan(raw)["max_trials"], 100)

        for bad in (
            {"objective": "profit"},
            {"objective": 1},
            {"max_trials": "12"},
            {"max_trials": 0},
            {"max_trials": 12.5},
        ):
            raw = self._valid_raw("weights")
            raw.update(bad)
            with self.subTest(bad=bad), self.assertRaises(ModelValidationError):
                normalize_tuning_plan(raw)

    def test_normalize_tuning_plan_rejects_bad_search_space(self) -> None:
        valid = self._search_space()
        cases = {
            "missing": None,
            "fewer_than_3": {
                "volatility_60": valid["volatility_60"],
                "roe": valid["roe"],
            },
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
            "too_many": {
                f"p{index}": {"min": 0.0, "max": 0.1, "step": 0.1}
                for index in range(5)
            },
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
        raw = FakeLLM().chat_json(
            TUNING_SYSTEM,
            [{"role": "user", "content": json.dumps({"tuning_request": True})}],
        )
        self.assertEqual(raw["approach"], "code")
        self.assertIn("build_transform(params=None)", raw["code"])
        self.assertIn("tilt", raw["param_space"])
        self.assertEqual(raw["param_space"]["tilt"]["default"], 0.5)
        self.assertEqual(raw["objective"], "sharpe")
        self.assertEqual(raw["max_trials"], 12)
        for key in ("summary", "why", "risks"):
            self.assertTrue(raw.get(key))
        plan = normalize_tuning_plan(raw)
        self.assertEqual(plan["approach"], "code")
        self.assertIn("tilt", plan["param_space"])

    def test_build_tuning_plan_with_fake_llm(self) -> None:
        plan = build_tuning_plan(FakeLLM(), self._state())
        self.assertEqual(plan["approach"], "code")
        self.assertEqual(plan["max_trials"], 12)
        self.assertEqual(plan["objective"], "sharpe")
        self.assertEqual(plan["repair_attempts"], 0)
        self.assertEqual(plan["param_space"]["tilt"]["default"], 0.5)
        self.assertIn("build_transform(params=None)", plan["code"])

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

    def test_build_tuning_plan_auto_repair(self) -> None:
        bad = self._valid_raw("weights")
        bad["search_space"] = {"volatility_60": {"min": 0.05, "max": 0.25, "step": 0.1}}
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
        bad["search_space"] = {"volatility_60": {"min": 0.05, "max": 0.25, "step": 0.1}}
        llm = SequenceLLM([bad, bad, bad])
        with self.assertRaises(ModelValidationError) as ctx:
            build_tuning_plan(llm, self._state())
        self.assertIn("自动修复 2 次", str(ctx.exception))
        self.assertEqual(len(llm.calls), 3)

    def test_build_tuning_plan_rejects_non_mapping_state(self) -> None:
        with self.assertRaises(ModelValidationError):
            build_tuning_plan(FakeLLM(), ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_build_tuning_plan_invalid_state_weights(self) -> None:
        state = self._state()
        state["weights"] = {"ep": -1}
        llm = FakeProposalLLM(self._valid_raw("weights"))
        with self.assertRaises(ModelValidationError):
            build_tuning_plan(llm, state)
        self.assertEqual(llm.calls, [])

    def test_fake_llm_planner_behavior_unchanged(self) -> None:
        plan = FakeLLM().chat_json(
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
