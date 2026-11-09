# -*- coding: utf-8 -*-
"""AI 多文件模型程序：契约校验、受限编译、第三方库申请等单元测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from backend.models import custom
from tests.support.scripted_llm import (
    SCRIPTED_PROGRAM_FILES,
    SCRIPTED_PROGRAM_PARAM_SPACE,
    SCRIPTED_REJECT_MARKER,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor_score": [0.1, 0.9, 0.5],
            "z_volatility_60": [0.2, -0.1, 0.0],
            "z_ep": [0.1, 0.2, -0.1],
        },
        index=["600000.SH", "000001.SZ", "600519.SH"],
    )


class ProgramValidationTest(unittest.TestCase):
    def test_valid_multi_file_program(self) -> None:
        info = custom.validate_program(SCRIPTED_PROGRAM_FILES, "main.py")
        self.assertTrue(info["ok"])
        self.assertEqual(info["files"], ["helpers.py", "main.py"])
        self.assertTrue(info["takes_params"])

    def test_forbidden_import_rejected(self) -> None:
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {"main.py": "import os\n\ndef build_transform():\n    return None\n"})
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {"main.py": (
                    "import scipy\n"
                    "import socket\n"
                    "def build_transform():\n    return None\n")},
                extra_roots=["scipy"],
            )

    def test_missing_entry_rejected(self) -> None:
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program({"helpers.py": "def helper():\n    return 1\n"},
                                    "main.py")

    def test_cycle_import_rejected(self) -> None:
        files = {
            "main.py": (
                "import helpers\n"
                "def build_transform():\n"
                "    return helpers.transform\n"),
            "helpers.py": (
                "import main\n"
                "def transform(z, weights):\n"
                "    return z.iloc[:, 0]\n"),
        }
        with self.assertRaises(custom.ModelCodeError) as context:
            custom.compile_program(files, "main.py", {})
        self.assertIn("循环导入", str(context.exception))

    def test_stdlib_filename_rejected(self) -> None:
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {"json.py": "def build_transform():\n    return None\n"}, "json.py")

    def test_rejected_marker_is_just_comment(self) -> None:
        files = dict(SCRIPTED_PROGRAM_FILES)
        files["main.py"] += f"\n# {SCRIPTED_REJECT_MARKER}\n"
        info = custom.validate_program(files, "main.py")
        self.assertTrue(info["ok"])

    def test_variable_named_code_is_allowed(self) -> None:
        files = {"main.py": (
            "import pandas as pd\n"
            "\n"
            "def build_transform(params=None):\n"
            "    code = 1\n"
            "    def transform(z, weights):\n"
            "        code = pd.Series(0.0, index=z.index, dtype=float)\n"
            "        return code\n"
            "    return transform\n")}
        info = custom.validate_program(files, "main.py")
        self.assertTrue(info["ok"])
        runtime = custom.compile_program(files, "main.py", {})
        frame = pd.DataFrame({"z_ep": [0.1]}, index=["600000.SH"])
        output = runtime.transform(frame, {"ep": 1.0})
        self.assertEqual(list(output), [0.0])

    def test_dangerous_call_still_rejected(self) -> None:
        files = {"main.py": (
            "def build_transform():\n"
            "    exec('print(1)')\n"
            "    def transform(z, weights):\n"
            "        return z.iloc[:, 0]\n"
            "    return transform\n")}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(files, "main.py")

    def test_compile_transform_and_select_hook(self) -> None:
        runtime = custom.compile_program(
            SCRIPTED_PROGRAM_FILES, "main.py", {"tilt": 0.5})
        frame = pd.DataFrame(
            {f"z_{name}": [0.1, -0.2] for name in (
                "ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
                "momentum_60", "reversal_5", "volatility_60", "liquidity_20", "size")},
            index=["600000.SH", "000001.SZ"],
        )
        weights = {name: 0.1 for name in (
            "ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
            "momentum_60", "reversal_5", "volatility_60", "liquidity_20", "size")}
        output = runtime.transform(frame, weights)
        self.assertEqual(len(output), 2)
        selected = runtime.select(_scores(), 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], "000001.SZ")

    def test_package_specs_and_roots(self) -> None:
        self.assertEqual(
            custom.normalize_package_specs(["xgboost>=2.0", "scikit-learn"]),
            ["xgboost>=2.0", "scikit-learn"])
        self.assertEqual(
            custom.package_import_roots(["xgboost>=2.0", "scikit-learn"]),
            ["sklearn", "xgboost"])
        for bad in ("../../etc/passwd", "https://example.com/a.whl",
                    "git+https://github.com/x/y", "pip"):
            with self.assertRaises(custom.ModelCodeError):
                custom.normalize_package_specs([bad])
        with self.assertRaises(custom.ModelCodeError):
            custom.normalize_package_specs(["a", "b", "c", "d", "e"])


if __name__ == "__main__":
    unittest.main()
