# -*- coding: utf-8 -*-
"""AI 多文件模型程序：契约校验、受限编译、第三方库申请等单元测试。"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import pandas as pd

from backend.models import custom
from tests.support.scripted_llm import (
    SCRIPTED_PROGRAM_FILES,
    SCRIPTED_PROGRAM_PARAM_SPACE,
    SCRIPTED_REJECT_MARKER,
)

_FACTOR_NAMES = (
    "ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
    "momentum_60", "reversal_5", "volatility_60", "liquidity_20", "size",
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


def _z_frame(rows: int = 3) -> pd.DataFrame:
    """10 个 z_* 因子列，供程序 transform 使用。"""
    data = {
        f"z_{name}": [round(0.1 * (row + position) - 0.2, 6)
                      for row in range(rows)]
        for position, name in enumerate(_FACTOR_NAMES)
    }
    return pd.DataFrame(
        data, index=["600000.SH", "000001.SZ", "600519.SH"][:rows])


def _weights() -> dict[str, float]:
    return {name: 0.1 for name in _FACTOR_NAMES}


def _entry_source() -> str:
    return (
        "def build_transform():\n"
        "    def transform(z, weights):\n"
        "        return z.iloc[:, 0]\n"
        "    return transform\n"
    )


class ProgramValidationTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # 既有契约
    # ------------------------------------------------------------------
    def test_valid_multi_file_program(self) -> None:
        info = custom.validate_program(SCRIPTED_PROGRAM_FILES, "main.py")
        self.assertTrue(info["ok"])
        self.assertEqual(info["files"], ["helpers.py", "main.py"])
        self.assertEqual(info["modules"], ["helpers", "main"])
        self.assertTrue(info["takes_params"])

    def test_forbidden_import_rejected(self) -> None:
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {"main.py": "import socket\n\ndef build_transform():\n    return None\n"})
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
        # 多级包里同样禁止用内置模块名做顶层包名
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {"json/helpers.py": "def build_transform():\n    return None\n"},
                "json/helpers.py")

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
            {f"z_{name}": [0.1, -0.2] for name in _FACTOR_NAMES},
            index=["600000.SH", "000001.SZ"],
        )
        output = runtime.transform(frame, _weights())
        self.assertEqual(len(output), 2)
        selected = runtime.select(_scores(), 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], "000001.SZ")

    # ------------------------------------------------------------------
    # 多级包 / 子目录
    # ------------------------------------------------------------------
    @staticmethod
    def _package_program() -> dict[str, str]:
        return {
            "main.py": (
                "import pandas as pd\n"
                "import pkg\n"
                "import pkg.helpers\n"
                "from pkg import helpers\n"
                "from pkg.helpers import weighted_score\n"
                "from pkg.sub import mod\n"
                "\n"
                "def build_transform(params=None):\n"
                "    def transform(z, weights):\n"
                "        base = weighted_score(z, weights)\n"
                "        bump = helpers.tag_ok(z)\n"
                "        extra = mod.constant_like(z)\n"
                "        return base + bump + extra\n"
                "    return transform\n"
            ),
            "pkg/__init__.py": "PACKAGE_TAG = 'pkg'\n",
            "pkg/helpers.py": (
                "import pandas as pd\n"
                "from pkg import PACKAGE_TAG\n"
                "\n"
                "def weighted_score(z, weights):\n"
                "    total = pd.Series(0.0, index=z.index, dtype=float)\n"
                "    for name, weight in weights.items():\n"
                "        column = 'z_' + name\n"
                "        if column in z.columns:\n"
                "            values = pd.to_numeric(z[column], errors='coerce').fillna(0.0)\n"
                "            total = total + float(weight) * values\n"
                "    return total\n"
                "\n"
                "def tag_ok(z):\n"
                "    return pd.Series(0.0 if PACKAGE_TAG == 'pkg' else 1.0,\n"
                "                     index=z.index, dtype=float)\n"
            ),
            "pkg/sub/__init__.py": "SUB_TAG = 'sub'\n",
            "pkg/sub/mod.py": (
                "import pandas as pd\n"
                "\n"
                "def constant_like(z):\n"
                "    return pd.Series(0.5, index=z.index, dtype=float)\n"
            ),
        }

    def test_multilevel_package_import_compile_and_run(self) -> None:
        files = self._package_program()
        info = custom.validate_program(files)  # 入口默认 main.py
        self.assertTrue(info["ok"])
        self.assertEqual(
            info["modules"],
            ["main", "pkg", "pkg.helpers", "pkg.sub", "pkg.sub.mod"])
        runtime = custom.compile_program(files)
        # 包 __init__ 已执行，并挂到父模块属性上
        self.assertEqual(runtime.modules["pkg"].PACKAGE_TAG, "pkg")
        self.assertIs(runtime.modules["pkg"].helpers,
                      runtime.modules["pkg.helpers"])
        self.assertIs(runtime.modules["pkg"].sub,
                      runtime.modules["pkg.sub"])
        self.assertIs(runtime.modules["pkg.sub"].mod,
                      runtime.modules["pkg.sub.mod"])
        output = runtime.transform(_z_frame(2), _weights())
        self.assertEqual(len(output), 2)
        self.assertTrue(bool(pd.notna(output).all()))
        # 期望 = 加权分（tag_ok 为 0）+ 0.5（constant_like）
        frame = _z_frame(2)
        expected = pd.Series(0.0, index=frame.index, dtype=float)
        for name in _FACTOR_NAMES:
            expected = expected + 0.1 * frame[f"z_{name}"]
        expected = expected + 0.5
        self.assertTrue(((output - expected).abs() < 1e-9).all())
        # service 网格寻优走 validate=False 复用已校验代码
        unchecked = custom.compile_program(files, "main.py", {}, validate=False)
        self.assertEqual(len(unchecked.transform(_z_frame(2), _weights())), 2)

    def test_implicit_namespace_package_supported(self) -> None:
        files = {
            "main.py": (
                "import pandas as pd\n"
                "from pkg.sub.mod import constant_like\n"
                "def build_transform():\n"
                "    def transform(z, weights):\n"
                "        return pd.Series(0.0, index=z.index) + constant_like(z)\n"
                "    return transform\n"
            ),
            "pkg/sub/mod.py": (
                "import pandas as pd\n"
                "def constant_like(z):\n"
                "    return pd.Series(0.25, index=z.index, dtype=float)\n"
            ),
        }
        runtime = custom.compile_program(files)
        output = runtime.transform(_z_frame(2), _weights())
        self.assertEqual(list(output), [0.25, 0.25])

    def test_path_traversal_and_invalid_paths_rejected(self) -> None:
        base = {"main.py": _entry_source()}
        bad_paths = [
            "../evil.py",
            "pkg/../evil.py",
            "/etc/evil.py",
            "C:/evil.py",
            r"pkg\evil.py",
            "pkg//mod.py",
            ".hidden/mod.py",
            "pkg/1bad.py",
            "pkg/mod.txt",
            "",
        ]
        for bad in bad_paths:
            with self.subTest(path=bad):
                with self.assertRaises(custom.ModelCodeError):
                    custom.validate_program({**base, bad: "VALUE = 1\n"}, "main.py")

    def test_module_name_collision_rejected(self) -> None:
        base = {"main.py": _entry_source()}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {**base, "pkg/__init__.py": "TAG = 1\n", "pkg.py": "TAG = 1\n"},
                "main.py")
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                [{"path": "main.py", "content": _entry_source()},
                 {"path": "main.py", "content": "VALUE = 1\n"}],
                "main.py")

    # ------------------------------------------------------------------
    # AST 放宽：类 / 装饰器 / 生成器
    # ------------------------------------------------------------------
    def test_class_init_private_attrs_and_decorators(self) -> None:
        files = {"main.py": (
            "import pandas as pd\n"
            "\n"
            "class Scorer(object):\n"
            "    def __init__(self, scale):\n"
            "        self._scale = float(scale)\n"
            "        self.__hits = 0\n"
            "\n"
            "    def __repr__(self):\n"
            "        return 'Scorer(%r)' % self._scale\n"
            "\n"
            "    def __len__(self):\n"
            "        return 10\n"
            "\n"
            "    def __getitem__(self, index):\n"
            "        return self._scale\n"
            "\n"
            "    def __eq__(self, other):\n"
            "        return isinstance(other, Scorer) and self._scale == other.scale\n"
            "\n"
            "    def __hash__(self):\n"
            "        return hash(self._scale)\n"
            "\n"
            "    def __call__(self, z, weights):\n"
            "        return pd.Series(self._scale, index=z.index, dtype=float)\n"
            "\n"
            "    @staticmethod\n"
            "    def build(scale):\n"
            "        return Scorer(scale)\n"
            "\n"
            "    @classmethod\n"
            "    def default(cls):\n"
            "        return cls(1.0)\n"
            "\n"
            "    @property\n"
            "    def scale(self):\n"
            "        return self._scale\n"
            "\n"
            "\n"
            "class Child(Scorer):\n"
            "    def __init__(self):\n"
            "        super().__init__(2.0)\n"
            "\n"
            "\n"
            "def build_transform(params=None):\n"
            "    return Child()\n"
        )}
        self.assertTrue(custom.validate_program(files, "main.py")["ok"])
        runtime = custom.compile_program(files, "main.py", {})
        output = runtime.transform(_z_frame(2), _weights())
        self.assertEqual(list(output), [2.0, 2.0])
        # 类/装饰器/私有属性在受限执行环境里真实可用
        module = runtime.modules["main"]
        self.assertEqual(module.Scorer.build(3.0).scale, 3.0)
        self.assertEqual(module.Scorer.default().scale, 1.0)
        self.assertEqual(len(module.Child()), 10)
        self.assertEqual(module.Child()[0], 2.0)
        self.assertIn("Scorer(2.0)", repr(module.Child()))

    def test_generator_allowed(self) -> None:
        files = {"main.py": (
            "import pandas as pd\n"
            "\n"
            "def _columns(z):\n"
            "    for column in z.columns:\n"
            "        yield column\n"
            "\n"
            "def _pairs(z):\n"
            "    yield from _columns(z)\n"
            "\n"
            "def build_transform():\n"
            "    def transform(z, weights):\n"
            "        total = pd.Series(0.0, index=z.index, dtype=float)\n"
            "        for column in _pairs(z):\n"
            "            total = total + z[column] * 0.0\n"
            "        return total\n"
            "    return transform\n")}
        runtime = custom.compile_program(files, "main.py", {})
        output = runtime.transform(_z_frame(3), _weights())
        self.assertEqual(list(output), [0.0, 0.0, 0.0])

    def test_validate_code_accepts_class_based_model(self) -> None:
        """单文件模型复用同一套 AST 规则：类/生成器可用且能通过子进程冒烟。"""
        source = (
            "import pandas as pd\n"
            "\n"
            "class Scaler(object):\n"
            "    def __init__(self, factor):\n"
            "        self._factor = float(factor)\n"
            "\n"
            "    def __call__(self, z, weights):\n"
            "        return pd.Series(self._factor, index=z.index, dtype=float)\n"
            "\n"
            "\n"
            "def _columns(frame):\n"
            "    for column in frame.columns:\n"
            "        yield column\n"
            "\n"
            "\n"
            "def build_transform(params=None):\n"
            "    options = params or {}\n"
            "    for _column in _columns(pd.DataFrame()):\n"
            "        break\n"
            "    return Scaler(options.get('factor', 1.0))\n"
        )
        info = custom.validate_code(source, {"factor": 2.0})
        self.assertTrue(info["ok"])
        self.assertTrue(info["takes_params"])
        transform = custom.compile_code(source, {"factor": 2.0}, validate=False)
        output = transform(_z_frame(2), _weights())
        self.assertEqual(list(output), [2.0, 2.0])

    def test_unsupported_decorator_rejected(self) -> None:
        files = {"main.py": (
            "import functools\n"
            "\n"
            "class Holder(object):\n"
            "    @functools.cached_property\n"
            "    def value(self):\n"
            "        return 1\n"
            "\n"
            "def build_transform():\n"
            "    return lambda z, weights: z.iloc[:, 0]\n")}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(files, "main.py")
        files = {"main.py": (
            "def deco(fn):\n"
            "    return fn\n"
            "\n"
            "@deco\n"
            "def helper():\n"
            "    return 1\n"
            "\n"
            "def build_transform():\n"
            "    return lambda z, weights: z.iloc[:, 0]\n")}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(files, "main.py")

    def test_class_metaclass_and_call_base_rejected(self) -> None:
        files = {"main.py": (
            "import abc\n"
            "class A(metaclass=abc.ABCMeta):\n"
            "    pass\n"
            "def build_transform():\n"
            "    return lambda z, weights: z.iloc[:, 0]\n")}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(files, "main.py")
        files = {"main.py": (
            "def make_base():\n"
            "    return object\n"
            "class A(make_base()):\n"
            "    pass\n"
            "def build_transform():\n"
            "    return lambda z, weights: z.iloc[:, 0]\n")}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(files, "main.py")

    def test_dangerous_dunder_still_rejected(self) -> None:
        cases = [
            (
                "class A(object):\n"
                "    def leak(self):\n"
                "        return self.__class__\n"
                "\n"
                "def build_transform():\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
            (
                "def build_transform():\n"
                "    value = ().__class__\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
            (
                "def build_transform():\n"
                "    value = (1).__class__.__bases__\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
            (
                "def helper():\n"
                "    return 1\n"
                "\n"
                "def build_transform():\n"
                "    value = helper.__globals__\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
            (
                "def build_transform():\n"
                "    value = object.__subclasses__()\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
            (
                "def build_transform():\n"
                "    value = __builtins__\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
        ]
        for source in cases:
            with self.subTest(source=source.splitlines()[-2]):
                with self.assertRaises(custom.ModelCodeError):
                    custom.validate_program({"main.py": source}, "main.py")

    def test_module_dunder_metadata_is_allowed(self) -> None:
        """__all__/__version__ 等模块元数据不应再被 AST 当成危险变量名拦截。"""
        files = {
            "main.py": (
                "__all__ = ['build_transform']\n"
                "__version__ = '1.0'\n"
                "\n"
                "def build_transform():\n"
                "    return lambda z, weights: z.iloc[:, 0]\n"
            ),
        }
        result = custom.validate_program(files, "main.py")
        self.assertTrue(result)

    def test_private_attr_on_other_object_rejected(self) -> None:
        files = {"main.py": (
            "class A(object):\n"
            "    def __init__(self):\n"
            "        self._value = 1\n"
            "\n"
            "    def read(self, other):\n"
            "        return other._value\n"
            "\n"
            "def build_transform():\n"
            "    return lambda z, weights: z.iloc[:, 0]\n")}
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(files, "main.py")

    # ------------------------------------------------------------------
    # context 透传
    # ------------------------------------------------------------------
    def test_transform_context_passthrough(self) -> None:
        files = {"main.py": (
            "import pandas as pd\n"
            "def build_transform():\n"
            "    def transform(z, weights, context=None):\n"
            "        scale = float((context or {}).get('scale', 1.0))\n"
            "        return pd.Series(scale, index=z.index, dtype=float)\n"
            "    return transform\n")}
        runtime = custom.compile_program(files, "main.py")
        self.assertTrue(runtime.transform.accepts_context)
        self.assertTrue(runtime.accepts_context)
        frame = _z_frame(2)
        output = runtime.transform(frame, _weights(), {"scale": 3.0})
        self.assertEqual(list(output), [3.0, 3.0])
        # 2 参调用保持兼容
        default_output = runtime.transform(frame, _weights())
        self.assertEqual(list(default_output), [1.0, 1.0])

    def test_transform_without_context_keeps_two_args_compatible(self) -> None:
        runtime = custom.compile_program(
            SCRIPTED_PROGRAM_FILES, "main.py", {"tilt": 0.5})
        self.assertFalse(runtime.transform.accepts_context)
        self.assertFalse(runtime.accepts_context)
        output = runtime.transform(_z_frame(2), _weights(), {"scale": 9.0})
        self.assertEqual(len(output), 2)
        self.assertTrue(bool(output.notna().all()))

    # ------------------------------------------------------------------
    # 白名单扩充
    # ------------------------------------------------------------------
    def test_expanded_stdlib_and_sklearn_imports(self) -> None:
        files = {"main.py": (
            "import json\n"
            "import dataclasses\n"
            "import enum\n"
            "import abc\n"
            "import copy\n"
            "import heapq\n"
            "import bisect\n"
            "import string\n"
            "import textwrap\n"
            "import types\n"
            "from functools import partial\n"
            "from dataclasses import dataclass\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.feature_selection import SelectKBest\n"
            "from sklearn.neural_network import MLPClassifier\n"
            "from sklearn.naive_bayes import GaussianNB\n"
            "from sklearn.gaussian_process import GaussianProcessRegressor\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.calibration import CalibratedClassifierCV\n"
            "from sklearn.inspection import permutation_importance\n"
            "from sklearn.dummy import DummyClassifier\n"
            "from sklearn.manifold import TSNE\n"
            "from sklearn.mixture import GaussianMixture\n"
            "from sklearn.multioutput import MultiOutputRegressor\n"
            "from sklearn.multiclass import OneVsRestClassifier\n"
            "from sklearn.semi_supervised import LabelSpreading\n"
            "from sklearn.feature_extraction import DictVectorizer\n"
            "from sklearn.covariance import EmpiricalCovariance\n"
            "from sklearn.kernel_approximation import Nystroem\n"
            "\n"
            "def build_transform(params=None):\n"
            "    def transform(z, weights):\n"
            "        return json.dumps({'ok': 1})\n"
            "    return transform\n")}
        info = custom.validate_program(files, "main.py")
        self.assertTrue(info["ok"])
        imported = "\n".join(info["imports"])
        self.assertIn("sklearn.model_selection", imported)
        self.assertIn("sklearn.kernel_approximation", imported)

    def test_json_import_is_usable_at_runtime(self) -> None:
        files = {"main.py": (
            "import json\n"
            "import pandas as pd\n"
            "def build_transform():\n"
            "    def transform(z, weights):\n"
            "        payload = json.loads(json.dumps({'n': 2}))\n"
            "        return pd.Series(float(payload['n']), index=z.index, dtype=float)\n"
            "    return transform\n")}
        runtime = custom.compile_program(files, "main.py", {})
        self.assertEqual(list(runtime.transform(_z_frame(2), _weights())), [2.0, 2.0])

    def test_sklearn_datasets_still_rejected(self) -> None:
        for bad_module in ("sklearn.datasets", "sklearn.externals"):
            with self.subTest(module=bad_module):
                with self.assertRaises(custom.ModelCodeError):
                    custom.validate_program(
                        {"main.py": (
                            f"import {bad_module}\n"
                            "def build_transform():\n"
                            "    return lambda z, weights: z.iloc[:, 0]\n")},
                        "main.py")
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(
                {"main.py": (
                    "from sklearn.datasets import load_iris\n"
                    "def build_transform():\n"
                    "    return lambda z, weights: z.iloc[:, 0]\n")},
                "main.py")

    # ------------------------------------------------------------------
    # 上限：默认值放宽 + 环境变量可配
    # ------------------------------------------------------------------
    def test_default_limits_are_relaxed(self) -> None:
        self.assertEqual(custom.MAX_SOURCE_CHARS, 60000)
        self.assertEqual(custom.MAX_PROGRAM_FILES, 64)
        self.assertEqual(custom.MAX_PROGRAM_CHARS, 500000)
        self.assertEqual(custom.MAX_PROGRAM_FILE_CHARS, 200000)
        self.assertEqual(custom.MAX_PACKAGES, 16)

    def test_env_limits_override_files_and_chars(self) -> None:
        main = _entry_source()
        # 文件数：默认 64 拒绝 65 个；环境变量放宽到 65 后通过
        many = {"main.py": main}
        for index in range(custom.MAX_PROGRAM_FILES):
            many[f"mod_{index}.py"] = "VALUE = 1\n"
        self.assertEqual(len(many), custom.MAX_PROGRAM_FILES + 1)
        with self.assertRaises(custom.ModelCodeError):
            custom.validate_program(many, "main.py")
        with mock.patch.dict(
                os.environ,
                {"AI_MAX_PROGRAM_FILES": str(custom.MAX_PROGRAM_FILES + 1)}):
            self.assertTrue(custom.validate_program(many, "main.py")["ok"])

        # 总长度 / 单文件长度：环境变量调小后立即生效
        short_main = (
            "def build_transform():\n"
            "    return lambda z, weights: z.iloc[:, 0]\n")
        long_file = "X = '" + "a" * 140 + "'\n"
        with mock.patch.dict(os.environ, {"AI_MAX_PROGRAM_CHARS": "200"}):
            with self.assertRaises(custom.ModelCodeError) as context:
                custom.validate_program(
                    {"main.py": short_main, "helpers.py": long_file}, "main.py")
            self.assertIn("总长度", str(context.exception))
        with mock.patch.dict(os.environ, {"AI_MAX_PROGRAM_FILE_CHARS": "100"}):
            with self.assertRaises(custom.ModelCodeError) as context:
                custom.validate_program(
                    {"main.py": short_main, "helpers.py": long_file}, "main.py")
            self.assertIn("单文件上限", str(context.exception))
        # 默认上限下同样的程序通过
        self.assertTrue(custom.validate_program(
            {"main.py": short_main, "helpers.py": long_file}, "main.py")["ok"])

    def test_env_limit_override_source_chars(self) -> None:
        source = _entry_source()
        self.assertGreater(len(source), 20)
        with mock.patch.dict(os.environ, {"AI_MAX_SOURCE_CHARS": "20"}):
            with self.assertRaises(custom.ModelCodeError) as context:
                custom.validate_code(source)
            self.assertIn("source_size", str(context.exception))

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
        # 旧上限 4 已放宽：默认 16 个以内都合法
        self.assertEqual(
            len(custom.normalize_package_specs([f"pkg_{index}" for index in range(5)])),
            5)
        with self.assertRaises(custom.ModelCodeError):
            custom.normalize_package_specs(
                [f"pkg_{index}" for index in range(custom.MAX_PACKAGES + 1)])

    def test_env_package_limit_override(self) -> None:
        with mock.patch.dict(os.environ, {"AI_MAX_PACKAGES": "3"}):
            with self.assertRaises(custom.ModelCodeError):
                custom.normalize_package_specs(["a", "b", "c", "d"])
        with mock.patch.dict(os.environ, {"AI_MAX_PACKAGES": "20"}):
            self.assertEqual(
                len(custom.normalize_package_specs(
                    [f"pkg_{index}" for index in range(17)])),
                17)


if __name__ == "__main__":
    unittest.main()
