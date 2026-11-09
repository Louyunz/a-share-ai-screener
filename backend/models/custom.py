# -*- coding: utf-8 -*-
"""AI 生成模型源码的安全校验、子进程冒烟与受限编译执行。

单文件模型（``validate_code`` / ``compile_code``）校验分三层，任何一层不通过都抛
:class:`ModelCodeError`：

1. 规模与语法：源码不超过 ``MAX_SOURCE_CHARS``（默认 60000，环境变量
   ``AI_MAX_SOURCE_CHARS`` 可配），且能被 ``ast.parse`` 解析；
2. AST 白名单：只允许纯计算模块（``math`` / ``numpy`` / ``pandas`` / ``sklearn`` /
   ``scipy`` 与安全的 stdlib）导入，禁止文件、网络、系统调用与动态执行；允许类、
   ``@staticmethod`` / ``@classmethod`` / ``@property``、生成器与 self/cls 私有属性，
   但禁止 ``__class__`` 等危险 dunder；顶层只允许 import、函数/类定义与常量赋值；
3. 子进程冒烟：在临时目录、独立 Python 子进程中把 10 列 ``z_*`` 与权重喂给
   ``build_transform()`` 返回的 transform，10 秒超时，校验返回 ``pandas.Series``、
   长度一致且全部为有限数值。

支持参数化引擎（供服务端网格寻优）：``def build_transform(params=None)``，
参数只能叫 ``params`` 且最多一个；冒烟子进程会把 ``params`` 传进去。

多文件程序（``validate_program`` / ``compile_program``）支持多级包
（``pkg/__init__.py``、``pkg/helpers.py``、``pkg/sub/mod.py``），模块名等于去掉
``.py`` 后把 ``/`` 换成 ``.``，导入白名单同时放行本程序内的模块；编译时按 import
依赖执行包 ``__init__`` 与子模块，子模块挂在父模块属性上。``compile_program`` 返回的
transform 统一包装成 ``transform(z, weights, context=None)`` 并在函数上标记
``accepts_context``：用户 transform 接受第三个参数时透传上下文，否则仍按 2 参调用。

``compile_code`` / ``compile_program`` 都在受限 ``__builtins__`` 环境里执行代码；
``validate=False`` 时可跳过重复校验，直接复用已经校验过的代码。
"""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from ..engine.factors import FACTOR_NAMES

__all__ = [
    "MAX_SOURCE_CHARS",
    "SMOKE_TIMEOUT_SECONDS",
    "MAX_PROGRAM_FILES",
    "MAX_PROGRAM_CHARS",
    "MAX_PROGRAM_FILE_CHARS",
    "MAX_PACKAGES",
    "PROGRAM_ENTRY_DEFAULT",
    "ModelValidationError",
    "ModelCodeError",
    "ProgramRuntime",
    "validate_code",
    "validate_program",
    "compile_code",
    "compile_program",
    "normalize_package_specs",
    "package_import_roots",
    "ai_packages_dir",
    "install_packages",
]

# ---------------------------------------------------------------------------
# 可配上限：导入时取环境变量，运行时在每次校验前重新读取（便于测试/部署放宽）
# ---------------------------------------------------------------------------


def _env_limit(name: str, default: int) -> int:
    """读取正整数环境变量；缺失或非法时回落到 ``default``。"""
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    return value if value >= 1 else int(default)


# 源码长度上限（契约 6.3；AI_MAX_SOURCE_CHARS 可配，默认 60000）
MAX_SOURCE_CHARS = _env_limit("AI_MAX_SOURCE_CHARS", 60000)
# 子进程冒烟超时（契约 6.3）
SMOKE_TIMEOUT_SECONDS = 10
# 冒烟用合成数据规模
_SMOKE_ROWS = 8
_SMOKE_SEED = 20240601

# 允许导入的模块根名（只放纯计算模块；文件/网络/系统/动态执行一律不在此列）
ALLOWED_IMPORT_ROOTS = (
    "math", "numpy", "pandas", "sklearn", "scipy",
    # 纯计算 stdlib：AI 编写程序/模型代码时可自由使用
    "itertools", "functools", "collections", "statistics", "numbers",
    "re", "random", "datetime", "typing", "decimal", "fractions", "operator",
    "json", "dataclasses", "enum", "abc", "copy", "heapq", "bisect",
    "string", "textwrap", "types",
)

# AI 程序（write_model_program）的上限：多文件、多级包、更长源码（环境变量可配）
MAX_PROGRAM_FILES = _env_limit("AI_MAX_PROGRAM_FILES", 64)
MAX_PROGRAM_CHARS = _env_limit("AI_MAX_PROGRAM_CHARS", 500000)
MAX_PROGRAM_FILE_CHARS = _env_limit("AI_MAX_PROGRAM_FILE_CHARS", 200000)
# 一次最多申请安装的第三方库数量（环境变量可配）
MAX_PACKAGES = _env_limit("AI_MAX_PACKAGES", 16)
PROGRAM_ENTRY_DEFAULT = "main.py"
PIP_INSTALL_TIMEOUT_SECONDS = 240

# 允许 AI 向审核申请安装的第三方包：包名 -> import 根名（无法自动推导时的映射）
PACKAGE_ROOT_ALIASES = {
    "scikit-learn": "sklearn",
    "scikit_learn": "sklearn",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "shap": "shap",
    "ta": "ta",
}
# 包名/版本约束的允许字符（排除 URL、git、本地路径等安装形式）
_PACKAGE_SPEC_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,63})"
    r"(?P<extras>\[[A-Za-z0-9_,. -]{1,64}\])?"
    r"(?P<spec>([=<>!~]=?[0-9A-Za-z.*+!_-]{1,32})*)$"
)
# 程序文件路径：每段都是合法 Python 标识符，用 / 分隔，末段以 .py 结束。
# 支持 pkg/__init__.py、pkg/helpers.py、pkg/sub/mod.py；拒绝 ..、绝对路径、反斜杠。
_PROGRAM_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*\.py$"
)

# sklearn 只允许模型/预处理/选型等常用子模块，排除 datasets/externals 等带文件 IO 的模块
ALLOWED_SKLEARN_PREFIXES = (
    "sklearn.preprocessing", "sklearn.svm", "sklearn.ensemble", "sklearn.linear_model",
    "sklearn.cluster", "sklearn.decomposition", "sklearn.pipeline", "sklearn.impute",
    "sklearn.neighbors", "sklearn.tree", "sklearn.metrics", "sklearn.utils",
    "sklearn.model_selection", "sklearn.feature_selection", "sklearn.neural_network",
    "sklearn.naive_bayes", "sklearn.gaussian_process", "sklearn.compose",
    "sklearn.calibration", "sklearn.inspection", "sklearn.dummy", "sklearn.manifold",
    "sklearn.mixture", "sklearn.multioutput", "sklearn.multiclass",
    "sklearn.semi_supervised", "sklearn.feature_extraction", "sklearn.covariance",
    "sklearn.kernel_approximation",
)
# 允许在函数/类方法上使用的装饰器（保持沙箱边界：不允许任意装饰器执行）
_ALLOWED_FUNCTION_DECORATORS = frozenset({"staticmethod", "classmethod", "property"})

# 允许访问的安全 dunder 属性：只允许以 self/cls 或 super() 为根
_SAFE_DUNDER_ATTRS = frozenset(
    {
        "__init__", "__repr__", "__str__", "__len__", "__getitem__", "__setitem__",
        "__iter__", "__next__", "__call__", "__enter__", "__exit__", "__eq__",
        "__lt__", "__hash__",
    }
)

# 受限执行环境里保留的内建函数/异常（其余内建全部移除）
SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs", "all", "any", "bool", "callable", "chr", "complex", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
        # 类/装饰器/生成器等写法需要的内建
        "super", "object", "property", "staticmethod", "classmethod", "NotImplemented",
        "__build_class__",
        "ArithmeticError", "AssertionError", "AttributeError", "Exception",
        "IndexError", "KeyError", "NameError", "NotImplementedError", "OverflowError",
        "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
    }
)

# 禁止出现的名字：文件/网络/系统/动态执行相关
FORBIDDEN_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
        "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr",
        "os", "sys", "subprocess", "importlib", "pathlib", "shutil", "socket", "urllib",
        "requests", "ctypes", "builtins", "io", "tempfile", "threading", "multiprocessing",
        "signal", "pickle", "marshal", "shelve", "sqlite3", "http", "ftplib", "runpy",
        "code", "codeop", "gc", "inspect", "dis", "tokenize", "zipfile", "tarfile",
        "gzip", "bz2", "lzma", "logging", "platform", "resource", "pty", "webbrowser",
    }
)

# 禁止访问/调用的属性：落盘、网络、系统、动态执行
FORBIDDEN_ATTRS = frozenset(
    {
        "system", "popen", "spawn", "spawnl", "spawnv", "fork", "exec", "execl",
        "execv", "execve", "kill", "open", "remove", "unlink", "rmdir", "mkdir",
        "makedirs", "chmod", "chown", "write", "writelines", "read", "readline",
        "readlines", "load", "dump", "save", "savez",
        "savez_compressed", "environ",
        # 数组/第三方库的文件 IO 与动态库加载（open/read_csv 之外的常见绕过）
        "fromfile", "tofile", "memmap", "open_memmap", "genfromtxt", "loadtxt",
        "savetxt", "load_npz", "save_npz", "fromregex", "load_library",
        "ctypeslib", "f2py", "datasets", "externals",
        "loadmat", "savemat", "mmread", "mmwrite", "whosmat", "loadarff",
        "readsav", "imread", "imsave", "netcdf_file", "netcdf_variable",
        "exec_command", "distutils",
        "to_csv", "to_excel", "to_json", "to_parquet", "to_pickle", "to_sql",
        "to_html", "to_clipboard", "to_feather", "to_hdf", "to_stata", "to_markdown",
        "to_latex", "to_xml", "to_orc",
        "read_csv", "read_excel", "read_json", "read_parquet", "read_pickle",
        "read_sql", "read_table", "read_html", "read_clipboard", "read_feather",
        "read_hdf", "read_stata", "read_xml", "read_fwf", "read_sas", "read_spss",
        "read_orc",
        "import_module", "load_module", "spec_from_file_location", "exec_module",
        "urlopen", "urlretrieve", "socket", "connect", "send", "sendall", "recv",
        "os", "sys", "subprocess", "importlib", "pathlib", "shutil", "ctypes",
        "builtins", "pickle", "marshal", "tempfile", "threading", "multiprocessing",
        "signal", "io",
        # types 的动态构造能力（CodeType/FunctionType 可绕过受限执行）；types 本身仍可用于
        # SimpleNamespace/MappingProxyType 等安全工具。
        "CodeType", "FunctionType", "LambdaType", "new_class", "prepare_class",
        "resolve_bases",
    }
)

# from ... import 的危险别名：import 白名单只校验模块根名，这里补齐别名拦截，
# 避免 from pandas import read_csv、from types import CodeType 这类绕过。
_FORBIDDEN_FROM_IMPORT_NAMES = frozenset(
    set(FORBIDDEN_NAMES)
    | set(FORBIDDEN_ATTRS)
    | {"CodeType", "FunctionType", "LambdaType", "new_class", "prepare_class",
       "resolve_bases"}
)

# 冒烟子进程驱动脚本：读取同目录下 <token>payload.json / <token>candidate.py，写出 <token>result.json
_SMOKE_DRIVER = '''# -*- coding: utf-8 -*-
import builtins
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""


def _path(name):
    return BASE / (TOKEN + name)


def _load_payload():
    with _path("payload.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _restricted_builtins(payload):
    allowed = {}
    for name in payload["safe_builtins"]:
        value = getattr(builtins, name, None)
        if value is not None:
            allowed[name] = value
    roots = set(payload["allowed_import_roots"])

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".")[0]
        if level or root not in roots:
            raise ImportError("不允许导入模块: " + str(name))
        return builtins.__import__(name, globals, locals, fromlist, level)

    allowed["__import__"] = _safe_import
    return allowed


def _run(payload, result):
    code = _path("candidate.py").read_text(encoding="utf-8")
    namespace = {
        "__builtins__": _restricted_builtins(payload),
        "__name__": "ai_candidate",
    }
    exec(compile(code, "<ai_candidate>", "exec"), namespace)

    build_transform = namespace.get("build_transform")
    if not callable(build_transform):
        raise RuntimeError("build_transform 未定义或不可调用")
    if payload.get("build_transform_takes_params"):
        transform = build_transform(dict(payload.get("params") or {}))
    else:
        transform = build_transform()
    if not callable(transform):
        raise RuntimeError("build_transform() 的返回值不可调用")

    factors = payload["factors"]
    frame = pd.DataFrame(
        {("z_" + name): payload["z_values"][name] for name in factors},
        index=payload["index"],
    )
    output = transform(frame, dict(payload["weights"]))
    if not isinstance(output, pd.Series):
        raise RuntimeError("transform 必须返回 pandas.Series，实际返回 " + type(output).__name__)
    if len(output) != len(frame):
        raise RuntimeError("返回 Series 长度 %d 与输入行数 %d 不一致" % (len(output), len(frame)))
    numeric = pd.to_numeric(output, errors="coerce")
    values = numeric.to_numpy(dtype="float64", na_value=float("nan"))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("返回 Series 含 NaN/Inf，必须全部为有限数值")
    result["ok"] = True
    result["detail"] = "子进程冒烟通过：%d 行输入返回 %d 个有限分数" % (len(frame), len(output))


def main():
    result = {"ok": False, "detail": "冒烟脚本未执行"}
    try:
        payload = _load_payload()
        _run(payload, result)
    except Exception:
        result["ok"] = False
        result["detail"] = "冒烟测试失败：\\n" + traceback.format_exc(limit=6)[-1500:]
    _path("result.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
'''


class ModelValidationError(ValueError):
    """模型提案/配置不合法。"""


class ModelCodeError(ModelValidationError):
    """AI 生成的模型源码未通过静态校验或冒烟测试。"""

    def __init__(
        self,
        message: str,
        checks: list[dict[str, str]] | None = None,
        imports: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.checks: list[dict[str, str]] = list(checks or [])
        self.imports: list[str] = list(imports or [])


class _RuleViolation(Exception):
    """AST 白名单检查失败（内部使用）。"""

    def __init__(self, check: str, detail: str) -> None:
        super().__init__(detail)
        self.check = check
        self.detail = detail


# ---------------------------------------------------------------------------
# AST 白名单
# ---------------------------------------------------------------------------


def _module_allowed(
    name: str,
    *,
    program_modules: frozenset[str] | set[str] | None = None,
    extra_roots: Sequence[str] = (),
) -> bool:
    if program_modules:
        if name in program_modules:
            return True
        # 允许多级包路径中的前缀目录（如 pkg、pkg.sub，即使没有 __init__.py）
        if any(module.startswith(name + ".") for module in program_modules):
            return True
    root = str(name).split(".")[0]
    if root in set(extra_roots):
        return True
    if root not in ALLOWED_IMPORT_ROOTS:
        return False
    if root == "sklearn":
        return name == "sklearn" or any(
            name == prefix or name.startswith(prefix + ".") for prefix in ALLOWED_SKLEARN_PREFIXES
        )
    return True


def _check_imports(
    tree: ast.Module,
    *,
    program_modules: frozenset[str] | set[str] | None = None,
    extra_roots: Sequence[str] = (),
) -> list[str]:
    """校验 import 白名单，返回人类可读的导入清单。"""
    imports: list[str] = []
    allowed_roots = " / ".join(
        list(ALLOWED_IMPORT_ROOTS) + sorted(set(extra_roots) - set(ALLOWED_IMPORT_ROOTS)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed(alias.name, program_modules=program_modules,
                                       extra_roots=extra_roots):
                    raise _RuleViolation(
                        "imports",
                        f"第 {node.lineno} 行禁止导入 {alias.name}；"
                        f"只允许 {allowed_roots} 以及本程序内的模块",
                    )
                imports.append(
                    alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise _RuleViolation("imports", f"第 {node.lineno} 行禁止相对导入")
            module = node.module or ""
            if not _module_allowed(module, program_modules=program_modules,
                                   extra_roots=extra_roots):
                raise _RuleViolation(
                    "imports",
                    f"第 {node.lineno} 行禁止从 {module or '未知模块'} 导入；"
                    f"只允许 {allowed_roots} 以及本程序内的模块",
                )
            for alias in node.names:
                if alias.name == "*":
                    raise _RuleViolation(
                        "imports", f"第 {node.lineno} 行禁止 import *"
                    )
                if alias.name in _FORBIDDEN_FROM_IMPORT_NAMES:
                    raise _RuleViolation(
                        "imports",
                        f"第 {node.lineno} 行禁止导入危险名字 "
                        f"{module}.{alias.name}",
                    )
                target = f"from {module} import {alias.name}"
                if alias.asname:
                    target += f" as {alias.asname}"
                imports.append(target)
    return imports


def _is_simple_target(node: ast.expr) -> bool:
    """顶层赋值的左侧只能是普通名字（不允许给属性/下标赋值）。"""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_simple_target(element) for element in node.elts)
    return False


def _check_module_structure(tree: ast.Module) -> None:
    """顶层只允许 import、函数定义与常量赋值，避免导入时的副作用。"""
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if not all(_is_simple_target(target) for target in targets):
                raise _RuleViolation(
                    "ast_safety",
                    f"第 {stmt.lineno} 行：顶层赋值只能给普通变量名赋值",
                )
            value = stmt.value
            if value is None:
                continue
            for child in ast.walk(value):
                if isinstance(child, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom)):
                    raise _RuleViolation(
                        "ast_safety",
                        f"第 {stmt.lineno} 行：顶层赋值只能使用常量表达式，禁止调用函数",
                    )
            continue
        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue  # 模块 docstring
            raise _RuleViolation(
                "ast_safety", f"第 {stmt.lineno} 行：顶层不允许执行表达式（可能有副作用）"
            )
        if isinstance(stmt, ast.Pass):
            continue
        raise _RuleViolation(
            "ast_safety",
            f"第 {stmt.lineno} 行：顶层只允许 import / 函数定义 / 常量赋值，"
            f"不允许 {type(stmt).__name__}",
        )


def _collect_defined_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _collect_bound_names(tree: ast.Module) -> set[str]:
    """收集所有可能被引用的名字：导入名、函数名、赋值名、参数名。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


def _attribute_root(node: ast.expr) -> ast.expr:
    """返回属性访问链最内层的表达式（``self._x.y`` -> ``Name('self')``）。"""
    root = node
    while isinstance(root, ast.Attribute):
        root = root.value
    return root


def _has_self_root(node: ast.expr) -> bool:
    root = _attribute_root(node)
    return isinstance(root, ast.Name) and root.id in {"self", "cls"}


def _has_super_call_root(node: ast.expr) -> bool:
    root = _attribute_root(node)
    return (
        isinstance(root, ast.Call)
        and isinstance(root.func, ast.Name)
        and root.func.id == "super"
    )


def _check_attribute(node: ast.Attribute) -> None:
    """属性访问：保留危险属性拦截，放宽 self/cls 的私有属性与安全 dunder。"""
    attr = node.attr
    if attr in FORBIDDEN_ATTRS:
        raise _RuleViolation(
            "ast_safety", f"第 {node.lineno} 行禁止访问属性 .{attr}"
        )
    if attr.startswith("__") and attr.endswith("__"):
        if attr in _SAFE_DUNDER_ATTRS and (
            _has_self_root(node) or _has_super_call_root(node)
        ):
            return
        raise _RuleViolation(
            "ast_safety",
            f"第 {node.lineno} 行禁止访问 dunder 属性 .{attr}"
            "（只允许 self/cls/super() 上的安全 dunder）",
        )
    if attr.startswith("_"):
        if _has_self_root(node):
            return  # self._cache / cls._REGISTRY 这类单下划线属性
        raise _RuleViolation(
            "ast_safety",
            f"第 {node.lineno} 行禁止访问下划线属性 .{attr}"
            "（单下划线属性只允许以 self/cls 为根）",
        )


def _check_call(
    node: ast.Call, defined_functions: set[str], bound_names: set[str]
) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        name = func.id
        if name.startswith("__"):
            raise _RuleViolation("ast_safety", f"第 {node.lineno} 行禁止调用 {name}()")
        # 程序自己定义的函数优先放行（允许命名为 code 等普通名字）
        if name in defined_functions:
            return
        if name in FORBIDDEN_NAMES:
            raise _RuleViolation("ast_safety", f"第 {node.lineno} 行禁止调用 {name}()")
        if name in SAFE_BUILTIN_NAMES or name in bound_names:
            return
        raise _RuleViolation(
            "ast_safety", f"第 {node.lineno} 行调用了未定义的函数 {name}()"
        )
    if isinstance(func, ast.Attribute):
        # 属性本身由 _check_attribute 统一校验（含 self/cls 私有属性放宽）
        root = _attribute_root(func.value)
        if isinstance(root, ast.Name) and root.id.startswith("__"):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行禁止通过 {root.id} 调用方法"
            )
        return
    if isinstance(func, ast.Lambda):
        return
    raise _RuleViolation(
        "ast_safety", f"第 {node.lineno} 行调用形式不受支持（只允许函数名、方法或 lambda）"
    )


def _check_ast_safety(tree: ast.Module) -> None:
    """遍历整棵 AST：只保留真正的安全边界。

    放宽（避免误杀正常的面向对象/生成器写法）：
    - 允许 ``ClassDef``，基类只能是 Name/Attribute，不允许装饰器与 metaclass；
    - 允许类内/普通函数定义 dunder 方法；装饰器只允许
      ``@staticmethod`` / ``@classmethod`` / ``@property``；
    - 允许以 ``self`` / ``cls`` 为根的单下划线属性；安全 dunder 属性只允许根为
      ``self`` / ``cls`` / ``super()``；
    - 允许生成器（``yield`` / ``yield from``），仍禁止 async。

    保留：危险导入/调用/属性/名称拦截（文件、网络、系统、动态执行）、
    ``__class__`` 等危险 dunder、``global`` / ``nonlocal``。
    """
    defined_functions = _collect_defined_functions(tree)
    bound_names = _collect_bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise _RuleViolation(
                    "ast_safety", f"第 {node.lineno} 行禁止使用名称 {node.id}"
                )
        elif isinstance(node, ast.Attribute):
            _check_attribute(node)
        elif isinstance(node, ast.Call):
            _check_call(node, defined_functions, bound_names)
        elif isinstance(node, ast.AsyncFunctionDef):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行不允许定义异步函数"
            )
        elif isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (
                    not isinstance(decorator, ast.Name)
                    or decorator.id not in _ALLOWED_FUNCTION_DECORATORS
                ):
                    raise _RuleViolation(
                        "ast_safety",
                        f"第 {node.lineno} 行函数装饰器不受支持；"
                        "只允许 @staticmethod / @classmethod / @property",
                    )
        elif isinstance(node, ast.ClassDef):
            if node.decorator_list:
                raise _RuleViolation(
                    "ast_safety", f"第 {node.lineno} 行不允许类装饰器"
                )
            if node.keywords:
                raise _RuleViolation(
                    "ast_safety",
                    f"第 {node.lineno} 行类定义不支持 metaclass/关键字参数",
                )
            for base in node.bases:
                if not isinstance(base, (ast.Name, ast.Attribute)):
                    raise _RuleViolation(
                        "ast_safety",
                        f"第 {node.lineno} 行类基类只允许名字或属性（Name/Attribute）",
                    )
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行不允许 global/nonlocal 声明"
            )
        elif isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行不允许异步语法"
            )


def _find_build_transform(tree: ast.Module) -> ast.FunctionDef | None:
    """返回顶层 ``build_transform`` 定义（没有则 None）。"""
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "build_transform":
            return stmt
    return None


def _build_transform_takes_params(build: ast.FunctionDef) -> bool:
    """``build_transform`` 是否为 ``def build_transform(params)`` 形式。"""
    positional = list(build.args.posonlyargs) + list(build.args.args)
    return len(positional) == 1 and positional[0].arg == "params"


def _check_entrypoint(tree: ast.Module) -> bool:
    """静态检查 ``def build_transform()`` / ``def build_transform(params=None)``。

    返回 ``build_transform`` 是否接收 ``params`` 参数。
    """
    build = _find_build_transform(tree)
    if build is None:
        raise _RuleViolation("entrypoint", "未找到顶层函数 def build_transform()")
    if build.decorator_list:
        raise _RuleViolation("entrypoint", "build_transform() 不允许使用装饰器")
    arguments = build.args
    positional = list(arguments.posonlyargs) + list(arguments.args)
    invalid = (
        arguments.vararg is not None
        or arguments.kwarg is not None
        or bool(arguments.kwonlyargs)
        or len(positional) > 1
        or (len(positional) == 1 and positional[0].arg != "params")
    )
    if invalid:
        raise _RuleViolation(
            "entrypoint",
            "build_transform() 只能无参数，或带一个名为 params 的参数"
            "（调参引擎写法：def build_transform(params=None)）",
        )
    has_return = any(
        isinstance(node, ast.Return) for node in ast.walk(build)
    )
    if not has_return:
        raise _RuleViolation(
            "entrypoint", "build_transform() 必须 return 一个 transform 函数"
        )
    return _build_transform_takes_params(build)


# ---------------------------------------------------------------------------
# 子进程冒烟
# ---------------------------------------------------------------------------


def _smoke_payload(
    takes_params: bool = False,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rng = random.Random(_SMOKE_SEED)
    factors = list(FACTOR_NAMES)
    z_values = {
        name: [round(rng.uniform(-2.5, 2.5), 6) for _ in range(_SMOKE_ROWS)]
        for name in factors
    }
    return {
        "factors": factors,
        "index": [f"6000{i:02d}" for i in range(_SMOKE_ROWS)],
        "weights": {name: round(1.0 / len(factors), 6) for name in factors},
        "z_values": z_values,
        "params": dict(params or {}),
        "build_transform_takes_params": bool(takes_params),
        "safe_builtins": sorted(SAFE_BUILTIN_NAMES),
        "allowed_import_roots": list(ALLOWED_IMPORT_ROOTS),
    }


def _smoke_base_dir() -> Path:
    """冒烟文件放在已有可写目录（不新建目录，避免沙箱对新建目录的 ACL 限制）。"""
    for base in (Path.cwd(), Path(__file__).resolve().parents[2],
                 Path(tempfile.gettempdir())):
        try:
            probe = base / (".dsh_write_probe_" + str(os.getpid()))
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return base
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def _run_smoke_test(
    source: str,
    *,
    takes_params: bool = False,
    params: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """在独立子进程中执行冒烟，返回 ``(是否通过, 说明)``。"""
    base = _smoke_base_dir()
    token = ".dsh_smoke_" + os.urandom(6).hex() + "_"
    candidate_path = base / (token + "candidate.py")
    payload_path = base / (token + "payload.json")
    driver_path = base / (token + "driver.py")
    result_path = base / (token + "result.json")
    try:
        candidate_path.write_text(source, encoding="utf-8")
        payload_path.write_text(
            json.dumps(
                _smoke_payload(takes_params=takes_params, params=params),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        driver_path.write_text(_SMOKE_DRIVER, encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            completed = subprocess.run(
                [sys.executable, str(driver_path), token],
                cwd=str(base),
                capture_output=True,
                timeout=SMOKE_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, f"子进程冒烟测试超时（超过 {SMOKE_TIMEOUT_SECONDS} 秒）"
        except OSError as exc:
            return False, f"无法启动冒烟测试子进程：{exc}"

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return False, f"冒烟测试结果无法读取：{exc}"
            if isinstance(result, dict):
                return bool(result.get("ok")), str(
                    result.get("detail") or "子进程冒烟测试失败"
                )

        stderr = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        stdout = (completed.stdout or b"").decode("utf-8", errors="replace").strip()
        output = (stderr or stdout)[-800:]
        return False, (
            f"冒烟测试子进程异常退出（returncode={completed.returncode}）："
            f"{output or '无输出'}"
        )
    finally:
        for path in (candidate_path, payload_path, driver_path, result_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 公开函数
# ---------------------------------------------------------------------------


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """把调参参数转成可 JSON 序列化的数值 dict。"""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ModelCodeError("params 必须是 dict（参数名 -> 数值）")
    cleaned: dict[str, Any] = {}
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ModelCodeError(f"params[{name!r}] 必须是数值")
        if isinstance(value, Integral):
            cleaned[str(name)] = int(value)
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ModelCodeError(f"params[{name!r}] 必须是有限数值")
        cleaned[str(name)] = number
    return cleaned


def validate_code(
    code: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """AST 白名单 + 子进程冒烟校验 AI 生成的模型源码。

    支持两种入口：``def build_transform()`` 或
    ``def build_transform(params=None)``（参数名必须叫 params，最多一个）。
    ``params`` 会传给参数化引擎的 ``build_transform`` 并用于冒烟测试。

    Returns
    -------
    dict
        ``{"ok": True, "checks": [{"name", "status", "detail"}, ...],
        "imports": [...], "takes_params": bool}``。

    Raises
    ------
    ModelCodeError
        任一层校验不通过；异常的 ``checks`` / ``imports`` 属性保留明细。
    """
    cleaned_params = _clean_params(params)
    checks: list[dict[str, str]] = []
    imports: list[str] = []

    def record(name: str, detail: str) -> None:
        checks.append({"name": name, "status": "passed", "detail": detail})

    def fail(name: str, detail: str) -> None:
        checks.append({"name": name, "status": "failed", "detail": detail})
        raise ModelCodeError(
            f"模型代码未通过校验（{name}）：{detail}",
            checks=list(checks),
            imports=list(imports),
        )

    if not isinstance(code, str):
        fail("source_size", "模型代码必须是字符串")
    source = code.strip()
    if not source:
        fail("source_size", "模型代码为空")
    source_limit = _env_limit("AI_MAX_SOURCE_CHARS", MAX_SOURCE_CHARS)
    if len(source) > source_limit:
        fail(
            "source_size",
            f"源码长度 {len(source)} 字符，超过上限 {source_limit}",
        )
    record("source_size", f"源码长度 {len(source)} / {source_limit}")

    try:
        tree = ast.parse(source, filename="<ai_model>")
    except SyntaxError as exc:
        fail("syntax", f"第 {exc.lineno} 行语法错误：{exc.msg}")
    record("syntax", "Python 语法解析通过")

    try:
        imports = _check_imports(tree)
    except _RuleViolation as exc:
        fail(exc.check, exc.detail)
    record("imports", "、".join(imports) if imports else "未使用 import")

    try:
        _check_module_structure(tree)
        _check_ast_safety(tree)
    except _RuleViolation as exc:
        fail(exc.check, exc.detail)
    record("ast_safety", "AST 白名单通过：无危险导入/调用/属性访问")

    try:
        takes_params = _check_entrypoint(tree)
    except _RuleViolation as exc:
        fail(exc.check, exc.detail)
    record(
        "entrypoint",
        "已定义 build_transform(params) 且带 return"
        if takes_params
        else "已定义 build_transform() 且带 return",
    )

    ok, detail = _run_smoke_test(
        source, takes_params=takes_params, params=cleaned_params
    )
    if not ok:
        fail("smoke", detail)
    record("smoke", detail or "子进程冒烟测试通过")

    return {
        "ok": True,
        "checks": checks,
        "imports": imports,
        "takes_params": takes_params,
    }


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """受限 ``__import__``：只放行 math / numpy / pandas / sklearn / scipy。"""
    root = str(name).split(".")[0]
    if level or root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"不允许导入模块: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _restricted_builtins(import_hook=None) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for name in SAFE_BUILTIN_NAMES:
        value = getattr(builtins, name, None)
        if value is not None:
            allowed[name] = value
    allowed["__import__"] = import_hook or _safe_import
    return allowed


def compile_code(
    code: str,
    params: Mapping[str, Any] | None = None,
    *,
    validate: bool = True,
) -> Callable[..., Any]:
    """编译模型代码，返回 ``transform(z, weights)``。

    Parameters
    ----------
    code:
        模型源码；``def build_transform()`` 或
        ``def build_transform(params=None)``。
    params:
        调参参数；参数化引擎编译时作为 ``build_transform(params)`` 的入参。
    validate:
        默认 ``True`` 会先做 AST + 子进程冒烟校验；服务端网格寻优复用同一份
        已校验代码时可传 ``False`` 跳过重复校验（不重复启动冒烟子进程）。

    Raises
    ------
    ModelCodeError
        校验、编译或 ``build_transform()`` 执行失败。
    """
    cleaned_params = _clean_params(params)
    if validate:
        info = validate_code(code, params=cleaned_params)
        takes_params = bool(info.get("takes_params"))
        source = code.strip()
    else:
        if not isinstance(code, str) or not code.strip():
            raise ModelCodeError("模型代码必须是非空字符串")
        source = code.strip()
        try:
            tree = ast.parse(source, filename="<ai_model>")
        except SyntaxError as exc:
            raise ModelCodeError(
                f"模型代码语法错误：第 {exc.lineno} 行 {exc.msg}"
            ) from exc
        build = _find_build_transform(tree)
        takes_params = build is not None and _build_transform_takes_params(build)

    namespace: dict[str, Any] = {
        "__builtins__": _restricted_builtins(),
        "__name__": "ai_custom_model",
    }
    try:
        exec(compile(source, "<ai_model>", "exec"), namespace)
    except Exception as exc:
        raise ModelCodeError(
            f"模型代码编译失败：{type(exc).__name__}: {exc}"
        ) from exc

    build_transform = namespace.get("build_transform")
    if not callable(build_transform):
        raise ModelCodeError("模型代码未提供可调用的 build_transform()")
    try:
        transform = (
            build_transform(cleaned_params) if takes_params else build_transform()
        )
    except Exception as exc:
        raise ModelCodeError(
            f"build_transform() 执行失败：{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(transform):
        raise ModelCodeError("build_transform() 未返回可调用的 transform(z, weights)")
    return transform


# ---------------------------------------------------------------------------
# AI 程序（write_model_program）：多文件模块 + 可选选股钩子 + 第三方库申请
# ---------------------------------------------------------------------------


class ProgramRuntime:
    """已编译的 AI 多文件程序运行时。"""

    def __init__(
        self,
        *,
        files: dict[str, str],
        entry: str,
        params: dict[str, Any],
        transform: Callable[..., Any],
        select_stocks: Callable[..., Any] | None,
        imports: list[str],
        modules: Mapping[str, types.ModuleType] | None = None,
    ) -> None:
        self.files = dict(files)
        self.entry = entry
        self.params = dict(params)
        self.transform = transform
        self.select_stocks = select_stocks
        self.imports = list(imports)
        self.modules = dict(modules or {})

    @property
    def accepts_context(self) -> bool:
        """transform 是否接受第三个 ``context`` 参数（供引擎判断是否构建 context）。"""
        return bool(getattr(self.transform, "accepts_context", False))

    def select(self, scores, top_n: int):
        """选股钩子：程序里定义了 select_stocks 时使用，否则 None。"""
        if self.select_stocks is None:
            return None
        return self.select_stocks(scores, int(top_n), dict(self.params))


def _program_module_name(path: str) -> str:
    """文件路径 -> 模块名：去掉 .py 后把 / 换成 .，包 __init__ 归到包名。"""
    stem = path[:-3] if path.endswith(".py") else path
    parts = stem.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _program_module_names(files: Mapping[str, str]) -> set[str]:
    return {_program_module_name(name) for name in files}


def _program_file_map(files: Mapping[str, str]) -> dict[str, str]:
    """模块名 -> 文件路径。"""
    return {_program_module_name(path): path for path in files}


def _program_ancestors(module_name: str) -> list[str]:
    """包的各级祖先包（pkg.sub.mod -> [pkg, pkg.sub]）。"""
    parts = module_name.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts))]


def _normalize_program_files(files: Mapping[str, Any] | list[Any]) -> dict[str, str]:
    if isinstance(files, Mapping):
        items = list(files.items())
    elif isinstance(files, (list, tuple)):
        items = []
        for item in files:
            if not isinstance(item, Mapping):
                raise ModelCodeError("files 每项必须是 {path, content} 对象")
            items.append((item.get("path"), item.get("content")))
    else:
        raise ModelCodeError("files 必须是文件列表或 {path: content} 字典")
    if not items:
        raise ModelCodeError("程序至少需要一个文件")
    file_count_limit = _env_limit("AI_MAX_PROGRAM_FILES", MAX_PROGRAM_FILES)
    if len(items) > file_count_limit:
        raise ModelCodeError(f"程序文件数 {len(items)} 超过上限 {file_count_limit}")
    total_limit = _env_limit("AI_MAX_PROGRAM_CHARS", MAX_PROGRAM_CHARS)
    file_char_limit = _env_limit(
        "AI_MAX_PROGRAM_FILE_CHARS", MAX_PROGRAM_FILE_CHARS)
    normalized: dict[str, str] = {}
    module_to_path: dict[str, str] = {}
    total = 0
    for path, content in items:
        name = str(path or "").strip()
        if "\\" in name or ".." in name or name.startswith("/") or re.match(
                r"^[A-Za-z]:", name):
            raise ModelCodeError(
                f"程序文件路径不合法：{name!r}；只允许 pkg/module.py 形式的分段路径"
                "（用 / 分隔，禁止 ..、绝对路径与反斜杠）")
        if not _PROGRAM_PATH_RE.match(name):
            raise ModelCodeError(
                f"程序文件路径不合法：{name!r}；每段必须是字母/数字/下划线组成的"
                "标识符并以 .py 结尾（如 helpers.py、pkg/helpers.py、pkg/__init__.py）")
        module_name = _program_module_name(name)
        if not module_name:
            raise ModelCodeError(
                f"程序文件路径不合法：{name!r}；__init__.py 必须位于包目录内")
        root = module_name.split(".")[0]
        if root in ALLOWED_IMPORT_ROOTS or root in sys.stdlib_module_names:
            raise ModelCodeError(f"程序文件名 {name!r} 与内置模块同名，请换一个名字")
        if name in normalized:
            raise ModelCodeError(f"程序文件路径重复：{name!r}")
        if module_name in module_to_path:
            raise ModelCodeError(
                f"程序文件 {name!r} 与 {module_to_path[module_name]!r} 的模块名重复"
                f"（{module_name}）")
        source = str(content if content is not None else "")
        if not source.strip():
            raise ModelCodeError(f"程序文件 {name} 内容为空")
        if len(source) > file_char_limit:
            raise ModelCodeError(
                f"程序文件 {name} 长度 {len(source)} 字符，"
                f"超过单文件上限 {file_char_limit}")
        total += len(source)
        normalized[name] = source
        module_to_path[module_name] = name
    if total > total_limit:
        raise ModelCodeError(f"程序总长度 {total} 字符超过上限 {total_limit}")
    return normalized


def _program_trees(files: Mapping[str, str]) -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for name, source in files.items():
        try:
            trees[name] = ast.parse(source, filename=name)
        except SyntaxError as exc:
            raise ModelCodeError(
                f"程序文件 {name} 第 {exc.lineno} 行语法错误：{exc.msg}") from exc
    return trees


def validate_program(
    files: Mapping[str, Any] | list[Any],
    entry: str | None = None,
    *,
    extra_roots: Sequence[str] = (),
) -> dict[str, Any]:
    """校验 AI 多文件程序：语法 + 导入白名单 + AST 安全 + 入口函数。

    支持多级包路径（``pkg/__init__.py``、``pkg/helpers.py``、``pkg/sub/mod.py``），
    模块名 = 去掉 ``.py`` 后把 ``/`` 换成 ``.``；导入白名单同时放行本程序内的模块。

    与旧 ``validate_code`` 不同：**不做子进程冒烟**（按产品要求），真正执行在
    网格寻优/回测阶段完成。
    """
    normalized = _normalize_program_files(files)
    entry_name = str(entry or PROGRAM_ENTRY_DEFAULT).strip()
    if entry_name not in normalized:
        raise ModelCodeError(f"程序入口文件不存在：{entry_name}")
    module_names = _program_module_names(normalized)
    trees = _program_trees(normalized)
    checks: list[dict[str, str]] = []
    all_imports: list[str] = []
    takes_params = False
    for name, tree in trees.items():
        try:
            imports = _check_imports(tree, program_modules=module_names,
                                     extra_roots=extra_roots)
            _check_module_structure(tree)
            _check_ast_safety(tree)
            if name == entry_name:
                takes_params = _check_entrypoint(tree)
        except _RuleViolation as exc:
            raise ModelCodeError(
                f"程序 {name} 未通过校验（{exc.check}）：{exc.detail}") from exc
        all_imports.extend(f"{name}: {item}" for item in imports)
        checks.append({"name": "ast_safety", "status": "passed",
                       "detail": f"{name}：语法/AST 白名单通过"})
    checks.append({"name": "entrypoint", "status": "passed",
                   "detail": f"{entry_name} 定义了 build_transform"})
    return {
        "ok": True,
        "checks": checks,
        "imports": all_imports,
        "files": sorted(normalized),
        "modules": sorted(module_names),
        "entry": entry_name,
        "takes_params": takes_params,
        "extra_roots": sorted(set(extra_roots)),
    }


def _program_dependency_order(
    trees: Mapping[str, ast.Module], module_names: set[str]
) -> list[str]:
    """按 import 依赖返回模块执行顺序（返回模块名，如 pkg.helpers / main）。"""
    stems = {_program_module_name(path): tree for path, tree in trees.items()}
    dependencies: dict[str, set[str]] = {name: set() for name in stems}
    for name, tree in stems.items():
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                module = node.module or ""
                if module:
                    targets.append(module)
                for alias in node.names:
                    if alias.name != "*":
                        targets.append(
                            f"{module}.{alias.name}" if module else alias.name)
            for target in targets:
                if target == name or target not in module_names:
                    continue
                # 子模块 import 自己的祖先包（from pkg import X）属于正常初始化顺序：
                # 祖先包本来就会先执行，不算循环依赖。
                if name.startswith(target + "."):
                    continue
                dependencies[name].add(target)
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ModelCodeError(f"程序模块存在循环导入：{name}")
        visiting.add(name)
        for dependency in sorted(dependencies.get(name, ())):
            visit(dependency)
        visiting.discard(name)
        visited.add(name)
        ordered.append(name)

    for name in stems:
        visit(name)
    return ordered


def _transform_accepts_context(transform: Callable[..., Any]) -> bool:
    """判断用户 transform 是否接受第三个位置参数（context）。"""
    try:
        signature = inspect.signature(transform)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind(None, None, None)
    except TypeError:
        return False
    return True


def _wrap_transform_context(transform: Callable[..., Any]) -> Callable[..., Any]:
    """把 transform 包装成 ``transform(z, weights, context=None)``。

    用户 transform 接受第 3 个参数时把 context 原样透传，否则仍按旧的 2 参方式调用；
    包装函数上设置 ``accepts_context`` 属性，供引擎判断是否需要构建 context。
    """
    accepts_context = _transform_accepts_context(transform)

    @functools.wraps(transform)
    def wrapped(z, weights, context=None):
        if accepts_context:
            return transform(z, weights, context)
        return transform(z, weights)

    wrapped.accepts_context = accepts_context  # type: ignore[attr-defined]
    return wrapped


def compile_program(
    files: Mapping[str, Any] | list[Any],
    entry: str | None = None,
    params: Mapping[str, Any] | None = None,
    *,
    validate: bool = True,
    extra_roots: Sequence[str] = (),
    extra_paths: Sequence[Path | str] = (),
) -> ProgramRuntime:
    """编译 AI 多文件程序（支持多级包），返回 ``ProgramRuntime``。

    - 入口默认 ``main.py``，入口模块顶层必须定义 ``build_transform``；
    - ``pkg/__init__.py`` 等包 __init__ 会先执行，子模块挂到父模块属性上；
    - 支持 ``import pkg.helpers`` / ``from pkg.helpers import x`` /
      ``from pkg import helpers``；
    - 返回的 transform 统一包装为 ``transform(z, weights, context=None)``，并在包装
      函数上标记 ``accepts_context`` 供引擎判断是否需要构建 context。

    ``extra_roots`` 是审核通过、已安装到 ``data/ai_packages`` 的第三方 import 根名；
    ``extra_paths`` 是这些包所在的目录（会加入 sys.path）。
    """
    cleaned_params = _clean_params(params)
    normalized = _normalize_program_files(files)
    if validate:
        info = validate_program(normalized, entry, extra_roots=extra_roots)
    else:
        info = {"entry": str(entry or PROGRAM_ENTRY_DEFAULT),
                "imports": [], "takes_params": True}
    entry_name = str(info.get("entry") or entry or PROGRAM_ENTRY_DEFAULT).strip()
    if entry_name not in normalized:
        raise ModelCodeError(f"程序入口文件不存在：{entry_name}")
    entry_module_name = _program_module_name(entry_name)
    module_names = _program_module_names(normalized)
    file_map = _program_file_map(normalized)
    trees = _program_trees(normalized)
    takes_params = bool(info.get("takes_params"))
    if not validate:
        try:
            takes_params = _check_entrypoint(trees[entry_name])
        except _RuleViolation as exc:
            raise ModelCodeError(f"程序入口检查失败：{exc.detail}") from exc
    for path in extra_paths or ():
        path_text = str(path)
        if path_text and path_text not in sys.path:
            sys.path.insert(0, path_text)
    allowed_roots = set(ALLOWED_IMPORT_ROOTS) | set(extra_roots)

    # 目录形式但没有 __init__.py 的包按隐式命名空间包处理
    implicit_packages: set[str] = set()
    for module_name in module_names:
        for ancestor in _program_ancestors(module_name):
            if ancestor not in module_names:
                implicit_packages.add(ancestor)
    node_names = set(module_names) | implicit_packages
    program_roots = {name.split(".")[0] for name in node_names}
    modules: dict[str, types.ModuleType] = {}
    executed: set[str] = set()
    executing: set[str] = set()

    def _execute(module_name: str) -> None:
        """按需执行程序模块（包 __init__ 先于子模块）。"""
        if module_name in executed:
            return
        for ancestor in _program_ancestors(module_name):
            if ancestor in executed or ancestor in executing:
                continue  # 父包正在初始化，允许导入其子模块（与 Python 行为一致）
            _execute(ancestor)
            if module_name in executed:
                return
        if module_name in executed:
            return
        if module_name in executing:
            raise ModelCodeError(f"程序模块存在循环导入：{module_name}")
        if module_name not in file_map:
            executed.add(module_name)  # 隐式命名空间包没有代码可执行
            return
        file_name = file_map[module_name]
        executing.add(module_name)
        try:
            exec(compile(normalized[file_name], file_name, "exec"),
                 modules[module_name].__dict__)
        except ModelCodeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelCodeError(
                f"程序模块 {file_name} 执行失败：{type(exc).__name__}: {exc}") from exc
        finally:
            executing.discard(module_name)
        executed.add(module_name)

    def _program_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise ImportError("程序内不允许相对导入")
        text = str(name)
        root = text.split(".")[0]
        if root in program_roots:
            if text not in modules:
                raise ImportError(f"程序内不存在模块：{text}")
            if text not in executed and text not in executing:
                _execute(text)
            for item in fromlist or ():
                child = f"{text}.{item}"
                if child in modules and child not in executed \
                        and child not in executing:
                    _execute(child)
            if fromlist:
                return modules[text]
            return modules[root]
        if root not in allowed_roots:
            raise ImportError(
                f"不允许导入模块: {name}；如需第三方库请先向审核申请安装")
        return builtins.__import__(name, globals, locals, fromlist, level)

    # 1) 创建隐式命名空间包（无代码，只提供包结构与属性挂载点）
    for package_name in sorted(implicit_packages, key=lambda item: item.count(".")):
        package = types.ModuleType(package_name)
        package.__dict__["__builtins__"] = _restricted_builtins(_program_import)
        package.__dict__["__file__"] = None
        package.__dict__["__package__"] = (
            package_name.rsplit(".", 1)[0] if "." in package_name else "")
        package.__dict__["__path__"] = []
        modules[package_name] = package
    # 2) 创建所有显式模块（先只建对象，执行顺序交给 import 依赖）
    for module_name in sorted(module_names, key=lambda item: item.count(".")):
        file_name = file_map[module_name]
        module = types.ModuleType(module_name)
        module.__dict__["__builtins__"] = _restricted_builtins(_program_import)
        module.__dict__["__file__"] = file_name
        module.__dict__["__package__"] = (
            module_name.rsplit(".", 1)[0] if "." in module_name else "")
        if any(other.startswith(module_name + ".")
               for other in node_names):
            module.__dict__["__path__"] = []
        modules[module_name] = module
    # 3) 子模块挂到父模块属性上（from pkg import helpers 依赖该属性）
    for node_name in sorted(node_names, key=lambda item: item.count(".")):
        if "." not in node_name:
            continue
        parent_name, leaf = node_name.rsplit(".", 1)
        parent = modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, modules[node_name])
    # 4) 执行全部模块（懒执行 + 依赖顺序兜底）
    for module_name in _program_dependency_order(trees, module_names):
        _execute(module_name)
    for module_name in sorted(module_names):
        _execute(module_name)
    entry_module = modules.get(entry_module_name)
    if entry_module is None:
        raise ModelCodeError(f"程序入口模块不存在：{entry_module_name}")
    build_transform = entry_module.__dict__.get("build_transform")
    if not callable(build_transform):
        raise ModelCodeError("程序入口未提供可调用的 build_transform()")
    try:
        transform = (
            build_transform(cleaned_params) if takes_params else build_transform())
    except Exception as exc:  # noqa: BLE001
        raise ModelCodeError(
            f"build_transform() 执行失败：{type(exc).__name__}: {exc}") from exc
    if not callable(transform):
        raise ModelCodeError(
            "build_transform() 必须返回可调用的 transform(z, weights, context=None)")
    select_raw = entry_module.__dict__.get("select_stocks")
    select_hook = select_raw if callable(select_raw) else None
    return ProgramRuntime(
        files=normalized,
        entry=entry_name,
        params=cleaned_params,
        transform=_wrap_transform_context(transform),
        select_stocks=select_hook,
        imports=list(info.get("imports") or []),
        modules=modules,
    )


# ---------------------------------------------------------------------------
# 第三方库：审核通过后的安装（data/ai_packages，data 目录不受 git 部署覆盖）
# ---------------------------------------------------------------------------


def normalize_package_specs(packages: Sequence[str] | None) -> list[str]:
    """校验 AI 申请的 PyPI 包名/版本约束，最多 ``MAX_PACKAGES`` 个（默认 16）。

    保留包名/版本约束格式校验与 pip 安全校验，拒绝 URL/路径/git 形式。
    """
    if not packages:
        return []
    if not isinstance(packages, (list, tuple)):
        raise ModelCodeError("packages 必须是字符串列表")
    limit = _env_limit("AI_MAX_PACKAGES", MAX_PACKAGES)
    if len(packages) > limit:
        raise ModelCodeError(f"一次最多申请安装 {limit} 个第三方库")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in packages:
        text = str(raw or "").strip()
        match = _PACKAGE_SPEC_RE.match(text)
        if not match:
            raise ModelCodeError(
                f"第三方库名不合法：{text!r}；只允许 PyPI 包名 + 版本约束，不支持 URL/路径/git")
        name = match.group("name").lower()
        if name in {"pip", "setuptools", "wheel"}:
            raise ModelCodeError(f"不允许通过该接口安装基础工具包：{text}")
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(text)
    return cleaned


def package_import_roots(packages: Sequence[str] | None) -> list[str]:
    """把包名映射成 import 根名（供程序导入白名单使用）。"""
    roots: set[str] = set()
    for spec in normalize_package_specs(packages):
        base = spec.split("[")[0]
        base = re.split(r"[=<>!~]", base)[0].strip().lower()
        roots.add(PACKAGE_ROOT_ALIASES.get(base, base.replace("-", "_").replace(".", "_")))
    return sorted(roots)


def ai_packages_dir(data_dir: Path | str | None = None) -> Path:
    """AI 审核通过后安装的第三方库目录（默认 data/ai_packages）。"""
    if data_dir is not None:
        return Path(data_dir) / "ai_packages"
    from ..config import DATA_DIR
    return Path(DATA_DIR) / "ai_packages"


def install_packages(
    packages: Sequence[str] | None,
    *,
    target_dir: Path | str | None = None,
    timeout: int = PIP_INSTALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """安装审核通过的第三方库到 ``data/ai_packages``，并把清单写入 manifest。"""
    specs = normalize_package_specs(packages)
    if not specs:
        return {"installed": [], "roots": [], "target": "", "output": ""}
    target = Path(target_dir) if target_dir is not None else ai_packages_dir()
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PIP_NO_INPUT", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable, "-m", "pip", "install",
        "--target", str(target), "--upgrade", "--no-cache-dir",
        "--disable-pip-version-check", *specs,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ModelCodeError(
            f"安装第三方库超时（>{timeout}s）：{', '.join(specs)}") from exc
    except OSError as exc:
        raise ModelCodeError(f"无法启动 pip 安装：{exc}") from exc
    output = ((completed.stdout or b"") + (completed.stderr or b"")).decode(
        "utf-8", errors="replace")[-1500:]
    if completed.returncode != 0:
        raise ModelCodeError(
            f"第三方库安装失败（{', '.join(specs)}）：{output or 'pip 返回非零退出码'}")
    manifest = target / "manifest.json"
    previous: list[str] = []
    if manifest.exists():
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                previous = [str(item) for item in loaded]
        except (OSError, json.JSONDecodeError):
            previous = []
    merged = sorted(set(previous) | set(specs))
    try:
        manifest.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    except OSError:
        pass
    return {
        "installed": specs,
        "roots": package_import_roots(specs),
        "target": str(target),
        "output": output,
    }
