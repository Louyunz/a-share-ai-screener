# -*- coding: utf-8 -*-
"""AI 生成模型源码的安全校验、子进程冒烟与受限编译执行。

校验分三层，任何一层不通过都抛 :class:`ModelCodeError`：

1. 规模与语法：源码 ≤ 6000 字符，且能被 ``ast.parse`` 解析；
2. AST 白名单：只允许 ``math`` / ``numpy`` / ``pandas`` / ``sklearn`` / ``scipy`` 导入，禁止
   文件、网络、系统调用与动态执行，禁止下划线属性访问；顶层只允许
   import、函数定义与常量赋值（禁止顶层副作用）；
3. 子进程冒烟：在临时目录、独立 Python 子进程中把 10 列 ``z_*`` 与
   权重喂给 ``build_transform()`` 返回的 transform，10 秒超时，校验返回
   ``pandas.Series``、长度一致且全部为有限数值。

支持参数化引擎（供服务端网格寻优）：``def build_transform(params=None)``，
参数只能叫 ``params`` 且最多一个；冒烟子进程会把 ``params`` 传进去。
``compile_code`` 复用同一套校验，并在受限 ``__builtins__`` 环境里执行代码；
``validate=False`` 时可跳过重复校验，直接复用已经校验过的代码。
"""

from __future__ import annotations

import ast
import builtins
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from ..engine.factors import FACTOR_NAMES

__all__ = [
    "MAX_SOURCE_CHARS",
    "SMOKE_TIMEOUT_SECONDS",
    "ModelValidationError",
    "ModelCodeError",
    "validate_code",
    "compile_code",
]

# 源码长度上限（契约 6.3）
MAX_SOURCE_CHARS = 6000
# 子进程冒烟超时（契约 6.3）
SMOKE_TIMEOUT_SECONDS = 10
# 冒烟用合成数据规模
_SMOKE_ROWS = 8
_SMOKE_SEED = 20240601

# 允许导入的模块根名
ALLOWED_IMPORT_ROOTS = ("math", "numpy", "pandas", "sklearn", "scipy")

# sklearn 只允许模型/预处理相关的常用子模块，避免 datasets 等带文件 IO 的模块
ALLOWED_SKLEARN_PREFIXES = (
    "sklearn.preprocessing", "sklearn.svm", "sklearn.ensemble", "sklearn.linear_model",
    "sklearn.cluster", "sklearn.decomposition", "sklearn.pipeline", "sklearn.impute",
    "sklearn.neighbors", "sklearn.tree", "sklearn.metrics", "sklearn.utils",
)

# 受限执行环境里保留的内建函数/异常（其余内建全部移除）
SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs", "all", "any", "bool", "callable", "chr", "complex", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
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
        "readlines", "load", "loads", "dump", "dumps", "save", "savez",
        "savez_compressed", "environ",
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
    }
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


def _module_allowed(name: str) -> bool:
    root = str(name).split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        return False
    if root == "sklearn":
        return name == "sklearn" or any(
            name == prefix or name.startswith(prefix + ".") for prefix in ALLOWED_SKLEARN_PREFIXES
        )
    return True


def _check_imports(tree: ast.Module) -> list[str]:
    """校验 import 白名单，返回人类可读的导入清单。"""
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if not _module_allowed(alias.name):
                    raise _RuleViolation(
                        "imports",
                        f"第 {node.lineno} 行禁止导入 {alias.name}；"
                        "只允许 math / numpy / pandas / sklearn（模型相关子模块）/ scipy",
                    )
                imports.append(
                    alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise _RuleViolation("imports", f"第 {node.lineno} 行禁止相对导入")
            module = node.module or ""
            root = module.split(".")[0]
            if not _module_allowed(module):
                raise _RuleViolation(
                    "imports",
                    f"第 {node.lineno} 行禁止从 {module or '未知模块'} 导入；"
                    "只允许 math / numpy / pandas / sklearn（模型相关子模块）/ scipy",
                )
            for alias in node.names:
                if alias.name == "*":
                    raise _RuleViolation(
                        "imports", f"第 {node.lineno} 行禁止 import *"
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
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
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
    return names


def _check_call(
    node: ast.Call, defined_functions: set[str], bound_names: set[str]
) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        name = func.id
        if name.startswith("__") or name in FORBIDDEN_NAMES:
            raise _RuleViolation("ast_safety", f"第 {node.lineno} 行禁止调用 {name}()")
        if name in SAFE_BUILTIN_NAMES or name in defined_functions or name in bound_names:
            return
        raise _RuleViolation(
            "ast_safety", f"第 {node.lineno} 行调用了未定义的函数 {name}()"
        )
    if isinstance(func, ast.Attribute):
        attr = func.attr
        if attr.startswith("_") or attr in FORBIDDEN_ATTRS:
            raise _RuleViolation("ast_safety", f"第 {node.lineno} 行禁止调用 .{attr}()")
        root: ast.expr = func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and (
            root.id.startswith("__") or root.id in FORBIDDEN_NAMES
        ):
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
    """遍历整棵 AST：禁止危险名字、下划线属性、危险方法调用与异步/类定义。"""
    defined_functions = _collect_defined_functions(tree)
    bound_names = _collect_bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                raise _RuleViolation(
                    "ast_safety", f"第 {node.lineno} 行禁止使用名称 {node.id}"
                )
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                raise _RuleViolation(
                    "ast_safety",
                    f"第 {node.lineno} 行禁止访问下划线属性 .{attr}（含 dunder）",
                )
            if attr in FORBIDDEN_ATTRS:
                raise _RuleViolation(
                    "ast_safety", f"第 {node.lineno} 行禁止访问属性 .{attr}"
                )
        elif isinstance(node, ast.Call):
            _check_call(node, defined_functions, bound_names)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise _RuleViolation(
                    "ast_safety", f"第 {node.lineno} 行不允许函数装饰器"
                )
        elif isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef)):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行不允许定义异步函数或类"
            )
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行不允许 global/nonlocal 声明"
            )
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            raise _RuleViolation(
                "ast_safety", f"第 {node.lineno} 行不允许生成器/异步语法"
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
    if len(source) > MAX_SOURCE_CHARS:
        fail(
            "source_size",
            f"源码长度 {len(source)} 字符，超过上限 {MAX_SOURCE_CHARS}",
        )
    record("source_size", f"源码长度 {len(source)} / {MAX_SOURCE_CHARS}")

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


def _restricted_builtins() -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for name in SAFE_BUILTIN_NAMES:
        value = getattr(builtins, name, None)
        if value is not None:
            allowed[name] = value
    allowed["__import__"] = _safe_import
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
