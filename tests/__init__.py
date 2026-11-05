# -*- coding: utf-8 -*-
"""测试包与测试工作目录工具。

注意：Windows 下把 SQLite 放在系统临时目录可能被权限策略拦截，
因此测试数据统一放在项目内 tests/_tmp/（git 忽略）。
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

TMP_ROOT = Path(__file__).resolve().parent / "_tmp"


def workdir(prefix: str) -> Path:
    path = TMP_ROOT / f"{prefix}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
