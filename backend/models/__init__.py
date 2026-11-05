# -*- coding: utf-8 -*-
"""AI 参与选股建模的公共入口。

包含三部分：
- 内置模型注册表（``registry``）：模型清单、权重校验；
- 自定义代码安全校验与编译（``custom``）：AST 白名单 + 子进程冒烟；
- 提案流水线（``pipeline``）：LLM 提案生成与结构化校验（不跑回测）。
"""

from __future__ import annotations

from .custom import (
    MAX_SOURCE_CHARS,
    SMOKE_TIMEOUT_SECONDS,
    ModelCodeError,
    ModelValidationError,
    compile_code,
    validate_code,
)
from .pipeline import (
    APPROACHES,
    build_candidate_proposal,
    describe_model,
    normalize_proposal,
)
from .registry import (
    DEFAULT_MODEL_ID,
    FACTOR_LABELS,
    MODEL_TYPES,
    ModelType,
    get_model,
    list_models,
    validate_weights,
)

__all__ = [
    # registry
    "MODEL_TYPES",
    "ModelType",
    "DEFAULT_MODEL_ID",
    "FACTOR_LABELS",
    "list_models",
    "get_model",
    "validate_weights",
    # custom code
    "MAX_SOURCE_CHARS",
    "SMOKE_TIMEOUT_SECONDS",
    "ModelValidationError",
    "ModelCodeError",
    "validate_code",
    "compile_code",
    # pipeline
    "APPROACHES",
    "normalize_proposal",
    "build_candidate_proposal",
    "describe_model",
]
