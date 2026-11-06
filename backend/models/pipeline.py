# -*- coding: utf-8 -*-
"""AI 建模提案流水线：LLM 生成提案 + 结构化校验。

职责边界：本模块只负责「生成提案」与「校验提案」，**不跑回测**；
回测与版本编排由 service 层完成。
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from typing import Any

from .custom import (
    ModelCodeError,
    ModelValidationError,
    validate_code,
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
    "APPROACHES",
    "normalize_proposal",
    "build_candidate_proposal",
    "describe_model",
]

# 允许的提案方式
APPROACHES = ("model_type", "weights", "code")

# state 必须包含的字段（契约 6.2 / 分工约定）
REQUIRED_STATE_KEYS = (
    "current_model",
    "current_metrics",
    "weights",
    "factor_stats",
    "recent_metrics",
)

# state 序列化进提示词的长度上限，避免上下文爆炸
MAX_STATE_CHARS = 12000

_OUTPUT_SCHEMA = """{
  "approach": "model_type | weights | code 三选一",
  "summary": "一句普通人能看懂的话，说明为什么这样改（不要出现夏普、因子暴露等术语）",
  "model_type": "切换到内置模型时必填，取下方清单里的 id；weights/code 方案可省略",
  "weights": {"ep": 0.12, "bp": 0.10, "...": "共 10 项，总和为 1，每项 0~0.4"},
  "code": "approach=code 时的完整 Python 源码；其他方案省略",
  "why": "数据依据，必须引用用户状态里的真实指标数字，不要编造",
  "expected_effect": "预期改善方向，一句话说清希望哪个指标变好",
  "risks": ["2~3 条主要风险，用普通人能懂的话"]
}"""

_CODE_SAMPLE = """def build_transform():
    def transform(z, weights):
        score = pd.Series(0.0, index=z.index, dtype=float)
        for factor, weight in weights.items():
            column = "z_" + factor
            if column in z.columns:
                values = pd.to_numeric(z[column], errors="coerce").fillna(0.0)
                score = score + float(weight) * values
        return score
    return transform"""


def _build_system_prompt() -> str:
    """构造建模助手系统提示词（模型清单与因子说明从注册表动态生成）。"""
    model_lines = "\n".join(
        f"- {item['id']}（{item['name']}）：{item['description']}"
        for item in list_models()
    )
    factor_lines = "、".join(
        f"{factor}（{FACTOR_LABELS.get(factor, factor)}）"
        for factor in FACTOR_LABELS
    )
    sections = [
        "你是「智投罗盘」的量化建模助手，负责在用户现有选股模型基础上提出可落地的改进方案。",
        "",
        "## 你的任务",
        "根据用户给出的当前模型、当前回测指标、10 因子暴露统计和近期表现，判断模型的主要短板，"
        "提出一个改进方案。只能使用下方给出的内置模型、10 个因子或符合代码契约的自定义代码，"
        "所有数字必须来自用户状态，禁止编造。",
        "",
        "## 输出格式",
        "只输出一个 JSON 对象，不要输出 Markdown 代码块、前后缀说明或多余文字，字段如下：",
        _OUTPUT_SCHEMA,
        "",
        "## 三种方案（approach 三选一）",
        '1. "model_type"：切换到更合适的模型风格；weights 省略时使用该内置模型自带权重。',
        '2. "weights"：保留当前模型风格，只调整 10 个因子的权重；weights 必填。',
        '3. "code"：生成自定义打分代码，适合内置方案都解决不了的情况；weights 作为基底权重'
        "（省略时沿用当前权重），code 必须通过静态安全检查和子进程冒烟测试。",
        "若 state.user_request 明确要求修改策略代码（如“改代码/修改源代码/build_transform/自定义代码”），"
        "必须使用 approach=\"code\"，不得用权重或内置模型替代；若 state.force_approach 非空，必须使用该 approach。",
        "在没有明确要求时，优先选择改动小、能解释清楚的方案；能用内置模型或权重调整解决时不要写代码。",
        "",
        "## 内置模型清单",
        model_lines,
        "",
        "## 可用因子",
        f"{factor_lines}。",
        "因子值（z 列）已经做过方向对齐：数值越大代表该因子越优（例如 volatility_60 越大代表波动越低）。",
        "",
        "## 自定义代码契约（approach=\"code\" 时必须遵守）",
        "- 必须定义顶层函数 def build_transform():，它返回可调用的 transform(z, weights)。",
        "- z 是 pandas.DataFrame，含 10 列 z_ep、z_bp、...，index 是股票代码；weights 是 {因子名: 权重} 的 dict。",
        "- transform 必须返回 pandas.Series，index 与 z 对齐、长度一致且不含 NaN/Inf。",
        "- 只允许 import math、import numpy as np、import pandas as pd；"
        "禁止文件/网络/系统调用与动态执行（open/exec/eval/compile/__import__ 等）；"
        "禁止访问下划线开头的属性（含 dunder）；源码不超过 6000 字符；不要打印、不要读写文件、不要定义类。",
        "- 参考写法（实际代码顶部必须补上 import numpy as np 与 import pandas as pd）：",
        _CODE_SAMPLE,
        "",
        "## 面向普通用户",
        "summary、why、expected_effect、risks 会直接展示给用户：用生活化语言，避免英文术语和公式；"
        "why 必须引用状态中的真实数字；risks 给 2~3 条。",
    ]
    return "\n".join(sections)


def _json_safe(value: Any) -> Any:
    """把 state 转成可 JSON 序列化的结构（NaN/Inf -> None，模型类型 -> dict）。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, ModelType):
        return _json_safe(value.to_dict(include_weights=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...（内容过长已截断）"


def _snippet(value: Any, limit: int = 400) -> str:
    """原始返回片段（用于错误信息）。"""
    try:
        text = json.dumps(_json_safe(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ModelValidationError("state 必须是 dict")
    payload = dict(state)
    missing = [key for key in REQUIRED_STATE_KEYS if key not in payload]
    if missing:
        raise ModelValidationError(
            f"state 缺少必要字段: {', '.join(missing)}；"
            f"需要包含: {', '.join(REQUIRED_STATE_KEYS)}"
        )
    return payload


def _state_base_weights(state: Mapping[str, Any]) -> dict[str, float]:
    try:
        return validate_weights(state["weights"])
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"state.weights 不合法：{exc}") from exc


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"提案缺少必填字段 {key}（非空字符串）")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ModelValidationError(f"提案字段 {key} 必须是字符串")
    return value.strip()


def _optional_risks(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("risks")
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError("提案字段 risks 必须是字符串数组")
    risks: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            risks.append(text)
    return risks


def _normalized_weights(weights: Any, field: str = "weights") -> dict[str, float]:
    try:
        return validate_weights(weights)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{field} 不合法：{exc}") from exc


def _resolve_base_weights(
    base_weights: Mapping[str, float] | None,
    model_payload: Mapping[str, Any] | None,
) -> dict[str, float]:
    """代码方案的基底权重：调用方传入 > 指定模型自带 > 默认模型。"""
    if base_weights is not None:
        return _normalized_weights(base_weights, "base_weights")
    if model_payload is not None:
        return dict(model_payload["weights"])
    return get_model(DEFAULT_MODEL_ID)["weights"]


def normalize_proposal(
    raw: dict[str, Any],
    base_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """校验并结构化 LLM 提案（契约 6.2）。

    Parameters
    ----------
    raw:
        LLM 返回的 JSON 对象，必须含 ``approach`` 与 ``summary``。
    base_weights:
        可选的当前基底权重；``approach="code"`` 且未提供 weights 时使用。

    Returns
    -------
    dict
        ``{approach, summary, model_type, weights, code, why,
        expected_effect, risks}``，其中 ``weights`` 已补齐并归一化。

    Raises
    ------
    ModelValidationError
        字段缺失/类型错误/未知模型/权重非法。
    ModelCodeError
        ``approach="code"`` 时代码未通过静态校验或冒烟测试。
    """
    if not isinstance(raw, Mapping):
        raise ModelValidationError("提案必须是 JSON 对象（dict）")
    payload = dict(raw)

    approach_raw = payload.get("approach")
    approach = str(approach_raw).strip().lower() if approach_raw is not None else ""
    if approach not in APPROACHES:
        raise ModelValidationError(
            f"approach 必须是 {' / '.join(APPROACHES)} 之一，当前为 {approach_raw!r}"
        )

    summary = _required_text(payload, "summary")

    # 模型类型：只要出现就必须合法；model_type 方案必须出现
    model_type: str | None = None
    model_payload: dict[str, Any] | None = None
    model_type_raw = payload.get("model_type")
    if model_type_raw is not None and str(model_type_raw).strip():
        if not isinstance(model_type_raw, str):
            raise ModelValidationError("model_type 必须是字符串")
        model_type = model_type_raw.strip()
        if model_type not in MODEL_TYPES:
            raise ModelValidationError(
                f"未知模型类型 {model_type!r}；可选: {', '.join(MODEL_TYPES)}"
            )
        model_payload = get_model(model_type)
    if approach == "model_type" and model_type is None:
        raise ModelValidationError("approach=model_type 时必须提供 model_type")

    # 代码方案：code 必填；其他方案不允许带 code
    code: str | None = None
    raw_code = payload.get("code")
    if approach == "code":
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise ModelValidationError("approach=code 时必须提供非空的 code 源码字符串")
        code = raw_code.strip()
    elif raw_code not in (None, ""):
        raise ModelValidationError(f"approach={approach} 时不应提供 code 字段")

    # 权重：三种方案最后都要有一份合法的 10 因子权重
    raw_weights = payload.get("weights")
    if approach == "code":
        if raw_weights is not None:
            weights = _normalized_weights(raw_weights)
        else:
            weights = _resolve_base_weights(base_weights, model_payload)
    elif approach == "model_type":
        if raw_weights is not None:
            weights = _normalized_weights(raw_weights)
        else:
            weights = dict(model_payload["weights"])
    else:  # approach == "weights"
        if raw_weights is None:
            raise ModelValidationError("approach=weights 时必须提供 weights")
        weights = _normalized_weights(raw_weights)

    # 代码方案最后做静态检查 + 子进程冒烟（耗时，放在权重校验之后）
    if code is not None:
        validate_code(code)

    return {
        "approach": approach,
        "summary": summary,
        "model_type": None if approach == "code" else model_type,
        "weights": weights,
        "code": code,
        "why": _optional_text(payload, "why"),
        "expected_effect": _optional_text(payload, "expected_effect"),
        "risks": _optional_risks(payload),
    }


def build_candidate_proposal(llm: Any, state: dict[str, Any]) -> dict[str, Any]:
    """调用 LLM 生成候选模型提案，并用 :func:`normalize_proposal` 校验。

    Parameters
    ----------
    llm:
        任何提供 ``chat_json(system, messages) -> dict`` 的客户端。
    state:
        至少包含 ``current_model`` / ``current_metrics`` / ``weights`` /
        ``factor_stats`` / ``recent_metrics``。

    Raises
    ------
    ModelValidationError
        state 不合法、LLM 调用失败、返回内容无法解析或提案校验不通过；
        错误信息包含原始返回片段。
    """
    state_payload = _validate_state(state)
    base_weights = _state_base_weights(state_payload)

    system_prompt = _build_system_prompt()
    state_text = _truncate(
        json.dumps(_json_safe(state_payload), ensure_ascii=False), MAX_STATE_CHARS
    )
    messages = [
        {
            "role": "user",
            "content": (
                "当前模型与近期表现（JSON）：\n"
                f"{state_text}\n\n"
                "请基于以上状态输出一个改进方案 JSON。"
            ),
        }
    ]

    try:
        raw = llm.chat_json(system_prompt, messages)
    except Exception as exc:
        raise ModelValidationError(
            f"LLM 未能返回可解析的提案（{type(exc).__name__}）：{exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise ModelValidationError(
            f"LLM 返回的提案不是 JSON 对象：{_snippet(raw)}"
        )

    try:
        return normalize_proposal(dict(raw), base_weights=base_weights)
    except ModelCodeError as exc:
        raise ModelValidationError(
            f"AI 生成的模型代码未通过校验：{exc}；原始返回：{_snippet(raw)}"
        ) from exc
    except ModelValidationError as exc:
        raise ModelValidationError(
            f"AI 返回的提案未通过校验：{exc}；原始返回：{_snippet(raw)}"
        ) from exc


def describe_model(model: Mapping[str, Any] | str) -> str:
    """把模型转成一句给普通用户看的中文描述。"""
    if isinstance(model, str):
        model = get_model(model)
    if not isinstance(model, Mapping):
        raise ModelValidationError("model 必须是模型 dict 或模型 id 字符串")

    name = str(model.get("name") or model.get("id") or "未命名模型").strip()
    description = str(model.get("description") or "").strip()
    text = f"{name}：{description}" if description else name

    weights = model.get("weights")
    highlights: list[tuple[float, str]] = []
    if isinstance(weights, Mapping):
        for factor, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            amount = float(value)
            if amount > 0:
                highlights.append((amount, str(factor)))
    if highlights:
        highlights.sort(reverse=True)
        top = "、".join(
            f"{FACTOR_LABELS.get(factor, factor)} {amount * 100:.0f}%"
            for amount, factor in highlights[:3]
        )
        text += f"（重点看：{top}）"
    return text
