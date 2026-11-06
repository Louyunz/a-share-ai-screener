# -*- coding: utf-8 -*-
"""AI 调参方案流水线：LLM 只产出「引擎 + 搜索范围」+ 结构化校验。

职责边界：本模块只负责生成/校验调参方案，**不跑回测**；
具体参数由 ``backend.engine.tuning`` 在服务端寻优，回测与版本由 service 编排。
"""

from __future__ import annotations

import dataclasses
import json
import keyword
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

from .custom import (
    ModelCodeError,
    ModelValidationError,
    validate_code,
)
from .registry import (
    DEFAULT_MODEL_ID,
    FACTOR_LABELS,
    FACTOR_NAMES,
    MAX_WEIGHT,
    MODEL_TYPES,
    ModelType,
    get_model,
    list_models,
    validate_weights,
)

__all__ = [
    "APPROACHES",
    "OBJECTIVES",
    "MAX_TUNING_PARAMS",
    "TUNING_SYSTEM",
    "TUNING_REPAIR_SYSTEM",
    "normalize_tuning_plan",
    "build_tuning_plan",
]

# 允许的引擎类型
APPROACHES = ("model_type", "weights", "code")
OBJECTIVES = ("sharpe", "cagr", "mdd")

MAX_STATE_CHARS = 12000

MAX_TUNING_PARAMS = 4
MIN_SEARCH_FACTORS = 3
DEFAULT_TUNING_MAX_TRIALS = 12
MAX_TUNING_REPAIR_ATTEMPTS = 2
_PARAM_TYPES = ("int", "float")

_TUNING_OUTPUT_SCHEMA = """{
  "approach": "weights | model_type | code 三选一",
  "summary": "一句普通人能看懂的话，说明为什么这样改",
  "why": "数据依据，必须引用状态里的真实指标数字，不要编造",
  "expected_effect": "预期改善方向",
  "risks": ["2~3 条主要风险，用普通人能懂的话"],
  "search_space": {"volatility_60": {"min": 0.05, "max": 0.2, "step": 0.05}, "...": "weights/model_type 方案必填，至少 3 个因子"},
  "param_space": {"参数名": {"min": 0.0, "max": 0.6, "step": 0.2, "type": "float", "default": 0.5}, "...": "code 方案必填，1~4 个参数"},
  "model_type": "仅 model_type 方案必填，取内置模型 id",
  "code": "仅 code 方案必填：参数化引擎源码",
  "objective": "sharpe | cagr | mdd，可选，默认 sharpe",
  "max_trials": "整数 8~48，服务端还会再限制范围"
}"""

_TUNING_CODE_SAMPLE = """def build_transform(params=None):
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

    return transform"""


def _build_tuning_system_prompt() -> str:
    """构造「只出引擎 + 调参范围」的系统提示词。"""
    model_lines = "\n".join(
        f"- {item['id']}（{item['name']}）：{item['description']}"
        for item in list_models()
    )
    factor_lines = "、".join(
        f"{factor}（{FACTOR_LABELS.get(factor, factor)}）"
        for factor in FACTOR_LABELS
    )
    sections = [
        "你是「智投罗盘」的量化调参规划助手，只负责两件事：",
        "1. 设计或选择一个底层打分引擎（模型本身）；",
        "2. 给出服务端网格寻优要搜索的参数范围。",
        "最终采用哪组权重/参数，由服务端在范围内逐一回测后决定；"
        "你不要自己选定最终权重或参数值，也不要声称已经调好参。"
        "你输出的是「引擎 + 搜索范围」的 JSON 方案。",
        "",
        "## 输出格式",
        "只输出一个 JSON 对象，不要输出 Markdown 代码块、前后缀说明或多余文字，字段如下：",
        _TUNING_OUTPUT_SCHEMA,
        "",
        "## 三种方案（approach 三选一）",
        '1. "weights"：保留当前 10 因子打分框架，只给 search_space（至少 3 个因子）。'
        "key 的顺序就是搜索优先级；每个因子给 {min,max,step}，满足 "
        f"0<=min<=max<={MAX_WEIGHT}、step>0，且 (max-min)/step>=1（至少 2 档）。"
        "weights 可选，作为基底权重；省略时用当前权重。",
        '2. "model_type"：切换到下方某个内置模型（model_type 必填），'
        "并给该模型权重的微调 search_space（规则同上）。",
        '3. "code"：写一个参数化打分引擎，让服务端用 param_space 里的参数网格寻优。'
        "必须 def build_transform(params=None)：先 params = params or {}，"
        '再用 params.get("参数名", 默认值) 读取；返回 transform(z, weights) -> pandas.Series。'
        f"param_space 给 1~{MAX_TUNING_PARAMS} 个参数，每个 "
        '{min,max,step,type:"int"|"float",default}，min<=max、step>0、(max-min)/step>=1、'
        "default 在范围内；int 类型要求 min/max/step/default 都是整数且 step>=1。"
        "code 方案不要给 search_space；weights 可选，作为基底权重（省略时用当前权重）。",
        "weights/model_type 方案不要给 param_space。",
        "",
        "## 执行指令（最高优先级，违反视为不合格）",
        "- state.directive 是 Planner 通过 tool_calls 明确给出的执行指令，必须遵守：",
        '  - directive.engine="code" → 必须 approach="code"，写新的参数化引擎，禁止用 model_type/weights 替代；',
        '  - directive.engine="model_type" → 必须 approach="model_type"，且 model_type 必须等于 directive.model_type；',
        '  - directive.engine="weights" → 必须 approach="weights"；',
        '  - directive.engine="auto" → 你可以根据数据选择 weights/model_type/code，但必须符合下面的升级规则。',
        "- directive.escalate=true（客户说继续优化/上一版未达预期）时："
        "若 directive.previous_approach=weights，必须使用 model_type 更换基础引擎；"
        "若 previous_approach=model_type 或 code，必须使用 code 重写参数化打分引擎；"
        "禁止 only-weights 的微调方案。",
        "- directive.objective 非空时必须使用该 objective；为空时才可自行判断。",
        "",
        "## 内置模型清单",
        model_lines,
        "",
        "## 可用因子（z 列已做方向对齐，数值越大代表越优）",
        f"{factor_lines}。",
        "",
        '## 自定义代码契约（approach="code" 时必须遵守）',
        "- build_transform 只能无参数或带一个名为 params 的参数；推荐 def build_transform(params=None)。",
        "- z：pandas.DataFrame，含 z_ep、z_bp、... 共 10 列，index 是股票代码；weights：dict[str, float]。",
        "- transform 必须返回 pandas.Series，index 与 z 对齐、长度一致、不含 NaN/Inf。",
        "- 允许 import math、numpy、pandas，以及 sklearn / scipy 的模型与预处理子模块；"
        "禁止文件/网络/系统调用与动态执行（open/exec/eval/compile/__import__ 等）；"
        "禁止访问下划线开头的属性（含 dunder）；源码不超过 6000 字符；不要打印、不要读写文件、不要定义类。",
        "- 引擎会被服务端用不同参数反复调用，必须把参数真正用于打分逻辑，不要忽略 params。",
        "- 参考写法（实际代码顶部必须补上 import numpy as np 与 import pandas as pd）：",
        _TUNING_CODE_SAMPLE,
        "",
        "## 其他字段",
        '- objective："sharpe" | "cagr" | "mdd"，默认 sharpe，表示网格寻优的目标。',
        "- max_trials：整数 8~48，表示最多回测多少组参数；服务端会再做一次范围限制。",
        "summary、why、expected_effect、risks 会直接展示给普通用户：用生活化语言，避免英文术语；"
        "why 必须引用状态中的真实数字；risks 给 2~3 条。",
    ]
    return "\n".join(sections)


TUNING_SYSTEM = _build_tuning_system_prompt()

TUNING_REPAIR_SYSTEM = """你是智投罗盘的调参方案修复助手。
给定原始调参方案、校验错误信息和当前模型状态，请修复并重新输出完整的调参方案 JSON。
要求：
1. 只输出一个 JSON 对象，不要输出 Markdown 代码块或解释文字；
2. 保留原方案的 approach（除非错误信息表明 approach 不合法），必须修复错误信息指出的字段；
3. weights/model_type 方案必须有 search_space：至少 3 个已知因子，每个 {min,max,step}，
   满足 0<=min<=max<=0.4、step>0、(max-min)/step>=1；
4. code 方案必须有 param_space（1~4 个合法标识符参数）和完整 code：
   def build_transform(params=None)，内部用 params.get(参数名, 默认值) 兜底；
   param_space 每个参数 {min,max,step,type,default}，范围与类型必须合法；
5. 不得删除必需字段，不得引入文件/网络/系统调用；不要解释修改过程；
6. repair_request.directive 是最高优先级执行指令：engine/model_type/objective/escalate
   必须严格遵守；escalate=true 时禁止只调权重的 weights 方案。
"""


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


def _as_real(value: Any, field: str) -> float:
    """把 JSON 数值转成有限 float；不合法抛 ``ModelValidationError``。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ModelValidationError(f"{field} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ModelValidationError(f"{field} 必须是有限数值")
    return number


def _grid_levels(minimum: float, maximum: float, step: float) -> int:
    """网格取值档数（含首尾端点）。"""
    return int(math.floor((maximum - minimum) / step + 1e-9)) + 1


def _normalize_search_space(value: Any) -> dict[str, dict[str, float]]:
    """校验权重搜索范围：{因子名: {min,max,step}}，至少 3 个因子，保留 key 顺序。"""
    if not isinstance(value, Mapping):
        raise ModelValidationError(
            "search_space 必须是对象：{因子名: {min, max, step}}"
        )
    normalized: dict[str, dict[str, float]] = {}
    for factor, spec in value.items():
        factor_name = str(factor)
        if factor_name not in FACTOR_NAMES:
            raise ModelValidationError(
                f"search_space 含未知因子 {factor_name!r}；可选: {', '.join(FACTOR_NAMES)}"
            )
        if not isinstance(spec, Mapping):
            raise ModelValidationError(
                f"search_space 因子 {factor_name} 的范围必须是 {{min, max, step}} 对象"
            )
        minimum = _as_real(spec.get("min"), f"search_space.{factor_name}.min")
        maximum = _as_real(spec.get("max"), f"search_space.{factor_name}.max")
        step = _as_real(spec.get("step"), f"search_space.{factor_name}.step")
        if minimum < 0 or maximum > MAX_WEIGHT:
            raise ModelValidationError(
                f"search_space 因子 {factor_name} 的范围必须在 0~{MAX_WEIGHT} 之内"
            )
        if minimum > maximum:
            raise ModelValidationError(
                f"search_space 因子 {factor_name} 的 min 不能大于 max"
            )
        if step <= 0:
            raise ModelValidationError(
                f"search_space 因子 {factor_name} 的 step 必须大于 0"
            )
        if _grid_levels(minimum, maximum, step) < 2:
            raise ModelValidationError(
                f"search_space 因子 {factor_name} 至少需要 2 档取值（max - min 不能小于 step）"
            )
        normalized[factor_name] = {
            "min": minimum,
            "max": maximum,
            "step": step,
        }
    if len(normalized) < MIN_SEARCH_FACTORS:
        raise ModelValidationError(
            f"search_space 至少需要 {MIN_SEARCH_FACTORS} 个因子"
            f"（当前 {len(normalized)} 个）"
        )
    return normalized


def _normalize_param_space(value: Any) -> dict[str, dict[str, Any]]:
    """校验代码引擎参数范围：1~4 个合法标识符参数，保留 key 顺序。"""
    if not isinstance(value, Mapping):
        raise ModelValidationError(
            "param_space 必须是对象：{参数名: {min, max, step, type, default}}"
        )
    if not 1 <= len(value) <= MAX_TUNING_PARAMS:
        raise ModelValidationError(
            f"param_space 需要 1~{MAX_TUNING_PARAMS} 个参数（当前 {len(value)} 个）"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, spec in value.items():
        param_name = str(name)
        if not param_name.isidentifier() or keyword.iskeyword(param_name):
            raise ModelValidationError(
                f"param_space 参数名 {param_name!r} 不是合法 Python 标识符"
            )
        if not isinstance(spec, Mapping):
            raise ModelValidationError(
                f"param_space 参数 {param_name} 的定义必须是对象"
            )
        param_type = spec.get("type", "float")
        if param_type not in _PARAM_TYPES:
            raise ModelValidationError(
                f"param_space 参数 {param_name} 的 type 必须是 int 或 float"
            )
        minimum = _as_real(spec.get("min"), f"param_space.{param_name}.min")
        maximum = _as_real(spec.get("max"), f"param_space.{param_name}.max")
        step = _as_real(spec.get("step"), f"param_space.{param_name}.step")
        if minimum > maximum:
            raise ModelValidationError(
                f"param_space 参数 {param_name} 的 min 不能大于 max"
            )
        if step <= 0:
            raise ModelValidationError(
                f"param_space 参数 {param_name} 的 step 必须大于 0"
            )
        if _grid_levels(minimum, maximum, step) < 2:
            raise ModelValidationError(
                f"param_space 参数 {param_name} 至少需要 2 档取值（max - min 不能小于 step）"
            )
        raw_default = spec.get("default")
        default = (
            minimum
            if raw_default is None
            else _as_real(raw_default, f"param_space.{param_name}.default")
        )
        if not minimum <= default <= maximum:
            raise ModelValidationError(
                f"param_space 参数 {param_name} 的 default 必须在 min~max 范围内"
            )
        if param_type == "int":
            for label, number in (
                ("min", minimum),
                ("max", maximum),
                ("step", step),
                ("default", default),
            ):
                if not float(number).is_integer():
                    raise ModelValidationError(
                        f"param_space 参数 {param_name} 是 int 类型，{label} 必须是整数"
                    )
            if step < 1:
                raise ModelValidationError(
                    f"param_space 参数 {param_name} 是 int 类型，step 必须是 >=1 的整数"
                )
            normalized[param_name] = {
                "min": int(minimum),
                "max": int(maximum),
                "step": int(step),
                "type": "int",
                "default": int(default),
            }
        else:
            normalized[param_name] = {
                "min": minimum,
                "max": maximum,
                "step": step,
                "type": "float",
                "default": default,
            }
    return normalized


def _default_params(param_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """取参数空间的默认值，供代码冒烟测试使用。"""
    return {name: spec["default"] for name, spec in param_space.items()}


def _normalize_objective(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "sharpe"
    if not isinstance(value, str):
        raise ModelValidationError("objective 必须是字符串")
    objective = value.strip().lower()
    if objective not in OBJECTIVES:
        raise ModelValidationError(
            f"objective 必须是 {' / '.join(OBJECTIVES)} 之一，当前为 {value!r}"
        )
    return objective


def _normalize_max_trials(value: Any) -> int:
    if value is None:
        return DEFAULT_TUNING_MAX_TRIALS
    if isinstance(value, bool):
        raise ModelValidationError("max_trials 必须是整数")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        number = int(value)
    else:
        raise ModelValidationError("max_trials 必须是整数")
    if number < 1:
        raise ModelValidationError("max_trials 必须 >= 1")
    return number


def _current_model_weights(current_model: Any) -> dict[str, float] | None:
    """从 current_model（模型 dict 或模型 id）取一份合法权重。"""
    if current_model is None:
        return None
    if isinstance(current_model, str):
        model_id = current_model.strip()
        return dict(MODEL_TYPES[model_id].weights) if model_id in MODEL_TYPES else None
    if isinstance(current_model, Mapping):
        raw_weights = current_model.get("weights")
        if raw_weights is not None:
            return _normalized_weights(raw_weights, "current_model.weights")
        for key in ("model_type", "model_id", "id", "name"):
            model_id = current_model.get(key)
            if isinstance(model_id, str) and model_id.strip() in MODEL_TYPES:
                return dict(MODEL_TYPES[model_id.strip()].weights)
    return None


def _resolve_tuning_base_weights(
    base_weights: Mapping[str, float] | None,
    current_model: Any,
    model_payload: Mapping[str, Any] | None,
) -> dict[str, float]:
    """调参基底权重：调用方传入 > 当前模型 > 指定模型自带 > 默认模型。"""
    if base_weights is not None:
        return _normalized_weights(base_weights, "base_weights")
    current = _current_model_weights(current_model)
    if current is not None:
        return current
    if model_payload is not None:
        return dict(model_payload["weights"])
    return get_model(DEFAULT_MODEL_ID)["weights"]


def _unwrap_plan(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """容忍 LLM 把方案包在 {"plan"/"proposal"/"tuning_plan": {...}} 里。"""
    for key in ("plan", "tuning_plan", "proposal"):
        inner = value.get(key)
        if isinstance(inner, Mapping):
            return inner
    return value


def normalize_tuning_plan(
    raw: dict[str, Any],
    base_weights: Mapping[str, float] | None = None,
    current_model: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """校验并结构化 LLM 的「引擎 + 调参范围」方案。

    Parameters
    ----------
    raw:
        LLM 返回的 JSON 对象，必须含 ``approach`` 与 ``summary``。
    base_weights:
        当前基底权重；``weights`` 方案与 ``code`` 方案省略 weights 时使用。
    current_model:
        当前模型（模型 dict 或模型 id），``base_weights`` 为空时从其取权重。

    Returns
    -------
    dict
        ``{approach, summary, model_type, weights, code, search_space,
        param_space, objective, max_trials, why, expected_effect, risks}``。
        ``weights`` 已补齐归一化；不用的 ``search_space`` / ``param_space``
        返回空 dict。

    Raises
    ------
    ModelValidationError
        字段缺失、范围非法、权重非法等。
    ModelCodeError
        ``approach="code"`` 时代码未通过静态校验或冒烟测试。
    """
    if not isinstance(raw, Mapping):
        raise ModelValidationError("调参方案必须是 JSON 对象（dict）")
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
            raise ModelValidationError(
                "approach=code 时必须提供非空的 code 源码字符串"
            )
        code = raw_code.strip()
    elif raw_code not in (None, ""):
        raise ModelValidationError(f"approach={approach} 时不应提供 code 字段")

    # 基底权重
    raw_weights = payload.get("weights")
    if raw_weights is not None:
        weights = _normalized_weights(raw_weights)
    elif approach == "model_type":
        weights = dict(model_payload["weights"])
    else:
        weights = _resolve_tuning_base_weights(
            base_weights, current_model, model_payload
        )

    search_space: dict[str, dict[str, float]] = {}
    param_space: dict[str, dict[str, Any]] = {}
    if approach == "code":
        if payload.get("search_space"):
            raise ModelValidationError(
                "approach=code 时不应提供 search_space（搜索范围请放在 param_space）"
            )
        param_space = _normalize_param_space(payload.get("param_space"))
        default_params = _default_params(param_space)
        info = validate_code(code, params=default_params)
        if not info.get("takes_params"):
            raise ModelValidationError(
                "code 方案的 build_transform 必须接收 params 参数"
                "（写法：def build_transform(params=None)），"
                "服务端会用 param_space 里的参数反复调用它"
            )
    else:
        if payload.get("param_space"):
            raise ModelValidationError(
                f"approach={approach} 时不应提供 param_space"
            )
        search_space = _normalize_search_space(payload.get("search_space"))

    objective = _normalize_objective(payload.get("objective"))
    max_trials = _normalize_max_trials(payload.get("max_trials"))

    return {
        "approach": approach,
        "summary": summary,
        "model_type": None if approach == "code" else model_type,
        "weights": weights,
        "code": code,
        "search_space": search_space,
        "param_space": param_space,
        "objective": objective,
        "max_trials": max_trials,
        "why": _optional_text(payload, "why"),
        "expected_effect": _optional_text(payload, "expected_effect"),
        "risks": _optional_risks(payload),
    }


def build_tuning_plan(llm: Any, state: dict[str, Any]) -> dict[str, Any]:
    """调用 LLM 生成「引擎 + 调参范围」方案，并自动修复至多 2 次。

    Parameters
    ----------
    llm:
        任何提供 ``chat_json(system, messages) -> dict`` 的客户端。
    state:
        当前模型/指标/因子统计等状态；``weights`` 会作为基底权重，
        ``current_model`` 用于兜底。整体会作为 ``tuning_request`` 发给 LLM。

    Returns
    -------
    dict
        :func:`normalize_tuning_plan` 的结构，额外带 ``repair_attempts``
        （自动修复次数，0 表示首次即通过）。

    Raises
    ------
    ModelValidationError
        LLM 调用失败、返回无法解析或修复 2 次后仍不合法。
    """
    if not isinstance(state, Mapping):
        raise ModelValidationError("state 必须是 dict")
    state_payload = dict(state)
    base_weights: dict[str, float] | None = None
    if state_payload.get("weights") is not None:
        base_weights = _normalized_weights(state_payload["weights"], "state.weights")
    current_model = state_payload.get("current_model")
    directive = state_payload.get("directive")
    if not isinstance(directive, Mapping):
        directive = {}
    required_engine = str(directive.get("engine") or "auto").strip().lower()
    required_model_type = str(directive.get("model_type") or "").strip()
    required_objective = str(directive.get("objective") or "").strip().lower()
    escalate = bool(directive.get("escalate"))
    previous_approach = str(directive.get("previous_approach") or "").strip()

    request_payload = {**_json_safe(state_payload), "tuning_request": True}
    state_text = json.dumps(request_payload, ensure_ascii=False)
    if len(state_text) > MAX_STATE_CHARS:
        state_text = _truncate(state_text, MAX_STATE_CHARS)
    messages = [{"role": "user", "content": state_text}]

    try:
        raw = llm.chat_json(TUNING_SYSTEM, messages)
    except Exception as exc:
        raise ModelValidationError(
            f"LLM 未能返回可解析的调参方案（{type(exc).__name__}）：{exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise ModelValidationError(
            f"LLM 返回的调参方案不是 JSON 对象：{_snippet(raw)}"
        )

    initial_raw = dict(_unwrap_plan(raw))
    current_raw = initial_raw
    attempts = 0
    while True:
        try:
            plan = normalize_tuning_plan(
                dict(current_raw),
                base_weights=base_weights,
                current_model=current_model,
            )
            if required_engine in APPROACHES and plan["approach"] != required_engine:
                raise ModelValidationError(
                    f"执行指令要求 approach={required_engine}，"
                    f"当前方案为 approach={plan['approach']}；请按指令重新输出"
                )
            if required_engine == "model_type" and required_model_type:
                if plan.get("model_type") != required_model_type:
                    raise ModelValidationError(
                        f"执行指令要求 model_type={required_model_type}，"
                        f"当前为 {plan.get('model_type')!r}；请按指令重新输出"
                    )
            if required_objective in ("sharpe", "cagr", "mdd"):
                if plan["objective"] != required_objective:
                    raise ModelValidationError(
                        f"执行指令要求 objective={required_objective}，"
                        f"当前为 {plan['objective']!r}；请按指令重新输出"
                    )
            if escalate:
                if plan["approach"] == "weights":
                    raise ModelValidationError(
                        "执行指令要求结构性升级，禁止只调权重的 weights 方案；"
                        "请按 previous_approach 规则改用 model_type 或 code"
                    )
                if previous_approach == "weights" and plan["approach"] != "model_type":
                    raise ModelValidationError(
                        "上一版是权重模型，继续优化时必须使用 model_type 更换基础引擎"
                    )
                if previous_approach in ("model_type", "code") and plan["approach"] != "code":
                    raise ModelValidationError(
                        "上一版是内置模型/代码引擎，继续优化时必须使用 code 重写打分引擎"
                    )
            plan["repair_attempts"] = attempts
            return plan
        except ModelCodeError as exc:
            error = exc
        except ModelValidationError as exc:
            error = exc

        if attempts >= MAX_TUNING_REPAIR_ATTEMPTS:
            raise ModelValidationError(
                f"AI 返回的调参方案未通过校验（自动修复 {attempts} 次）：{error}；"
                f"原始返回：{_snippet(initial_raw)}"
            ) from error
        attempts += 1
        repair_request = {
            "error": str(error),
            "original_plan": _json_safe(current_raw),
            "user_request": state_payload.get("user_request", ""),
            "current_model": state_payload.get("current_model", {}),
            "current_metrics": state_payload.get("current_metrics", {}),
            "directive": _json_safe(directive),
        }
        try:
            repaired = llm.chat_json(
                TUNING_REPAIR_SYSTEM,
                [
                    {
                        "role": "user",
                        "content": json.dumps(repair_request, ensure_ascii=False),
                    }
                ],
            )
        except Exception as exc:
            raise ModelValidationError(
                f"AI 调参方案修复调用失败：{type(exc).__name__}：{exc}"
            ) from exc
        if not isinstance(repaired, Mapping):
            raise ModelValidationError("AI 调参方案修复返回的不是 JSON 对象")
        current_raw = dict(_unwrap_plan(repaired))
