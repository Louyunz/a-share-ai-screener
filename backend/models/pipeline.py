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
import os
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

# 允许的引擎类型（调参方案只产出 weights / model_type；code 引擎改由
# ``write_model_program`` 工具直接编写）
APPROACHES = ("model_type", "weights", "code")
OBJECTIVES = ("sharpe", "cagr", "mdd", "calmar", "sortino", "ir", "win_rate")

# 调参请求状态文本上限（默认 60000 字符），可用 TUNING_STATE_MAX_CHARS 覆盖
MAX_STATE_CHARS = 60000

# 兼容别名：param_space 参数个数已不再设硬上限（>=1 即可），该常量仅保留给
# ``backend.models`` 的导出/旧引用，不再参与任何校验。
MAX_TUNING_PARAMS: int | None = None
# 最少搜索因子数：允许只搜索 1 个因子
MIN_SEARCH_FACTORS = 1
# AI 未给 max_trials 时的默认评估预算
DEFAULT_TUNING_MAX_TRIALS = 32
# 调参方案自动修复次数上限（默认 3），可用 TUNING_MAX_REPAIR_ATTEMPTS 覆盖
MAX_TUNING_REPAIR_ATTEMPTS = 3
_PARAM_TYPES = ("int", "float")


def _env_positive_int(name: str, default: int) -> int:
    """读取正整数环境变量；缺省/非法/<=0 时退回默认值。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    return value if value >= 1 else int(default)


def _state_char_limit() -> int:
    """当前生效的状态文本上限（每次调用读取环境变量）。"""
    return _env_positive_int("TUNING_STATE_MAX_CHARS", MAX_STATE_CHARS)


def _repair_attempt_limit() -> int:
    """当前生效的调参方案自动修复次数上限（每次调用读取环境变量）。"""
    return _env_positive_int("TUNING_MAX_REPAIR_ATTEMPTS", MAX_TUNING_REPAIR_ATTEMPTS)


_TUNING_OUTPUT_SCHEMA = """{
  "approach": "weights | model_type 二选一（本方案不再产出 code 方案）",
  "summary": "一句普通人能看懂的话，说明为什么这样改",
  "why": "数据依据，必须引用状态里的真实指标数字，不要编造",
  "expected_effect": "预期改善方向",
  "risks": ["2~3 条主要风险，用普通人能懂的话"],
  "search_space": {"volatility_60": {"min": 0.05, "max": 0.2, "step": 0.05}, "...": "weights/model_type 方案必填，至少 1 个因子"},
  "model_type": "仅 model_type 方案必填，取内置模型 id",
  "objective": "sharpe | cagr | mdd | calmar | sortino | ir | win_rate，可选，默认 sharpe",
  "max_trials": "正整数，可选，默认 32；服务端有上限"
}"""


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
        "## 两种方案（approach 二选一）",
        '1. "weights"：保留当前 10 因子打分框架，只给 search_space（至少 1 个因子）。'
        "key 的顺序就是搜索优先级；每个因子给 {min,max,step}，满足 "
        f"0<=min<=max<={MAX_WEIGHT}、step>0，且 (max-min)/step>=1（至少 2 档）。"
        "weights 可选，作为基底权重；省略时用当前权重。",
        '2. "model_type"：切换到下方某个内置模型（model_type 必填），'
        "并给该模型权重的微调 search_space（至少 1 个因子，规则同上）。",
        '注意：本方案不产出 approach="code" 的自定义代码引擎；'
        "需要重写打分程序时由 write_model_program 工具直接编写，不要在这里输出 code / param_space。",
        "weights/model_type 方案不要给 code / param_space。",
        "",
        "## 执行指令（最高优先级，违反视为不合格）",
        "- state.directive 是 Planner 通过 tool_calls 明确给出的执行指令，必须遵守：",
        '  - directive.engine="weights" → 必须 approach="weights"；',
        '  - directive.engine="model_type" → 必须 approach="model_type"，且 model_type 必须等于 directive.model_type；',
        '  - directive.engine="auto" → 你只能根据数据在 weights / model_type 中选择，禁止输出 code；',
        "- directive.escalate=true（客户说继续优化/上一版未达预期）时："
        "若 directive.previous_approach=weights，必须使用 model_type 更换基础引擎；"
        "若 previous_approach=model_type/code，说明需要重写打分引擎，本方案不再处理。"
        "禁止 only-weights 的微调方案。",
        "- directive.objective 非空时必须使用该 objective；为空时才可自行判断。",
        "",
        "## 内置模型清单",
        model_lines,
        "",
        "## 可用因子（z 列已做方向对齐，数值越大代表越优）",
        f"{factor_lines}。",
        "",
        "## 其他字段",
        '- objective："sharpe"（夏普）| "cagr"（年化收益）| "mdd"（最大回撤）| '
        '"calmar"（卡玛）| "sortino"（索提诺）| "ir"（信息比率）| "win_rate"（月胜率），'
        "默认 sharpe，表示网格寻优的目标。",
        "- max_trials：正整数，表示最多回测多少组参数；未给时默认 32，服务端还有上限。",
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
3. 方案只允许 approach="weights" 或 "model_type"；不允许输出 code / param_space
   （自定义打分程序由 write_model_program 工具负责）；
4. weights/model_type 方案必须有 search_space：至少 1 个已知因子，每个 {min,max,step}，
   满足 0<=min<=max<=0.4、step>0、(max-min)/step>=1；
5. max_trials 必须是正整数（服务端有上限），objective 必须是允许的目标指标；
6. 不得删除必需字段，不得引入文件/网络/系统调用；不要解释修改过程；
7. repair_request.directive 是最高优先级执行指令：engine/model_type/objective/escalate
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
    """校验权重搜索范围：{因子名: {min,max,step}}，至少 1 个因子，保留 key 顺序。"""
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
    """校验代码引擎参数范围：>=1 个合法标识符参数（不设个数上限），保留 key 顺序。"""
    if not isinstance(value, Mapping):
        raise ModelValidationError(
            "param_space 必须是对象：{参数名: {min, max, step, type, default}}"
        )
    if len(value) < 1:
        raise ModelValidationError("param_space 至少需要 1 个参数")
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
    """调用 LLM 生成「引擎 + 调参范围」方案，并自动修复至多 3 次。

    说明：调参方案只产出 ``weights`` / ``model_type`` 两种 approach；
    需要重写打分引擎（``directive.engine="code"`` 或结构性升级）时直接报错，
    引导调用方改用 ``write_model_program`` 工具，不再由本函数生成代码方案。
    自动修复次数可用环境变量 ``TUNING_MAX_REPAIR_ATTEMPTS`` 覆盖（默认 3）。

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
        LLM 调用失败、返回无法解析、方案不合法或修复次数用尽。
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

    if required_engine == "code":
        raise ModelValidationError(
            "调参方案不再生成 code 引擎：请改用 write_model_program 工具直接编写模型程序；"
            "本工具只处理 weights / model_type 方案"
        )
    if escalate and previous_approach in ("model_type", "code"):
        raise ModelValidationError(
            "结构性升级需要重写打分引擎：请改用 write_model_program 工具；"
            "调参方案只支持 weights / model_type"
        )

    request_payload = {**_json_safe(state_payload), "tuning_request": True}
    state_text = json.dumps(request_payload, ensure_ascii=False)
    state_limit = _state_char_limit()
    if len(state_text) > state_limit:
        state_text = _truncate(state_text, state_limit)
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
            if plan["approach"] == "code":
                raise ModelValidationError(
                    'engine=auto 时不再允许 approach="code"（代码程序请改用 '
                    "write_model_program 工具）；请改出 weights 或 model_type 方案"
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
            if required_objective in OBJECTIVES:
                if plan["objective"] != required_objective:
                    raise ModelValidationError(
                        f"执行指令要求 objective={required_objective}，"
                        f"当前为 {plan['objective']!r}；请按指令重新输出"
                    )
            if escalate:
                if plan["approach"] == "weights":
                    raise ModelValidationError(
                        "执行指令要求结构性升级，禁止只调权重的 weights 方案；"
                        "请改用 model_type 更换基础引擎"
                    )
                if previous_approach == "weights" and plan["approach"] != "model_type":
                    raise ModelValidationError(
                        "上一版是权重模型，继续优化时必须使用 model_type 更换基础引擎"
                    )
            plan["repair_attempts"] = attempts
            return plan
        except ModelCodeError as exc:
            error = exc
        except ModelValidationError as exc:
            error = exc

        repair_limit = _repair_attempt_limit()
        if attempts >= repair_limit:
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
