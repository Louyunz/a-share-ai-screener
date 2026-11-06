# -*- coding: utf-8 -*-
"""AI 建模提案流水线：LLM 生成提案 + 结构化校验。

职责边界：本模块只负责「生成提案」与「校验提案」，**不跑回测**；
回测与版本编排由 service 层完成。
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
    "normalize_proposal",
    "build_candidate_proposal",
    "normalize_tuning_plan",
    "build_tuning_plan",
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

_SKLEARN_SAMPLE = """import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def build_transform():
    def transform(z, weights):
        x = z.fillna(0.0)
        base = np.zeros(len(x), dtype=float)
        for factor, weight in weights.items():
            column = "z_" + factor
            if column in x.columns:
                base = base + float(weight) * x[column].to_numpy(dtype=float)
        model = RandomForestRegressor(n_estimators=60, max_depth=3, random_state=42)
        model.fit(x.to_numpy(dtype=float), base)
        return pd.Series(model.predict(x.to_numpy(dtype=float)), index=z.index)
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
        "若 state.requested_custom_model 非空，或客户点名内置清单之外的模型（如 SVM、随机森林、梯度提升、聚类、PCA 等），"
        "必须使用 approach=\"code\"，用 sklearn 在 transform 内实现该模型；不得拒绝、不得声称没有选项，也不得用内置模型替代。",
        "优化模型时不要默认只微调因子权重。若当前模型存在明显短板、客户对收益/回撤不满意、"
        "或 state.escalation.active 为 true，应优先考虑更换基础模型（approach=\"model_type\"）"
        "或用 sklearn 重写数学打分逻辑（approach=\"code\"）。",
        "当 state.escalation.active 为 true（客户点了继续优化/上一版未达预期）时，必须做结构性升级："
        "禁止 only-weights 的微调；若 strict_requirement 要求 code，则必须重写数学打分逻辑，"
        "并与 previous_proposal/previous_model 不同。",
        "在没有明确要求且当前模型表现尚可时，优先选择改动小、能解释清楚的方案；不要为了写代码而写代码。",
        "",
        "## 内置模型清单",
        model_lines,
        "",
        "",
        "## 可用因子",
        f"{factor_lines}。",
        "因子值（z 列）已经做过方向对齐：数值越大代表该因子越优（例如 volatility_60 越大代表波动越低）。",
        "",
        "## 自定义代码契约（approach=\"code\" 时必须遵守）",
        "- 必须定义顶层函数 def build_transform():，它返回可调用的 transform(z, weights)。",
        "- z 是 pandas.DataFrame，含 10 列 z_ep、z_bp、...，index 是股票代码；weights 是 {因子名: 权重} 的 dict。",
        "- transform 必须返回 pandas.Series，index 与 z 对齐、长度一致且不含 NaN/Inf。",
        "- 允许 import math、numpy、pandas，以及 sklearn / scipy 的模型与预处理子模块；"
        "禁止文件/网络/系统调用与动态执行（open/exec/eval/compile/__import__ 等）；"
        "禁止访问下划线开头的属性（含 dunder）；源码不超过 6000 字符；不要打印、不要读写文件、不要定义类。",
        "- sklearn 模型只能基于当前 z 截面做转导式/无监督拟合，或先用加权因子合成分构造伪标签再训练；"
        "不得假设存在未来收益标签，不得使用 sklearn.datasets 或任何文件读写。"
        "禁止导入 xgboost/lightgbm/torch/tensorflow 等未安装库；需要等价模型时用 sklearn 对应实现。",
        "- 参考写法（简单因子组合，实际代码顶部必须补上 import numpy as np 与 import pandas as pd）：",
        _CODE_SAMPLE,
        "- sklearn 参考写法（伪标签蒸馏，演示如何让 AI 自写任意模型）：",
        _SKLEARN_SAMPLE,
        "",
        "## 面向普通用户",
        "summary、why、expected_effect、risks 会直接展示给用户：用生活化语言，避免英文术语和公式；"
        "why 必须引用状态中的真实数字；risks 给 2~3 条。",
    ]
    return "\n".join(sections)


CODE_REPAIR_SYSTEM = """你是智投罗盘的量化模型代码修复助手。
给定原始提案、校验错误信息和客户诉求，请修复 Python 模型代码并重新输出完整提案 JSON。
要求：
1. 只输出一个 JSON 对象，字段与原始提案一致，approach 必须为 code；
2. 修复必须针对错误信息，不得删除安全检查要求、不得引入文件/网络/系统调用；
3. 代码必须定义 build_transform()，返回 transform(z, weights)，输出有限 pandas.Series；
4. 只允许 math/numpy/pandas/sklearn（模型相关子模块）/scipy；
5. 不要解释修改过程。
"""

# ---------------------------------------------------------------------------
# 调参方案（AI 只产出「引擎 + 搜索范围」，最终参数由服务端网格寻优决定）
# ---------------------------------------------------------------------------

OBJECTIVES = ("sharpe", "cagr", "mdd")
MIN_SEARCH_FACTORS = 3
MAX_TUNING_PARAMS = 4
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
5. 不得删除必需字段，不得引入文件/网络/系统调用；不要解释修改过程。
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

    original_raw = dict(raw)
    attempts = 0
    while True:
        try:
            proposal = normalize_proposal(dict(raw), base_weights=base_weights)
            if attempts:
                proposal["repair_attempts"] = attempts
            return proposal
        except ModelCodeError as exc:
            error = exc
        except ModelValidationError as exc:
            error = exc
        if attempts >= 2 or str(original_raw.get("approach", "")) != "code":
            if isinstance(error, ModelCodeError):
                raise ModelValidationError(
                    f"AI 生成的模型代码未通过校验（自动修复 {attempts} 次）：{error}；"
                    f"原始返回：{_snippet(original_raw)}"
                ) from error
            raise ModelValidationError(
                f"AI 返回的提案未通过校验（自动修复 {attempts} 次）：{error}；"
                f"原始返回：{_snippet(original_raw)}"
            ) from error
        attempts += 1
        repair_request = {
            "error": str(error),
            "original_proposal": _json_safe(raw),
            "user_request": state_payload.get("user_request", ""),
            "current_model": state_payload.get("current_model", {}),
            "current_metrics": state_payload.get("current_metrics", {}),
        }
        try:
            repaired = llm.chat_json(
                CODE_REPAIR_SYSTEM,
                [{"role": "user", "content": json.dumps(repair_request, ensure_ascii=False)}],
            )
        except Exception as exc:
            raise ModelValidationError(
                f"AI 代码修复调用失败：{type(exc).__name__}：{exc}"
            ) from exc
        if isinstance(repaired, Mapping) and "proposal" in repaired:
            repaired = repaired["proposal"]
        if not isinstance(repaired, Mapping):
            raise ModelValidationError("AI 代码修复返回的不是 JSON 对象")
        raw = dict(repaired)


# ---------------------------------------------------------------------------
# 调参方案校验 / 生成（AI 只产出「引擎 + 搜索范围」，寻优回测在 service 层）
# ---------------------------------------------------------------------------


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
