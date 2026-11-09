# -*- coding: utf-8 -*-
"""对话动作（工具）契约：Pydantic 模型 + OpenAI function calling schema。

这是 Planner 与执行层之间的**唯一契约**：

- 运行时用同一份 Pydantic 模型校验 LLM 的 tool_calls 参数；
- 发给模型的 ``tools`` 参数由同一份模型生成的 JSON Schema 得到；
- 文档与测试直接引用 ``TOOL_DEFINITIONS``，避免提示词/代码/文档三处漂移。

服务端不再解析自然语言来判断“要换模型还是调参”；这个判断只由 LLM 在
tool_calls 的 args 中给出，服务端只做 schema 校验与确定性策略执行。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..models.registry import MODEL_TYPES
from ..schemas import SLOT_KEYS

__all__ = [
    "ACTION_NAMES",
    "AnalyzeModelArgs",
    "WriteModelProgramArgs",
    "ToolArgsError",
    "TOOL_DEFINITIONS",
    "validate_tool_call",
]

ACTION_NAMES = (
    "update_profile",
    "confirm_profile",
    "run_selection",
    "explain_results",
    "analyze_model",
    "write_model_program",
    "apply_pending_change",
    "rollback_version",
)

MODEL_TYPE_IDS = tuple(MODEL_TYPES)

# 可选优化目标：sharpe/cagr/mdd/calmar/sortino/ir/win_rate；空字符串=默认夏普
OBJECTIVE_KEYS = ("sharpe", "cagr", "mdd", "calmar", "sortino", "ir", "win_rate")

# AI 程序可声明的执行配置键（值由服务端校验，不在这里限制具体数值）
PROGRAM_CONFIG_KEYS = (
    "top_n", "backtest_years", "rebalance_days",
    "cost_buy", "cost_sell", "portfolio_method",
)


def _validate_objective(value: str) -> str:
    text = str(value or "").strip().lower()
    if text and text not in OBJECTIVE_KEYS:
        raise ValueError(f"未知优化目标 {value!r}；可选：{', '.join(OBJECTIVE_KEYS)} 或留空")
    return text


class ToolArgsError(ValueError):
    """LLM 的 tool_call 参数未通过 schema 校验。"""

    def __init__(self, tool: str, detail: str) -> None:
        super().__init__(f"{tool} 参数不合法：{detail}")
        self.tool = tool
        self.detail = detail


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EmptyArgs(_StrictModel):
    """无参数动作。"""


class ExplainResultsArgs(_StrictModel):
    question: str = Field(default="", max_length=500)

    def clean(self) -> dict[str, Any]:
        return {"question": self.question.strip()[:500]}


class ApplyPendingChangeArgs(_StrictModel):
    """启用待审批的候选方案。

    ``candidate`` 是候选方案编号（按方案出现顺序从 1 开始）；0 表示不指定，
    默认启用最新的待审批方案。客户在一轮里生成了多个候选方案时，需要靠这个
    参数区分“用第一个/第二个”，服务端不解析自然语言。
    """

    candidate: int = Field(default=0, ge=0)

    def clean(self) -> dict[str, Any]:
        return {"candidate": int(self.candidate)}


class UpdateProfileArgs(_StrictModel):
    """客户本轮新提供的画像信息（工具调用，不再混在回复 JSON 里）。"""

    slots: dict[str, str] = Field(default_factory=dict)

    @field_validator("slots")
    @classmethod
    def _check_slots(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, raw in value.items():
            if key not in SLOT_KEYS:
                raise ValueError(f"未知画像槽位 {key!r}")
            text = str(raw).strip()
            if text:
                cleaned[key] = text[:200]
        return cleaned


class AnalyzeModelArgs(_StrictModel):
    """模型优化动作：AI 必须明确本轮用哪种内置引擎方向、什么目标、是否升级。"""

    engine: Literal["auto", "weights", "model_type"] = "auto"
    model_type: str = Field(default="", max_length=40)
    objective: str = ""
    escalate: bool = False
    reason: str = Field(default="", max_length=500)

    @field_validator("model_type")
    @classmethod
    def _check_model_type(cls, value: str) -> str:
        if value and value not in MODEL_TYPES:
            raise ValueError(
                f"未知内置模型 {value!r}；可留空或用 "
                + " / ".join(MODEL_TYPES)
            )
        return value

    @field_validator("objective")
    @classmethod
    def _check_objective(cls, value: str) -> str:
        return _validate_objective(value)

    def clean(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model_type": self.model_type,
            "objective": self.objective,
            "escalate": bool(self.escalate),
            "reason": self.reason.strip(),
        }


class ProgramFile(_StrictModel):
    """AI 程序里的一个文件（可多级包路径，如 pkg/helpers.py）。"""

    path: str = Field(min_length=3, max_length=256)
    content: str = Field(min_length=1, max_length=500000)


class WriteModelProgramArgs(_StrictModel):
    """AI 直接编写/编辑完整模型程序（多文件、多级包）。

    入口文件必须定义 ``build_transform(params=None) -> transform(z, weights, context=None)``；
    可选定义 ``select_stocks(scores, top_n, params=None)`` 接管选股（可返回列表或
    {代码: 权重} 字典）。``param_space`` 可省略（省略则只做单次回测）；
    ``config`` 可覆盖 top_n/回测年限/调仓周期/交易成本/组合方法；
    ``packages`` 是向审核申请的第三方库。
    """

    files: list[ProgramFile] = Field(min_length=1)
    entry: str = Field(default="main.py", max_length=256)
    summary: str = Field(default="", max_length=1000)
    param_space: dict[str, dict[str, Any]] = Field(default_factory=dict)
    objective: str = ""
    max_trials: int | None = Field(default=None, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)
    packages: list[str] = Field(default_factory=list)
    escalate: bool = False
    reason: str = Field(default="", max_length=500)
    expected_effect: str = Field(default="", max_length=1000)
    risks: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("objective")
    @classmethod
    def _check_objective(cls, value: str) -> str:
        return _validate_objective(value)

    @field_validator("config")
    @classmethod
    def _check_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(str(key) for key in value if str(key) not in PROGRAM_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"config 不支持这些键：{', '.join(unknown)}；"
                f"可选：{', '.join(PROGRAM_CONFIG_KEYS)}")
        return value

    def clean(self) -> dict[str, Any]:
        return {
            "files": {item.path.strip(): item.content for item in self.files},
            "entry": self.entry.strip() or "main.py",
            "summary": self.summary.strip(),
            "param_space": dict(self.param_space),
            "objective": self.objective,
            "max_trials": int(self.max_trials) if self.max_trials is not None else None,
            "config": dict(self.config),
            "packages": [str(item).strip() for item in self.packages if str(item).strip()],
            "escalate": bool(self.escalate),
            "reason": self.reason.strip(),
            "expected_effect": self.expected_effect.strip(),
            "risks": [str(item).strip() for item in self.risks if str(item).strip()],
        }


_TOOL_MODELS: dict[str, type[BaseModel]] = {
    "update_profile": UpdateProfileArgs,
    "confirm_profile": EmptyArgs,
    "run_selection": EmptyArgs,
    "explain_results": ExplainResultsArgs,
    "analyze_model": AnalyzeModelArgs,
    "write_model_program": WriteModelProgramArgs,
    "apply_pending_change": ApplyPendingChangeArgs,
    "rollback_version": EmptyArgs,
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "update_profile": "把客户本轮新提供的画像信息写入 slots（只放本轮新事实，没有就不调用）。",
    "confirm_profile": "客户本轮明确同意其投资画像时调用；画像必填项必须已齐全。",
    "run_selection": "画像已确认后，按当前会话模型执行 A 股筛选与回测；客户说开始/帮我选股时调用。",
    "explain_results": "最近已有选股结果、客户在追问原因/风险/怎么看时调用，回答只基于给定事实。",
    "analyze_model": (
        "调整当前会话投资模型的内置方向（不写代码）：点名内置模型 → engine=model_type 并填 model_type；"
        "普通调参优化 → engine=auto；继续优化/上一版未达预期 → escalate=true。"
        "客户要求构建/更换/编辑新的数学模型、打分模型、引擎、程序或公式时，"
        "必须调用 write_model_program 直接编写完整程序，不要用本工具。"
        "objective 可用 sharpe/cagr/mdd/calmar/sortino/ir/win_rate，留空默认夏普。"
    ),
    "write_model_program": (
        "AI 直接编写/编辑完整模型程序（多文件、多级包 Python）。客户要求构建、更换、重写、"
        "定制数学/打分模型或修改现有模型程序时调用。入口文件定义 "
        "build_transform(params=None) -> transform(z, weights, context=None)（context 含 "
        "asof/历史行情/财务/行业，可用于因子计算或训练）；可选定义 "
        "select_stocks(scores, top_n, params=None)，返回代码列表或 {代码: 权重} 字典接管选股/仓位。"
        "程序可写多个模块并互相 import（支持 pkg/helpers.py 多级包），可用类/装饰器/生成器；"
        "param_space 可省略（省略则单次回测），数量不限，max_trials 可指定寻优预算；"
        "config 可覆盖 top_n/backtest_years/rebalance_days/cost_buy/cost_sell/portfolio_method；"
        "需要额外第三方库时在 packages 里申请，由 LLM 安全审核决定是否安装。"
        "生成过程会实时流式展示，生成后经 LLM 安全审核与服务端校验，不会自动启用。"
    ),
    "apply_pending_change": (
        "存在待审批模型方案且客户本轮明确同意启用时调用。"
        "客户指定第 N 个候选方案（如“用第一个/第二个方案”）时传 candidate=N"
        "（按方案出现顺序从 1 开始）；未指定则留空，默认启用最新的待审批方案。"
    ),
    "rollback_version": "客户明确要求回退到当前会话上一版模型时调用。",
}


def _schema_for(name: str) -> dict[str, Any]:
    schema = _TOOL_MODELS[name].model_json_schema()
    schema.pop("title", None)
    return schema


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": _TOOL_DESCRIPTIONS[name],
            "parameters": _schema_for(name),
        },
    }
    for name in ACTION_NAMES
]


def validate_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    """校验一个 tool call 的参数并返回干净 dict（供执行层使用）。

    失败抛 :class:`ToolArgsError`；调用方应把错误回传 LLM 让它重新调用工具，
    不得在服务端用关键词猜测意图。
    """
    tool = str(name or "").strip()
    if tool not in _TOOL_MODELS:
        raise ToolArgsError(tool or "<unknown>", f"未知工具；可选 {', '.join(ACTION_NAMES)}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolArgsError(tool, "参数必须是 JSON 对象")
    try:
        model = _TOOL_MODELS[tool].model_validate(arguments)
    except ValidationError as exc:
        detail = "；".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'args'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ToolArgsError(tool, detail) from exc
    if isinstance(model, AnalyzeModelArgs):
        return model.clean()
    if isinstance(model, WriteModelProgramArgs):
        return model.clean()
    if isinstance(model, ExplainResultsArgs):
        return model.clean()
    if isinstance(model, ApplyPendingChangeArgs):
        return model.clean()
    if isinstance(model, UpdateProfileArgs):
        return {"slots": dict(model.slots)}
    return {}
