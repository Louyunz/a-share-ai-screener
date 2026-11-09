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
    "apply_pending_change",
    "rollback_version",
)

MODEL_TYPE_IDS = tuple(MODEL_TYPES)


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

    candidate: int = Field(default=0, ge=0, le=20)

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
    """模型优化动作：AI 必须明确本轮用哪种引擎、什么目标、是否结构性升级。"""

    engine: Literal["auto", "weights", "model_type", "code"] = "auto"
    model_type: str = Field(default="", max_length=40)
    objective: Literal["", "sharpe", "cagr", "mdd"] = ""
    escalate: bool = False
    reason: str = Field(default="", max_length=120)

    @field_validator("model_type")
    @classmethod
    def _check_model_type(cls, value: str) -> str:
        if value and value not in MODEL_TYPES:
            raise ValueError(
                f"未知内置模型 {value!r}；可留空或用 "
                + " / ".join(MODEL_TYPES)
            )
        return value

    def clean(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model_type": self.model_type,
            "objective": self.objective,
            "escalate": bool(self.escalate),
            "reason": self.reason.strip(),
        }


_TOOL_MODELS: dict[str, type[BaseModel]] = {
    "update_profile": UpdateProfileArgs,
    "confirm_profile": EmptyArgs,
    "run_selection": EmptyArgs,
    "explain_results": ExplainResultsArgs,
    "analyze_model": AnalyzeModelArgs,
    "apply_pending_change": ApplyPendingChangeArgs,
    "rollback_version": EmptyArgs,
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "update_profile": "把客户本轮新提供的画像信息写入 slots（只放本轮新事实，没有就不调用）。",
    "confirm_profile": "客户本轮明确同意其投资画像时调用；画像必填项必须已齐全。",
    "run_selection": "画像已确认后，按当前会话模型执行 A 股筛选与回测；客户说开始/帮我选股时调用。",
    "explain_results": "最近已有选股结果、客户在追问原因/风险/怎么看时调用，回答只基于给定事实。",
    "analyze_model": (
        "优化当前会话的投资模型（必须通过参数明确本轮引擎方向）。"
        "客户要求构建/更换/新建一套新的投资数学模型、打分模型、引擎或公式时，"
        "必须立即调用本工具并设 engine=code，由 AI 重写参数化打分引擎；"
        "不要反问‘想换成哪种模型’、不要声称没有权限、不要只在文字里说会优化。"
        "点名内置模型时 engine=model_type 并填 model_type；"
        "客户说继续优化/上一版未达预期时 escalate=true；普通优化 engine=auto。"
        "objective 取值来自客户本轮诉求，不确定留空。"
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
    if isinstance(model, ExplainResultsArgs):
        return model.clean()
    if isinstance(model, ApplyPendingChangeArgs):
        return model.clean()
    if isinstance(model, UpdateProfileArgs):
        return {"slots": dict(model.slots)}
    return {}
