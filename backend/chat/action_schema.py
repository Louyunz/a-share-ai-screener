# -*- coding: utf-8 -*-
"""对话动作（工具）契约：Pydantic 模型 + OpenAI function calling schema。

这是 Planner 与执行层之间的**唯一契约**：

- 运行时用同一份 Pydantic 模型校验 LLM 的 tool_calls 参数；
- 发给模型的 ``tools`` 参数由同一份模型生成的 JSON Schema 得到；
- 每个工具的「参数模型 + 给模型看的描述 + 用法说明」都放在同一个
  :class:`ToolSpec` 里（定义在对应参数模型旁边），``render_tool_docs()`` 与
  ``TOOL_DEFINITIONS`` 都从这份单一事实源生成，避免提示词/代码/文档三处漂移；
- 所有动作参数模型都带可选 ``thought`` 字段：ReAct 每一步的可见思考文本，
  只用于流式展示与审计，**不会**进入 ``validate_tool_call`` 的执行参数。

服务端不再解析自然语言来判断“要换模型还是调参”；这个判断只由 LLM 在
tool_calls 的 args 中给出，服务端只做 schema 校验与确定性策略执行。
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..models.registry import MODEL_TYPES
from ..schemas import SLOT_KEYS

__all__ = [
    "ACTION_NAMES",
    "AnalyzeModelArgs",
    "WriteModelProgramArgs",
    "ToolArgsError",
    "ToolSpec",
    "TOOL_SPECS",
    "TOOL_DEFINITIONS",
    "tool_definitions",
    "render_tool_docs",
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

# thought：ReAct 每一步的可见思考文本；只做展示/审计，不参与业务执行。
THOUGHT_MAX_LENGTH = 2000
THOUGHT_FIELD_DESCRIPTION = (
    "当前这一步的思考（会被实时展示给客户）：用一句简短中文说明你的判断与打算；"
    "建议每次工具调用都填写，但它不参与业务执行，"
    "除 thought 之外的业务参数仍必须严格按 schema 完整、正确填写。"
)
TOOL_THOUGHT_NOTE = (
    "可选参数 thought：用一句简短中文说明当前这一步的思考（会实时展示给客户）；"
    "thought 之外的业务参数仍必须完整、正确填写。"
)


def _doc(text: str) -> str:
    """整理工具用法说明：去掉统一缩进与首尾空行，保证渲染结果稳定。"""
    return textwrap.dedent(text).strip()


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


class _ActionArgs(_StrictModel):
    """所有动作参数的公共基类（``thought`` + ``clean`` 契约）。

    ``thought`` 是 ReAct 每步的可见思考文本，只用于流式展示/审计：
    它会被 Pydantic 校验长度，但 :meth:`clean` 返回的业务参数里永远不含它。
    """

    thought: str = Field(
        default="",
        max_length=THOUGHT_MAX_LENGTH,
        description=THOUGHT_FIELD_DESCRIPTION,
    )

    def clean(self) -> dict[str, Any]:
        """返回供执行层使用的业务参数；``thought`` 永不进入返回值。"""
        return {}


class EmptyArgs(_ActionArgs):
    """无业务参数动作（仅可选 thought）。"""


class ExplainResultsArgs(_ActionArgs):
    question: str = Field(default="", max_length=500)

    def clean(self) -> dict[str, Any]:
        return {"question": self.question.strip()[:500]}


class ApplyPendingChangeArgs(_ActionArgs):
    """启用待审批的候选方案。

    ``candidate`` 是候选方案编号（按方案出现顺序从 1 开始）；0 表示不指定，
    默认启用最新的待审批方案。客户在一轮里生成了多个候选方案时，需要靠这个
    参数区分“用第一个/第二个”，服务端不解析自然语言。
    """

    candidate: int = Field(default=0, ge=0)

    def clean(self) -> dict[str, Any]:
        return {"candidate": int(self.candidate)}


class UpdateProfileArgs(_ActionArgs):
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

    def clean(self) -> dict[str, Any]:
        return {"slots": dict(self.slots)}


class AnalyzeModelArgs(_ActionArgs):
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


class WriteModelProgramArgs(_ActionArgs):
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


@dataclass(frozen=True)
class ToolSpec:
    """一个工具（动作）的单一事实源：参数模型 + 给模型看的描述 + 用法说明。

    - ``model``：运行时校验与 ``parameters`` JSON Schema 的唯一来源；
    - ``description``：发给模型的 function description（用途 + 关键约束速览）；
    - ``usage``：``render_tool_docs()`` 使用的 Markdown 用法（用途/何时调用/关键参数/行为注意）。
    """

    name: str
    model: type[BaseModel]
    description: str
    usage: str


_OBJECTIVES_DOC = " / ".join(OBJECTIVE_KEYS)
_CONFIG_KEYS_DOC = "/".join(PROGRAM_CONFIG_KEYS)


# 每个工具的元数据就写在它的参数模型旁边：name / model / description / usage。
# 这里按「自然分组」书写，_ordered_tool_specs() 会强制按 ACTION_NAMES 固定排序。
_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="update_profile",
        model=UpdateProfileArgs,
        description="把客户本轮新提供的画像信息写入 slots（只放本轮新事实，没有就不调用）。",
        usage=_doc("""
            - 用途：把客户本轮新提供的画像事实写入 `slots`，合并进当前画像。
            - 何时调用：客户本轮说出或补充了画像信息（目标、期限、资金用途、可接受回撤、亏损反应、经验、偏好、称呼等）时调用；没有新事实不要调用，也不要传空 slots。
            - 关键参数：
              - `slots`（object，必需）：键必须是画像槽位 key（见上方槽位列表），值是客户原意，不要改写或推测；一次可写多个槽位。
            - 行为注意：
              - 只放本轮新事实，画像里已有的或客户没有表达的信息不要写。
              - 调用后 content 仍要自然回应客户，不要在 content 里复述槽位或参数。
        """),
    ),
    ToolSpec(
        name="confirm_profile",
        model=EmptyArgs,
        description="客户本轮明确同意其投资画像时调用；画像必填项必须已齐全。",
        usage=_doc("""
            - 用途：记录客户对当前画像的确认；确认后才能执行选股或模型优化。
            - 何时调用：客户本轮明确同意画像，且必填项 goal、horizon、drawdown_tolerance、loss_reaction 已齐全。
            - 关键参数：无业务参数（仅可选 thought）。
            - 行为注意：
              - 客户只是含糊应答、还在补充信息、或必填项不齐时不要调用。
              - 本轮画像刚好补齐时，可以同一轮同时调用 confirm_profile 与 run_selection。
        """),
    ),
    ToolSpec(
        name="run_selection",
        model=EmptyArgs,
        description="画像已确认后，按当前会话模型执行 A 股筛选与回测；客户说开始/帮我选股时调用。",
        usage=_doc("""
            - 用途：按当前会话模型执行 A 股筛选与回测。
            - 何时调用：画像已确认（或本轮同时确认），客户表示开始/帮我选股/按这个来执行。
            - 关键参数：无业务参数（仅可选 thought）。
            - 行为注意：
              - 选股与回测由服务端执行；调用时 content 给客户一句正在执行的简短确认。
              - 画像未确认时不要调用，先补齐画像并请客户确认。
        """),
    ),
    ToolSpec(
        name="explain_results",
        model=ExplainResultsArgs,
        description="最近已有选股结果、客户在追问原因/风险/怎么看时调用，回答只基于给定事实。",
        usage=_doc("""
            - 用途：基于最近一次选股/回测结果回答客户追问。
            - 何时调用：当前会话已有最近一次选股结果，客户在问原因、风险、指标含义或怎么看。
            - 关键参数：
              - `question`（string，≤500 字）：客户本轮问题的简要摘录，用于让解释聚焦。
            - 行为注意：
              - 只能基于服务端给出的结果事实回答，不得编造收益或荐股；没有结果时不要调用，先说明还没有可解读的结果。
        """),
    ),
    ToolSpec(
        name="analyze_model",
        model=AnalyzeModelArgs,
        description=(
            "调整当前会话投资模型的内置方向（不写代码）：点名内置模型 → engine=model_type 并填 model_type；"
            "普通调参优化 → engine=auto；继续优化/上一版未达预期 → escalate=true。"
            "客户要求构建/更换/编辑新的数学模型、打分模型、引擎、程序或公式时，"
            "必须调用 write_model_program 直接编写完整程序，不要用本工具。"
            "objective 可用 sharpe/cagr/mdd/calmar/sortino/ir/win_rate，留空默认夏普。"
        ),
        usage=_doc(f"""
            - 用途：调整当前会话投资模型的内置方向或权重（不写代码）。
            - 何时调用：客户只想换内置模型或做参数优化时调用；点名内置模型 → engine=model_type 并填 model_type；普通调参优化 → engine=auto；继续优化且上一版未达预期 → escalate=true。
            - 关键参数：
              - `engine`（string）：auto（普通调参优化）/ weights（只调因子权重）/ model_type（点名内置模型）。
              - `model_type`（string）：engine=model_type 时的内置模型 id，其余情况留空。
              - `objective`（string）：{_OBJECTIVES_DOC} 之一，留空默认夏普。
              - `escalate`（bool）：继续优化、上一版未达预期时为 true。
              - `reason`（string）：一句话审计原因，不参与路由。
            - 行为注意：
              - 客户要求构建、更换、重写、定制或编辑数学模型/打分模型/引擎/程序/公式时，必须改用 write_model_program，不要用本工具。
              - 优化结果不会自动启用，需要客户审批。
        """),
    ),
    ToolSpec(
        name="write_model_program",
        model=WriteModelProgramArgs,
        description=(
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
        usage=_doc(f"""
            - 用途：AI 直接编写/编辑完整模型程序（多文件、多级包 Python）。
            - 何时调用：客户要求构建、更换、重写、定制、编辑新的数学模型/打分模型/引擎/程序/公式，或点名 SVM、随机森林、神经网络等清单外模型时调用。
            - 关键参数：
              - `files`（array，必需）：每项包含 `path` 与 `content` 两个字符串字段；path 可以是 `main.py` 或 `pkg/helpers.py` 等多级包路径，content 必须是完整可运行源码，不能写占位符或省略。
              - `entry`（string）：入口文件名，默认 `main.py`，必须是 files 里的一个路径。
              - `summary`（string）：给客户看的一句话说明。
              - `param_space`（object）：参数名映射到 `min`/`max`/`step` 范围对象；省略表示只做一次回测，参数数量不限。
              - `objective`（string）：优化目标，{_OBJECTIVES_DOC} 之一。
              - `max_trials`（integer）：可选寻优预算。
              - `config`（object）：可覆盖 {_CONFIG_KEYS_DOC}。
              - `packages`（array）：需要额外第三方库时在这里申请，由 LLM 安全审核决定是否安装。
              - `escalate`（bool）：继续优化、上一版未达预期时为 true。
              - `reason` / `expected_effect` / `risks`：可选审计说明。
            - 行为注意：
              - 入口文件必须定义 `build_transform(params=None)`，返回 `transform(z, weights, context=None) -> pandas.Series`；可选定义 `select_stocks(scores, top_n, params=None)` 接管选股/仓位。
              - 源码只放在 files 里，不要写进 content；程序生成后会经安全审核与服务端校验，不会自动启用。
        """),
    ),
    ToolSpec(
        name="apply_pending_change",
        model=ApplyPendingChangeArgs,
        description=(
            "存在待审批模型方案且客户本轮明确同意启用时调用。"
            "客户指定第 N 个候选方案（如“用第一个/第二个方案”）时传 candidate=N"
            "（按方案出现顺序从 1 开始）；未指定则留空，默认启用最新的待审批方案。"
        ),
        usage=_doc("""
            - 用途：启用待审批的候选模型方案。
            - 何时调用：存在待审批候选方案，且客户本轮明确同意启用或明确指定要用哪一个。
            - 关键参数：
              - `candidate`（integer，默认 0）：客户指定第 N 个方案时传 N（按方案出现顺序从 1 开始）；未指定或说“最新这个/就按这个”时留空（0 表示默认启用最新待审批方案）。
            - 行为注意：
              - status=applied 的候选已启用，不要重复启用；available=false 的旧会话记录不能启用。
              - 在多个未启用候选之间做选择不是回滚，不要调用 rollback_version。
        """),
    ),
    ToolSpec(
        name="rollback_version",
        model=EmptyArgs,
        description="客户明确要求回退到当前会话上一版模型时调用。",
        usage=_doc("""
            - 用途：回滚到当前会话上一版已启用的模型。
            - 何时调用：客户明确要求回退到上一版模型，且 state.versions 非空。
            - 关键参数：无业务参数（仅可选 thought）。
            - 行为注意：
              - state.versions 为空表示从未启用过模型版本，此时不要调用，只在 content 说明当前没有可回滚的版本。
        """),
    ),
)


def _ordered_tool_specs(specs: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
    """按 ``ACTION_NAMES`` 固定顺序返回工具元数据；缺失/多余/重复即契约错误。"""
    by_name: dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in by_name:
            raise RuntimeError(f"工具元数据重复定义：{spec.name}")
        by_name[spec.name] = spec
    missing = [name for name in ACTION_NAMES if name not in by_name]
    unknown = sorted(name for name in by_name if name not in ACTION_NAMES)
    if missing or unknown:
        raise RuntimeError(
            f"工具元数据与 ACTION_NAMES 不一致：缺少 {missing}；多出 {unknown}")
    return tuple(by_name[name] for name in ACTION_NAMES)


TOOL_SPECS: tuple[ToolSpec, ...] = _ordered_tool_specs(_TOOL_SPECS)
_SPEC_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def _definition_for(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description + "\n\n" + TOOL_THOUGHT_NOTE,
            "parameters": _schema_for(spec.model),
        },
    }


def tool_definitions() -> list[dict[str, Any]]:
    """按 ``ACTION_NAMES`` 固定顺序返回 OpenAI function calling 工具定义。

    每次都从 :data:`TOOL_SPECS` 重新构建（返回新列表），保证概念上「只有一份事实源」；
    :data:`TOOL_DEFINITIONS` 是它的模块级快照，保留名字兼容既有调用方。
    """
    return [_definition_for(spec) for spec in TOOL_SPECS]


TOOL_DEFINITIONS: list[dict[str, Any]] = tool_definitions()

_TOOL_DOCS_INTRO = (
    "## 可用工具（按固定顺序，共 {count} 个）\n\n"
    "所有工具都支持可选字段 `thought`：用一句简短中文说明当前这一步的思考"
    "（会实时展示给客户）；`thought` 不参与业务执行，但除它之外的业务参数"
    "仍必须按 schema 完整、正确填写。\n"
)


def render_tool_docs() -> str:
    """按 ``ACTION_NAMES`` 固定顺序渲染工具用法说明（供 planner 提示词动态注入）。

    - 只包含模型真正需要的用法：用途 / 何时调用 / 关键参数 / 行为注意；
    - 不包含 JSON Schema 全文，避免提示词过长；
    - 输出完全确定：同一份元数据每次渲染结果完全一致，便于提示词缓存与测试稳定。
    """
    blocks = [_TOOL_DOCS_INTRO.format(count=len(TOOL_SPECS))]
    for index, spec in enumerate(TOOL_SPECS, start=1):
        blocks.append(f"### {index}. `{spec.name}`\n\n{spec.usage}\n")
    return "\n".join(blocks).rstrip("\n") + "\n"


def validate_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    """校验一个 tool call 的参数并返回干净 dict（供执行层使用）。

    ``thought`` 只用于流式展示/审计，永远不会出现在返回的执行参数里。

    失败抛 :class:`ToolArgsError`；调用方应把错误回传 LLM 让它重新调用工具，
    不得在服务端用关键词猜测意图。
    """
    tool = str(name or "").strip()
    spec = _SPEC_BY_NAME.get(tool)
    if spec is None:
        raise ToolArgsError(tool or "<unknown>",
                            f"未知工具；可选 {', '.join(ACTION_NAMES)}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolArgsError(tool, "参数必须是 JSON 对象")
    try:
        model = spec.model.model_validate(arguments)
    except ValidationError as exc:
        detail = "；".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'args'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ToolArgsError(tool, detail) from exc
    cleaned = dict(model.clean())
    # 双保险：即使某个 clean() 意外带上了 thought，也不允许它进入执行参数。
    cleaned.pop("thought", None)
    return cleaned
