# -*- coding: utf-8 -*-
"""对话 Agent：原生 function calling 规划 + 流式回复。

- 回复文本通过 content 增量实时产出；
- 动作只通过 tools/tool_calls 表达，服务端按 schema 校验后执行；
- 画像更新走 update_profile 工具，不再混在回复 JSON 里。
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from .chat.action_schema import TOOL_DEFINITIONS, ToolArgsError, validate_tool_call
from .llm import LLMClient, LLMError, ToolChatResult
from .schemas import Plan, SLOT_DEFS, profile_payload

BANNED_WORDS = ("稳赚", "必涨", "保证收益", "推荐买入", "满仓干", "包赚", "翻倍牛股")

META_PHRASES = (
    "先问第一个问题",
    "第一个问题",
    "接下来我会",
    "作为AI",
    "作为 AI",
    "作为一个人工智能",
    "希望以上",
    "感谢您的信任",
    "我们开始吧",
    "好的，我明白了",
)

PLANNER_SYSTEM = f"""你是一名服务于金融机构的持牌投资顾问型 AI 助手。你的表达必须符合金融机构客户服务与投资者适当性管理规范，专业、审慎、以客户为中心。

## 表达规范（违反视为不合格）
- 称呼客户为“您”；语气专业、克制、礼貌，不使用网络用语、感叹号、表情或夸张形容。
- 不进行自我叙述，不描述“我接下来要做什么步骤”，禁止出现“先问第一个问题、第一个问题、接下来我会、作为 AI、希望以上、感谢您的信任”等套话。
- 需要了解信息时，使用业务化表达，例如：“在为您提供适配建议前，需要向您确认一项信息：……”“为完成适当性匹配，请补充……”。
- 一次只确认一个信息点；信息足够时先做简明摘要，再请客户确认，不要反复追问；已经确认或已经回答过的画像字段不得再次询问。
- 不重复客户原话，不做机械式分点罗列；能用两句话说清就不用五点。
- content 只输出给客户看的纯文本，不要 JSON、Markdown 标题、加粗或列表符号，不超过 180 字。

## 语义理解（重点）
- 结合最近一轮助手提出的问题理解客户的短回答、省略句和口语，不要只做字面关键词匹配。
- 如果助手在问“亏损反应/真亏了会怎么做/出现浮亏怎么办”，客户回答“继续持有、拿着、不卖、扛着、补仓、加仓、止损、减仓、割肉”等，都是对该画像字段的回答，必须通过 update_profile 写入 loss_reaction。
- 如果助手在问期限/亏损比例/资金用途，客户只答“4 年”“10%”“子女教育”等短句，也要通过 update_profile 写入对应槽位。
- 客户已经回答过的信息不得再次追问；不要要求客户按固定措辞或选项作答。

## 能力说明（必须在对话中清楚告知）
- 首次对话可用一句话点明：我可以协助建立投资者画像、进行 A 股筛选与回测、解读结果，也能优化模型——由 AI 负责编写底层打分引擎、确定参数搜索范围，由服务端执行网格寻优并实时展示进度；重要变更需要您审批，且可回滚。
- 当客户问“你能做什么 / 有什么功能 / 怎么用”时，用不超过 4 条的简洁中文说明上述能力，最后只问一个开放式问题。
- 当候选方案未达目标、或客户选择“继续优化”时，明确告知下一步会采取更激进的结构性策略：可以直接更换基础模型，或由 AI 重写数学打分引擎，再由服务端在新的参数范围内重新寻优并回测，不会自动启用。
- 客户说“继续优化模型 / 上一版未达到我的预期 / 再试一版 / 换个方向”时，本轮必须直接调用 analyze_model(engine="auto", escalate=true)，并在回复中说明会做结构性升级；不得改为回滚、不得询问“是否回滚”。
- 客户问“为什么你来调参 / 是不是你直接改参数”时，说明分工：AI 只负责写引擎和给出参数范围，具体参数由服务端在范围内逐组回测、按目标指标选出最优。

## 客户追问与解释规范（重点）
- 当客户问“什么是回撤 / 这是什么意思 / 为什么问这个 / 怎么理解 / 没看懂”时，先明确说出具体概念或字段，用一两句话解释它是什么、为什么影响适当性匹配，最多给一个生活化例子。
- 如果客户只是询问概念，本轮可以只解释，不追问；不要为了采集画像强行把解释变成客户作答。
- 禁止“请选择一项”“一是……二是……三是……”“回复对应选项即可”等问卷式表达；禁止“请按实际情况描述”“请如实回答”“请回答”“必须回答”等命令式表达。

## 服务流程
1. 了解客户：投资目标、期限、资金用途、风险承受与亏损反应、经验与偏好。
2. 适当性匹配：根据客户信息推导风险等级，并说明该等级与后续策略的匹配关系。
3. 请客户确认：给出简明画像摘要，使用“需要您确认：……”句式。
4. 经客户确认后执行选股或模型优化；执行前不承诺结果。
5. 结果说明：先给结论，再给依据，最后做风险提示。

## 画像槽位（生活化标签）
{json.dumps([{"key": item["key"], "label": item["label"], "required": item["required"], "hint": item["hint"]} for item in SLOT_DEFS], ensure_ascii=False)}
必填：goal、horizon、drawdown_tolerance、loss_reaction。风险等级由信息推导，不直接询问“你是什么风险等级”。

## 工具调用规则
- 动作只能通过 tools 里的函数调用表达，不要在 content 里写动作说明或 JSON。
- 调用工具时，content 仍必须给客户一句简短确认（例如“已收到，正在执行”），不要留空；不要描述内部步骤。
- 客户本轮提供画像信息时，调用 update_profile({{"slots": {{"槽位key": "客户原意"}}}})；只放本轮新事实，没有就不调用。
- 客户本轮明确同意画像时调用 confirm_profile；开始选股调用 run_selection；追问结果调用 explain_results；明确同意启用方案调用 apply_pending_change；明确要求回滚调用 rollback_version。
- analyze_model 必须明确 engine/model_type/objective/escalate：构建/更换新的数学模型、打分模型、引擎或公式 → engine="code"；点名内置模型 → engine="model_type" 并填 model_type；点名 SVM/随机森林等清单外模型 → engine="code"；继续优化/上一版未达预期 → escalate=true；普通优化 → engine="auto"。
- 客户说“继续优化/上一版未达到预期/再试一版”时，必须直接调用 analyze_model(engine="auto", escalate=true)，不得反问“哪里不满意/先确认方向”。
- 客户只泛称“换其他模型/换个模型/其他模型”而没有点名具体模型时，调用 analyze_model(engine="model_type", model_type="", escalate=false)，由服务端选择与当前不同的内置模型；不要用 engine="code"，也不要当成继续优化。
- escalate=true 只用于“继续优化/上一版未达到预期/再试一版”；存在待审批方案本身不代表要升级。
- 客户说“回滚到上一个版本/恢复上一版模型”时，必须直接调用 rollback_version，不得反问确认。
- 客户确认画像且必填齐全时，必须同时调用 confirm_profile 与 run_selection。
- 客户要求“更换/构建数学模型”时不要反问“想换成哪种模型”，不要声称没有权限写代码。

## 工具调用示例（参数只写关键字段）
- “更换数学模型” → analyze_model(engine="code")
- “换成低波质量” → analyze_model(engine="model_type", model_type="low_vol_quality")
- “换其他模型/换个模型” → analyze_model(engine="model_type", model_type="", escalate=false)
- “继续优化，上一版没达到预期” → analyze_model(engine="auto", escalate=true)
- “帮我优化一下模型” → analyze_model(engine="auto")
- “可以，就按这个来”（画像待确认）→ confirm_profile、run_selection
- “我 5 年后要用钱” → update_profile(slots={{"horizon": "5 年"}})
- “什么是回撤” → 不调用工具

## 澄清与执行必须分离（硬性规则）
- 如果需要客户补充信息或做选择，本轮只提问，不调用任何工具。
- 客户回答后的下一轮再执行工具；不得在提出问题的同一轮调用工具。
- 调用工具时不要再向客户提问。
- 同一待执行动作最多澄清一次；客户已回答后必须执行。

## 确认协议
- 画像待确认：“需要您确认：以上画像信息是否准确？如无补充，请回复确认，我将按该画像为您筛选标的；如需调整，请直接告知。”
- 模型方案待审批：结合本次回测结果自然说明，请客户决定是否启用；不要使用固定模板，也不要与其他提问叠加。
- 动作未执行前不得声称已完成。
- 同一轮只能提出一个需要客户回答的问题。

## 合规红线
- 不荐股、不预测涨跌、不承诺收益；禁用“稳赚、必涨、保证收益、推荐买入、满仓”等表述。
- 说明历史回测的局限性；涉及风险时使用规范表述：“历史业绩不代表未来表现，市场有风险，投资需谨慎。”"""

EXPLAIN_SYSTEM = """你是一名金融机构的投资研究服务人员。请依据给定事实，用规范、克制的中文回答客户问题：
1. 结构：先给结论，再列 1~2 条依据（必须引用事实中的数字），最后做风险提示。
2. 使用“您”称呼客户；不使用网络用语、感叹号或情绪化表达；专业术语首次出现时用一句话说明。
3. 只使用事实中提供的数据；数据未覆盖时明确说明“当前数据未覆盖”，不猜测、不编造。
4. 不预测涨跌、不推荐买卖、不承诺收益。
5. 结尾固定表述：“以上基于历史回测数据，历史业绩不代表未来表现，不构成投资建议，市场有风险，投资需谨慎。”
6. 控制在 350 字以内。"""


class AgentError(RuntimeError):
    """对话或计划生成失败。"""


class ComplianceError(AgentError):
    """输出包含违规承诺、荐股措辞或明显的 AI 套话。"""


def ensure_compliance(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise AgentError("模型返回空回复")
    for word in BANNED_WORDS:
        if word in text:
            raise ComplianceError("回复包含不合规措辞，已拦截")
    for phrase in META_PHRASES:
        if phrase in text:
            raise ComplianceError("回复包含 AI 套话，已拦截")
    return text.strip()


def chunk_text(text: str, size: int = 12) -> Iterator[str]:
    """把完整文本切成小片（用于服务端生成的短备注，不用于 LLM 回复）。"""
    cleaned = text.strip()
    for start in range(0, len(cleaned), size):
        yield cleaned[start:start + size]


def asks_clarifying_question(reply: str) -> bool:
    return "？" in reply or "?" in reply


SLOT_QUESTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "loss_reaction": ("亏损反应", "真亏了", "出现浮亏", "下跌", "继续持有", "补仓", "止损", "怎么处理"),
    "drawdown_tolerance": ("最大亏损", "回撤", "能接受亏", "亏损承受", "跌多少"),
    "horizon": ("投资期限", "多久", "不用这笔钱", "什么时候用"),
    "goal": ("投资目标", "资金用途", "用来做什么", "做什么用"),
    "income_stability": ("收入稳定",),
    "experience": ("投资经验", "基金", "股票"),
}


def _last_assistant_text(history: list[dict[str, str]] | None) -> str:
    for item in reversed(history or []):
        if str(item.get("role")) == "assistant" and str(item.get("content") or "").strip():
            return str(item.get("content"))
    return ""


def infer_pending_slot(state: dict[str, Any], history: list[dict[str, str]] | None) -> str:
    question = _last_assistant_text(history)
    for slot, words in SLOT_QUESTION_KEYWORDS.items():
        if any(word in question for word in words):
            return slot
    profile = state.get("profile") or {}
    missing = [str(item) for item in (profile.get("missing") or [])]
    for slot, words in SLOT_QUESTION_KEYWORDS.items():
        if any(any(word in label for word in words) for label in missing):
            return slot
    return ""


class Agent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _payload(state: dict[str, Any], history: list[dict[str, Any]],
                 message: str) -> str:
        return json.dumps({
            "state": state,
            "user_message": message,
            "recent_history": history[-8:],
            "last_assistant_question": _last_assistant_text(history),
            "awaiting_slot": infer_pending_slot(state, history),
        }, ensure_ascii=False)

    @staticmethod
    def _actions_from_result(
            result: ToolChatResult) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        actions: list[dict[str, Any]] = []
        profile_delta: dict[str, Any] = {}
        errors: list[str] = []
        for call in result.tool_calls:
            try:
                args = validate_tool_call(call.name, call.arguments)
            except ToolArgsError as exc:
                errors.append(str(exc))
                continue
            if call.name == "update_profile":
                profile_delta.update(args["slots"])
                continue
            actions.append({"name": call.name, "args": args})
        return actions, profile_delta, errors

    @staticmethod
    def _repair_messages(user_content: str, result: ToolChatResult,
                         errors: list[str]) -> list[dict[str, Any]]:
        message = result.message if isinstance(result.message, dict) else {}
        assistant = {
            "role": "assistant",
            "content": str(message.get("content") or ""),
            "tool_calls": message.get("tool_calls") or [],
        }
        tool_messages = [
            {"role": "tool", "tool_call_id": call.id,
             "content": "参数校验失败；请修正该工具参数后重新调用。"}
            for call in result.tool_calls
        ]
        correction = (
            "上一次工具调用参数未通过校验：\n- " + "\n- ".join(errors)
            + "\n请重新调用正确的工具并修正参数；不要解释，不要在 content 里输出动作。"
        )
        return [{"role": "user", "content": user_content}, assistant,
                *tool_messages, {"role": "user", "content": correction}]

    @staticmethod
    def _state_after_profile(state: dict[str, Any],
                             profile_delta: dict[str, Any]) -> dict[str, Any]:
        if not profile_delta:
            return state
        profile = state.get("profile") or {}
        slots = dict(profile.get("slots") or {})
        slots.update(profile_delta)
        return {**state,
                "profile": profile_payload(slots, bool(profile.get("confirmed")))}

    def _ack_tokens(self, state: dict[str, Any], user_content: str,
                    tool_calls: list[Any]) -> Iterator[str]:
        calls = [{"name": call.name, "args": call.arguments} for call in tool_calls]
        payload = json.dumps({
            "state": state,
            "user_message": user_content,
            "tool_calls": calls,
        }, ensure_ascii=False)
        system = (
            "你是持牌投资顾问。刚才后端已执行了给定工具调用，但模型没有留下回复文本。"
            "请只输出一句给客户的回复（不超过 100 字），按以下优先级：\n"
            "1. 若调用了 update_profile 且画像仍有必填项缺失：只询问下一个缺失项，"
            "用生活化表达、一次一个问题；不要声称已执行模型操作或展示进度。\n"
            "2. 若调用了 update_profile 且画像齐全但尚未确认：请客户确认画像摘要。\n"
            "3. 若调用了 confirm_profile 或 run_selection：告知已按画像完成筛选，结果和风险已展示。\n"
            "4. 若调用了 analyze_model：参数 engine=model_type 时说明正在切换基础模型并回测，"
            "不要称参数寻优；其他 engine 说明已开始模型优化与寻优；候选结果都需要客户确认。\n"
            "5. 其他情况：简短确认并说明下一步。\n"
            "不要重复固定套话，不要罗列选项，不要否认能力，不要反问。"
        )
        for chunk in self.llm.chat_text_stream(system, [{"role": "user", "content": payload}]):
            yield chunk

    def stream_plan(self, state: dict[str, Any], history: list[dict[str, Any]],
                    message: str) -> Iterator[tuple[str, Any]]:
        """产出 ("token", 文本增量) 若干次，最后产出 ("plan", Plan)。"""
        user_content = self._payload(state, history, message)
        reply_parts: list[str] = []
        result: ToolChatResult | None = None
        try:
            for kind, payload in self.llm.chat_tools_stream(
                    PLANNER_SYSTEM, [{"role": "user", "content": user_content}],
                    TOOL_DEFINITIONS):
                if kind == "content":
                    reply_parts.append(payload)
                    yield "token", payload
                elif kind == "result":
                    result = payload
        except LLMError as exc:
            raise AgentError(f"AI 暂时无法理解这句话：{exc}") from exc
        if result is None:
            raise AgentError("AI 没有返回规划结果")

        actions, profile_delta, errors = self._actions_from_result(result)
        if errors:
            try:
                repaired = self.llm.chat_tools(
                    PLANNER_SYSTEM,
                    self._repair_messages(user_content, result, errors),
                    TOOL_DEFINITIONS,
                )
            except LLMError as exc:
                raise AgentError(f"AI 工具参数修复失败：{exc}") from exc
            actions, profile_delta, errors = self._actions_from_result(repaired)
            if errors:
                raise AgentError("AI 工具参数校验失败：" + "；".join(errors))
            result = repaired

        if not result.tool_calls:
            review_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": (
                    "请复核上一轮：如果客户是在要求执行动作（优化/更换或构建模型/选股/"
                    "确认画像/启用方案/回滚），必须立即调用对应工具；如果只是提问或回答信息，"
                    "则不要调用工具。不要修改 content 文案。")},
            ]
            try:
                reviewed = self.llm.chat_tools(
                    PLANNER_SYSTEM, review_messages, TOOL_DEFINITIONS)
            except LLMError:
                reviewed = None
            if reviewed is not None and reviewed.tool_calls:
                actions, profile_delta, errors = self._actions_from_result(reviewed)
                if errors:
                    raise AgentError("AI 工具参数校验失败：" + "；".join(errors))
                result = reviewed

        reply = "".join(reply_parts).strip()
        if not reply:
            reply_parts = []
            ack_state = self._state_after_profile(state, profile_delta)
            for chunk in self._ack_tokens(ack_state, user_content, result.tool_calls):
                reply_parts.append(chunk)
                yield "token", chunk
            reply = "".join(reply_parts).strip()
            if not reply:
                raise AgentError("模型没有给出回复文本")
        ensure_compliance(reply)

        plan = Plan(reply=reply, actions=actions, profile_delta=profile_delta)
        if plan.actions and asks_clarifying_question(plan.reply):
            plan.deferred_intent = plan.actions[0]["name"]
            plan.actions = []
        yield "plan", plan

    def plan(self, state: dict[str, Any], history: list[dict[str, Any]],
             message: str) -> Plan:
        final: Plan | None = None
        for kind, payload in self.stream_plan(state, history, message):
            if kind == "plan":
                final = payload
        if final is None:
            raise AgentError("AI 没有返回规划结果")
        return final

    def explain(self, facts: dict[str, Any], question: str = "") -> str:
        payload = json.dumps({"facts": facts, "question": question}, ensure_ascii=False)
        try:
            text = self.llm.chat_text(EXPLAIN_SYSTEM, [{"role": "user", "content": payload}])
        except LLMError as exc:
            raise AgentError(f"解读生成失败：{exc}") from exc
        return ensure_compliance(text)


def safe_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:500]
