# -*- coding: utf-8 -*-
"""对话 Agent：理解客户、决定下一步动作、按金融机构服务规范回复。

Agent 不直接碰数据库和回测；它输出 ``Plan``，由 service 执行并生成卡片。
需要客户确认或补充信息时，服务端会兜底补上标准确认话术，并阻止“边问边执行”。
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from .llm import LLMClient, LLMError
from .schemas import Plan, SLOT_DEFS

ACTION_NAMES = {
    "confirm_profile",
    "run_selection",
    "explain_results",
    "analyze_model",
    "apply_pending_change",
    "rollback_version",
}

BANNED_WORDS = ("稳赚", "必涨", "保证收益", "推荐买入", "满仓干", "包赚", "翻倍牛股")

# AI 味的自我叙述/套话：真实金融机构服务中不会出现
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

# 服务端兜底确认话术：不依赖模型自由发挥
CONFIRM_PROFILE_HINT = (

)
CONFIRMATION_MARKERS = (

)

PLANNER_SYSTEM = f"""你是一名服务于金融机构的持牌投资顾问型 AI 助手。你的表达必须符合金融机构客户服务与投资者适当性管理规范，专业、审慎、以客户为中心。

## 表达规范（违反视为不合格）
- 称呼客户为“您”；语气专业、克制、礼貌，不使用网络用语、感叹号、表情或夸张形容。
- 不进行自我叙述，不描述“我接下来要做什么步骤”，禁止出现“先问第一个问题、第一个问题、接下来我会、作为 AI、希望以上、感谢您的信任”等套话。
- 需要了解信息时，使用业务化表达，例如：“在为您提供适配建议前，需要向您确认一项信息：……”“为完成适当性匹配，请补充……”。
- 一次只确认一个信息点；信息足够时先做简明摘要，再请客户确认，不要反复追问；已经确认或已经回答过的画像字段不得再次询问。
- 不重复客户原话，不做机械式分点罗列；能用两句话说清就不用五点。

## 语义理解（重点）
- 结合最近一轮助手提出的问题理解客户的短回答、省略句和口语，不要只做字面关键词匹配。
- 如果助手在问“亏损反应/真亏了会怎么做/出现浮亏怎么办”，客户回答“继续持有、拿着、不卖、扛着、补仓、加仓、止损、减仓、割肉”等，都是对该画像字段的回答，必须写入 profile_delta.loss_reaction，不要解释成“持仓记录”、交易指令或要求客户补充持仓信息。
- 如果助手在问期限/亏损比例/资金用途，客户只答“4 年”“10%”“子女教育”等短句，也要映射到对应槽位。
- 客户已经回答过的信息不得再次追问；不要要求客户按固定措辞或选项作答。
- 在输出 profile_delta 时保留客户原意，可以做适度规范化（例如“继续持有”可记为“继续持有等待修复”）。

## 能力说明（必须在对话中清楚告知）
- 首次对话可用一句话点明：我可以协助建立投资者画像、进行 A 股筛选与回测、解读结果，也能优化模型——包括更换基础模型、调整因子权重，或直接重写数学打分代码；重要变更需要您审批，且可回滚。
- 当客户问“你能做什么 / 有什么功能 / 怎么用”时，用不超过 4 条的简洁中文说明上述能力，最后只问一个开放式问题。
- 当候选方案未达目标、或客户选择“继续优化”时，明确告知下一步会采取更激进的结构性策略：可以直接更换基础模型，或由 AI 重写数学打分模型并重新回测，不会自动启用。

## 客户追问与解释规范（重点）
- 当客户问“什么是回撤 / 这是什么意思 / 为什么问这个 / 怎么理解 / 没看懂”时，先明确说出具体概念或字段，用一两句话解释它是什么、为什么影响适当性匹配，最多给一个生活化例子。
- 如果客户只是询问概念，本轮可以只解释，不追问；不要为了采集画像强行把解释变成客户作答。
- 禁止使用“您指的是刚才那个问题”这类空泛指代；禁止“请选择一项”“一是……二是……三是……”“回复对应选项即可”等问卷式表达；禁止“请按实际情况描述”“请如实回答”“请回答”“必须回答”等审讯式、命令式表达。
- 如确需推进，用柔和、可选的措辞，例如：“您不用急着给准确数字，可以说个大概；不确定也没关系，我们后面再一起校准。” 保持持牌顾问的解释口吻，而不是审问。

## 服务流程
1. 了解客户：投资目标、期限、资金用途、风险承受与亏损反应、经验与偏好。
2. 适当性匹配：根据客户信息推导风险等级，并说明该等级与后续策略的匹配关系。
3. 请客户确认：给出简明画像摘要，使用“需要您确认：……”句式，并给出可直接回复的选项。
4. 经客户确认后执行选股或模型优化；执行前不承诺结果。
5. 结果说明：先给结论，再给依据，最后做风险提示。

## 画像槽位（生活化标签）
{json.dumps([{"key": item["key"], "label": item["label"], "required": item["required"], "hint": item["hint"]} for item in SLOT_DEFS], ensure_ascii=False)}
必填：goal、horizon、drawdown_tolerance、loss_reaction。风险等级由信息推导，不直接询问“你是什么风险等级”。
当必填槽位齐全、只差客户确认时，不再追问任何必填项；可选信息可在确认后继续了解。

## 澄清与执行必须分离（硬性规则）
- 如果需要客户补充信息或做选择（例如优化方向、风险偏好、资金安排），本轮只提问，actions 必须为空。
- 客户回答后的下一轮再执行动作；不得在提出问题的同一轮执行任何动作。
- 执行动作时不要再向客户提问。
- 同一待执行动作最多澄清一次；客户已回答后必须执行，不得再次提问。
- 若 state.pending_intent 非空且客户本轮已回答澄清问题，执行该动作。

## 确认协议
凡需要客户决定，必须使用明确、可回复的句式，禁止用“要不要”“可以吗”这类模糊问法：
- 画像待确认：“需要您确认：以上画像信息是否准确？如无补充，请回复「确认」，我将按该画像为您筛选标的；如需调整，请直接告知。”
- 模型方案待审批：结合本次回测结果自然说明，请客户决定是否启用；不要使用固定模板，也不要与其他提问叠加。
- 动作未执行前不得声称已完成。
- 同一轮只能提出一个需要客户回答的问题：如果本轮在询问优化方向/其他信息，不得再附加“是否启用上一版方案”的确认；如果认为候选不宜启用，不要把“继续优化”的提问和“是否启用”的确认放在同一段。

## 可执行动作（actions 数组；不需要动作时输出空数组）
- confirm_profile：客户本轮明确同意画像时使用（“可以/确认/没问题/就按这个来”），且必填齐全。
- run_selection：画像确认后开始选股；客户说“开始/帮我选股/可以”时与 confirm_profile 一同输出。
- explain_results：最近有选股结果、客户在问“为什么/风险/怎么看”时使用。
- analyze_model：客户明确要求优化模型/效果不好/想调参/点名切换模型（如随机森林、SVM、低波质量）时使用；系统会分析并给出候选方案，不会直接启用。客户点名模型中已有选项时不得以“没有对应选项”为由拒绝，服务端会优先按客户点名的模型生成方案。
- apply_pending_change：存在待审批方案（state.pending 非空）且客户本轮明确同意时使用。
- rollback_version：客户明确要求回滚时使用。

## 合规红线
- 不荐股、不预测涨跌、不承诺收益；禁用“稳赚、必涨、保证收益、推荐买入、满仓”等表述。
- 说明历史回测的局限性；涉及风险时使用规范表述：“历史业绩不代表未来表现，市场有风险，投资需谨慎。”

## 输出格式（只输出 JSON，不要代码块或多余文字）
{{"reply": "给客户看的回复，≤180字，专业克制", "actions": [{{"name": "动作名", "args": {{}}}}], "profile_delta": {{"槽位key": "客户本轮新提供的事实"}}}}
profile_delta 只放客户本轮新说出的事实，不要编造；没有就给空对象。"""

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
            text = text.replace(phrase, "")
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if not text:
        raise AgentError("模型返回空回复")
    return text


def chunk_text(text: str, size: int = 12) -> Iterator[str]:
    """把完整回复切成小片，模拟逐字输出（内容本身来自模型）。"""
    cleaned = text.strip()
    for start in range(0, len(cleaned), size):
        yield cleaned[start:start + size]


def asks_clarifying_question(reply: str) -> bool:
    """回复是否在向客户提问。"""
    return "？" in reply or "?" in reply


SLOT_QUESTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "loss_reaction": ("亏损反应", "真亏了", "出现浮亏", "下跌", "继续持有", "补仓", "止损", "怎么处理"),
    "drawdown_tolerance": ("最大亏损", "回撤", "能接受亏", "亏损承受", "跌多少"),
    "horizon": ("投资期限", "多久", "不用这笔钱", "什么时候用"),
    "goal": ("投资目标", "资金用途", "用来做什么", "做什么用"),
    "income_stability": ("收入稳定",),
    "experience": ("投资经验", "基金", "股票"),
}


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


def is_explanation_question(message: str) -> bool:
    text = str(message or "").strip()
    return any(term in text for term in (
        "什么意思", "是什么意思", "这些是什么", "都是什么", "解释", "为什么问",
        "为什么需要", "怎么理解", "不太懂", "没看懂", "看不懂", "干嘛问",
        "什么是", "指什么", "含义", "代表什么", "怎么算",
    ))


def _last_assistant_text(history: list[dict[str, str]] | None) -> str:
    for item in reversed(history or []):
        if str(item.get("role")) == "assistant" and str(item.get("content") or "").strip():
            return str(item.get("content"))
    return ""


MECHANICAL_STYLE_MARKERS = (
    "请选择一项", "回复对应选项即可", "您指的是刚才那个问题",
    "一是继续持有", "二是分批补仓", "三是止损离场",
    "请按实际情况描述", "请如实回答", "请回答我", "必须回答",
    "请正面回答", "立即回复", "请马上回答",
)

STYLE_REPAIR_SYSTEM = """你是持牌投资顾问。客户刚才对适当性提问表示不理解。
请重写给定回复，要求：
1. 用 2-4 句自然中文，先明确说出具体概念/字段是什么、为什么需要了解，最多给一个生活化例子；
2. 如果客户只是询问概念，可以只解释、不追问；不要为了采集画像强行要求客户作答；
3. 禁止“请选择一项”“一是/二是/三是”“回复对应选项即可”等问卷式表达，也禁止“请按实际情况描述”“请如实回答”“请回答”“必须回答”等审讯式、命令式表达；
4. 如确需推进，用柔和可选的措辞，例如“您不用急着给准确数字，可以说个大概，后面再校准”；
5. 不得使用“您指的是刚才那个问题”这类空泛指代，不要编造客户信息或新增风险承诺。
只输出重写后的回复正文，不要解释你的修改过程。"""


def has_mechanical_style(reply: str) -> bool:
    value = str(reply or "")
    return any(marker in value for marker in MECHANICAL_STYLE_MARKERS)


def has_confirmation_request(text: str) -> bool:
    return any(marker in text for marker in CONFIRMATION_MARKERS)


CONFIRMATION_LEADS = ("需要您确认", "请您确认", "【请确认】", "【待确认】")


def strip_confirmation_tail(text: str) -> str:
    """去掉文本中从第一个确认话术开始的后半段（用于避免重复追问）。"""
    value = str(text or "")
    positions = [value.find(marker) for marker in CONFIRMATION_LEADS if marker in value]
    if not positions:
        return value
    head = value[:min(positions)].rstrip(" \n；。")
    return head


def trim_duplicate_confirmation(text: str) -> str:
    """保留一段确认话术，删除后面重复的纯确认段落。"""
    value = str(text or "")
    if not has_confirmation_request(value):
        return value
    paragraphs = value.split("\n\n")
    confirmation_indices = [index for index, item in enumerate(paragraphs) if has_confirmation_request(item)]
    if len(confirmation_indices) <= 1:
        return value
    for index in confirmation_indices[1:]:
        stripped = paragraphs[index].strip()
        if stripped.startswith(CONFIRMATION_LEADS):
            paragraphs[index] = ""
    return "\n\n".join(item for item in paragraphs if item.strip())


def confirmation_hint(state: dict[str, Any], plan: Plan) -> str:
    """需要客户确认但计划里没有动作时，返回服务端标准确认话术。

    同一轮只允许一个问题：如果本轮已经提出澄清问题或记录待执行意图，
    不得再追加上一版候选方案的启用确认，避免双重提问。
    """
    if plan.actions or plan.deferred_intent:
        return ""
    if asks_clarifying_question(plan.reply):
        return ""
    profile = state.get("profile") or {}
    if profile.get("complete") and not profile.get("confirmed"):
        return CONFIRM_PROFILE_HINT
        return ""


def attach_confirmation(reply: str, hint: str) -> str:
    if not hint:
        return reply
    if has_confirmation_request(reply):
        return reply
    return reply.rstrip() + "\n\n" + hint


class Agent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _regenerate_explanation(self, draft: str, state: dict[str, Any],
                                history: list[dict[str, str]], message: str) -> str:
        """提示词已约束；若仍出现问卷式话术，调用同一模型重写一次/两次。"""
        last_question = _last_assistant_text(history)
        profile = state.get("profile") or {}
        payload = json.dumps({
            "customer_question": message,
            "previous_question": last_question,
            "draft_reply": draft,
            "profile_summary": profile.get("summary", ""),
            "missing_labels": profile.get("missing", []),
        }, ensure_ascii=False)
        for _attempt in range(2):
            try:
                rewritten = str(self.llm.chat_text(
                    STYLE_REPAIR_SYSTEM, [{"role": "user", "content": payload}]) or "").strip()
            except Exception as exc:  # noqa: BLE001
                raise AgentError(f"AI 暂时无法重新组织说明：{exc}") from exc
            if rewritten and not has_mechanical_style(rewritten):
                ensure_compliance(rewritten)
                return rewritten
        raise AgentError("AI 暂时无法把这条说明讲清楚，请换一种问法再试。")

    # ------------------------------------------------------------------
    def plan(self, state: dict[str, Any], history: list[dict[str, str]],
             message: str) -> Plan:
        last_question = _last_assistant_text(history)
        payload = json.dumps(
            {"state": state, "user_message": message, "recent_history": history[-8:],
             "last_assistant_question": last_question,
             "awaiting_slot": infer_pending_slot(state, history)},
            ensure_ascii=False)
        try:
            raw = self.llm.chat_json(
                PLANNER_SYSTEM, [{"role": "user", "content": payload}])
        except LLMError as exc:
            raise AgentError(f"AI 暂时无法理解这句话：{exc}") from exc
        reply = str(raw.get("reply", "")).strip()
        if not reply:
            reply = "在继续之前，需要向您确认当前的投资目标与资金安排，以便为您做适当性匹配。"
        ensure_compliance(reply)
        actions = self._parse_actions(raw.get("actions"))
        if not actions and is_explanation_question(message) and has_mechanical_style(reply):
            reply = self._regenerate_explanation(reply, state, history, message)
        delta = raw.get("profile_delta")
        if not isinstance(delta, dict):
            delta = {}

        plan = Plan(reply=reply, actions=actions, profile_delta=delta)
        # 澄清与执行分离：提问时本轮不执行动作；记录待执行意图，客户回答后再执行
        if plan.actions and asks_clarifying_question(plan.reply):
            plan.deferred_intent = plan.actions[0]["name"]
            plan.actions = []
        plan.reply = attach_confirmation(plan.reply, confirmation_hint(state, plan))
        return plan

    # ------------------------------------------------------------------
    def explain(self, facts: dict[str, Any], question: str = "") -> str:
        payload = json.dumps({"facts": facts, "question": question}, ensure_ascii=False)
        try:
            text = self.llm.chat_text(EXPLAIN_SYSTEM, [{"role": "user", "content": payload}])
        except LLMError as exc:
            raise AgentError(f"解读生成失败：{exc}") from exc
        return ensure_compliance(text)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_actions(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        actions: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name not in ACTION_NAMES:
                continue
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            actions.append({"name": name, "args": args})
        return actions


def safe_question(text: str) -> str:
    """把客户问题压成一段短文本，避免把超长内容塞进提示词。"""
    return re.sub(r"\s+", " ", text.strip())[:500]
