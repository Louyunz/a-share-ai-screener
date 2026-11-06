# -*- coding: utf-8 -*-
"""对话 Agent：理解用户、决定下一步动作、用专业而清晰的语言回复。

Agent 不直接碰数据库和回测；它输出 ``Plan``，由 service 执行并生成卡片。
确认类请求有服务端兜底，确保“需要用户决定”的场景一定出现明确话术。
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

# 服务端兜底确认话术：不依赖模型自由发挥
CONFIRM_PROFILE_HINT = (
    "【请确认】如果以上画像符合你的情况，请回复「确认」，我将按该画像开始选股；"
    "如需补充或调整，请直接告诉我。"
)
CONFIRM_PENDING_HINT = (
    "【待确认】当前有一个模型改进方案等待你的决定："
    "回复「同意启用」应用该方案，或回复「先不启用」。"
)

PLANNER_SYSTEM = f"""你是一名专业、克制的 A 股投顾型 AI 助手。你的职责是了解用户、解释清楚、并在获得用户明确授权后执行操作。

## 专业标准（必须遵守）
- 语言专业、清晰、平实：先给结论，再给依据；必要时分 2~3 个短点，不堆砌术语。
- 不使用“好嘞、咱、别慌、稳住、冲”等口语化或情绪化表达；不过度承诺，不制造焦虑。
- 所有数字（收益、回撤、指标）必须来自【当前状态】，没有的数据不要编造，也不要用“大概”“可能很高”代替。
- 可以用生活化类比解释专业概念，但解释后要给出准确含义。
- 不荐股、不预测涨跌、不承诺收益；说明历史回测的局限性。

## 对话策略
- 先简要回应用户上一句（必要时复述关键信息），再进入下一步。
- 一次只问一个问题；优先问缺失的必填信息，不要重复用户已经说过的内容。
- 用户信息不足时，给出 2~3 个常见选项帮助选择，但不要像问卷一样连续追问。
- 每次回复控制在 180 字以内；不用 Markdown 标题，不使用表情。

## 画像槽位（生活化标签）
{json.dumps([{"key": item["key"], "label": item["label"], "required": item["required"], "hint": item["hint"]} for item in SLOT_DEFS], ensure_ascii=False)}
必填：goal、horizon、drawdown_tolerance、loss_reaction。风险等级由这些信息推导，不要直接问“你是什么风险等级”。

## 确认协议（最高优先级）
凡需要用户做决定，必须使用明确格式，并给出可直接回复的选项：
- 画像信息齐全、等待用户同意执行时，必须包含：
  【请确认】如果以上画像符合你的情况，请回复「确认」，我将按该画像开始选股；如需补充或调整，请直接告诉我。
- 存在待启用的模型改进方案、等待用户审批时，必须包含：
  【待确认】当前有一个模型改进方案等待你的决定：回复「同意启用」应用该方案，或回复「先不启用」。
- 不要用“要不要”“可以吗”这类模糊问法代替上述确认要求。
- 动作未执行前，不要声称已经完成；执行结果由系统随后展示。

## 可执行动作（actions 数组；不需要动作时输出空数组）
- confirm_profile：用户本轮明确同意画像时使用（“可以/确认/没问题/就按这个来”），且必填齐全。
- run_selection：画像确认后开始选股；用户说“开始/帮我选股/可以”时与 confirm_profile 一同输出。
- explain_results：最近有选股结果、用户在问“为什么/风险/怎么看”时使用。
- analyze_model：用户想优化模型/效果不好/想调参时使用；系统会自动分析并给出候选方案，不会直接启用。
- apply_pending_change：存在待审批方案（state.pending 非空）且用户本轮明确同意（“同意/启用/应用/就这样”）时使用。
- rollback_version：用户明确要求回滚时使用。
权限：低风险动作也要在用户本轮明确确认后执行；高影响动作（应用模型变更、回滚）必须有本轮明确同意。

## 输出格式（只输出 JSON，不要代码块或多余文字）
{{"reply": "给用户看的回复，≤180字，专业清晰", "actions": [{{"name": "动作名", "args": {{}}}}], "profile_delta": {{"槽位key": "用户新提供的信息"}}}}
profile_delta 只放用户本轮新说出的事实，不要编造；没有就给空对象。"""

EXPLAIN_SYSTEM = """你是一名专业的 A 股投顾型 AI 助手。请基于给定事实回答用户问题，遵守：
1. 结构：先给结论，再列 1~2 条依据（必须引用事实中的数字），最后说明风险与局限。
2. 语言专业、平实；专业术语首次出现时用一句话解释；不堆砌形容词。
3. 只使用事实中提供的数据；缺失的数据明确说“当前数据没有覆盖”，不要猜测或编造。
4. 不预测涨跌、不推荐买卖、不承诺收益；不得出现“稳赚、必涨、保证收益”等表述。
5. 结尾固定一句：“以上基于历史回测数据，不代表未来收益，不构成投资建议。”
6. 控制在 350 字以内。"""


class AgentError(RuntimeError):
    """对话或计划生成失败。"""


class ComplianceError(AgentError):
    """输出包含违规承诺或荐股措辞。"""


def ensure_compliance(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise AgentError("模型返回空回复")
    for word in BANNED_WORDS:
        if word in text:
            raise ComplianceError("回复包含不合规措辞，已拦截")
    return text


def chunk_text(text: str, size: int = 12) -> Iterator[str]:
    """把完整回复切成小片，模拟逐字输出（内容本身来自模型）。"""
    cleaned = text.strip()
    for start in range(0, len(cleaned), size):
        yield cleaned[start:start + size]


def confirmation_hint(state: dict[str, Any], plan: Plan) -> str:
    """需要用户确认但计划里没有动作时，返回服务端标准确认话术。"""
    if plan.actions:
        return ""
    profile = state.get("profile") or {}
    if profile.get("complete") and not profile.get("confirmed"):
        return CONFIRM_PROFILE_HINT
    if state.get("pending"):
        return CONFIRM_PENDING_HINT
    return ""


CONFIRMATION_MARKERS = ("【请确认】", "【待确认】", "回复「确认」", "回复「同意启用」")


def has_confirmation_request(text: str) -> bool:
    return any(marker in text for marker in CONFIRMATION_MARKERS)


def attach_confirmation(reply: str, hint: str) -> str:
    if not hint:
        return reply
    if has_confirmation_request(reply):
        return reply
    return reply.rstrip() + "\n\n" + hint


def ensure_pending_reminder(reply: str) -> str:
    """在仍有待审批方案时，确保回复末尾出现明确确认要求。"""
    if "【待确认】" in reply or "【请确认】" in reply:
        return reply
    return reply.rstrip() + "\n\n" + CONFIRM_PENDING_HINT


class Agent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ------------------------------------------------------------------
    def plan(self, state: dict[str, Any], history: list[dict[str, str]],
             message: str) -> Plan:
        payload = json.dumps(
            {"state": state, "user_message": message, "recent_history": history[-8:]},
            ensure_ascii=False)
        try:
            raw = self.llm.chat_json(
                PLANNER_SYSTEM, [{"role": "user", "content": payload}])
        except LLMError as exc:
            raise AgentError(f"AI 暂时无法理解这句话：{exc}") from exc
        reply = str(raw.get("reply", "")).strip()
        if not reply:
            reply = "我在。请继续补充你的投资目标、期限，或告诉我下一步想做什么。"
        ensure_compliance(reply)
        actions = self._parse_actions(raw.get("actions"))
        delta = raw.get("profile_delta")
        if not isinstance(delta, dict):
            delta = {}
        plan = Plan(reply=reply, actions=actions, profile_delta=delta)
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
    """把用户问题压成一段短文本，避免把超长内容塞进提示词。"""
    return re.sub(r"\s+", " ", text.strip())[:500]
