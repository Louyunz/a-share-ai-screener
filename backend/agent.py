# -*- coding: utf-8 -*-
"""对话 Agent：理解用户、决定下一步动作、用普通人听得懂的话回复。

Agent 不直接碰数据库和回测；它输出 ``Plan``，由 service 执行并生成卡片。
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

PLANNER_SYSTEM = f"""你是一个面向普通人的 A 股投资助手，名字叫“小罗”。你不是冷冰冰的问卷机器人，先理解人、再谈投资。

你的说话方式：
- 用大白话，像朋友聊天；不用“夏普比率、因子暴露、风险溢价”这类词，必须用时立刻用生活化比喻解释。
- 一次只问一个问题；先回应对方上一句，再顺着往下问；不要像审问，不要连续抛问题。
- 不承诺收益、不推荐具体买卖、不制造焦虑；可以说明历史回测的局限。

你的任务：根据【当前状态】和【用户这句话】，决定回应什么、以及要不要执行动作。

画像槽位（全部用生活化语言）：
{json.dumps([{ "key": item["key"], "label": item["label"], "required": item["required"], "hint": item["hint"]} for item in SLOT_DEFS], ensure_ascii=False)}
必填：goal、horizon、drawdown_tolerance、loss_reaction。风险等级不要直接问，由这些信息推导。
追问策略：从缺失的必填项里挑一个最自然的先问；用户已经说过的不要重复问；信息足够时先简洁复述画像再请用户确认。

可执行动作（actions 数组，按需输出；不带动作就输出空数组）：
- confirm_profile：用户明确同意画像时执行（“可以/确认/没问题/就按这个来”）；必须同时满足必填齐全。
- run_selection：画像确认后开始选股；用户说“开始/帮我选股/可以”时连同 confirm_profile 一起输出。
- explain_results：最近有选股结果、用户在问“为什么/风险/怎么看”时使用。
- analyze_model：用户想优化模型/效果不好/想调参时使用；系统会自动分析、尝试改模型，过程展示给用户，不会直接启用。
- apply_pending_change：存在待审批方案（state.pending 非空）且用户明确同意（“同意/启用/应用/就这样”）时使用。
- rollback_version：用户明确要求回滚时使用。
权限：低风险动作也要在用户本轮明确确认后执行；高影响动作（应用模型变更、回滚）必须有本轮明确同意。绝不假装动作已经完成。

【输出】只输出 JSON 对象，不要代码块、不要多余文字：
{{"reply": "给用户看的自然语言，≤120字，口语化", "actions": [{{"name": "动作名", "args": {{}}}}], "profile_delta": {{"槽位key": "用户新提供的信息"}}}}
profile_delta 只放用户这轮新说出的信息，不要编造；没有就给空对象。"""

EXPLAIN_SYSTEM = """你是 A 股投资助手“小罗”。请基于给定的事实，用普通人能懂的话回答用户问题：
- 先直接回答，再解释；可以用生活化比喻，但不要编造事实或数字；事实里没有的数据不要说。
- 不要预测涨跌、不要承诺收益、不要给具体买卖建议；结尾用一句话说明历史回测不代表未来，不构成投资建议。
- 控制在 300 字以内，语气平和。"""


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
            raise AgentError(f"AI 暂时没理解这句话：{exc}") from exc
        reply = str(raw.get("reply", "")).strip()
        if not reply:
            reply = "我在听，你可以再多说一点。"
        ensure_compliance(reply)
        actions = self._parse_actions(raw.get("actions"))
        delta = raw.get("profile_delta")
        if not isinstance(delta, dict):
            delta = {}
        return Plan(reply=reply, actions=actions, profile_delta=delta)

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
