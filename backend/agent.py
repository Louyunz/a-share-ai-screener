# -*- coding: utf-8 -*-
"""Agent 门面：多轮对话画像、解读生成与合规校验。

设计要点：
- 不降级：LLM/校验失败直接抛异常，由接口层转成 SSE error；
- 画像只走对话：LLM 抽取槽位，缺失项由回复追问，没有问卷路径；
- 合规校验：拦截收益承诺、买卖点等敏感表述。
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from .llm import LLMClient
from .schemas import (
    REQUIRED_SLOTS,
    normalize_slots,
    profile_complete,
    profile_payload,
    risk_level,
)
from .store import Store

PROMPT_PROFILE = """你是 A 股智能选股助手的画像抽取器。
根据对话历史与用户最新发言，抽取以下画像槽位（全部为字符串）：
- risk_level：conservative / balanced / aggressive 三选一，依据风险偏好与回撤容忍判断；
- horizon：投资期限，例如“3-5 年”；
- drawdown_tolerance：可接受的最大回撤，例如“20%”；
- goal：投资目标，例如“稳健增值”“长期成长”；
- experience：投资经验，例如“新手”“3-5 年”；
- excluded_industries：不投资的行业，多个用“、”连接；
- capital：计划投入资金，例如“50 万”。
只输出 JSON，格式：{"slots": {...}}。未提及的槽位不要编造，直接省略。
若用户的表述模糊（如只说“稳一点”），不要自行猜测风险等级。"""

PROMPT_REPLY = """你是“智投罗盘”A 股智能选股助手，语气专业、简洁、克制。
当前已收集的画像槽位：{slots}
仍缺失的必填项：{missing}
规则：
1. 若仍有缺失项，用自然语言追问（一次最多问两个），不要罗列清单式问卷；
2. 若信息齐全，用一两句话总结画像并请用户确认画像卡；
3. 不预测涨跌、不承诺收益、不给出个股买卖点；
4. 回复控制在 120 字以内，中文。"""

PROMPT_EXPLAIN = """你是 A 股投研解读助手。只根据给定事实写一段解读，不得引入外部数据。
事实：{facts}
用户追问：{question}
规则：
1. 每个数字必须来自事实，否则不要使用；
2. 必须包含至少一条风险提示；
3. 禁止出现“稳赚 / 必涨 / 保证收益 / 推荐买入 / 满仓”等表述；
4. 结尾注明“历史模拟不代表未来收益，不构成投资建议”；
5. 控制在 220 字以内。"""

BANNED_WORDS = ("稳赚", "必涨", "保证收益", "推荐买入", "满仓", "包赚")


class AgentError(RuntimeError):
    """Agent 业务错误。"""


class ComplianceError(AgentError):
    """合规校验未通过。"""


class GuardChain:
    """两层校验：槽位结构校验 + 输出合规校验。"""

    def validate_slots(self, slots: dict[str, str]) -> dict[str, str]:
        normalized = normalize_slots(slots)
        unknown_risk = normalized.get("risk_level") and normalized["risk_level"] not in (
            "conservative", "balanced", "aggressive")
        if unknown_risk:
            normalized.pop("risk_level")
        return normalized

    def missing_required(self, slots: dict[str, str]) -> list[str]:
        return [key for key in REQUIRED_SLOTS if not slots.get(key)]

    def check_compliance(self, text: str) -> str:
        hits = [word for word in BANNED_WORDS if word in (text or "")]
        if hits:
            raise ComplianceError(f"输出包含违规表述：{'、'.join(hits)}")
        if len(text or "") > 2000:
            raise AgentError("解读过长，已拒绝返回")
        return text


class AgentFacade:
    def __init__(self, llm: LLMClient, store: Store) -> None:
        self.llm = llm
        self.store = store
        self.guard = GuardChain()

    # ------------------------------------------------------------------
    def stream_chat(self, session_id: str, content: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """处理一轮对话，产出 (event, payload)：slots / token / profile_ready。"""
        history = self.store.get_messages(session_id, limit=12)
        self.store.add_message(session_id, "user", content)

        previous = self.store.get_profile(session_id) or {"slots": {}}
        extraction = self.llm.chat_json(
            PROMPT_PROFILE,
            [*history, {"role": "user", "content": content}],
        )
        slot_delta = extraction.get("slots") if isinstance(extraction.get("slots"), dict) else {}
        slots = self.guard.validate_slots({**(previous.get("slots") or {}), **slot_delta})
        missing = self.guard.missing_required(slots)
        if slots:
            self.store.save_profile(session_id, slots, risk_level(slots), confirmed=False)
        yield "slots", {"slots": slots, "missing": missing}

        system = PROMPT_REPLY.format(
            slots=json.dumps(slots, ensure_ascii=False),
            missing="、".join(missing) if missing else "无",
        )
        reply_parts: list[str] = []
        for chunk in self.llm.chat_stream(system, [{"role": "user", "content": content}]):
            reply_parts.append(chunk)
            yield "token", {"text": chunk}
        reply = "".join(reply_parts).strip()
        self.guard.check_compliance(reply)
        self.store.add_message(session_id, "assistant", reply or "(空)")
        if not missing:
            yield "profile_ready", profile_payload(slots)

    # ------------------------------------------------------------------
    def explain(self, facts: dict[str, Any], question: str = "") -> str:
        prompt = PROMPT_EXPLAIN.format(
            facts=json.dumps(facts, ensure_ascii=False), question=question or "无")
        text = self.llm.chat_text("你是专业的 A 股投研解读助手。", [{"role": "user", "content": prompt}])
        return self.guard.check_compliance(text)

    def confirm_profile(self, session_id: str, slots: dict[str, str]) -> dict[str, Any]:
        normalized = self.guard.validate_slots(slots)
        missing = self.guard.missing_required(normalized)
        if missing:
            raise AgentError(f"画像缺少必填项：{'、'.join(missing)}")
        self.store.save_profile(session_id, normalized, risk_level(normalized), confirmed=True)
        self.store.audit("confirm_profile", to_version=None, reason="user_confirmed")
        return profile_payload(normalized, confirmed=True)
