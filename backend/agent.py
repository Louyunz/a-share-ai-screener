# -*- coding: utf-8 -*-
"""对话 Agent：原生 function calling 规划 + 流式回复。

- 回复文本通过 content 增量实时产出；
- 动作只通过 tools/tool_calls 表达，服务端按 schema 校验后执行；
- 画像更新走 update_profile 工具，不再混在回复 JSON 里。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from .chat.action_schema import TOOL_DEFINITIONS, ToolArgsError, validate_tool_call
from .llm import LLMClient, LLMError, ToolChatResult
from .schemas import Plan, SLOT_DEFS, profile_payload

BANNED_WORDS = ("稳赚", "必涨", "保证收益", "推荐买入", "满仓干", "包赚", "翻倍牛股")

META_PHRASES = (
    "先问第一个问题",
    "第一个问题",
    "作为AI",
    "作为 AI",
    "作为一个人工智能",
    "希望以上",
)
PROMPT_DIR = Path(__file__).with_name("prompts")


def _load_prompt(name: str, **values: str) -> str:
    try:
        template = (PROMPT_DIR / name).read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentError(f"提示词文件读取失败：{name}") from exc
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _planner_system() -> str:
    slots = json.dumps(
        [{"key": item["key"], "label": item["label"],
          "required": item["required"], "hint": item["hint"]}
         for item in SLOT_DEFS],
        ensure_ascii=False,
    )
    return _load_prompt("planner.txt", SLOT_DEFS=slots)


def chunk_text(text: str, size: int = 16) -> Iterator[str]:
    if size <= 0:
        raise ValueError("分片长度必须大于 0")
    return (text[index:index + size] for index in range(0, len(text), size))




class AgentError(RuntimeError):
    """对话或计划生成失败。"""


class ComplianceError(AgentError):
    """输出包含违规承诺、荐股措辞或明显的 AI 套话。"""


def ensure_compliance(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise AgentError("模型异常，请重试")
    for word in BANNED_WORDS:
        if word in text:
            raise ComplianceError("回复出错，请重试")
    for phrase in META_PHRASES:
        if phrase in text:
            raise ComplianceError("回复出错，请重试")
    return text.strip()




def asks_clarifying_question(reply: str) -> bool:
    return "？" in reply or "?" in reply


def _state_blocked_action(action: dict[str, Any], state: dict[str, Any]) -> bool:
    """按会话状态判断动作是否不可能执行（只做状态校验，不解析自然语言）。"""
    name = str(action.get("name") or "")
    if name == "rollback_version":
        return not bool(state.get("versions"))
    if name == "apply_pending_change":
        candidates = state.get("pending_candidates")
        waiting = [item for item in candidates or []
                   if isinstance(item, dict)
                   and str(item.get("status") or "pending") != "applied"]
        return not (state.get("pending") or waiting)
    return False


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
    def _content_repair_messages(
            user_content: str, result: ToolChatResult) -> list[dict[str, Any]]:
        message = result.message if isinstance(result.message, dict) else {}
        assistant = {
            "role": "assistant",
            "content": str(message.get("content") or ""),
            "tool_calls": message.get("tool_calls") or [],
        }
        tool_messages = [
            {"role": "tool", "tool_call_id": call.id,
             "content": "工具调用已接受。"}
            for call in result.tool_calls
        ]
        correction = (
            "必须在保留正确工具调用的同时，补充一句给客户看的简短确认，"
            "不要输出 JSON、工具参数或内部步骤。"
        )
        return [{"role": "user", "content": user_content}, assistant,
                *tool_messages, {"role": "user", "content": correction}]

    def stream_plan(self, state: dict[str, Any], history: list[dict[str, Any]],
                    message: str) -> Iterator[tuple[str, Any]]:
        """产出 ("token", 文本增量) 若干次，最后产出 ("plan", Plan)。"""
        user_content = self._payload(state, history, message)
        reply_parts: list[str] = []
        result: ToolChatResult | None = None
        try:
            for kind, payload in self.llm.chat_tools_stream(
                    _planner_system(), [{"role": "user", "content": user_content}],
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
                    _planner_system(),
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
                    _planner_system(), review_messages, TOOL_DEFINITIONS)
            except LLMError:
                reviewed = None
            if reviewed is not None and reviewed.tool_calls:
                actions, profile_delta, errors = self._actions_from_result(reviewed)
                if errors:
                    raise AgentError("AI 工具参数校验失败：" + "；".join(errors))
                result = reviewed

        if not reply_parts and result.content:
            reply_parts.append(result.content)
            yield "token", result.content

        if not reply_parts and result.tool_calls:
            try:
                repaired = self.llm.chat_tools(
                    _planner_system(),
                    self._content_repair_messages(user_content, result),
                    TOOL_DEFINITIONS,
                )
            except LLMError as exc:
                raise AgentError(f"AI 回复修复失败：{exc}") from exc
            repaired_actions, repaired_profile_delta, errors = (
                self._actions_from_result(repaired))
            if errors:
                raise AgentError("AI 工具参数校验失败：" + "；".join(errors))
            if repaired.tool_calls:
                actions = repaired_actions
                profile_delta = repaired_profile_delta
            result = repaired
            if result.content:
                reply_parts.append(result.content)
                yield "token", result.content

        reply = "".join(reply_parts).strip()
        if not reply:
            raise AgentError("模型没有给出回复文本")
        ensure_compliance(reply)

        plan = Plan(reply=reply, actions=actions, profile_delta=profile_delta)
        if plan.actions and asks_clarifying_question(plan.reply):
            plan.deferred_intent = plan.actions[0]["name"]
            plan.actions = []
        # 状态兜底：模型偶尔会在“没有可回滚版本/没有待审批方案”时仍调用工具。
        # 这类动作不可能执行，直接丢弃（服务端 chat_stream 仍会做二次校验），
        # 避免用户只看到一条错误、却被留下一条没有回复的孤消息。
        plan.actions = [action for action in plan.actions
                        if not _state_blocked_action(action, state)]
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
            text = self.llm.chat_text(_load_prompt("explain.txt"), [{"role": "user", "content": payload}])
        except LLMError as exc:
            raise AgentError(f"解读生成失败：{exc}") from exc
        return ensure_compliance(text)


def safe_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:500]
