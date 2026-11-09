# -*- coding: utf-8 -*-
"""对话 Agent：原生 function calling 规划 + 流式回复。

- 回复文本通过 content 增量实时产出；
- 动作只通过 tools/tool_calls 表达，服务端按 schema 校验后执行；
- 画像更新走 update_profile 工具，不再混在回复 JSON 里。
"""
from __future__ import annotations

import json
import re
import uuid
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


# 从流式 tool_call 参数 JSON 中尽力恢复 write_model_program 的文件内容，用于边生成边展示
_JSON_PATH_RE = re.compile(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"')
_JSON_CONTENT_RE = re.compile(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)')


def _decode_json_string_fragment(fragment: str) -> str:
    """解码 JSON 字符串片段（允许尾部是尚未写完的转义序列）。"""
    for cut in range(len(fragment), max(-1, len(fragment) - 16), -1):
        try:
            return json.loads('"' + fragment[:cut] + '"')
        except (json.JSONDecodeError, ValueError):
            continue
    return ""


def preview_program_arguments(raw_arguments: str) -> dict[str, Any] | None:
    """从 write_model_program 的流式参数里提取当前已生成的文件快照。"""
    raw = str(raw_arguments or "")
    if '"content"' not in raw:
        return None
    paths = [(match.start(), _decode_json_string_fragment(match.group(1)))
             for match in _JSON_PATH_RE.finditer(raw)]
    files: dict[str, str] = {}
    active = ""
    for match in _JSON_CONTENT_RE.finditer(raw):
        path = "main.py"
        for position, candidate in paths:
            if position >= match.start():
                break
            if candidate:
                path = candidate
        content = _decode_json_string_fragment(match.group(1))
        files[path] = content
        active = path
    if not files:
        return None
    return {"files": files, "active": active}


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
        # 富卡片 blocks（净值序列、程序 diff、寻优历史等）只用于前端渲染，
        # 不发给 Planner，避免上下文被大量结构化数据撑爆。
        compact_history = [
            {"role": str(item.get("role") or ""),
             "content": str(item.get("content") or "")[:1200]}
            for item in history[-8:]
            if isinstance(item, dict)
        ]
        return json.dumps({
            "state": state,
            "user_message": message,
            "recent_history": compact_history,
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
        program_preview: dict[str, Any] = {"id": "", "last_total": -1,
                                           "raw": "", "active": ""}
        try:
            for kind, payload in self.llm.chat_tools_stream(
                    _planner_system(), [{"role": "user", "content": user_content}],
                    TOOL_DEFINITIONS):
                if kind == "content":
                    reply_parts.append(payload)
                    yield "token", payload
                elif kind == "result":
                    result = payload
                elif kind == "tool_args":
                    if str(payload.get("name") or "") != "write_model_program":
                        continue
                    raw_arguments = str(payload.get("arguments") or "")
                    snapshot = preview_program_arguments(raw_arguments)
                    if not snapshot:
                        continue
                    program_preview["raw"] = raw_arguments
                    if not program_preview["id"]:
                        program_preview["id"] = "program-" + uuid.uuid4().hex[:8]
                    total = sum(len(value) for value in snapshot["files"].values())
                    if total - program_preview["last_total"] >= 60:
                        program_preview["last_total"] = total
                        yield "program", {
                            "id": program_preview["id"],
                            "type": "program",
                            "status": "writing",
                            "files": snapshot["files"],
                            "active": snapshot["active"],
                        }
        except LLMError as exc:
            raise AgentError(f"AI 暂时无法理解这句话：{exc}") from exc
        if program_preview["id"] and program_preview["raw"]:
            snapshot = preview_program_arguments(program_preview["raw"])
            if snapshot:
                yield "program", {
                    "id": program_preview["id"],
                    "type": "program",
                    "status": "generated",
                    "files": snapshot["files"],
                    "active": snapshot["active"],
                }
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

    # ------------------------------------------------------------------
    # AI 程序（write_model_program）：二次审核 + 审核不通过时自动修复一次
    # ------------------------------------------------------------------
    def review_program(self, *, files: dict[str, str], entry: str,
                       summary: str, objective: str, packages: list[str],
                       request_text: str, repair_attempt: bool = False) -> dict[str, Any]:
        """让 LLM 对 AI 刚生成的程序做二次审核，返回结构化审核结果。"""
        payload = {
            "customer_request": request_text,
            "summary": summary,
            "objective": objective,
            "files": [{"path": path, "content": files[path]} for path in sorted(files)],
            "packages": list(packages),
            "entry": entry,
            "repair_attempt": bool(repair_attempt),
            "contract": (
                "入口文件定义 build_transform(params=None) -> transform(z, weights)；"
                "可选 select_stocks(scores, top_n, params=None)；z 只含 10 个 z_* 因子列；"
                "多文件模块按文件名互相 import；沙箱禁止文件/网络/系统/动态执行。"
            ),
        }
        try:
            raw = self.llm.chat_json(
                _load_prompt("program_review.txt"),
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            )
        except LLMError as exc:
            raise AgentError(f"程序二次审核失败：{exc}") from exc
        if not isinstance(raw, dict) or "approved" not in raw:
            raise AgentError("程序二次审核没有返回有效的 JSON 结果")
        issues = [str(item).strip() for item in raw.get("issues") or [] if str(item).strip()]
        suggestions = [str(item).strip() for item in raw.get("suggestions") or []
                       if str(item).strip()]
        approved_packages = [str(item).strip().lower()
                             for item in raw.get("approved_packages") or []
                             if str(item).strip()]
        risk = str(raw.get("risk_level") or "medium").strip().lower()
        if risk not in ("low", "medium", "high"):
            risk = "medium"
        return {
            "approved": bool(raw.get("approved")),
            "risk_level": risk,
            "issues": issues[:8],
            "suggestions": suggestions[:8],
            "approved_packages": approved_packages,
            "package_notes": [str(item).strip() for item in raw.get("package_notes") or []
                              if str(item).strip()][:8],
            "summary": str(raw.get("summary") or "").strip()[:300],
        }

    def repair_program(self, *, files: dict[str, str], summary: str, objective: str,
                       packages: list[str], request_text: str,
                       review: dict[str, Any]) -> dict[str, Any]:
        """审核不通过时让 LLM 按审核意见自动改写程序（只自动修复一次）。"""
        payload = {
            "customer_request": request_text,
            "files": [{"path": path, "content": files[path]} for path in sorted(files)],
            "packages": list(packages),
            "review": review,
            "objective": objective,
        }
        try:
            raw = self.llm.chat_json(
                _load_prompt("program_repair.txt"),
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            )
        except LLMError as exc:
            raise AgentError(f"程序自动修复失败：{exc}") from exc
        if not isinstance(raw, dict):
            raise AgentError("程序自动修复没有返回有效的 JSON 结果")
        try:
            return validate_tool_call("write_model_program", raw)
        except ToolArgsError as exc:
            raise AgentError(f"自动修复后的程序参数不合法：{exc.detail}") from exc


def safe_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:500]
