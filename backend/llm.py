# -*- coding: utf-8 -*-
"""LLM 客户端（OpenAI 兼容接口）。"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from .config import LLM_BASE_URL, LLM_MODEL


class LLMError(RuntimeError):
    """模型调用或输出解析失败。"""


@dataclass
class ToolCall:
    """LLM 发起的一次工具调用（OpenAI function calling 协议）。"""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ToolChatResult:
    """带工具调用的一次 chat 响应。"""

    content: str
    tool_calls: list[ToolCall]
    message: dict[str, Any]


class LLMClient(Protocol):
    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str: ...

    def chat_tools(self, system: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]],
                   tool_choice: str = "auto") -> ToolChatResult: ...

    def chat_tools_stream(self, system: str, messages: list[dict[str, Any]],
                          tools: list[dict[str, Any]],
                          tool_choice: str = "auto") -> Iterator[tuple[str, Any]]: ...

    def chat_text_stream(self, system: str,
                         messages: list[dict[str, str]]) -> Iterator[str]: ...


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出中提取第一个 JSON 对象（容忍 ```json 包裹）。"""
    if not isinstance(text, str):
        raise LLMError("模型输出不是文本")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("模型输出中没有 JSON")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型输出 JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("模型输出必须是 JSON 对象")
    return data


class OpenAICompatClient:
    """OpenAI 兼容 /chat/completions 客户端（DeepSeek 等）。"""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"模型接口 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"模型接口连接失败：{exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"模型接口响应异常：{exc}") from exc
        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型接口返回格式异常：{body}") from exc

    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        try:
            message = self._post(payload)
        except LLMError:
            # 部分兼容实现不支持 response_format，去掉后重试一次
            payload.pop("response_format", None)
            message = self._post(payload)
        return extract_json(str(message.get("content", "")))

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.6,
        }
        message = self._post(payload)
        text = str(message.get("content", "")).strip()
        if not text:
            raise LLMError("模型返回空文本")
        return text

    def chat_text_stream(self, system: str,
                         messages: list[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.6,
            "stream": True,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        produced = False
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    text = (choices[0].get("delta") or {}).get("content") or ""
                    if text:
                        produced = True
                        yield text
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"模型接口 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"模型接口连接失败：{exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"模型接口流式响应异常：{exc}") from exc
        if not produced:
            raise LLMError("模型返回空文本")

    @staticmethod
    def _tool_calls_from_message(message: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            name = str(function.get("name") or "").strip()
            raw = function.get("arguments")
            if isinstance(raw, str):
                raw_text = raw
                try:
                    arguments = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError as exc:
                    raise LLMError(
                        f"工具 {name or '<unknown>'} 参数不是合法 JSON：{raw[:200]}"
                    ) from exc
            elif isinstance(raw, dict):
                arguments = raw
                raw_text = json.dumps(raw, ensure_ascii=False)
            else:
                arguments = {}
                raw_text = ""
            if not isinstance(arguments, dict):
                raise LLMError(f"工具 {name or '<unknown>'} 参数必须是 JSON 对象")
            calls.append(ToolCall(
                id=str(item.get("id") or f"call_{index + 1}"),
                name=name,
                arguments=arguments,
                raw_arguments=raw_text,
            ))
        return calls

    def chat_tools(self, system: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]],
                   tool_choice: str = "auto") -> ToolChatResult:
        """OpenAI 兼容的 function calling：模型主动调用工具并返回参数。"""
        if not tools:
            raise LLMError("chat_tools 至少需要一个工具定义")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0.2,
        }
        message = self._post(payload)
        content = str(message.get("content") or "").strip()
        calls = self._tool_calls_from_message(message)
        if not content and not calls:
            raise LLMError("模型既没有返回文本，也没有调用工具")
        return ToolChatResult(content=content, tool_calls=calls, message=message)

    def chat_tools_stream(self, system: str, messages: list[dict[str, Any]],
                          tools: list[dict[str, Any]],
                          tool_choice: str = "auto") -> Iterator[tuple[str, Any]]:
        """流式 function calling：逐个产出 ("content", 增量) / ("result", ToolChatResult)。"""
        if not tools:
            raise LLMError("chat_tools_stream 至少需要一个工具定义")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0.2,
            "stream": True,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        content_parts: list[str] = []
        pending: dict[int, dict[str, str]] = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        content_parts.append(text)
                        yield "content", text
                    for item in delta.get("tool_calls") or []:
                        index = int(item.get("index", len(pending)))
                        entry = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if item.get("id"):
                            entry["id"] = str(item["id"])
                        function = item.get("function") or {}
                        if function.get("name"):
                            entry["name"] += str(function["name"])
                        if function.get("arguments"):
                            entry["arguments"] += str(function["arguments"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"模型接口 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"模型接口连接失败：{exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"模型接口流式响应异常：{exc}") from exc

        message: dict[str, Any] = {"role": "assistant",
                                   "content": "".join(content_parts)}
        if pending:
            wire_calls = []
            for index in sorted(pending):
                entry = pending[index]
                wire_calls.append({
                    "id": entry["id"] or f"call_{index + 1}",
                    "type": "function",
                    "function": {"name": entry["name"],
                                 "arguments": entry["arguments"] or "{}"},
                })
            message["tool_calls"] = wire_calls
        calls = self._tool_calls_from_message(message)
        content = message["content"].strip()
        if not content and not calls:
            raise LLMError("模型既没有返回文本，也没有调用工具")
        yield "result", ToolChatResult(content=content, tool_calls=calls, message=message)


def build_llm() -> LLMClient:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise LLMError("未配置 DEEPSEEK_API_KEY；请设置环境变量后重启")
    return OpenAICompatClient(api_key=api_key, base_url=LLM_BASE_URL, model=LLM_MODEL)
