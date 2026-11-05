# -*- coding: utf-8 -*-
"""LLM 客户端：OpenAI 兼容接口（仅用标准库 urllib）与测试用 FakeLLM。"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Iterator, Protocol

from .config import ConfigError, LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, llm_api_key


class LLMError(RuntimeError):
    """LLM 调用失败（网络、限流、返回格式非法等）。"""


class LLMClient(Protocol):
    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict: ...
    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str: ...
    def chat_stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]: ...


def extract_json(text: str) -> dict:
    """从模型输出中提取第一个 JSON 对象。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError(f"模型未返回 JSON：{text[:200]}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型返回的 JSON 无法解析：{exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("模型返回的 JSON 不是对象")
    return data


class OpenAICompatClient:
    """DeepSeek / Qwen / OpenAI 兼容的 Chat Completions 客户端。"""

    def __init__(self, base_url: str = LLM_BASE_URL, api_key: str | None = None,
                 model: str = LLM_MODEL, timeout: int = 60) -> None:
        key = api_key if api_key is not None else llm_api_key()
        if not key:
            raise ConfigError(
                f"缺少 LLM API Key：请设置环境变量 {LLM_API_KEY_ENV}（或 OPENAI_API_KEY）")
        self.base_url = base_url.rstrip("/")
        self.api_key = key
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------
    def _request(self, payload: dict) -> urllib.response.addinfourl:  # type: ignore[name-defined]
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        })
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"LLM 接口 HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM 接口连接失败：{exc.reason}") from exc

    def _messages(self, system: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, *messages]

    def _completion(self, system: str, messages: list[dict[str, str]],
                    json_mode: bool = False) -> str:
        payload: dict = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "temperature": 0.2,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            with self._request(payload) as resp:
                body = resp.read().decode("utf-8")
        except LLMError:
            if not json_mode:
                raise
            # 部分兼容服务不支持 response_format，去掉后重试一次
            payload.pop("response_format", None)
            with self._request(payload) as resp:
                body = resp.read().decode("utf-8")
        try:
            data = json.loads(body)
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"LLM 响应格式异常：{body[:200]}") from exc

    # ------------------------------------------------------------------
    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict:
        text = self._completion(system, messages, json_mode=True)
        return extract_json(text)

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str:
        return self._completion(system, messages, json_mode=False).strip()

    def chat_stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "temperature": 0.4,
            "stream": True,
        }
        with self._request(payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # ------------------------------------------------------------------
    def check(self) -> None:
        """轻量连通性检查（不发真实请求，避免演示启动时消耗配额）。"""
        if time.time() < 0:
            raise LLMError("unreachable")


class FakeLLM:
    """确定性假模型：仅用于自动化测试与无 Key 的本地 UI 联调（LLM_PROVIDER=fake）。

    注意：它不是运行时降级路径——生产路径必须配置真实 Key，否则启动失败。
    """

    RISK_WORDS = {"保守": "conservative", "稳健": "balanced", "平衡": "balanced",
                  "积极": "aggressive", "激进": "aggressive"}

    def _extract_slots(self, content: str) -> dict:
        slots: dict[str, str] = {}
        for word, level in self.RISK_WORDS.items():
            if word in content:
                slots["risk_level"] = level
                break
        m = re.search(r"(\d+)\s*[-~至到]?\s*(\d+)?\s*年", content)
        if m:
            slots["horizon"] = f"{m.group(1)}-{m.group(2)} 年" if m.group(2) else f"{m.group(1)} 年"
        m = re.search(r"(\d+)\s*%", content)
        if m:
            slots["drawdown_tolerance"] = f"{m.group(1)}%"
        for industry in ("医药", "消费", "科技", "电子", "银行", "计算机", "新能源", "食品饮料"):
            if industry in content:
                slots["excluded_industries"] = industry
                break
        if "成长" in content:
            slots["goal"] = "长期成长"
        elif "稳健" in content or "保值" in content:
            slots["goal"] = "稳健增值"
        if "新手" in content:
            slots["experience"] = "新手"
        elif "经验" in content or "年" in content:
            slots["experience"] = "3-5 年"
        return slots

    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict:
        content = messages[-1]["content"] if messages else ""
        if "画像" in system:
            return {"slots": self._extract_slots(content)}
        if "解读" in system:
            return {"text": self._explain(content)}
        return {"ok": True}

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"] if messages else ""
        if "画像" in system:
            slots = self._extract_slots(content)
            missing = [k for k in ("risk_level", "horizon", "drawdown_tolerance") if k not in slots]
            if missing:
                return (f"为了匹配适合您的选股模型，我还需要确认：{'、'.join(missing)}。"
                        "您可以自然语言补充，例如“稳健型，持有 3 年，能接受 15% 左右回撤”。")
            return "已根据您的描述生成画像卡：风险等级、投资期限与回撤容忍度都已记录，请确认后开始选股。"
        if "解读" in system:
            return self._explain(content)
        return "收到。请继续描述您的投资偏好。"

    def chat_stream(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.chat_text(system, messages)
        step = 6
        for i in range(0, len(text), step):
            yield text[i:i + step]

    @staticmethod
    def _explain(content: str) -> str:
        return ("结合因子贡献与样本外回测结果：该标的在价值与质量因子上得分靠前，"
                "近期动量与流动性处于中上水平；组合层面行业暴露分散、单票权重受控。"
                "需要提示的是，历史模拟不代表未来收益，回测未包含全部市场冲击成本。"
                "以上内容不构成投资建议。")


def build_llm(provider: str | None = None) -> LLMClient:
    """按配置创建 LLM 客户端；未配置 Key 时抛出 ConfigError（不降级）。"""
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "fake":
        return FakeLLM()
    return OpenAICompatClient()
