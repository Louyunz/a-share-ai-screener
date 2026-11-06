# -*- coding: utf-8 -*-
"""LLM 客户端（OpenAI 兼容接口）与测试/构建用确定性模型。

生产路径只使用真实模型；``LLM_PROVIDER=fake`` 仅用于本地无 Key 测试与自动化验收。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Iterator, Protocol

from .config import LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER


class LLMError(RuntimeError):
    """模型调用或输出解析失败。"""


class LLMClient(Protocol):
    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str: ...


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


class FakeLLM:
    """确定性模型：驱动无 Key 的自动化测试与现场演示。

    匹配规则只依赖输入 JSON 的键（planner / proposal_request / facts），
    不猜测自然语言；真实产品路径使用 OpenAICompatClient。
    """

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def chat_json(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        content = messages[-1]["content"] if messages else ""
        payload = self._parse(content)
        if not payload:
            try:
                payload = extract_json(content)
            except LLMError:
                payload = {}
        if "proposal_request" in payload:
            request = payload["proposal_request"]
            return self._proposal(request if isinstance(request, dict) else payload)
        if "current_model" in payload and "factor_stats" in payload:
            return self._proposal(payload)
        if "facts" in payload:
            return {"text": self._explain(payload["facts"], payload.get("question", ""))}
        return self._plan(payload)

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"] if messages else ""
        payload = self._parse(content)
        if "facts" in payload:
            return self._explain(payload["facts"], payload.get("question", ""))
        return str(payload.get("text", "我在，有什么可以帮你？"))

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _parse(content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _slots_from_text(text: str) -> dict[str, str]:
        slots: dict[str, str] = {}
        if not text:
            return slots
        if re.search(r"养老|退休", text):
            slots["goal"] = "退休养老"
        elif re.search(r"买房|首付", text):
            slots["goal"] = "攒钱买房"
        elif re.search(r"孩子|教育", text):
            slots["goal"] = "孩子教育"
        elif re.search(r"增值|攒钱|理财|闲钱", text):
            slots["goal"] = "让闲钱稳健增值"
        horizon = re.search(r"(\d+)\s*年", text)
        if horizon:
            slots["horizon"] = f"{horizon.group(1)} 年"
        elif "长期" in text:
            slots["horizon"] = "长期（3 年以上）"
        loss = re.search(r"(\d+)\s*%", text)
        if loss:
            slots["drawdown_tolerance"] = f"{loss.group(1)}%"
        if re.search(r"慌|睡不着|害怕|担心", text):
            slots["loss_reaction"] = "会有点慌，但先不卖"
        if re.search(r"继续持有|不卖|扛住", text):
            slots["loss_reaction"] = "能扛住，继续持有"
        if re.search(r"割肉|马上卖", text):
            slots["loss_reaction"] = "会想先卖掉"
        capital = re.search(r"(\d+)\s*万", text)
        if capital:
            slots["capital"] = f"{capital.group(1)} 万左右"
        if re.search(r"不碰|坚决不买|不买|排除", text):
            for industry in ("白酒", "消费", "医药", "银行", "新能源", "地产", "半导体", "军工", "食品饮料"):
                if industry in text:
                    slots["excluded_industries"] = industry
                    break
        if re.search(r"看好|偏好", text):
            for industry in ("消费", "医药", "科技", "新能源", "银行", "半导体"):
                if industry in text:
                    slots["preferred_industries"] = industry
                    break
        if re.search(r"没买过|第一次|新手", text):
            slots["experience"] = "新手，没怎么买过"
        elif re.search(r"买过基金|买过股票|有经验", text):
            slots["experience"] = "有一些经验"
        if re.search(r"稳定|工资|固定收入", text):
            slots["income_stability"] = "收入比较稳定"
        if "灵活" in text:
            slots["trade_frequency"] = "希望灵活调整"
        return slots

    @staticmethod
    def _confirm(text: str) -> bool:
        return any(word in text for word in ("可以", "确认", "没问题", "同意", "好的", "就按这个", "开始吧", "按这个来"))

    @staticmethod
    def _agree(text: str) -> bool:
        return any(word in text for word in ("同意", "启用", "应用", "换上", "就这样", "可以"))

    @staticmethod
    def _wants_optimize(text: str) -> bool:
        return any(word in text for word in ("优化", "调参", "改进", "效果不好", "模型", "提升", "调整"))

    @staticmethod
    def _wants_rollback(text: str) -> bool:
        return any(word in text for word in ("回滚", "退回", "恢复上一个", "换回之前"))

    @staticmethod
    def _is_question(text: str) -> bool:
        return any(mark in text for mark in ("为什么", "风险", "怎么看", "如何", "怎么", "?", "？", "吗"))

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------
    def _plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload.get("state") or {}
        message = str(payload.get("user_message", ""))
        profile = state.get("profile") or {}
        slots = dict(profile.get("slots") or {})
        delta = {key: value for key, value in self._slots_from_text(message).items()
                 if not slots.get(key)}
        slots.update(delta)
        complete = all(slots.get(key) for key in ("goal", "horizon", "drawdown_tolerance", "loss_reaction"))
        pending = state.get("pending")
        actions: list[dict[str, Any]] = []

        if self._wants_rollback(message) and state.get("versions"):
            actions = [{"name": "rollback_version", "args": {}}]
            reply = "收到。我将切换回上一版本，随后给出当前生效的模型信息。"
        elif pending and self._agree(message):
            actions = [{"name": "apply_pending_change", "args": {}}]
            reply = "收到。我将应用该改进方案并切换模型版本，随后给出对比结果。"
        elif self._wants_optimize(message) and profile.get("confirmed"):
            actions = [{"name": "analyze_model", "args": {}}]
            reply = ("我会先评估当前模型的表现与不足，再给出可验证的改进方案；"
                     "分析与回测过程会逐步展示，未经你确认不会启用。")
        elif self._confirm(message) and complete and not profile.get("confirmed"):
            actions = [{"name": "confirm_profile", "args": {}},
                       {"name": "run_selection", "args": {}}]
            reply = "已收到确认。我将按该画像执行选股，并给出回测指标与风险说明。"
        elif not complete:
            missing = [key for key in ("goal", "horizon", "drawdown_tolerance", "loss_reaction")
                       if not slots.get(key)]
            labels = {"goal": "这笔钱主要计划用于什么", "horizon": "这笔钱大概多久不需要动用",
                      "drawdown_tolerance": "最多能接受多大的亏损",
                      "loss_reaction": "如果出现亏损，你通常会怎么处理"}
            question = labels.get(missing[0], "请再补充一些你的情况")
            reply = f"为了给出合适的方案，我还需要确认一点：{question}？"
        elif not profile.get("confirmed"):
            reply = ("根据你目前提供的信息，我已整理出投资画像。\n\n"
                     "【请确认】如果画像符合你的情况，请回复「确认」，我将按该画像开始选股；"
                     "如需补充或调整，请直接说明。")
        elif state.get("last_selection") and self._is_question(message):
            actions = [{"name": "explain_results", "args": {"question": message}}]
            reply = "我会基于本次选股的真实回测数据说明，并标注数据边界。"
        else:
            reply = ("已记录。你可以继续补充投资情况，或让我执行：选股、解释本次结果、优化模型。")

        plan: dict[str, Any] = {"reply": reply, "actions": actions, "profile_delta": delta}
        return plan

    # ------------------------------------------------------------------
    # 建模提案
    # ------------------------------------------------------------------
    @staticmethod
    def _proposal(request: dict[str, Any]) -> dict[str, Any]:
        base = dict(request.get("weights") or {})
        weights = {name: float(base.get(name, 0.1)) for name in (
            "ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
            "momentum_60", "reversal_5", "volatility_60", "liquidity_20", "size")}
        total = sum(weights.values()) or 1.0
        weights = {name: round(value / total, 4) for name, value in weights.items()}
        code = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "def build_transform():\n"
            "    def transform(z, weights):\n"
            "        score = pd.Series(0.0, index=z.index, dtype=float)\n"
            "        for factor, weight in weights.items():\n"
            "            column = 'z_' + factor\n"
            "            if column in z.columns:\n"
            "                values = pd.to_numeric(z[column], errors='coerce').fillna(0.0)\n"
            "                score = score + float(weight) * values\n"
            "        # 让低波动和短期反转多拿一点分，降低组合大起大落\n"
            "        score = score + 0.12 * pd.to_numeric(z['z_volatility_60'], errors='coerce').fillna(0.0)\n"
            "        score = score + 0.08 * pd.to_numeric(z['z_reversal_5'], errors='coerce').fillna(0.0)\n"
            "        return score\n"
            "    return transform\n"
        )
        return {
            "approach": "code",
            "summary": "把打分逻辑往「低波动 + 短期反转」多偏一点，让组合更稳、回撤更小。",
            "weights": weights,
            "code": code,
            "why": "回测显示当前组合波动偏大，低波动和反转因子最近贡献更稳定。",
            "expected_effect": "波动和最大回撤有望下降，收益略降但持有体验更好。",
            "risks": ["低波动策略在快速上涨行情里可能跑输", "这是历史回测，不代表未来"],
        }

    # ------------------------------------------------------------------
    # 解释
    # ------------------------------------------------------------------
    @staticmethod
    def _explain(facts: dict[str, Any], question: str = "") -> str:
        metrics = facts.get("metrics") or {}
        top = facts.get("top") or []
        codes = "、".join(str(item.get("code", "")) for item in top[:3]) or "本次组合"
        parts = [
            f"结论：本次组合主要持有 {codes}，筛选逻辑侧重估值合理、盈利质量与走势稳定。",
        ]
        if metrics:
            parts.append(
                f"依据：历史回测年化收益约 {float(metrics.get('cagr', 0)) * 100:.0f}%，"
                f"最大回撤约 {abs(float(metrics.get('mdd', 0))) * 100:.0f}%，"
                f"夏普比率约 {float(metrics.get('sharpe', 0)):.2f}。"
            )
        if question:
            parts.append("风险与局限：上述指标均为历史模拟结果，短期表现可能明显偏离；"
                         "回测无法覆盖政策、流动性与市场结构变化。")
        else:
            parts.append("风险与局限：历史回测无法覆盖未来政策、流动性与市场结构变化。")
        parts.append("以上基于历史回测数据，不代表未来收益，不构成投资建议。")
        return "".join(parts)


def build_llm(provider: str | None = None) -> LLMClient:
    selected = (provider or LLM_PROVIDER or "openai").strip().lower()
    if selected == "fake":
        return FakeLLM()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise LLMError("未配置 DEEPSEEK_API_KEY；请设置环境变量后重启，或仅在本地测试时设置 LLM_PROVIDER=fake")
    return OpenAICompatClient(api_key=api_key, base_url=LLM_BASE_URL, model=LLM_MODEL)
