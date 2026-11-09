# -*- coding: utf-8 -*-
"""测试专用确定性 LLM：只用于结构测试与 E2E，不进入生产包。

生产路径是 ``backend.llm.OpenAICompatClient``（原生 function calling）；
本文件位于 tests/support/，不会被打包进 backend 生产模块。
"""
from __future__ import annotations

import json
import re
from typing import Any

from backend.llm import LLMError, ToolCall, ToolChatResult, extract_json

# 测试/演示用：write_model_program 的确定性多文件程序（覆盖 helpers 模块与选股钩子）
SCRIPTED_PROGRAM_FILES = {
    "main.py": (
        "import pandas as pd\n"
        "import helpers\n"
        "\n"
        "def build_transform(params=None):\n"
        "    options = params or {}\n"
        "    tilt = float(options.get('tilt', 0.5))\n"
        "    def transform(z, weights):\n"
        "        base = helpers.weighted_score(z, weights)\n"
        "        vol = pd.to_numeric(z['z_volatility_60'], errors='coerce').fillna(0.0)\n"
        "        return base + tilt * vol\n"
        "    return transform\n"
        "\n"
        "def select_stocks(scores, top_n, params=None):\n"
        "    ordered = scores.sort_values('factor_score', ascending=False)\n"
        "    return list(ordered.index[:int(top_n)])\n"
    ),
    "helpers.py": (
        "import pandas as pd\n"
        "\n"
        "def weighted_score(z, weights):\n"
        "    total = pd.Series(0.0, index=z.index, dtype=float)\n"
        "    for name, weight in weights.items():\n"
        "        column = 'z_' + name\n"
        "        if column in z.columns:\n"
        "            values = pd.to_numeric(z[column], errors='coerce').fillna(0.0)\n"
        "            total = total + float(weight) * values\n"
        "    return total\n"
    ),
}
SCRIPTED_PROGRAM_PARAM_SPACE = {
    "tilt": {"min": 0.0, "max": 1.0, "step": 0.25, "type": "float", "default": 0.5},
}
SCRIPTED_REJECT_MARKER = "SCRIPTED_REJECT"



class ScriptedLLM:
    """确定性模型：驱动无 Key 的自动化测试与现场演示。

    匹配规则只依赖输入 JSON 的键（tuning_request / planner / proposal_request / facts），
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
        if "files" in payload and "review" in payload:
            return self._repair_program(payload)
        if "files" in payload and "contract" in payload:
            return self._review_program(payload)
        if payload.get("tuning_request"):
            return self._tuning_plan(payload)
        if "facts" in payload:
            return {"text": self._explain(payload["facts"], payload.get("question", ""))}
        return self._plan(payload)

    # ------------------------------------------------------------------
    # AI 程序二次审核（测试替身）
    # ------------------------------------------------------------------
    @staticmethod
    def _program_text(files: Any) -> str:
        if isinstance(files, dict):
            return "\n".join(str(value) for value in files.values())
        if isinstance(files, list):
            return "\n".join(str((item or {}).get("content") or "") for item in files)
        return ""

    def _review_program(self, payload: dict[str, Any]) -> dict[str, Any]:
        packages = [str(item) for item in payload.get("packages") or []]
        text = self._program_text(payload.get("files"))
        if SCRIPTED_REJECT_MARKER in text:
            return {
                "approved": False,
                "risk_level": "high",
                "issues": ["代码包含必被拒绝的测试标记，请移除后重写"],
                "suggestions": ["按审核意见修复后再复审"],
                "approved_packages": [],
                "package_notes": ["测试标记存在时不批准任何依赖"],
                "summary": "审核未通过：存在阻塞问题",
            }
        return {
            "approved": True,
            "risk_level": "low",
            "issues": [],
            "suggestions": [],
            "approved_packages": packages,
            "package_notes": [f"{name}：测试替身默认批准" for name in packages],
            "summary": "审核通过：程序符合契约与客户要求（测试替身）",
        }

    @staticmethod
    def _repair_program(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "files": [{"path": path, "content": content}
                      for path, content in SCRIPTED_PROGRAM_FILES.items()],
            "entry": "main.py",
            "summary": "按审核意见修复后的模型程序（测试替身）",
            "param_space": dict(SCRIPTED_PROGRAM_PARAM_SPACE),
            "packages": [],
            "expected_effect": "修复后程序应能通过复审并参与寻优回测。",
            "risks": ["历史回测不代表未来表现"],
        }

    def chat_text(self, system: str, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"] if messages else ""
        payload = self._parse(content)
        if "facts" in payload:
            return self._explain(payload["facts"], payload.get("question", ""))
        return str(payload.get("text", "我在，有什么可以帮你？"))

    def chat_text_stream(self, system, messages):
        text = self.chat_text(system, messages)
        for start in range(0, len(text), 6):
            yield text[start:start + 6]

    def _plan_for_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for message in reversed(messages):
            if str(message.get("role")) != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                parsed = self._parse(content)
                if parsed:
                    payload = parsed
                    break
        return self._plan(payload)

    @staticmethod
    def _calls_from_plan(plan: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        delta = plan.get("profile_delta") if isinstance(plan.get("profile_delta"), dict) else {}
        if delta:
            calls.append(ToolCall(id="call_update_profile", name="update_profile",
                                  arguments={"slots": delta},
                                  raw_arguments=json.dumps({"slots": delta}, ensure_ascii=False)))
        for index, action in enumerate(plan.get("actions") or []):
            if not isinstance(action, dict):
                continue
            name = str(action.get("name") or "").strip()
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            calls.append(ToolCall(
                id=f"call_{index + 1}", name=name, arguments=args,
                raw_arguments=json.dumps(args, ensure_ascii=False)))
        return calls

    @staticmethod
    def _wire_message(reply: str, calls: list[ToolCall]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply}
        if calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.name,
                              "arguments": call.raw_arguments or "{}"}}
                for call in calls
            ]
        return message

    def chat_tools(self, system: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]],
                   tool_choice: str = "auto") -> ToolChatResult:
        """确定性测试替身：把 ``_plan`` 的规则结果包装成原生 tool_calls。"""
        plan = self._plan_for_messages(messages)
        reply = str(plan.get("reply") or "")
        calls = self._calls_from_plan(plan)
        return ToolChatResult(content=reply, tool_calls=calls,
                              message=self._wire_message(reply, calls))

    def chat_tools_stream(self, system: str, messages: list[dict[str, Any]],
                          tools: list[dict[str, Any]],
                          tool_choice: str = "auto"):
        plan = self._plan_for_messages(messages)
        reply = str(plan.get("reply") or "")
        for start in range(0, len(reply), 6):
            yield "content", reply[start:start + 6]
        calls = self._calls_from_plan(plan)
        yield "result", ToolChatResult(content=reply, tool_calls=calls,
                                       message=self._wire_message(reply, calls))

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
        return any(word in text for word in (
            "同意", "启用", "应用", "换上", "就这样", "可以", "就按", "最新", "用这个"))

    @staticmethod
    def _wants_optimize(text: str) -> bool:
        return any(word in text for word in ("优化", "调参", "改进", "效果不好", "模型", "提升", "调整"))

    @staticmethod
    def _wants_rollback(text: str) -> bool:
        return any(word in text for word in ("回滚", "退回", "恢复上一个", "换回之前"))

    @staticmethod
    def _candidate_choice(text: str) -> int:
        """测试替身：把“第一个/第二个/第 3 个”映射成候选方案编号。"""
        for token, index in (
                ("第一", 1), ("第 1", 1), ("第1", 1),
                ("第二", 2), ("第 2", 2), ("第2", 2),
                ("第三", 3), ("第 3", 3), ("第3", 3),
                ("第四", 4), ("第 4", 4), ("第4", 4)):
            if token in text:
                return index
        return 0

    @staticmethod
    def _is_question(text: str) -> bool:
        return any(mark in text for mark in ("为什么", "风险", "怎么看", "如何", "怎么", "?", "？", "吗"))

    # ------------------------------------------------------------------
    # Planner（测试替身）
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_intent(message: str) -> dict[str, Any]:
        """ScriptedLLM 的确定性意图映射（仅测试/结构演示）。"""
        text = str(message or "")
        lower = text.lower()
        args: dict[str, Any] = {"engine": "auto", "model_type": "",
                                "objective": "", "escalate": False, "reason": "fake-llm"}
        if any(word in text for word in (
                "继续优化", "未达到我的预期", "未达预期", "再试一版", "继续调",
                "换个方向", "更激进")):
            args.update({"engine": "auto", "escalate": True})
            return args
        if any(word in text for word in (
                "数学模型", "数学建模", "打分模型", "打分引擎", "投资模型", "选股模型",
                "模型公式", "新的模型", "新模型", "自建模型", "自己建模",
                "构建模型", "新建模型", "创建模型", "设计模型", "搭建模型",
                "写一个模型", "写个模型", "换一套模型", "更换模型", "换模型",
                "重写模型", "重写打分")):
            args.update({"engine": "code"})
        elif any(word in lower for word in (
                "svm", "支持向量机", "随机森林", "randomforest", "xgboost",
                "lightgbm", "梯度提升", "gbdt", "神经网络", "机器学习",
                "sklearn", "kmeans", "聚类", "pca", "降维", "lasso", "ridge")):
            args.update({"engine": "code"})
        else:
            aliases = {
                "multi_factor": ("均衡多因子", "均衡配置"),
                "low_vol_quality": ("低波质量", "低波稳健"),
                "momentum_growth": ("趋势成长", "动量成长"),
                "value_dividend": ("价值红利", "红利策略"),
                "contrarian": ("反转策略", "反转模型", "反转"),
            }
            for model_id, words in aliases.items():
                if any(word in text for word in words):
                    args.update({"engine": "model_type", "model_type": model_id})
                    break
        if any(word in text for word in ("收益", "回报", "赚钱", "涨幅")):
            args["objective"] = "cagr"
        elif any(word in text for word in ("回撤", "风险", "波动")):
            args["objective"] = "mdd"
        return args

    @staticmethod
    def _program_action(analyze_intent: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": "write_model_program",
            "args": {
                "files": [{"path": path, "content": content}
                          for path, content in SCRIPTED_PROGRAM_FILES.items()],
                "entry": "main.py",
                "summary": "测试替身生成的多文件模型程序：加权打分 + 低波倾斜 + 选股钩子。",
                "param_space": dict(SCRIPTED_PROGRAM_PARAM_SPACE),
                "objective": analyze_intent.get("objective") or "",
                "packages": [],
                "escalate": bool(analyze_intent.get("escalate")),
                "reason": "fake-llm-program",
                "expected_effect": "在低波因子上做小幅倾斜，预期回撤控制更稳。",
                "risks": ["历史回测不代表未来表现"],
            },
        }

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
        candidates = state.get("pending_candidates")
        candidates = candidates if isinstance(candidates, list) else []
        pending_intent = state.get("pending_intent")
        choice = self._candidate_choice(message)
        analyze_intent = self._analyze_intent(message)
        # 结构化升级统一走“AI 直接重写程序”；普通优化仍交给 analyze_model（AI 定范围）
        program_needed = (analyze_intent["engine"] == "code"
                          or bool(analyze_intent["escalate"]))
        actions: list[dict[str, Any]] = []

        if pending_intent and not self._is_question(message):
            actions = [{"name": str(pending_intent), "args": {}}]
            reply = ("已记录您的补充信息。我将按此执行模型优化，分析与回测过程会逐步展示，"
                     "方案不会自动启用，需经您确认。")
        elif candidates and choice:
            actions = [{"name": "apply_pending_change", "args": {"candidate": int(choice)}}]
            reply = f"收到。我将启用第 {choice} 个候选方案并切换模型版本，随后给出对比结果。"
        elif self._wants_rollback(message) and state.get("versions"):
            actions = [{"name": "rollback_version", "args": {}}]
            reply = "收到。我将为您切换回上一版本，随后提供当前生效的模型信息。"
        elif (pending or candidates) and self._agree(message):
            actions = [{"name": "apply_pending_change", "args": {}}]
            reply = "收到。我将应用该改进方案并切换模型版本，随后给出对比结果。"
        elif program_needed and profile.get("confirmed"):
            actions = [self._program_action(analyze_intent)]
            reply = ("收到。我会按您的要求直接编写完整的模型程序，先经 LLM 二次审核与安全校验，"
                     "再由服务端寻优回测；确认前不会启用。")
        elif (self._wants_optimize(message)
              or analyze_intent["engine"] != "auto"
              or analyze_intent["escalate"]) and profile.get("confirmed"):
            actions = [{"name": "analyze_model", "args": analyze_intent}]
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
            reply = f"在为您提供适配建议前，需要向您确认一项信息：{question}？"
        elif not profile.get("confirmed"):
            reply = ("根据您目前提供的信息，我已整理出投资画像。\n\n"
                     "需要您确认：以上画像信息是否准确？如无补充，请回复「确认」，我将按该画像为您筛选标的；"
                     "如需调整，请直接告知。")
        elif state.get("last_selection") and self._is_question(message):
            actions = [{"name": "explain_results", "args": {"question": message}}]
            reply = "我将基于本次选股的回测数据说明，并注明数据边界与风险提示。"
        else:
            reply = ("已记录。您可以继续补充投资情况，或指示我执行选股、结果解读或模型优化。")

        plan: dict[str, Any] = {"reply": reply, "actions": actions, "profile_delta": delta}
        return plan

    # ------------------------------------------------------------------
    # 调参方案（AI 只给引擎 + 搜索范围，具体参数由服务端网格寻优）
    # ------------------------------------------------------------------
    @staticmethod
    def _tuning_plan(request: dict[str, Any]) -> dict[str, Any]:
        from backend.models import registry

        directive = request.get("directive") if isinstance(request.get("directive"), dict) else {}
        engine = str(directive.get("engine") or "auto").strip().lower()
        objective = str(directive.get("objective") or "sharpe").strip().lower()
        if objective not in ("sharpe", "cagr", "mdd"):
            objective = "sharpe"
        max_trials = 12

        def _space(weights: dict[str, float]) -> dict[str, dict[str, float]]:
            top = sorted(weights.items(), key=lambda pair: pair[1], reverse=True)[:5]
            space: dict[str, dict[str, float]] = {}
            for name, weight in top:
                low = max(0.0, round(float(weight) - 0.08, 2))
                high = min(0.4, round(float(weight) + 0.08, 2))
                if high - low < 0.04:
                    high = min(0.4, round(low + 0.04, 2))
                space[name] = {"min": low, "max": high, "step": 0.04}
            return space

        if engine == "model_type":
            model_id = str(directive.get("model_type") or "").strip()
            if model_id not in registry.MODEL_TYPES:
                model_id = "low_vol_quality"
            info = registry.get_model(model_id)
            return {
                "approach": "model_type",
                "model_type": model_id,
                "summary": f"按指令切换到{info.get('name')}，并在该风格权重附近做小范围寻优。",
                "why": "客户明确点名该基础模型，服务端将围绕其风格权重验证更优组合。",
                "expected_effect": "风格与客户要求一致，回撤/收益以本次寻优回测为准。",
                "risks": ["切换风格后短期表现可能差异较大", "历史回测不代表未来表现"],
                "weights": dict(info["weights"]),
                "search_space": _space(dict(info["weights"])),
                "objective": objective,
                "max_trials": max_trials,
            }

        if engine == "weights":
            base = dict(request.get("weights") or registry.get_model("multi_factor")["weights"])
            return {
                "approach": "weights",
                "summary": "保留现有打分框架，在 AI 给定的因子权重范围内由服务端寻优。",
                "why": "当前模型表现需要改善，先在不改变引擎结构的前提下搜索更优权重。",
                "expected_effect": "目标指标有望改善，具体以寻优回测为准。",
                "risks": ["历史回测不代表未来表现"],
                "weights": base,
                "search_space": _space(base),
                "objective": objective,
                "max_trials": max_trials,
            }

        code = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "def build_transform(params=None):\n"
            "    options = params or {}\n"
            "    tilt = float(options.get('tilt', 0.5))\n"
            "\n"
            "    def transform(z, weights):\n"
            "        score = pd.Series(0.0, index=z.index, dtype=float)\n"
            "        for factor, weight in weights.items():\n"
            "            column = 'z_' + factor\n"
            "            if column in z.columns:\n"
            "                values = pd.to_numeric(z[column], errors='coerce').fillna(0.0)\n"
            "                score = score + float(weight) * values\n"
            "        extra = pd.to_numeric(z['z_volatility_60'], errors='coerce').fillna(0.0)\n"
            "        return score + tilt * extra\n"
            "\n"
            "    return transform\n"
        )
        return {
            "approach": "code",
            "summary": "让引擎自带一个可调的「稳健倾斜」旋钮，服务端会在 0~0.6 之间自动试出最合适的值。",
            "why": "当前组合波动偏大，低波动因子还有加码空间。",
            "expected_effect": "网格寻优后波动和最大回撤有望下降。",
            "risks": ["低波动倾斜过大时，快速上涨行情可能跑输", "历史回测不代表未来表现"],
            "code": code,
            "param_space": {
                "tilt": {
                    "min": 0,
                    "max": 0.6,
                    "step": 0.2,
                    "type": "float",
                    "default": 0.5,
                }
            },
            "objective": objective,
            "max_trials": max_trials,
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
