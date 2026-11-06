# -*- coding: utf-8 -*-
"""对话式业务服务：所有操作通过聊天流完成，AI 决定动作，服务负责执行与展示。

没有模块页签、没有独立动作接口；一次聊天 = 理解 → 回复 → 执行动作 → 富卡片。
"""
from __future__ import annotations

import difflib
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .agent import (Agent, AgentError, CONFIRM_PROFILE_HINT,
                     asks_clarifying_question, chunk_text, has_confirmation_request, safe_question,
                     strip_confirmation_tail, trim_duplicate_confirmation)
from .config import (
    A_SHARE_DIR,
    BACKTEST_YEARS,
    COST_BUY,
    COST_SELL,
    DB_PATH,
    DEFAULT_TOP_N,
    REBALANCE_DAYS,
    RISK_FREE_RATE,
    SOURCE,
)
from .data import DataBundle, load_bundle
from .llm import LLMClient, build_llm
from .schemas import (
    FACTOR_PLAIN,
    SLOT_LABELS,
    ValidationError,
    merge_slots,
    profile_complete,
    profile_payload,
    profile_summary,
    top_reasons,
)
from .store import Store

METRIC_NOTES = {
    "cagr": "过去几年平均每年涨多少",
    "sharpe": "每承担一份波动，换来多少收益（越高越划算）",
    "mdd": "最难受的时候，从高点跌了多少",
    "ir": "相对大盘表现稳不稳",
    "win_rate": "有多少个月是赚钱的",
}

WORK_STEP_TEMPLATES = {
    "analysis": "看当前表现",
    "data": "翻数据找线索",
    "plan": "提出调参方案",
    "tune": "网格寻优",
    "code": "修改模型引擎",
    "check": "检查代码安不安全",
    "backtest": "拿候选方案回测",
    "compare": "和现在比一比",
}

TUNING_NOTE = ("网格寻优基于历史回测逐组评估，最优参数可能存在过拟合；"
               "历史业绩不代表未来实际表现，候选方案需要您确认后才会启用。")


MODEL_REQUEST_ALIASES: dict[str, tuple[str, ...]] = {
    "low_vol_quality": ("低波质量", "低波稳健"),
    "multi_factor": ("均衡多因子", "均衡配置"),
    "momentum_growth": ("趋势成长", "动量成长"),
    "value_dividend": ("价值红利", "红利策略"),
    "contrarian": ("反转策略", "反转模型"),
}


CUSTOM_MODEL_KEYWORDS = (
    "svm", "支持向量机", "随机森林", "randomforest", "xgboost", "lightgbm",
    "梯度提升", "gbdt", "神经网络", "机器学习", "sklearn", "kmeans", "pca",
    "聚类", "降维", "孤立森林", "isolationforest", "lasso", "ridge",
)


class ServiceError(RuntimeError):
    """业务错误（对话流里以 error 事件返回）。"""


class ConflictError(ServiceError):
    """状态不允许执行该动作。"""


class NotFoundError(ServiceError):
    """资源不存在。"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        return round(value, 6)
    return value


def _downsample(series: pd.Series, points: int = 120) -> list[dict[str, Any]]:
    series = series.dropna()
    if len(series) == 0:
        return []
    if len(series) > points:
        series = series.iloc[::max(1, len(series) // points)]
    base = float(series.iloc[0]) or 1.0
    return [{"date": idx.strftime("%Y-%m-%d"), "value": round(float(value) / base, 6)}
            for idx, value in series.items()]


class AppService:
    def __init__(self, *, data_dir: Path | None = None, db_path: Path | None = None,
                 llm: LLMClient | None = None, bundle: DataBundle | None = None) -> None:
        self.data_dir = Path(data_dir or A_SHARE_DIR)
        self.store = Store(db_path or DB_PATH)
        self.data = bundle if bundle is not None else load_bundle(self.data_dir)
        self.llm = llm or build_llm()
        self.agent = Agent(self.llm)

    # ------------------------------------------------------------------
    # 基础
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": self.data.meta.get("source", SOURCE),
            "as_of": self.data.as_of,
            "universe": len(self.data.universe),
            "llm": type(self.llm).__name__,
        }

    def create_session(self) -> dict[str, Any]:
        session_id = self.store.create_session()
        return {"session_id": session_id}

    def list_sessions(self) -> dict[str, Any]:
        sessions = []
        for item in self.store.list_sessions():
            title = str(item.get("title") or "").strip().replace("\n", " ")
            if len(title) > 28:
                title = title[:28] + "…"
            sessions.append({
                "id": item["id"],
                "title": title or "新对话",
                "created_at": item["created_at"],
                "updated_at": item.get("last_at") or item["created_at"],
                "message_count": int(item.get("message_count") or 0),
                "profile_summary": str(item.get("profile_summary") or ""),
                "profile_confirmed": bool(item.get("profile_confirmed")),
            })
        return {"sessions": sessions, "current": None}

    def messages(self, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        return {"messages": self.store.get_messages(session_id)}

    def versions(self, session_id: str | None = None) -> dict[str, Any]:
        """模型版本清单：带 session 时只返回该会话的版本链。"""
        if not session_id:
            return {"versions": [self._version_public(item) for item in self.store.list_versions()],
                    "active_id": None, "model_scope": "session"}
        self._require_session(session_id)
        runtime = self.store.get_state(session_id)
        active_id = runtime.get("active_version_id")
        return {
            "versions": [self._version_public(item, active_id)
                         for item in self._session_versions(session_id)],
            "active_id": active_id,
            "model_scope": "session",
        }

    def get_state(self, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        profile = self._profile(session_id)
        runtime = self.store.get_state(session_id)
        model = self._session_active_model(session_id)
        active_version_id = runtime.get("active_version_id")
        versions = self._session_versions(session_id)
        return {
            "profile": profile_payload(profile["slots"] if profile else {},
                                       profile["confirmed"] if profile else False),
            "active_model": self._model_public(
                model, version_id=active_version_id,
                metrics=runtime.get("active_metrics") or {}),
            "versions": [self._version_public(item, active_version_id) for item in versions],
            "pending_proposal": self._pending_public(runtime.get("pending")),
            "last_selection": runtime.get("last_selection"),
            "model_scope": "session",
            "default_model": self._model_public(self._default_model()),
        }

    # ------------------------------------------------------------------
    # 对话主流程
    # ------------------------------------------------------------------
    def chat_stream(self, session_id: str, content: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self._require_session(session_id)
        content = (content or "").strip()
        if not content:
            raise ValidationError("说点什么吧")
        self.store.add_message(session_id, "user", content)
        history = self.store.get_messages(session_id)
        public_state = self._planner_state(session_id)

        try:
            plan = self.agent.plan(public_state, history[:-1], content)
        except Exception as exc:  # noqa: BLE001
            yield "error", {"message": str(exc), "retryable": True}
            yield "done", {"message_id": uuid.uuid4().hex[:8]}
            return

        # 同一待执行动作最多澄清一次：客户已回答后直接执行，避免无限追问
        pending_intent = public_state.get("pending_intent")
        if pending_intent and plan.deferred_intent:
            plan.reply = ("已收到您的确认，现按该方向执行，分析与回测过程将逐步展示。")
            plan.actions = [{"name": str(pending_intent), "args": {}}]
            plan.deferred_intent = ""

        # 审批卡「继续优化」是产品级明确指令：若模型误判成回滚/一般提问，服务端纠正为
        # analyze_model（结构性升级），避免演示时“继续优化”被反问或走错动作。
        if self._wants_aggressive_retry(content) and not plan.has_action("analyze_model"):
            if "apply_pending_change" not in plan.action_names():
                plan.actions = [{"name": "analyze_model", "args": {}}]
                plan.deferred_intent = ""
                if any(word in plan.reply for word in ("回滚", "上一版", "是否", "确认")):
                    plan.reply = ("收到。我会在现有模型基础上做结构性升级：更换基础引擎或重写数学打分逻辑，"
                                  "再由服务端在新的参数范围内重新寻优并回测；候选方案不会自动启用。")

        # 澄清与执行分离：提问轮只记录待执行意图与客户原始诉求，等客户回答后再执行
        runtime_before = self.store.get_state(session_id)
        deferred_request = str(runtime_before.get("deferred_request") or "")
        request_text = content
        if plan.deferred_intent:
            self._save_runtime(session_id, pending_intent=plan.deferred_intent,
                               deferred_request=content)
        elif plan.actions:
            if deferred_request:
                request_text = deferred_request + "；" + content
            self._save_runtime(session_id, pending_intent=None, deferred_request=None)

        saved_blocks: dict[str, dict[str, Any]] = {}

        # 1) 先更新画像，让客户马上看到
        profile = self._profile(session_id)
        slots = profile["slots"] if profile else {}
        confirmed = profile["confirmed"] if profile else False
        if plan.profile_delta:
            merged = merge_slots(slots, plan.profile_delta)
            if merged != slots:
                slots = merged
                self.store.save_profile(session_id, slots, profile_summary(slots), confirmed)
                profile_block = self._profile_block(session_id)
                saved_blocks[profile_block["id"]] = profile_block
                yield "ui", {"block": profile_block}

        # 画像齐全等待确认时，用确定的专业总结覆盖自相矛盾的追问，保证确认要求唯一且明确
        updated_profile = self.store.get_profile(session_id)
        if (updated_profile and not updated_profile["confirmed"]
                and profile_complete(updated_profile["slots"])
                and not plan.has_action("confirm_profile")):
            if not has_confirmation_request(plan.reply):
                plan.reply = ("我已根据你提供的信息整理出投资画像："
                              + profile_summary(updated_profile["slots"])
                              + "\n\n" + CONFIRM_PROFILE_HINT)

        # 1.5) 动作前置校验：没有 pending 时不能让模型直接假设已审批，
        #       自动转为重新分析，避免“回复说已启用、随后报错”的不一致。
        runtime_pre = self.store.get_state(session_id)
        action_names = plan.action_names()
        fallback_analyze = False
        if "apply_pending_change" in action_names and not runtime_pre.get("pending"):
            plan.actions = [item for item in plan.actions
                            if str(item.get("name")) != "apply_pending_change"]
            if not any(str(item.get("name")) == "analyze_model" for item in plan.actions):
                plan.actions.append({"name": "analyze_model", "args": {}})
            plan.reply = ("当前没有等待您确认的模型改进方案。我会按最新模型重新分析，"
                          "生成候选方案并完成回测，结果仍需要您确认后才会启用。")
            fallback_analyze = True
        if "rollback_version" in action_names and not self._can_rollback(session_id):
            plan.actions = [item for item in plan.actions
                            if str(item.get("name")) != "rollback_version"]
            if not plan.actions:
                plan.reply = ("当前会话没有可以回滚的历史模型版本。您可以先让我分析当前模型，"
                              "生成新的改进方案。")
        if (not plan.actions and not runtime_pre.get("pending")
                and "模型改进方案" in plan.reply and "同意启用" in plan.reply):
            plan.actions = [{"name": "analyze_model", "args": {}}]
            plan.reply = ("我先对当前模型做一次完整分析与回测，生成候选方案；"
                          "方案不会自动启用，仍需您确认。")
            fallback_analyze = True
        if fallback_analyze:
            request_text = "根据当前对话重新分析模型并生成候选方案"

        # 2) 流式回复（去掉模型自行重复的确认段落）
        plan.reply = trim_duplicate_confirmation(plan.reply)
        notes: list[str] = []
        try:
            for piece in chunk_text(plan.reply):
                yield "token", {"text": piece}
            for action in plan.actions:
                events, note, should_stop = yield from self._run_action(
                    session_id, action, request_text)
                for block in events or []:
                    if isinstance(block, dict) and block.get("id"):
                        saved_blocks[str(block["id"])] = block
                if note:
                    if has_confirmation_request(note) and (
                            has_confirmation_request(plan.reply)
                            or asks_clarifying_question(plan.reply)):
                        note = strip_confirmation_tail(note).strip()
                    if note:
                        notes.append(note)
                        yield "token", {"text": "\n\n"}
                        for piece in chunk_text(note):
                            yield "token", {"text": piece}
                if should_stop:
                    break
        except (ServiceError, AgentError, ValidationError) as exc:
            self._remember_assistant(session_id, plan.reply, notes, list(saved_blocks.values()))
            yield "error", {"message": str(exc), "retryable": False}
            yield "done", {"message_id": uuid.uuid4().hex[:8]}
            return
        except Exception as exc:  # noqa: BLE001
            self._remember_assistant(session_id, plan.reply, notes, list(saved_blocks.values()))
            yield "error", {"message": f"处理这句时出了点问题：{exc}", "retryable": True}
            yield "done", {"message_id": uuid.uuid4().hex[:8]}
            return

        self._remember_assistant(session_id, plan.reply, notes, list(saved_blocks.values()))
        yield "done", {"message_id": uuid.uuid4().hex[:8]}

    def _run_action(self, session_id: str, action: dict[str, Any], request_text: str = ""):
        name = action.get("name")
        if name == "confirm_profile":
            return (yield from self._confirm_profile(session_id))
        if name == "run_selection":
            return (yield from self._run_selection(session_id))
        if name == "explain_results":
            question = str(action.get("args", {}).get("question", ""))
            return (yield from self._explain_results(session_id, question))
        if name == "analyze_model":
            return (yield from self._analyze_model(session_id, request_text))
        if name == "apply_pending_change":
            return (yield from self._apply_pending(session_id))
        if name == "rollback_version":
            return (yield from self._rollback(session_id))
        return [], "", False

    # ------------------------------------------------------------------
    # 动作：画像确认
    # ------------------------------------------------------------------
    def _confirm_profile(self, session_id: str):
        profile = self._profile(session_id)
        slots = profile["slots"] if profile else {}
        if not profile_complete(slots):
            missing = [SLOT_LABELS[key] for key in ("goal", "horizon", "drawdown_tolerance",
                                                     "loss_reaction") if not slots.get(key)]
            raise ConflictError("画像还差一点信息：" + "、".join(missing))
        self.store.save_profile(session_id, slots, profile_summary(slots), True)
        block = self._profile_block(session_id)
        yield "ui", {"block": block}
        return [block], "画像已经确认。", False

    # ------------------------------------------------------------------
    # 动作：选股
    # ------------------------------------------------------------------
    def _run_selection(self, session_id: str):
        profile = self._profile(session_id)
        if not profile or not profile["confirmed"]:
            raise ConflictError("画像还没确认，先和我说清楚你的情况吧")
        model = self._session_active_model(session_id)
        weights, transform = self._model_engine(model)
        from .engine import selection

        top_df, _scores = selection.screen(self.data, weights, top_n=DEFAULT_TOP_N,
                                           score_transform=transform)
        portfolio = selection.build_portfolio(top_df, self.data.prices,
                                              method="equal_weight", capital=1_000_000.0)
        start, end = self._backtest_window()
        backtest = selection.run_backtest(
            self.data, weights, start=start, end=end, top_n=DEFAULT_TOP_N,
            rebalance_days=REBALANCE_DAYS, cost_buy=COST_BUY, cost_sell=COST_SELL,
            risk_free_rate=RISK_FREE_RATE, score_transform=transform)

        contributions: dict[str, float] = {}
        top_payload: list[dict[str, Any]] = []
        for _, row in top_df.iterrows():
            stock_contributions = {
                name: round(float(row.get(f"z_{name}", 0.0)) * float(weights.get(name, 0.0)), 4)
                for name in weights
            }
            for name, value in stock_contributions.items():
                contributions[name] = contributions.get(name, 0.0) + value
            top_payload.append({
                "code": str(row["code"]),
                "industry": str(row["industry"]),
                "score": round(float(row["score"]), 3),
                "reasons": top_reasons(stock_contributions),
            })
        factor_chart = [
            {"name": name, "label": FACTOR_PLAIN.get(name, name),
             "value": round(value / max(1, len(top_payload)), 4)}
            for name, value in sorted(contributions.items(), key=lambda pair: pair[1], reverse=True)
        ]
        facts = {"as_of": self.data.as_of, "top": top_payload[:5],
                 "metrics": backtest["metrics"],
                 "model": model.get("name", "均衡多因子")}
        explanation = self.agent.explain(facts)
        block = {
            "id": "selection-" + uuid.uuid4().hex[:8],
            "type": "selection",
            "as_of": self.data.as_of,
            "model": {"name": model.get("name", "均衡多因子"),
                      "description": model.get("description", "")},
            "top": top_payload,
            "metrics": _json_safe(backtest["metrics"]),
            "metric_notes": METRIC_NOTES,
            "portfolio": _json_safe(portfolio.to_dict("records")),
            "factor_contributions": factor_chart,
            "nav": _downsample(backtest["nav"]),
            "benchmark_nav": _downsample(backtest["benchmark_nav"]),
            "explanation": explanation,
            "weights": _json_safe(weights),
        }
        self._save_runtime(session_id, last_selection=block)
        yield "ui", {"block": block}
        return [block], "已经选好 10 只股票，理由和回测结果都在上面的卡片里。", False

    # ------------------------------------------------------------------
    # 动作：解释追问
    # ------------------------------------------------------------------
    def _explain_results(self, session_id: str, question: str):
        runtime = self.store.get_state(session_id)
        block = runtime.get("last_selection")
        if not block:
            raise ConflictError("还没有选股结果，先让我帮你选一次吧")
        facts = {"as_of": block.get("as_of"), "top": block.get("top", [])[:5],
                 "metrics": block.get("metrics") or {},
                 "model": (block.get("model") or {}).get("name")}
        text = self.agent.explain(facts, safe_question(question))
        for piece in chunk_text(text):
            yield "token", {"text": piece}
        return [], "", False

    # ------------------------------------------------------------------
    # 动作：AI 定引擎与调参范围 + 服务端执行网格寻优
    # ------------------------------------------------------------------
    def _analyze_model(self, session_id: str, request_text: str = ""):
        """AI 只负责写引擎、定搜索范围；具体参数由服务端网格寻优选出。"""
        from .engine import selection, tuning
        from .models import custom, pipeline, registry

        model = self._session_active_model(session_id)
        base_weights, base_transform = self._model_engine(model)
        base_metrics = self._current_metrics(session_id)
        expectation = self._expectation(request_text, session_id)
        objective = expectation.get("objective") or "sharpe"
        objective_label = tuning.OBJECTIVE_LABELS.get(objective, objective)
        tolerance = expectation.get("tolerance")

        runtime_before = self.store.get_state(session_id)
        previous_pending = runtime_before.get("pending") or {}
        previous_approach = self._previous_approach(previous_pending)
        aggressive = self._wants_aggressive_retry(request_text, previous_pending)
        requested_model = self._requested_model_type(request_text)
        custom_model = self._requested_custom_model(request_text)
        force_approach = ""
        force_model_type = ""
        if custom_model or self._wants_code_change(request_text):
            force_approach = "code"
        elif requested_model:
            force_approach = "model_type"
            force_model_type = requested_model
        if aggressive:
            if previous_approach == "weights":
                force_approach = "model_type"
                force_model_type = self._different_builtin_model(
                    model, previous_pending.get("model") or {})
            else:
                force_approach = "code"
                force_model_type = ""

        work_id = "work-" + uuid.uuid4().hex[:8]
        tune_id = "tune-" + uuid.uuid4().hex[:8]
        steps: list[dict[str, Any]] = []
        tuning_state: dict[str, Any] = {"start": None, "history": [], "best": None,
                                        "started": time.time()}

        def publish(status: str = "running", comparison: dict | None = None,
                    proposal: dict | None = None, extra: dict | None = None) -> dict[str, Any]:
            block = {
                "id": work_id, "type": "work", "title": "AI 正在优化模型",
                "status": status, "steps": list(steps),
                "comparison": comparison or {},
                "proposal": proposal or {},
            }
            if extra:
                block.update(extra)
            return {"block": block}

        def add_step(title: str, detail: str, status: str, kind: str = "analysis") -> None:
            steps.append({"title": title, "detail": detail, "status": status, "kind": kind})

        def best_public(best: dict[str, Any] | None, engine_kind: str) -> dict[str, Any]:
            if not isinstance(best, dict) or not best:
                return {}
            params = _json_safe(best.get("params") or {})
            metric_value = best.get("metric_value")
            payload = {
                "trial": best.get("trial"),
                "label": f"第 {best.get('trial')} 组",
                # 展示真实目标指标（如夏普 0.16）；排序分可能因回撤超限被惩罚，
                # 单独放在 rank_score，避免前端把惩罚分误当成模型指标。
                "score": metric_value if metric_value is not None else best.get("score"),
                "metric_value": metric_value,
                "rank_score": best.get("score"),
                "metrics": _json_safe(best.get("metrics") or {}),
                "params": params,
                "constraint_ok": best.get("constraint_ok", True),
            }
            if engine_kind == "weight":
                payload["weights"] = params
            return payload

        def build_tuning_block(status: str, payload: dict[str, Any] | None = None,
                               progress: dict[str, Any] | None = None,
                               current: dict[str, Any] | None = None,
                               best: dict[str, Any] | None = None,
                               history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
            meta = payload or tuning_state.get("start") or {}
            return {
                "id": tune_id,
                "type": "tuning",
                "title": "参数网格寻优",
                "status": status,
                "method": meta.get("method") or "网格寻优",
                "objective": meta.get("objective") or objective,
                "objective_label": meta.get("objective_label") or objective_label,
                "space": meta.get("space") or [],
                "progress": progress or {"trial": 0,
                                         "total": meta.get("total") or 0,
                                         "percent": 0, "elapsed": 0},
                "current": current or {},
                "best": best or {},
                "history": history or [],
                "note": TUNING_NOTE,
            }

        base_annual = float(base_metrics.get("cagr", 0)) * 100
        add_step("看当前表现",
                 f"近两年年化 {base_annual:.1f}%，"
                 f"最大回撤 {abs(float(base_metrics.get('mdd', 0))) * 100:.1f}%，"
                 f"夏普 {float(base_metrics.get('sharpe', 0)):.2f}。", "done", "analysis")
        yield "ui", publish()

        top_df, scores = selection.screen(self.data, base_weights, top_n=DEFAULT_TOP_N,
                                          score_transform=base_transform)
        focus = self._factor_focus(top_df, base_weights)
        add_step("翻数据找线索", f"最近组合更看重：{'、'.join(focus[:3]) or '均衡配置'}；"
                                 "先把偏弱的环节找出来。", "done", "data")
        factor_stats = self._factor_stats(scores, base_weights)
        recent_metrics = self._recent_metrics(model)
        yield "ui", publish()

        # 1) AI 只产出「引擎 + 搜索范围」，不直接给参数值
        try:
            plan = pipeline.build_tuning_plan(self.llm, self._tuning_state(
                model, base_metrics, factor_stats, recent_metrics, request_text,
                force_approach=force_approach, force_model_type=force_model_type,
                custom_model=custom_model, aggressive=aggressive,
                previous=previous_pending, objective=objective, tolerance=tolerance))
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"AI 没能给出调参方案：{exc}") from exc
        if isinstance(plan, dict) and isinstance(plan.get("plan"), dict):
            plan = plan["plan"]
        if not isinstance(plan, dict) or not plan.get("approach"):
            raise AgentError("AI 返回的调参方案格式不对")
        if aggressive:
            add_step("升级优化策略",
                     ("上一版未达预期，本轮不再微调，改为直接更换基础模型并由服务端重新寻优。"
                      if force_approach == "model_type" else
                      "上一版未达预期，本轮不再微调，改由 AI 重写数学打分引擎，再由服务端重新寻优。"),
                     "done", "plan")
        repair_attempts = int(plan.get("repair_attempts") or 0)
        if repair_attempts:
            add_step("自动修复调参方案",
                     f"AI 首次给出的引擎/范围未通过校验，已根据错误自动修复 {repair_attempts} 次。",
                     "done", "check")
        add_step("提出调参方案", self._plan_detail(plan), "done", "plan")
        yield "ui", publish(proposal=self._proposal_public(plan))

        # 2) 服务端在 AI 给定范围内执行逐轴网格寻优
        engine_kind = "param" if plan["approach"] == "code" else "weight"
        engine_weights = normalize_weights(plan.get("weights") or base_weights)
        code = str(plan.get("code") or "")
        if engine_kind == "weight":
            search_space = dict(plan.get("search_space") or {})
            labels = {name: FACTOR_PLAIN.get(name, name) for name in search_space}
        else:
            search_space = dict(plan.get("param_space") or {})
            labels = {name: name for name in search_space}

        start_date, end_date = self._backtest_window()
        rebalance_days = REBALANCE_DAYS
        cost_buy = COST_BUY
        cost_sell = COST_SELL
        risk_free_rate = RISK_FREE_RATE

        def evaluator(params: dict[str, Any]) -> dict[str, Any]:
            if engine_kind == "weight":
                tuned_weights = normalize_weights(params)
                backtest = selection.run_backtest(
                    self.data, tuned_weights, start=start_date, end=end_date,
                    top_n=DEFAULT_TOP_N, rebalance_days=rebalance_days,
                    cost_buy=cost_buy, cost_sell=cost_sell,
                    risk_free_rate=risk_free_rate)
                return {"metrics": backtest["metrics"],
                        "params": tuned_weights, "nav": backtest["nav"]}
            trial_transform = custom.compile_code(code, params, validate=False)  # type: ignore[attr-defined]
            backtest = selection.run_backtest(
                self.data, engine_weights, start=start_date, end=end_date,
                top_n=DEFAULT_TOP_N, rebalance_days=rebalance_days,
                cost_buy=cost_buy, cost_sell=cost_sell,
                risk_free_rate=risk_free_rate, score_transform=trial_transform)
            return {"metrics": backtest["metrics"],
                    "params": dict(params), "nav": backtest["nav"]}

        step_title = "运行参数网格寻优" if engine_kind == "param" else "运行因子网格寻优"
        add_step(step_title,
                 (f"AI 已给出 {len(search_space)} 个参数的搜索范围，服务端逐个组合回测评估，"
                  f"目标：{objective_label}。具体进度实时显示在寻优卡片中。"),
                 "running", "tune")
        yield "ui", publish()

        final_event: dict[str, Any] | None = None
        try:
            for event in tuning.run_search(
                    evaluator,
                    space=search_space,
                    base_params=engine_weights if engine_kind == "weight" else {},
                    objective=objective,
                    max_trials=plan.get("max_trials"),
                    tolerance=tolerance,
                    kind=engine_kind,
                    labels=labels):
                event_name = event.get("event")
                if event_name == "start":
                    tuning_state["start"] = event
                    yield "ui", {"block": build_tuning_block(
                        "running",
                        progress={"trial": 0, "total": event.get("total") or 0,
                                  "percent": 0, "elapsed": 0})}
                    continue
                if event_name == "trial":
                    total = event.get("total") or 1
                    elapsed = round(time.time() - tuning_state["started"], 1)
                    percent = int(round(float(event.get("progress") or 0) * 100))
                    progress = {"trial": event.get("trial", 0), "total": total,
                                "percent": min(99, percent), "elapsed": elapsed}
                    if event.get("error"):
                        current = {"trial": event.get("trial"),
                                   "label": event.get("label") or "该组合回测失败",
                                   "error": str(event.get("error") or "")[:200]}
                        yield "ui", {"block": build_tuning_block(
                            "running", progress=progress, current=current,
                            best=best_public(tuning_state.get("best"), engine_kind),
                            history=list(tuning_state["history"]))}
                        continue
                    tuning_state["history"].append(
                        {"trial": event.get("trial"),
                         "score": event.get("metric_value", event.get("score")) or 0})
                    tuning_state["best"] = event.get("best") or tuning_state.get("best")
                    current = {"trial": event.get("trial"), "label": event.get("label"),
                               "score": event.get("metric_value", event.get("score")),
                               "rank_score": event.get("score"),
                               "metrics": _json_safe(event.get("metrics") or {}),
                               "params": _json_safe(event.get("params") or {})}
                    best_payload = best_public(tuning_state.get("best"), engine_kind)
                    best_metric = float((tuning_state.get("best") or {}).get("metric_value")
                                        or 0)
                    steps[-1] = dict(
                        steps[-1],
                        detail=(f"已评估 {event.get('trial')}/{total} 组，"
                                f"当前最优{objective_label} {best_metric:.2f}"
                                + ("（本组更优）" if event.get("improved") else "") + "。"))
                    yield "ui", {"block": build_tuning_block(
                        "running", progress=progress, current=current,
                        best=best_payload, history=list(tuning_state["history"]))}
                    yield "ui", publish()
                    continue
                if event_name == "done":
                    final_event = event
                    break
        except tuning.TuningError as exc:
            raise ServiceError(f"参数网格寻优失败：{exc}") from exc
        if final_event is None:
            raise ServiceError("参数网格寻优没有返回结果")

        final_best = final_event.get("best") or {}
        tuned_params = dict(final_best.get("params") or {})
        candidate_nav = final_best.get("nav")
        candidate_metrics = _json_safe(final_best.get("metrics") or {})
        evaluated = int(final_event.get("evaluated") or final_event.get("total") or 0)
        total_trials = int(final_event.get("total") or evaluated)
        if engine_kind == "weight":
            candidate_weights = normalize_weights(tuned_params)
        else:
            candidate_weights = engine_weights
            try:
                validation = custom.validate_code(code, tuned_params)  # type: ignore[attr-defined]
            except custom.ModelValidationError as exc:  # type: ignore[attr-defined]
                raise AgentError(f"AI 生成的模型引擎用最优参数校验失败：{exc}") from exc
            checks = "、".join(str(item.get("name")) for item in validation.get("checks", []))
        steps[-1] = dict(
            steps[-1], status="done",
            detail=(f"共评估 {evaluated} 组参数，按{objective_label}选出最优一组；"
                    "结果下方寻优卡片可查看范围与得分走势。"))
        final_tune_meta = {
            "method": final_event.get("method"),
            "objective": final_event.get("objective"),
            "objective_label": final_event.get("objective_label"),
            "space": final_event.get("space") or [],
            "total": total_trials,
        }
        tune_public = {
            "method": final_tune_meta["method"],
            "objective": final_tune_meta["objective"],
            "objective_label": final_tune_meta["objective_label"],
            "trials": evaluated,
            "total": total_trials,
            "best_score": ((final_best.get("metric_value") if final_best else None)
                           if final_best and final_best.get("metric_value") is not None
                           else (final_best.get("score") if final_best else None)),
            "best_rank_score": final_best.get("score") if final_best else None,
            "best_constraint_ok": final_best.get("constraint_ok", True) if final_best else True,
            "best_params": _json_safe(tuned_params),
            "space": _json_safe(final_tune_meta["space"]),
            "elapsed": final_event.get("elapsed"),
        }
        final_tuning_block = build_tuning_block(
            "done", payload=final_tune_meta,
            progress={"trial": evaluated, "total": total_trials, "percent": 100,
                      "elapsed": final_event.get("elapsed")},
            current={"trial": evaluated, "label": "寻优完成"},
            best=best_public(final_best, engine_kind),
            history=list(tuning_state["history"]))
        yield "ui", {"block": final_tuning_block}

        # 3) 展示候选方案改动：权重 diff / 模型切换 / 代码改写
        final_plan = dict(plan)
        final_plan["weights"] = candidate_weights
        model_name = self._candidate_model_name(final_plan)
        change_preview = self._change_preview(model, final_plan)
        diff_lines = str(change_preview.get("diff") or "").split("\n")
        if final_plan["approach"] == "code":
            step_title = "写入 AI 重写的模型引擎"
            running_detail = "AI 已重写打分逻辑，代码逐行写入中…"
        elif final_plan["approach"] == "model_type":
            step_title = "切换基础模型并写入寻优权重"
            running_detail = "正在写入模型类型与寻优后的配置，改动逐行展示中…"
        else:
            step_title = "写入寻优后的因子权重"
            running_detail = "正在写入网格寻优得到的因子权重，改动逐行展示中…"
        add_step(step_title, running_detail, "running", "code")
        partial_change = dict(change_preview)
        partial_change["diff"] = ""
        partial_change["added"] = 0
        partial_change["removed"] = 0
        yield "ui", publish(proposal=self._proposal_public(final_plan, model_name=model_name),
                            extra={"code_change": partial_change})
        for index, line in enumerate(diff_lines):
            if not line and index == len(diff_lines) - 1:
                continue
            partial = diff_lines[:index + 1]
            partial_change = dict(change_preview)
            partial_change["diff"] = "\n".join(partial)
            partial_change["added"] = len([item for item in partial
                                           if item.startswith("+") and not item.startswith("+++")])
            partial_change["removed"] = len([item for item in partial
                                             if item.startswith("-") and not item.startswith("---")])
            yield "ui", publish(proposal=self._proposal_public(final_plan, model_name=model_name),
                                extra={"code_change": partial_change})
            time.sleep(0.03)

        if final_plan["approach"] == "code":
            steps[-1] = dict(steps[-1], status="done",
                             detail=f"改写策略打分引擎：新增 {change_preview['added']} 行、"
                                    f"删除 {change_preview['removed']} 行；"
                                    f"通过安全检查（{checks or '导入、危险调用、返回格式'}）"
                                    "及子进程冒烟测试，最优参数已写入。")
        elif final_plan["approach"] == "model_type":
            info = registry.get_model(str(final_plan.get("model_type")))
            steps[-1] = dict(steps[-1], status="done",
                             detail=f"切换基础模型为{info.get('name')}，并按 AI 给出的范围完成"
                                    "网格寻优，具体配置改动见下方改动框。")
        else:
            changed = len([line for line in change_preview["diff"].splitlines()
                           if line.startswith("+") and not line.startswith("+++")])
            steps[-1] = dict(steps[-1], status="done",
                             detail=f"寻优后调整 {changed} 个因子权重，具体改动见下方改动框。")
        yield "ui", publish(proposal=self._proposal_public(final_plan, model_name=model_name),
                            extra={"code_change": change_preview})

        # 4) 候选回测结果、对比与目标核对
        add_step("拿候选方案回测",
                 f"候选策略年化 {float(candidate_metrics.get('cagr', 0)) * 100:.1f}%，"
                 f"最大回撤 {abs(float(candidate_metrics.get('mdd', 0))) * 100:.1f}%。",
                 "done", "backtest")
        comparison = {"current": _json_safe(base_metrics), "candidate": candidate_metrics}
        yield "ui", publish(proposal=self._proposal_public(final_plan, model_name=model_name),
                            comparison=comparison)
        add_step("和现在比一比",
                 f"夏普从 {float(base_metrics.get('sharpe', 0)):.2f} 变为 "
                 f"{float(candidate_metrics.get('sharpe', 0)):.2f}，等你确认后再启用。",
                 "done", "compare")
        expectation_met, expectation_reason = self._meets_expectation(
            expectation, base_metrics, candidate_metrics)
        add_step("核对您的目标", expectation_reason,
                 "done" if expectation_met else "error", "check")
        proposal_public = self._proposal_public(final_plan, model_name=model_name)
        proposal_public["tuning"] = tune_public
        yield "ui", publish(proposal=proposal_public, comparison=comparison)

        candidate_model = self._candidate_model_from_plan(final_plan, tuned_params)
        pending = {
            "id": "pending-" + uuid.uuid4().hex[:8],
            "proposal": proposal_public,
            "model": candidate_model,
            "metrics": candidate_metrics,
            "nav": _downsample(candidate_nav) if candidate_nav is not None else [],
            "base_metrics": _json_safe(base_metrics),
            "tuning": tune_public,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_runtime(session_id, pending=pending)
        final_extra = {"code_change": change_preview, "tuning": tune_public}
        final_work = publish(status="done", proposal=proposal_public,
                             comparison=comparison, extra=final_extra)
        yield "ui", final_work

        approval_actions = [
            {"label": "同意", "message": "同意启用这个改进"},
            {"label": "暂存", "message": "先别启用，我再想想"},
        ]
        if not expectation_met:
            approval_actions.append(
                {"label": "继续优化", "message": "继续优化模型，上一版未达到我的预期"})
        approval = {
            "id": "approval-" + uuid.uuid4().hex[:8],
            "type": "approval",
            "title": "是否启用这个网格寻优结果？" if expectation_met else "候选未达预期，是否仍要启用？",
            "summary": (f"服务端网格寻优（{tune_public.get('method', '逐轴网格搜索')}）共评估 "
                        f"{evaluated} 组参数，按{objective_label}"
                        + ("并在回撤上限约束下" if tolerance is not None else "")
                        + f"选出最优一组；回测夏普 "
                        f"{float(base_metrics.get('sharpe', 0)):.2f} → "
                        f"{float(candidate_metrics.get('sharpe', 0)):.2f}。"
                        + ("" if expectation_met else f" 目标核对：{expectation_reason}")),
            "risk": ("最优参数由历史回测选出，可能存在过拟合；"
                     + ("；".join(final_plan.get("risks") or []) or "历史回测不代表未来表现。")
                     + ("" if expectation_met
                        else " 注意：候选方案未达到您的目标，建议继续优化或先保留现模型。")),
            "actions": approval_actions,
        }
        yield "ui", {"block": approval}
        note_text = ""
        if not expectation_met:
            note_text = ("候选方案未达到您的目标：" + expectation_reason
                         + " 您可以先不启用，或回复继续优化；我会更换基础引擎或重写打分逻辑，"
                           "再由服务端在新的参数范围内重新寻优。")
        return [final_work["block"], final_tuning_block, approval], note_text, False

    def _tuning_state(self, model: dict[str, Any], base_metrics: dict[str, Any],
                      factor_stats: dict[str, Any], recent_metrics: dict[str, Any],
                      request_text: str, *, force_approach: str = "",
                      force_model_type: str = "", custom_model: str = "",
                      aggressive: bool = False, previous: dict[str, Any] | None = None,
                      objective: str = "sharpe",
                      tolerance: float | None = None) -> dict[str, Any]:
        """构造给 AI 的调参状态：AI 只依据它决定引擎与搜索范围。"""
        previous = previous or {}
        current_model = {
            "name": model.get("name"),
            "kind": model.get("kind"),
            "model_id": model.get("model_id"),
            "description": model.get("description"),
            "weights": model.get("weights") or {},
            "code": model.get("code") or "",
            "params": model.get("params") or {},
        }
        return {
            "tuning_request": True,
            "current_model": current_model,
            "current_metrics": _json_safe(base_metrics),
            "weights": normalize_weights(model.get("weights") or {}),
            "factor_stats": factor_stats,
            "recent_metrics": recent_metrics,
            "user_request": request_text,
            "force_approach": force_approach,
            "force_model_type": force_model_type,
            "requested_custom_model": custom_model,
            "objective": objective,
            "target_drawdown": tolerance,
            "escalation": {
                "active": bool(aggressive),
                "previous_approach": self._previous_approach(previous),
                "previous_model": ((previous.get("model") or {}).get("model_id") or ""),
                "previous_name": ((previous.get("model") or {}).get("name") or ""),
            },
            "previous_proposal": previous.get("proposal") or {},
            "previous_metrics": previous.get("metrics") or {},
            "market": {"as_of": self.data.as_of, "universe": len(self.data.universe)},
        }

    @staticmethod
    def _plan_detail(plan: dict[str, Any]) -> str:
        approach = str(plan.get("approach") or "")
        space = plan.get("search_space") or plan.get("param_space") or {}
        engine = {"weights": "线性因子权重引擎",
                  "model_type": "内置模型引擎",
                  "code": "自定义代码引擎"}.get(approach, "模型引擎")
        return (f"AI 选定{engine}，只负责编写底层逻辑并给出 {len(space)} 个参数的搜索范围；"
                "具体取值由服务端网格寻优逐组回测选出，AI 不直接拍参数。")

    def _candidate_model_name(self, plan: dict[str, Any]) -> str:
        from .models import registry

        approach = str(plan.get("approach") or "")
        if approach == "model_type":
            info = registry.get_model(str(plan.get("model_type")))
            return f"{info.get('name')}（网格寻优）"
        if approach == "code":
            return "AI 生成模型"
        return "网格寻优权重"

    def _candidate_model_from_plan(self, plan: dict[str, Any],
                                   tuned_params: dict[str, Any]) -> dict[str, Any]:
        from .models import registry

        approach = str(plan.get("approach") or "")
        if approach == "code":
            return {
                "kind": "code", "model_id": "ai_generated", "name": "AI 生成模型",
                "description": str(plan.get("summary") or ""),
                "weights": plan.get("weights") or {},
                "code": plan.get("code") or "",
                "params": dict(tuned_params),
                "summary": str(plan.get("summary") or ""),
            }
        if approach == "model_type":
            info = registry.get_model(str(plan.get("model_type")))
            return {
                "kind": "builtin", "model_id": info["id"],
                "name": self._candidate_model_name(plan),
                "description": info.get("description", ""),
                "weights": plan.get("weights") or {},
                "summary": str(plan.get("summary") or ""),
            }
        return {
            "kind": "builtin", "model_id": "ai_tuned_weights",
            "name": "网格寻优权重",
            "description": str(plan.get("summary") or ""),
            "weights": plan.get("weights") or {},
            "summary": str(plan.get("summary") or ""),
        }

    @staticmethod
    def _requested_model_type(request_text: str) -> str:
        text = str(request_text or "").lower().replace(" ", "").replace("-", "")
        if not text:
            return ""
        for model_id, aliases in MODEL_REQUEST_ALIASES.items():
            for alias in aliases:
                if alias.lower().replace(" ", "") in text:
                    return model_id
        return ""

    @staticmethod
    def _requested_custom_model(request_text: str) -> str:
        text = str(request_text or "").lower().replace(" ", "")
        for keyword in CUSTOM_MODEL_KEYWORDS:
            if keyword.lower() in text:
                return keyword
        return ""

    def _target_drawdown(self, request_text: str, session_id: str) -> float | None:
        text = str(request_text or "")
        match = re.search(r"回撤[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)\s*%", text)
        if not match:
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%[。；;\s]{0,4}回撤", text)
        if match:
            try:
                return abs(float(match.group(1))) / 100.0
            except ValueError:
                return None
        profile = self.store.get_profile(session_id) or {}
        value = str((profile.get("slots") or {}).get("drawdown_tolerance") or "")
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", value)
        if match:
            try:
                return abs(float(match.group(1))) / 100.0
            except ValueError:
                return None
        return None

    def _expectation(self, request_text: str, session_id: str) -> dict[str, Any]:
        text = str(request_text or "")
        if any(word in text for word in ("收益", "回报", "赚钱", "涨幅")):
            objective = "cagr"
        elif any(word in text for word in ("回撤", "风险", "波动")):
            objective = "mdd"
        else:
            objective = "sharpe"
        return {"objective": objective,
                "tolerance": self._target_drawdown(request_text, session_id)}

    @staticmethod
    def _meets_expectation(expectation: dict[str, Any], base: dict[str, Any],
                           candidate: dict[str, Any]) -> tuple[bool, str]:
        objective = expectation.get("objective") or "sharpe"
        tolerance = expectation.get("tolerance")
        names = {"cagr": "年化收益", "sharpe": "夏普比率", "mdd": "最大回撤"}
        try:
            base_value = float(base.get(objective) or 0.0)
            cand_value = float(candidate.get(objective) or 0.0)
        except (TypeError, ValueError):
            return False, "回测指标不完整，暂时无法判断是否达到目标。"
        improved = cand_value > base_value + 1e-9
        if objective == "mdd":
            improved = cand_value > base_value + 1e-9
        reason = (f"目标：{names.get(objective, objective)}较当前改善；"
                  f"当前 {base_value:.4f} → 候选 {cand_value:.4f}。")
        met = improved
        if tolerance is not None:
            cand_mdd = abs(float(candidate.get("mdd") or 0.0))
            if cand_mdd > tolerance + 1e-9:
                met = False
                reason += (f"候选最大回撤 {cand_mdd * 100:.1f}% 仍高于您设定的 "
                           f"{tolerance * 100:.0f}% 上限。")
            else:
                reason += f"候选最大回撤 {cand_mdd * 100:.1f}%，在 {tolerance * 100:.0f}% 上限内。"
        elif not improved:
            reason += "候选指标没有优于当前模型。"
        return met, reason


    @classmethod
    def _change_preview(cls, current_model: dict[str, Any],
                        proposal: dict[str, Any]) -> dict[str, Any]:
        """生成模型改动预览：代码方案展示源码 diff，权重/模型类型展示配置 diff。"""
        approach = proposal.get("approach")
        if approach == "code":
            return cls._code_change(current_model, proposal["code"])
        from .engine.factors import FACTOR_NAMES
        from .models import registry

        if approach == "model_type":
            info = registry.get_model(str(proposal.get("model_type")))
            file_name = "models/strategy.py · 模型类型与权重"
            current = (current_model or {})
            current_weights = current.get("weights") or {}
            candidate_weights = proposal.get("weights") or info.get("weights") or {}
            diff_lines = [
                "--- " + file_name + "（当前）",
                "+++ " + file_name + "（候选）",
                "- model_type: " + str(current.get("model_id") or "multi_factor"),
                "+ model_type: " + str(info.get("id")),
                "- name: " + str(current.get("name") or "均衡多因子"),
                "+ name: " + str(info.get("name")) + "（网格寻优）",
                "@@ 因子权重（AI 定范围，服务端寻优） @@",
            ]
            added = removed = 2
            for name in FACTOR_NAMES:
                old_value = round(float(current_weights.get(name, 0.0)), 4)
                new_value = round(float(candidate_weights.get(name, 0.0)), 4)
                if abs(old_value - new_value) < 0.0005:
                    diff_lines.append(f"  {name}: {old_value:.4f}")
                    continue
                diff_lines.append(f"- {name}: {old_value:.4f}")
                diff_lines.append(f"+ {name}: {new_value:.4f}")
                added += 1
                removed += 1
            current_desc = str(current.get("description") or "").strip()
            new_desc = str(info.get("description") or "").strip()
            if current_desc != new_desc:
                diff_lines.append("- description: " + current_desc)
                diff_lines.append("+ description: " + new_desc)
                added += 1
                removed += 1
            return {"file": file_name, "language": "text", "kind": "model_type",
                    "diff": "\n".join(diff_lines), "added": added, "removed": removed}

        current = (current_model or {}).get("weights") or {}
        candidate = proposal.get("weights") or {}
        file_name = "models/strategy.py · 因子权重"
        diff_lines = ["--- " + file_name + "（当前）", "+++ " + file_name + "（候选）"]
        added = removed = 0
        for name in FACTOR_NAMES:
            old_value = round(float(current.get(name, 0.0)), 4)
            new_value = round(float(candidate.get(name, 0.0)), 4)
            if abs(old_value - new_value) < 0.0005:
                continue
            diff_lines.append(f"- {name}: {old_value:.4f}")
            diff_lines.append(f"+ {name}: {new_value:.4f}")
            added += 1
            removed += 1
        if not added:
            diff_lines.append("  （权重无显著变化）")
        return {"file": file_name, "language": "text", "kind": "weights",
                "diff": "\n".join(diff_lines), "added": added, "removed": removed}

    @staticmethod
    def _code_change(current_model: dict[str, Any], code: str) -> dict[str, Any]:
        """生成代码方案的行级 diff。"""
        current_code = str((current_model or {}).get("code") or "")
        new_lines = str(code or "").splitlines()
        file_name = "models/strategy.py"
        if current_code.strip():
            diff_lines = list(difflib.unified_diff(
                current_code.splitlines(), new_lines,
                fromfile=file_name + "（当前）", tofile=file_name + "（候选）", lineterm=""))
        else:
            diff_lines = ["--- /dev/null", "+++ " + file_name + "（AI 新增）"]
            diff_lines += ["+" + line for line in new_lines]
        added = sum(1 for line in diff_lines
                    if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff_lines
                      if line.startswith("-") and not line.startswith("---"))
        return {"file": file_name, "language": "python", "kind": "code",
                "diff": "\n".join(diff_lines), "added": added, "removed": removed}

    @staticmethod
    def _proposal_public(proposal: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
        approach_label = {"model_type": "服务端网格寻优（切换基础模型）",
                          "weights": "服务端网格寻优（因子权重）",
                          "code": "服务端网格寻优（自定义引擎）"}.get(
            str(proposal.get("approach")), "网格寻优")
        return {
            "summary": proposal.get("summary", ""),
            "approach": approach_label,
            "model_name": model_name or proposal.get("model_type") or "网格寻优权重",
            "diff": proposal.get("code", ""),
            "why": proposal.get("why", ""),
            "expected_effect": proposal.get("expected_effect", ""),
            "risks": proposal.get("risks") or [],
        }

    # ------------------------------------------------------------------
    # 动作：应用/回滚（模型只在当前会话内生效，新对话回到默认模型）
    # ------------------------------------------------------------------
    def _apply_pending(self, session_id: str):
        runtime = self.store.get_state(session_id)
        pending = runtime.get("pending")
        if not pending:
            raise ConflictError("没有等你确认的改进方案")
        active_version_id = runtime.get("active_version_id")
        previous = self.store.get_version(active_version_id) if active_version_id else None
        if previous is None:
            # 第一次应用 AI 改动前，先登记当前默认模型为基线，保证随时可以回滚
            baseline_id = self.store.create_version(
                model=self._default_model(),
                metrics=pending.get("base_metrics") or {},
                status="candidate", parent_id=None, source="baseline",
                summary="初始模型", session_id=session_id)
            previous = self.store.get_version(baseline_id)
        version_id = self.store.create_version(
            model=pending["model"], metrics=pending["metrics"],
            status="candidate", parent_id=previous["id"] if previous else None,
            source="ai_optimize", summary=pending["proposal"].get("summary", ""),
            session_id=session_id)
        self.store.audit("ai_apply", from_version=previous["id"] if previous else None,
                         to_version=version_id, reason="user_approved_in_chat")
        self._save_runtime(session_id, pending=None,
                           active_model=pending["model"],
                           active_metrics=pending.get("metrics") or {},
                           active_version_id=version_id)
        version_block = self._version_block(session_id)
        yield "ui", {"block": version_block}
        metrics = pending.get("metrics") or {}
        return [version_block], (f"已启用 {self._model_name_from_model(pending['model'])}；回测夏普 "
                    f"{float(metrics.get('sharpe', 0)):.2f}，之后当前会话的选股会用这个新模型"
                    "（新建对话将回到默认模型）。"), False

    def _rollback(self, session_id: str):
        runtime = self.store.get_state(session_id)
        active_version_id = runtime.get("active_version_id")
        if not active_version_id:
            raise ConflictError("当前会话还没有可回滚的模型版本")
        current = self.store.get_version(active_version_id)
        if current is None:
            raise ConflictError("找不到当前会话的模型版本")
        parent = self.store.get_version(current["parent_id"]) if current.get("parent_id") else None
        if parent is not None:
            model = dict(parent.get("model") or self._default_model())
            metrics = parent.get("metrics") or {}
            previous_id = parent["id"]
        else:
            model = self._default_model()
            metrics = {}
            previous_id = None
        self.store.audit("rollback", from_version=active_version_id,
                         to_version=previous_id, reason="user_rollback")
        self._save_runtime(session_id, active_model=model, active_metrics=metrics,
                           active_version_id=previous_id)
        version_block = self._version_block(session_id)
        yield "ui", {"block": version_block}
        return [version_block], f"已切回{self._model_name_from_model(model)}（仅当前会话生效）。", False

    # ------------------------------------------------------------------
    # 状态与模型
    # ------------------------------------------------------------------
    def _profile(self, session_id: str) -> dict[str, Any] | None:
        return self.store.get_profile(session_id)

    def _profile_block(self, session_id: str) -> dict[str, Any]:
        profile = self._profile(session_id)
        payload = profile_payload(profile["slots"] if profile else {},
                                  profile["confirmed"] if profile else False)
        return {"id": "profile-" + session_id[:6], "type": "profile", **payload}

    def _session_active_model(self, session_id: str) -> dict[str, Any]:
        """当前会话生效的模型：没有则使用默认数学模型（模型仅会话内生效）。"""
        runtime = self.store.get_state(session_id)
        model = runtime.get("active_model")
        if isinstance(model, dict) and model.get("weights"):
            payload = dict(model)
            payload.setdefault("kind", "builtin")
            return payload
        return self._default_model()

    def _session_versions(self, session_id: str) -> list[dict[str, Any]]:
        return self.store.list_versions(session_id=session_id)

    def _can_rollback(self, session_id: str) -> bool:
        """当前会话是否存在可回滚的上一版模型。"""
        runtime = self.store.get_state(session_id)
        version_id = runtime.get("active_version_id")
        if not version_id:
            return False
        current = self.store.get_version(version_id)
        return bool(current and current.get("parent_id"))

    @staticmethod
    def _default_model() -> dict[str, Any]:
        from .models import registry
        info = registry.get_model("multi_factor")
        return {
            "kind": "builtin", "model_id": "multi_factor",
            "name": info.get("name", "均衡多因子"),
            "description": info.get("description", ""),
            "weights": info["weights"],
        }

    @staticmethod
    def _model_name_from_model(model: dict[str, Any] | None) -> str:
        model = model or {}
        return str(model.get("name") or model.get("model_id") or "默认模型")

    @staticmethod
    def _model_name(version: dict[str, Any]) -> str:
        model = version.get("model") or {}
        return str(model.get("name") or model.get("model_id") or version.get("id", ""))

    @classmethod
    def _model_public(cls, model: dict[str, Any] | None, *,
                      version_id: str | None = None,
                      metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        model = model or cls._default_model()
        return {
            "version_id": version_id,
            "name": cls._model_name_from_model(model),
            "description": str(model.get("description") or ""),
            "kind": str(model.get("kind") or "builtin"),
            "metrics": metrics or {},
        }

    def _model_engine(self, model: dict[str, Any]) -> tuple[dict[str, float], Any]:
        weights = normalize_weights(model.get("weights") or {})
        transform = None
        if model.get("kind") == "code" and model.get("code"):
            from .models import custom
            params = model.get("params") if isinstance(model.get("params"), dict) else None
            transform = custom.compile_code(str(model["code"]), params)
        return weights, transform

    @staticmethod
    def _wants_aggressive_retry(text: str, pending: dict[str, Any] | None = None) -> bool:
        value = str(text or "")
        return any(word in value for word in (
            "继续优化", "上一版未达", "未达到我的预期", "没达到预期", "未达预期",
            "再试一版", "换个方向", "更激进", "直接换", "重新优化模型",
        )) or bool(pending and any(word in value for word in ("继续", "再试", "不满意")))

    @staticmethod
    def _previous_approach(pending: dict[str, Any] | None) -> str:
        model = (pending or {}).get("model") or {}
        if model.get("kind") == "code":
            return "code"
        if model.get("model_id") == "ai_tuned_weights":
            return "weights"
        if model.get("kind") == "builtin":
            return "model_type"
        return ""

    @staticmethod
    def _different_builtin_model(current: dict[str, Any],
                                 previous: dict[str, Any] | None = None) -> str:
        from .models import registry
        excluded = {str((current or {}).get("model_id") or "multi_factor"),
                    str((previous or {}).get("model_id") or "")}
        for info in registry.list_models():
            if info["id"] not in excluded:
                return str(info["id"])
        return "multi_factor" if "multi_factor" not in excluded else "low_vol_quality"

    @staticmethod
    def _wants_code_change(text: str) -> bool:
        raw = str(text or "")
        negative = ("不要改代码", "不用改代码", "别改代码", "不需要改代码")
        if any(word in raw for word in negative):
            return False
        return any(word in raw for word in (
            "修改代码", "改代码", "源代码", "build_transform", "自定义代码", "代码方案"))

    def _current_metrics(self, session_id: str) -> dict[str, Any]:
        """当前会话生效模型在同一回测窗口下的指标（用于寻优前后对比）。"""
        from .engine import selection

        runtime = self.store.get_state(session_id)
        model = self._session_active_model(session_id)
        if runtime.get("active_version_id") and runtime.get("active_metrics"):
            return _json_safe(runtime["active_metrics"])
        weights, transform = self._model_engine(model)
        start, end = self._backtest_window()
        backtest = selection.run_backtest(
            self.data, weights, start=start, end=end, top_n=DEFAULT_TOP_N,
            rebalance_days=REBALANCE_DAYS, cost_buy=COST_BUY, cost_sell=COST_SELL,
            risk_free_rate=RISK_FREE_RATE, score_transform=transform)
        return _json_safe(backtest["metrics"])

    @staticmethod
    def _factor_stats(scores: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for name in weights:
            column = f"z_{name}"
            if column not in scores.columns:
                continue
            values = pd.to_numeric(scores[column], errors="coerce").dropna()
            if values.empty:
                continue
            stats[name] = {
                "label": FACTOR_PLAIN.get(name, name),
                "market_mean": round(float(values.mean()), 3),
                "p20": round(float(values.quantile(0.2)), 3),
                "p80": round(float(values.quantile(0.8)), 3),
            }
        return stats

    def _recent_metrics(self, model: dict[str, Any]) -> dict[str, Any]:
        from .engine import selection
        weights, transform = self._model_engine(model)
        index = self.data.prices.index
        end = index[-1]
        start = max(end - pd.Timedelta(days=365), index[min(len(index) - 1, 200)])
        backtest = selection.run_backtest(
            self.data, weights, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"), top_n=DEFAULT_TOP_N,
            rebalance_days=REBALANCE_DAYS, cost_buy=COST_BUY, cost_sell=COST_SELL,
            risk_free_rate=RISK_FREE_RATE, score_transform=transform)
        return _json_safe(backtest["metrics"])

    @staticmethod
    def _factor_focus(top_df: pd.DataFrame, weights: dict[str, float], limit: int = 4) -> list[str]:
        totals: dict[str, float] = {}
        for name, weight in weights.items():
            column = f"z_{name}"
            if column not in top_df.columns or len(top_df) == 0:
                continue
            totals[name] = float(top_df[column].mean()) * float(weight)
        ordered = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
        return [FACTOR_PLAIN.get(name, name) for name, _ in ordered[:limit]]

    def _planner_state(self, session_id: str) -> dict[str, Any]:
        profile = self._profile(session_id)
        runtime = self.store.get_state(session_id)
        model = self._session_active_model(session_id)
        active_version_id = runtime.get("active_version_id")
        versions = self._session_versions(session_id)
        last = runtime.get("last_selection")
        pending = runtime.get("pending")
        return {
            "market": {"as_of": self.data.as_of, "universe": len(self.data.universe)},
            "profile": profile_payload(profile["slots"] if profile else {},
                                       profile["confirmed"] if profile else False),
            "model": self._model_public(
                model, version_id=active_version_id,
                metrics=runtime.get("active_metrics") or {}),
            "model_scope": "session",
            "versions": [{"id": item["id"], "name": self._model_name(item),
                          "status": ("active" if item["id"] == active_version_id
                                     else "archived")}
                         for item in versions[-5:]],
            "pending": ({"summary": (pending.get("proposal") or {}).get("summary", ""),
                         "tuning": pending.get("tuning") or {}}
                        if pending else None),
            "pending_intent": runtime.get("pending_intent"),
            "last_selection": ({"as_of": last.get("as_of"),
                                "top": [item.get("code") for item in (last.get("top") or [])[:5]],
                                "metrics": last.get("metrics")} if last else None),
        }

    def _save_runtime(self, session_id: str, **updates: Any) -> None:
        runtime = self.store.get_state(session_id)
        for key, value in updates.items():
            if value is None:
                runtime.pop(key, None)
            else:
                runtime[key] = value
        self.store.save_state(session_id, runtime)

    # ------------------------------------------------------------------
    # 版本展示（按会话隔离：新对话默认模型，旧对话保留各自版本链）
    # ------------------------------------------------------------------
    def _version_public(self, version: dict[str, Any],
                        active_version_id: str | None = None) -> dict[str, Any]:
        status = "active" if (active_version_id
                              and version["id"] == active_version_id) else "archived"
        return {"id": version["id"], "name": self._model_name(version),
                "status": status, "created_at": version["created_at"],
                "metrics": version.get("metrics") or {},
                "summary": version.get("summary", "")}

    def _version_block(self, session_id: str) -> dict[str, Any]:
        runtime = self.store.get_state(session_id)
        model = self._session_active_model(session_id)
        active_version_id = runtime.get("active_version_id")
        versions = [self._version_public(item, active_version_id)
                    for item in self._session_versions(session_id)]
        return {"id": "version-" + uuid.uuid4().hex[:8], "type": "version",
                "active": self._model_public(
                    model, version_id=active_version_id,
                    metrics=runtime.get("active_metrics") or {}),
                "versions": versions[-6:]}

    @staticmethod
    def _pending_public(pending: dict[str, Any] | None) -> dict[str, Any] | None:
        if not pending:
            return None
        return {"summary": (pending.get("proposal") or {}).get("summary", ""),
                "metrics": pending.get("metrics") or {},
                "base_metrics": pending.get("base_metrics") or {},
                "model_name": (pending.get("model") or {}).get("name", ""),
                "tuning": pending.get("tuning") or {}}

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _require_session(self, session_id: str) -> None:
        if not self.store.session_exists(session_id):
            raise NotFoundError(f"会话不存在：{session_id}")

    def _backtest_window(self) -> tuple[str, str]:
        index = self.data.prices.index
        if len(index) == 0:
            raise ServiceError("行情数据为空，无法回测")
        end = index[-1]
        start = end - pd.Timedelta(days=int(365 * BACKTEST_YEARS))
        warmup_start = index[min(len(index) - 1, 200)]
        start = max(start, warmup_start)
        if start >= end:
            raise ServiceError("可用历史数据不足，无法回测")
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    @staticmethod
    def _assistant_text(reply: str, notes: list[str]) -> str:
        clean_notes = [note for note in notes if note]
        return reply + ("\n\n" + "\n".join(clean_notes) if clean_notes else "")

    def _remember_assistant(self, session_id: str, reply: str, notes: list[str],
                            blocks: list[dict[str, Any]] | None = None) -> None:
        text = self._assistant_text(reply, notes)
        if text.strip():
            self.store.add_message(session_id, "assistant", text, blocks or [])

    def close(self) -> None:
        self.store.close()


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in (weights or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            values[str(name)] = number
    total = sum(values.values())
    if total <= 0:
        raise ServiceError("模型权重为空")
    return {name: value / total for name, value in values.items()}
