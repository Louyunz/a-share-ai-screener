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
    "plan": "提出改进方案",
    "code": "修改模型代码",
    "check": "检查代码安不安全",
    "backtest": "拿候选方案回测",
    "compare": "和现在比一比",
}


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

    def get_state(self, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        profile = self._profile(session_id)
        active = self.store.active_version()
        versions = self.store.list_versions()
        runtime = self.store.get_state(session_id)
        return {
            "profile": profile_payload(profile["slots"] if profile else {},
                                       profile["confirmed"] if profile else False),
            "active_model": self._active_model_public(active),
            "versions": [self._version_public(item) for item in versions],
            "pending_proposal": self._pending_public(runtime.get("pending")),
            "last_selection": runtime.get("last_selection"),
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
        if "rollback_version" in action_names and self.store.parent_version() is None:
            plan.actions = [item for item in plan.actions
                            if str(item.get("name")) != "rollback_version"]
            if not plan.actions:
                plan.reply = ("当前没有可以回滚的历史模型版本。您可以先让我分析当前模型，"
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
        model = self._active_model()
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
    # 动作：AI 分析和建模
    # ------------------------------------------------------------------
    def _analyze_model(self, session_id: str, request_text: str = ""):
        from .models import registry

        model = self._active_model()
        weights, _ = self._model_engine(model)
        runtime_before = self.store.get_state(session_id)
        previous_pending = runtime_before.get("pending") or {}
        previous_approach = self._previous_approach(previous_pending)
        aggressive = self._wants_aggressive_retry(request_text, previous_pending)
        aggressive_model = ""
        aggressive_code = False
        if aggressive:
            if previous_approach == "weights":
                aggressive_model = self._different_builtin_model(
                    model, previous_pending.get("model") or {})
            else:
                aggressive_code = True
        from .engine import selection

        work_id = "work-" + uuid.uuid4().hex[:8]
        steps: list[dict[str, Any]] = []

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

        base_metrics = self._current_metrics(session_id)
        add_step("看当前表现",
                 f"近两年年化 {float(base_metrics.get('cagr', 0)) * 100:.1f}%，"
                 f"最大回撤 {abs(float(base_metrics.get('mdd', 0))) * 100:.1f}%，"
                 f"夏普 {float(base_metrics.get('sharpe', 0)):.2f}。", "done", "analysis")
        yield "ui", publish()

        top_df, _scores = selection.screen(self.data, weights, top_n=DEFAULT_TOP_N,
                                           score_transform=self._model_engine(model)[1])
        focus = self._factor_focus(top_df, weights)
        add_step("翻数据找线索", f"最近组合更看重：{'、'.join(focus[:3]) or '均衡配置'}；"
                                "先把偏弱的环节找出来。", "done", "data")
        if aggressive:
            if aggressive_model:
                info = registry.get_model(aggressive_model)
                add_step("升级优化策略",
                         f"上一版未达预期，本轮不再微调权重，直接更换基础模型为「{info.get('name')}」并重新回测。",
                         "done", "plan")
            else:
                add_step("升级优化策略",
                         "上一版未达预期，本轮不再微调权重，改为由 AI 重写数学打分模型并重新回测。",
                         "done", "plan")
        yield "ui", publish()

        factor_stats = self._factor_stats(_scores, weights)
        recent_metrics = self._recent_metrics(model)
        custom_model = self._requested_custom_model(request_text)
        force_code = self._wants_code_change(request_text) or bool(custom_model) or aggressive_code
        requested_model = "" if custom_model else (aggressive_model or self._requested_model_type(request_text))
        if requested_model:
            # 客户点名或激进重试要求更换基础模型：优先按目标模型生成确定性方案
            proposal = self._normalize_proposal(self._model_switch_proposal(requested_model), weights)
        else:
            proposal_raw = self._propose_change(model, base_metrics, factor_stats, recent_metrics,
                                                request_text, force_code, custom_model=custom_model,
                                                aggressive=aggressive, previous=previous_pending)
            proposal = self._normalize_proposal(proposal_raw, weights)
            if force_code and proposal["approach"] != "code":
                # 客户点名自定义模型、要求改代码，或激进重试时必须让 AI 写代码实现
                proposal_raw = self._propose_change(model, base_metrics, factor_stats, recent_metrics,
                                                    request_text, True, strict=True,
                                                    custom_model=custom_model,
                                                    aggressive=aggressive, previous=previous_pending)
                proposal = self._normalize_proposal(proposal_raw, weights)
        add_step("提出改进方案", str(proposal.get("summary", "")), "done", "plan")
        repair_attempts = int(proposal.get("repair_attempts") or 0)
        if repair_attempts:
            add_step("自动修复代码",
                     f"首次生成的代码未通过校验，AI 已根据错误自动修复 {repair_attempts} 次，"
                     "并通过静态检查与子进程冒烟。", "done", "check")
        yield "ui", publish(proposal=self._proposal_public(proposal))

        candidate_weights = proposal.get("weights") or weights
        transform = None
        checks = ""
        change_preview = self._change_preview(model, proposal)
        diff_lines = str(change_preview.get("diff") or "").split("\n")

        # 代码 / 配置改动流式写入：先发空改动框，再逐行推送，最后补全统计
        info = None
        if proposal["approach"] == "code":
            step_title = "修改模型代码"
            running_detail = "正在根据方案改写策略打分逻辑，代码逐行写入中…"
        elif proposal["approach"] == "model_type":
            info = registry.get_model(str(proposal.get("model_type")))
            candidate_weights = info["weights"]
            step_title = "切换模型类型"
            running_detail = "正在写入模型类型与配置，改动逐行展示中…"
        else:
            step_title = "调整因子权重"
            running_detail = "正在写入新的因子权重，改动逐行展示中…"
        add_step(step_title, running_detail, "running", "code")
        partial_change = dict(change_preview)
        partial_change["diff"] = ""
        partial_change["added"] = 0
        partial_change["removed"] = 0
        yield "ui", publish(proposal=self._proposal_public(proposal), extra={"code_change": partial_change})
        for index, line in enumerate(diff_lines):
            if not line and index == len(diff_lines) - 1:
                continue
            partial = diff_lines[:index + 1]
            partial_change = dict(change_preview)
            partial_change["diff"] = "\n".join(partial)
            partial_change["added"] = len([item for item in partial if item.startswith("+") and not item.startswith("+++")])
            partial_change["removed"] = len([item for item in partial if item.startswith("-") and not item.startswith("---")])
            yield "ui", publish(proposal=self._proposal_public(proposal), extra={"code_change": partial_change})
            time.sleep(0.03)

        if proposal["approach"] == "code":
            from .models import custom
            validation = custom.validate_code(proposal["code"])
            checks = "、".join(str(item.get("name")) for item in validation.get("checks", []))
            steps[-1] = dict(steps[-1], status="done",
                             detail=f"改写策略打分逻辑：新增 {change_preview['added']} 行、"
                                    f"删除 {change_preview['removed']} 行，具体改动见下方代码框。")
            yield "ui", publish(proposal=self._proposal_public(proposal),
                                extra={"code_change": change_preview})
            add_step("执行代码安全检查", f"通过静态检查（{checks or '导入、危险调用、返回格式'}）"
                                    "及子进程冒烟测试。", "done", "check")
            transform = custom.compile_code(proposal["code"])
            yield "ui", publish(proposal=self._proposal_public(proposal),
                                extra={"code_change": change_preview})
        elif proposal["approach"] == "model_type":
            steps[-1] = dict(steps[-1], status="done",
                             detail=f"切换模型类型为「{info.get('name')}」：{info.get('description')}；"
                                    "具体配置改动见下方改动框。")
            yield "ui", publish(proposal=self._proposal_public(proposal, model_name=info.get("name")),
                                extra={"code_change": change_preview})
        else:
            changed = len([line for line in change_preview["diff"].splitlines()
                           if line.startswith("+") and not line.startswith("+++")])
            steps[-1] = dict(steps[-1], status="done",
                             detail=f"调整 {changed} 个因子权重，具体改动见下方改动框。")
            yield "ui", publish(proposal=self._proposal_public(proposal),
                                extra={"code_change": change_preview})

        start, end = self._backtest_window()
        candidate_bt = selection.run_backtest(
            self.data, candidate_weights, start=start, end=end, top_n=DEFAULT_TOP_N,
            rebalance_days=REBALANCE_DAYS, cost_buy=COST_BUY, cost_sell=COST_SELL,
            risk_free_rate=RISK_FREE_RATE, score_transform=transform)
        candidate_metrics = _json_safe(candidate_bt["metrics"])
        add_step("拿候选方案回测",
                 f"候选策略年化 {float(candidate_metrics.get('cagr', 0)) * 100:.1f}%，"
                 f"最大回撤 {abs(float(candidate_metrics.get('mdd', 0))) * 100:.1f}%。",
                 "done", "backtest")
        comparison = {"current": _json_safe(base_metrics), "candidate": candidate_metrics}
        yield "ui", publish(proposal=self._proposal_public(proposal), comparison=comparison)

        add_step("和现在比一比",
                 f"夏普从 {float(base_metrics.get('sharpe', 0)):.2f} 变为 "
                 f"{float(candidate_metrics.get('sharpe', 0)):.2f}，等你确认后再启用。",
                 "done", "compare")
        expectation = self._expectation(request_text, session_id)
        expectation_met, expectation_reason = self._meets_expectation(
            expectation, base_metrics, candidate_metrics)
        add_step("核对您的目标", expectation_reason,
                 "done" if expectation_met else "error", "check")
        yield "ui", publish(proposal=self._proposal_public(proposal),
                            comparison=comparison)
        candidate_model = self._candidate_model(proposal, model)
        pending = {
            "id": proposal.get("id") or uuid.uuid4().hex[:8],
            "proposal": self._proposal_public(proposal),
            "model": candidate_model,
            "metrics": candidate_metrics,
            "nav": _downsample(candidate_bt["nav"]),
            "base_metrics": _json_safe(base_metrics),
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_runtime(session_id, pending=pending)
        final_extra = {"nav": _downsample(candidate_bt["nav"])}
        final_extra["code_change"] = change_preview
        final_work = publish(status="done",
                             proposal=self._proposal_public(proposal),
                             comparison=comparison,
                             extra=final_extra)
        yield "ui", final_work
        approval_actions = [
            {"label": "同意，启用吧", "message": "同意启用这个改进"},
            {"label": "先不启用", "message": "先别启用，我再想想"},
        ]
        if not expectation_met:
            approval_actions.append({"label": "继续优化", "message": "继续优化模型，上一版未达到我的预期"})
        approval = {
            "id": "approval-" + uuid.uuid4().hex[:8],
            "type": "approval",
            "title": "是否启用这个改进？" if expectation_met else "候选未达预期，是否仍要启用？",
            "summary": (f"AI 建议：{proposal.get('summary', '')} "
                        f"回测夏普 {float(base_metrics.get('sharpe', 0)):.2f} → "
                        f"{float(candidate_metrics.get('sharpe', 0)):.2f}。"
                        + ("" if expectation_met else f" 目标核对：{expectation_reason}")),
            "risk": ("；".join(proposal.get("risks") or []) or "历史回测不代表未来表现。")
                    + ("" if expectation_met else " 注意：候选方案未达到您的目标，建议继续优化或先保留现模型。"),
            "actions": approval_actions,
        }
        yield "ui", {"block": approval}
        note_text = ""
        if not expectation_met:
            note_text = ("候选方案未达到您的目标：" + expectation_reason
                         + " 您可以先不启用，或回复「继续优化」；我会直接更换基础模型或重写数学打分模型再试一版。")
        return [final_work["block"], approval], note_text, False

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

    @staticmethod
    def _model_switch_proposal(model_type: str) -> dict[str, Any]:
        from .models import registry
        info = registry.get_model(model_type)
        return {
            "approach": "model_type",
            "model_type": model_type,
            "summary": f"按您的要求切换为「{info.get('name')}」模型：{info.get('description')}",
            "why": "您明确点名了该模型；本次先保留既有因子框架，切换为对应风格权重后再做同区间回测。",
            "expected_effect": "风格与您指定的模型一致；收益与回撤变化以本次候选回测数据为准。",
            "risks": ["历史回测不代表未来实际表现。", "切换风格后短期表现可能与原模型差异较大。"],
        }

    def _propose_change(self, model: dict[str, Any], base_metrics: dict[str, Any],
                        factor_stats: dict[str, Any], recent_metrics: dict[str, Any],
                        request_text: str = "", force_code: bool = False,
                        strict: bool = False, custom_model: str = "",
                        aggressive: bool = False,
                        previous: dict[str, Any] | None = None) -> dict[str, Any]:
        from .models import pipeline
        previous = previous or {}
        previous_model = previous.get("model") or {}
        state = {
            "proposal_request": True,
            "escalation": {
                "active": bool(aggressive),
                "previous_approach": self._previous_approach(previous),
                "previous_model": previous_model.get("model_id") or "",
                "previous_name": previous_model.get("name") or "",
            },
            "previous_proposal": previous.get("proposal") or {},
            "previous_metrics": previous.get("metrics") or {}, 
            "current_model": {"name": model.get("name"), "kind": model.get("kind"),
                              "model_id": model.get("model_id"),
                              "description": model.get("description")},
            "weights": model.get("weights") or {},
            "current_metrics": _json_safe(base_metrics),
            "factor_stats": factor_stats,
            "recent_metrics": recent_metrics,
            "user_request": request_text,
            "force_approach": "code" if force_code else "",
            "requested_custom_model": custom_model,
            "market": {"as_of": self.data.as_of, "universe": len(self.data.universe)},
        }
        if strict:
            state["strict_requirement"] = (
                "客户明确要求修改策略代码、点名自定义模型，或上一版未达预期需要激进升级；"
                "本次必须输出 approach=code 的代码方案，主动重写数学打分逻辑，"
                "不得用内置模型或权重调整替代，且应与上一版代码结构不同。"
            )
        try:
            raw = pipeline.build_candidate_proposal(self.llm, state)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"AI 没能给出改进方案：{exc}") from exc
        if isinstance(raw, dict) and "proposal" in raw:
            raw = raw["proposal"]
        return raw

    def _normalize_proposal(self, raw: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
        from .models import pipeline
        if not isinstance(raw, dict):
            raise AgentError("AI 返回的改进方案格式不对")
        if "weights" not in raw:
            raw = dict(raw)
            raw["weights"] = weights
        return pipeline.normalize_proposal(raw, base_weights=weights)

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
                "+ name: " + str(info.get("name")),
                "@@ 因子权重 @@",
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
        approach_label = {"model_type": "切换模型类型", "weights": "调整因子权重",
                          "code": "AI 生成新策略代码"}.get(str(proposal.get("approach")), "优化")
        return {
            "summary": proposal.get("summary", ""),
            "approach": approach_label,
            "model_name": model_name or proposal.get("model_type") or "AI 调整权重",
            "diff": proposal.get("code", ""),
            "why": proposal.get("why", ""),
            "expected_effect": proposal.get("expected_effect", ""),
            "risks": proposal.get("risks") or [],
        }

    # ------------------------------------------------------------------
    # 动作：应用/回滚
    # ------------------------------------------------------------------
    def _apply_pending(self, session_id: str):
        runtime = self.store.get_state(session_id)
        pending = runtime.get("pending")
        if not pending:
            raise ConflictError("没有等你确认的改进方案")
        previous = self.store.active_version()
        if previous is None:
            # 第一次应用 AI 改动前，先登记当前模型为基线，保证随时可以回滚
            baseline_id = self.store.create_version(
                model=self._default_model(), metrics=pending.get("base_metrics") or {},
                status="active", parent_id=None, source="baseline",
                summary="初始模型")
            previous = self.store.get_version(baseline_id)
        version_id = self.store.create_version(
            model=pending["model"], metrics=pending["metrics"],
            status="candidate", parent_id=previous["id"] if previous else None,
            source="ai_optimize", summary=pending["proposal"].get("summary", ""))
        active = self.store.activate(version_id, reason="ai_approved")
        self.store.audit("ai_apply", from_version=previous["id"] if previous else None,
                         to_version=version_id, reason="user_approved_in_chat")
        self._save_runtime(session_id, pending=None)
        version_block = self._version_block()
        yield "ui", {"block": version_block}
        metrics = active.get("metrics") or {}
        return [version_block], (f"已启用「{self._model_name(active)}」；回测夏普 "
                    f"{float(metrics.get('sharpe', 0)):.2f}，之后选股会用这个新模型。"), False

    def _rollback(self, session_id: str):
        previous = self.store.parent_version()
        if previous is None:
            raise ConflictError("找不到可以回滚的旧版本")
        self.store.rollback(previous["id"], reason="user_rollback")
        version_block = self._version_block()
        yield "ui", {"block": version_block}
        return [version_block], f"已切回「{self._model_name(previous)}」。", False

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

    def _active_model(self) -> dict[str, Any]:
        active = self.store.active_version()
        if active and active.get("model"):
            model = dict(active["model"])
            model["version_id"] = active["id"]
            return model
        return self._default_model()

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
    def _model_name(version: dict[str, Any]) -> str:
        model = version.get("model") or {}
        return str(model.get("name") or model.get("model_id") or version.get("id", ""))

    def _active_model_public(self, active: dict[str, Any] | None) -> dict[str, Any]:
        name = self._model_name(active) if active else self._default_model()["name"]
        description = (active.get("model") or {}).get("description", "") if active else ""
        metrics = (active.get("metrics") or {}) if active else {}
        return {"version_id": active.get("id") if active else None,
                "name": name, "description": description,
                "kind": (active.get("model") or {}).get("kind", "builtin") if active else "builtin",
                "metrics": metrics}

    def _model_engine(self, model: dict[str, Any]) -> tuple[dict[str, float], Any]:
        weights = normalize_weights(model.get("weights") or {})
        transform = None
        if model.get("kind") == "code" and model.get("code"):
            from .models import custom
            transform = custom.compile_code(str(model["code"]))
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

    def _candidate_model(self, proposal: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        from .models import registry
        if proposal["approach"] == "model_type":
            info = registry.get_model(str(proposal["model_type"]))
            return {"kind": "builtin", "model_id": info["id"], "name": info.get("name", ""),
                    "description": info.get("description", ""), "weights": info["weights"],
                    "summary": proposal.get("summary", "")}
        if proposal["approach"] == "code":
            return {"kind": "code", "model_id": "ai_generated", "name": "AI 生成模型",
                    "description": proposal.get("summary", ""),
                    "weights": proposal["weights"], "code": proposal["code"],
                    "summary": proposal.get("summary", "")}
        return {"kind": "builtin", "model_id": "ai_tuned_weights", "name": "AI 调整权重",
                "description": proposal.get("summary", ""), "weights": proposal["weights"],
                "summary": proposal.get("summary", "")}

    def _current_metrics(self, session_id: str) -> dict[str, Any]:
        runtime = self.store.get_state(session_id)
        block = runtime.get("last_selection")
        if block and block.get("metrics"):
            return block["metrics"]
        model = self._active_model()
        weights, transform = self._model_engine(model)
        from .engine import selection
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
        active = self.store.active_version()
        last = runtime.get("last_selection")
        pending = runtime.get("pending")
        return {
            "market": {"as_of": self.data.as_of, "universe": len(self.data.universe)},
            "profile": profile_payload(profile["slots"] if profile else {},
                                       profile["confirmed"] if profile else False),
            "model": self._active_model_public(active),
            "versions": [{"id": item["id"], "name": self._model_name(item),
                          "status": item["status"]} for item in self.store.list_versions()[-5:]],
            "pending": ({"summary": (pending.get("proposal") or {}).get("summary", "")}
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
    # 版本展示
    # ------------------------------------------------------------------
    def _version_public(self, version: dict[str, Any]) -> dict[str, Any]:
        return {"id": version["id"], "name": self._model_name(version),
                "status": version["status"], "created_at": version["created_at"],
                "metrics": version.get("metrics") or {},
                "summary": version.get("summary", "")}

    def _version_block(self) -> dict[str, Any]:
        active = self.store.active_version()
        versions = [self._version_public(item) for item in self.store.list_versions()]
        return {"id": "version-" + uuid.uuid4().hex[:8], "type": "version",
                "active": self._active_model_public(active), "versions": versions[-6:]}

    @staticmethod
    def _pending_public(pending: dict[str, Any] | None) -> dict[str, Any] | None:
        if not pending:
            return None
        return {"summary": (pending.get("proposal") or {}).get("summary", ""),
                "metrics": pending.get("metrics") or {},
                "base_metrics": pending.get("base_metrics") or {},
                "model_name": (pending.get("model") or {}).get("name", "")}

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
