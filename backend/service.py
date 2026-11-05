# -*- coding: utf-8 -*-
"""对话式业务服务：所有操作通过聊天流完成，AI 决定动作，服务负责执行与展示。

没有模块页签、没有独立动作接口；一次聊天 = 理解 → 回复 → 执行动作 → 富卡片。
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .agent import Agent, AgentError, chunk_text, safe_question
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

        # 1) 先更新画像，让用户马上看到
        profile = self._profile(session_id)
        slots = profile["slots"] if profile else {}
        confirmed = profile["confirmed"] if profile else False
        if plan.profile_delta:
            merged = merge_slots(slots, plan.profile_delta)
            if merged != slots:
                slots = merged
                self.store.save_profile(session_id, slots, profile_summary(slots), confirmed)
                yield "ui", {"block": self._profile_block(session_id)}

        # 2) 流式回复
        notes: list[str] = []
        try:
            for piece in chunk_text(plan.reply):
                yield "token", {"text": piece}
            for action in plan.actions:
                events, note, should_stop = yield from self._run_action(session_id, action)
                if note:
                    notes.append(note)
                    yield "token", {"text": "\n\n"}
                    for piece in chunk_text(note):
                        yield "token", {"text": piece}
                if should_stop:
                    break
        except (ServiceError, AgentError, ValidationError) as exc:
            self._remember_assistant(session_id, plan.reply, notes)
            yield "error", {"message": str(exc), "retryable": False}
            yield "done", {"message_id": uuid.uuid4().hex[:8]}
            return
        except Exception as exc:  # noqa: BLE001
            self._remember_assistant(session_id, plan.reply, notes)
            yield "error", {"message": f"处理这句时出了点问题：{exc}", "retryable": True}
            yield "done", {"message_id": uuid.uuid4().hex[:8]}
            return

        self._remember_assistant(session_id, plan.reply, notes)
        yield "done", {"message_id": uuid.uuid4().hex[:8]}

    def _run_action(self, session_id: str, action: dict[str, Any]):
        name = action.get("name")
        if name == "confirm_profile":
            return (yield from self._confirm_profile(session_id))
        if name == "run_selection":
            return (yield from self._run_selection(session_id))
        if name == "explain_results":
            question = str(action.get("args", {}).get("question", ""))
            return (yield from self._explain_results(session_id, question))
        if name == "analyze_model":
            return (yield from self._analyze_model(session_id))
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
        yield "ui", {"block": self._profile_block(session_id)}
        return [], "画像已经确认。", False

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
        return [], "已经选好 10 只股票，理由和回测结果都在上面的卡片里。", False

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
    def _analyze_model(self, session_id: str):
        from .models import registry

        model = self._active_model()
        weights, _ = self._model_engine(model)
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
        yield "ui", publish()

        factor_stats = self._factor_stats(_scores, weights)
        recent_metrics = self._recent_metrics(model)
        proposal_raw = self._propose_change(model, base_metrics, factor_stats, recent_metrics)
        proposal = self._normalize_proposal(proposal_raw, weights)
        add_step("提出改进方案", str(proposal.get("summary", "")), "done", "plan")
        yield "ui", publish(proposal=self._proposal_public(proposal))

        candidate_weights = proposal.get("weights") or weights
        transform = None
        checks = ""
        if proposal["approach"] == "code":
            from .models import custom
            validation = custom.validate_code(proposal["code"])
            checks = "、".join(str(item.get("name")) for item in validation.get("checks", []))
            add_step("修改模型代码",
                     f"新增策略代码 {len(proposal['code'].splitlines())} 行，"
                     "让它自己决定低波动和反转因子怎么加权。", "done", "code")
            yield "ui", publish(proposal=self._proposal_public(proposal))
            add_step("检查代码安不安全", f"静态检查通过：{checks or '导入、危险调用、返回格式'}；"
                                    "并在小样本上跑通。", "done", "check")
            transform = custom.compile_code(proposal["code"])
            yield "ui", publish(proposal=self._proposal_public(proposal))
        elif proposal["approach"] == "model_type":
            info = registry.get_model(str(proposal.get("model_type")))
            candidate_weights = info["weights"]
            add_step("修改模型代码",
                     f"切换模型类型为「{info.get('name')}」：{info.get('description')}",
                     "done", "code")
            yield "ui", publish(proposal=self._proposal_public(proposal, model_name=info.get("name")))
        else:
            add_step("修改模型代码", "不写新代码，直接调整 10 个因子的权重分配。", "done", "code")
            yield "ui", publish(proposal=self._proposal_public(proposal))

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
        yield "ui", publish(status="done",
                            proposal=self._proposal_public(proposal),
                            comparison=comparison,
                            extra={"nav": _downsample(candidate_bt["nav"])})
        approval = {
            "id": "approval-" + uuid.uuid4().hex[:8],
            "type": "approval",
            "title": "是否启用这个改进？",
            "summary": (f"AI 建议：{proposal.get('summary', '')} "
                        f"回测夏普 {float(base_metrics.get('sharpe', 0)):.2f} → "
                        f"{float(candidate_metrics.get('sharpe', 0)):.2f}。"),
            "risk": "；".join(proposal.get("risks") or []) or "历史回测不代表未来表现。",
            "actions": [
                {"label": "同意，启用吧", "message": "同意启用这个改进"},
                {"label": "先不启用", "message": "先别启用，我再想想"},
            ],
        }
        yield "ui", {"block": approval}
        return [], "新方案已经跑完回测，要不要启用由你决定。", False

    def _propose_change(self, model: dict[str, Any], base_metrics: dict[str, Any],
                        factor_stats: dict[str, Any],
                        recent_metrics: dict[str, Any]) -> dict[str, Any]:
        from .models import pipeline
        state = {
            "proposal_request": True,
            "current_model": {"name": model.get("name"), "kind": model.get("kind"),
                              "model_id": model.get("model_id"),
                              "description": model.get("description")},
            "weights": model.get("weights") or {},
            "current_metrics": _json_safe(base_metrics),
            "factor_stats": factor_stats,
            "recent_metrics": recent_metrics,
            "market": {"as_of": self.data.as_of, "universe": len(self.data.universe)},
        }
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
        yield "ui", {"block": self._version_block()}
        metrics = active.get("metrics") or {}
        return [], (f"已启用「{self._model_name(active)}」；回测夏普 "
                    f"{float(metrics.get('sharpe', 0)):.2f}，之后选股会用这个新模型。"), False

    def _rollback(self, session_id: str):
        previous = self.store.parent_version()
        if previous is None:
            raise ConflictError("找不到可以回滚的旧版本")
        self.store.rollback(previous["id"], reason="user_rollback")
        yield "ui", {"block": self._version_block()}
        return [], f"已切回「{self._model_name(previous)}」。", False

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

    def _remember_assistant(self, session_id: str, reply: str, notes: list[str]) -> None:
        text = self._assistant_text(reply, notes)
        if text.strip():
            self.store.add_message(session_id, "assistant", text)

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
