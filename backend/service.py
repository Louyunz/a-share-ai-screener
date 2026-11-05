# -*- coding: utf-8 -*-
"""与 Web 框架无关的业务服务层。

FastAPI（main.py）与标准库 dev_server.py 共用本层：
会话对话、选股任务、参数调优任务、版本启用/回滚、SSE 事件流。
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .agent import AgentFacade
from .config import (
    A_SHARE_DIR,
    BACKTEST_YEARS,
    COST_BUY,
    COST_SELL,
    DB_PATH,
    DEFAULT_TOP_N,
    DEFAULT_TRIALS,
    REBALANCE_DAYS,
    RISK_FREE_RATE,
    SOURCE,
)
from .data import DataBundle, load_bundle
from .events import EventBus
from .llm import LLMClient, build_llm
from .schemas import (
    ValidationError,
    excluded_industries,
    normalize_slots,
    profile_complete,
    profile_payload,
    risk_level,
)
from .store import Store


class ServiceError(RuntimeError):
    """业务错误（会映射为 HTTP 4xx/5xx 或 SSE error）。"""


class ConflictError(ServiceError):
    """状态冲突（如画像未确认）。"""


class NotFoundError(ServiceError):
    """资源不存在。"""


def _json_safe(value: Any) -> Any:
    """把 numpy/pandas 类型递归转成可 JSON 序列化的原生类型。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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
        step = max(1, len(series) // points)
        series = series.iloc[::step]
    base = float(series.iloc[0]) or 1.0
    return [{"date": idx.strftime("%Y-%m-%d"), "value": round(float(val) / base, 6)}
            for idx, val in series.items()]


class Job:
    """后台任务：保存状态、事件历史与订阅（供 SSE 回放）。"""

    def __init__(self, job_id: str, kind: str) -> None:
        self.id = job_id
        self.kind = kind
        self.status = "running"
        self.seq = 0
        self.history: list[dict[str, Any]] = []
        self.lock = threading.RLock()

    def record(self, event: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.seq += 1
            self.history.append({"seq": self.seq, "event": event, "data": data})


class AppService:
    def __init__(self, *, data_dir: Path | None = None, db_path: Path | None = None,
                 llm: LLMClient | None = None, bundle: DataBundle | None = None) -> None:
        self.data_dir = Path(data_dir or A_SHARE_DIR)
        self.store = Store(db_path or DB_PATH)
        self.bus = EventBus()
        self.data = bundle if bundle is not None else load_bundle(self.data_dir)
        self.llm = llm or build_llm()
        self.agent = AgentFacade(self.llm, self.store)
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.RLock()
        self._last_results: dict[str, dict[str, Any]] = {}
        self._results_lock = threading.RLock()

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
            "data_ready": True,
        }

    def create_session(self) -> dict[str, Any]:
        session_id = self.store.create_session()
        return {"session_id": session_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def messages(self, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        return {"messages": self.store.get_messages(session_id)}

    def get_profile(self, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        row = self.store.get_profile(session_id)
        if row is None:
            return profile_payload({}, confirmed=False)
        return profile_payload(row["slots"], confirmed=row["confirmed"])

    def put_profile(self, session_id: str, slots: dict[str, str], confirmed: bool = False) -> dict[str, Any]:
        self._require_session(session_id)
        normalized = normalize_slots(slots)
        if confirmed:
            return self.agent.confirm_profile(session_id, normalized)
        self.store.save_profile(session_id, normalized, risk_level(normalized), confirmed=False)
        return profile_payload(normalized, confirmed=False)

    # ------------------------------------------------------------------
    # 对话（含追问）
    # ------------------------------------------------------------------
    def chat_events(self, session_id: str, content: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self._require_session(session_id)
        content = (content or "").strip()
        if not content:
            raise ValidationError("消息内容不能为空")

        if self._looks_like_followup(content):
            result = self._last_result(session_id)
            if result is not None:
                try:
                    text = self.agent.explain(self._facts_from_result(result), content)
                except Exception as exc:  # noqa: BLE001
                    self.store.add_message(session_id, "user", content)
                    yield "error", {"message": f"解读生成失败：{exc}"}
                    return
                self.store.add_message(session_id, "user", content)
                self.store.add_message(session_id, "assistant", text)
                yield "token", {"text": text}
                yield "done", {"kind": "explanation"}
                return

        try:
            yield from self.agent.stream_chat(session_id, content)
            yield "done", {"kind": "chat"}
        except Exception as exc:  # noqa: BLE001
            yield "error", {"message": f"对话处理失败：{exc}"}

    @staticmethod
    def _looks_like_followup(content: str) -> bool:
        markers = ("为什么", "风险", "解读", "怎么", "如何", "吗", "?", "？", "原因")
        return any(marker in content for marker in markers)

    # ------------------------------------------------------------------
    # 选股任务
    # ------------------------------------------------------------------
    def start_selection(self, session_id: str, top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
        self._require_session(session_id)
        profile = self.store.get_profile(session_id)
        if not profile or not profile.get("confirmed"):
            raise ConflictError("画像尚未确认，请先完成画像对话并确认画像卡")
        task_id = "task_" + uuid.uuid4().hex[:10]
        job = Job(task_id, "selection")
        with self._jobs_lock:
            self._jobs[task_id] = job
        thread = threading.Thread(target=self._run_selection, args=(job, session_id, top_n),
                                  name=f"selection-{task_id}", daemon=True)
        thread.start()
        return {"task_id": task_id}

    def _run_selection(self, job: Job, session_id: str, top_n: int) -> None:
        try:
            self._publish(job, "started", {"task_id": job.id})
            profile = self.store.get_profile(session_id) or {}
            data = self.data
            weights = self._weights_for(profile)
            self._publish(job, "progress", {"stage": "因子计算", "percent": 20,
                                            "message": "计算 A 股 10 因子并打分"})
            from .engine import selection
            top_df, all_scores = selection.screen(data, weights, top_n=top_n)
            self._publish(job, "progress", {"stage": "组合构建", "percent": 45,
                                            "message": "按权重方案构建组合"})
            portfolio = selection.build_portfolio(top_df, data.prices,
                                                  method="equal_weight", capital=1_000_000.0)
            self._publish(job, "progress", {"stage": "回测验证", "percent": 65,
                                            "message": "按 A 股交易规则运行样本外回测"})
            end = data.prices.index[-1]
            start = end - pd.Timedelta(days=int(365 * BACKTEST_YEARS))
            bt = selection.run_backtest(
                data, weights, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"), top_n=top_n,
                rebalance_days=REBALANCE_DAYS, cost_buy=COST_BUY,
                cost_sell=COST_SELL, risk_free_rate=RISK_FREE_RATE)
            self._publish(job, "progress", {"stage": "AI 解读", "percent": 85,
                                            "message": "基于因子贡献与回测指标生成解读"})
            top_payload = self._top_payload(top_df, weights)
            facts = {"as_of": data.as_of, "top": top_payload[:5],
                     "metrics": bt["metrics"], "portfolio": portfolio.head(10).to_dict("records")}
            explanation = self.agent.explain(facts)
            result = {
                "task_id": job.id,
                "as_of": data.as_of,
                "top_n": top_n,
                "top": top_payload,
                "portfolio": _json_safe(portfolio.to_dict("records")),
                "metrics": _json_safe(bt["metrics"]),
                "nav": _downsample(bt["nav"]),
                "benchmark_nav": _downsample(bt["benchmark_nav"]),
                "explanation": explanation,
                "weights": _json_safe(weights),
            }
            with self._results_lock:
                self._last_results[session_id] = result
            self._publish(job, "result", result)
            self._finish(job, "done")
        except Exception as exc:  # noqa: BLE001
            self._publish(job, "error", {"message": f"选股任务失败：{exc}"})
            traceback.print_exc()
            self._finish(job, "error")

    # ------------------------------------------------------------------
    # 调优任务
    # ------------------------------------------------------------------
    def start_tuning(self, session_id: str, trials: int = DEFAULT_TRIALS) -> dict[str, Any]:
        self._require_session(session_id)
        if trials < 3 or trials > 120:
            raise ValidationError("trials 必须是 3-120 的整数")
        job_id = "job_" + uuid.uuid4().hex[:10]
        job = Job(job_id, "tuning")
        with self._jobs_lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_tuning, args=(job, session_id, trials),
                                  name=f"tuning-{job_id}", daemon=True)
        thread.start()
        return {"job_id": job_id}

    def _run_tuning(self, job: Job, session_id: str, trials: int) -> None:
        try:
            profile = self.store.get_profile(session_id) or {}
            base_weights = self._weights_for(profile)
            active = self.store.active_version()
            end = self.data.prices.index[-1]
            start = end - pd.Timedelta(days=int(365 * BACKTEST_YEARS))
            start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            from .engine import selection, tuner

            base_bt = selection.run_backtest(
                self.data, base_weights, start=start_s, end=end_s,
                top_n=DEFAULT_TOP_N, rebalance_days=REBALANCE_DAYS,
                cost_buy=COST_BUY, cost_sell=COST_SELL, risk_free_rate=RISK_FREE_RATE)
            self._publish(job, "tuning_started", {
                "job_id": job.id, "trials_total": trials,
                "base_weights": _json_safe(base_weights),
                "base_metrics": _json_safe(base_bt["metrics"]),
            })
            best = {"objective": float("-inf"), "weights": base_weights, "metrics": base_bt["metrics"]}

            def on_trial(trial: dict[str, Any]) -> None:
                nonlocal best
                objective = float(trial.get("objective", float("-inf")))
                is_best = objective > best["objective"]
                if is_best:
                    best = {"objective": objective, "weights": trial["weights"],
                            "metrics": trial["metrics"]}
                self._publish(job, "trial", {
                    "trial_index": trial.get("trial_index"),
                    "weights": _json_safe(trial.get("weights", {})),
                    "metrics": _json_safe(trial.get("metrics", {})),
                    "objective": round(objective, 6),
                    "is_best": is_best,
                })
                if is_best:
                    self._publish(job, "best_update", {
                        "best_weights": _json_safe(best["weights"]),
                        "best_metrics": _json_safe(best["metrics"]),
                        "objective": round(objective, 6),
                    })

            outcome = tuner.run_tuning(
                self.data, base_weights=base_weights, trials=trials,
                top_n=DEFAULT_TOP_N, start=start_s, end=end_s,
                seed=42, progress_cb=on_trial)
            candidate_id = self.store.create_version(
                weights=outcome["best_weights"], metrics=outcome["best_metrics"],
                status="candidate", source="auto_tune",
                parent_id=active["id"] if active else None)
            self.store.audit("tune", from_version=active["id"] if active else None,
                             to_version=candidate_id, reason=f"trials={trials}")
            self._publish(job, "done", {
                "candidate_version_id": candidate_id,
                "compare": {
                    "current": {"weights": _json_safe(base_weights),
                                "metrics": _json_safe(base_bt["metrics"])},
                    "candidate": {"weights": _json_safe(outcome["best_weights"]),
                                  "metrics": _json_safe(outcome["best_metrics"])},
                },
            })
            self._finish(job, "done")
        except Exception as exc:  # noqa: BLE001
            self._publish(job, "error", {"message": f"调优任务失败：{exc}"})
            traceback.print_exc()
            self._finish(job, "error")

    # ------------------------------------------------------------------
    # 事件流
    # ------------------------------------------------------------------
    def selection_events(self, session_id: str, task_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self._require_session(session_id)
        yield from self._stream_job(task_id)

    def tuning_events(self, session_id: str, job_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        self._require_session(session_id)
        yield from self._stream_job(job_id)

    def _stream_job(self, job_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        job = self._get_job(job_id)
        q = self.bus.subscribe(job_id)
        try:
            with job.lock:
                snapshot = list(job.history)
            last_seq = 0
            for item in snapshot:
                last_seq = max(last_seq, item["seq"])
                yield item["event"], item["data"]
            if job.status in ("done", "error"):
                return
            while True:
                try:
                    item = q.get(timeout=120)
                except queue.Empty:
                    yield "error", {"message": "任务超时未响应"}
                    return
                if item is None:
                    return
                if item.get("seq", 0) <= last_seq:
                    continue
                last_seq = item["seq"]
                yield item["event"], item["data"]
                if item["event"] in ("error",):
                    return
                if item["event"] in ("result", "done") and job.kind == "selection":
                    # 选股任务在 result 之后还有 done，继续读完再退出
                    pass
        finally:
            self.bus.unsubscribe(job_id, q)

    # ------------------------------------------------------------------
    # 模型版本
    # ------------------------------------------------------------------
    def versions(self) -> dict[str, Any]:
        versions = self.store.list_versions()
        active = self.store.active_version()
        return {"versions": versions, "active_id": active["id"] if active else None}

    def activate_version(self, version_id: str, reason: str = "user_action") -> dict[str, Any]:
        try:
            active = self.store.activate(version_id, reason=reason)
        except KeyError as exc:
            raise NotFoundError(str(exc)) from exc
        return {"active_id": active["id"]}

    def rollback_version(self, version_id: str, reason: str = "user_rollback") -> dict[str, Any]:
        try:
            active = self.store.rollback(version_id, reason=reason)
        except KeyError as exc:
            raise NotFoundError(str(exc)) from exc
        return {"active_id": active["id"]}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _require_session(self, session_id: str) -> None:
        if not self.store.session_exists(session_id):
            raise NotFoundError(f"会话不存在: {session_id}")

    def _get_job(self, job_id: str) -> Job:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"任务不存在: {job_id}")
        return job

    def _publish(self, job: Job, event: str, data: dict[str, Any]) -> None:
        job.record(event, data)
        self.bus.publish(job.id, event, data, seq=job.seq)

    def _finish(self, job: Job, status: str) -> None:
        job.status = status
        self.bus.close(job.id)

    def _weights_for(self, profile: dict[str, Any]) -> dict[str, float]:
        active = self.store.active_version()
        if active:
            return dict(active["weights"])
        from .engine.factors import weights_for
        level = (profile or {}).get("risk_level", "balanced")
        return dict(weights_for(level))

    def _last_result(self, session_id: str) -> dict[str, Any] | None:
        with self._results_lock:
            return self._last_results.get(session_id)

    @staticmethod
    def _facts_from_result(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "as_of": result.get("as_of"),
            "top": [{k: item[k] for k in ("code", "score", "industry", "contributions")
                     if k in item} for item in result.get("top", [])[:5]],
            "metrics": result.get("metrics"),
            "portfolio": result.get("portfolio", [])[:10],
        }

    @staticmethod
    def _top_payload(top_df: pd.DataFrame, weights: dict[str, float],
                     limit: int = 20) -> list[dict[str, Any]]:
        if top_df is None or len(top_df) == 0:
            return []
        records: list[dict[str, Any]] = []
        for _, row in top_df.head(limit).iterrows():
            code = str(row.get("code", row.name))
            factors: dict[str, float] = {}
            contributions: dict[str, float] = {}
            for name, weight in weights.items():
                z = row.get(f"z_{name}")
                if z is None or pd.isna(z):
                    continue
                factors[name] = round(float(z), 3)
                contributions[name] = round(float(z) * float(weight), 4)
            records.append({
                "code": code,
                "score": round(float(row.get("factor_score", row.get("score", 0.0))), 4),
                "industry": str(row.get("industry", "综合")),
                "factors": factors,
                "contributions": contributions,
            })
        return records

    def close(self) -> None:
        self.store.close()
