# -*- coding: utf-8 -*-
"""每日行情同步：收盘后用 Tushare API 全量刷新缓存，并热加载到运行中的服务。

- 默认每个工作日 18:05（Asia/Shanghai）执行一次；
- 缓存先写入临时目录，再原子替换，失败时保留旧缓存；
- 同步成功后调用 ``AppService.reload_data()``，新请求立即使用新数据；
- 状态写入 ``data/sync_status.json``，管理端可查看/手动触发。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from ..config import (
    A_SHARE_DIR,
    DATA_SYNC_ENABLED,
    DATA_SYNC_HOUR,
    DATA_SYNC_MINUTE,
    DATA_SYNC_SOURCE,
    DATA_SYNC_TIMEZONE,
    DEFAULT_UNIVERSE,
)
from .base import DataBundle, DataError, load_bundle, save_bundle

logger = logging.getLogger(__name__)


class SyncManager:
    def __init__(self, service: Any, *,
                 source: str | None = None,
                 universe: str | None = None,
                 hour: int | None = None,
                 minute: int | None = None,
                 timezone_name: str | None = None,
                 enabled: bool | None = None,
                 adapter_factory: Callable[..., Any] | None = None) -> None:
        self.service = service
        self.data_dir = Path(service.data_dir)
        self.status_path = self.data_dir.parent / "sync_status.json"
        self.source = source or DATA_SYNC_SOURCE
        self.universe = universe or DEFAULT_UNIVERSE
        self.hour = DATA_SYNC_HOUR if hour is None else int(hour)
        self.minute = DATA_SYNC_MINUTE if minute is None else int(minute)
        self.timezone_name = timezone_name or DATA_SYNC_TIMEZONE
        self.enabled = DATA_SYNC_ENABLED if enabled is None else bool(enabled)
        self._adapter_factory = adapter_factory
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()
        self._running = False
        self._last_attempt_date = ""
        self._status: dict[str, Any] = self._read_status()

    # ------------------------------------------------------------------
    # 调度
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self.enabled:
            self._update_status(status="disabled", enabled=False,
                                message="DATA_SYNC_ENABLED=0")
            return
        if self.source != "tushare":
            self._update_status(status="disabled", enabled=False,
                                message=f"数据源 {self.source} 不支持 API 同步")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="daily-data-sync", daemon=True)
        self._thread.start()
        logger.info("每日行情同步已启动：工作日 %02d:%02d %s",
                    self.hour, self.minute, self.timezone_name)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _zone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone_name)
        except Exception:  # noqa: BLE001
            return ZoneInfo("Asia/Shanghai")

    def _target_time(self) -> clock_time:
        return clock_time(max(0, min(23, self.hour)), max(0, min(59, self.minute)))

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = datetime.now(self._zone())
                today = now.date().isoformat()
                target = datetime.combine(now.date(), self._target_time(), tzinfo=now.tzinfo)
                if (self.enabled and now.weekday() < 5 and now >= target
                        and not self._synced_after(target)
                        and self._last_attempt_date != today):
                    self._last_attempt_date = today
                    self.run_once(reason="scheduled")
            except Exception as exc:  # noqa: BLE001 - 调度线程不能退出
                logger.warning("每日同步调度异常：%s", exc)
            self._stop_event.wait(30)

    def _synced_after(self, target: datetime) -> bool:
        """今天收盘后是否已经成功同步过（手动同步在收盘前完成时，收盘后仍要再同步）。"""
        raw = str(self._status.get("last_success") or "")
        if not raw:
            return False
        try:
            success = datetime.fromisoformat(raw)
        except ValueError:
            return False
        if success.tzinfo is None:
            success = success.replace(tzinfo=self._zone())
        return success >= target

    def trigger(self) -> dict[str, Any]:
        """管理端手动触发；已在运行时直接返回当前状态。"""
        if not self.enabled:
            return {"status": "disabled", "message": "每日同步已关闭"}
        if self._running:
            return self.status()
        threading.Thread(target=self.run_once, kwargs={"reason": "manual"},
                         name="manual-data-sync", daemon=True).start()
        return {"status": "running", "message": "同步已开始"}

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def run_once(self, reason: str = "manual") -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            return self.status()
        self._running = True
        started = time.time()
        self._update_status(status="running", reason=reason, enabled=self.enabled,
                            last_attempt=datetime.now(self._zone()).isoformat(timespec="seconds"),
                            error="")
        try:
            if self.source != "tushare":
                raise DataError(f"当前同步只支持 tushare，收到 source={self.source}")
            old = load_bundle(self.data_dir)
            if len(old.prices) == 0:
                raise DataError("现有缓存为空，无法确定同步起点")
            start = old.prices.index[0].strftime("%Y%m%d")
            limit = len(old.universe) or None
            adapter = self._build_adapter(limit)
            bundle = adapter.load(start=start, end=date.today().strftime("%Y%m%d"))
            self._validate(bundle)
            self._save_atomic(bundle)
            self.service.reload_data()
            duration = round(time.time() - started, 1)
            self._update_status(
                status="success", reason=reason, enabled=self.enabled,
                last_success=datetime.now(self._zone()).isoformat(timespec="seconds"),
                as_of=bundle.as_of, universe=len(bundle.universe),
                rows=len(bundle.prices), duration_seconds=duration, error="",
            )
            logger.info("行情同步完成：%s 只 × %s 个交易日，as_of=%s，耗时 %.1fs",
                        len(bundle.universe), len(bundle.prices), bundle.as_of, duration)
            return self.status()
        except Exception as exc:  # noqa: BLE001
            self._update_status(
                status="error", reason=reason, enabled=self.enabled,
                error=str(exc)[:500], duration_seconds=round(time.time() - started, 1),
            )
            logger.warning("行情同步失败：%s", exc)
            return self.status()
        finally:
            self._running = False
            self._run_lock.release()

    def _build_adapter(self, limit: int | None) -> Any:
        if self._adapter_factory is not None:
            return self._adapter_factory(self.source, self.universe, limit)
        from .tushare_adapter import TushareAdapter
        return TushareAdapter(universe=self.universe, limit=limit)

    @staticmethod
    def _validate(bundle: DataBundle) -> None:
        if not isinstance(bundle, DataBundle) or len(bundle.prices) == 0:
            raise DataError("同步返回空行情")
        if len(bundle.universe) == 0:
            raise DataError("同步返回空股票池")
        if not all(column in bundle.prices.columns for column in bundle.universe):
            raise DataError("同步行情列与股票池不一致")

    def _save_atomic(self, bundle: DataBundle) -> None:
        tmp_dir = self.data_dir.parent / (self.data_dir.name + ".sync_tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            save_bundle(bundle, tmp_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            names = {path.name for path in tmp_dir.iterdir() if path.is_file()}
            # 同一份数据可能因 pyarrow 可用性切换 parquet/csv；先清掉另一后缀，避免 load_bundle 读到旧格式
            for name in names:
                stem = Path(name).stem
                for suffix in (".parquet", ".csv"):
                    stale = self.data_dir / (stem + suffix)
                    if stale.name not in names and stale.exists():
                        stale.unlink()
            # meta 最后替换：前面任何失败都不会让旧 meta 指向不完整的新文件
            ordered = sorted(names, key=lambda name: name == "meta.json")
            for name in ordered:
                os.replace(tmp_dir / name, self.data_dir / name)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def _read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _update_status(self, **updates: Any) -> None:
        self._status.update(updates)
        self._status.setdefault("source", self.source)
        self._status.setdefault("schedule", f"工作日 {self.hour:02d}:{self.minute:02d}")
        self._status.setdefault("timezone", self.timezone_name)
        self._status["next_run"] = self._next_run_text()
        self._write_status()

    def _write_status(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.status_path)

    def _next_run_text(self) -> str:
        now = datetime.now(self._zone())
        target = datetime.combine(now.date(), self._target_time(), tzinfo=now.tzinfo)
        if now >= target or now.weekday() >= 5:
            target += timedelta(days=1)
            while target.weekday() >= 5:
                target += timedelta(days=1)
        return target.isoformat(timespec="minutes")

    def status(self) -> dict[str, Any]:
        current = dict(self._status)
        current.update({"enabled": self.enabled, "running": self._running,
                        "source": self.source})
        return current
