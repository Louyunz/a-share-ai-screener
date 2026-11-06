# -*- coding: utf-8 -*-
"""Tushare 数据适配器（真实数据源，需 TUSHARE_TOKEN）。

低权限 token 适配要点：
- ``index_weight`` / ``fina_indicator`` 可能无权限，第一次失败后不再重复请求；
- ``stock_basic`` 与 ``daily_basic`` 批量接口频率很低（部分 token 为 1 次/分钟），
  因此只调用一次并缓存，用批量行情选择股票池、取估值/市值；
- 个股行情按股票逐个请求；数据源只在 ``build_cache`` 时联网。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .base import DataBundle

logger = logging.getLogger(__name__)

INDEX_CODE = "000300.SH"

# 低权限 token 的 stock_basic 频率可能极低（1 次/小时）；
# 批量接口不可用时，用一组流动性较好的 A 股核心标的作为真实数据池。
FALLBACK_UNIVERSE = [
    "600519.SH", "600036.SH", "601318.SH", "600030.SH", "600276.SH",
    "600887.SH", "601166.SH", "600000.SH", "601888.SH", "600900.SH",
    "601012.SH", "600585.SH", "600309.SH", "601088.SH", "600028.SH",
    "000001.SZ", "000333.SZ", "000651.SZ", "000858.SZ", "002415.SZ",
    "002594.SZ", "300750.SZ", "300059.SZ", "300015.SZ", "002475.SZ",
    "000725.SZ", "002304.SZ", "000568.SZ", "002714.SZ", "300124.SZ",
]


def _ts_code(code: str) -> str:
    code = str(code).split(".")[0].zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


class TushareAdapter:
    """通过 Tushare Pro 拉取 A 股数据并组装为 DataBundle。"""

    def __init__(self, universe: str = "csi800", limit: int | None = None) -> None:
        self.universe = universe
        self.limit = limit
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            raise RuntimeError("未配置 TUSHARE_TOKEN，无法使用 Tushare 数据源")
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - 依赖真实环境
            raise RuntimeError("未安装 tushare：pip install tushare") from exc
        self.pro = ts.pro_api(token)
        self._basic: pd.DataFrame | None = None
        self._daily_basic: pd.DataFrame | None = None
        self._index_weight_available = True
        self._fina_available = True
        self.latest_trade_date = ""

    # ------------------------------------------------------------------
    # 低频接口：只调一次并缓存
    # ------------------------------------------------------------------
    def _load_basic(self) -> pd.DataFrame:
        if self._basic is None:
            try:
                df = self.pro.stock_basic(
                    exchange="", list_status="L",
                    fields="ts_code,name,industry,market,list_date")
                self._basic = df if df is not None else pd.DataFrame()
            except Exception as exc:  # noqa: BLE001
                logger.warning("stock_basic 获取失败：%s", exc)
                self._basic = pd.DataFrame()
        return self._basic

    def _latest_date(self, end: str) -> str:
        """用平安银行日线确定数据最新交易日，避免用非交易日批量报价。"""
        try:
            df = self.pro.daily(ts_code="000001.SZ", start_date=end[:4] + "0101", end_date=end)
            if df is not None and not df.empty:
                return str(df["trade_date"].max())
        except Exception as exc:  # noqa: BLE001
            logger.warning("探测最新交易日失败：%s", exc)
        return end

    def _load_daily_basic(self) -> pd.DataFrame:
        if self._daily_basic is None:
            try:
                df = self.pro.daily_basic(
                    trade_date=self.latest_trade_date,
                    fields="ts_code,pe,pb,total_mv,circ_mv")
                self._daily_basic = df if df is not None else pd.DataFrame()
            except Exception as exc:  # noqa: BLE001
                logger.warning("daily_basic 批量获取失败：%s", exc)
                self._daily_basic = pd.DataFrame()
        return self._daily_basic

    # ------------------------------------------------------------------
    # 股票池
    # ------------------------------------------------------------------
    def _universe(self) -> list[str]:
        frames = []
        if self._index_weight_available:
            for idx in ("000300.SH", "000905.SH"):
                try:
                    df = self.pro.index_weight(
                        index_code=idx, start_date="20240101",
                        end_date=self.latest_trade_date)
                    if df is not None and not df.empty:
                        last_date = df["trade_date"].max()
                        frames.append(df[df["trade_date"] == last_date]["con_code"].astype(str).tolist())
                except Exception as exc:  # noqa: BLE001
                    logger.info("指数成分无权限，改用批量行情选池：%s", exc)
                    self._index_weight_available = False
                    break
        codes = sorted(set(code for frame in frames for code in frame))
        if not codes:
            basic = self._load_basic()
            if not basic.empty and "ts_code" in basic.columns:
                basic = basic.copy()
                basic = basic[~basic["ts_code"].astype(str).str.endswith(".BJ")]
                if "list_date" in basic.columns:
                    basic = basic[basic["list_date"].astype(str) <= "20150101"]
                daily = self._load_daily_basic()
                if not daily.empty and "total_mv" in daily.columns:
                    ranked = daily.sort_values("total_mv", ascending=False)["ts_code"].astype(str).tolist()
                    available = set(basic["ts_code"].astype(str))
                    codes = [code for code in ranked if code in available]
                if not codes:
                    preferred = basic[basic["ts_code"].astype(str).str.match(r"^(60|00|30)")]
                    codes = preferred["ts_code"].astype(str).tolist()
        if not codes:
            logger.info("批量接口不可用，使用内置核心 A 股标的池")
            codes = list(FALLBACK_UNIVERSE)
        if not codes:
            raise RuntimeError("Tushare 股票池为空")
        return codes[: self.limit] if self.limit else codes[:50]

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def load(self, start: str = "20190101", end: str | None = None) -> DataBundle:
        end = (end or date.today().strftime("%Y%m%d")).replace("-", "")
        start = start.replace("-", "")
        self.latest_trade_date = self._latest_date(end)
        codes = self._universe()
        if not codes:
            raise RuntimeError("Tushare 股票池为空")

        prices, raw, amounts, ok = {}, {}, {}, []
        for i, code in enumerate(codes, 1):
            for attempt in (1, 2):
                try:
                    df = self.pro.daily(ts_code=code, start_date=start, end_date=self.latest_trade_date)
                    if df is None or df.empty:
                        break
                    df = df.set_index(pd.to_datetime(df["trade_date"])).sort_index()
                    prices[code] = df["close"].astype(float)
                    raw[code] = df["close"].astype(float)
                    amounts[code] = df["amount"].astype(float) * 1000.0  # 千元 → 元
                    ok.append(code)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("行情失败 %s（第 %d 次）：%s", code, attempt, exc)
                    time.sleep(0.6)
            if i % 10 == 0:
                logger.info("Tushare 已拉取 %d/%d", i, len(codes))
            time.sleep(0.08)
        if not ok:
            raise RuntimeError("Tushare 行情全部失败，请检查 token 与积分权限")

        prices_df = pd.DataFrame(prices).sort_index()
        close_raw = pd.DataFrame(raw).reindex(prices_df.index)
        amount_df = pd.DataFrame(amounts).reindex(prices_df.index).fillna(0.0)

        fundamentals = self._fundamentals(ok)
        industry = fundamentals[["industry"]].copy()
        try:
            idx = self.pro.index_daily(
                ts_code=INDEX_CODE, start_date=start, end_date=self.latest_trade_date)
            idx = idx.set_index(pd.to_datetime(idx["trade_date"])).sort_index()
            benchmark = idx["close"].astype(float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("沪深300 获取失败（%s），用等权收益代理", exc)
            benchmark = 3000.0 * (1.0 + prices_df.pct_change().mean(axis=1).fillna(0.0)).cumprod()
        benchmark = benchmark.reindex(prices_df.index).ffill()

        meta = {"source": "tushare", "as_of": prices_df.index[-1].strftime("%Y-%m-%d"),
                "n_stocks": len(ok), "universe": self.universe,
                "note": "Tushare 真实行情；指数成分/财务指标权限不足时自动降级为核心标的池与价格因子"}
        return DataBundle(prices=prices_df, close_raw=close_raw, amount=amount_df,
                          fundamentals=fundamentals, industry=industry,
                          benchmark=benchmark, universe=ok, meta=meta)

    # ------------------------------------------------------------------
    # 基本面：批量字段优先，财务指标权限不足时只跳过该部分
    # ------------------------------------------------------------------
    def _fundamentals(self, codes: list[str]) -> pd.DataFrame:
        basic = self._load_basic()
        industries = {}
        if not basic.empty and {"ts_code", "industry"} <= set(basic.columns):
            for row in basic[["ts_code", "industry"]].itertuples(index=False):
                industries[str(row.ts_code)] = str(row.industry or "综合")
        daily = self._load_daily_basic()
        daily_map: dict[str, dict[str, float]] = {}
        if not daily.empty:
            for row in daily.itertuples(index=False):
                item: dict[str, float] = {}
                for src, dst in (("pe", "pe"), ("pb", "pb"), ("total_mv", "total_mv"),
                                 ("circ_mv", "float_mv")):
                    value = getattr(row, src, None)
                    if pd.notna(value):
                        item[dst] = float(value) * (10000.0 if "mv" in src else 1.0)
                daily_map[str(row.ts_code)] = item

        fina_map: dict[str, dict[str, float]] = {}
        if self._fina_available and codes:
            try:
                fin = self.pro.fina_indicator(ts_code=codes[0])
                if fin is not None and not fin.empty:
                    for code in codes:
                        try:
                            data = self.pro.fina_indicator(ts_code=code)
                            if data is None or data.empty:
                                continue
                            last = data.sort_values("end_date").iloc[-1]
                            item = {}
                            if pd.notna(last.get("roe")):
                                item["roe"] = float(last["roe"]) / 100.0
                            if pd.notna(last.get("or_yoy")):
                                item["revenue_growth"] = float(last["or_yoy"]) / 100.0
                            fina_map[code] = item
                        except Exception:  # noqa: BLE001
                            time.sleep(0.2)
            except Exception as exc:  # noqa: BLE001
                logger.info("fina_indicator 无权限，跳过财务质量因子：%s", exc)
                self._fina_available = False

        rows: dict[str, dict[str, Any]] = {}
        for code in codes:
            row: dict[str, Any] = {"industry": industries.get(code, "综合")}
            row.update(fina_map.get(code, {}))
            row.update(daily_map.get(code, {}))
            if row.get("pe") and row["pe"] > 0:
                row["ep"] = 1.0 / row["pe"]
            if row.get("pb") and row["pb"] > 0:
                row["bp"] = 1.0 / row["pb"]
            rows[code] = row
        df = pd.DataFrame.from_dict(rows, orient="index")
        for col in ("ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
                    "pe", "pb", "total_mv", "float_mv"):
            if col not in df.columns:
                df[col] = np.nan
        return df
