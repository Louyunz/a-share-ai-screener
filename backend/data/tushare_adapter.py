# -*- coding: utf-8 -*-
"""Tushare 数据适配器（辅助数据源，需 TUSHARE_TOKEN）。

说明：仅在 `python -m backend.data.build_cache --source tushare` 时联网拉取；
未安装 tushare 或未配置 token 时抛出明确错误。
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .base import DataBundle

logger = logging.getLogger(__name__)

INDEX_CODE = "000300.SH"


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

    def _universe(self, trade_date: str) -> list[str]:
        frames = []
        for idx in ("000300.SH", "000905.SH"):
            try:
                df = self.pro.index_weight(index_code=idx, start_date="20240101", end_date=trade_date)
                if df is not None and not df.empty:
                    last_date = df["trade_date"].max()
                    frames.append(df[df["trade_date"] == last_date]["con_code"].astype(str).tolist())
            except Exception as exc:  # noqa: BLE001
                logger.warning("成分股 %s 获取失败: %s", idx, exc)
        if not frames:
            basic = self.pro.stock_basic(exchange="", list_status="L", fields="ts_code")
            codes = basic["ts_code"].astype(str).tolist()
            return codes[: self.limit or 800]
        codes = sorted(set(c for frame in frames for c in frame))
        return codes[: self.limit] if self.limit else codes

    def load(self, start: str = "20190101", end: str | None = None) -> DataBundle:
        end = (end or date.today().strftime("%Y%m%d")).replace("-", "")
        start = start.replace("-", "")
        codes = self._universe(end)
        if not codes:
            raise RuntimeError("Tushare 股票池为空")

        prices, raw, amounts, ok = {}, {}, {}, []
        for i, code in enumerate(codes, 1):
            try:
                df = self.pro.daily(ts_code=code, start_date=start, end_date=end)
                if df is None or df.empty:
                    continue
                df = df.set_index(pd.to_datetime(df["trade_date"])).sort_index()
                prices[code] = df["close"].astype(float)
                raw[code] = df["close"].astype(float)
                amounts[code] = df["amount"].astype(float) * 1000.0  # 千元 → 元
                ok.append(code)
            except Exception as exc:  # noqa: BLE001
                logger.warning("行情失败 %s: %s", code, exc)
            if i % 100 == 0:
                logger.info("Tushare 已拉取 %d/%d", i, len(codes))
        if not ok:
            raise RuntimeError("Tushare 行情全部失败，请检查 token 与积分权限")

        prices_df = pd.DataFrame(prices).sort_index()
        close_raw = pd.DataFrame(raw).reindex(prices_df.index)
        amount_df = pd.DataFrame(amounts).reindex(prices_df.index).fillna(0.0)

        fundamentals = self._fundamentals(ok)
        industry = fundamentals[["industry"]].copy()
        try:
            idx = self.pro.index_daily(ts_code=INDEX_CODE, start_date=start, end_date=end)
            idx = idx.set_index(pd.to_datetime(idx["trade_date"])).sort_index()
            benchmark = idx["close"].astype(float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("沪深300 获取失败（%s），用等权收益代理", exc)
            benchmark = 3000.0 * (1.0 + prices_df.pct_change().mean(axis=1).fillna(0.0)).cumprod()
        benchmark = benchmark.reindex(prices_df.index).ffill()

        meta = {"source": "tushare", "as_of": prices_df.index[-1].strftime("%Y-%m-%d"),
                "n_stocks": len(ok), "universe": self.universe,
                "note": "Tushare 拉取，字段口径见 docs/CONTRACT.md"}
        return DataBundle(prices=prices_df, close_raw=close_raw, amount=amount_df,
                          fundamentals=fundamentals, industry=industry,
                          benchmark=benchmark, universe=ok, meta=meta)

    def _fundamentals(self, codes: list[str]) -> pd.DataFrame:
        rows: dict[str, dict[str, Any]] = {}
        for code in codes:
            row: dict[str, Any] = {"industry": "综合"}
            try:
                basic = self.pro.stock_basic(ts_code=code, fields="ts_code,industry")
                if basic is not None and not basic.empty:
                    row["industry"] = str(basic.iloc[0]["industry"] or "综合")
            except Exception:  # noqa: BLE001
                pass
            try:
                db = self.pro.daily_basic(ts_code=code, fields="trade_date,pe,pb,total_mv,circ_mv")
                if db is not None and not db.empty:
                    last = db.sort_values("trade_date").iloc[-1]
                    for src, dst in (("pe", "pe"), ("pb", "pb"), ("total_mv", "total_mv"),
                                     ("circ_mv", "float_mv")):
                        if pd.notna(last.get(src)):
                            row[dst] = float(last[src]) * (10000.0 if "mv" in src else 1.0)
                    if row.get("pe") and row["pe"] > 0:
                        row["ep"] = 1.0 / row["pe"]
                    if row.get("pb") and row["pb"] > 0:
                        row["bp"] = 1.0 / row["pb"]
            except Exception:  # noqa: BLE001
                pass
            try:
                fin = self.pro.fina_indicator(ts_code=code)
                if fin is not None and not fin.empty:
                    last = fin.sort_values("end_date").iloc[-1]
                    if pd.notna(last.get("roe")):
                        row["roe"] = float(last["roe"]) / 100.0
                    if pd.notna(last.get("or_yoy")):
                        row["revenue_growth"] = float(last["or_yoy"]) / 100.0
            except Exception:  # noqa: BLE001
                pass
            rows[code] = row
        df = pd.DataFrame.from_dict(rows, orient="index")
        for col in ("ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
                    "pe", "pb", "total_mv", "float_mv"):
            if col not in df.columns:
                df[col] = np.nan
        return df
