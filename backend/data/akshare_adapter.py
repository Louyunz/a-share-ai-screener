# -*- coding: utf-8 -*-
"""AkShare 数据适配器（主数据源，免费、无需 token）。

说明：本适配器在服务运行期不直接调用；仅在 `python -m backend.data.build_cache --source akshare`
时联网拉取并落盘。字段口径见 docs/CONTRACT.md 第 1 节。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import DataBundle

logger = logging.getLogger(__name__)

INDEX_CODES = {"hs300": "000300", "zz500": "000905", "csi800": "000300,000905"}
INDUSTRY_NAMES = [
    "食品饮料", "医药生物", "电子", "计算机", "电力设备",
    "银行", "非银金融", "机械设备", "基础化工", "汽车",
]


def _to_code6(code: str) -> str:
    return str(code).split(".")[0].zfill(6)


def _with_suffix(code: str) -> str:
    code6 = _to_code6(code)
    if code6.startswith(("6", "9")):
        return f"{code6}.SH"
    if code6.startswith(("4", "8")):
        return f"{code6}.BJ"
    return f"{code6}.SZ"


class AkShareAdapter:
    """通过 AkShare 拉取 A 股数据并组装为 DataBundle。"""

    def __init__(self, universe: str = "csi800", limit: int | None = None) -> None:
        self.universe = universe
        self.limit = limit

    # ------------------------------------------------------------------
    # 股票池
    # ------------------------------------------------------------------
    def _index_members(self, index_code: str) -> list[str]:
        import akshare as ak

        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
            col = next((c for c in ("成分券代码", "品种代码", "证券代码") if c in df.columns), None)
            if col is None:
                raise ValueError(f"无法识别成分股列: {list(df.columns)}")
            return [_with_suffix(c) for c in df[col].astype(str).tolist()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("指数 %s 成分股获取失败（%s），回退到全市场市值前 300", index_code, exc)
            spot = ak.stock_zh_a_spot_em()
            spot = spot.sort_values("总市值", ascending=False).head(300)
            return [_with_suffix(c) for c in spot["代码"].astype(str).tolist()]

    def universe_codes(self) -> list[str]:
        codes: list[str] = []
        for part in INDEX_CODES.get(self.universe, INDEX_CODES["csi800"]).split(","):
            codes.extend(self._index_members(part))
        unique = sorted(set(codes))
        return unique[: self.limit] if self.limit else unique

    # ------------------------------------------------------------------
    # 行情
    # ------------------------------------------------------------------
    def _daily_one(self, code: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        import akshare as ak

        code6 = _to_code6(code)
        raw = ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start.replace("-", ""),
                                 end_date=end.replace("-", ""), adjust="")
        qfq = ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start.replace("-", ""),
                                 end_date=end.replace("-", ""), adjust="qfq")
        frames = []
        for df, name in ((raw, "close_raw"), (qfq, "prices")):
            if df is None or df.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            df = df.set_index(pd.to_datetime(df["日期"]))
            frames.append(df[["收盘"]].rename(columns={"收盘": name}))
        close_raw, prices = frames
        amount = raw.set_index(pd.to_datetime(raw["日期"]))[["成交额"]].rename(columns={"成交额": "amount"})
        return close_raw, prices, amount

    def load(self, start: str = "2019-01-01", end: str | None = None) -> DataBundle:
        from datetime import date

        end = end or date.today().strftime("%Y-%m-%d")
        codes = self.universe_codes()
        if not codes:
            raise RuntimeError("AkShare 股票池为空")

        logger.info("AkShare 拉取 %d 只股票行情：%s ~ %s", len(codes), start, end)
        raw_frames, qfq_frames, amount_frames, ok_codes = {}, {}, {}, []
        for i, code in enumerate(codes, 1):
            try:
                close_raw, prices, amount = self._daily_one(code, start, end)
                if close_raw.empty:
                    continue
                raw_frames[code] = close_raw["close_raw"]
                qfq_frames[code] = prices["prices"]
                amount_frames[code] = amount["amount"]
                ok_codes.append(code)
            except Exception as exc:  # noqa: BLE001
                logger.warning("行情拉取失败 %s: %s", code, exc)
            if i % 50 == 0:
                logger.info("已拉取 %d/%d", i, len(codes))

        if not ok_codes:
            raise RuntimeError("AkShare 行情全部失败，请检查网络或接口是否变更")

        prices = pd.DataFrame(qfq_frames).sort_index()
        close_raw = pd.DataFrame(raw_frames).reindex(prices.index)
        amount = pd.DataFrame(amount_frames).reindex(prices.index).fillna(0.0)

        # ---- 财务与估值（以 stock_a_indicator_lg 为主，缺失字段留 NaN）----
        fundamentals = self._load_fundamentals(ok_codes)
        if "industry" not in fundamentals.columns:
            fundamentals["industry"] = "综合"

        industry = fundamentals[["industry"]].copy()

        # ---- 基准指数（沪深 300）----
        import akshare as ak

        try:
            bench_df = ak.stock_zh_index_daily_em(symbol="sh000300")
            bench_df = bench_df.set_index(pd.to_datetime(bench_df["date"]))
            benchmark = bench_df["close"].astype(float).loc[start:end]
        except Exception as exc:  # noqa: BLE001
            logger.warning("沪深300 拉取失败（%s），用成分股等权收益代理", exc)
            bench_ret = prices.pct_change().mean(axis=1).fillna(0.0)
            benchmark = 3000.0 * (1.0 + bench_ret).cumprod()

        benchmark = benchmark.reindex(prices.index).ffill()
        meta = {"source": "akshare", "as_of": prices.index[-1].strftime("%Y-%m-%d"),
                "n_stocks": len(ok_codes), "universe": self.universe,
                "note": "AkShare 拉取，字段口径见 docs/CONTRACT.md"}
        return DataBundle(prices=prices, close_raw=close_raw, amount=amount,
                          fundamentals=fundamentals, industry=industry,
                          benchmark=benchmark, universe=ok_codes, meta=meta)

    def _load_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """逐只读取估值指标；财务质量指标缺失时留 NaN，由因子引擎按中性值处理。"""
        import akshare as ak

        rows: dict[str, dict[str, Any]] = {}
        for code in codes:
            code6 = _to_code6(code)
            row: dict[str, Any] = {"industry": "综合"}
            try:
                ind = ak.stock_a_indicator_lg(symbol=code6).dropna(subset=["trade_date"])
                last = ind.iloc[-1]
                for src, dst in (("pe", "pe"), ("pb", "pb"), ("total_mv", "total_mv")):
                    if src in last and pd.notna(last[src]):
                        row[dst] = float(last[src])
                if "total_mv" in row:
                    row["float_mv"] = row["total_mv"] * 0.8
                if row.get("pe") and row["pe"] > 0:
                    row["ep"] = 1.0 / row["pe"]
                if row.get("pb") and row["pb"] > 0:
                    row["bp"] = 1.0 / row["pb"]
            except Exception as exc:  # noqa: BLE001
                logger.debug("估值指标失败 %s: %s", code, exc)
            try:
                fin = ak.stock_financial_analysis_indicator(symbol=code6)
                if fin is not None and not fin.empty:
                    last = fin.iloc[-1]
                    if "净资产收益率(%)" in fin.columns and pd.notna(last.get("净资产收益率(%)")):
                        row["roe"] = float(last["净资产收益率(%)"]) / 100.0
                    for col in ("主营业务收入增长率(%)", "营业收入增长率(%)"):
                        if col in fin.columns and pd.notna(last.get(col)):
                            row["revenue_growth"] = float(last[col]) / 100.0
                            break
            except Exception as exc:  # noqa: BLE001
                logger.debug("财务指标失败 %s: %s", code, exc)
            rows[code] = row
        df = pd.DataFrame.from_dict(rows, orient="index")
        for col in ("ep", "bp", "roe", "ocf_to_profit", "revenue_growth",
                    "pe", "pb", "total_mv", "float_mv"):
            if col not in df.columns:
                df[col] = np.nan
        df["ocf_to_profit"] = df.get("ocf_to_profit", np.nan)
        return df
