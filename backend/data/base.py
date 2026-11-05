# -*- coding: utf-8 -*-
"""数据包与数据端口协议。

DataBundle 是引擎、服务与适配器之间唯一的数据交换格式；字段语义见 docs/CONTRACT.md。

缓存格式：有 pyarrow 时写 parquet（生产推荐）；否则自动改用 csv（无 pyarrow 环境可跑）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class DataError(RuntimeError):
    """数据缺失或格式非法。"""


@dataclass
class DataBundle:
    prices: pd.DataFrame       # 前复权收盘价（宽表：行=交易日，列=代码）
    close_raw: pd.DataFrame    # 未复权收盘价（判断涨跌停）
    amount: pd.DataFrame       # 成交额（元）
    fundamentals: pd.DataFrame  # index=代码，含 pe/pb/roe/total_mv/float_mv 等
    industry: pd.DataFrame      # index=代码，列 industry
    benchmark: pd.Series        # 基准指数收盘价
    universe: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def slice(self, start: str | None = None, end: str | None = None) -> "DataBundle":
        """按日期切片，返回新的 DataBundle（用于回测/调优的子区间）。"""
        prices = self.prices.loc[start:end] if start or end else self.prices
        close_raw = self.close_raw.loc[prices.index]
        amount = self.amount.loc[prices.index]
        benchmark = self.benchmark.loc[start:end] if start or end else self.benchmark
        return DataBundle(
            prices=prices,
            close_raw=close_raw,
            amount=amount,
            fundamentals=self.fundamentals,
            industry=self.industry,
            benchmark=benchmark,
            universe=list(self.universe),
            meta=dict(self.meta),
        )

    @property
    def as_of(self) -> str:
        if self.meta.get("as_of"):
            return str(self.meta["as_of"])
        return self.prices.index[-1].strftime("%Y-%m-%d") if len(self.prices) else ""


class IDataPort(Protocol):
    """数据源适配器协议（AkShare / Tushare / Sample 均实现该协议）。"""

    def load(self) -> DataBundle: ...


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


PARQUET_SUFFIX = ".parquet"
CSV_SUFFIX = ".csv"


def _write_frame(frame: pd.DataFrame, base: Path) -> str:
    """写 DataFrame：优先 parquet，无 pyarrow 时写 csv；返回实际格式标记。"""
    if _has_parquet():
        frame.to_parquet(base.with_suffix(PARQUET_SUFFIX))
        return "parquet"
    frame.to_csv(base.with_suffix(CSV_SUFFIX), encoding="utf-8-sig")
    return "csv"


def _read_frame(base: Path, *, index_is_date: bool = False) -> pd.DataFrame:
    parquet_path = base.with_suffix(PARQUET_SUFFIX)
    csv_path = base.with_suffix(CSV_SUFFIX)
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ValueError, OSError):
            if not csv_path.exists():
                raise DataError(f"缓存文件无法读取：{parquet_path}")
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0,
                           parse_dates=index_is_date, encoding="utf-8-sig")
    raise DataError(f"缓存文件缺失：{parquet_path} / {csv_path}")


def load_bundle(data_dir: Path) -> DataBundle:
    """从本地缓存目录读取 DataBundle；文件缺失时抛出明确提示。"""
    root = Path(data_dir)
    needed_bases = ["prices", "close_raw", "amount", "fundamentals", "benchmark"]
    missing = [str(root / base) for base in needed_bases
               if not (root / f"{base}.parquet").exists() and not (root / f"{base}.csv").exists()]
    for extra in ("industry.csv", "universe.csv"):
        if not (root / extra).exists():
            missing.append(str(root / extra))
    if missing:
        raise DataError(
            "A 股数据缓存缺失：\n  " + "\n  ".join(missing) +
            "\n请先运行：python -m backend.data.build_cache --source sample"
        )

    prices = _read_frame(root / "prices", index_is_date=True)
    close_raw = _read_frame(root / "close_raw", index_is_date=True)
    amount = _read_frame(root / "amount", index_is_date=True)
    fundamentals = _read_frame(root / "fundamentals")
    industry = pd.read_csv(root / "industry.csv", index_col=0, encoding="utf-8-sig")
    benchmark_frame = _read_frame(root / "benchmark", index_is_date=True)
    benchmark = benchmark_frame.iloc[:, 0] if benchmark_frame.shape[1] else benchmark_frame.iloc[:, 0]
    benchmark.index = pd.to_datetime(benchmark.index)
    universe = pd.read_csv(root / "universe.csv", header=None,
                           encoding="utf-8-sig")[0].astype(str).tolist()

    for frame in (prices, close_raw, amount):
        frame.index = pd.to_datetime(frame.index)
    meta = _read_json(root / "meta.json")
    meta.setdefault("cache_format", "parquet" if _has_parquet() else "csv")
    return DataBundle(
        prices=prices.sort_index(),
        close_raw=close_raw.sort_index(),
        amount=amount.sort_index(),
        fundamentals=fundamentals,
        industry=industry,
        benchmark=benchmark.sort_index(),
        universe=list(universe),
        meta=meta,
    )


def save_bundle(bundle: DataBundle, data_dir: Path) -> None:
    """把 DataBundle 写入本地缓存目录（供 build_cache 与测试使用）。"""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    formats = {
        "prices": _write_frame(bundle.prices, root / "prices"),
        "close_raw": _write_frame(bundle.close_raw, root / "close_raw"),
        "amount": _write_frame(bundle.amount, root / "amount"),
        "fundamentals": _write_frame(bundle.fundamentals, root / "fundamentals"),
        "benchmark": _write_frame(bundle.benchmark.to_frame(name="benchmark"), root / "benchmark"),
    }
    bundle.industry.to_csv(root / "industry.csv", encoding="utf-8-sig")
    pd.Series(bundle.universe, dtype="string").to_csv(
        root / "universe.csv", index=False, header=False, encoding="utf-8-sig")
    meta = dict(bundle.meta)
    meta["cache_format"] = formats["prices"]
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
