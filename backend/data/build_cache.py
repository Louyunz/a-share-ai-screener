# -*- coding: utf-8 -*-
"""A 股数据缓存构建 CLI。

用法：
    python -m backend.data.build_cache --source sample --stocks 150 --days 750
    python -m backend.data.build_cache --source akshare --universe csi800 --limit 200
    python -m backend.data.build_cache --source tushare --limit 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..config import A_SHARE_DIR
from .base import DataBundle, save_bundle


def build(source: str, out_dir: Path, *, stocks: int, days: int,
          universe: str, limit: int | None, start: str, end: str | None) -> DataBundle:
    if source == "sample":
        from .sample import generate_bundle
        bundle = generate_bundle(n_stocks=stocks, n_days=days)
    elif source == "akshare":
        from .akshare_adapter import AkShareAdapter
        bundle = AkShareAdapter(universe=universe, limit=limit).load(start=start, end=end)
    elif source == "tushare":
        from .tushare_adapter import TushareAdapter
        bundle = TushareAdapter(universe=universe, limit=limit).load(start=start, end=end)
    else:
        raise SystemExit(f"未知数据源: {source}（可选 sample / akshare / tushare）")
    save_bundle(bundle, out_dir)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 A 股数据缓存")
    parser.add_argument("--source", default="sample", choices=["sample", "akshare", "tushare"])
    parser.add_argument("--out", default=str(A_SHARE_DIR), help="缓存输出目录")
    parser.add_argument("--stocks", type=int, default=150, help="示例数据股票数")
    parser.add_argument("--days", type=int, default=750, help="示例数据交易日数")
    parser.add_argument("--universe", default="csi800", help="真实数据股票池：csi800/hs300/zz500")
    parser.add_argument("--limit", type=int, default=None, help="真实数据最多拉取股票数")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    print(f"[build_cache] source={args.source} out={args.out}")
    bundle = build(args.source, Path(args.out), stocks=args.stocks, days=args.days,
                   universe=args.universe, limit=args.limit, start=args.start, end=args.end)
    print(f"[build_cache] 完成：{len(bundle.universe)} 只股票 × {len(bundle.prices)} 交易日，"
          f"区间 {bundle.prices.index[0].date()} ~ {bundle.prices.index[-1].date()}，"
          f"耗时 {time.time() - started:.1f}s")
    print(f"[build_cache] 数据源={bundle.meta.get('source')} as_of={bundle.as_of}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
