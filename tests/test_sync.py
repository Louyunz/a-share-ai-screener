# -*- coding: utf-8 -*-
"""每日行情同步的调度/原子替换/失败保留测试（不访问网络）。"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from backend.data.base import DataBundle, load_bundle, save_bundle
from backend.data.sample import generate_bundle
from backend.data.sync import SyncManager
from tests import cleanup, workdir


class DummyService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.reloads = 0

    def reload_data(self) -> DataBundle:
        self.reloads += 1
        return load_bundle(self.data_dir)


class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = workdir("sync")
        self.data_dir = self.dir / "a_share"
        self.bundle = generate_bundle(n_stocks=8, n_days=120, seed=13)
        save_bundle(self.bundle, self.data_dir)
        self.service = DummyService(self.data_dir)

    def tearDown(self) -> None:
        cleanup(self.dir)

    @staticmethod
    def _fresh_bundle(base: DataBundle) -> DataBundle:
        prices = base.prices.copy()
        close_raw = base.close_raw.copy()
        amount = base.amount.copy()
        benchmark = base.benchmark.copy()
        new_index = prices.index[-1] + pd.Timedelta(days=1)
        prices.loc[new_index] = prices.iloc[-1] * 1.02
        close_raw.loc[new_index] = close_raw.iloc[-1] * 1.02
        amount.loc[new_index] = amount.iloc[-1]
        benchmark.loc[new_index] = benchmark.iloc[-1] * 1.01
        meta = dict(base.meta)
        meta["as_of"] = new_index.strftime("%Y-%m-%d")
        return DataBundle(prices=prices.sort_index(), close_raw=close_raw.sort_index(),
                          amount=amount.sort_index(), fundamentals=base.fundamentals,
                          industry=base.industry, benchmark=benchmark.sort_index(),
                          universe=list(base.universe), meta=meta)

    def test_success_sync_writes_cache_and_reloads(self) -> None:
        fresh = self._fresh_bundle(self.bundle)

        class FakeAdapter:
            def load(self, start=None, end=None):
                return fresh

        manager = SyncManager(
            self.service, source="tushare", enabled=True,
            adapter_factory=lambda source, universe, limit: FakeAdapter())
        result = manager.run_once(reason="test")
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.service.reloads, 1)
        loaded = load_bundle(self.data_dir)
        self.assertEqual(loaded.as_of, result["as_of"])
        self.assertGreater(len(loaded.prices), len(self.bundle.prices))

    def test_failure_keeps_old_cache(self) -> None:
        class Broken:
            def load(self, start=None, end=None):
                raise RuntimeError("tushare 限频")

        manager = SyncManager(
            self.service, source="tushare", enabled=True,
            adapter_factory=lambda source, universe, limit: Broken())
        old_as_of = self.bundle.as_of
        result = manager.run_once(reason="test")
        self.assertEqual(result["status"], "error")
        self.assertIn("限频", result["error"])
        self.assertEqual(load_bundle(self.data_dir).as_of, old_as_of)
        self.assertEqual(self.service.reloads, 0)

    def test_non_tushare_source_is_rejected(self) -> None:
        manager = SyncManager(self.service, source="sample", enabled=False)
        result = manager.run_once(reason="test")
        self.assertEqual(result["status"], "error")
        self.assertIn("tushare", result["error"])


if __name__ == "__main__":
    unittest.main()
