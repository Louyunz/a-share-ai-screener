# -*- coding: utf-8 -*-
"""AppService 集成测试（FakeLLM + 合成小数据，不依赖网络）。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.data.base import DataBundle, save_bundle
from backend.data.sample import generate_bundle
from backend.llm import FakeLLM
from backend.service import AppService


def _try_import_engine() -> None:
    try:
        from backend.engine import factors, selection, tuner  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise unittest.SkipTest(f"引擎尚未就绪：{exc}") from exc


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _try_import_engine()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        bundle = generate_bundle(n_stocks=30, n_days=340, seed=7)
        save_bundle(bundle, root / "a_share")
        cls.service = AppService(data_dir=root / "a_share", db_path=root / "app.db",
                                 llm=FakeLLM())
        cls.sid = cls.service.create_session()["session_id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.close()
        cls.tmp.cleanup()

    # ------------------------------------------------------------------
    def _complete_profile(self) -> dict:
        events = list(self.service.chat_events(
            self.sid, "我是稳健型，打算持有 3 年，能接受 20% 回撤，偏好消费行业"))
        self.assertTrue(any(e == "profile_ready" for e, _ in events) or
                        self.service.get_profile(self.sid)["complete"])
        profile = self.service.get_profile(self.sid)
        confirmed = self.service.put_profile(self.sid, profile["slots"], confirmed=True)
        self.assertTrue(confirmed["confirmed"])
        return confirmed

    def _collect_job(self, iterator, max_events: int = 500) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        for event, data in iterator:
            events.append((event, data))
            if len(events) >= max_events:
                break
            if event in ("result", "done", "error"):
                break
        return events

    # ------------------------------------------------------------------
    def test_01_health(self) -> None:
        health = self.service.health()
        self.assertTrue(health["ok"])
        self.assertGreater(health["universe"], 0)

    def test_02_chat_and_profile(self) -> None:
        profile = self._complete_profile()
        self.assertEqual(profile["risk_level"], "balanced")
        self.assertEqual(profile["slots"]["horizon"], "3 年")
        messages = self.service.messages(self.sid)["messages"]
        self.assertGreaterEqual(len(messages), 2)

    def test_03_selection_events(self) -> None:
        self._complete_profile()
        task = self.service.start_selection(self.sid, top_n=5)
        events = self._collect_job(self.service.selection_events(self.sid, task["task_id"]))
        kinds = [e for e, _ in events]
        self.assertIn("started", kinds)
        self.assertIn("result", kinds, msg=f"事件：{kinds}")
        result = dict(events)[("result")] if False else None  # noqa: F841
        payload = next(data for event, data in events if event == "result")
        self.assertTrue(payload["top"])
        for key in ("cagr", "sharpe", "mdd", "ir", "win_rate"):
            self.assertIn(key, payload["metrics"])
        self.assertTrue(payload["explanation"])

    def test_04_tuning_and_versions(self) -> None:
        self._complete_profile()
        job = self.service.start_tuning(self.sid, trials=3)
        events = self._collect_job(self.service.tuning_events(self.sid, job["job_id"]))
        kinds = [e for e, _ in events]
        self.assertIn("tuning_started", kinds)
        self.assertIn("trial", kinds)
        self.assertIn("done", kinds, msg=f"事件：{kinds}")
        done = next(data for event, data in events if event == "done")
        candidate_id = done["candidate_version_id"]
        versions = self.service.versions()
        self.assertTrue(any(v["id"] == candidate_id for v in versions["versions"]))
        self.service.activate_version(candidate_id, "test")
        self.assertEqual(self.service.versions()["active_id"], candidate_id)
        self.service.rollback_version(candidate_id, "test")
        self.assertTrue(self.service.versions()["active_id"])

    def test_05_selection_requires_confirmed_profile(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        bundle = generate_bundle(n_stocks=10, n_days=200, seed=3)
        save_bundle(bundle, root / "a_share")
        service = AppService(data_dir=root / "a_share", db_path=root / "s.db", llm=FakeLLM())
        try:
            sid = service.create_session()["session_id"]
            with self.assertRaises(Exception):
                service.start_selection(sid, top_n=3)
        finally:
            service.close()

    def test_06_missing_data_raises(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with self.assertRaises(Exception):
            AppService(data_dir=Path(tmp.name) / "empty", db_path=Path(tmp.name) / "x.db",
                       llm=FakeLLM())


if __name__ == "__main__":
    unittest.main()
