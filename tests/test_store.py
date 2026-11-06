# -*- coding: utf-8 -*-
"""SQLite 存储测试（v2：会话/消息/画像/模型版本/运行态）。"""
from __future__ import annotations

import unittest

from backend.store import SCHEMA_VERSION, Store
from tests import cleanup, workdir


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = workdir("store")
        self.store = Store(self.dir / "test.db")

    def tearDown(self) -> None:
        self.store.close()
        cleanup(self.dir)

    def test_schema_version(self) -> None:
        value = self.store._conn.execute("PRAGMA user_version").fetchone()[0]  # noqa: SLF001
        self.assertEqual(int(value), SCHEMA_VERSION)

    def test_session_and_messages(self) -> None:
        sid = self.store.create_session()
        self.assertTrue(self.store.session_exists(sid))
        self.store.add_message(sid, "user", "你好")
        self.store.add_message(sid, "assistant", "在的")
        messages = self.store.get_messages(sid)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(messages[-1]["content"], "在的")

    def test_message_blocks_persisted(self) -> None:
        sid = self.store.create_session()
        blocks = [{"id": "work-1", "type": "work", "status": "done",
                   "code_change": {"file": "models/strategy.py", "diff": "+x", "added": 1,
                                   "removed": 0}}]
        self.store.add_message(sid, "assistant", "已生成代码方案", blocks)
        messages = self.store.get_messages(sid)
        self.assertTrue(messages[0]["blocks"])
        self.assertEqual(messages[0]["blocks"][0]["code_change"]["diff"], "+x")

    def test_profile_upsert(self) -> None:
        sid = self.store.create_session()
        self.store.save_profile(sid, {"goal": "养老"}, "想养老", False)
        profile = self.store.get_profile(sid)
        self.assertEqual(profile["slots"]["goal"], "养老")
        self.assertFalse(profile["confirmed"])
        self.store.save_profile(sid, {"goal": "养老", "horizon": "5 年"}, "想养老，5 年", True)
        profile = self.store.get_profile(sid)
        self.assertEqual(profile["slots"]["horizon"], "5 年")
        self.assertTrue(profile["confirmed"])

    def test_runtime_state(self) -> None:
        sid = self.store.create_session()
        self.assertEqual(self.store.get_state(sid), {})
        self.store.save_state(sid, {"pending": {"summary": "换模型"}})
        self.assertEqual(self.store.get_state(sid)["pending"]["summary"], "换模型")
        self.store.save_state(sid, {"last_selection": {"as_of": "2024-01-01"}})
        state = self.store.get_state(sid)
        self.assertNotIn("pending", state)
        self.assertIn("last_selection", state)

    def test_versions_activate_rollback_and_parent(self) -> None:
        base = self.store.create_version(
            {"kind": "builtin", "model_id": "multi_factor", "name": "均衡多因子",
             "weights": {"ep": 1.0}}, {"sharpe": 1.0}, status="active", parent_id=None)
        candidate = self.store.create_version(
            {"kind": "code", "model_id": "ai_generated", "name": "AI 生成模型",
             "weights": {"ep": 0.5, "bp": 0.5}, "code": "def build_transform():\n    pass\n"},
            {"sharpe": 1.2}, status="candidate", parent_id=base)
        self.assertEqual(self.store.active_version()["id"], base)
        self.store.activate(candidate, reason="test")
        self.assertEqual(self.store.active_version()["id"], candidate)
        parent = self.store.parent_version()
        self.assertEqual(parent["id"], base)
        self.store.rollback(base, reason="test")
        self.assertEqual(self.store.active_version()["id"], base)
        self.assertEqual(len(self.store.list_versions()), 2)
        self.assertTrue(self.store.get_version(candidate)["model"]["code"])

    def test_unknown_version(self) -> None:
        with self.assertRaises(KeyError):
            self.store.activate("not-exist")


if __name__ == "__main__":
    unittest.main()
