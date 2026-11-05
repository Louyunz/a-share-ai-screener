# -*- coding: utf-8 -*-
"""Store（SQLite）单元测试。"""
from __future__ import annotations

import unittest

from backend.store import Store
from tests import cleanup, workdir


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = workdir("store")
        self.store = Store(self.dir / "test.db")

    def tearDown(self) -> None:
        self.store.close()
        cleanup(self.dir)

    def test_session_and_messages(self) -> None:
        sid = self.store.create_session()
        self.assertTrue(self.store.session_exists(sid))
        self.store.add_message(sid, "user", "你好")
        self.store.add_message(sid, "assistant", "您好")
        messages = self.store.get_messages(sid)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], "您好")
        self.assertEqual(self.store.get_messages(sid, limit=1)[0]["content"], "您好")

    def test_profile_upsert(self) -> None:
        sid = self.store.create_session()
        self.assertIsNone(self.store.get_profile(sid))
        self.store.save_profile(sid, {"risk_level": "balanced"}, "balanced", False)
        profile = self.store.get_profile(sid)
        self.assertEqual(profile["risk_level"], "balanced")
        self.assertFalse(profile["confirmed"])
        self.store.save_profile(sid, {"risk_level": "aggressive"}, "aggressive", True)
        profile = self.store.get_profile(sid)
        self.assertEqual(profile["risk_level"], "aggressive")
        self.assertTrue(profile["confirmed"])

    def test_versions_activate_rollback(self) -> None:
        v1 = self.store.create_version({"ep": 0.5, "bp": 0.5}, {"sharpe": 1.0}, "active", "manual")
        v2 = self.store.create_version({"ep": 0.4, "bp": 0.6}, {"sharpe": 1.2}, "candidate", "auto_tune", v1)
        self.assertEqual(self.store.active_version()["id"], v1)
        self.store.activate(v2, reason="test")
        self.assertEqual(self.store.active_version()["id"], v2)
        self.assertEqual(self.store.get_version(v1)["status"], "archived")
        self.store.rollback(v1, reason="test_rollback")
        self.assertEqual(self.store.active_version()["id"], v1)
        audit = self.store.list_audit()
        self.assertEqual(audit[0]["action"], "rollback")
        self.assertEqual(len(self.store.list_versions()), 2)

    def test_unknown_version_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.store.activate("not-exist")


if __name__ == "__main__":
    unittest.main()
