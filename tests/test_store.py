# -*- coding: utf-8 -*-
"""SQLite 存储测试（v2：会话/消息/画像/模型版本/运行态）。"""
from __future__ import annotations

import sqlite3
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

    def test_list_sessions_archive(self) -> None:
        first = self.store.create_session()
        self.store.add_message(first, "user", "第一个会话的问题")
        self.store.add_message(first, "assistant", "第一个会话的回复")
        self.store.save_profile(first, {"goal": "养老"}, "目标：养老", False)
        second = self.store.create_session()
        self.store.add_message(second, "user", "第二个会话")
        sessions = self.store.list_sessions()
        self.assertEqual([item["id"] for item in sessions], [second, first])
        latest = sessions[0]
        self.assertEqual(latest["message_count"], 1)
        self.assertEqual(latest["title"], "第二个会话")
        older = sessions[1]
        self.assertEqual(older["message_count"], 2)
        self.assertTrue(older["profile_summary"])

    def test_migrates_legacy_version_table(self) -> None:
        """旧库（v3，version 表无 session_id）应自动迁移而不是启动报错。"""
        path = self.dir / "legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE version (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                model_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'ai_optimize',
                status TEXT NOT NULL DEFAULT 'candidate',
                created_at TEXT NOT NULL
            );
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            PRAGMA user_version=3;
            """
        )
        conn.execute(
            "INSERT INTO version(id, parent_id, model_json, metrics_json, summary, source, status, created_at)"
            " VALUES ('v-legacy', NULL, '{\"name\":\"旧模型\",\"weights\":{\"ep\":1}}', '{}',"
            " '旧版本', 'ai_optimize', 'active', '2024-01-01T00:00:00')"
        )
        conn.commit()
        conn.close()
        store = Store(path)
        try:
            columns = [row[1] for row in store._conn.execute("PRAGMA table_info(version)")]
            self.assertIn("session_id", columns)
            version = store.get_version("v-legacy")
            self.assertIsNotNone(version)
            self.assertEqual(version["session_id"], None)
            self.assertEqual(store.list_versions(session_id="no-session"), [])
            user_version = int(store._conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(user_version, SCHEMA_VERSION)
        finally:
            store.close()

    def test_unknown_version(self) -> None:
        with self.assertRaises(KeyError):
            self.store.activate("not-exist")


if __name__ == "__main__":
    unittest.main()
