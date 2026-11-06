# -*- coding: utf-8 -*-
"""SQLite 存储：会话、消息、画像、模型版本与运行态。

v2 数据库结构由 ``SCHEMA_VERSION`` 管理；版本不匹配时重建表（运行期数据，不入库）。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:10]


class Store:
    def __init__(self, db_path: Path | str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS session (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES session(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                blocks_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_message_session ON message(session_id, id);
            CREATE TABLE IF NOT EXISTS profile (
                session_id TEXT PRIMARY KEY REFERENCES session(id),
                slots_json TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                confirmed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS version (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                session_id TEXT,
                model_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'ai_optimize',
                status TEXT NOT NULL DEFAULT 'candidate',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY REFERENCES session(id),
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                from_version TEXT,
                to_version TEXT,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        columns = [row[1] for row in self._conn.execute("PRAGMA table_info(message)")]
        if "blocks_json" not in columns:
            self._conn.execute(
                "ALTER TABLE message ADD COLUMN blocks_json TEXT NOT NULL DEFAULT '[]'")
        version_columns = [row[1] for row in self._conn.execute("PRAGMA table_info(version)")]
        if "session_id" not in version_columns:
            self._conn.execute("ALTER TABLE version ADD COLUMN session_id TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_version_session "
            "ON version(session_id, created_at)")
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self._conn.commit()

    # ------------------------------------------------------------------
    # 会话与消息
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._conn.execute(
                "INSERT INTO session(id, created_at) VALUES (?, ?)", (session_id, _now()))
            self._conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM session WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def add_message(self, session_id: str, role: str, content: str,
                    blocks: list[dict[str, Any]] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO message(session_id, role, content, blocks_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content,
                 json.dumps(blocks or [], ensure_ascii=False), _now()))
            self._conn.commit()

    def list_sessions(self, limit: int = 80) -> list[dict[str, Any]]:
        """历史对话归档列表：按最近消息时间倒序。"""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.id AS id,
                       s.created_at AS created_at,
                       (SELECT content FROM message WHERE session_id = s.id AND role = 'user'
                        ORDER BY id LIMIT 1) AS title,
                       (SELECT COUNT(*) FROM message WHERE session_id = s.id) AS message_count,
                       (SELECT MAX(created_at) FROM message WHERE session_id = s.id) AS last_at,
                       COALESCE(p.summary, '') AS profile_summary,
                       COALESCE(p.confirmed, 0) AS profile_confirmed
                FROM session s
                LEFT JOIN profile p ON p.session_id = s.id
                ORDER BY COALESCE(last_at, s.created_at) DESC, s.rowid DESC
                LIMIT ?
                """,
                (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, blocks_json, created_at FROM message "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, int(limit))).fetchall()
        messages: list[dict[str, Any]] = []
        for row in reversed(rows):
            item = dict(row)
            item["blocks"] = json.loads(item.pop("blocks_json") or "[]")
            messages.append(item)
        return messages

    # ------------------------------------------------------------------
    # 画像
    # ------------------------------------------------------------------
    def save_profile(self, session_id: str, slots: dict[str, Any], summary: str,
                     confirmed: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO profile(session_id, slots_json, summary, confirmed, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "slots_json=excluded.slots_json, summary=excluded.summary, "
                "confirmed=excluded.confirmed, updated_at=excluded.updated_at",
                (session_id, json.dumps(slots, ensure_ascii=False), summary,
                 1 if confirmed else 0, _now()))
            self._conn.commit()

    def get_profile(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT slots_json, summary, confirmed, updated_at FROM profile WHERE session_id = ?",
                (session_id,)).fetchone()
        if row is None:
            return None
        return {
            "slots": json.loads(row["slots_json"]),
            "summary": row["summary"],
            "confirmed": bool(row["confirmed"]),
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # 会话运行态（待审批方案、最近结果等，不写 Git）
    # ------------------------------------------------------------------
    def get_state(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM session_state WHERE session_id = ?",
                (session_id,)).fetchone()
        return json.loads(row["state_json"]) if row else {}

    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_state(session_id, state_json, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "state_json=excluded.state_json, updated_at=excluded.updated_at",
                (session_id, json.dumps(state, ensure_ascii=False), _now()))
            self._conn.commit()

    # ------------------------------------------------------------------
    # 模型版本
    # ------------------------------------------------------------------
    def create_version(self, model: dict[str, Any], metrics: dict[str, Any], *,
                       status: str = "candidate", parent_id: str | None = None,
                       source: str = "ai_optimize", summary: str = "",
                       session_id: str | None = None) -> str:
        version_id = _new_id("v")
        with self._lock:
            self._conn.execute(
                "INSERT INTO version(id, parent_id, session_id, model_json, metrics_json, summary, "
                "source, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (version_id, parent_id, session_id, json.dumps(model, ensure_ascii=False),
                 json.dumps(metrics, ensure_ascii=False), summary, source, status, _now()))
            self._conn.commit()
        return version_id

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM version WHERE id = ?", (version_id,)).fetchone()
        return self._version_row(row) if row else None

    def list_versions(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if session_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM version ORDER BY created_at ASC, id ASC").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM version WHERE session_id = ? "
                    "ORDER BY created_at ASC, id ASC", (session_id,)).fetchall()
        return [self._version_row(row) for row in rows]

    def active_version(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM version WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self._version_row(row) if row else None

    def activate(self, version_id: str, *, reason: str = "user_action") -> dict[str, Any]:
        target = self.get_version(version_id)
        if target is None:
            raise KeyError(f"版本不存在：{version_id}")
        previous = self.active_version()
        with self._lock:
            self._conn.execute("UPDATE version SET status = 'archived' WHERE status = 'active'")
            self._conn.execute("UPDATE version SET status = 'active' WHERE id = ?", (version_id,))
            self._conn.execute(
                "INSERT INTO audit_log(action, from_version, to_version, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("activate", previous["id"] if previous else None, version_id, reason, _now()))
            self._conn.commit()
        return self.get_version(version_id) or target

    def rollback(self, version_id: str, *, reason: str = "user_rollback") -> dict[str, Any]:
        return self.activate(version_id, reason=reason)

    def parent_version(self, version_id: str | None = None) -> dict[str, Any] | None:
        current = self.get_version(version_id) if version_id else self.active_version()
        if current is None:
            return None
        if current.get("parent_id"):
            return self.get_version(current["parent_id"])
        archived = [v for v in self.list_versions() if v["id"] != current["id"]]
        return archived[-1] if archived else None

    def audit(self, action: str, *, from_version: str | None = None,
              to_version: str | None = None, reason: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log(action, from_version, to_version, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (action, from_version, to_version, reason, _now()))
            self._conn.commit()

    def list_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _version_row(row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        return {
            "id": row["id"],
            "parent_id": row["parent_id"],
            "session_id": row["session_id"] if "session_id" in keys else None,
            "model": json.loads(row["model_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "summary": row["summary"],
            "source": row["source"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
