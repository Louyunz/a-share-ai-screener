# -*- coding: utf-8 -*-
"""SQLite 存储层：会话、消息、画像、模型版本与审计。

所有写操作加锁，可被后台线程安全调用（选股/调优任务在独立线程执行）。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  tokens INTEGER DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS profiles (
  session_id TEXT PRIMARY KEY,
  slots_json TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  confirmed INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  weights_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  status TEXT NOT NULL,
  source TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor TEXT,
  action TEXT NOT NULL,
  from_version TEXT,
  to_version TEXT,
  reason TEXT,
  created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # 会话与消息
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:16]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions(id, created_at) VALUES (?, ?)", (session_id, _now()))
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
        return row is not None

    def add_message(self, session_id: str, role: str, content: str, tokens: int = 0) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, tokens, _now()))

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id"
        args: tuple[Any, ...] = (session_id,)
        if limit:
            sql = ("SELECT role, content, created_at FROM ("
                   "SELECT role, content, created_at, id FROM messages WHERE session_id=?"
                   " ORDER BY id DESC LIMIT ?) ORDER BY id")
            args = (session_id, int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 画像
    # ------------------------------------------------------------------
    def save_profile(self, session_id: str, slots: dict[str, str],
                     risk_level: str, confirmed: bool) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO profiles(session_id, slots_json, risk_level, confirmed, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET slots_json=excluded.slots_json,"
                " risk_level=excluded.risk_level, confirmed=excluded.confirmed,"
                " updated_at=excluded.updated_at",
                (session_id, json.dumps(slots, ensure_ascii=False), risk_level,
                 1 if confirmed else 0, _now()))

    def get_profile(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT slots_json, risk_level, confirmed, updated_at FROM profiles"
                " WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            return None
        return {
            "slots": json.loads(row["slots_json"]),
            "risk_level": row["risk_level"],
            "confirmed": bool(row["confirmed"]),
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # 模型版本
    # ------------------------------------------------------------------
    def create_version(self, weights: dict, metrics: dict, status: str = "candidate",
                       source: str = "auto_tune", parent_id: str | None = None) -> str:
        version_id = "v" + uuid.uuid4().hex[:10]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO versions(id, parent_id, weights_json, metrics_json,"
                " status, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (version_id, parent_id, json.dumps(weights, ensure_ascii=False),
                 json.dumps(metrics, ensure_ascii=False), status, source, _now()))
        return version_id

    @staticmethod
    def _version_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "parent_id": row["parent_id"],
            "weights": json.loads(row["weights_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "status": row["status"],
            "source": row["source"],
            "created_at": row["created_at"],
        }

    def list_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM versions ORDER BY created_at DESC").fetchall()
        return [self._version_row(r) for r in rows]

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
        return self._version_row(row) if row else None

    def active_version(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM versions WHERE status='active'"
                " ORDER BY created_at DESC LIMIT 1").fetchone()
        return self._version_row(row) if row else None

    def set_status(self, version_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE versions SET status=? WHERE id=?", (status, version_id))

    def activate(self, version_id: str, actor: str = "demo", reason: str = "user_action") -> dict[str, Any]:
        """启用指定版本：当前 active 置 archived，目标置 active，并写审计。"""
        version = self.get_version(version_id)
        if version is None:
            raise KeyError(f"版本不存在: {version_id}")
        previous = self.active_version()
        with self._lock, self._conn:
            if previous:
                self._conn.execute(
                    "UPDATE versions SET status='archived' WHERE id=?", (previous["id"],))
            self._conn.execute(
                "UPDATE versions SET status='active' WHERE id=?", (version_id,))
            self._conn.execute(
                "INSERT INTO audit_logs(actor, action, from_version, to_version, reason, created_at)"
                " VALUES (?, 'activate', ?, ?, ?, ?)",
                (actor, previous["id"] if previous else None, version_id, reason, _now()))
        return self.active_version()  # type: ignore[return-value]

    def rollback(self, version_id: str, actor: str = "demo", reason: str = "rollback") -> dict[str, Any]:
        current = self.active_version()
        if current and current["id"] == version_id:
            return current
        version = self.get_version(version_id)
        if version is None:
            raise KeyError(f"版本不存在: {version_id}")
        with self._lock, self._conn:
            if current:
                self._conn.execute(
                    "UPDATE versions SET status='archived' WHERE id=?", (current["id"],))
            self._conn.execute(
                "UPDATE versions SET status='active' WHERE id=?", (version_id,))
            self._conn.execute(
                "INSERT INTO audit_logs(actor, action, from_version, to_version, reason, created_at)"
                " VALUES (?, 'rollback', ?, ?, ?, ?)",
                (actor, current["id"] if current else None, version_id, reason, _now()))
        return self.active_version()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 审计
    # ------------------------------------------------------------------
    def audit(self, action: str, actor: str = "demo", from_version: str | None = None,
              to_version: str | None = None, reason: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit_logs(actor, action, from_version, to_version, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (actor, action, from_version, to_version, reason, _now()))

    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
