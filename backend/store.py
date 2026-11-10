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

SCHEMA_VERSION = 7


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
                created_at TEXT NOT NULL,
                active_version_id TEXT
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
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_account (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );
            CREATE TABLE IF NOT EXISTS auth_session (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_auth_session_expiry
                ON auth_session(expires_at);
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
        session_columns = [row[1] for row in self._conn.execute("PRAGMA table_info(session)")]
        if "active_version_id" not in session_columns:
            self._conn.execute("ALTER TABLE session ADD COLUMN active_version_id TEXT")
        audit_columns = [row[1] for row in self._conn.execute("PRAGMA table_info(audit_log)")]
        if "payload_json" not in audit_columns:
            self._conn.execute(
                "ALTER TABLE audit_log ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        self._backfill_active_version()
        self._migrate_auth()
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self._conn.commit()

    def _migrate_auth(self) -> None:
        """多用户迁移：旧单管理员 auth_user → auth_account，并给旧会话补 owner。"""
        session_columns = [row[1] for row in self._conn.execute("PRAGMA table_info(session)")]
        if "owner_id" not in session_columns:
            self._conn.execute("ALTER TABLE session ADD COLUMN owner_id INTEGER")
        auth_session_columns = [
            row[1] for row in self._conn.execute("PRAGMA table_info(auth_session)")]
        if "user_id" not in auth_session_columns:
            self._conn.execute(
                "ALTER TABLE auth_session ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0")
        legacy_tables = {
            row[0] for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "auth_user" in legacy_tables:
            row = self._conn.execute(
                "SELECT username, password_hash, created_at FROM auth_user WHERE id = 1"
            ).fetchone()
            if row and not self._conn.execute(
                    "SELECT 1 FROM auth_account WHERE username = ?",
                    (row["username"],)).fetchone():
                self._conn.execute(
                    "INSERT INTO auth_account(username, password_hash, role, active, "
                    "created_at, updated_at) VALUES (?, ?, 'admin', 1, ?, ?)",
                    (row["username"], row["password_hash"], row["created_at"], row["created_at"]))
            self._conn.execute("DROP TABLE auth_user")
        admin = self._conn.execute(
            "SELECT id FROM auth_account WHERE role = 'admin' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if admin:
            self._conn.execute(
                "UPDATE session SET owner_id = ? WHERE owner_id IS NULL", (int(admin["id"]),))

    def _backfill_active_version(self) -> None:
        """把旧版 session_state 里的 active_version_id 迁移到 session 表（单一状态源）。"""
        rows = self._conn.execute(
            """
            SELECT s.id AS sid, s.active_version_id AS active_id, st.state_json AS state_json
            FROM session s
            LEFT JOIN session_state st ON st.session_id = s.id
            WHERE s.active_version_id IS NULL
            """
        ).fetchall()
        for row in rows:
            state: dict[str, Any] = {}
            try:
                state = json.loads(row["state_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                state = {}
            active_id = state.get("active_version_id")
            if active_id:
                self._conn.execute(
                    "UPDATE session SET active_version_id = ? WHERE id = ?",
                    (str(active_id), row["sid"]))
            if any(key in state for key in ("active_model", "active_metrics", "active_version_id")):
                for key in ("active_model", "active_metrics", "active_version_id"):
                    state.pop(key, None)
                self._conn.execute(
                    "UPDATE session_state SET state_json = ? WHERE session_id = ?",
                    (json.dumps(state, ensure_ascii=False), row["sid"]))

    # ------------------------------------------------------------------
    # 会话与消息
    # ------------------------------------------------------------------
    def create_session(self, owner_id: int | None = None) -> str:
        session_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._conn.execute(
                "INSERT INTO session(id, created_at, owner_id) VALUES (?, ?, ?)",
                (session_id, _now(), owner_id))
            self._conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM session WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def session_owner(self, session_id: str) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT owner_id FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"会话不存在：{session_id}")
        return int(row["owner_id"]) if row["owner_id"] is not None else None

    def claim_orphan_sessions(self, owner_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE session SET owner_id = ? WHERE owner_id IS NULL", (int(owner_id),))
            self._conn.commit()
        return cur.rowcount

    def count_sessions(self, owner_id: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM session WHERE owner_id = ?", (int(owner_id),)).fetchone()
        return int(row["n"]) if row else 0

    def add_message(self, session_id: str, role: str, content: str,
                    blocks: list[dict[str, Any]] | None = None) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO message(session_id, role, content, blocks_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content,
                 json.dumps(blocks or [], ensure_ascii=False), _now()))
            self._conn.commit()
            return int(cursor.lastrowid)

    def update_assistant_message(self, session_id: str, message_id: int, content: str,
                                 blocks: list[dict[str, Any]] | None = None) -> None:
        """Update one streaming response, preserving its identity and creation time."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE message SET content = ?, blocks_json = ? "
                "WHERE id = ? AND session_id = ? AND role = 'assistant'",
                (content, json.dumps(blocks or [], ensure_ascii=False), message_id, session_id))
            self._conn.commit()
            if cursor.rowcount != 1:
                raise ValueError("找不到当前会话的 AI 消息")

    def finish_interrupted_message(self, session_id: str, note: str) -> list[dict[str, Any]]:
        """Mark only this turn's unfinished cards as ended on explicit stop/error."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, content, blocks_json FROM message WHERE session_id = ? "
                "AND role = 'assistant' AND id > COALESCE((SELECT MAX(id) FROM message "
                "WHERE session_id = ? AND role = 'user'), 0) ORDER BY id DESC LIMIT 1",
                (session_id, session_id)).fetchone()
            if row is None:
                self.add_message(session_id, "assistant", note)
                return []
            blocks = json.loads(row["blocks_json"] or "[]")
            changed = []
            for block in blocks:
                if block.get("status") in ("running", "writing", "generated"):
                    block["status"] = "error"
                    changed.append(block)
            text = str(row["content"] or "")
            self.update_assistant_message(session_id, row["id"],
                                          (text + "\n\n" + note).strip(), blocks)
            return changed

    def list_sessions(self, limit: int = 80, owner_id: int | None = None) -> list[dict[str, Any]]:
        """历史对话归档列表：按最近消息时间倒序。"""
        where = "WHERE s.owner_id = ?" if owner_id is not None else ""
        params: tuple[Any, ...] = (int(owner_id), int(limit)) if owner_id is not None else (int(limit),)
        with self._lock:
            rows = self._conn.execute(
                f"""
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
                {where}
                ORDER BY COALESCE(last_at, s.created_at) DESC, s.rowid DESC
                LIMIT ?
                """,
                params).fetchall()
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

    def list_versions_for_owner(self, owner_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT v.* FROM version v JOIN session s ON s.id = v.session_id "
                "WHERE s.owner_id = ? ORDER BY v.created_at ASC, v.id ASC",
                (int(owner_id),)).fetchall()
        return [self._version_row(row) for row in rows]

    def session_active_version_id(self, session_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT active_version_id FROM session WHERE id = ?",
                (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"会话不存在：{session_id}")
        value = row["active_version_id"]
        return str(value) if value else None

    def set_session_active_version(self, session_id: str, version_id: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE session SET active_version_id = ? WHERE id = ?",
                (version_id, session_id))
            self._conn.commit()

    def session_active_version(self, session_id: str) -> dict[str, Any] | None:
        version_id = self.session_active_version_id(session_id)
        return self.get_version(version_id) if version_id else None

    def audit(self, action: str, *, from_version: str | None = None,
              to_version: str | None = None, reason: str = "",
              payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log(action, from_version, to_version, reason, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (action, from_version, to_version, reason,
                 json.dumps(payload or {}, ensure_ascii=False), _now()))
            self._conn.commit()

    def list_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json", "{}") or "{}")
            except (TypeError, json.JSONDecodeError):
                item["payload"] = {}
            items.append(item)
        return items

    # ------------------------------------------------------------------
    # 用户与登录会话
    # ------------------------------------------------------------------
    @staticmethod
    def _auth_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def auth_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, username, password_hash, role, active, created_at, "
                "updated_at, last_login_at FROM auth_account WHERE username = ?",
                (str(username).strip(),)).fetchone()
        return self._auth_row(row)

    def auth_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, username, password_hash, role, active, created_at, "
                "updated_at, last_login_at FROM auth_account WHERE id = ?",
                (int(user_id),)).fetchone()
        return self._auth_row(row)

    def list_auth_users(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, username, role, active, created_at, updated_at, last_login_at "
                "FROM auth_account ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]

    def create_auth_user(self, username: str, password_hash: str,
                         role: str = "user", active: int = 1) -> int | None:
        now = _now()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO auth_account(username, password_hash, role, active, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (username, password_hash, role, int(active), now, now))
                self._conn.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                return None

    def update_auth_user(self, user_id: int, *, password_hash: str | None = None,
                         role: str | None = None, active: int | None = None,
                         last_login: bool = False) -> bool:
        fields: list[str] = []
        values: list[Any] = []
        if password_hash is not None:
            fields.append("password_hash = ?")
            values.append(password_hash)
        if role is not None:
            fields.append("role = ?")
            values.append(role)
        if active is not None:
            fields.append("active = ?")
            values.append(int(active))
        if last_login:
            fields.append("last_login_at = ?")
            values.append(_now())
        if not fields:
            return True
        fields.append("updated_at = ?")
        values.append(_now())
        values.append(int(user_id))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE auth_account SET {', '.join(fields)} WHERE id = ?", values)
            self._conn.commit()
        return cur.rowcount > 0

    def delete_auth_user(self, user_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM auth_account WHERE id = ?", (int(user_id),))
            self._conn.commit()
        return cur.rowcount > 0

    def count_active_admins(self, exclude_id: int | None = None) -> int:
        with self._lock:
            if exclude_id is None:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM auth_account "
                    "WHERE role = 'admin' AND active = 1").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM auth_account "
                    "WHERE role = 'admin' AND active = 1 AND id <> ?",
                    (int(exclude_id),)).fetchone()
        return int(row["n"]) if row else 0

    def create_auth_session(self, token_hash: str, expires_at: int, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO auth_session(token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (token_hash, int(user_id), int(datetime.now().timestamp()), expires_at))
            self._conn.commit()

    def auth_session(self, token_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, expires_at FROM auth_session WHERE token_hash = ?",
                (token_hash,)).fetchone()
        return dict(row) if row else None

    def delete_auth_session(self, token_hash: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM auth_session WHERE token_hash = ?", (token_hash,))
            self._conn.commit()

    def delete_auth_sessions_for_user(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM auth_session WHERE user_id = ?", (int(user_id),))
            self._conn.commit()

    def purge_expired_auth_sessions(self, before: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM auth_session WHERE expires_at <= ?", (int(before),))
            self._conn.commit()

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
