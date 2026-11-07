# -*- coding: utf-8 -*-
"""管理员认证核心逻辑测试（不启动 HTTP 服务）。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi.responses import Response

from backend.auth import AuthManager, hash_password, validate_password, verify_password
from backend.store import Store
from tests import cleanup, workdir


class AuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = workdir("auth")
        self.store = Store(self.dir / "app.db")
        self.auth = AuthManager(self.store)

    def tearDown(self) -> None:
        self.store.close()
        cleanup(self.dir)

    @staticmethod
    def _request(host="127.0.0.1", cookies=None):
        return SimpleNamespace(client=SimpleNamespace(host=host), cookies=cookies or {})

    def test_password_hash_roundtrip(self) -> None:
        encoded = hash_password("Safe-pass-2026")
        self.assertTrue(verify_password("Safe-pass-2026", encoded))
        self.assertFalse(verify_password("wrong-pass", encoded))
        self.assertNotIn("Safe-pass-2026", encoded)
        with self.assertRaises(Exception):
            validate_password("short1")

    def test_setup_login_and_logout(self) -> None:
        request = self._request()
        token = self.auth.setup(request, "admin", "Safe-pass-2026")
        self.assertTrue(token)
        self.assertTrue(self.auth.authenticated(self._request(cookies={"zhitou_admin_session": token})))
        login_token = self.auth.login(request, "admin", "Safe-pass-2026")
        self.assertTrue(self.auth.authenticated(self._request(cookies={"zhitou_admin_session": login_token})))
        self.auth.logout(self._request(cookies={"zhitou_admin_session": login_token}))
        self.assertIsNone(self.auth.authenticated(self._request(cookies={"zhitou_admin_session": login_token})))

    def test_session_cookie_is_secure_by_default(self) -> None:
        response = Response()
        self.auth.set_cookie(response, "session-token")
        self.assertIn("Secure", response.headers["set-cookie"])

    def test_setup_only_from_loopback(self) -> None:
        with self.assertRaises(Exception):
            self.auth.setup(self._request(host="10.0.0.8"), "admin", "Safe-pass-2026")
        self.assertFalse(self.auth.configured())

    def test_login_rate_limit(self) -> None:
        self.auth.setup(self._request(), "admin", "Safe-pass-2026")
        for _ in range(5):
            with self.assertRaises(Exception):
                self.auth.login(self._request(), "admin", "wrong-pass-1")
        with self.assertRaises(Exception):
            self.auth.login(self._request(), "admin", "Safe-pass-2026")

    def test_admin_user_management_and_session_owner(self) -> None:
        token = self.auth.setup(self._request(), "admin", "Safe-pass-2026")
        admin = self.auth.authenticated(
            self._request(cookies={"zhitou_admin_session": token}))
        self.assertEqual(admin["role"], "admin")

        user = self.auth.create_user(admin, "alice", "User-pass-2026", "user")
        self.assertEqual(user["role"], "user")
        self.assertEqual(len(self.auth.list_users(admin)), 2)
        with self.assertRaises(Exception):
            self.auth.create_user(admin, "alice", "Other-pass-2026", "user")

        user_token = self.auth.login(self._request(), "alice", "User-pass-2026")
        alice = self.auth.authenticated(
            self._request(cookies={"zhitou_admin_session": user_token}))
        self.assertEqual(alice["id"], user["id"])

        self.auth.update_user(admin, user["id"], active=False)
        with self.assertRaises(Exception):
            self.auth.login(self._request(), "alice", "User-pass-2026")

        self.auth.update_user(admin, user["id"], active=True)
        self.auth.delete_user(admin, user["id"])
        self.assertEqual(len(self.auth.list_users(admin)), 1)

        admin_session = self.store.create_session(admin["id"])
        other_session = self.store.create_session(999)
        listed = self.store.list_sessions(owner_id=admin["id"])
        self.assertTrue(any(item["id"] == admin_session for item in listed))
        self.assertTrue(all(item["id"] != other_session for item in listed))


if __name__ == "__main__":
    unittest.main()
