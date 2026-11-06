# -*- coding: utf-8 -*-
"""管理员认证核心逻辑测试（不启动 HTTP 服务）。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
