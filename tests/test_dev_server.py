# -*- coding: utf-8 -*-
"""标准库 dev_server 冒烟测试（不依赖 fastapi）。"""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from backend.data.sample import generate_bundle
from backend.data.base import save_bundle
from backend.dev_server import ApiHandler
from backend.llm import FakeLLM
from backend.service import AppService


class DevServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from backend.engine import factors, selection, tuner  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"引擎尚未就绪：{exc}") from exc
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        save_bundle(generate_bundle(n_stocks=12, n_days=220, seed=11), root / "a_share")
        cls.service = AppService(data_dir=root / "a_share", db_path=root / "app.db", llm=FakeLLM())
        ApiHandler.service = cls.service
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.service.close()
        cls.tmp.cleanup()

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)

    def test_health(self) -> None:
        conn = self._conn()
        conn.request("GET", "/api/health")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertTrue(data["ok"])

    def test_session_and_profile(self) -> None:
        conn = self._conn()
        conn.request("POST", "/api/sessions", body=b"{}",
                     headers={"Content-Type": "application/json"})
        sid = json.loads(conn.getresponse().read())["session_id"]
        conn.request("GET", f"/api/sessions/{sid}/profile")
        profile = json.loads(conn.getresponse().read())
        conn.close()
        self.assertFalse(profile["confirmed"])
        self.assertTrue(sid)

    def test_unknown_api_404(self) -> None:
        conn = self._conn()
        conn.request("GET", "/api/nope")
        resp = conn.getresponse()
        conn.close()
        self.assertEqual(resp.status, 404)


if __name__ == "__main__":
    unittest.main()
