# -*- coding: utf-8 -*-
"""标准库 dev_server 的 HTTP/SSE 冒烟测试。"""
from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from backend.data.base import save_bundle
from backend.data.sample import generate_bundle
from backend.dev_server import ApiHandler
from tests.support.scripted_llm import ScriptedLLM
from backend.service import AppService
from tests import cleanup, workdir


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event, payload = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
        if event and payload is not None:
            try:
                events.append((event, json.loads(payload)))
            except json.JSONDecodeError:
                continue
    return events


class DevServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = workdir("devserver")
        save_bundle(generate_bundle(n_stocks=12, n_days=240, seed=11), cls.dir / "a_share")
        cls.service = AppService(data_dir=cls.dir / "a_share", db_path=cls.dir / "app.db",
                                 llm=ScriptedLLM())
        ApiHandler.service = cls.service
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.service.close()
        cleanup(cls.dir)

    def _request(self, method: str, path: str, body: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        conn.request(method, path, body=json.dumps(body or {}).encode(),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        conn.close()
        return response.status, raw

    def test_health_and_session(self) -> None:
        status, body = self._request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        status, body = self._request("POST", "/api/sessions")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["session_id"])

    def test_chat_stream(self) -> None:
        status, body = self._request("POST", "/api/sessions")
        sid = json.loads(body)["session_id"]
        status, text = self._request(
            "POST", f"/api/sessions/{sid}/chat",
            {"content": "我想为养老攒点钱，大概 5 年不用，最多能接受亏 15%，真亏了会有点慌但先不卖"})
        self.assertEqual(status, 200)
        events = _parse_sse(text)
        self.assertTrue(any(event == "token" for event, _ in events))
        self.assertTrue(any(event == "done" for event, _ in events))
        self.assertTrue(any(event == "ui" and data["block"]["type"] == "profile"
                            for event, data in events))

    def test_sessions_archive(self) -> None:
        status, body = self._request("POST", "/api/sessions")
        sid = json.loads(body)["session_id"]
        self._request("POST", f"/api/sessions/{sid}/chat", {"content": "我想为养老攒点钱"})
        status, body = self._request("GET", "/api/sessions")
        self.assertEqual(status, 200)
        sessions = json.loads(body)["sessions"]
        self.assertTrue(any(item["id"] == sid for item in sessions))
        self.assertTrue(all("title" in item and "message_count" in item for item in sessions))

    def test_state_and_messages(self) -> None:
        status, body = self._request("POST", "/api/sessions")
        sid = json.loads(body)["session_id"]
        self._request("POST", f"/api/sessions/{sid}/chat", {"content": "我想为养老攒点钱"})
        status, body = self._request("GET", f"/api/sessions/{sid}/state")
        self.assertEqual(status, 200)
        self.assertIn("profile", json.loads(body))
        status, body = self._request("GET", f"/api/sessions/{sid}/messages")
        self.assertGreaterEqual(len(json.loads(body)["messages"]), 2)

    def _confirmed_session_with_candidate(self) -> str:
        status, body = self._request("POST", "/api/sessions")
        sid = json.loads(body)["session_id"]
        self._request("POST", f"/api/sessions/{sid}/chat", {"content": (
            "我想为养老攒点钱，大概 5 年不用，最多能接受亏 15%，真亏了会有点慌但先不卖")})
        self._request("POST", f"/api/sessions/{sid}/chat", {"content": "可以，就按这个来"})
        self._request("POST", f"/api/sessions/{sid}/chat", {"content": "帮我优化一下模型"})
        return sid

    def test_workbench_apply_candidate_and_activate_version(self) -> None:
        sid = self._confirmed_session_with_candidate()
        status, body = self._request("GET", f"/api/sessions/{sid}/state")
        state = json.loads(body)
        self.assertIn("pending_candidates", state)
        pending = [item for item in state["pending_candidates"]
                   if item["status"] == "pending"]
        self.assertTrue(pending)
        target = pending[-1]
        status, body = self._request(
            "POST", f"/api/sessions/{sid}/candidates/{target['index']}/apply")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"]["active_model"]["name"], target["name"])
        archived = [item for item in result["state"]["versions"]
                    if item["status"] != "active"]
        self.assertTrue(archived)
        target_version = archived[0]
        status, body = self._request(
            "POST", f"/api/sessions/{sid}/versions/{target_version['id']}/activate")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"]["active_model"]["name"],
                         target_version["name"])
        status, _ = self._request("POST", f"/api/sessions/{sid}/candidates/99/apply")
        self.assertIn(status, (400, 404, 409))
        status, _ = self._request("POST", f"/api/sessions/{sid}/versions/v-missing/activate")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
