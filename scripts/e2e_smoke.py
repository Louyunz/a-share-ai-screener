# -*- coding: utf-8 -*-
"""端到端冒烟脚本：通过 HTTP + SSE 走通完整业务闭环（不依赖浏览器）。

用法：
    python scripts/e2e_smoke.py                      # 使用 data/a_share 缓存 + FakeLLM
    python scripts/e2e_smoke.py --use-cache          # 同上
    set LLM_PROVIDER=fake & python scripts/e2e_smoke.py

流程：健康检查 → 创建会话 → AI 对话（SSE）→ 确认画像 → 选股（SSE）→
      调参（SSE）→ 启用候选版本 → 回滚 → 打印摘要。
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import A_SHARE_DIR  # noqa: E402
from backend.dev_server import ApiHandler  # noqa: E402
from backend.llm import FakeLLM  # noqa: E402
from backend.service import AppService  # noqa: E402


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def request(self, method: str, path: str, body: dict | None = None, stream: bool = False):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        conn.request(method, path, body=json.dumps(body or {}).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if stream:
            text = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return resp.status, text
        data = resp.read()
        conn.close()
        return resp.status, json.loads(data) if data else {}

    def sse(self, path: str) -> list[tuple[str, dict]]:
        status, text = self.request("GET", path, stream=True)
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

    def post_sse(self, path: str, body: dict) -> list[tuple[str, dict]]:
        status, text = self.request("POST", path, body, stream=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A 股 AI 选股 MVP 端到端冒烟")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--data-dir", default=str(A_SHARE_DIR))
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args(argv)

    work = ROOT / "tests" / "_tmp"
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / f"e2e_{int(time.time())}.db"
    service = AppService(data_dir=Path(args.data_dir), db_path=db_path, llm=FakeLLM())
    ApiHandler.service = service
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), ApiHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    client = Client(port)
    print(f"[e2e] 服务启动: http://127.0.0.1:{port}  (data={service.data.as_of})")
    try:
        status, health = client.request("GET", "/api/health")
        assert status == 200 and health["ok"], health
        print(f"[e2e] health: {health}")

        status, session = client.request("POST", "/api/sessions")
        sid = session["session_id"]
        print(f"[e2e] session: {sid}")

        chat = client.post_sse(f"/api/sessions/{sid}/messages",
                               {"content": "稳健型，持有 3 年，能接受 20% 回撤，偏好消费行业"})
        kinds = [e for e, _ in chat]
        assert "profile_ready" in kinds or any(e == "slots" for e in kinds), kinds
        print(f"[e2e] chat events: {kinds}")

        status, profile = client.request("GET", f"/api/sessions/{sid}/profile")
        status, confirmed = client.request("PUT", f"/api/sessions/{sid}/profile",
                                           {"slots": profile["slots"], "confirmed": True})
        assert confirmed["confirmed"], confirmed
        print(f"[e2e] profile confirmed: {confirmed['slots']}")

        status, task = client.request("POST", f"/api/sessions/{sid}/selection", {"top_n": 5})
        selection = client.sse(f"/api/sessions/{sid}/selection/events?task_id={task['task_id']}")
        kinds = [e for e, _ in selection]
        assert "result" in kinds, f"selection events={kinds}"
        result = next(data for event, data in selection if event == "result")
        print(f"[e2e] selection: top={[x['code'] for x in result['top'][:5]]} "
              f"metrics={ {k: round(v, 4) for k, v in result['metrics'].items()} }")
        assert result["top"] and result["metrics"]["sharpe"] is not None

        status, job = client.request("POST", f"/api/sessions/{sid}/tune", {"trials": args.trials})
        tuning = client.sse(f"/api/sessions/{sid}/tune/events?job_id={job['job_id']}")
        kinds = [e for e, _ in tuning]
        assert "done" in kinds, f"tuning events={kinds}"
        done = next(data for event, data in tuning if event == "done")
        trials = [d for e, d in tuning if e == "trial"]
        print(f"[e2e] tuning: {len(trials)} trials, candidate={done['candidate_version_id']}, "
              f"best_sharpe={done['compare']['candidate']['metrics']['sharpe']:.3f} "
              f"(base {done['compare']['current']['metrics']['sharpe']:.3f})")

        status, versions = client.request("GET", "/api/versions")
        active_before = versions["active_id"]
        status, activated = client.request(
            "POST", f"/api/versions/{done['candidate_version_id']}/activate", {"reason": "e2e"})
        assert activated["active_id"] == done["candidate_version_id"], activated
        print(f"[e2e] version activated: {activated['active_id']}")

        status, rolled = client.request(
            "POST", f"/api/versions/{active_before}/rollback", {"reason": "e2e"})
        assert rolled["active_id"] == active_before, rolled
        print(f"[e2e] version rolled back: {rolled['active_id']}")

        print("[e2e] ✅ 端到端冒烟全部通过")
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()
        service.close()
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
