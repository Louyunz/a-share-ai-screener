# -*- coding: utf-8 -*-
"""单对话全链路冒烟：画像 → AI 确认并选股 → 追问 → AI 优化 → 审批启用 → 回滚。

用法：
    python scripts/e2e_smoke.py
需要 data/a_share 缓存（先运行 build_cache --source sample）。
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
from tests.support.scripted_llm import ScriptedLLM  # noqa: E402
from backend.service import AppService  # noqa: E402


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=240)

    def get(self, path: str) -> dict:
        conn = self._conn()
        conn.request("GET", path)
        data = json.loads(conn.getresponse().read())
        conn.close()
        return data

    def post(self, path: str, body: dict | None = None) -> dict:
        conn = self._conn()
        conn.request("POST", path, body=json.dumps(body or {}).encode(),
                     headers={"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        conn.close()
        return data

    def chat(self, session_id: str, content: str) -> list[tuple[str, dict]]:
        conn = self._conn()
        conn.request("POST", f"/api/sessions/{session_id}/chat",
                     body=json.dumps({"content": content}).encode(),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        text = response.read().decode("utf-8")
        conn.close()
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


def _blocks(events: list[tuple[str, dict]], kind: str | None = None) -> list[dict]:
    blocks = [data["block"] for event, data in events if event == "ui"]
    return [block for block in blocks if kind is None or block["type"] == kind]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A 股 AI 选股助手：单对话全链路冒烟")
    parser.add_argument("--data-dir", default=str(A_SHARE_DIR))
    args = parser.parse_args(argv)

    work = ROOT / "tests" / "_tmp"
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / f"e2e_chat_{int(time.time())}.db"
    service = AppService(data_dir=Path(args.data_dir), db_path=db_path, llm=ScriptedLLM())
    ApiHandler.service = service
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    client = Client(port)
    print(f"[e2e] 服务启动 http://127.0.0.1:{port}（数据 {service.data.as_of}）")
    try:
        health = client.get("/api/health")
        assert health["ok"], health
        sid = client.post("/api/sessions")["session_id"]
        print(f"[e2e] 会话 {sid}")

        events = client.chat(sid, "我想为养老攒点钱")
        blocks = _blocks(events)
        # 当前产品行为：逐题采集阶段只追问、不弹画像卡；画像齐全后再出现画像卡。
        assert not any(block["type"] == "profile" for block in blocks), blocks
        assert any(event == "token" for event, _ in events)
        print("[e2e] 画像采集开始（逐题阶段不弹卡）")

        events = client.chat(sid, "大概 5 年不用这笔钱，最多能接受亏 15%，真亏了会有点慌但先不卖")
        profile = _blocks(events, "profile")[-1]
        assert profile["complete"], profile
        print("[e2e] 画像完整:", profile["summary"])

        events = client.chat(sid, "可以，就按这个来")
        selection = _blocks(events, "selection")[-1]
        assert selection["top"] and selection["metrics"], selection
        print(f"[e2e] AI 已确认画像并选股 Top {len(selection['top'])}；"
              f"夏普 {selection['metrics']['sharpe']:.2f}，回撤 {selection['metrics']['mdd']:.1%}")

        events = client.chat(sid, "为什么最大回撤会这么大？")
        text = "".join(data.get("text", "") for event, data in events if event == "token")
        assert "不构成投资建议" in text
        print("[e2e] 追问解读 OK")

        events = client.chat(sid, "帮我优化一下模型")
        work = _blocks(events, "work")[-1]
        tuning = _blocks(events, "tuning")[-1]
        approval = _blocks(events, "approval")[-1]
        assert work["status"] == "done" and len(work["steps"]) >= 6, work
        assert approval["actions"], approval
        assert tuning["status"] == "done", tuning
        assert tuning["progress"]["total"] > 1, tuning
        assert tuning["space"], "应展示 AI 给出的调参范围"
        code_change = work.get("code_change") or {}
        assert code_change.get("diff"), "AI 修改代码时应返回逐行 diff"
        assert code_change.get("added", 0) > 0
        print(f"[e2e] AI 定引擎/范围 + 服务端寻优 {tuning['progress']['trial']} 组，"
              f"等待用户审批：{work['proposal']['summary']}")

        events = client.chat(sid, "同意，启用这个改进")
        version = _blocks(events, "version")[-1]
        assert version["active"]["version_id"], version
        state = client.get(f"/api/sessions/{sid}/state")
        print(f"[e2e] 当前会话已启用新模型：{state['active_model']['name']}")

        # 新对话必须回到默认模型，不能沿用上一个对话的模型
        other_sid = client.post("/api/sessions")["session_id"]
        other_state = client.get(f"/api/sessions/{other_sid}/state")
        assert other_state["active_model"]["name"] == "均衡多因子", other_state["active_model"]
        print("[e2e] 新对话使用默认模型：均衡多因子")

        events = client.chat(sid, "回滚到上一个版本")
        assert _blocks(events, "version"), events
        state = client.get(f"/api/sessions/{sid}/state")
        assert state["active_model"]["name"] == "均衡多因子", state["active_model"]
        print(f"[e2e] 回滚后当前会话模型：{state['active_model']['name']}，"
              f"会话版本 {len(state['versions'])} 个")

        # AI 直接写程序：更换数学模型 → LLM 二次审核 → 寻优 → 审批 → 启用
        events = client.chat(sid, "更换数学模型")
        work = _blocks(events, "work")[-1]
        tuning = _blocks(events, "tuning")[-1]
        assert work["status"] == "done", work
        assert tuning["status"] == "done", tuning
        assert any(step.get("title") == "LLM 二次审核" for step in work["steps"]), work
        code_change = work.get("code_change") or {}
        assert code_change.get("diff"), "AI 程序应按行流式输出 diff"
        assert "helpers.py" in "\n".join(code_change.get("files") or []), code_change
        print(f"[e2e] AI 直接写程序（{len(code_change.get('files') or [])} 个模块）"
              f" + 二次审核通过")
        events = client.chat(sid, "同意，启用这个改进")
        state = client.get(f"/api/sessions/{sid}/state")
        assert state["active_model"]["name"] == "AI 生成模型", state["active_model"]
        print(f"[e2e] 程序模型已启用：{state['active_model']['name']}")
        print("[e2e] [OK] 单对话全链路通过")
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
