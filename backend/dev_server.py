# -*- coding: utf-8 -*-
"""标准库 HTTP 服务器：在没有安装 fastapi/uvicorn 的环境里本地运行完整 MVP。

与 FastAPI 版共用 backend.service，接口与 SSE 事件完全一致；仅用于开发/本地预览，
生产与提交建议使用 `uvicorn backend.main:app`。

用法：
    python -m backend.dev_server --port 8000
    set LLM_PROVIDER=fake & python -m backend.dev_server --port 8000   # 无 Key 仅看 UI
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

from .config import FRONTEND_DIR
from .data import DataError
from .events import sse_format
from .schemas import ValidationError
from .service import AppService, ConflictError, NotFoundError, ServiceError

MIME_FALLBACK = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
}


class ApiHandler(BaseHTTPRequestHandler):
    service: AppService
    server_version = "AStockAIScreener/1.2"

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # 降低噪音
        sys.stderr.write("[dev_server] " + (fmt % args) + "\n")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, exc: Exception) -> None:
        if isinstance(exc, NotFoundError):
            status = 404
        elif isinstance(exc, ConflictError):
            status = 409
        elif isinstance(exc, ValidationError):
            status = 400
        elif isinstance(exc, (DataError,)):
            status = 503
        elif isinstance(exc, ServiceError):
            status = 400
        else:
            status = 500
        self._send_json(status, {"error": str(exc)})

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValidationError("非法 Content-Length") from exc
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise ValidationError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ValidationError("请求体必须是 JSON 对象")
        return data

    def _stream_sse(self, events: Iterator[tuple[str, dict[str, Any]]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        try:
            for event, data in events:
                self.wfile.write(sse_format(event, data).encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能发 error
            try:
                self.wfile.write(sse_format("error", {"message": str(exc)}).encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        try:
            m = re.match(r"^/api/sessions/([^/]+)/messages$", path)
            if path == "/api/health":
                return self._send_json(200, self.service.health())
            if path == "/api/versions":
                return self._send_json(200, self.service.versions())
            if m:
                return self._send_json(200, self.service.messages(m.group(1)))
            m = re.match(r"^/api/sessions/([^/]+)/profile$", path)
            if m:
                return self._send_json(200, self.service.get_profile(m.group(1)))
            m = re.match(r"^/api/sessions/([^/]+)/selection/events$", path)
            if m:
                task_id = (query.get("task_id") or [""])[0]
                return self._stream_sse(self.service.selection_events(m.group(1), task_id))
            m = re.match(r"^/api/sessions/([^/]+)/tune/events$", path)
            if m:
                job_id = (query.get("job_id") or [""])[0]
                return self._stream_sse(self.service.tuning_events(m.group(1), job_id))
            if path.startswith("/api/"):
                raise NotFoundError(f"未知接口: {path}")
            return self._serve_static(path)
        except Exception as exc:  # noqa: BLE001
            return self._send_error(exc)

    # ------------------------------------------------------------------
    # POST / PUT
    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/sessions":
                return self._send_json(200, self.service.create_session())
            m = re.match(r"^/api/sessions/([^/]+)/messages$", path)
            if m:
                sid = m.group(1)
                content = str(body.get("content", ""))
                self.service._require_session(sid)  # noqa: SLF001 - 提前校验，避免流开始后报错
                return self._stream_sse(self.service.chat_events(sid, content))
            m = re.match(r"^/api/sessions/([^/]+)/selection$", path)
            if m:
                return self._send_json(200, self.service.start_selection(m.group(1), int(body.get("top_n", 10))))
            m = re.match(r"^/api/sessions/([^/]+)/tune$", path)
            if m:
                return self._send_json(200, self.service.start_tuning(m.group(1), int(body.get("trials", 24))))
            m = re.match(r"^/api/versions/([^/]+)/(activate|rollback)$", path)
            if m:
                vid, action = m.group(1), m.group(2)
                reason = str(body.get("reason", "user_action"))
                if action == "activate":
                    return self._send_json(200, self.service.activate_version(vid, reason))
                return self._send_json(200, self.service.rollback_version(vid, reason))
            raise NotFoundError(f"未知接口: {path}")
        except Exception as exc:  # noqa: BLE001
            return self._send_error(exc)

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            m = re.match(r"^/api/sessions/([^/]+)/profile$", path)
            if not m:
                raise NotFoundError(f"未知接口: {path}")
            slots = body.get("slots") if isinstance(body.get("slots"), dict) else body
            confirmed = bool(body.get("confirmed", False))
            return self._send_json(200, self.service.put_profile(m.group(1), slots, confirmed))
        except Exception as exc:  # noqa: BLE001
            return self._send_error(exc)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ------------------------------------------------------------------
    # 静态资源
    # ------------------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        target = (FRONTEND_DIR / path.lstrip("/")).resolve()
        try:
            target.relative_to(FRONTEND_DIR.resolve())
        except ValueError as exc:
            raise NotFoundError("非法路径") from exc
        if not target.exists() or not target.is_file():
            raise NotFoundError(f"文件不存在: {path}")
        suffix = target.suffix.lower()
        content_type = MIME_FALLBACK.get(suffix) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A 股 AI 选股 MVP 标准库预览服务器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    print("[dev_server] 初始化服务（加载数据缓存与 LLM 配置）...")
    try:
        service = AppService()
    except Exception as exc:  # noqa: BLE001
        print(f"[dev_server] 启动失败：{exc}")
        print("[dev_server] 提示：先运行 python -m backend.data.build_cache --source sample；"
              "无 Key 时可设置 LLM_PROVIDER=fake 仅预览 UI。")
        return 2

    ApiHandler.service = service
    httpd = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    httpd.daemon_threads = True
    print(f"[dev_server] 已启动: http://{args.host}:{args.port}  "
          f"(source={service.data.meta.get('source')}, as_of={service.data.as_of})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dev_server] 停止")
    finally:
        httpd.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
