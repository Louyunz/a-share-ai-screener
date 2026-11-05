# -*- coding: utf-8 -*-
"""标准库 HTTP 服务器：无 fastapi 时本地运行同一套对话服务（仅开发/预览）。"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from .config import FRONTEND_DIR
from .data import DataError
from .events import sse_format
from .llm import LLMError
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
    server_version = "AStockAIAssistant/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # 降低噪音
        sys.stderr.write("[dev_server] " + (fmt % args) + "\n")

    # ------------------------------------------------------------------
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
        elif isinstance(exc, (LLMError, DataError)):
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
        except Exception as exc:  # noqa: BLE001 - 流已开始
            try:
                self.wfile.write(sse_format("error", {"message": str(exc)}).encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                return self._send_json(200, self.service.health())
            if path == "/api/versions":
                active = self.service.store.active_version()
                return self._send_json(200, {
                    "versions": [self.service._version_public(item)
                                 for item in self.service.store.list_versions()],
                    "active_id": active["id"] if active else None})
            match = re.match(r"^/api/sessions/([^/]+)/messages$", path)
            if match:
                return self._send_json(200, self.service.messages(match.group(1)))
            match = re.match(r"^/api/sessions/([^/]+)/state$", path)
            if match:
                return self._send_json(200, self.service.get_state(match.group(1)))
            if path.startswith("/api/"):
                raise NotFoundError(f"未知接口：{path}")
            return self._serve_static(path)
        except Exception as exc:  # noqa: BLE001
            return self._send_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/sessions":
                return self._send_json(200, self.service.create_session())
            match = re.match(r"^/api/sessions/([^/]+)/chat$", path)
            if match:
                session_id = match.group(1)
                content = str(body.get("content", ""))
                self.service._require_session(session_id)  # noqa: SLF001
                return self._stream_sse(self.service.chat_stream(session_id, content))
            raise NotFoundError(f"未知接口：{path}")
        except Exception as exc:  # noqa: BLE001
            return self._send_error(exc)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

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
            raise NotFoundError(f"文件不存在：{path}")
        content_type = (MIME_FALLBACK.get(target.suffix.lower())
                        or mimetypes.guess_type(str(target))[0]
                        or "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A 股 AI 选股助手（标准库预览服务器）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    print("[dev_server] 初始化服务（加载数据缓存与模型配置）...")
    try:
        service = AppService()
    except Exception as exc:  # noqa: BLE001
        print(f"[dev_server] 启动失败：{exc}")
        print("[dev_server] 提示：先运行 python -m backend.data.build_cache --source sample；"
              "本地测试可设置 LLM_PROVIDER=fake。")
        return 2

    ApiHandler.service = service
    httpd = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    httpd.daemon_threads = True
    print(f"[dev_server] 已启动：http://{args.host}:{args.port} "
          f"(数据 {service.data.as_of}，模型 {type(service.llm).__name__})")
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
