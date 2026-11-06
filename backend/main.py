# -*- coding: utf-8 -*-
"""FastAPI 入口：单页应用 + 对话 SSE。

启动：``uvicorn backend.main:app --port 8000``
"""
from __future__ import annotations

from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR
from .data import DataError
from .events import sse_format
from .llm import LLMError
from .schemas import ValidationError
from .service import AppService, ConflictError, NotFoundError, ServiceError

_service: AppService | None = None


def get_service() -> AppService:
    global _service
    if _service is None:
        _service = AppService()
    return _service


def _error(exc: Exception) -> JSONResponse:
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
    return JSONResponse({"error": str(exc)}, status_code=status)


def _sse(events: Iterator[tuple[str, dict[str, Any]]]) -> StreamingResponse:
    def generator() -> Iterator[str]:
        try:
            for event, data in events:
                yield sse_format(event, data)
        except Exception as exc:  # noqa: BLE001 - 响应已开始，只能发 error 事件
            yield sse_format("error", {"message": str(exc), "retryable": False})

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"请求体不是合法 JSON：{exc}") from exc
    if not isinstance(body, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return body


def create_app(service: AppService | None = None) -> FastAPI:
    global _service
    _service = service
    app = FastAPI(title="智投罗盘 · AI 选股助手", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health() -> Any:
        try:
            return get_service().health()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions")
    def list_sessions() -> Any:
        try:
            return get_service().list_sessions()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/sessions")
    def create_session() -> Any:
        try:
            return get_service().create_session()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/sessions/{sid}/chat")
    async def chat(sid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            content = str(body.get("content", ""))
            service = get_service()
            service._require_session(sid)  # noqa: SLF001 - 提前校验，避免 SSE 已开始才报错
            return _sse(service.chat_stream(sid, content))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/messages")
    def messages(sid: str) -> Any:
        try:
            return get_service().messages(sid)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/state")
    def state(sid: str) -> Any:
        try:
            return get_service().get_state(sid)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/versions")
    def versions(request: Request) -> Any:
        try:
            session_id = request.query_params.get("session") or None
            return get_service().versions(session_id)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app


app = create_app()
