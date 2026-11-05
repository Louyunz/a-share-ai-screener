# -*- coding: utf-8 -*-
"""FastAPI 应用：REST + SSE + 前端静态托管（单进程同时提供页面与 API）。

启动：
    uvicorn backend.main:app --port 8000
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR
from .events import sse_format
from .schemas import ValidationError
from .service import (
    AppService,
    ConflictError,
    NotFoundError,
    ServiceError,
)
from .config import ConfigError
from .data import DataError

_service: AppService | None = None


def get_service() -> AppService:
    """首次请求时初始化服务（数据缺失/未配置 Key 时返回明确错误）。"""
    global _service
    if _service is None:
        _service = AppService()
    return _service


def _error(exc: Exception) -> JSONResponse:
    if isinstance(exc, NotFoundError):
        return JSONResponse({"error": str(exc)}, status_code=404)
    if isinstance(exc, ConflictError):
        return JSONResponse({"error": str(exc)}, status_code=409)
    if isinstance(exc, ValidationError):
        return JSONResponse({"error": str(exc)}, status_code=400)
    if isinstance(exc, (ConfigError, DataError)):
        return JSONResponse({"error": str(exc)}, status_code=503)
    if isinstance(exc, ServiceError):
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"error": f"服务异常：{exc}"}, status_code=500)


def _sse(events: Iterator[tuple[str, dict[str, Any]]]) -> StreamingResponse:
    def generator() -> Iterator[str]:
        try:
            for event, data in events:
                yield sse_format(event, data)
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能发送 error 事件
            yield sse_format("error", {"message": str(exc)})
    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"请求体不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return data


def create_app(service: AppService | None = None) -> FastAPI:
    global _service
    _service = service
    app = FastAPI(title="智投罗盘 · A股 AI 选股 MVP", version="1.2.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ---------------- 基础 ----------------
    @app.get("/api/health")
    def health() -> Any:
        try:
            return get_service().health()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ---------------- 会话与对话 ----------------
    @app.post("/api/sessions")
    def create_session() -> Any:
        try:
            return get_service().create_session()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/messages")
    def messages(sid: str) -> Any:
        try:
            return get_service().messages(sid)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/sessions/{sid}/messages")
    async def post_message(sid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            content = str(body.get("content", ""))
            service = get_service()
            service._require_session(sid)  # noqa: SLF001 - 提前校验，避免 SSE 已开始才报错
            return _sse(service.chat_events(sid, content))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/profile")
    def get_profile(sid: str) -> Any:
        try:
            return get_service().get_profile(sid)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.put("/api/sessions/{sid}/profile")
    async def put_profile(sid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            slots = body.get("slots") if isinstance(body.get("slots"), dict) else body
            confirmed = bool(body.get("confirmed", False))
            return get_service().put_profile(sid, slots, confirmed)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ---------------- 选股 ----------------
    @app.post("/api/sessions/{sid}/selection")
    async def start_selection(sid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            top_n = int(body.get("top_n", 10))
            return get_service().start_selection(sid, top_n)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/selection/events")
    def selection_events(sid: str, task_id: str) -> Any:
        try:
            return _sse(get_service().selection_events(sid, task_id))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ---------------- 调优 ----------------
    @app.post("/api/sessions/{sid}/tune")
    async def start_tuning(sid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            trials = int(body.get("trials", 24))
            return get_service().start_tuning(sid, trials)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/tune/events")
    def tuning_events(sid: str, job_id: str) -> Any:
        try:
            return _sse(get_service().tuning_events(sid, job_id))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ---------------- 版本 ----------------
    @app.get("/api/versions")
    def versions() -> Any:
        try:
            return get_service().versions()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/versions/{vid}/activate")
    async def activate(vid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            return get_service().activate_version(vid, str(body.get("reason", "user_action")))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/versions/{vid}/rollback")
    async def rollback(vid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            return get_service().rollback_version(vid, str(body.get("reason", "user_rollback")))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ---------------- 前端静态资源（必须最后挂载） ----------------
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
