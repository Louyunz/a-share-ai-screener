# -*- coding: utf-8 -*-
"""FastAPI 入口：登录保护的单页应用 + 对话 SSE。

启动：``uvicorn backend.main:app --port 8000``
"""
from __future__ import annotations

from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .auth import AuthError, AuthManager
from .config import DB_PATH, FRONTEND_DIR
from .data import DataError
from .data.sync import SyncManager
from .events import sse_format
from .llm import LLMError
from .schemas import ValidationError
from .service import AppService, ConflictError, NotFoundError, ServiceError
from .store import Store

_service: AppService | None = None
_auth: AuthManager | None = None
_sync: SyncManager | None = None

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
        "img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'"
    ),
}


def get_service() -> AppService:
    global _service
    if _service is None:
        _service = AppService()
    return _service


def get_auth() -> AuthManager:
    global _auth
    if _auth is None:
        _auth = AuthManager(Store(DB_PATH))
    return _auth


def _error(exc: Exception) -> JSONResponse:
    if isinstance(exc, AuthError):
        status = exc.status
    elif isinstance(exc, NotFoundError):
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
    global _service, _auth, _sync
    _service = service
    _auth = AuthManager(service.store) if service is not None else None
    _sync = None
    app = FastAPI(title="智投罗盘 · AI 选股助手", version="2.1.0")

    @app.middleware("http")
    async def protect_api(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and request.method in UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin:
                expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
                if origin.rstrip("/") != expected:
                    return JSONResponse({"error": "跨站请求被拒绝"}, status_code=403)
        if path.startswith("/api/") and not path.startswith("/api/auth/"):
            user = get_auth().authenticated(request)
            if user is None:
                return JSONResponse({"error": "未登录或登录已过期"}, status_code=401)
            request.state.user = user
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if path.startswith("/api/") or path in {"/", "/login"}:
            response.headers.setdefault("Cache-Control", "no-store")
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    # ------------------------------------------------------------------
    # 认证
    # ------------------------------------------------------------------
    @app.get("/api/auth/status")
    def auth_status(request: Request) -> Any:
        return get_auth().status_payload(request)

    @app.get("/api/auth/session")
    def auth_session(request: Request) -> Any:
        user = get_auth().authenticated(request)
        if user is None:
            return JSONResponse({"error": "未登录或登录已过期"}, status_code=401)
        return {"authenticated": True, "username": str(user.get("username") or ""),
                "role": str(user.get("role") or "")}

    @app.post("/api/auth/setup")
    async def auth_setup(request: Request) -> Any:
        try:
            body = await _json_body(request)
            token = get_auth().setup(
                request, str(body.get("username", "")), str(body.get("password", "")))
            response = JSONResponse({"ok": True, "username": str(body.get("username", "")).strip()})
            get_auth().set_cookie(response, token, secure=get_auth().cookie_secure_for(request))
            return response
        except AuthError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.post("/api/auth/login")
    async def auth_login(request: Request) -> Any:
        try:
            body = await _json_body(request)
            token = get_auth().login(
                request, str(body.get("username", "")), str(body.get("password", "")))
            response = JSONResponse({"ok": True, "username": str(body.get("username", "")).strip()})
            get_auth().set_cookie(response, token, secure=get_auth().cookie_secure_for(request))
            return response
        except AuthError as exc:
            headers = {"Retry-After": "60"} if exc.status == 429 else None
            return JSONResponse({"error": str(exc)}, status_code=exc.status, headers=headers)

    @app.post("/api/auth/logout")
    def auth_logout(request: Request) -> Any:
        get_auth().logout(request)
        response = JSONResponse({"ok": True})
        get_auth().clear_cookie(response)
        return response

    # ------------------------------------------------------------------
    # 业务 API（全部需要登录）
    # ------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> Any:
        try:
            payload = get_service().health()
            if _sync is not None:
                payload["sync"] = _sync.status()
            return payload
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions")
    def list_sessions(request: Request) -> Any:
        try:
            user = getattr(request.state, "user", {})
            return get_service().list_sessions(owner_id=int(user["id"]))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/sessions")
    def create_session(request: Request) -> Any:
        try:
            user = getattr(request.state, "user", {})
            return get_service().create_session(owner_id=int(user["id"]))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/sessions/{sid}/chat")
    async def chat(sid: str, request: Request) -> Any:
        try:
            body = await _json_body(request)
            content = str(body.get("content", ""))
            user = getattr(request.state, "user", {})
            service = get_service()
            service._require_session(sid, int(user["id"]))  # noqa: SLF001
            return _sse(service.chat_stream(sid, content, int(user["id"])))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/messages")
    def messages(sid: str, request: Request) -> Any:
        try:
            user = getattr(request.state, "user", {})
            return get_service().messages(sid, owner_id=int(user["id"]))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/sessions/{sid}/state")
    def state(sid: str, request: Request) -> Any:
        try:
            user = getattr(request.state, "user", {})
            return get_service().get_state(sid, owner_id=int(user["id"]))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.get("/api/versions")
    def versions(request: Request) -> Any:
        try:
            session_id = request.query_params.get("session") or None
            user = getattr(request.state, "user", {})
            return get_service().versions(session_id, owner_id=int(user["id"]))
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ------------------------------------------------------------------
    # 管理员：用户管理
    # ------------------------------------------------------------------
    def _admin(request: Request) -> dict[str, Any]:
        user = getattr(request.state, "user", None)
        if not user:
            raise AuthError("未登录或登录已过期", 401)
        get_auth().require_admin(user)
        return user

    @app.get("/api/admin/users")
    def admin_users(request: Request) -> Any:
        try:
            return {"users": get_auth().list_users(_admin(request))}
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/admin/users")
    async def admin_create_user(request: Request) -> Any:
        try:
            actor = _admin(request)
            body = await _json_body(request)
            user = get_auth().create_user(
                actor, str(body.get("username", "")), str(body.get("password", "")),
                str(body.get("role", "user")))
            get_auth().store.audit(
                "user_create", reason="admin", payload={
                    "actor": actor.get("username"), "target": user.get("username"),
                    "role": user.get("role"),
                })
            return user
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.patch("/api/admin/users/{user_id}")
    async def admin_update_user(user_id: int, request: Request) -> Any:
        try:
            actor = _admin(request)
            body = await _json_body(request)
            active = body.get("active")
            user = get_auth().update_user(
                actor, int(user_id),
                password=str(body["password"]) if "password" in body else None,
                role=str(body["role"]) if "role" in body else None,
                active=None if active is None else bool(active))
            get_auth().store.audit(
                "user_update", reason="admin", payload={
                    "actor": actor.get("username"), "target": user.get("username"),
                    "active": user.get("active"), "role": user.get("role"),
                })
            return user
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.delete("/api/admin/users/{user_id}")
    def admin_delete_user(user_id: int, request: Request) -> Any:
        try:
            actor = _admin(request)
            target = get_auth().store.auth_user_by_id(int(user_id))
            get_auth().delete_user(actor, int(user_id))
            get_auth().store.audit(
                "user_delete", reason="admin", payload={
                    "actor": actor.get("username"),
                    "target": (target or {}).get("username", ""),
                })
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ------------------------------------------------------------------
    # 管理员：行情同步
    # ------------------------------------------------------------------
    @app.get("/api/admin/data/sync")
    def admin_sync_status(request: Request) -> Any:
        try:
            _admin(request)
            if _sync is None:
                return {"status": "disabled", "message": "同步任务尚未启动"}
            return _sync.status()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @app.post("/api/admin/data/sync")
    def admin_sync_now(request: Request) -> Any:
        try:
            actor = _admin(request)
            if _sync is None:
                raise AuthError("同步任务尚未启动", 503)
            get_auth().store.audit(
                "data_sync", reason="admin_manual", payload={"actor": actor.get("username")})
            return _sync.trigger()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    # ------------------------------------------------------------------
    # 页面
    # ------------------------------------------------------------------
    @app.get("/")
    def index(request: Request):
        if get_auth().authenticated(request) is None:
            return RedirectResponse(url="/login", status_code=303)
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/login")
    def login_page():
        return FileResponse(str(FRONTEND_DIR / "login.html"))

    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.on_event("startup")
    def _start_daily_sync() -> None:
        global _sync
        if _sync is None:
            _sync = SyncManager(get_service())
            _sync.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        global _auth, _sync
        if _sync is not None:
            _sync.stop()
            _sync = None
        if _auth is not None:
            _auth.store.close()
            _auth = None

    return app


app = create_app()
