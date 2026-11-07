# -*- coding: utf-8 -*-
"""多用户认证：PBKDF2 密码哈希、服务端会话、登录限速、管理员用户管理。

- 第一个账号由首次初始化创建，角色固定为 admin；
- 后续由管理员创建普通用户或管理员；
- 密码只存 PBKDF2-HMAC-SHA256 哈希，会话只存令牌 SHA-256；
- Cookie 为 HttpOnly + SameSite=Strict；HTTPS 下可开 Secure；
- 连续登录失败触发短时锁定。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import re
import secrets
import threading
import time
from typing import Any

from fastapi import Request
from fastapi.responses import Response

from .config import (
    AUTH_ALLOW_SETUP,
    AUTH_COOKIE_NAME,
    AUTH_COOKIE_SECURE,
    AUTH_PASSWORD_HASH,
    AUTH_SESSION_HOURS,
    AUTH_USERNAME,
)
from .store import Store

PBKDF2_ITERATIONS = 390_000
LOGIN_WINDOW_SECONDS = 300
LOGIN_MAX_FAILURES = 5
MIN_PASSWORD_LENGTH = 10
ROLES = {"admin", "user"}


class AuthError(RuntimeError):
    """认证/用户管理失败；status 直接映射为 HTTP 状态码。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        iterations = int(iterations_text)
    except (AttributeError, ValueError):
        return False
    if scheme != "pbkdf2_sha256" or not 100_000 <= iterations <= 2_000_000:
        return False
    try:
        salt = _unb64(salt_text)
        expected = _unb64(digest_text)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def validate_username(value: str) -> str:
    username = str(value or "").strip()
    if not 3 <= len(username) <= 32 or re.search(r"\s", username):
        raise AuthError("用户名需为 3~32 个不含空白的字符")
    return username


def validate_password(value: str) -> str:
    password = str(value or "")
    if not MIN_PASSWORD_LENGTH <= len(password) <= 128:
        raise AuthError(f"密码长度需为 {MIN_PASSWORD_LENGTH}~128 位")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("密码需同时包含字母和数字")
    return password


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user["id"]),
        "username": str(user["username"]),
        "role": str(user["role"]),
        "active": bool(user["active"]),
        "created_at": str(user.get("created_at") or ""),
        "updated_at": str(user.get("updated_at") or ""),
        "last_login_at": str(user.get("last_login_at") or ""),
    }


class AuthManager:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._bootstrap_env_admin()

    def _bootstrap_env_admin(self) -> None:
        if not (AUTH_USERNAME and AUTH_PASSWORD_HASH):
            return
        if self.store.list_auth_users():
            return
        user_id = self.store.create_auth_user(
            AUTH_USERNAME, AUTH_PASSWORD_HASH, role="admin", active=1)
        if user_id is not None:
            self.store.claim_orphan_sessions(user_id)

    # ------------------------------------------------------------------
    # 账号状态
    # ------------------------------------------------------------------
    def configured(self) -> bool:
        return bool(self.store.list_auth_users())

    def setup_required(self) -> bool:
        return not self.configured() and AUTH_ALLOW_SETUP

    @staticmethod
    def _client_host(request: Request) -> str:
        return str(getattr(request.client, "host", "") or "")

    def setup_allowed(self, request: Request) -> bool:
        if self.configured() or not AUTH_ALLOW_SETUP:
            return False
        host = self._client_host(request)
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host in {"localhost", "testclient"}

    # ------------------------------------------------------------------
    # 登录/登出/初始化
    # ------------------------------------------------------------------
    def _check_rate(self, client_key: str) -> None:
        now = time.time()
        with self._lock:
            attempts = [item for item in self._failures.get(client_key, [])
                        if now - item < LOGIN_WINDOW_SECONDS]
            self._failures[client_key] = attempts
            if len(attempts) >= LOGIN_MAX_FAILURES:
                retry = int(LOGIN_WINDOW_SECONDS - (now - attempts[0])) + 1
                raise AuthError(f"登录尝试过于频繁，请 {retry} 秒后再试", 429)

    def _record_failure(self, client_key: str) -> None:
        with self._lock:
            self._failures.setdefault(client_key, []).append(time.time())

    def _clear_failures(self, client_key: str) -> None:
        with self._lock:
            self._failures.pop(client_key, None)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _issue_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = int(time.time() + AUTH_SESSION_HOURS * 3600)
        self.store.purge_expired_auth_sessions(int(time.time()))
        self.store.create_auth_session(self._token_hash(token), expires_at, user_id)
        return token

    def setup(self, request: Request, username: str, password: str) -> str:
        if self.configured():
            raise AuthError("管理员账号已初始化，不能重复设置", 409)
        if not self.setup_allowed(request):
            raise AuthError("初始化只允许在服务器本机完成", 403)
        username = validate_username(username)
        password = validate_password(password)
        user_id = self.store.create_auth_user(
            username, hash_password(password), role="admin", active=1)
        if user_id is None:
            raise AuthError("管理员账号已初始化，不能重复设置", 409)
        self.store.claim_orphan_sessions(user_id)
        return self._issue_session(user_id)

    def login(self, request: Request, username: str, password: str) -> str:
        client_key = self._client_host(request) or "unknown"
        self._check_rate(client_key)
        user = self.store.auth_user_by_username(username)
        if user is None:
            self._record_failure(client_key)
            raise AuthError("账号或密码不正确", 401)
        if not verify_password(str(password or ""), str(user.get("password_hash") or "")):
            self._record_failure(client_key)
            raise AuthError("账号或密码不正确", 401)
        if not bool(user.get("active")):
            raise AuthError("该账号已停用，请联系管理员", 403)
        self._clear_failures(client_key)
        self.store.update_auth_user(int(user["id"]), last_login=True)
        return self._issue_session(int(user["id"]))

    def authenticated(self, request: Request) -> dict[str, Any] | None:
        token = request.cookies.get(AUTH_COOKIE_NAME)
        if not token:
            return None
        token_hash = self._token_hash(token)
        session = self.store.auth_session(token_hash)
        if not session or int(session["expires_at"]) <= int(time.time()):
            if session:
                self.store.delete_auth_session(token_hash)
            return None
        user = self.store.auth_user_by_id(int(session["user_id"]))
        if user is None or not bool(user.get("active")):
            self.store.delete_auth_session(token_hash)
            return None
        return _public_user(user)

    def logout(self, request: Request) -> None:
        token = request.cookies.get(AUTH_COOKIE_NAME)
        if token:
            self.store.delete_auth_session(self._token_hash(token))

    # ------------------------------------------------------------------
    # 管理员：用户管理
    # ------------------------------------------------------------------
    @staticmethod
    def require_admin(user: dict[str, Any]) -> None:
        if str(user.get("role") or "") != "admin":
            raise AuthError("需要管理员权限", 403)

    def list_users(self, actor: dict[str, Any]) -> list[dict[str, Any]]:
        self.require_admin(actor)
        return [_public_user(user) for user in self.store.list_auth_users()]

    def create_user(self, actor: dict[str, Any], username: str, password: str,
                    role: str = "user") -> dict[str, Any]:
        self.require_admin(actor)
        role = str(role or "user")
        if role not in ROLES:
            raise AuthError("角色只能是 admin 或 user")
        username = validate_username(username)
        password = validate_password(password)
        user_id = self.store.create_auth_user(
            username, hash_password(password), role=role, active=1)
        if user_id is None:
            raise AuthError("用户名已存在", 409)
        user = self.store.auth_user_by_id(user_id)
        return _public_user(user or {})

    def update_user(self, actor: dict[str, Any], user_id: int, *,
                    password: str | None = None, role: str | None = None,
                    active: bool | None = None) -> dict[str, Any]:
        self.require_admin(actor)
        target = self.store.auth_user_by_id(user_id)
        if target is None:
            raise AuthError("用户不存在", 404)
        target_id = int(target["id"])
        if role is not None and role not in ROLES:
            raise AuthError("角色只能是 admin 或 user")
        demoting = role is not None and role != "admin" and target["role"] == "admin"
        disabling = active is False and target["role"] == "admin"
        if (demoting or disabling) and self.store.count_active_admins(exclude_id=target_id) == 0:
            raise AuthError("不能停用或降级最后一个管理员", 400)
        password_hash = hash_password(validate_password(password)) if password else None
        self.store.update_auth_user(
            target_id, password_hash=password_hash, role=role,
            active=None if active is None else int(active))
        if password_hash is not None or active is False:
            self.store.delete_auth_sessions_for_user(target_id)
        updated = self.store.auth_user_by_id(target_id)
        return _public_user(updated or {})

    def delete_user(self, actor: dict[str, Any], user_id: int) -> None:
        self.require_admin(actor)
        target = self.store.auth_user_by_id(user_id)
        if target is None:
            raise AuthError("用户不存在", 404)
        target_id = int(target["id"])
        if target_id == int(actor["id"]):
            raise AuthError("不能删除当前登录账号", 400)
        if target["role"] == "admin" and self.store.count_active_admins(exclude_id=target_id) == 0:
            raise AuthError("不能删除最后一个管理员", 400)
        if self.store.count_sessions(target_id):
            raise AuthError("该用户仍有关联会话数据，请先停用账号，不要直接删除", 409)
        self.store.delete_auth_sessions_for_user(target_id)
        self.store.delete_auth_user(target_id)

    # ------------------------------------------------------------------
    # Cookie / 状态
    # ------------------------------------------------------------------
    def set_cookie(self, response: Response, token: str, *, secure: bool = True) -> None:
        """Web 路由必须传入 cookie_secure_for(request)；默认 Secure 便于安全测试/默认安全。"""
        response.set_cookie(
            AUTH_COOKIE_NAME, token,
            max_age=int(AUTH_SESSION_HOURS * 3600),
            httponly=True, samesite="strict", secure=bool(secure), path="/",
        )

    @staticmethod
    def request_is_https(request: Request) -> bool:
        """直连 HTTPS 或反向代理声明 X-Forwarded-Proto: https。"""
        forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        return request.url.scheme == "https" or forwarded == "https"

    @classmethod
    def cookie_secure_for(cls, request: Request) -> bool:
        """显式 1/0 优先；auto 时根据 HTTPS/反向代理判断。"""
        mode = str(AUTH_COOKIE_SECURE or "auto").strip().lower()
        if mode in ("1", "true", "yes", "always", "force"):
            return True
        if mode in ("0", "false", "no", "never"):
            return False
        return cls.request_is_https(request)

    @staticmethod
    def clear_cookie(response: Response) -> None:
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")

    def status_payload(self, request: Request) -> dict[str, Any]:
        user = self.authenticated(request)
        return {
            "configured": self.configured(),
            "setup_required": self.setup_required() and not user,
            "setup_allowed": self.setup_allowed(request),
            "authenticated": user is not None,
            "username": str(user.get("username") or "") if user else "",
            "role": str(user.get("role") or "") if user else "",
        }
