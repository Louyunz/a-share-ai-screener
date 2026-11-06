# -*- coding: utf-8 -*-
"""单管理员认证：PBKDF2 密码哈希、服务端会话、登录限速。

设计目标：
- 密码只存 PBKDF2-HMAC-SHA256 哈希，绝不明文落库；
- 浏览器只持有随机 256bit 会话令牌，服务端只存令牌的 SHA-256；
- Cookie 为 HttpOnly + SameSite=Strict；HTTPS 下可开 Secure；
- 首次初始化只允许本机访问，之后设置关闭；
- 连续登录失败触发短时锁定，避免在线暴力猜密码。
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


class AuthError(RuntimeError):
    """认证/初始化失败；status 直接映射为 HTTP 状态码。"""

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


class AuthManager:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 账号状态
    # ------------------------------------------------------------------
    def _env_user(self) -> dict[str, str] | None:
        if AUTH_USERNAME and AUTH_PASSWORD_HASH:
            return {"username": AUTH_USERNAME, "password_hash": AUTH_PASSWORD_HASH}
        return None

    def current_user(self) -> dict[str, Any] | None:
        return self._env_user() or self.store.auth_user()

    def configured(self) -> bool:
        return self.current_user() is not None

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

    def _issue_session(self) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = int(time.time() + AUTH_SESSION_HOURS * 3600)
        self.store.purge_expired_auth_sessions(int(time.time()))
        self.store.create_auth_session(self._token_hash(token), expires_at)
        return token

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def setup(self, request: Request, username: str, password: str) -> str:
        if self.configured():
            raise AuthError("管理员账号已初始化，不能重复设置", 409)
        if not self.setup_allowed(request):
            raise AuthError("初始化只允许在服务器本机完成", 403)
        username = validate_username(username)
        password = validate_password(password)
        if not self.store.create_auth_user(username, hash_password(password)):
            raise AuthError("管理员账号已初始化，不能重复设置", 409)
        return self._issue_session()

    def login(self, request: Request, username: str, password: str) -> str:
        client_key = self._client_host(request) or "unknown"
        self._check_rate(client_key)
        user = self.current_user()
        if user is None:
            raise AuthError("管理员账号尚未初始化", 409)
        if not hmac.compare_digest(str(user.get("username") or ""), str(username or "").strip()):
            self._record_failure(client_key)
            raise AuthError("账号或密码不正确", 401)
        if not verify_password(str(password or ""), str(user.get("password_hash") or "")):
            self._record_failure(client_key)
            raise AuthError("账号或密码不正确", 401)
        self._clear_failures(client_key)
        return self._issue_session()

    def authenticated(self, request: Request) -> dict[str, Any] | None:
        token = request.cookies.get(AUTH_COOKIE_NAME)
        if not token:
            return None
        token_hash = self._token_hash(token)
        expires_at = self.store.auth_session_expiry(token_hash)
        if not expires_at or expires_at <= int(time.time()):
            if expires_at:
                self.store.delete_auth_session(token_hash)
            return None
        return self.current_user()

    def logout(self, request: Request) -> None:
        token = request.cookies.get(AUTH_COOKIE_NAME)
        if token:
            self.store.delete_auth_session(self._token_hash(token))

    # ------------------------------------------------------------------
    # Cookie
    # ------------------------------------------------------------------
    def set_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            AUTH_COOKIE_NAME, token,
            max_age=int(AUTH_SESSION_HOURS * 3600),
            httponly=True, samesite="strict", secure=AUTH_COOKIE_SECURE, path="/",
        )

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
        }
