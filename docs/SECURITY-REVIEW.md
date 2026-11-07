# 安全复查（2026-11-07）

## 范围

针对本机 `http://127.0.0.1:8000` 的浏览器侧攻击面复查，重点验证：

- `AUTH_COOKIE_SECURE=1` 是否真正生效；
- 未登录/普通用户能否越权；
- CSRF、路径穿越、Prompt Injection 是否能获取 DeepSeek Key。

测试仅访问本机自有服务，未对任何外部目标发起攻击。

## 结论：Secure Cookie 设置已生效

登录响应：

```text
Set-Cookie: zhitou_admin_session=...; HttpOnly; Max-Age=43200; Path=/; SameSite=strict; Secure
```

真实 Edge 浏览器（headless，访问 `http://127.0.0.1`）：

```text
LOGIN=200 | SESSION=200 | USER=browser_cookie_test ROLE=admin | JS_COOKIE=<empty>
```

说明：

- `Secure` 设置确实生效；
- 现代浏览器把 localhost/127.0.0.1 视为安全上下文，HTTP 下也会保存并回传 Secure Cookie，所以正常登录不受影响；
- `HttpOnly` 生效，JavaScript 读不到会话 Cookie；
- 之前用 Python `http.cookiejar` 测试时，严格 HTTP 客户端不会回传 Secure Cookie，导致“登录失效”的误判；不是服务端设置未生效。

## 其他实测

| 项目 | 结果 |
|---|---|
| `/.env`、`/backend/config.py`、`/data/app.db`、`/data/sync_status.json`、路径穿越 | 全部 404 |
| 未登录访问业务/管理 API | 401 |
| 普通用户访问管理 API | 403 |
| 普通用户读取管理员会话 | 404 |
| 伪造 Cookie / Bearer / `X-User` / `X-Admin` / `X-Forwarded-For` | 401 |
| POST 携带恶意 `Origin` | 403 |
| Prompt Injection 要求输出 Key / `.env` | 模型拒绝，未出现任何真实密钥 |
| 前端静态 JS/CSS/HTML 扫描 | 无 DeepSeek/Tushare 密钥 |

## 配置语义

`AUTH_COOKIE_SECURE`：

- `auto`（默认）：按 HTTPS 或 `X-Forwarded-Proto: https` 自动判断；
- `1`：强制 Secure；生产 HTTPS 推荐；
- `0`：强制不 Secure；仅本机 HTTP 调试。

## 残余风险

- 本机 HTTP 仍不防链路窃听；公网/局域网部署必须使用 HTTPS 反向代理。
- 能直接读取服务器 `.env` 或 `data/app.db` 的人不受浏览器登录保护。
- 无 MFA、无 SSO、无登录审计流水；高风险环境应接入机构统一身份认证。
- 仍需持续防范 XSS；CSP 已限制脚本来源，但不代表前端绝对无漏洞。
