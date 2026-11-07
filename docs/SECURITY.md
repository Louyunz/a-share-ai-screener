# 安全说明（管理员登录与部署）

## 已实现

- **多用户登录与数据隔离**：每个账号只看到自己的会话、画像和模型版本；管理员可以创建用户、停用/启用账号、重置密码、调整角色（至少保留一个管理员）。
- **所有业务 API 默认需要登录**：`/api/auth/*` 之外的所有 `/api/**` 在未认证时返回 401；管理员接口额外校验 admin 角色。
- **密码哈希**：PBKDF2-HMAC-SHA256（390,000 次迭代、随机盐）；密码明文不落库、不回显。重置密码会立刻注销该用户的所有会话。
- **服务端会话**：浏览器只持有随机 256-bit 会话令牌（HttpOnly Cookie）；数据库只存令牌的 SHA-256，可服务端注销。
- **Cookie**：默认启用 `Secure`，并使用 `HttpOnly` + `SameSite=Strict` + `Path=/`；仅本机 HTTP 开发调试时才应显式设置 `AUTH_COOKIE_SECURE=0`。
- **CSRF**：SameSite=Strict，并对带 `Origin` 的写请求校验同源；已移除通配 CORS。
- **暴力破解**：同一来源 5 分钟内连续失败 5 次后短时锁定。
- **首次初始化**：没有账号时，只允许从服务器本机（loopback）创建管理员；设置完成后初始化接口自动关闭。
- **安全响应头**：`X-Content-Type-Options`、`X-Frame-Options: DENY`、`Referrer-Policy`、CSP、COOP/CORP；HTTPS 下附加 HSTS。
- **预览服务器**：`backend.dev_server` 不带登录，默认拒绝启动；仅本机临时预览可显式设置 `DEV_SERVER_ALLOW_INSECURE=1`。

## 配置管理员账号（生产/长期使用）

首次打开页面创建管理员即可。也可以把账号写进 `.env`：

```powershell
.\.venv\Scripts\python.exe scripts\set_password.py
```

把输出的 `AUTH_USERNAME` / `AUTH_PASSWORD_HASH` 写入 `.env`，重启服务。设置后初始化页面会自动关闭。

## 必须知道的边界

**不存在“绝对安全”**。当前项目是本地单机、小规模多用户演示：

- 默认通过 HTTP 暴露在 `127.0.0.1`，但 Cookie 默认带 `Secure`，因此 HTTP 访问不会发送管理员会话 Cookie；如果要在局域网/公网使用，**必须**放在 HTTPS 反向代理后，禁止关闭 `AUTH_COOKIE_SECURE`，否则 Cookie 和密码可能在链路上被截获。
- SQLite 数据文件本身不加密；能读到 `data/app.db` 的人仍能看到会话内容。生产环境需要磁盘加密、最小权限运行账号和备份加密。
- 账号体系只有 admin/user 两种角色，无 MFA、无 SSO/LDAP；需要机构级合规时应接入统一身份认证。
- 前端仍有内联样式和动态 `innerHTML` 渲染卡片；CSP 已限制脚本来源，但 XSS 防护仍需持续审计。
- 目前记录用户管理操作审计，没有登录成功/失败审计流水；登录只做限速。

上线检查清单：

1. HTTPS 反向代理，并保持 `AUTH_COOKIE_SECURE=1`；
2. 仅监听内网/本机，不用 `--host 0.0.0.0` 直接暴露；
3. 使用强密码（≥ 10 位，字母+数字；建议 16 位以上随机密码）；
4. 定期更新依赖与备份 `data/app.db`（备份本身要加密）。
