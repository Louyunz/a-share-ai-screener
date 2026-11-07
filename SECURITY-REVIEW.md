# LLM 流式输出审查

审查对象：`3cd286b (v1.1.0) feat(stream): LLM 真流式输出，画像更新改用 update_profile 工具`

审查范围：

- `backend/llm.py` 的文本流式输出与流式工具调用
- `backend/agent.py` 的工具规划与参数处理
- `backend/service.py` 的会话流与状态更新
- `backend/main.py` 的 SSE 边界
- 相关测试与调用链

## 发现

| # | Severity | File | Lines | Vulnerability | Confidence |
|---|----------|------|-------|---------------|------------|
| 1 | 🟠 HIGH | `backend/llm.py` | 148, 250 | 流式请求将 `Authorization` 固定为字面量 `******`，未使用 `self.api_key`，导致真实 LLM 服务认证失败。 | 10/10 |
| 2 | 🟡 MEDIUM | `backend/service.py` | 191-280 | 同一会话的流式请求没有互斥或乐观并发控制，可能交错更新画像、待审批方案及模型版本。 | 8/10 |
| 3 | 🟡 MEDIUM | `backend/llm.py` | 164-173, 264-286 | SSE 解析未统一处理 `UnicodeDecodeError`、`ValueError` 及异常 chunk 结构，异常供应商响应会中断流。 | 9/10 |
| 4 | ⚪ LOW | `backend/llm.py` | 264-308 | 流式工具调用依赖整数 `index` 和字符串片段格式，兼容网关缺少字段或返回不同类型时可能丢失调用或失败。 | 9/10 |

## 影响与建议

1. 统一封装请求头并使用 `Authorization: Bearer {self.api_key}`；增加 HTTP mock 测试验证非流式和流式请求均携带真实认证头。
2. 为 `session_id` 增加互斥锁，或使用数据库事务/版本号实现乐观并发控制，并补充并发聊天、审批和回滚测试。
3. 对 SSE 行、JSON chunk、工具调用索引和字段类型做结构化校验，将解析异常统一转换为 `LLMError`。
4. 使用稳定的 call-id 映射处理缺失或非法 `index`，并覆盖不同 OpenAI 兼容实现的流式工具调用格式。

## 验证

已运行：

```text
python -m pytest -q tests/test_agent.py tests/test_service.py tests/test_chat_contract.py
49 passed, 1 warning
```

现有测试未覆盖真实 Authorization 请求头、同会话并发一致性或异常 SSE chunk。

## 管理员认证与密钥暴露审查

审查范围：网页前端及相关 API，重点检查无需密码绕过管理员登录和获取
DeepSeek API key 的路径。审查过程中未读取、导出或披露任何密钥值。

| # | Severity | File | Lines | Vulnerability | Confidence |
|---|----------|------|-------|---------------|------------|
| 1 | 🟠 HIGH | `backend/auth.py` | 134-147 | 初始管理员设置仅信任 `request.client.host` 判断本地请求。反向代理部署时，远程请求可能被应用识别为 loopback，从而调用 `/api/auth/setup` 并设置攻击者可控的管理员密码。 | 8/10 |
| 2 | 🟠 HIGH | `backend/config.py` | 63 | `AUTH_COOKIE_SECURE` 默认关闭。若服务通过 HTTP 暴露，管理员会话 cookie 可能以明文传输并被窃取，攻击者可复用会话而无需密码。 | 9/10 |

### 修复建议

1. 默认关闭管理员初始化，并要求一次性 bootstrap secret 或已认证的本地
   管理通道。若保留代理感知的 IP 判断，应显式配置并校验可信代理地址。
2. 默认启用 `AUTH_COOKIE_SECURE`，非 loopback 部署强制 HTTPS，并拒绝或重定向
   生产环境中的不安全请求。
3. DeepSeek API key 仅保留在后端受保护的环境变量或密钥管理系统中，确保
   API 响应和日志不会返回该 key。
