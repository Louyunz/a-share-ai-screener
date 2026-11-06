# 智投罗盘

一个面向 A 股回测的 AI 选股原型。它会根据对话整理投资偏好，使用本地行情缓存做选股和回测，并把结果、模型版本和调参过程展示在网页中。

项目目前是单用户演示应用，不是实盘交易系统，也不构成投资建议。

## 功能

- 用对话收集投资期限、风险承受能力等信息。
- 基于 10 个因子筛选股票，展示指标、因子贡献、组合和净值曲线。
- 支持示例数据、AkShare 和 Tushare。数据源只在构建缓存时访问网络，服务运行时读取本地缓存。
- **原生 function calling**：Planner 通过 tools/tool_calls 表达动作与参数（例如 `analyze_model(engine="code")`），服务端只校验和执行，不再从聊天文本里用关键词猜意图。
- 支持内置模型和 AI 生成的参数化模型。AI 只决定「引擎 + 参数搜索范围」，服务端执行真实网格寻优并实时展示进度；生成代码会经过 AST 检查、子进程冒烟和回测。
- 真流式回复：模型文本以 SSE token 增量实时显示，前端收到即渲染；管理动作通过原生 tool_calls 表达，画像更新走 update_profile 工具。
- 代码改动卡也逐行流式输出，且**先于网格寻优卡**出现；效果对比独立成卡，**排在网格寻优结果之后**；寻优期间代码窗口保持可见，新卡片延迟平滑聚焦，不会在弹出瞬间把页面强行拉到底部。
- 执行模型优化时，对话里会说明**具体怎么改**（重写引擎/切换模型/搜索哪些因子或参数/目标指标），不只是“已开始优化”。
- 模型版本按会话隔离：候选模型需确认后才在当前会话启用，也可以回滚；新建对话回到默认「均衡多因子」。
- 前端是原生 JavaScript 和 SVG，无前端构建步骤。

## 快速开始

以下命令适用于 Windows PowerShell。项目要求 Python 3.12 或更高版本。

```powershell
git clone <repository-url>
cd a-share-ai-screener

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 生成离线示例数据
python -m backend.data.build_cache --source sample

# 配置模型密钥；也可以复制 .env.example 为 .env 后填写
$env:DEEPSEEK_API_KEY = "sk-..."

# 启动 FastAPI 应用
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>。首次启动必须先生成数据缓存；没有模型密钥时可以打开页面，但无法进行 AI 对话。

### 使用 `.env`

项目会自动读取根目录下的 `.env`，不会覆盖已经存在的环境变量：

```powershell
Copy-Item .env.example .env
```

至少填写：

```dotenv
DEEPSEEK_API_KEY=你的模型密钥
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

也可以使用其他 OpenAI 兼容接口，修改 `LLM_BASE_URL` 和 `LLM_MODEL` 即可。不要把密钥提交到 Git。

## 构建真实行情缓存

### AkShare

```powershell
python -m backend.data.build_cache `
  --source akshare `
  --universe csi800 `
  --limit 200 `
  --start 2022-01-01
```

### Tushare

先在 `.env` 中填写 `TUSHARE_TOKEN`，再执行：

```powershell
python -m backend.data.build_cache `
  --source tushare `
  --universe csi800 `
  --limit 200 `
  --start 2022-01-01
```

缓存默认写入 `data/a_share/`。构建完成后重启服务，服务不会在运行期间自动拉取行情。数据源权限不足时，适配器可能无法补齐部分财务或指数数据；终端输出会报告构建结果。

常用参数：

- `--universe`：`csi800`、`hs300` 或 `zz500`。
- `--limit`：限制真实数据拉取的股票数量，适合先做小规模验证。
- `--start`、`--end`：设置行情日期范围。
- `--out`：修改缓存目录；如果修改了目录，同时设置 `.env` 中的 `DATA_DIR`。

## 截图

| 对话与投资上下文 | 选股结果 |
|---|---|
| ![对话与投资上下文](docs/screenshots/01_ai_chat.png) | ![选股结果](docs/screenshots/02_selection.png) |

| 模型管理 | 历史记录 |
|---|---|
| ![模型管理](docs/screenshots/03_model.png) | ![历史记录](docs/screenshots/04_history.png) |

## 代码结构

```text
backend/
  main.py            FastAPI 入口
  dev_server.py      无 FastAPI 依赖的本地预览服务
  service.py         对话编排与卡片
  agent.py           原生 function calling 规划 + 工具参数修复 + 合规后处理
  chat/              工具契约（action_schema）与确定性策略（directives）
  llm.py             OpenAI 兼容客户端（chat_json / chat_text / chat_tools）
  store.py           SQLite 持久化（会话 active_version_id 为单一模型状态源）
  data/              数据源适配器和缓存 CLI
  engine/            因子、选股、组合、回测和服务端网格寻优
  models/            模型注册、代码校验、调参方案生成
frontend/             页面、对话和图表
tests/                Python 测试（含动作契约与“禁止关键词路由”回归）
scripts/              本地开发、E2E 冒烟、真实 LLM 语义矩阵
```

## 测试

```powershell
python -m unittest discover -s tests -v
node --check frontend/app.js
node --check frontend/charts.js
```

结构 E2E（tests/support 的 ScriptedLLM，无需 Key；生产包不含任何假模型）：

```powershell
python scripts\e2e_smoke.py
```

真实 LLM 语义矩阵（需要 `DEEPSEEK_API_KEY`，验证“更换数学模型/换内置模型/随机森林/继续优化/回滚”等意图是否通过 tool_calls 正确表达，11/11 通过才允许发版）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
.\.venv\Scripts\python.exe scripts\semantic_eval.py
```

## 说明

- 回测结果依赖数据范围、缓存内容和参数，不能代表未来收益。
- `backend.dev_server` 只用于不安装 FastAPI 时的本地预览；完整功能请使用 `backend.main:app`。
- 项目没有认证、权限管理、任务队列和生产级数据库配置，不建议直接暴露到公网。
