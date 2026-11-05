# 智投罗盘 · AI 选股助手（MVP）

一个「只用一个聊天框」完成全部操作的 A 股 AI 选股应用：AI 先像朋友一样了解你，再自动完成选股、解读，并能自己分析模型、生成/修改选股代码、回测对比，最后等你点头后才启用。

- 前端：原生 JS 单页（无构建链、无 CDN、纯 SVG 图表），一个聊天流 + 右侧 AI 工作台
- 后端：FastAPI（单进程同时托管页面与 API）+ SQLite + pandas
- 数据：AkShare 为主、Tushare 为辅；离线缓存自动选择 parquet/CSV；无网络时可用内置示例数据
- 模型：10 因子选股引擎 + 5 种内置模型类型 + AI 生成自定义打分代码（AST 白名单 + 子进程冒烟 + 回测验证）

## 它怎么工作

1. **聊天建画像**：AI 用大白话一次问一个问题，自动补齐 12 个画像槽位；信息够了会先复述并请你确认。
2. **AI 主动执行**：你回一句“可以”，AI 自己去确认画像并开始选股（无需点按钮、无需切页面）。
3. **结果与解读**：选股卡展示 Top 10、白话指标、因子贡献、组合权重、净值曲线，并给出基于事实的解读；继续追问即可。
4. **AI 参与建模**：你说“帮我优化模型”，AI 会：
   - 看当前表现 → 翻因子数据找短板 → 提出改进方案（切换内置模型 / 调权重 / 直接生成新的策略代码）
   - 对生成的代码做静态安全检查 + 子进程冒烟 + 全区间回测
   - 把每一步实时展示在聊天里，给出对比指标和风险，**等你明确同意后才启用**，且随时可回滚
5. **版本可回滚**：每次模型变更都落版本（基线 + 候选），聊天里说“回滚到上一个版本”即可。

## 快速开始

```powershell
cd a-share-ai-screener
$py = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

# 1) 数据缓存（示例数据，离线可用；真实数据用 --source akshare / tushare）
& $py -m backend.data.build_cache --source sample

# 2) 无 Key 本地体验（FakeLLM，仅用于演示与测试）
$env:LLM_PROVIDER = "fake"
& $py -m backend.dev_server --port 8000
# 或正式入口：
# & $py -m uvicorn backend.main:app --port 8000

# 3) 浏览器打开 http://127.0.0.1:8000
```

正式使用需配置真实模型（OpenAI 兼容接口）：

```powershell
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "sk-..."
$env:LLM_MODEL = "deepseek-chat"
& $py -m uvicorn backend.main:app --port 8000
```

真实 A 股数据：

```powershell
& $py -m backend.data.build_cache --source akshare --limit 300
# 或 Tushare（需要 TUSHARE_TOKEN）
& $py -m backend.data.build_cache --source tushare --limit 300
```

## 界面

| 对话建画像 | 选股结果 |
|---|---|
| ![对话建画像](docs/screenshots/01_ai_chat.png) | ![选股结果](docs/screenshots/02_selection.png) |

| AI 优化过程（方案/代码/对比/审批） | 模型版本 |
|---|---|
| ![AI 优化过程](docs/screenshots/03_interpretation.png) | ![模型版本](docs/screenshots/04_tuning.png) |

截图来自真实浏览器联调（示例数据 + FakeLLM）；真实模型下文字与解读由所选模型生成。

## 代码结构

```
backend/
  main.py            FastAPI：页面 + 聊天 SSE + 状态接口
  dev_server.py      标准库 HTTP 服务（无 fastapi 时本地预览）
  service.py         对话编排：理解 → 回复 → 执行动作 → 生成卡片
  agent.py           Planner 提示词、动作白名单、合规校验
  llm.py             OpenAI 兼容客户端 + FakeLLM（测试/演示）
  schemas.py         画像槽位、计划、白话文案
  store.py           SQLite：会话/消息/画像/模型版本/运行态
  events.py          SSE 格式化
  data/              DataBundle、示例数据、AkShare/Tushare 适配器、缓存 CLI
  engine/            A 股 10 因子、筛选/组合、T+1/涨跌停/停牌/成本回测
  models/            内置模型类型、AI 代码安全校验与编译、提案流水线
frontend/
  index.html / app.js / style.css / charts.js
tests/               87+ 个 unittest（引擎/模型/Agent/存储/对话服务/HTTP）
scripts/             构建缓存、启动、重置、全量测试、E2E 冒烟
docs/                接口契约、运行手册、设计摘要、界面截图
```

## 测试与验收

```powershell
& $py -m unittest discover -s tests -v
$node --check frontend/app.js; $node --check frontend/charts.js
& $py scripts\e2e_smoke.py
```

- 全量测试：引擎、模型安全校验（含拒绝危险代码/死循环超时）、Agent 计划、对话全流程、SQLite、HTTP/SSE
- 浏览器 E2E：真实 Edge 驱动“说话 → AI 追问 → 确认 → 选股 → 优化 → 同意 → 版本更新”全流程
- 风险提示：本项目为比赛演示原型，历史回测不代表未来收益，不构成投资建议。
