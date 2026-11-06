# 智投罗盘 · AI 投资助手（MVP）

一个面向 A 股投资场景的 AI 智能投资顾问：先用自然语言建立投资者画像，再生成适配策略、完成选股与解读；模型优化时由 AI 编写底层打分引擎并设定参数搜索范围，服务端执行网格寻优、实时展示调参进度，最后等待用户确认后才启用。

- 前端：原生 JS + 纯 SVG 图表，无构建链、无 CDN；**Professional Financial SaaS** 三栏布局（左侧导航 / 中间 AI 对话工作区 / 右侧投资上下文）
- 后端：FastAPI（单进程同时托管页面与 API）+ SQLite + pandas
- 数据：支持 AkShare / Tushare 真实行情；已接入 Tushare 缓存（当前 29 只核心 A 股，真实日线覆盖 2022-01-04 至 2026-11-05），离线缓存自动选择 parquet/CSV
- 模型：10 因子选股引擎 + 5 种内置模型类型 + AI 按需生成 sklearn 自定义模型 + AI 编写参数化打分引擎 + 服务端逐轴网格寻优（AST 白名单 + 子进程冒烟 + 全区间回测）

## 它怎么工作

1. **聊天建画像**：AI 用大白话一次问一个问题，自动补齐 12 个画像槽位；信息够了会先复述并请你确认。
2. **AI 主动执行**：你回一句“可以”，AI 自己去确认画像并开始选股（无需点按钮、无需切页面）。
3. **结果与解读**：选股卡展示 Top 10、白话指标、因子贡献、组合权重、净值曲线，并给出基于事实的解读；继续追问即可。
4. **AI 定引擎、服务端调参**：你说“帮我优化模型”，AI 会：
   - 先看当前表现 → 翻因子数据找短板 → 决定底层引擎（保留因子权重 / 切换内置模型 / 重写参数化打分代码）
   - AI 只给出参数搜索范围（因子权重的上下限与步长，或代码引擎的参数范围），不直接拍最终参数
   - 服务端在范围内执行逐轴网格寻优，逐个组合回测、按目标指标保留最优；网页实时显示进度、当前最优参数与得分走势
   - 对 AI 生成的代码做静态安全检查 + 子进程冒烟 + 全区间回测；候选方案必须经你明确同意才启用，且随时可回滚
5. **版本与会话隔离**：每个会话都有自己的模型版本链（基线 + 候选），聊天里说“回滚到上一个版本”即可；模型改动只对当前会话生效，**新建对话默认回到均衡多因子模型**。

## 快速开始

```powershell
cd a-share-ai-screener
$py = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

# 1) 数据缓存（示例数据，离线可用；真实数据用 --source akshare / tushare）
& $py -m backend.data.build_cache --source sample

# 2) 配置真实模型后启动
$env:DEEPSEEK_API_KEY = "sk-..."
& $py -m backend.dev_server --port 8000
# 或：
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

真实 A 股数据（推荐把 Token 放在项目根目录 `.env`，该文件已被 gitignore，不会入库）：

```powershell
# .env
# TUSHARE_TOKEN=你的 token
# SOURCE=tushare
# UNIVERSE=csi800

# 拉取真实行情并写入 data/a_share 缓存；重启服务后生效
& $py -m backend.data.build_cache --source tushare --limit 30 --start 2022-01-01
```

说明：低积分 Tushare token 可能没有 `index_weight`、`fina_indicator`、`daily_basic` 或 `index_daily` 权限/额度。适配器会自动降级：用核心 A 股标的池、真实日线行情和等权基准，缺失的财务因子按 0 处理；额度恢复后重新 build 即可自动补齐。服务运行期只读缓存，不联网。

## 界面

| 三栏工作区（导航 / AI 对话 / 投资上下文） | 选股结果与回测指标 |
|---|---|
| ![三栏工作区](docs/screenshots/01_ai_chat.png) | ![选股结果](docs/screenshots/02_selection.png) |

| 模型管理（版本与指标） | 历史记录 |
|---|---|
| ![模型管理](docs/screenshots/03_model.png) | ![历史记录](docs/screenshots/04_history.png) |

| 代码改动 diff | 投资组合 |
|---|---|
| ![代码改动](docs/screenshots/06_code.png) | ![投资组合](docs/screenshots/07_portfolio.png) |

| 收起右侧上下文 | 移动端导航抽屉 |
|---|---|
| ![收起上下文](docs/screenshots/05_collapsed.png) | ![移动端导航](docs/screenshots/08_mobile_nav.png) |

移动端投资上下文抽屉：

![移动端上下文](docs/screenshots/09_mobile_context.png)

截图来自真实浏览器联调（示例数据 + 真实 DeepSeek）。

**流式输出**：AI 回复通过 SSE `token` 事件逐字渲染（前端打字机队列）；模型优化时，`tuning` 卡片按同一 id 实时 upsert，进度条、当前最优指标、搜索范围与得分走势随每次回测更新；AI 修改代码/权重/模型类型时，改动框在 `work` 卡片内按行流式写入，`提出调参方案 → 服务端网格寻优 → 写入改动 → 对比 → 审批` 步骤同步推进：

![代码改动流式写入](docs/screenshots/10_streaming_code.png)

## 设计系统

- **定位**：Professional Financial SaaS。借鉴 Linear 的极简信息层级、Bloomberg Terminal 的金融数据密度、Stripe Dashboard 的卡片与表格规范；去装饰化，每个视觉元素都有信息功能。
- **布局**：顶部工具栏 56px；左侧导航 224px（<1024px 收为图标，移动端为抽屉）；中间对话区最小 560px；右侧投资上下文 320px（可收起，窄屏为抽屉）；页面边距 24px、栏间距 16px。
- **颜色**：主背景 `#F8FAFC`、卡片 `#FFFFFF`、品牌 Navy `#1E3A5F`、交互蓝 `#2563EB`、辅助蓝 `#DBEAFE`；边框 `#E2E8F0`、强边框 `#CBD5E1`；正文 `#0F172A` / `#475569`，辅助文字 `#94A3B8`。
- **状态与金融语义**：成功 `#15803D` / 警告 `#B45309` / 危险 `#DC2626` / 信息 `#2563EB`；A 股涨跌沿用中国大陆习惯，红色上涨、绿色下跌；所有收益、回撤、夏普等指标旁标注「历史回测数据，不代表未来实际表现」。
- **卡片 / 按钮 / 输入**：卡片白底、1px `#E2E8F0` 边框、10px 圆角、默认无阴影；按钮 6px 圆角、默认 36px 高；输入框 8px 圆角、白色背景、最小 56px，聚焦边框变蓝。
- **字体与数字**：`Inter, PingFang SC, Microsoft YaHei, Noto Sans CJK SC`；金融指标和表格数字使用 `font-variant-numeric: tabular-nums`，避免变化时跳动。
- **克制交互**：页面淡入 150ms、按钮/输入聚焦 120–150ms；不使用动态渐变背景、玻璃拟态、持续发光 AI 图标或无意义悬浮动画；支持 `prefers-reduced-motion`。
- **AI 对话组织**：AI 回复不只输出大段文字，还以指标卡、数据表格、代码 diff、方案对比、风险提示等结构化卡片呈现；审批与启用必须经过聊天确认。
- **换模型与重复提问**：内置 5 种基础风格；客户点名其他模型（SVM、随机森林、梯度提升、聚类等）时，AI 会用 sklearn 写参数化自定义模型代码，再由服务端网格寻优并回测，不再以“没有对应选项”拒绝。模型收益未达目标时会显式核对并给出「继续优化」选项；继续优化会更换基础引擎或重写打分逻辑，而不是只微调参数。确认话术在服务端去重，已确认画像字段不会被重复询问。
- **调参进度可见**：新增「参数网格寻优」卡片，展示 AI 给出的搜索范围、目标指标、已评估组数、当前最优参数与得分走势；寻优在服务端真实执行，进度由回测事件驱动，不伪造。
- **模型会话隔离**：模型改动只在当前会话生效；新建对话自动使用默认均衡多因子模型，避免上一个对话的调参结果意外影响新分析。

## 导航与对话管理

- **左侧导航**：AI 投资顾问、投资组合、策略分析、历史记录、模型管理、系统设置；投资组合 / 策略分析 / 模型管理使用真实会话状态生成只读详情弹窗，不改变后端接口。
- **对话区头部**：显示当前会话名与运行状态；提供历史、清空、新对话；输入区提供分析组合、查看模型、生成报告、上传资料等快捷操作，并支持停止生成、重新生成、清空输入。
- **投资上下文**：自动展示画像、AI 活动、当前模型与版本、最近选股结果；宽屏可收起，窄屏通过「上下文」按钮打开抽屉。
- **历史记录**：按最近更新时间列出全部会话（首条用户消息作为标题、消息数、画像摘要、是否已确认），可切换回任意会话并恢复聊天卡片、寻优进度卡与代码框；切换会话时右侧工作台同步该会话的模型版本链。
- 所有会话、画像、模型版本和聊天富卡片都保存在本地 SQLite 中，刷新页面或重启服务不会丢失；模型版本按会话隔离，新建对话自动使用默认模型。

## AI 表达与确认规范

- **金融机构服务口吻**：称呼“您”、结论先行、数据只引用真实回测结果、不做自我叙述（禁止“先问第一个问题/接下来我会”等 AI 套话），风险提示使用“历史业绩不代表未来表现，市场有风险，投资需谨慎”。提示词参考了公募基金销售与投资者适当性管理中的规范话术（见文末参考）。
- **能力透明**：首次对话与用户询问功能时，AI 会明确说明可做投资者画像、A 股筛选与回测、结果解读、模型优化（由 AI 编写底层引擎并设定搜索范围、服务端执行网格寻优）以及审批与版本回滚。
- **继续优化升级**：候选未达目标后点击「继续优化」不会继续微调参数，而是结构性升级：上一版是调权重 → 直接更换基础模型；上一版是切模型或代码 → 由 AI 重写数学打分引擎，再由服务端在新的参数范围内重新寻优并回测。
- **短回答语义理解**：客户只回“继续持有/4 年/10%”等省略句时，结合上一轮问题映射到对应画像槽位；不会把“继续持有”误判为持仓查询，也不会重复追问已回答字段。
- **客户追问解释规范**：客户问“什么是回撤/这是什么意思/为什么问这个”时，先说清概念与用途；纯概念提问可以只解释、不追问。禁止“请选择一项”“一是/二是/三是”“回复对应选项即可”等问卷式表达，也禁止“请按实际情况描述/请如实回答/必须回答”等审讯式表达。若首轮仍触发这些特征，会调用同一模型在约束提示词下重写，而不是套静态话术。
- **澄清与执行分离**：AI 需要客户补充信息时只提问、本轮不执行任何动作；客户回答后的下一轮才执行。服务端对“边问边改”做硬性拦截，并记录待执行意图（`pending_intent`）。
- **确认话术**：画像确认由 AI 自然询问；模型方案审批由 AI 结合回测结果说明，并提供审批卡操作。服务端不再追加固定「需要您确认：是否启用……」审批兜底文本。
- **代码自检与自动修复**：AI 生成的模型代码先做 AST 安全检查与子进程冒烟；失败时会把错误信息和原提案回传给模型，自动修复并重新校验（最多 2 次），work 卡展示「自动修复代码」步骤。调参方案（引擎 + 搜索范围）校验失败时同样会自动修复最多 2 次，并展示「自动修复调参方案」步骤。
- **调参分工与合规**：AI 不直接给最终参数，只决定引擎结构与搜索范围；具体参数由服务端在历史回测中逐组评估选出，审批卡明确提示「最优参数可能过拟合、历史回测不代表未来表现」，不会自动启用。
- **代码可读性**：代码框带 Python 语法高亮、逐行 diff，并在代码下方保留风险提示（AI 生成代码仅供回测演示，历史业绩不代表未来表现）。
- **代码改动可见**：AI 生成/修改策略代码时，聊天内展示 `models/strategy.py` 的行号 diff 代码框，并在审批卡中给出风险与回测对比；富卡片随消息持久化，刷新或重进会话可恢复。

## 代码结构

```
backend/
  main.py            FastAPI：页面 + 聊天 SSE + 状态接口
  dev_server.py      标准库 HTTP 服务（无 fastapi 时本地预览）
  service.py         对话编排：理解 → 回复 → 执行动作 → 生成卡片
  agent.py           Planner 提示词、动作白名单、合规校验
  llm.py             OpenAI 兼容客户端
  schemas.py         画像槽位、计划、白话文案
  store.py           SQLite：会话/消息/画像/模型版本/运行态
  events.py          SSE 格式化
  data/              DataBundle、示例数据、AkShare/Tushare 适配器、缓存 CLI
  engine/            A 股 10 因子、筛选/组合、T+1/涨跌停/停牌/成本回测、服务端网格寻优
  models/            内置模型类型、AI 代码安全校验与编译、调参方案流水线
frontend/
  index.html / app.js / style.css / charts.js
tests/               150+ 个 unittest（引擎/寻优/模型/Agent/存储/对话服务/HTTP）
scripts/             构建缓存、启动、重置、全量测试、E2E 冒烟
docs/                接口契约、运行手册、设计摘要、界面截图
```

## 测试与验收

```powershell
& $py -m unittest discover -s tests -v
$node --check frontend/app.js
$node --check frontend/charts.js
& $py scripts\e2e_smoke.py
```

- 全量测试：引擎、网格寻优（范围校验/预算/目标/回撤约束）、模型安全校验（含拒绝危险代码/死循环超时）、调参方案自动修复、Agent 计划、对话全流程、SQLite、HTTP/SSE
- 浏览器 E2E：真实 Edge 驱动“说话 → AI 追问 → 确认 → 选股 → 优化（观察寻优进度）→ 同意 → 当前会话模型更新 → 新对话回到默认模型 → 回滚”全流程；另验证三栏尺寸（56/224/320）、卡片 Token、对话→画像→选股→刷新恢复、清空对话与移动端抽屉。
- 风险提示：本项目为比赛演示原型，历史回测不代表未来收益，不构成投资建议。

## 参考话术来源（提示词依据）

- 公募基金销售适当性与风险揭示流程话术：[德邦基金公募产品开立交易操作指南](https://www.dbfund.com.cn/upload/pdf/1698046211058-debangjijinguanliyouxiangongsigongmuchanpinkaihujiaoyicaozuozhinan.pdf)
- 理财产品销售告知与期限/风险披露话术：[杭银理财产品话术](http://jsbchina.cn/files/licai/FYG25031A.pdf)
- 高净值客户 KYC 与沟通技巧：[《高净值客户深度 KYC 及沟通技巧》](https://m.jiangshi99.com/course/content/185767.html)
- 证券投资顾问业务与适当性管理要求：[证券投资顾问业务暂行规定](https://www.10jqka.com.cn)
- 财富管理路演支持话术参考：[FinClaw bank wealth-management skill](https://github.com/aifinlab/finclaw/blob/main/skills/bank-t191-wealth-management-roadshow-support-assistant/SKILL.md)

## 设计参考

- [Linear](https://linear.app/)：极简布局、信息层级与克制的交互。
- [Bloomberg Terminal](https://www.bloomberg.com/professional/products/bloomberg-terminal/)：金融数据密度、指标层次与表格表达。
- [Stripe Dashboard](https://stripe.com/docs/dashboard)：卡片、字体、表格与状态设计规范。
