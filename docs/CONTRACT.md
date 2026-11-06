# A 股 AI 选股助手 · v2 契约（对话优先 + AI 主动执行 + AI 参与建模）

> 本文件是 v2 唯一契约。旧版接口/页面/兜底设计直接替换，不保留兼容层；历史版本见 git。

## 0. 环境

- 项目根：`C:\Users\77578\Documents\deepseek-harness\default-workspace\a-share-ai-screener`
- Python：`C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`（已装 fastapi/uvicorn；无 akshare/tushare）
- Node：`C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe`
- git 由 Lead 统一提交；子任务不要执行 git，不要改自己写作用域之外的文件。

## 1. 产品原则（v2）

1. **一个对话框完成全部操作**：画像采集、确认、选股、解读、追问、模型优化、版本启用/回滚，全部在同一个聊天流里完成。没有模块页签、没有独立的“运行/确认”按钮页面。
2. **AI 主动执行（用户确认后）**：AI 判断用户意图并调用后端动作。画像确认、发起选股等低风险动作，在用户对话中确认后由 AI 直接执行；应用模型变更/启用版本等高影响动作，在聊天中给出审批卡，用户点击聊天内快捷回复或直接说“同意”后执行。
3. **普通用户语言**：AI 不用专业术语提问；必须把“回撤、夏普、因子”等翻译成生活化解释；一次只问一个问题；先共情再提问。
4. **AI 真正参与建模**：AI 可以切换内置模型类型、调整参数、生成/修改选股模型源码（受静态检查与冒烟测试约束），并用回测结果向用户解释为什么更好。
5. **AI 工作过程可见**：优化过程中，前端实时展示 AI 的步骤（分析→提出方案→改代码→检查→回测→对比→等待确认），步骤由后端真实动作产生，不伪造。

## 2. 后端 API（精简）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | `{ok, source, as_of, universe, llm}` |
| GET | `/api/sessions` | 历史对话归档 `{sessions:[{id,title,created_at,updated_at,message_count,profile_summary,profile_confirmed}]}` |
| POST | `/api/sessions` | `{session_id}` |
| POST | `/api/sessions/{sid}/chat` | `{content, client_action?}` → **SSE 聊天流**（见第 3 节） |
| GET | `/api/sessions/{sid}/messages` | 文本 + 富卡片历史 `{messages:[{role,content,blocks,created_at}]}` |
| GET | `/api/sessions/{sid}/state` | 页面刷新恢复用：`{profile, active_model, versions, pending_proposal, last_selection}` |
| GET | `/api/versions` | 版本列表（兼容保留给状态面板） |

不提供独立的 profile/selection/tune 动作接口——这些动作只能由聊天流里的 AI 触发。

## 3. 聊天 SSE 事件

`event: token` → `{text}`：AI 自然语言增量。
`event: ui` → `{block: {...}}`：插入或更新一个富卡片（按 `id` upsert）。
`event: done` → `{message_id}`。
`event: error` → `{message, retryable}`。

### 3.1 卡片（block）

**profile**（画像卡）
```json
{"id":"profile-1","type":"profile",
 "summary":"用一句话描述用户（AI 生成，通俗）",
 "items":[{"label":"风险感受","value":"亏 15% 会有点慌"},...],
 "completeness":0.75,"missing":["投资期限"],
 "confirmed":false}
```

**selection**（选股结果卡）
```json
{"id":"selection-1","type":"selection",
 "as_of":"2024-11-18","model":{"name":"均衡多因子","description":"..."},
 "top":[{"code":"600519.SH","industry":"食品饮料","score":2.4,
         "reasons":["盈利质量高","价格相对便宜"]}],
 "metrics":{"cagr":0.19,"sharpe":1.0,"mdd":-0.17,"ir":3.5,"win_rate":0.6},
 "metric_notes":{"cagr":"过去几年平均每年涨多少","sharpe":"同样的波动下赚得多不多"},
 "portfolio":[{"code":"...","weight_pct":12.0}],
 "nav":[{"date":"...","value":1.0}],"benchmark_nav":[...],
 "explanation":"给普通用户看的解读"}
```

**work**（AI 工作过程卡，优化/分析时出现）
```json
{"id":"work-1","type":"work","title":"AI 正在优化模型",
 "status":"running|done|error",
 "steps":[{"title":"看当前表现","detail":"近两年年化 12%，最大回撤 18%","status":"done","kind":"analysis"},
          {"title":"提出方案","detail":"把权重更多放给低波动和盈利质量","status":"running","kind":"plan"}],
 "comparison":{"current":{"sharpe":1.0},"candidate":{"sharpe":1.2}},
 "proposal":{"summary":"...","approach":"model_type|weights|code",
             "model_name":"低波质量","diff":"（code 方案时的 diff 摘要）",
             "risks":["..."]}}
```

**approval**（审批卡；用户可在聊天里点快捷回复，也可直接说话）
```json
{"id":"approval-1","type":"approval","title":"是否启用这个改进？",
 "summary":"AI 建议把模型切成「低波质量」并微调权重，回测 Sharpe 从 1.02 升到 1.21。",
 "risk":"这是演示系统，不构成投资建议。",
 "actions":[{"label":"同意，启用吧","message":"同意启用这个改进"},
            {"label":"先看看细节","message":"先不启用，展开说说风险"}]}
```

**version**（版本卡）
```json
{"id":"version-1","type":"version","active":{"id":"v...","name":"低波质量",
 "metrics":{"sharpe":1.2}},
 "versions":[{"id":"v...","name":"低波质量","status":"candidate","created_at":"..."}]}
```

## 4. AI 动作与权限（后端执行，前端不直接调）

Planner 每轮输出：
```json
{"reply":"给用户看的自然语言（≤120字，通俗）",
 "actions":[{"name":"...","args":{}}],
 "profile_delta":{"horizon":"3-5 年"}}
```

| action | 何时允许 | 行为 |
|---|---|---|
| `confirm_profile` | 用户本轮明确确认（可以/确认/没问题/就按这个来）且画像必填齐全 | 写入 confirmed，返回 profile 卡，AI 回复确认 |
| `run_selection` | 画像已确认（可同轮先 confirm） | 运行 10 因子选股+回测，返回 selection 卡 |
| `explain_results` | 有最近一次选股结果 | 基于事实回答追问 |
| `analyze_model` | 画像已确认即可 | AI 分析当前模型表现 → 生成候选方案（切模型/调参/改代码）→ 静态检查 → 冒烟 → 回测 → work 卡 + approval 卡，**不自动启用** |
| `apply_pending_change` | 存在待审批方案且用户本轮明确同意 | 保存新模型版本并设为 active，返回 version 卡 |
| `rollback_version` | 用户明确要求回滚 | 切回上一版本，返回 version 卡 |

低风险动作不允许在缺少用户确认时触发；高影响动作必须有本轮明确同意。所有动作由后端校验，不信任前端。

## 5. 画像槽位（12 项，全部用生活化标签）

`goal`(投资目标)、`horizon`(投资期限)、`capital`(大致投入)、`liquidity_need`(多久可能要用钱)、`loss_reaction`(亏钱时的真实反应)、`drawdown_tolerance`(最多能接受亏多少)、`income_stability`(收入稳定程度)、`experience`(投资经验)、`preferred_industries`(偏好行业)、`excluded_industries`(不碰行业)、`trade_frequency`(希望多久调整一次)、`concerns`(最担心什么)。
必填：`goal`、`horizon`、`drawdown_tolerance`、`loss_reaction`。风险等级由 AI 根据这些推导，不直接问“你是什么风险等级”。

AI 追问原则：一次一个问题；先回应/共情，再问；提供生活化选项；不重复问已知信息；最多 3 轮内形成画像总结并请用户确认。

## 6. AI 建模流水线（`backend/models/`）

### 6.1 内置模型类型（registry）

内置 5 种：`multi_factor`（均衡多因子）、`low_vol_quality`（低波质量）、`momentum_growth`（趋势成长）、`value_dividend`（价值红利）、`contrarian`（反转）。客户点名清单外模型（SVM、随机森林、梯度提升、聚类等）时，走 `approach=code`，由 AI 用 sklearn 生成自定义模型代码并回测，不再维护固定模型白名单。每个模型：`id/name/description/weights(10因子)`。

### 6.2 LLM 提案 JSON（pipeline 输入）

```json
{"approach":"model_type|weights|code",
 "summary":"一句普通用户能懂的话，说明为什么这样改",
 "model_type":"low_vol_quality",
 "weights":{"ep":0.1,...10项...},
 "code":"python 源码（approach=code 时）",
 "why":"数据依据（引用真实指标）",
 "expected_effect":"预期改善方向",
 "risks":["..."]}
```

### 6.3 自定义模型代码契约

AI 生成的代码必须定义：
```python
def build_transform():
    def transform(z, weights):
        # z: pandas.DataFrame，列含 z_ep/z_bp/...（10 个方向对齐后的因子 z 值）
        # weights: dict[str,float]
        # 返回 pandas.Series（index 与 z 对齐）
        return ...
    return transform
```
允许 import：`math`、`numpy as np`、`pandas as pd`、`sklearn`（仅 preprocessing/svm/ensemble/linear_model/cluster/decomposition/pipeline/impute/neighbors/tree/metrics/utils 子模块）与 `scipy`；禁止：文件/网络/系统调用/动态执行/dunder 属性；源码 ≤6000 字符；必须通过 AST 白名单检查 + 子进程冒烟（10s 超时）。sklearn 模型基于当前截面做转导/无监督拟合或用加权因子合成分构造伪标签，不假设未来收益标签。

### 6.4 预期核对与继续优化

候选回测完成后，服务端根据用户本轮诉求（收益优先 / 回撤优先 / 默认夏普）与画像回撤上限核对是否达到目标，并在 work 卡新增「核对您的目标」步骤。未达预期时：审批卡标题改为「候选未达预期，是否仍要启用？」并增加「继续优化」动作；聊天回复明确差距，不会自动启用模型。用户选择「继续优化」会重新进入 analyze_model，生成下一轮候选。

### 6.4 pipeline 公开函数

```python
def list_models() -> list[dict]                       # 内置模型清单
def normalize_proposal(raw: dict) -> Proposal         # 校验字段、权重、模型类型
def validate_code(code: str) -> dict                  # AST+冒烟，失败抛 ModelCodeError
def compile_code(code: str) -> callable               # 返回 transform(z,weights)
def build_candidate_proposal(llm, state) -> Proposal  # 调 LLM 生成提案（prompt 在 pipeline 内）
```
`selection.screen/run_backtest` 增加可选参数 `score_transform=None`（Lead 实现，供自定义代码接入）。

## 7. 前端（Professional Financial SaaS 三栏）

- 文件：`frontend/index.html`、`app.js`、`style.css`、`charts.js`（纯原生 JS + SVG，无构建链/无 CDN）。
- 布局：**顶部工具栏 56px + 左侧导航 224px + 中间 AI 对话工作区（最小 560px）+ 右侧投资上下文 320px**；页面边距 24px、栏间距 16px。小于 1024px 时导航收为图标，移动端为抽屉；小于 1200px 时右侧上下文改为可开合抽屉。
- 导航：左侧为 AI 投资顾问 / 投资组合 / 策略分析 / 历史记录 / 模型管理 / 系统设置；投资组合、策略分析、模型管理、设置使用当前会话状态渲染只读弹窗，不改后端接口。对话区头部提供历史、清空、新对话；输入区支持快捷操作、停止生成、重新生成、清空输入与文本附件。
- 交互：对话是唯一的写操作入口；卡片里的「快捷回复」按钮只是把一句话发进聊天，不直接调接口；审批、启用、回滚都必须经过聊天消息。
- SSE：`POST /api/sessions/{sid}/chat` 用 `fetch` + `ReadableStream` 解析；`token` 事件逐字追加（前端打字机渲染）；`ui` 事件按 id upsert 卡片（work 卡步骤实时更新）。**代码/权重/模型类型改动流式契约**：服务端先发一个 `work` 块（`status=running`、最后一步 `修改模型代码/调整因子权重/切换模型类型` 为 running、`code_change.diff` 为空），随后按行重复发同一 id 的 `work` 块，`code_change.diff` 逐行增长，最后发 `status=done` 的完整块；前端按 id 覆盖渲染，因此代码框逐行出现。`error` 显示错误条+重试；停止生成通过 `AbortController` 取消请求。
- 首次进入自动创建会话；刷新后调用 `/state` 恢复右侧上下文，从 `/messages` 恢复聊天文本与富卡片（`blocks`）；历史对话通过 `GET /api/sessions` 归档列表切换。
- 视觉：**Professional Financial SaaS / Light Mode**，参考 Linear + Bloomberg Terminal + Stripe Dashboard。Token：背景 `#F8FAFC`、卡片 `#FFFFFF`、品牌 Navy `#1E3A5F`、交互蓝 `#2563EB`、边框 `#E2E8F0`；卡片 1px 边框 / 10px 圆角 / 默认无阴影，按钮 6px 圆角，输入框 8px 圆角；正文 14px、辅助 12px、卡片标题 16px。**不使用动态渐变背景、玻璃拟态与大阴影**。
- 模型切换：内置 5 种基础风格；客户点名其他模型时，服务端强制 `approach=code`，由 AI 使用 numpy/pandas/sklearn/scipy 写自定义模型代码并通过静态检查+冒烟后回测，不再由 LLM 以“没有对应选项”拒绝。模型代码只允许基于当前截面做转导/无监督拟合或伪标签蒸馏。
- 确认话术：同一轮只允许出现一段确认请求；服务端修剪计划回复中的重复确认段落，并在动作 note 与回复重复时剥离 note 中的确认尾段；已确认画像字段不得重复询问。
- 状态与金融语义：成功绿 / 警告琥珀 / 危险红 / 信息蓝；A 股涨跌沿用大陆习惯红色上涨、绿色下跌；收益、回撤、夏普等指标统一标注「历史回测数据，不代表未来实际表现」。
- 禁止：mock 数据、独立页签、页面内直接调 selection/tune/profile 接口、伪造进度。

## 8. 文件与分工

- Lead：`backend/agent.py`、`backend/service.py`、`backend/main.py`、`backend/dev_server.py`、`backend/store.py`、`backend/schemas.py`、`backend/engine/selection.py`（score_transform 钩子）、`tests/` 集成测试、文档。
- **chat-ui-dev**：只写 `frontend/` 下 `index.html`、`app.js`、`style.css`、`charts.js`。
- **model-pipeline-dev**：只写 `backend/models/*.py`、`tests/test_models.py`。
- 完成标准：`node --check` 通过；后端 `unittest discover` 通过；真实浏览器可完成“说目标→AI 追问→AI 总结并确认→AI 选股→AI 解读→说‘帮我优化模型’→看 AI 步骤→聊天内同意→版本更新”全流程。
