# A 股 AI 选股助手 · v3.0 契约（原生 function calling + AI 定引擎/范围 + 服务端网格寻优）

> 本文件是当前唯一契约。旧版接口/页面/关键词路由/legacy proposal 协议直接删除，不保留兼容层；历史版本见 git。

## 0. 环境

- 项目根：`C:\Users\77578\Documents\deepseek-harness\default-workspace\a-share-ai-screener`
- Python：项目内 `.venv`（Python 3.12，含 fastapi/uvicorn/pandas/scikit-learn/tushare；也可用 DSH runtime python）。真实 Key 通过环境变量 `DEEPSEEK_API_KEY` 或 `.env` 注入，不入库。
- Node：`C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe`
- 测试：`.venv\Scripts\python.exe -m unittest discover -s tests -v`
- git 由 Lead 统一提交；子任务不要执行 git，不要改自己写作用域之外的文件。

## 1. 产品原则（v3）

1. **一个对话框完成全部操作**：画像采集、确认、选股、解读、追问、模型优化、版本启用/回滚，全部在同一个聊天流里完成。没有模块页签、没有独立的“运行/确认”按钮页面。
2. **AI 主动执行（用户确认后）**：AI 通过**原生 function calling** 调用后端工具。画像确认、发起选股等低风险动作，在用户对话中确认后由 AI 直接执行；应用模型变更/启用版本等高影响动作，在聊天中给出审批卡，用户点击聊天内快捷回复或直接说“同意”后执行。
3. **普通用户语言**：AI 不用专业术语提问；必须把“回撤、夏普、因子”等翻译成生活化解释；一次只问一个问题；先共情再提问。
4. **AI 定引擎，服务端调参**：AI 负责选择/编写底层打分引擎并给出参数搜索范围；最终权重或代码参数由服务端在范围内网格寻优、逐组回测选出。AI 不直接拍最终参数。
5. **AI 工作过程可见**：优化过程中，前端实时展示 AI 的步骤（分析 → 提出调参方案 → 服务端网格寻优 → 写入改动 → 回测 → 对比 → 等待确认），步骤由后端真实动作产生，不伪造。寻优进度卡按真实回测事件实时更新。
6. **模型会话隔离**：模型改动只在被启用的会话内生效；新建对话默认使用「均衡多因子」模型，不继承上一个对话的模型。每个会话保留自己的版本链，可独立回滚。
7. **意图唯一权威**：意图只来自 LLM 的 tool_calls（typed args）；服务端只校验 schema、执行确定性策略、审计，**禁止**从自然语言关键词推断意图（`_wants_*` / `_requested_*` / 基于回复文本的路由均删除）。

## 1.1 数据源（真实数据）

- Tushare token 通过 `.env` 的 `TUSHARE_TOKEN` 注入；`.env` 已 gitignore，不提交。
- `python -m backend.data.build_cache --source tushare --limit 30 --start 2022-01-01` 生成 `data/a_share/` 缓存；服务运行期只读缓存。
- 低积分 token 的 `index_weight` / `fina_indicator` / `daily_basic` / `index_daily` 可能无权限或限频；适配器首次失败后不再重复请求，并降级为：核心 A 股标的池、真实日线行情、等权基准，缺失财务因子在打分时按 0 处理。额度恢复后重新 build 即自动补齐。
- 当前缓存：29 只核心 A 股，2022-01-04 ~ 2026-11-05，`meta.source=tushare`。

## 2. 后端 API（精简）

所有 `/api/**` 默认需要登录 Cookie（`/api/auth/*` 除外）；未登录返回 401，前端跳转 `/login`。会话按用户隔离：普通用户只能看到自己的会话，管理员也只能看到自己的会话；管理员可额外管理账号。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/auth/status` | 登录/初始化状态 `{configured, setup_required, setup_allowed, authenticated, username, role}` |
| POST | `/api/auth/setup` | 首次初始化管理员（仅本机、仅未配置时）；成功后写入 HttpOnly Cookie |
| POST | `/api/auth/login` | 登录；连续失败 5 次锁定 5 分钟 |
| POST | `/api/auth/logout` | 注销当前会话 |
| GET | `/api/auth/session` | 当前登录用户与角色；未登录 401 |
| GET | `/api/admin/users` | 管理员：用户列表 |
| POST | `/api/admin/users` | 管理员：创建用户 `{username,password,role:admin|user}` |
| PATCH | `/api/admin/users/{id}` | 管理员：启用/停用、重置密码、调整角色 |
| DELETE | `/api/admin/users/{id}` | 管理员：删除无数据用户；有会话数据时返回 409 |
| GET | `/api/admin/data/sync` | 管理员：行情同步状态（上次成功、as_of、下次运行） |
| POST | `/api/admin/data/sync` | 管理员：手动触发一次行情同步 |
| GET | `/api/health` | `{ok, source, as_of, universe, llm}`（需登录） |
| GET | `/api/sessions` | 历史对话归档 `{sessions:[{id,title,created_at,updated_at,message_count,profile_summary,profile_confirmed}]}` |
| POST | `/api/sessions` | `{session_id}` |
| POST | `/api/sessions/{sid}/chat` | `{content, client_action?}` → **SSE 聊天流**（见第 3 节） |
| GET | `/api/sessions/{sid}/messages` | 文本 + 富卡片历史 `{messages:[{role,content,blocks,created_at}]}` |
| GET | `/api/sessions/{sid}/state` | 页面刷新恢复用：`{profile, active_model, versions, pending_proposal, pending_candidates, last_selection, model_scope:"session", default_model}`；`active_model`/`versions` 均为该会话级别；`pending_proposal.candidate_count` 是待确认候选方案数量；`pending_candidates` 是候选方案列表（`index/name/summary/status/available/metrics`） |
| POST | `/api/sessions/{sid}/candidates/{index}/apply` | 工作台「启用」按钮：直接启用第 `index` 个待确认候选方案（不经过 LLM），返回 `{ok, note, blocks, state}` |
| POST | `/api/sessions/{sid}/versions/{vid}/activate` | 工作台「启用」按钮：把当前会话切到该会话的任意历史模型版本（含 baseline），返回 `{ok, note, blocks, state}` |
| GET | `/api/versions?session={sid}` | 版本列表；带 `session` 时只返回该会话版本链（`model_scope=session`），不带时返回全局归档 |

不提供独立的 profile/selection/tune 动作接口——这些动作只能由聊天流里的 AI 触发。模型候选方案与历史版本额外提供工作台直连「启用」接口（仅做状态校验与执行，不做语义判断），审计记为 `ai_apply` / `ui_activate_version`。

**状态单一来源（SCHEMA_VERSION=7）**：会话当前模型 = `session.active_version_id` 指向的 `version` 行（为空则默认「均衡多因子」）；`session_state` 只保存 pending/待执行意图/最近结果，不再保存 active_model/metrics/version_id。旧的全局 `store.active_version/activate/parent_version` 接口已删除，版本链按 `session_id` 隔离。`auth_account` / `auth_session` 保存多用户账号与会话令牌哈希；`session.owner_id` 用于用户数据隔离，第一次初始化管理员时接管历史无主会话。

## 3. 聊天 SSE 事件

`event: token` → `{text}`：**LLM 流式生成的真实文本增量**（content 直接以 delta 形式转发，前端收到即渲染）。
`event: ui` → `{block: {...}}`：插入或更新一个富卡片（按 `id` upsert）。
`event: done` → `{message_id}`。
`event: error` → `{message, retryable}`。

Planner 使用 OpenAI 兼容的 **stream + tools**：文本 delta 实时发给浏览器；tool_calls 在本轮流结束后由服务端校验执行。画像更新通过 `update_profile` 工具完成，不再出现在回复 JSON 里。

### 3.1 卡片（block）

**profile**（画像卡）
```json
{"id":"profile-1","type":"profile",
 "summary":"用一句话描述用户（AI 生成，通俗）",
 "items":[{"label":"风险感受","value":"亏 15% 会有点慌"},...],
 "completeness":0.75,"missing":["投资期限"],
 "confirmed":false}
```
画像卡只在四项必填信息齐全、进入最终确认时首次发送；逐题作答阶段不发送。客户确认后用同一 id 更新 `confirmed=true`，不新弹一张卡。

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

**work**（AI 工作过程卡，优化/分析时出现；代码类方案会先于此卡逐行写入 diff）
```json
{"id":"work-1","type":"work","title":"AI 正在优化模型",
 "status":"running|done|error",
 "steps":[{"title":"看当前表现","detail":"近两年年化 12%，最大回撤 18%","status":"done","kind":"analysis"},
          {"title":"运行因子网格寻优","detail":"已评估 18/21 组，当前最优夏普 0.83","status":"running","kind":"tune"}],
 "code_change":{"file":"models/strategy.py","language":"python","kind":"code|weights|model_type",
                "diff":"逐行 diff","added":12,"removed":3},
 "proposal":{"summary":"...","approach":"服务端网格寻优（因子权重）",
             "model_name":"网格寻优权重","why":"...","expected_effect":"...","risks":["..."]}}
```
`work` 卡不承载效果对比；代码类改动的 `code_change.diff` 必须在 `tuning` 卡出现之前逐行推送完成。

**tuning**（服务端参数网格寻优进度卡；同一 id 实时 upsert）
```json
{"id":"tune-1","type":"tuning","title":"参数网格寻优","status":"running|done|error",
 "method":"逐因子网格搜索|参数网格搜索",
 "objective":"sharpe|cagr|mdd","objective_label":"夏普比率",
 "space":[{"key":"roe","label":"赚钱能力","min":0.04,"max":0.24,"step":0.04}],
 "progress":{"trial":12,"total":21,"percent":57,"elapsed":4.8},
 "current":{"trial":12,"label":"赚钱能力 = 0.2","score":0.31,
            "metrics":{"cagr":0.11,"sharpe":0.31,"mdd":-0.16},"params":{"roe":0.2}},
 "best":{"trial":10,"label":"第 10 组","score":0.33,
         "metrics":{"cagr":0.12,"sharpe":0.33,"mdd":-0.15},
         "params":{"ep":0.1,"roe":0.2},"weights":{"ep":0.1,"roe":0.2},"constraint_ok":true},
 "history":[{"trial":1,"score":0.12},{"trial":2,"score":0.18}],
 "note":"网格寻优基于历史回测逐组评估，最优参数可能存在过拟合；候选仍需确认后才会启用。"}
```
说明：`space` 是 AI 给出的搜索范围（权重类为 10 因子，代码类为引擎参数）；进度、最优指标与得分走势由服务端每次回测后实时推送，前端不得伪造。`best.score`/`history.score` 是展示用真实目标指标（如夏普），选取排序时若回撤超限会在内部使用 `rank_score` 惩罚，`constraint_ok` 表示最优组是否满足回撤约束。

**comparison**（效果对比卡；独立卡，必须晚于 `tuning` 的 `done` 出现）
```json
{"id":"compare-1","type":"comparison","title":"效果对比","status":"done",
 "comparison":{"current":{"cagr":0.09,"sharpe":0.4,"mdd":-0.19},
               "candidate":{"cagr":0.12,"sharpe":0.8,"mdd":-0.15}},
 "note":"历史回测数据，不代表未来实际表现；候选方案需您确认后才会启用。"}
```
卡片顺序固定为：`work`（含代码改动）→ `tuning`（寻优结果）→ `comparison`（效果对比）→ `approval`（审批）。

**approval**（审批卡；用户可在聊天里点快捷回复，也可直接说话）
```json
{"id":"approval-1","type":"approval","title":"是否启用这个网格寻优结果？",
 "summary":"服务端网格寻优（逐因子网格搜索）共评估 21 组参数，按夏普比率选出最优一组；回测夏普 0.40 → 0.83。",
 "risk":"最优参数由历史回测选出，可能存在过拟合；历史回测不代表未来实际表现。",
 "actions":[{"label":"同意","message":"同意启用这个改进"},
            {"label":"暂存","message":"先别启用，我再想想"}]}
```

**version**（版本卡；版本与当前模型均为会话级）
```json
{"id":"version-1","type":"version","active":{"version_id":"v...","name":"网格寻优权重",
 "kind":"builtin","metrics":{"sharpe":1.2}},
 "versions":[{"id":"v...","name":"网格寻优权重","status":"active","created_at":"..."}]}
```
用户同意后仅当前会话启用新模型；新建对话时 `active_model` 为默认「均衡多因子」，`versions` 为空。

## 4. AI 动作与权限（原生 function calling，服务端只执行）

Planner 每轮输出：
- `content`：直接给客户看的纯文本回复（模型流式生成，不加 JSON 包装）；
- `tool_calls`：动作，由 Action Gateway 按 schema 校验后执行。

```json
// tool_calls（模型主动调用）
[{"name":"write_model_program","args":{"files":[{"path":"main.py","content":"..."}],"entry":"main.py","summary":"按客户要求编写非线性打分程序","param_space":{"tilt":{"min":0,"max":1,"step":0.25}},"objective":"mdd","packages":[]}}]
```

工具 schema 的唯一来源：`backend/chat/action_schema.py`（Pydantic 模型同时用于运行时校验、发给模型的 JSON Schema、测试）。服务端不再解析自然语言判断意图。

| tool | args | 何时允许 | 行为 |
|---|---|---|---|
| `update_profile` | `{slots:{key:value}}` | 客户本轮提供画像信息 | 合并画像槽位，返回/更新 profile 卡 |
| `confirm_profile` | `{}` | 画像必填齐全且用户本轮明确确认 | 写入 confirmed，返回 profile 卡 |
| `run_selection` | `{}` | 画像已确认（可同轮先 confirm） | 运行 10 因子选股+回测，返回 selection 卡 |
| `explain_results` | `{question}` | 有最近一次选股结果 | 基于事实回答追问 |
| `analyze_model` | `{engine, model_type, objective, escalate, reason}` | 画像已确认即可 | 只做内置模型/权重层面调整；`engine=model_type` 切换内置模型并使用预设权重，其他 engine 走服务端调参方案 → 网格寻优 → work/tuning/approval 卡，**不自动启用**。`engine="code"` 已废弃，服务端会要求改用 `write_model_program` |
| `write_model_program` | `{files[], entry, summary, param_space, objective, packages[], escalate, reason, expected_effect, risks[]}` | 客户要求构建/更换/编辑模型程序，或结构性升级 | **AI 直接写完整多文件程序** → LLM 二次审核（不通过自动修复 1 次）→ 安装审核通过的第三方库 → 受限沙箱静态校验 → 服务端网格寻优/回测 → 逐行流式 diff → work/tuning/comparison/approval 卡，**不自动启用** |
| `apply_pending_change` | `{candidate?}` | 存在待审批候选方案且用户本轮明确同意/指定 | `candidate` 为候选方案编号（按出现顺序 1..N；缺省=最新待审批方案），当前会话启用该候选模型版本，返回 version 卡 |
| `rollback_version` | `{}` | 用户明确要求回滚且当前会话有可回滚版本 | 沿当前会话版本链切回上一版 |

**AI 程序接口（write_model_program）**：
- 程序是完整 Python 多文件模块（1~8 个顶层 `module.py`，总长 ≤20000 字符，模块间用 `import helpers` / `from helpers import x` 互相引用）；
- 入口文件（默认 `main.py`）必须定义 `build_transform(params=None) -> transform(z, weights) -> pandas.Series`；可选定义 `select_stocks(scores, top_n, params=None) -> list[str]` 接管选股（`scores` 是 pandas.DataFrame，index=股票代码，含 `factor_score` 列与 z_* 因子列）；
- 服务端**不做子进程冒烟测试**；执行前做 AST 白名单 / 受限 builtins / 入口契约校验，真正执行发生在网格寻优与回测阶段；
- 程序生成后先经 **LLM 安全审核**（`program_review.txt`，只检查沙箱/越权/恶意行为，不评价策略逻辑与收益），审核不通过时自动修复一次并复审；仍不通过则返回 review 卡与问题清单，不生成候选、不启用；
- 第三方库：AI 在 `packages` 里申请，LLM 审核通过后服务端用 pip `--target data/ai_packages` 安装并写入审计 `program_install`；未申请的库会被静态校验拒绝；
- 沙箱边界：禁止文件、网络、系统命令、动态执行、下划线属性访问；只允许白名单库 + 本程序模块 + 审核通过的第三方库；
- 程序改动沿用现有 `code_change` 卡片逐行流式输出（多文件以文件名分段），候选/审批/版本/回滚机制不变。

**多候选方案**：同一会话可先后产生多个候选方案（每次 `analyze_model` 一个），服务端按出现顺序编号 1..N 并保存在 `session_state.candidates`；Planner 状态中 `pending_candidates` 给出 `index/name/summary/status/available`。客户说“用第一个/第二个/第三个方案”时通过 `apply_pending_change(candidate=N)` 指定；旧会话（升级前只存了最新 pending）会从已持久化的 work/comparison/approval 卡片中尽力还原候选列表，无法无损还原的旧代码方案标记为 `available=false`，引导客户重新优化。`rollback_version` 只适用于已启用过的版本链，与“在未启用候选之间做选择”是不同动作。

**换模型与调参分离**：客户说“换其他模型/换个模型”但未点名时，AI 调用 `analyze_model(engine="model_type", model_type="", escalate=false)`；服务端选择与当前不同的内置模型，**直接使用预设权重回测**，不跑网格寻优。客户要求构建/更换/编辑模型程序、或“继续优化/上一版未达预期”的结构性升级，则通过 `write_model_program` 由 AI 直接写程序（上一版是权重/内置模型时也优先写程序）。

**`analyze_model` 参数语义**（由 AI 判断，服务端只做确定性策略）：
- `engine`: `auto`（普通优化，由 pipeline 决定） / `weights`（只调因子权重） / `model_type`（点名内置模型）；`code` 已废弃（服务端拒绝并要求使用 `write_model_program`）
- `model_type`: `engine=model_type` 时的内置模型 id（`multi_factor/low_vol_quality/momentum_growth/value_dividend/contrarian`）
- `objective`: `sharpe|cagr|mdd`，空字符串表示默认夏普
- `escalate`: 客户说继续优化/上一版未达预期时为 true；结构性升级由 `write_model_program` 完成
- `reason`: 一句话审计原因（不参与路由）

**动作前置校验**：服务端校验“当前状态是否允许该动作”（apply 需要存在待审批候选方案、rollback 需要可回滚版本、选股需要已确认画像等）；不合法时不猜意图，而是把结构化错误回传 LLM，让它重新调用工具（最多一次）。仍不合法则返回明确错误。

**“更换数学模型”约定**：`数学模型/打分模型/引擎/程序/公式` 指底层打分逻辑，必须调用 `write_model_program` 直接写完整程序；不允许降级成切内置模型或只调权重，也不允许声称没有权限写代码；旧的 `analyze_model(engine="code")` 隐藏生成路径已删除。

**AI 与服务的调参分工**：
- AI 直接负责程序源码（`files`）与参数搜索范围（`param_space`），**不输出最终参数值**；
- 服务端在范围内执行逐轴网格搜索，逐组回测后按目标指标保留最优（见 6.5）；
- 最终候选的参数由服务端写回模型与改动框，并在 tuning 卡中展示搜索范围与得分走势。

## 5. 画像槽位（生活化标签）

`name`(称呼，可选)、`goal`(投资目标)、`horizon`(投资期限)、`capital`(大致投入)、`liquidity_need`(多久可能要用钱)、`loss_reaction`(亏钱时的真实反应)、`drawdown_tolerance`(最多能接受亏多少)、`income_stability`(收入稳定程度)、`experience`(投资经验)、`preferred_industries`(偏好行业)、`excluded_industries`(不碰行业)、`trade_frequency`(希望多久调整一次)、`concerns`(最担心什么)。
必填：`goal`、`horizon`、`drawdown_tolerance`、`loss_reaction`。`name` 只用于称呼，不参与适当性判断，也不能单独推导风险等级；风险等级只有在四项必填信息齐全后才生成。
首次对话先简短自我介绍并自然询问怎么称呼；客户给出称呼后，先开放式询问来意（今天想了解哪方面/希望解决什么问题），不要直接问资金用途；客户表达投资需求后，再进入适当性问题。客户直接提出投资需求或不想透露称呼时不纠缠，顺着客户内容继续。风险等级由 AI 根据信息推导，不直接问“你是什么风险等级”。

AI 追问原则：一次一个问题；先回应/共情，再问；提供生活化选项；不重复问已知信息；最多 3 轮内形成画像总结并请用户确认。

短回答语义：服务端会把最近一轮助手问题、待填槽位和客户原话一起交给模型；客户只答“继续持有/分批补仓/止损”“4 年”“10%”等短句时，必须结合上一问写入对应 `profile_delta`，不得解释为持仓记录或要求补充持仓信息，也不得重复追问已回答字段。服务端会把最近一轮问题、待填槽位和客户原话一起作为上下文交给模型，由模型完成语义映射，不做固定选项匹配。

客户追问概念（“什么是回撤/什么意思/为什么问/没看懂”）时：必须明确说出具体概念并解释其用途；纯概念提问可以只解释不追问。禁止“您指的是刚才那个问题”“请选择一项”“一是/二是/三是”“回复对应选项即可”等问卷式引导，禁止“请按实际情况描述/请如实回答/请回答/必须回答”等审讯式、命令式表达。若首轮仍出现机械或审讯式话术，服务端会调用同一模型在强约束提示词下重写一次/两次；仍不合格则明确报错提示换一种问法，不使用静态模板替换。

## 6. AI 建模流水线（`backend/models/`）

### 6.1 内置模型类型（registry）

内置 5 种：`multi_factor`（均衡多因子）、`low_vol_quality`（低波质量）、`momentum_growth`（趋势成长）、`value_dividend`（价值红利）、`contrarian`（反转）。客户点名清单外模型（SVM、随机森林、梯度提升、聚类等）时，走 `approach=code`，由 AI 用 sklearn 生成自定义模型代码并回测，不再维护固定模型白名单。每个模型：`id/name/description/weights(10因子)`。

### 6.2 LLM 调参方案 JSON（AI 只出引擎 + 范围）

`build_tuning_plan` 要求 LLM 只输出「引擎 + 搜索范围」，由服务端执行真实调参。调用时 state 里包含 `directive`：

```json
{"engine":"auto|weights|model_type|code", "model_type":"low_vol_quality", "objective":"sharpe|mdd|cagr", "escalate":false, "previous_approach":"weights|model_type|code|"}
```

`directive` 由 `backend/chat/directives.resolve_directive()` 从 analyze_model 的 tool args + 上一版结构确定性生成；LLM 必须遵守（engine/model_type/objective/escalate），违反会触发修复重试。LLM 的方案 JSON：

```json
{"approach":"weights|model_type|code",
 "summary":"一句普通用户能懂的话，说明为什么这样改",
 "model_type":"low_vol_quality（仅 model_type）",
 "search_space":{"ep":{"min":0.04,"max":0.2,"step":0.04},"roe":{...}},
 "param_space":{"lookback":{"min":10,"max":60,"step":10,"type":"int","default":20}},
 "weights":{"ep":0.1,...},         // 可选：基底权重，省略时用当前权重
 "code":"python 源码（仅 code）",
 "objective":"sharpe|cagr|mdd",
 "max_trials":24,
 "why":"数据依据（引用真实指标）",
 "expected_effect":"预期改善方向",
 "risks":["..."]}
```

校验规则：
- `weights/model_type`：`search_space` 至少 3 个已知因子，每个 `{min,max,step}` 满足 `0<=min<=max<=0.4`、`step>0`、网格档数 >=2；键顺序即搜索优先级。
- `code`：必须提供 `param_space`（1~4 个合法参数名，`{min,max,step,type,default}`，`type=int|float`）与参数化代码；`code` 方案不能带 `search_space`，`weights/model_type` 不能带 `param_space`。
- `objective` 缺省 `sharpe`；`max_trials` 为整数，服务端寻优再按 4~64 收敛。
- 校验失败时服务端把错误信息 + 原方案回传 LLM 自动修复，最多 2 次（`TUNING_REPAIR_SYSTEM`），成功返回 `repair_attempts`，work 卡展示「自动修复调参方案」。

### 6.3 参数化自定义模型代码契约

AI 生成的代码必须定义：
```python
import numpy as np
import pandas as pd

def build_transform(params=None):
    params = params or {}
    lookback = int(params.get("lookback", 20))

    def transform(z, weights):
        # z: pandas.DataFrame，列含 z_ep/z_bp/...（10 个方向对齐后的因子 z 值）
        # weights: dict[str,float]
        # params: AI 在 param_space 中声明的参数，服务端寻优时逐组传入
        # 返回 pandas.Series（index 与 z 对齐）
        return ...
    return transform
```
- `build_transform` 允许无参数，或恰好一个名为 `params` 的参数；其他签名由 entrypoint 检查拒绝。
- 寻优循环复用已校验代码（`compile_code(code, params, validate=False)`），不重复跑子进程冒烟；候选确定后再用最终参数做一次 `validate_code(code, tuned_params)`。
- 静态检查/AST 白名单/子进程冒烟规则不变（math/numpy/pandas/sklearn/scipy、禁止文件/网络/系统调用/动态执行/下划线属性、源码 ≤6000 字符、10s 冒烟超时）。

### 6.4 预期核对与继续优化

候选回测完成后，服务端根据用户本轮诉求（收益优先 / 回撤优先 / 默认夏普）与画像回撤上限核对是否达到目标，并在 work 卡新增「核对您的目标」步骤。未达预期时：审批卡标题改为「候选未达预期，是否仍要启用？」并增加「继续优化」动作；聊天回复明确差距，不会自动启用模型。用户选择「继续优化」会重新进入 analyze_model，并按结构性升级策略更换引擎或重写代码，再由服务端重新寻优。

### 6.5 服务端网格寻优（`backend/engine/tuning.py`）

- 输入：AI 给出的 `search_space/param_space`、当前基底权重/参数、目标指标 `objective`、回撤上限 `tolerance`、预算 `max_trials`。
- 算法：基线评估 → 按 search_space 顺序逐轴轮转 → 每个轴取若干网格候选（默认每轴 2~5 个）→ 其余参数固定为当前最优 → 每组跑同一区间回测 → 按目标指标保留最优；最大回撤超过画像上限时重罚（优先选不超限的组合）。
- 事件：`run_search` 逐个 yield `start/trial/done` 事件，service 将其映射为 `tuning` 卡片进度推送；事件包含当前试次、总试次、当前参数、最优参数/指标与得分走势。
- 评估次数硬约束在 `max_trials` 内；单组回测异常被跳过，全部失败才报错。
- `mdd` 目标直接比较回撤值（越接近 0 越好）；`cagr/sharpe` 越大越好。

### 6.6 pipeline 与安全模块公开函数

```python
# backend/models/pipeline.py（只保留调参方案协议；legacy proposal 已删除）
def normalize_tuning_plan(raw, base_weights=None, current_model=None) -> dict
def build_tuning_plan(llm, state) -> dict          # 生成「引擎+范围」，含自动修复；按 state.directive 强制方向
TUNING_SYSTEM / TUNING_REPAIR_SYSTEM               # 调参方案提示词

# backend/chat/action_schema.py（唯一动作契约）
TOOL_DEFINITIONS                                    # 发给 LLM 的 OpenAI tools schema
validate_tool_call(name, arguments) -> dict         # Pydantic 校验；失败抛 ToolArgsError

# backend/chat/directives.py（确定性策略，无自然语言解析）
resolve_directive(args, previous_approach="") -> EngineDirective
drawdown_tolerance_ratio(slots) -> float | None     # 画像槽位数值化，非意图识别

# backend/models/custom.py
def validate_code(code, params=None) -> dict        # AST + 冒烟；返回 takes_params 等
def compile_code(code, params=None, *, validate=True) -> transform(z, weights)
```
`backend/models/registry.py` 继续提供 5 种内置模型；`selection.screen/run_backtest` 的 `score_transform` 钩子用于接入参数化自定义引擎。

## 7. 前端（Professional Financial SaaS 三栏）

- 文件：`frontend/index.html`、`app.js`、`style.css`、`charts.js`（纯原生 JS + SVG，无构建链/无 CDN）；登录页为 `login.html`、`login.css`、`login.js`。
- 登录：所有业务 API 需要管理员会话 Cookie；未登录访问 `/` 跳转 `/login`；首次使用在本机创建管理员账号；登出入口在顶部工具栏。
- 用户管理：管理员侧栏可创建、停用/启用、重置密码、删除用户；普通用户只能看到自己的会话，管理员也不能查看其他用户的对话内容。
- 行情同步：服务启动后每日工作日 18:05（Asia/Shanghai）自动通过 Tushare 刷新缓存并热加载；管理端「用户管理」页可查看状态并手动触发。缓存先写临时目录再原子替换，失败保留旧缓存。
- 布局：**顶部工具栏 56px + 左侧导航 224px + 中间 AI 对话工作区（最小 560px）+ 右侧投资上下文 320px**；页面边距 24px、栏间距 16px。小于 1024px 时导航收为图标，移动端为抽屉；小于 1200px 时右侧上下文改为可开合抽屉。
- 导航：左侧为 AI 投资顾问 / 投资组合 / 策略分析 / 历史记录 / 模型管理 / 系统设置；投资组合、策略分析、模型管理、设置使用当前会话状态渲染只读弹窗，不改后端接口。对话区头部提供历史、清空、新对话；输入区支持快捷操作、停止生成、重新生成、清空输入与文本附件。
- 交互：对话是唯一的写操作入口；卡片里的「快捷回复」按钮只是把一句话发进聊天，不直接调接口；审批、启用、回滚都必须经过聊天消息。
- SSE：`POST /api/sessions/{sid}/chat` 用 `fetch` + `ReadableStream` 解析；`token` 事件是 **LLM 流式增量**，前端收到即追加渲染（不再等待整段完成后再打字机播放）；`ui` 事件按 id upsert 卡片。**计划旁白**：`analyze_model` 在方案生成后、代码/寻优开始前，会追加一段 `token` 流，明确说明本轮如何改模型（引擎/因子/参数/目标），并随消息持久化。**代码改动先于寻优**：AI 返回 `approach=code` 的方案后，服务端先逐行推送 `work.code_change.diff`（代码改动卡逐行增长），完成后再发 `tuning` 卡并开始网格寻优；不允许 tuning 卡先于代码改动出现。**代码改动不得丢失**：同一 work id 写入过 `code_change` 后，后续 tuning/回测阶段的每次 work 更新都必须继续带完整 `code_change`，否则前端重渲染会把代码窗口折叠掉。**代码窗口固定**：代码卡出现后，前端在寻优运行期间保持它留在视口（不因 tuning 进度自动滚底），寻优结束后再平滑切到寻优/对比结果。**卡片聚焦**：新卡片首次插入后前端不立即滚底，延迟约 1s 再平滑滚动到卡片；用户已向上滚动时不抢滚动。**work 改动流式契约**：`work` 块先发 `status=running`、`code_change.diff` 为空，随后按行重复发同一 id 的 `work` 块，`code_change.diff` 逐行增长，最后发完整块；前端按 id 覆盖渲染，因此代码框逐行出现。**寻优进度契约**：服务端先发 `tuning` 块（`status=running`、`progress.trial=0`），之后每完成一组回测就发同一 id、同一范围的 `tuning` 块（`progress`、`current`、`best`、`history` 更新），结束时发 `status=done/progress.percent=100` 的块。`error` 显示错误条+重试；停止生成通过 `AbortController` 取消请求。
- 首次进入自动创建会话；刷新后调用 `/state` 恢复右侧上下文（会话级 active_model/versions），从 `/messages` 恢复聊天文本与富卡片（`blocks`，含 tuning 卡）；历史对话通过 `GET /api/sessions` 归档列表切换，切换后右侧模型/版本随会话同步。新建对话必须回到默认「均衡多因子」模型。
- 视觉：**Professional Financial SaaS / Light Mode**，参考 Linear + Bloomberg Terminal + Stripe Dashboard。Token：背景 `#F8FAFC`、卡片 `#FFFFFF`、品牌 Navy `#1E3A5F`、交互蓝 `#2563EB`、边框 `#E2E8F0`；卡片 1px 边框 / 10px 圆角 / 默认无阴影，按钮 6px 圆角，输入框 8px 圆角；正文 14px、辅助 12px、卡片标题 16px。**不使用动态渐变背景、玻璃拟态与大阴影**。
- 模型切换：内置 5 种基础风格；客户点名其他模型时，服务端强制 `approach=code`，由 AI 用 numpy/pandas/sklearn/scipy 写参数化模型代码并通过静态检查+冒烟，再由服务端网格寻优；不再由 LLM 以“没有对应选项”拒绝。模型代码只允许基于当前截面做转导/无监督拟合或伪标签蒸馏。
- 调参进度卡：`tuning` 卡展示进度条（已评估/总组数）、目标指标、当前最优指标与参数、搜索范围（可折叠）和得分走势；所有数值来自服务端真实回测事件，禁止前端伪造进度。
- 确认话术：同一轮只允许出现一段确认请求；画像确认由 AI 自然询问。模型方案审批由 AI 结合回测结果说明 + 审批卡完成；服务端不再追加上一版候选的固定「需要您确认：是否启用……」兜底，避免与其他提问叠加。重复确认段落仍会被修剪；已确认画像字段不得重复询问。
- 状态与金融语义：成功绿 / 警告琥珀 / 危险红 / 信息蓝；A 股涨跌沿用大陆习惯红色上涨、绿色下跌；收益、回撤、夏普等指标统一标注「历史回测数据，不代表未来实际表现」。
- 禁止：mock 数据、独立页签、页面内直接调 selection/tune/profile 接口、伪造进度。

## 8. 文件与分工

- Lead：`backend/agent.py`（流式规划 + tool_calls）、`backend/llm.py`（chat_tools / chat_tools_stream）、`backend/chat/action_schema.py`（工具契约）、`backend/chat/directives.py`（确定性策略）、`backend/service.py`（编排）、`backend/main.py`、`backend/dev_server.py`、`backend/store.py`（会话 active_version_id 单一状态源）、`backend/schemas.py`、`backend/engine/selection.py`（score_transform 钩子）、`backend/engine/tuning.py`（服务端网格寻优）、`tests/` 集成测试、`scripts/e2e_smoke.py`、`scripts/semantic_eval.py`、文档。
- **chat-ui-dev**：只写 `frontend/` 下 `index.html`、`app.js`、`style.css`、`charts.js`（含 tuning 卡与实时进度）。
- **model-pipeline-dev**：只写 `backend/models/*.py`、`tests/test_models.py`。
- 完成标准：`node --check` 通过；后端 `unittest discover` 通过；`scripts/e2e_smoke.py` 通过；真实 LLM 语义矩阵 `scripts/semantic_eval.py` ≥10/10；真实浏览器可完成“说目标→AI 追问→AI 总结并确认→AI 选股→AI 解读→说‘更换数学模型’→AI 写新引擎+寻优进度→审批→当前会话模型更新→新对话回到默认模型→回滚”全流程。
