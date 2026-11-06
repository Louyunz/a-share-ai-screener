# 设计决策摘要（v3.0）

## 产品形态
- **AI 对话工作区**：画像、选股、解读、模型优化、版本管理仍在一个对话流完成；外层为 Professional Financial SaaS 三栏（左侧导航、中间对话、右侧上下文）。
- **AI 主动执行（原生 function calling）**：Planner 通过 OpenAI 兼容的 `tools/tool_calls` 调用后端工具；`content` 只输出客户可见 JSON（reply/profile_delta）。服务端校验工具参数并执行，不再从自然语言关键词推断意图。
- **右侧投资上下文**：自动展示画像摘要、AI 活动时间线、当前会话模型/版本、最近结果；宽屏可收起，窄屏为抽屉，不取代聊天操作。

## AI 建模（本版核心差异）
- **单一意图权威**：动作参数契约在 `backend/chat/action_schema.py`（Pydantic 模型 + OpenAI tools schema）。`analyze_model` 必须带 `{engine, model_type, objective, escalate, reason}`，例如“更换数学模型”必须 `engine="code"`。服务端只做 schema 校验、确定性策略与审计，不猜语义。
- **AI 只写引擎 + 定范围**：`approach=weights/model_type` 给 10 因子 `search_space`；`approach=code` 给参数化引擎代码（`build_transform(params=None)`）与 1~4 个参数的 `param_space`。AI 不直接给最终参数。
- **服务端网格寻优**：`backend/engine/tuning.py` 在 AI 给定范围内做逐轴网格搜索：当前参数基线 → 按优先级逐因子轮转 → 每轴取网格候选、其余固定 → 每组参数跑全区间回测 → 按目标指标（sharpe/cagr/mdd，支持回撤上限惩罚）保留最优。评估次数受 `max_trials` 硬约束，并可逐组推送进度。
- **实时进度**：`tuning` 卡片按同一 id upsert；前端实时显示进度条、已评估组数、当前最优指标、Top 参数、搜索范围与得分走势。进度由真实回测事件驱动，不伪造。
- 生成的代码必须通过：长度/语法检查 → AST 白名单（math/numpy/pandas/sklearn/scipy，禁止文件/网络/系统调用/动态执行/下划线属性）→ 子进程冒烟（10s 超时）→ 全区间回测；调参方案校验失败自动修复最多 2 次。
- 任何模型变更都生成候选，用户同意后才在当前会话激活；会话各自保留版本链，聊天一句话即可回滚。
- **会话级模型**：模型只在被启用的会话内生效，新对话自动回到默认均衡多因子；`session.active_version_id` 是唯一模型状态源，`session_state` 不再保存 active_model/metrics/version_id。

## 技术取舍
- 轻量前后端分离：FastAPI 单进程同时托管静态页面与 API；聊天用 `fetch + ReadableStream` 解析 SSE（POST 不能使用 EventSource）。
- **真流式回复**：LLM 客户端使用 OpenAI 兼容 `stream=true` 解析 SSE，content delta 实时转发为 `token` 事件；前端收到即渲染，不再等整段结束后再打字机播放。tool_calls 随流增量拼接，流结束后校验执行；画像更新走 `update_profile` 工具，content 不再包 JSON。
- 无构建链、无 CDN、纯 SVG 图表，保证比赛现场一键启动、弱网可演示。
- SQLite 存会话/画像/模型版本/审计；`version` 表带 `session_id`，`session` 表带 `active_version_id`（SCHEMA_VERSION=5）。
- 真实模型路径走 OpenAI 兼容接口；LLM 客户端自实现 `chat_tools` / `chat_tools_stream`（复用现有 urllib 传输，未引入额外 SDK），支持一次工具参数修复与“无工具调用时的语义复核”。
- 生产包不含任何假模型；结构测试使用 `tests/support/scripted_llm.py`（仅测试目录）。真实语义由 `scripts/semantic_eval.py` 语义矩阵验收（10 个意图用例，10/10 才发版）。

## 合规与边界
- AI 不承诺收益、不荐股、不说“稳赚”；回复有违禁词拦截，解读基于真实回测事实。
- 寻优结果来自历史回测，审批卡与寻优卡都提示“最优参数可能过拟合、历史业绩不代表未来表现”。
- 历史回测与示例数据仅用于技术演示，不构成投资建议。
