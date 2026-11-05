# 设计摘要（V1.2 · A 股 AI 选股 MVP）

> 完整设计见 `docs/智投罗盘_A股AI选股_MVP精简设计说明书_v1.2.pdf`，接口契约见 `docs/CONTRACT.md`。

## 目标与范围

3 天可交付的 A 股 AI 选股演示系统：AI 多轮对话建立画像 → 按画像适配 10 因子选股 → A 股规则回测 → 基于事实的解读与追问 → AI 参数调优（可视化调参台 + 版本回滚）。不做代码级自优化、不做降级兜底、不做多服务与容器编排。

## 关键决策

1. **轻量前后端分离**：单进程 FastAPI 同时托管 SPA 与 API，无 Nginx、无构建链、无消息队列；对话/进度/调参 trial 均走 SSE。
2. **画像只走 AI 对话**：AI 多轮追问补全 6 项槽位，用户确认后落库；AI 不可用直接报错。
3. **A 股口径**：股票池沪深 300+中证 500；10 因子；T+1、涨跌停、停牌、双边成本进入回测；数据 AkShare 为主、Tushare 为辅，离线缓存。
4. **无兜底**：所有异常返回明确错误（HTTP 5xx / SSE error），前端提示重试，不展示降级内容。
5. **自优化只调参数**：随机搜索 10 因子权重，样本外评估 + 人工启用/回滚；调参台由真实 `trial` 事件驱动，可美化但不伪造。

## 角色与模块

- 前端 SPA：对话画像、选股结果、AI 解读、AI 调参台。
- 后端：api-server、agent-core、factor-engine、backtest-engine、tuner、store（SQLite）、data-adapter（AkShare/Tushare）。
- 存储：SQLite（sessions/messages/profiles/versions/audit_logs）+ parquet 缓存。

## 3 天排期

- D1：数据缓存与 10 因子计算、FastAPI 骨架、AI 对话画像（SSE 打通）。
- D2：选股/组合/A 股回测 API、AI 解读与追问、结果页。
- D3：调参 tuner + AI 调参台可视化、版本启用/回滚、异常与演示打磨。

## 演示路径（3 分钟）

对话建立画像 → 确认 → 选股（Top N/因子贡献/组合/回测指标）→ AI 解读与追问 → AI 调参台（参数演化动画、指标对比、启用候选/回滚）→ 合规声明。
