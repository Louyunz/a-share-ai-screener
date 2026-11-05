# A 股 AI 选股 MVP · 接口契约（开发冻结版）

本文件是并开发包的唯一契约。所有模块按此实现，禁止擅自改字段名/语义；确需变更先告知 Lead。

## 0. 环境

- 项目根：`C:\Users\77578\Documents\deepseek-harness\default-workspace\a-share-ai-screener`
- Python：`C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`（3.12，已装 pandas 3 / numpy 2 / scipy）
- Node：`C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe`（24）
- **没有** fastapi/uvicorn/pytest/akshare/tushare，也不能联网安装。后端同时提供标准库 dev_server 供本地运行。
- git 由 Lead 统一管理，队友**不要执行任何 git 命令**。

## 1. 数据层契约（DataBundle）

`backend/data/base.py` 定义：

```python
@dataclass
class DataBundle:
    prices: pd.DataFrame       # 前复权收盘价，宽表 index=交易日(Timestamp,升序), columns=代码
    close_raw: pd.DataFrame    # 未复权收盘价，与 prices 对齐（判断涨跌停）
    amount: pd.DataFrame       # 成交额（元），与 prices 对齐（流动性、停牌判断）
    fundamentals: pd.DataFrame # index=代码；列：ep/bp/roe/ocf_to_profit/revenue_growth/pe/pb/total_mv/float_mv
    industry: pd.DataFrame     # index=代码；列：industry（申万一级行业名）
    benchmark: pd.Series       # 沪深300 收盘价，index=交易日
    universe: list[str]        # 股票池代码
    meta: dict                 # {"source": "sample"/"akshare"/"tushare", "as_of": "YYYY-MM-DD", ...}

class IDataPort(Protocol):
    def load(self) -> DataBundle: ...
```

缓存文件（`data/a_share/`）：`prices`、`close_raw`、`amount`、`fundamentals`、`benchmark` 五类数值表（有 pyarrow 时写 `.parquet`，否则写 `.csv`，读取时自动识别），以及 `industry.csv`、`universe.csv`、`meta.json`。服务运行期只读缓存。

## 2. 引擎契约（engine-dev 实现）

文件：`backend/engine/factors.py`、`selection.py`、`tuner.py`。只允许 import：`pandas`、`numpy`、`scipy`、标准库；**不得 import fastapi/akshare/tushare/其他 backend 模块**（engine 是纯函数库）。

### 2.1 factors.py

```python
FACTOR_NAMES = ["ep","bp","roe","ocf_to_profit","revenue_growth","momentum_60","reversal_5","volatility_60","liquidity_20","size"]
FACTOR_DIRECTION = {"ep":1,"bp":1,"roe":1,"ocf_to_profit":1,"revenue_growth":1,
                    "momentum_60":1,"reversal_5":-1,"volatility_60":-1,"liquidity_20":-1,"size":-1}
RISK_WEIGHTS = {"conservative": {...10 项和为1...}, "balanced": {...}, "aggressive": {...}}
DEFAULT_WEIGHTS = RISK_WEIGHTS["balanced"]
WEIGHT_PRESETS: dict[str, dict] = {...}

def weights_for(risk_level: str) -> dict            # 未知等级回退 balanced
def validate_weights(weights: dict) -> None          # 10 项齐全、和为1±1e-6、每项 ∈[0,0.4]，否则 ValueError
def register_preset(name: str, weights: dict) -> None
def compute_factor_scores(prices, close_raw, amount, fundamentals, industry,
                          weights=None, asof=None, min_history=120) -> pd.DataFrame
```

`compute_factor_scores` 返回：index=代码，列包含每个因子的 `raw_<name>`、`z_<name>`（行业中性化后），以及 `factor_score`；按 score 降序。规则：
- 用 `asof` 之前（含）的行情切片；`momentum_60`=近60交易日收益、`reversal_5`=近5日收益、`volatility_60`=近60日日收益年化波动、`liquidity_20`=近20日日均成交额取对数、`size`=ln(流通市值)。
- 去极值 1%/99%、z-score；行业中性化=按 industry 分组 z-score（行业样本<3 时用全市场 z）。
- 方向对齐：`z * FACTOR_DIRECTION`，保证越大越好；缺失值 z=0。
- 数据不足 `min_history` 的代码剔除。

### 2.2 selection.py

```python
def screen(data, weights, top_n=10, asof=None, exclude_codes=None) -> tuple[pd.DataFrame, pd.DataFrame]
# data 可为 DataBundle 或 dict；top_df 列：code, score, industry, z 列（至少 z_ep 等 10 个）
def build_portfolio(top_df, prices, method="equal_weight", capital=1_000_000.0) -> pd.DataFrame
# 列：code, weight_pct, amount, method；method ∈ {equal_weight, risk_parity}
def run_backtest(data, weights, start, end, top_n=10, rebalance_days=20,
                 cost_buy=0.0003, cost_sell=0.0013, risk_free_rate=0.02) -> dict
# 返回 {"nav": pd.Series, "benchmark_nav": pd.Series,
#       "metrics": {"cagr","sharpe","mdd","ir","win_rate"}, "rebalances": int}
```

A 股规则：
- T+1：调仓日 t 收盘出信号，t+1 收盘价成交（价格缺失则顺延）。
- 停牌/无成交（price 或 amount 为 NaN/0）标的买卖跳过。
- 涨跌停：以 `close_raw` 日收益判断，买入时涨幅 ≥ +9.8% 跳过，卖出时跌幅 ≤ −9.8% 跳过。
- 成本：买入收 `cost_buy`，卖出收 `cost_sell`（含印花税）；按换手计。
- 风险平价：用调仓日前 60 日日收益年化波动倒数加权。
- 指标：CAGR、Sharpe（日频年化、减无风险）、MDD、IR（相对基准）、月胜率。

### 2.3 tuner.py

```python
def run_tuning(data, base_weights=None, trials=60, top_n=10, start=None, end=None,
               seed=42, progress_cb=None) -> dict
# 返回 {"best_weights","base_metrics","best_metrics","objective","trials":[...]}
# progress_cb(trial: dict) 每完成一组调用一次；trial 字段：
#   {"trial_index":int,"weights":{10项},"metrics":{cagr,sharpe,mdd,ir},"objective":float,"is_best":bool}
# 约束：随机采样 10 项权重、和为 1、每项 ∈[0,0.4]；目标 J = sharpe - 0.5*abs(mdd) + 0.2*ir
# 防过拟合：同时计算 IS（前 70% 区间）与 OOS（后 30%）；要求 IS cagr >= 0.8*base IS cagr
```

### 2.4 测试（engine-dev 负责）

`tests/test_engine.py`（unittest，不依赖 pytest）：构造 60 只股票、400 个交易日的合成数据（固定 seed），验证：10 因子列齐全/权重校验/行业中性化稳健/筛选排序/组合权重和为 100/回测指标有限且 MDD≤0/调优输出约束与进度回调次数。运行命令：

```
& $py -m unittest tests.test_engine -v
```

## 3. 后端 API 契约（Lead 实现，前端按此对接）

统一前缀 `/api`，JSON；SSE 格式 `event: <type>\ndata: <json>\n\n`。

| 方法 | 路径 | 请求 | 响应 / SSE 事件 |
|---|---|---|---|
| GET | `/api/health` | — | `{ok, source, as_of, llm:"ok"}` |
| POST | `/api/sessions` | — | `{session_id}` |
| POST | `/api/sessions/{sid}/messages` | `{content}` | SSE：`token{text}`、`slots{slots}`、`profile_ready{profile}`、`error{message}` |
| GET | `/api/sessions/{sid}/messages` | — | `{messages:[{role,content,created_at}]}` |
| GET | `/api/sessions/{sid}/profile` | — | `{slots, risk_level, confirmed, complete}` |
| PUT | `/api/sessions/{sid}/profile` | `{slots, confirmed}` | `{slots, risk_level, confirmed, complete}` |
| POST | `/api/sessions/{sid}/selection` | `{top_n:"10"}` | `{task_id}` |
| GET | `/api/sessions/{sid}/selection/events?task_id=` | — | SSE：`started`、`progress{stage,percent,message}`、`result{top,portfolio,metrics,nav}`、`error` |
| POST | `/api/sessions/{sid}/tune` | `{trials:60}` | `{job_id}` |
| GET | `/api/sessions/{sid}/tune/events?job_id=` | — | SSE：`tuning_started{job_id,trials_total,base_weights,base_metrics}`、`trial{trial_index,weights,metrics,objective,is_best}`、`best_update{best_weights,best_metrics}`、`done{candidate_version_id,compare}`、`error` |
| GET | `/api/versions` | — | `{versions:[{id,weights,metrics,status,created_at,parent_id}], active_id}` |
| POST | `/api/versions/{vid}/activate` | `{reason}` | `{active_id}` |
| POST | `/api/versions/{vid}/rollback` | `{reason}` | `{active_id}` |

画像槽位（`slots`，均字符串）：`risk_level`(conservative/balanced/aggressive)、`horizon`、`drawdown_tolerance`、`goal`、`experience`、`excluded_industries`（逗号分隔）、`capital`。`complete` 判定：`risk_level/horizon/drawdown_tolerance` 非空。

## 4. 前端契约（frontend-dev 实现）

- 文件：`frontend/index.html`、`app.js`、`charts.js`、`style.css`；可选 `mock-data.js`（仅在 `?mock=1` 时加载，用于无后端时查看 UI，代码中注明 dev-only；默认不启用）。
- 零外部依赖：**不引入 ECharts/CDN/构建工具**，图表用纯 SVG/CSS 实现（`charts.js` 暴露 `Charts.bar/radar/line/scatter/gauge` 等函数即可）。
- 深色科技风仪表盘，配色：背景 `#0B1220→#111C2E`、卡片 `#151F33`、边框 `#2A3A55`、主色 `#22D3EE`、强调橙 `#F59E0B`、成功 `#34D399`、危险 `#FB7185`。
- 四个视图（顶部导航）：**AI 对话画像**（聊天气泡、流式 token、槽位 chips、画像卡确认/编辑）；**选股结果**（Top N 表、因子贡献条、组合权重、指标卡、净值线）；**AI 解读**（解读文本、追问输入、免责声明）；**AI 调参台**（三栏：左 AI 操作流时间轴 / 中因子权重对比条+雷达+候选散点 / 右指标卡+净值对比+版本时间轴；底部启用/回滚；由 `tune` SSE 驱动；支持无数据骨架屏）。
- 交互：`localStorage.session_id`；SSE `EventSource`（对话用 fetch 流式读取，POST 无法用 EventSource，请用 `fetch`+`ReadableStream` 解析 SSE）；错误显示红色错误条+重试按钮；不实现降级/假数据（除 mock 模式）。
- 验收：`node --check frontend/app.js`、`node --check frontend/charts.js` 通过；`frontend/index.html` 与后端 dev_server 联调可跑通对话→选股→解读→调参。

## 5. 协作与交付

- engine-dev 只写：`backend/engine/*.py`、`tests/test_engine.py`。
- frontend-dev 只写：`frontend/*`。
- Lead 写其余全部后端文件与测试，并统一 git 提交。
- 完成后向 Lead 报告：文件清单、运行过的命令与结果、遗留风险；不要执行 git/pip/git commit。
