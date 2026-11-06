# 运行手册（v3：原生 function calling + 服务端网格寻优）

## 1. 最快跑起来

```powershell
cd a-share-ai-screener
$py = ".venv\Scripts\python.exe"   # Windows 启动脚本；也可用 DSH runtime python

& $py -m backend.data.build_cache --source sample
$env:DEEPSEEK_API_KEY = "sk-..."
& $py -m backend.dev_server --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

正式入口（需要 `pip install -r requirements.txt`）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:LLM_MODEL = "deepseek-chat"
& $py -m uvicorn backend.main:app --port 8000
```

## 2. 演示脚本（3 分钟）

1. 输入“我想为养老攒点钱”，AI 会先问一个缺失的信息；
2. 补充“大概 5 年不用、最多能接受亏 15%、真亏了会有点慌但先不卖”；
3. 回“可以，就按这个来”——AI 自己完成画像确认 + 选股，展示画像卡、选股卡、指标与解读；
4. 追问“为什么最大回撤这么大”，AI 基于这次回测事实解释；
5. 说“帮我优化一下模型”——AI 通过 tool 调用发起优化：看表现 → 翻数据 → 提出「引擎 + 参数搜索范围」方案；
6. 服务端开始网格寻优：「参数网格寻优」卡片实时推进（已评估组数、当前最优指标、Top 参数、搜索范围、得分走势）；work 卡同步展示步骤与改动 diff；
7. 说“更换数学模型”——AI 必须调用 `analyze_model(engine="code")`，写一套新的参数化打分引擎，再由服务端重新寻优（而不是只调权重或换内置模型）；
8. 审批卡说明寻优组数、目标指标、回测对比与过拟合风险；点“同意”在当前会话启用新模型；
9. 点顶部「新对话」验证：新会话回到默认「均衡多因子」，不会沿用上一个对话的模型；
10. 回到原会话说“回滚到上一个版本”，切回该会话基线。

## 3. 使用真实 A 股数据

推荐把 token 写入项目根目录 `.env`（已 gitignore）：

```powershell
# .env
# TUSHARE_TOKEN=...
# SOURCE=tushare
# UNIVERSE=csi800
```

```powershell
& $py -m backend.data.build_cache --source tushare --limit 30 --start 2022-01-01
# 重启服务以加载新缓存
```

当前真实缓存：29 只核心 A 股 × 1171 个交易日（2022-01-04 至 2026-11-05）。若 token 额度较低，`index_weight/fina_indicator/daily_basic/index_daily` 会降级：使用核心标的池、真实日线、等权基准，财务质量因子缺失；额度恢复后重新 build 即自动补齐。服务运行期只读 `data/a_share/` 缓存，不访问网络。历史数据仅用于回测演示，不构成投资建议。

## 4. 常见问题

| 现象 | 处理 |
|---|---|
| 启动报缺少 LLM Key | 设置 `DEEPSEEK_API_KEY` 后重启服务 |
| 启动报数据缓存缺失 | `python -m backend.data.build_cache --source sample` |
| AkShare 拉取慢 | 加 `--limit 200`，或先用示例数据演示 |
| AI 生成的代码被拒绝 | 代码安全检查/冒烟/回测拦截属于预期；可再说一次“重新给个方案” |
| 网格寻优较慢 | 正常：每一组参数都要跑一遍全区间回测，进度卡会实时显示组数与当前最优；真实数据约 0.3~0.5 秒/组 |
| 新对话模型不是上一个对话的 | 预期行为：模型改动只在当前会话生效，新对话默认使用均衡多因子；如需继续用调优模型，请回到原会话 |
| 想清空会话和版本 | `.\scripts\demo_reset.ps1`（保留数据缓存） |
| 端口占用 | 启动时换 `--port 8010` |

## 5. 验收命令

```powershell
$py = ".venv\Scripts\python.exe"
& $py -m unittest discover -s tests -v
$node --check frontend/app.js
$node --check frontend/charts.js
& $py scripts\e2e_smoke.py

# 真实 LLM 语义矩阵（需要 DEEPSEEK_API_KEY；10/10 才允许发版）
& $py scripts\semantic_eval.py
```
