# 运行手册（Runbook）

## 1. 首次运行（离线演示，最快的路径）

```powershell
cd a-share-ai-screener
$py = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $py -m backend.data.build_cache --source sample     # 生成示例 A 股数据缓存（CSV/parquet 自动选择）
$env:LLM_PROVIDER = "fake"                            # 无 Key 时仅本地联调；正式演示请配真实 Key
& $py -m backend.dev_server --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

正式环境（安装了 requirements 的 Python）：

```powershell
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "sk-..."
$env:LLM_MODEL = "deepseek-chat"
uvicorn backend.main:app --port 8000
```

## 2. 使用真实 A 股数据

```powershell
& $py -m backend.data.build_cache --source akshare --universe csi800 --limit 300
# 或（需要 TUSHARE_TOKEN，字段更全）
& $py -m backend.data.build_cache --source tushare --limit 300
```

- AkShare 免费、无需 token；数据量小时用 `--limit` 控制首次拉取时间。
- 服务运行期只读 `data/a_share/` 缓存，不访问网络。
- 历史数据只用于回测与演示，不构成投资建议。

## 3. 演示流程（3 分钟）

1. 打开页面 → 「AI 对话画像」：输入“稳健型、3-5 年、能接受 20% 回撤、偏好消费、不碰 ST”；
2. AI 追问缺失项 → 画像卡确认；
3. 点击「开始选股」→ 查看 Top N、10 因子贡献、组合权重、OOS 指标；
4. 「AI 解读」：查看基于事实的解读，追问「为什么选它/风险在哪」；
5. 「AI 调参台」：点击开始调优 → 观察参数演化与指标变化 → 启用候选版本 → 回滚；
6. 收尾声明：历史模拟不代表未来收益，不构成投资建议。

## 4. 常见问题

| 现象 | 处理 |
|---|---|
| 启动报“缺少 LLM API Key” | 设置 `DEEPSEEK_API_KEY`；无 Key 仅本地看 UI 可用 `LLM_PROVIDER=fake` |
| 启动报“A 股数据缓存缺失” | 运行 `python -m backend.data.build_cache --source sample` |
| AkShare 拉取慢/限流 | 加 `--limit 200`，或改用示例数据；服务运行期不联网 |
| 端口占用 | `--port 8010` 换端口 |
| 想重置演示状态 | `.\scripts\demo_reset.ps1`（清空 SQLite，保留数据缓存） |
| 调参太慢 | 前端传入 trials 更小（如 8），或环境变量 `DEFAULT_TRIALS` 调小 |

## 5. 验收命令

```powershell
& $py -m unittest discover -s tests -v
node --check frontend/app.js
node --check frontend/charts.js
```
