# 重置演示状态：清空 SQLite（会话/画像/版本/审计）但保留 A 股数据缓存
# 用法： .\scripts\demo_reset.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$db = Join-Path $root "data\app.db"
$wal = Join-Path $root "data\app.db-wal"
$shm = Join-Path $root "data\app.db-shm"
foreach ($f in @($db, $wal, $shm)) {
    if (Test-Path $f) { Remove-Item $f -Force; Write-Host "[demo_reset] 已删除 $f" }
}
Write-Host "[demo_reset] 完成：下次启动服务将创建干净的 SQLite。"
Write-Host "[demo_reset] 如需重建数据缓存： .\scripts\build_cache.ps1 -Source sample"
