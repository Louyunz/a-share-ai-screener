# 启动本地预览服务器（无需 fastapi/uvicorn）
# 用法： .\scripts\run_dev.ps1            （默认 127.0.0.1:8000）
#        .\scripts\run_dev.ps1 -Port 8010
param(
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
Set-Location $root

if (-not (Test-Path "data\a_share\prices.parquet")) {
    Write-Host "[run_dev] 未发现数据缓存，使用内置示例数据构建..."
    & $py -m backend.data.build_cache --source sample
}

& $py -m backend.dev_server --host $HostName --port $Port
