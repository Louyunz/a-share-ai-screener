# 构建 A 股数据缓存
# 用法： .\scripts\build_cache.ps1                 （示例数据，离线可用）
#        .\scripts\build_cache.ps1 -Source akshare -Limit 200
param(
    [ValidateSet("sample", "akshare", "tushare")]
    [string]$Source = "sample",
    [int]$Stocks = 150,
    [int]$Days = 750,
    [int]$Limit = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
Set-Location $root

$argsList = @("-m", "backend.data.build_cache", "--source", $Source)
if ($Source -eq "sample") {
    $argsList += @("--stocks", "$Stocks", "--days", "$Days")
} elseif ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}
& $py @argsList
