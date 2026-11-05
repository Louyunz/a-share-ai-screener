# 一键验收：Python 单元/集成测试 + 前端语法检查
# 用法： .\scripts\test_all.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$node = "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
Set-Location $root

Write-Host "== Python tests =="
& $py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Python 测试失败" }

Write-Host "== Frontend syntax check =="
foreach ($f in @("frontend\app.js", "frontend\charts.js", "frontend\mock-data.js")) {
    if (Test-Path $f) {
        & $node --check $f
        if ($LASTEXITCODE -ne 0) { throw "语法检查失败: $f" }
        Write-Host "OK $f"
    }
}
Write-Host "== 全部通过 =="
