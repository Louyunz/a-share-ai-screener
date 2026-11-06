# 一键验收：Python 单元/集成测试 + 结构 E2E + 前端语法检查
# 用法： .\scripts\test_all.ps1
#       可选：设置 DEEPSEEK_API_KEY 后加 -Semantic 运行真实 LLM 语义矩阵
param([switch]$Semantic)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venv) { $venv } else {
    "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
}
$node = if (Get-Command node -ErrorAction SilentlyContinue) {
    "node"
} else {
    "C:\Users\77578\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe"
}
Set-Location $root

Write-Host "== Python tests =="
& $py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Python 测试失败" }

Write-Host "== Structural E2E (FakeLLM) =="
& $py scripts\e2e_smoke.py
if ($LASTEXITCODE -ne 0) { throw "E2E 冒烟失败" }

Write-Host "== Frontend syntax check =="
foreach ($f in @("frontend\app.js", "frontend\charts.js")) {
    if (Test-Path $f) {
        & $node --check $f
        if ($LASTEXITCODE -ne 0) { throw "语法检查失败: $f" }
        Write-Host "OK $f"
    }
}

if ($Semantic) {
    Write-Host "== Real LLM semantic matrix =="
    & $py scripts\semantic_eval.py
    if ($LASTEXITCODE -ne 0) { throw "真实 LLM 语义矩阵未通过" }
}

Write-Host "== 全部通过 =="
