# Reset demo state: clear SQLite (sessions/profiles/versions/audit) but keep A-share cache.
# Usage: .\scripts\demo_reset.ps1
# Stop the server before running this script.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$files = @(
    (Join-Path $root "data\app.db"),
    (Join-Path $root "data\app.db-wal"),
    (Join-Path $root "data\app.db-shm")
)
foreach ($file in $files) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "[demo_reset] removed $file"
    }
}
Write-Host "[demo_reset] done: next server start creates a clean SQLite database."
Write-Host "[demo_reset] rebuild cache if needed: .\scripts\build_cache.ps1 -Source sample"
