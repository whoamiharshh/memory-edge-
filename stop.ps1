# Stops everything started by start.ps1
# See demo\stop_demo.ps1 for why taskkill's stderr is redirected inside cmd.exe and not in PowerShell.
$root = $PSScriptRoot

function Stop-Tree([string]$ProcId) {
  if ($ProcId -match '^\d+$') { cmd /c "taskkill /PID $ProcId /T /F >nul 2>nul" }
}

# Demo services (Qdrant, fleet cloud, edge devices)
$f = Join-Path $root "runtime\pids.json"
if (Test-Path $f) {
  try { $pids = Get-Content $f -Raw | ConvertFrom-Json } catch { $pids = $null }
  if ($pids) { foreach ($p in $pids.PSObject.Properties) { Stop-Tree "$($p.Value)" } }
  Remove-Item $f -Force -ErrorAction SilentlyContinue
}

# Unified gateway (one clean URL)
$upid = Join-Path $root "runtime\unified_pid.txt"
if (Test-Path $upid) {
  Stop-Tree ((Get-Content $upid -Raw).Trim())
  Remove-Item $upid -Force -ErrorAction SilentlyContinue
}

Write-Host "All services stopped." -ForegroundColor Green
