# Stops everything started by start.ps1
$root = $PSScriptRoot

# Stop demo services (Qdrant, cloud, edge devices)
$f = Join-Path $root "runtime\pids.json"
if (Test-Path $f) {
  $pids = Get-Content $f | ConvertFrom-Json
  foreach ($p in $pids.PSObject.Properties) { & taskkill /PID $p.Value /T /F 2>$null | Out-Null }
  Remove-Item $f
}

# Stop unified gateway
$upid = Join-Path $root "runtime\unified_pid.txt"
if (Test-Path $upid) {
  try { & taskkill /PID (Get-Content $upid).Trim() /T /F 2>$null | Out-Null } catch {}
  Remove-Item $upid
}

Write-Host "All services stopped." -ForegroundColor Green
