# Stops every process started by demo\run_demo.ps1 (and their child interpreters).
$root = Split-Path -Parent $PSScriptRoot
$f = Join-Path $root "runtime\pids.json"
if (Test-Path $f) {
  $pids = Get-Content $f | ConvertFrom-Json
  foreach ($p in $pids.PSObject.Properties) { & taskkill /PID $p.Value /T /F 2>$null | Out-Null }
  Remove-Item $f
}
Write-Host "demo stopped"
