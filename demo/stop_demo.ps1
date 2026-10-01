# Stops every process started by demo\run_demo.ps1 (and their child interpreters).
# taskkill's stderr is redirected inside cmd.exe, NOT in PowerShell: PowerShell 5.1 wraps a native
# command's redirected stderr in a terminating NativeCommandError, so a PID that had already exited
# used to abort the caller (run_demo.ps1 -Reset) before it started anything.
$root = Split-Path -Parent $PSScriptRoot
$f = Join-Path $root "runtime\pids.json"
if (Test-Path $f) {
  try { $pids = Get-Content $f -Raw | ConvertFrom-Json } catch { $pids = $null }
  if ($pids) {
    foreach ($p in $pids.PSObject.Properties) { cmd /c "taskkill /PID $($p.Value) /T /F >nul 2>nul" }
  }
  Remove-Item $f -Force -ErrorAction SilentlyContinue
}
Write-Host "demo stopped"
