# ONE-CLICK LAUNCHER: starts all services and opens the unified app in your browser.
# Usage:   powershell -ExecutionPolicy Bypass -File start.ps1
#          powershell -ExecutionPolicy Bypass -File start.ps1 -Reset   (wipe runtime/ and start fresh)
#
# Opens: http://127.0.0.1:9000  (one clean page — Dashboard, Episodes, Search, Fleet)
# Stop:  powershell -ExecutionPolicy Bypass -File stop.ps1

param([switch]$Reset)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "runtime\logs"

if ($Reset) {
    & (Join-Path $root "demo\stop_demo.ps1") 2>$null
    $upid = Join-Path $root "runtime\unified_pid.txt"
    if (Test-Path $upid) { try { Stop-Process -Id (Get-Content $upid) -Force -ErrorAction SilentlyContinue } catch {} }
    Remove-Item -Recurse -Force (Join-Path $root "runtime") -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $logs | Out-Null

Write-Host ""
Write-Host "=== Machine Memory at the Edge ===" -ForegroundColor Cyan
Write-Host "Starting services..." -ForegroundColor Gray
Write-Host ""

# 1. Qdrant Server
$env:QDRANT__STORAGE__STORAGE_PATH = Join-Path $root "runtime\qdrant\storage"
$env:QDRANT__STORAGE__SNAPSHOTS_PATH = Join-Path $root "runtime\qdrant\snapshots"
$env:QDRANT__TELEMETRY_DISABLED = "true"
Write-Host "[1/4] Starting Qdrant Server..." -ForegroundColor Yellow
$q = Start-Process -FilePath (Join-Path $root "qdrant_server\qdrant.exe") -WorkingDirectory (Join-Path $root "qdrant_server") `
     -RedirectStandardOutput "$logs\qdrant.out.log" -RedirectStandardError "$logs\qdrant.err.log" -WindowStyle Hidden -PassThru
for ($i = 0; $i -lt 60; $i++) { try { Invoke-RestMethod http://127.0.0.1:6333/readyz -TimeoutSec 1 | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }
Write-Host "       Qdrant ready (pid $($q.Id))" -ForegroundColor Green

# 2. Fleet cloud
Write-Host "[2/4] Starting Fleet Cloud..." -ForegroundColor Yellow
$c = Start-Process -FilePath $py -ArgumentList "-m","cloud.main","--bootstrap","--port","8100" -WorkingDirectory $root `
     -RedirectStandardOutput "$logs\cloud.out.log" -RedirectStandardError "$logs\cloud.err.log" -WindowStyle Hidden -PassThru
$boot = Join-Path $root "runtime\cloud\bootstrap.json"
for ($i = 0; $i -lt 120; $i++) { if ((Test-Path $boot) -and (Test-NetConnection 127.0.0.1 -Port 8100 -InformationLevel Quiet -WarningAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 500 }
$b = Get-Content $boot | ConvertFrom-Json
Write-Host "       Fleet Cloud ready (pid $($c.Id))" -ForegroundColor Green

# 3. One edge device
Write-Host "[3/4] Starting edge device..." -ForegroundColor Yellow
$tok = $b.devices.devA.token
$dev = Start-Process -FilePath $py -ArgumentList "-m","edge.main","--name","devA","--site","site1","--port","8101","--cloud","http://127.0.0.1:8100",`
       "--device-token",$tok,"--operator-token","operator-devA","--fit-baseline" -WorkingDirectory $root `
       -RedirectStandardOutput "$logs\devA.out.log" -RedirectStandardError "$logs\devA.err.log" -WindowStyle Hidden -PassThru
Write-Host "       Device ready (pid $($dev.Id))" -ForegroundColor Green

$pids = @{ qdrant = $q.Id; cloud = $c.Id; devA = $dev.Id }
$pids | ConvertTo-Json | Set-Content (Join-Path $root "runtime\pids.json")

# 4. Unified gateway (the single URL)
Write-Host "[4/4] Starting unified gateway on port 9000..." -ForegroundColor Yellow
$gw = Start-Process -FilePath $py -ArgumentList "-m","app.unified","--port","9000" -WorkingDirectory $root `
      -RedirectStandardOutput "$logs\unified.out.log" -RedirectStandardError "$logs\unified.err.log" -WindowStyle Hidden -PassThru
$gw.Id | Set-Content (Join-Path $root "runtime\unified_pid.txt")
for ($i = 0; $i -lt 30; $i++) { try { Invoke-RestMethod http://127.0.0.1:9000/health -TimeoutSec 1 | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }
Write-Host "       Gateway ready (pid $($gw.Id))" -ForegroundColor Green

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Open in your browser:" -ForegroundColor White
Write-Host ""
Write-Host "  http://127.0.0.1:9000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Stop: powershell -ExecutionPolicy Bypass -File stop.ps1" -ForegroundColor Gray
Write-Host "=============================================" -ForegroundColor Cyan

# Open browser automatically
Start-Process "http://127.0.0.1:9000"
