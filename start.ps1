# start.ps1 - run from the repo root, or via run.bat
# Starts Qdrant, the fleet cloud, the three edge devices and the unified gateway, then opens the one
# URL a person is meant to use. Stop everything with stop.bat / stop.ps1.
param([switch]$Reset)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "  Edge Memory -- starting..." -ForegroundColor Cyan

# Free the ports a previous instance may still hold
foreach ($port in @(6333, 8100, 8101, 8102, 8103, 9000)) {
    $lines = netstat -ano 2>$null | Select-String (":$port ")
    foreach ($line in $lines) {
        $procId = ($line.ToString().Trim() -split '\s+')[-1]
        if ($procId -match '^\d+$') { Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue }
    }
}
Start-Sleep -Milliseconds 800

# 1. Backend stack (Qdrant + fleet cloud + devices A/B/C)
$demoArgs = @("-ExecutionPolicy","Bypass","-File","demo\run_demo.ps1")
if ($Reset) { $demoArgs += "-Reset" }
Start-Process powershell -ArgumentList $demoArgs -WindowStyle Hidden

Write-Host "  Waiting for the device" -ForegroundColor DarkGray -NoNewline
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Milliseconds 1000
    Write-Host "." -NoNewline -ForegroundColor DarkGray
    try { $null = Invoke-RestMethod "http://127.0.0.1:8101/api/health" -TimeoutSec 1; $ready = $true; break } catch {}
}
Write-Host ""
if (-not $ready) {
    Write-Host ""
    Write-Host "  The device server did not start in time. Check runtime\logs\ for errors." -ForegroundColor Red
    exit 1
}

# 2. Unified gateway: one URL, and it injects the auth tokens itself so nobody has to paste one
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "runtime") | Out-Null
$logs = Join-Path $PSScriptRoot "runtime\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$gw = Start-Process -FilePath (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") `
      -ArgumentList "-m","app.unified","--port","9000" -WorkingDirectory $PSScriptRoot `
      -RedirectStandardOutput "$logs\unified.out.log" -RedirectStandardError "$logs\unified.err.log" `
      -WindowStyle Hidden -PassThru
"$($gw.Id)" | Set-Content (Join-Path $PSScriptRoot "runtime\unified_pid.txt") -Encoding ascii

Write-Host "  Waiting for the app" -ForegroundColor DarkGray -NoNewline
$appReady = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    Write-Host "." -NoNewline -ForegroundColor DarkGray
    try { $null = Invoke-RestMethod "http://127.0.0.1:9000/health" -TimeoutSec 1; $appReady = $true; break } catch {}
}
Write-Host ""
if (-not $appReady) {
    Write-Host ""
    Write-Host "  The app did not start. Check runtime\logs\unified.err.log" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Open this:   http://127.0.0.1:9000/" -ForegroundColor Green
Write-Host ""
Write-Host "  Also available, for engineering detail:" -ForegroundColor DarkGray
Write-Host "    Device A   http://127.0.0.1:8101/      Device B  http://127.0.0.1:8102/" -ForegroundColor DarkGray
Write-Host "    Fleet      http://127.0.0.1:8100/      Qdrant    http://127.0.0.1:6333/dashboard" -ForegroundColor DarkGray
Write-Host ""
Start-Process "http://127.0.0.1:9000/"
