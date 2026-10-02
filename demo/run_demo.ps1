# Starts the whole demo on one laptop: Qdrant Server, the fleet cloud, Device A (site1), Device B (site2) and
# Device C (site3).
# Usage (from the repo root):   powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
#                               powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset   (wipe runtime/ first)
param([switch]$Reset)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "runtime\logs"

if ($Reset) { & (Join-Path $PSScriptRoot "stop_demo.ps1"); Remove-Item -Recurse -Force (Join-Path $root "runtime") -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $logs | Out-Null

# 1. Qdrant Server (official Windows release binary, v1.19.1)
$env:QDRANT__STORAGE__STORAGE_PATH = Join-Path $root "runtime\qdrant\storage"
$env:QDRANT__STORAGE__SNAPSHOTS_PATH = Join-Path $root "runtime\qdrant\snapshots"
$env:QDRANT__TELEMETRY_DISABLED = "true"
$q = Start-Process -FilePath (Join-Path $root "qdrant_server\qdrant.exe") -WorkingDirectory (Join-Path $root "qdrant_server") `
     -RedirectStandardOutput "$logs\qdrant.out.log" -RedirectStandardError "$logs\qdrant.err.log" -WindowStyle Hidden -PassThru
for ($i = 0; $i -lt 60; $i++) { try { Invoke-RestMethod http://127.0.0.1:6333/readyz -TimeoutSec 1 | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }
Write-Host "Qdrant Server  : http://127.0.0.1:6333/dashboard  (pid $($q.Id))"

# 2. Fleet cloud (issues demo tokens on first run -> runtime\cloud\bootstrap.json)
$c = Start-Process -FilePath $py -ArgumentList "-m","cloud.main","--bootstrap","--port","8100" -WorkingDirectory $root `
     -RedirectStandardOutput "$logs\cloud.out.log" -RedirectStandardError "$logs\cloud.err.log" -WindowStyle Hidden -PassThru
$boot = Join-Path $root "runtime\cloud\bootstrap.json"
for ($i = 0; $i -lt 120; $i++) { if ((Test-Path $boot) -and (Test-NetConnection 127.0.0.1 -Port 8100 -InformationLevel Quiet -WarningAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 500 }
$b = Get-Content $boot | ConvertFrom-Json

# 3. Devices. Operator tokens are fixed here for the demo; change them for anything real.
$pids = @{ qdrant = $q.Id; cloud = $c.Id }
foreach ($d in @(@{n="devA"; s="site1"; p=8101; deny="Ravi,Priya,Pune"}, @{n="devB"; s="site2"; p=8102; deny="Anil"},
                 @{n="devC"; s="site3"; p=8103; deny="Meena"})) {
  $tok = $b.devices.($d.n).token
  $proc = Start-Process -FilePath $py -ArgumentList "-m","edge.main","--name",$d.n,"--site",$d.s,"--port",$d.p,"--cloud","http://127.0.0.1:8100",`
          "--device-token",$tok,"--operator-token","operator-$($d.n)","--denylist",$d.deny,"--fit-baseline" -WorkingDirectory $root `
          -RedirectStandardOutput "$logs\$($d.n).out.log" -RedirectStandardError "$logs\$($d.n).err.log" -WindowStyle Hidden -PassThru
  $pids[$d.n] = $proc.Id
}
$pids | ConvertTo-Json | Set-Content (Join-Path $root "runtime\pids.json")
Write-Host "Fleet cloud UI : http://127.0.0.1:8100/   admin token: $($b.admin)"
Write-Host "Device A UI    : http://127.0.0.1:8101/   operator token: operator-devA"
Write-Host "Device B UI    : http://127.0.0.1:8102/   operator token: operator-devB"
Write-Host "Device C UI    : http://127.0.0.1:8103/   operator token: operator-devC   (site 3: the disagreeing report)"
Write-Host "Logs           : $logs      Stop everything: demo\stop_demo.ps1"
# 4. Unified gateway
$gw = Start-Process -FilePath $py -ArgumentList "-m","app.unified","--port","9000" -WorkingDirectory $root `
     -RedirectStandardOutput "$logs\unified.out.log" -RedirectStandardError "$logs\unified.err.log" -WindowStyle Hidden -PassThru
$gw.Id | Set-Content (Join-Path $root "runtime\unified_pid.txt")
for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-RestMethod http://127.0.0.1:9000/health -TimeoutSec 1 | Out-Null; break }
    catch { Start-Sleep -Milliseconds 500 }
}
Write-Host "Unified gateway : http://127.0.0.1:9000/   (pid $($gw.Id))"
