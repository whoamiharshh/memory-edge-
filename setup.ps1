# Edge Memory portable setup for Windows PowerShell 5.1+
[CmdletBinding()]
param(
  [switch]$SkipQdrant,
  [switch]$SkipModels,
  [switch]$Dev
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Require-Command([string]$Name, [string]$InstallHint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name was not found. $InstallHint"
  }
}

Write-Host "Edge Memory setup" -ForegroundColor Cyan
Require-Command "python" "Install Python 3.12+ from https://www.python.org/downloads/ and enable PATH."

$python = (Get-Command python).Source
$version = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$parts = $version.Split('.')
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 12)) {
  throw "Python 3.12 or newer is required; detected $version."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "Creating .venv with Python $version..." -ForegroundColor DarkGray
  & $python -m venv .venv
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
Write-Host "Installing pinned dependencies..." -ForegroundColor DarkGray
# The prebuilt CPU wheel for llama-cpp-python (the optional evidence-brief model) lives on the author's
# own index, so there is no compiler step. Only --extra-index-url is passed: --index-strategy is a uv
# flag that pip rejects outright, which made every run of this script fail at the install step. The
# misspelled "abeten.github.io" mirror that used to sit beside it is gone: that domain is nobody's, and
# an unowned host has no business being trusted as a package index.
$extra = @("--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu")
if ($Dev) { & $venvPython -m pip install -r requirements.txt @extra }
else {
  $runtime = Get-Content requirements.txt | Where-Object { $_ -notmatch '^\s*(pytest|playwright|psutil|scikit-learn|pandas)\b' }
  $runtimePath = Join-Path $env:TEMP "edge-memory-runtime-$PID.txt"
  $runtime | Set-Content -Encoding UTF8 $runtimePath
  try { & $venvPython -m pip install -r $runtimePath @extra } finally { Remove-Item $runtimePath -Force -ErrorAction SilentlyContinue }
}

New-Item -ItemType Directory -Force runtime, models_cache, backups | Out-Null
if (-not $SkipQdrant) {
  Write-Host "Installing Qdrant Server for this PC..." -ForegroundColor DarkGray
  & $venvPython -m tools.qdrant_local download
}
if (-not $SkipModels) {
  Write-Host "Downloading the offline text embedding model..." -ForegroundColor DarkGray
  & $venvPython -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='models_cache')"
}

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Start: .\start.ps1   (or run.bat)" -ForegroundColor White
Write-Host "Open:  http://127.0.0.1:9000/   - one page, no sign-in" -ForegroundColor White
if ($SkipModels) { Write-Host "Models were skipped; first use may download them." -ForegroundColor Yellow }
