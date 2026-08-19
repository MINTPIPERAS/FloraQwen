# FloraQwen one-click runner (English-only on purpose: avoid PS GBK/UTF-8 BOM issues)
# Usage:
#   .\scripts\run.ps1 scripts\check_env.py
#   .\scripts\run.ps1 scripts\infer.py --image data\raw\images\rose_001.jpg
# What it does:
#   1) cd to project root (no matter where you call it from)
#   2) set HF_* env vars so the model cache on F: is used (never C:)
#   3) run the script with the floraqwen conda env python directly (no activation needed)
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# model cache / mirror env (all on F: drive)
$env:HF_HOME = 'F:\hf-cache\huggingface'
$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_OFFLINE = '1'          # model already cached -> offline load, no network
$env:MODELSCOPE_CACHE = 'F:\hf-cache\modelscope_cache'

$envPy = 'D:\DataMining\Anaconda3\envs\floraqwen\python.exe'
if (-not (Test-Path $envPy)) {
    Write-Host "[ERROR] python not found: $envPy" -ForegroundColor Red
    exit 1
}

$target = Join-Path $root $Script
if (-not (Test-Path $target)) {
    Write-Host "[ERROR] script not found: $target (run from anywhere, script path is relative to project root)" -ForegroundColor Red
    exit 1
}

Write-Host "[FloraQwen] run: $Script" -ForegroundColor Cyan
Write-Host "[FloraQwen] python: $envPy" -ForegroundColor Cyan
& $envPy $target @ScriptArgs
exit $LASTEXITCODE
