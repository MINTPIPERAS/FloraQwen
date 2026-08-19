# Single-connection resume download of Qwen3-VL safetensors (hf-mirror) + sha256 verify
# v3: verify-fail -> delete tmp and re-download (loop), no early exit
$ErrorActionPreference = 'Stop'
$curl = 'C:\Windows\System32\curl.exe'
$py = 'D:\DataMining\Anaconda3\envs\floraqwen\python.exe'
$hf = 'F:\hf-cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct'
$base = 'https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct/resolve/main/'
$commit = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
$verifyPy = 'F:\GithubDeskClone\FloraQwen\logs\verify_sha256.py'

# 注意 (2026-08-18 已修正): 以下 sha256 曾与文件名写反, 导致即使下载正确也会校验失败并删掉重下 (死循环)。
# 已用 HF tree JSON (commit ebb281ec) 核实: 00001 -> 30a01a05..., 00002 -> 046296a2...
$jobs = @(
    @{ File = 'model-00001-of-00002.safetensors'; Blob = '30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9'; Size = [long]4967229296 },
    @{ File = 'model-00002-of-00002.safetensors'; Blob = '046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6'; Size = [long]3908490048 }
)

foreach ($j in $jobs) {
    $url = $base + $j.File
    $done = Join-Path $hf "blobs\$($j.Blob)"
    $tmp = "$done.dl"

    Write-Host "=== $($j.File) ==="

    # skip if already verified
    if (Test-Path $done) {
        & $py $verifyPy $done $j.Blob
        if ($LASTEXITCODE -eq 0) { Write-Host "  already verified, skip"; continue }
        Write-Host "  existing blob corrupt, remove and re-download"
        Remove-Item $done -Force
    }

    $ok = $false
    for ($round = 1; $round -le 12; $round++) {
        # 1) download / resume via direct CDN url
        $final = & $curl -s -o NUL -w "%{url_effective}" -L --max-time 30 $url
        if (-not $final -or $final -notmatch '^http') {
            Write-Host "  round $round : resolve URL failed, wait"
            Start-Sleep -Seconds 15
            continue
        }
        & $curl -s -C - --retry 80 --retry-delay 3 --max-time 0 -o $tmp $final

        # 2) size check
        if (Test-Path $tmp) { $len = (Get-Item $tmp).Length } else { $len = 0 }
        Write-Host "  round $round : $([math]::Round($len/1MB)) MB / $([math]::Round($j.Size/1MB)) MB"
        if ($len -ne $j.Size) {
            Write-Host "  size mismatch, retry"
            Start-Sleep -Seconds 5
            continue
        }

        # 3) sha256 verify
        & $py $verifyPy $tmp $j.Blob
        if ($LASTEXITCODE -eq 0) {
            $ok = $true
            break
        }
        Write-Host "  sha256 mismatch, delete tmp and re-download"
        Remove-Item $tmp -Force
        Start-Sleep -Seconds 5
    }

    if (-not $ok) { Write-Host "FAIL after 12 rounds: $($j.File)"; exit 1 }
    Move-Item $tmp $done -Force

    # snapshot hardlink
    $snap = Join-Path $hf "snapshots\$commit\$($j.File)"
    if (-not (Test-Path $snap)) {
        New-Item -ItemType HardLink -Path $snap -Target $done | Out-Null
    }
    Write-Host "OK: $($j.File)"
}
Write-Host 'ALL DONE'
