# -*- coding: utf-8 -*-
# 分片并行下载 torch cu126 wheel v2 (严格校验 + 多轮重试)
$ErrorActionPreference = 'Stop'
$curl = 'C:\Windows\System32\curl.exe'
$url = 'https://download.pytorch.org/whl/cu126/torch-2.11.0%2Bcu126-cp311-cp311-win_amd64.whl'
$proxy = 'http://127.0.0.1:7890'
$out = 'F:\GithubDeskClone\FloraQwen\logs\torch_dl'
$size = [long]2596413186
$parts = 8
$per = [long][math]::Ceiling($size / $parts)

Write-Host "总大小: $size bytes; 每片: $per"

# 期望每片大小
$expects = @()
for ($i = 0; $i -lt $parts; $i++) {
    $start = [long]($i * $per)
    $end = [long][math]::Min($start + $per - 1, $size - 1)
    $expects += [long]($end - $start + 1)
}

# 多轮下载, 直到所有分片完整或达到轮数上限
for ($round = 1; $round -le 6; $round++) {
    $procs = @()
    $missing = 0
    for ($i = 0; $i -lt $parts; $i++) {
        $part = "$out\part_$i.bin"
        if (Test-Path $part) { $len = (Get-Item $part).Length } else { $len = 0 }
        if ($len -eq $expects[$i]) { continue }
        $missing++
        Remove-Item $part -Force -ErrorAction SilentlyContinue
        $start = [long]($i * $per)
        $end = [long]($expects[$i] + $start - 1)
        Write-Host "第${round}轮: 下载 part_$i [$start-$end]"
        $p = Start-Process -FilePath $curl -ArgumentList @('-s','--retry','40','--retry-delay','3','-x',$proxy,'-r',"$start-$end",'-o',$part,$url) -PassThru -NoNewWindow
        $procs += $p
    }
    if ($missing -eq 0) { Write-Host "所有分片完整"; break }
    Write-Host "第${round}轮: $missing 个分片待下载, 等待..."
    $procs | Wait-Process
}

# 最终校验
$fail = 0
for ($i = 0; $i -lt $parts; $i++) {
    $part = "$out\part_$i.bin"
    if (Test-Path $part) { $len = (Get-Item $part).Length } else { $len = 0 }
    if ($len -ne $expects[$i]) { Write-Host "part_$i 大小不符: $len / $($expects[$i])"; $fail++ }
    else { Write-Host "part_$i OK ($len)" }
}
if ($fail -gt 0) { Write-Host "仍有 $fail 个分片失败, 请重跑脚本"; exit 2 }

# 合并
$merged = "$out\torch-2.11.0+cu126.whl"
Remove-Item $merged -Force -ErrorAction SilentlyContinue
$fs = [System.IO.File]::OpenWrite($merged)
try {
    for ($i = 0; $i -lt $parts; $i++) {
        $bytes = [System.IO.File]::ReadAllBytes("$out\part_$i.bin")
        $fs.Write($bytes, 0, $bytes.Length)
    }
} finally { $fs.Close() }
$final = Get-Item $merged
Write-Host "合并完成: $($final.Length) bytes (期望 $size)"
if ($final.Length -ne $size) { Write-Host '大小不匹配!'; exit 3 }
Write-Host 'DONE'
