# ============================================================================
# FloraQwen 环境搭建脚本 (Windows / PowerShell)
# 用法: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#        powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Mirror <pypi镜像>
#
# 目标环境: RTX 4060 Laptop 8GB (或同级 8GB 独显), Windows + conda
# 要点:
#   - 使用 Python 3.11 (bitsandbytes / unsloth 对 3.12/3.13 支持滞后)
#   - torch 使用 cu126 wheel (适配较新 NVIDIA 驱动, 610.47 已确认可用)
#   - 默认走清华镜像 (国内网络, download.pytorch.org 常超时); 镜像不可用时自动回退官方源
#   - unsloth 为可选 (训练加速); 装不上时可用 LLaMA-Factory 或纯 transformers 训练
# ============================================================================

param(
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$EnvName = "floraqwen"
$PythonVer = "3.11"
$TorchIndex = "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu126"
$TorchIndexFallback = "https://download.pytorch.org/whl/cu126"
$env:CONDA_REPORT_ERRORS = "false"   # 避免 conda 错误报告交互提示卡住脚本

Write-Host "=== [1/5] 检查 conda ===" -ForegroundColor Cyan
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 conda, 请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Red
    exit 1
}
conda --version

Write-Host "`n=== [2/5] 创建 conda 环境 $EnvName (python $PythonVer) ===" -ForegroundColor Cyan
$exists = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if ($exists) {
    Write-Host "环境已存在, 跳过创建" -ForegroundColor Yellow
} else {
    conda create -n $EnvName python=$PythonVer -y
}

Write-Host "`n=== [3/5] 安装项目依赖 (镜像: $Mirror) ===" -ForegroundColor Cyan
conda run -n $EnvName pip install -r requirements.txt -i $Mirror --timeout 60

Write-Host "`n=== [4/5] 安装 PyTorch (CUDA 12.6, 清华镜像, 最后安装保证覆盖 CPU 版) ===" -ForegroundColor Cyan
conda run -n $EnvName pip install --force-reinstall torch torchvision --index-url $TorchIndex --timeout 120
# 验证是否为 CUDA 版, 若不是则回退官方源
$cudaOk = conda run -n $EnvName python -c "import torch; print(torch.version.cuda or 'CPU')" 2>$null
Write-Host "  当前 torch CUDA 版本: '$cudaOk'"
if ("$cudaOk" -match "^(CPU|None)?\s*$") {
    Write-Host "  清华镜像未提供 CUDA wheel, 回退官方源安装..." -ForegroundColor Yellow
    conda run -n $EnvName pip install --force-reinstall torch torchvision --index-url $TorchIndexFallback --timeout 180
} else {
    Write-Host "  torch CUDA 版本确认: $cudaOk" -ForegroundColor Green
}

Write-Host "`n=== [5/5] 可选: 安装 Unsloth (QLoRA 加速, Windows 支持见官方文档) ===" -ForegroundColor Cyan
try {
    conda run -n $EnvName pip install unsloth -i $Mirror --timeout 60
    Write-Host "unsloth 安装成功" -ForegroundColor Green
} catch {
    Write-Host "unsloth 安装失败(不影响 transformers 基线推理)。
        训练可改用 LLaMA-Factory: conda run -n $EnvName pip install llamafactory -i $Mirror
        或 WSL2 中安装 unsloth: https://github.com/unslothai/unsloth" -ForegroundColor Yellow
}

Write-Host "`n=== 环境检查 ===" -ForegroundColor Cyan
conda run -n $EnvName python scripts/check_env.py

Write-Host "`n完成! 后续命令示例:" -ForegroundColor Green
Write-Host "  conda activate $EnvName"
Write-Host "  python scripts/infer.py --image data/raw/images/xxx.jpg  # 基线推理"
