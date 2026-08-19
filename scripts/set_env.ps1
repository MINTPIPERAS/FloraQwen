# ============================================================================
# FloraQwen 运行环境变量设置 (Windows)
# 用法: 每次开新终端运行一次 (或手动执行下面命令):
#     . .\scripts\set_env.ps1        (dot-source, 仅当前会话生效)
#
# 说明: 模型缓存 (~9GB) 放在 F 盘, 避免写满 C 盘系统盘
# ============================================================================

$env:HF_HOME = "F:\hf-cache\huggingface"     # HF 模型缓存目录 (F 盘)
$env:HF_ENDPOINT = "https://hf-mirror.com"   # HF 国内镜像
$env:HF_HUB_DISABLE_XET = "1"                # 禁用 Xet 协议 (否则下载报 401)
$env:HF_HUB_OFFLINE = "1"                    # 模型已缓存完整: 纯离线加载, 绝不联网/绝不重复下载 (需要联网下新模型时改成 "0")

Write-Host "[FloraQwen] 环境变量已设置:" -ForegroundColor Cyan
Write-Host "  HF_HOME          = $env:HF_HOME"
Write-Host "  HF_ENDPOINT      = $env:HF_ENDPOINT"
Write-Host "  HF_HUB_DISABLE_XET = $env:HF_HUB_DISABLE_XET"
