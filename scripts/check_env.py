#!/usr/bin/env python
"""FloraQwen 环境检查脚本。

验证: Python 版本 / torch / CUDA / 显存 / 关键依赖是否就绪。
缺失项会给出修复提示, 不会直接崩溃。
用法: python scripts/check_env.py
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

OK = "\033[92m[OK]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def main() -> None:
    print(f"Python        : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    if sys.version_info >= (3, 13):
        print(f"  {WARN} Python >=3.13: bitsandbytes/unsloth 支持可能滞后, 建议 conda 环境用 3.11")

    # torch / CUDA
    try:
        import torch
        print(f"torch         : {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU           : {name} ({total:.1f} GB)")
            if total < 9:
                print(f"  {WARN} 显存 {total:.1f}GB: 只能走 QLoRA/4-bit 路线, 见 docs 方案")
            print(f"torch CUDA    : {torch.version.cuda}")
        else:
            print(f"  {FAIL} CUDA 不可用! 检查驱动 (nvidia-smi) 与 torch wheel 是否匹配 (cu126)")
            print(f"        重装: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    except ImportError:
        print(f"torch         : {FAIL} 未安装. 运行 scripts\\setup_env.ps1 或 pip install torch --index-url https://download.pytorch.org/whl/cu126")

    # 关键依赖 (bitsandbytes 等 import 时可能因环境/沙箱限制失败, 容错处理)
    for pkg in ["transformers", "bitsandbytes", "peft", "accelerate", "datasets", "trl", "PIL"]:
        try:
            mod = __import__(pkg if pkg != "PIL" else "PIL")
            ver = getattr(mod, "__version__", "?")
            print(f"{pkg:<13}: {OK} {ver}")
        except Exception as e:
            hint = "pip install unsloth" if pkg == "unsloth" else f"pip install {pkg.lower()}"
            print(f"{pkg:<13}: {WARN} import 失败 ({type(e).__name__}: {str(e)[:60]}) -> {hint}")

    # unsloth (可选)
    try:
        import unsloth  # noqa: F401
        print(f"{'unsloth':<13}: {OK} 已安装 (训练加速可用)")
    except ImportError:
        print(f"{'unsloth':<13}: {WARN} 未安装(可选). Windows 安装参考 https://github.com/unslothai/unsloth ; 或改用 LLaMA-Factory")

    # nvidia-smi (失败不影响其余检查)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            print("nvidia-smi   : " + out.stdout.strip().replace("\n", " | "))
        else:
            print(f"nvidia-smi   : {WARN} 执行失败 (rc={out.returncode})")
    except Exception as e:
        print(f"nvidia-smi   : {WARN} 不可用 ({e})")

    # 磁盘空间提醒 (模型 + 数据约需 20-40GB)
    try:
        import shutil as _sh
        usage = _sh.disk_usage(str(Path.cwd()))
        print(f"磁盘剩余     : {usage.free / 1024**3:.0f} GB (建议 >= 40GB)")
    except Exception:
        pass

    print("\n检查完成. 若 torch/CUDA 为 [OK], 即可运行: python scripts/infer.py --image <图片路径>")


if __name__ == "__main__":
    main()
