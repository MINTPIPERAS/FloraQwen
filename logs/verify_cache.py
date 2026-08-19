# -*- coding: utf-8 -*-
"""临时: 验证 HF 缓存完整性 (offline 模式, 验证后删除)"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HOME"] = r"F:\hf-cache\huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"

from huggingface_hub import snapshot_download

try:
    p = snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
    print("缓存完整! 路径:", p)
except Exception as e:
    print("缓存不完整:", type(e).__name__, str(e)[:200])
