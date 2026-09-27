#!/usr/bin/env python
"""训练前冒烟测试: 只加载 Qwen3-VL-4B (4-bit NF4) + processor 然后退出。

目的: 在正式训练前验证模型加载路径 (bitsandbytes 4bit + device_map=auto + sdpa)
在本机 Windows 上可正常工作, 提前暴露 0xC0000005 类问题。
用法: python logs/smoke_load_model.py
"""
import os
import sys

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = r"F:\hf-cache\huggingface"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

print("[1/3] torch cuda:", torch.cuda.is_available(), torch.__version__)

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

print("[2/3] 加载 4-bit NF4 模型 ...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="sdpa",
)
print("[2/3] 模型加载成功")

print("[3/3] 加载 processor ...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
print("[3/3] processor 加载成功")

# 检查视觉编码器属性 (训练脚本 freeze_vision 依赖)
vision = getattr(getattr(model, "model", model), "visual", None)
print("visual 属性:", type(vision).__name__ if vision is not None else None)

# 显存占用
gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"峰值显存: {gb:.2f} GB")
print("SMOKE TEST PASSED")
