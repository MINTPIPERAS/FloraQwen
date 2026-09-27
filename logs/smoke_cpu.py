#!/usr/bin/env python
"""CPU 加载对照: 4bit 量化 + device_map='cpu', 验证崩溃是否与 CUDA 上下文相关。

若 CPU 加载也崩溃 -> 与 CUDA 无关, 是加载路径/内存问题;
若 CPU 加载成功 -> 与 CUDA 上下文/显存转移相关。
用法: python logs/smoke_cpu.py
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
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("[load] device_map='cpu' 开始 ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    quantization_config=bnb,
    device_map="cpu",
    attn_implementation="sdpa",
)
print("[load] 成功", flush=True)
print("CPU LOAD PASSED", flush=True)
