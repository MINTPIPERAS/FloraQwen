#!/usr/bin/env python
"""加载变体对照测试: 复现 2026-08-18 诊断报告的 9 组对照思路, 找出本机当前可用的加载参数组合。

用法:
    python logs/smoke_variant.py v1   # 不传 dtype + 去双量化 + device_map=auto
    python logs/smoke_variant.py v2   # 不传 dtype + 双量化 + device_map="cuda:0"
    python logs/smoke_variant.py v3   # 显式 fp16 + 双量化 + device_map=auto
    python logs/smoke_variant.py v4   # 不传 dtype + 双量化 + low_cpu_mem_usage=True
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

variant = sys.argv[1] if len(sys.argv) > 1 else "v1"

base = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
kw = dict(device_map="auto", attn_implementation="sdpa")

if variant == "v1":
    bnb = BitsAndBytesConfig(**base, bnb_4bit_use_double_quant=False)
elif variant == "v2":
    bnb = BitsAndBytesConfig(**base, bnb_4bit_use_double_quant=True)
    kw["device_map"] = "cuda:0"
elif variant == "v3":
    bnb = BitsAndBytesConfig(**base, bnb_4bit_use_double_quant=True)
    kw["torch_dtype"] = torch.float16
elif variant == "v4":
    bnb = BitsAndBytesConfig(**base, bnb_4bit_use_double_quant=True)
    kw["low_cpu_mem_usage"] = True
else:
    raise SystemExit(f"未知变体 {variant}")

print(f"[{variant}] 开始加载, kwargs={kw}", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    quantization_config=bnb,
    **kw,
)
print(f"[{variant}] 加载成功, device={model.device}", flush=True)
print(f"[{variant}] RESULT: PASSED", flush=True)
