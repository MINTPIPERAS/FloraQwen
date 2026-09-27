#!/usr/bin/env python
"""带内存监控的 4bit 加载测试: 采样系统可用内存与进程峰值, 验证"提交内存耗尽"假设。

用法: python logs/smoke_memwatch.py
"""
import os
import sys
import threading
import time

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

stop = threading.Event()
samples = []


def monitor():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    while not stop.is_set():
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        samples.append((time.time(), m.ullAvailPhys / 2**30, m.ullAvailPageFile / 2**30, m.dwMemoryLoad))
        time.sleep(0.5)


mon = threading.Thread(target=monitor, daemon=True)
mon.start()

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("[load] 开始 ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="sdpa",
)
stop.set()
print("[load] 成功", flush=True)
print(f"peak_avail_phys_gb={max(s[1] for s in samples):.2f}", flush=True)
print(f"min_avail_pagefile_gb={min(s[2] for s in samples):.2f}", flush=True)
print(f"max_mem_load={max(s[3] for s in samples)}%", flush=True)
print("MEMWATCH PASSED", flush=True)
