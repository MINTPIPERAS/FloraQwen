#!/usr/bin/env python
"""最小复现: 单独量化第 216 号张量 (layers.8.self_attn.o_proj.weight), 定位崩溃环节。

逐步打印, 确认崩溃发生在 safetensors 读取 / torch CPU 处理 / bnb 量化 / 上传 GPU 中的哪一步。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

T = "model.language_model.layers.8.self_attn.o_proj.weight"
SHARD = r"F:\hf-cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17\model-00001-of-00002.safetensors"

print("[1] 读取 safetensors ...", flush=True)
from safetensors import safe_open
with safe_open(SHARD, framework="pt") as f:
    t = f.get_tensor(T)
print(f"[1] OK shape={tuple(t.shape)} dtype={t.dtype}", flush=True)

print("[2] torch CPU 拷贝/变形 ...", flush=True)
import torch
t2 = t.float().contiguous()
print("[2] OK", flush=True)

print("[3] bnb 4bit 量化 (CPU) ...", flush=True)
import bitsandbytes as bnb
p4 = bnb.nn.Params4bit(t2, quant_type="nf4", compress_statistics=True)
print("[3] OK", flush=True)

print("[4] 上传 GPU ...", flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)
p4 = p4.cuda()
print("[4] OK", flush=True)

print("[5] 量化后反量化一致性检查 ...", flush=True)
dq = bnb.functional.dequantize_4bit(p4.data, p4.quant_state)
err = (dq.float() - t2.cuda()).abs().max().item()
print(f"[5] OK max_abs_err={err:.4f}", flush=True)
print("MINIMAL REPRO PASSED", flush=True)
