#!/usr/bin/env python
"""隔离测试: 不用 transformers, 直接按加载顺序切片 shard1 的所有张量。

复现 _materialize_copy 的 `tensor[...]` 操作, 判断崩溃是否属于 safetensors/torch 读取层。
"""
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from safetensors import safe_open

SHARD = r"F:\hf-cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17\model-00001-of-00002.safetensors"


def dot_natural_key(s: str):
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s))


with safe_open(SHARD, framework="pt") as f:
    keys = sorted(f.keys(), key=dot_natural_key)
    print(f"共 {len(keys)} 个张量", flush=True)
    for i, k in enumerate(keys):
        t = f.get_tensor(k)
        t2 = t[...]  # 与 _materialize_copy 相同的切片操作
        if i % 20 == 0:
            print(f"[{i}] {k} {tuple(t2.shape)}", flush=True)

print("ISOLATION TEST PASSED (全部切片完成)", flush=True)
