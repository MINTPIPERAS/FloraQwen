#!/usr/bin/env python
"""训练前验证 collator: 确认视觉 token 进入 input_ids, labels 掩码正确。

用法: python logs/collator_test.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_lora_trl import build_dataset, VLDataCollator  # noqa: E402

from transformers import AutoProcessor  # noqa: E402

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
print("processor 加载完成", flush=True)

ds = build_dataset("data/processed/train.json")[:3]
print(f"样本数: {len(ds)}", flush=True)

# 惰性格式: 数据集条目含 image_path, 图片在 collator 内按 batch 解码
for s in ds:
    print(f"  image_path: {s['image_path']}", flush=True)

collator = VLDataCollator(processor)
batch = collator(ds)

ids = batch["input_ids"]
labels = batch["labels"]
print(f"input_ids shape: {tuple(ids.shape)}, labels shape: {tuple(labels.shape)}", flush=True)

vision_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
print(f"vision_start token id: {vision_start}", flush=True)

for i in range(len(ds)):
    row = ids[i]
    lab = labels[i]
    has_vision = bool((row == vision_start).any())
    n_labeled = int((lab != -100).sum())
    n_masked = int((lab == -100).sum())
    # 回答部分应含中文"这是"等 token, 且未被 mask 的部分长度 > 0
    print(f"  [{i}] len={row.shape[0]} vision_tokens={'YES' if has_vision else 'NO'} "
          f"labeled={n_labeled} masked={n_masked}", flush=True)

assert all(bool((ids[i] == vision_start).any()) for i in range(len(ds))), "视觉 token 缺失!"
assert all(int((labels[i] != -100).sum()) > 0 for i in range(len(ds))), "回答部分被完全 mask!"
print("COLLATOR TEST PASSED", flush=True)
