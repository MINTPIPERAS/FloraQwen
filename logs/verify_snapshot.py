# -*- coding: utf-8 -*-
"""临时: 按 HF tree JSON 核对缓存快照 (无 torch 依赖, 纯文件系统校验)"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HF = Path(r"F:\hf-cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct")
COMMIT = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
snap = HF / "snapshots" / COMMIT
tree = json.loads((HF / "trees" / f"{COMMIT}.json").read_text(encoding="utf-8"))
files = tree["files"]

ok = True
for name, meta in files.items():
    p = snap / name
    if not p.exists():
        print(f"[缺失] {name}")
        ok = False
        continue
    size = p.stat().st_size
    if size != meta["size"]:
        print(f"[大小不符] {name}: 期望 {meta['size']} 实际 {size}")
        ok = False
    else:
        print(f"[OK] {name}  {size/1e6:.1f}MB")

print()
print("结果:", "全部一致, HF 缓存可离线加载 ✓" if ok else "存在不一致 ✗")
