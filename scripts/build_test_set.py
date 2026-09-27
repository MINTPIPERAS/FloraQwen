#!/usr/bin/env python
"""从 Flowers102 官方 test 划分构建 evaluate.py 期望的评估集。

产出: data/test/<中文名>/image_xxxxx.jpg (硬链接, 不占额外磁盘)
      默认每类最多 30 张 (全 102 类 ≈ 3060 张), 可用 --max-per-class 调整。

用法:
    python scripts/build_test_set.py
    python scripts/build_test_set.py --max-per-class 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from make_flowers102_labels import SPECIES  # noqa: E402  (同目录脚本)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Flowers102 评估集")
    parser.add_argument("--src", default="data/raw/flowers102")
    parser.add_argument("--out", default="data/test")
    parser.add_argument("--max-per-class", type=int, default=30)
    args = parser.parse_args()

    src = Path(args.src)
    splits = json.loads((src / "splits.json").read_text(encoding="utf-8"))
    test_set = splits["test"]  # 6149 张官方 test

    with (src / "labels.csv").open("r", encoding="utf-8-sig") as f:
        rows = {Path(r["image"]).name: r["species_en"] for r in csv.DictReader(f)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_class = defaultdict(list)
    for name in test_set:
        cls = rows.get(name)
        if cls not in SPECIES:
            continue
        zh = SPECIES[cls][0]
        per_class[zh].append(name)

    total = 0
    for zh, names in sorted(per_class.items()):
        cls_dir = out / zh
        cls_dir.mkdir(parents=True, exist_ok=True)
        for name in names[: args.max_per_class]:
            src_img = src / "images" / name
            dst = cls_dir / name
            if src_img.is_file() and not dst.exists():
                try:
                    os.link(src_img, dst)  # 硬链接, 零额外磁盘
                    total += 1
                except OSError:
                    pass  # 跨盘/不支持时跳过

    n_cls = len(per_class)
    print(f"评估集就绪: {total} 张 ({n_cls} 类) -> {out}")
    print("评估: python scripts/evaluate.py --test-dir data/test --limit N")


if __name__ == "__main__":
    main()
