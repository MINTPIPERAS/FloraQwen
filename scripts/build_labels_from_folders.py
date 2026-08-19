#!/usr/bin/env python
"""从"按物种分目录"的图片文件夹批量生成 labels.csv (通用整理工具)。

适用场景:
    - iNaturalist 下载包 (解压后每类一个目录)
    - 用户自拍按花名分目录整理
    - 任何 "目录名 = 物种名" 的数据

用法:
    python scripts/build_labels_from_folders.py --src D:/photos/flowers \
        --out data/raw/labels.csv --images-dir data/raw/images

行为:
    - 递归扫描 --src 下所有子目录, 目录名作为物种名 (中文)
    - 收集每个目录下的图片 (jpg/png/webp/bmp)
    - 复制(或软链)到 --images-dir, 文件名加前缀避免重名
    - 生成 labels.csv: image,species_zh,species_en,family,features,habitat
      (species_en/family/features 留空, 可后续人工/脚本补充)
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="按文件夹批量生成 labels.csv")
    parser.add_argument("--src", required=True, help="按物种分目录的图片根目录")
    parser.add_argument("--out", default="data/raw/labels.csv", help="输出 labels.csv")
    parser.add_argument("--images-dir", default="data/raw/images", help="图片存放目录")
    parser.add_argument("--min-per-class", type=int, default=1, help="每类最少图片数")
    parser.add_argument("--max-per-class", type=int, default=0, help="每类最多图片数 (0=不限)")
    parser.add_argument("--copy", action="store_true", help="复制文件 (默认建立硬链接, 省空间)")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"[错误] 目录不存在: {src}")
        raise SystemExit(1)

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(p for p in src.iterdir() if p.is_dir())
    print(f"发现 {len(classes)} 个物种目录")

    rows: list[dict] = []
    copied = 0
    for i, cls_dir in enumerate(classes, 1):
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_SUFFIX]
        if args.max_per_class > 0:
            imgs = imgs[: args.max_per_class]
        if len(imgs) < args.min_per_class:
            print(f"  [跳过] {cls_dir.name}: 仅 {len(imgs)} 张")
            continue
        species_zh = cls_dir.name.strip()
        for j, p in enumerate(imgs, 1):
            dst = images_dir / f"{i:03d}_{j:04d}{p.suffix.lower()}"
            if not dst.exists():
                if args.copy:
                    shutil.copy2(p, dst)
                else:
                    try:
                        dst.hardlink_to(p)
                    except OSError:
                        shutil.copy2(p, dst)  # 跨盘不支持硬链接时回退复制
            rows.append({
                "image": dst.name,
                "species_zh": species_zh,
                "species_en": "",
                "family": "",
                "features": "",
                "habitat": "",
            })
            copied += 1

    if not rows:
        print("[错误] 没有生成任何行, 请检查目录结构")
        raise SystemExit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    stat = Counter(r["species_zh"] for r in rows)
    print(f"已整理 {copied} 张图片 -> {images_dir}")
    print(f"labels.csv -> {out} (共 {len(stat)} 类: " + ", ".join(f"{k}({v})" for k, v in stat.most_common(10)) + (" ..." if len(stat) > 10 else "") + ")")
    print("提示: species_en/family/features 留空, 可用 LLM 批量补全 (脚本见后续里程碑)")


if __name__ == "__main__":
    main()
