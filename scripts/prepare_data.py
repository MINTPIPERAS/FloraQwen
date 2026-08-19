#!/usr/bin/env python
"""FloraQwen 数据准备脚本: 图片目录 + 标签 CSV -> ShareGPT 训练集 (train.json)。

用法:
    python scripts/prepare_data.py                          # 使用默认路径
    python scripts/prepare_data.py --labels my_labels.csv \
        --images-dir data/raw/images --out data/processed/train.json \
        --max-per-class 100

labels.csv 格式 (UTF-8, 参考 data/raw/labels.example.csv):
    image,species_zh,species_en,family,features,habitat
    rose_001.jpg,月季,Rosa chinensis,蔷薇科,"羽状复叶;茎生皮刺;重瓣杯状花",喜光耐旱

输出: data/processed/train.json (ShareGPT 格式, LLaMA-Factory / Unsloth 通用)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

random.seed(42)

# 多样化提问模板 (随机组合, 增强数据多样性)
QUESTION_TEMPLATES = [
    "这张照片里是什么植物？请给出中文名、学名和科属。",
    "这是什么花？它有哪些典型特征？",
    "帮我识别一下这张图里的植物，并说说它属于什么科。",
    "图片里的是什么植物？请简要介绍它的特征。",
    "这株植物叫什么名字？识别依据是什么？",
    "请问图中是什么花？如果可能，请说明容易和它混淆的物种。",
]

# 回答模板: {0}=中文名 {1}=学名 {2}=科属 {3}=特征描述 {4}=生境/习性
ANSWER_TEMPLATES = [
    "这是**{0}**（{1}，{2}）。识别依据：{3}。{4}",
    "图中是{0}（学名 {1}，属于{2}）。典型特征：{3}。{4}",
    "这是{0}，{2}植物，学名 {1}。可以通过以下特征识别：{3}。{4}",
    "图中植物为{0}（{1}）。主要特征：{3}。它属于{2}。{4}",
]


def parse_features(features: str) -> str:
    """把分号分隔的特征列表转成通顺的中文描述。"""
    parts = [p.strip() for p in features.split(";") if p.strip()]
    if not parts:
        return "外观特征待确认"
    if len(parts) == 1:
        return parts[0]
    return "、".join(parts[:-1]) + "和" + parts[-1]


def build_samples(rows: list[dict], images_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for row in rows:
        img_rel = row["image"].strip()
        img_path = images_dir / img_rel
        if not img_path.is_file():
            print(f"  [跳过] 图片不存在: {img_path}")
            continue

        features = parse_features(row.get("features", ""))
        habitat = row.get("habitat", "").strip() or "常见观赏/野生植物"

        # 为每张图生成 2-4 条不同问法的样本
        n_variants = random.randint(2, 4)
        for _ in range(n_variants):
            q = random.choice(QUESTION_TEMPLATES)
            a = random.choice(ANSWER_TEMPLATES).format(
                row["species_zh"].strip(),
                row.get("species_en", "").strip() or "sp.",
                row.get("family", "").strip() or "未知科",
                features,
                habitat,
            )
            samples.append(
                {
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{q}"},
                        {"from": "gpt", "value": a},
                    ],
                    "images": [str(img_path).replace("\\", "/")],
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 ShareGPT 格式训练集")
    parser.add_argument("--labels", default="data/raw/labels.csv",
                        help="标签 CSV 路径 (默认 data/raw/labels.csv)")
    parser.add_argument("--images-dir", default="data/raw/images",
                        help="图片目录 (默认 data/raw/images)")
    parser.add_argument("--out", default="data/processed/train.json",
                        help="输出 JSON 路径")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="每类最多样本数 (0=不限制, 用于平衡类别)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print(f"[错误] 找不到标签文件: {labels_path}")
        print("请先准备 data/raw/labels.csv (参考 data/raw/labels.example.csv), 图片放入 data/raw/images/")
        raise SystemExit(1)

    with labels_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"读取 {len(rows)} 行标签")

    if args.max_per_class > 0:
        from collections import defaultdict
        by_class: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_class[r["species_zh"]].append(r)
        rows = [r for lst in by_class.values() for r in lst[: args.max_per_class]]
        print(f"按每类最多 {args.max_per_class} 条过滤后: {len(rows)} 行")

    samples = build_samples(rows, Path(args.images_dir))
    if not samples:
        print("[错误] 没有生成任何样本, 请检查图片路径与 labels.csv")
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(samples)} 条训练样本 -> {out_path}")

    # 统计
    from collections import Counter
    classes = Counter(r["species_zh"].strip() for r in rows)
    print(f"覆盖 {len(classes)} 个物种: " + ", ".join(f"{k}({v})" for k, v in classes.items()))


if __name__ == "__main__":
    main()
