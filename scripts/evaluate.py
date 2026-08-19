#!/usr/bin/env python
"""FloraQwen 识别评估脚本 (M4)。

用法:
    # 测试集结构: data/test/<物种中文名>/*.jpg
    python scripts/evaluate.py --test-dir data/test
    python scripts/evaluate.py --test-dir data/test --adapter models/lora-xxx

输出: 总准确率 / 每类准确率 / 失败样例
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_test_set(test_dir: Path) -> list[tuple[Path, str]]:
    """扫描 data/test/<物种>/*.jpg, 返回 [(图片路径, 真值物种名)]"""
    if not test_dir.is_dir():
        print(f"[错误] 测试目录不存在: {test_dir}")
        print("请按 <物种名>/<图片> 结构放置测试集, 例如 data/test/月季/photo1.jpg")
        sys.exit(1)
    samples = []
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_SUFFIX]
        for img in imgs:
            samples.append((img, cls_dir.name.strip()))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="FloraQwen 评估")
    parser.add_argument("--test-dir", default="data/test", help="测试集目录 (子目录=物种名)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--limit", type=int, default=0, help="每类最多测试图片数 (0=不限)")
    parser.add_argument("--detail", action="store_true", help="打印失败样例明细")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from floraqwen import FloraAgent

    samples = collect_test_set(Path(args.test_dir))
    if not samples:
        print("[错误] 测试集中没有图片")
        sys.exit(1)
    print(f"测试集: {len(samples)} 张图片, {len({s for _, s in samples})} 个物种")

    print(f"[加载模型] {args.model} (quant={args.quant}, adapter={args.adapter or '无'})")
    agent = FloraAgent(model_name=args.model, adapter=args.adapter, quant=args.quant)

    per_class = defaultdict(lambda: {"correct": 0, "total": 0, "fails": []})
    for i, (img, truth) in enumerate(samples, 1):
        try:
            res = agent.recognize(img)
        except Exception as e:
            print(f"  [错误] {img.name}: {e}")
            continue
        pred = None
        if res.get("ok") and res["data"]:
            pred = res["data"].get("中文名")
        correct = pred is not None and pred == truth
        per_class[truth]["total"] += 1
        if correct:
            per_class[truth]["correct"] += 1
        else:
            per_class[truth]["fails"].append((img.name, pred))
        if i % 10 == 0 or i == len(samples):
            print(f"  进度 {i}/{len(samples)}")

    total_correct = sum(v["correct"] for v in per_class.values())
    total = sum(v["total"] for v in per_class.values())
    print(f"\n===== 评估结果 =====")
    print(f"Top-1 准确率: {total_correct}/{total} = {total_correct / total * 100:.1f}%")
    print(f"\n每类准确率:")
    for cls in sorted(per_class):
        v = per_class[cls]
        acc = v["correct"] / v["total"] * 100 if v["total"] else 0
        print(f"  {cls}: {v['correct']}/{v['total']} ({acc:.0f}%)")
        if args.detail and v["fails"]:
            for name, pred in v["fails"][:5]:
                print(f"      - {name}: 预测={pred!r}")


if __name__ == "__main__":
    main()
