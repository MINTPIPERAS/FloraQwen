#!/usr/bin/env python
"""下载并整理 Oxford 102 Flowers 数据集 (国际基线/评估集)。

用法:
    python scripts/download_flowers102.py --out data/raw/flowers102
    python scripts/download_flowers102.py --out data/raw/flowers102 --skip-download  # 只整理已下载文件

数据源: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
    - 102flowers.tgz     ~648MB, 8189 张图片
    - imagelabels.mat    每张图的类别标签 (1-102)
    - setid.mat          官方 train/val/test 划分

产出:
    data/raw/flowers102/
        images/          所有图片 (按类别子目录整理)
        labels.csv       图片 -> 类别 (image, species_en, species_zh 留空待补中文)
        splits.json      官方划分 (train/val/test 图片清单)

依赖: scipy (读取 .mat), 可选: pip install scipy
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FILES = {
    "102flowers.tgz": "102flowers.tgz",
    "imagelabels.mat": "imagelabels.mat",
    "setid.mat": "setid.mat",
}


def download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  [跳过] 已存在: {dest.name}")
        return
    print(f"  [下载] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r     {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Oxford 102 Flowers 数据集")
    parser.add_argument("--out", default="data/raw/flowers102")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载, 只做整理")
    args = parser.parse_args()

    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("[1/3] 下载文件 (images ~648MB)")
        for name, local in FILES.items():
            download(BASE + name, raw / local)
    else:
        print("[1/3] 跳过下载")

    # 解压图片
    print("[2/3] 解压图片")
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tgz = raw / "102flowers.tgz"
    if not images_dir.exists() or not any(images_dir.iterdir()):
        if not tgz.is_file():
            print("[错误] 缺少 102flowers.tgz, 请先运行 --skip-download 之外的下载")
            sys.exit(1)
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(out / "_tmp_images")
        # tgz 内含 jpg_1.jpg ... jpg_8189.jpg
        for p in (out / "_tmp_images").glob("*.jpg"):
            p.rename(images_dir / p.name)
        (out / "_tmp_images").rmdir()

    # 读标签与划分
    print("[3/3] 生成 labels.csv 与 splits.json")
    try:
        from scipy.io import loadmat
    except ImportError:
        print("[错误] 需要 scipy: pip install scipy")
        sys.exit(1)

    labels = loadmat(raw / "imagelabels.mat")["labels"][0]          # (8189,)
    setid = loadmat(raw / "setid.mat")
    trnid = setid["trnid"][0].tolist()
    valid = setid["valid"][0].tolist()
    tstid = setid["tstid"][0].tolist()

    rows = []
    splits = {"train": [], "val": [], "test": []}
    for idx, label in enumerate(labels, start=1):
        name = f"image_{idx:05d}.jpg"
        src = images_dir / f"jpg_{idx}.jpg"
        dst = images_dir / name
        if src.exists() and not dst.exists():
            src.rename(dst)
        cls = int(label)
        rows.append({"image": f"images/{name}", "species_en": f"flower_{cls:03d}", "species_zh": ""})
        if idx in trnid:
            splits["train"].append(name)
        elif idx in valid:
            splits["val"].append(name)
        elif idx in tstid:
            splits["test"].append(name)

    # 按类别分子目录 (硬链接, 便于人工核查, 不占额外空间)
    cls_dir = out / "by_class"
    linked = 0
    for r in rows:
        species_dir = cls_dir / r["species_en"]
        species_dir.mkdir(parents=True, exist_ok=True)
        p = out / r["image"]
        dst = species_dir / p.name
        if p.exists() and not dst.exists():
            try:
                dst.hardlink_to(p)
                linked += 1
            except OSError:
                pass  # 跨盘/文件系统不支持硬链接时跳过, 不影响主流程
    print(f"  by_class 目录已整理 ({linked} 个硬链接)")

    (out / "labels.csv").write_text(
        "image,species_en,species_zh,features,habitat\n"
        + "\n".join(f"{r['image']},{r['species_en']},{r['species_zh']}," for r in rows),
        encoding="utf-8",
    )
    (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: {len(rows)} 张图片 -> {out}")
    print("提示: species_zh 列留空, 请用代码/人工补充中文名后复制为 data/raw/labels.csv 使用")


if __name__ == "__main__":
    main()
