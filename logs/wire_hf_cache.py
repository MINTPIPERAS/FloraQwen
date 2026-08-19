# -*- coding: utf-8 -*-
"""临时: modelscope 下载完成后 -> 校验 sha256 -> 以硬链接接入 HF 缓存布局 (F 盘)
用法: python wire_hf_cache.py
前置: F:\hf-cache\models\Qwen3-VL-4B-Instruct 已由 modelscope download 完整下载
目标: F:\hf-cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct 变为可离线加载
"""
import hashlib
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(r"F:\hf-cache\models\Qwen3-VL-4B-Instruct")
HF = Path(r"F:\hf-cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct")
COMMIT = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

# 两个大分片: 期望 sha256 (= HF 仓库 lfs_sha256, 已用 HF tree JSON 核实)
# 注意: 旧 download_single.ps1 中此映射曾写反, 已修正
EXPECT = {
    "model-00001-of-00002.safetensors": "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "model-00002-of-00002.safetensors": "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1 << 22)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def main() -> int:
    if not SRC.is_dir():
        print(f"[错误] 源目录不存在: {SRC}")
        return 1

    # 1) 校验两个大分片
    print("== 1) sha256 校验 ==")
    for name, expect in EXPECT.items():
        p = SRC / name
        if not p.is_file() or p.stat().st_size == 0:
            print(f"  [FAIL] {name} 不存在或为空")
            return 1
        got = sha256(p)
        ok = got == expect
        print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}  {p.stat().st_size/1e9:.2f}GB")
        if not ok:
            print(f"    expect {expect}")
            print(f"    got    {got}")
            return 1

    blobs = HF / "blobs"
    snaps = HF / "snapshots" / COMMIT
    blobs.mkdir(parents=True, exist_ok=True)
    snaps.mkdir(parents=True, exist_ok=True)

    # 2) 小文件: 计算 blob 名 -> 接入 (已有则跳过)
    print("== 2) 元数据文件接入 ==")
    for p in sorted(SRC.iterdir()):
        if not p.is_file():
            continue
        if p.name in EXPECT:  # 大分片单独处理
            continue
        h = sha256(p)
        blob = blobs / h
        if not blob.exists():
            try:
                os.link(p, blob)
            except OSError as e:
                print(f"  [WARN] 硬链接失败 {p.name}: {e} (改用复制)")
                import shutil
                shutil.copy2(p, blob)
        link = snaps / p.name
        if not link.exists():
            try:
                link.symlink_to(Path("..") / ".." / "blobs" / h)
            except OSError as e:
                print(f"  [WARN] 符号链接失败 {p.name}: {e}")
        print(f"  [OK] {p.name} -> blob {h[:12]}...")

    # 3) 大分片: 硬链接 blob + 快照符号链接
    print("== 3) 大分片接入 ==")
    for name, expect in EXPECT.items():
        blob = blobs / expect
        if not blob.exists():
            os.link(SRC / name, blob)
        link = snaps / name
        if not link.exists():
            link.symlink_to(Path("..") / ".." / "blobs" / expect)
        print(f"  [OK] {name} -> blob {expect[:12]}...")

    # 4) 汇总
    total = sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
    print(f"== 完成: HF 缓存 blobs 共 {total/1e9:.2f} GB ==")
    print("验证: 设置 HF_HOME=F:\\hf-cache\\huggingface 后可用 from_pretrained('Qwen/Qwen3-VL-4B-Instruct') 离线加载")
    return 0


if __name__ == "__main__":
    sys.exit(main())
