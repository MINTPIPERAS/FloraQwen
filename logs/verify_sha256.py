# -*- coding: utf-8 -*-
"""临时: 校验文件 sha256 是否等于期望 blob 名 (验证后删除)
用法: python verify_sha256.py <file> <expect_sha256>
"""
import hashlib
import sys


def main():
    path, expect = sys.argv[1], sys.argv[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1 << 22)
            if not c:
                break
            h.update(c)
    ok = h.hexdigest() == expect
    print(f"sha256={h.hexdigest()[:16]}... {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
