#!/usr/bin/env python
"""FloraQwen 推理脚本 (transformers 4-bit, 兼容 Qwen3-VL / Qwen2.5-VL)。

用法:
    # 基座模型直接推理 (首次运行自动下载 ~9GB)
    python scripts/infer.py --image data/raw/images/rose_001.jpg

    # 加载微调后的 LoRA adapter
    python scripts/infer.py --image xxx.jpg --adapter models/lora-qwen3vl-4b-flora

    # 指定模型 / 量化 / 自定义提问
    python scripts/infer.py --image xxx.jpg --model Qwen/Qwen2.5-VL-3B-Instruct \
        --quant 8bit --prompt "请只回答中文名和学名" --max-tokens 256

8GB 显存建议: --quant 4bit (默认), 或 GGUF/Ollama 部署见 docs 方案。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 安全网: 若未手动设置环境变量 (如忘记 . .\scripts\set_env.ps1),
# 自动指向 F 盘已下载好的 HF 缓存并启用离线模式, 避免去 C 盘重复下载 8.9GB。
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = r"F:\hf-cache\huggingface"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("[提示] 未检测到 HF_HOME, 已自动使用 F 盘模型缓存 (离线模式)")

# Windows 控制台默认 GBK, 模型输出可能含 emoji/中文符号 -> 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FloraQwen 花草识别推理")
    p.add_argument("--image", required=True, help="图片路径 (支持单张或目录, 目录时逐张识别)")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                   help="基座模型 (默认 Qwen3-VL-4B; 可换 Qwen/Qwen2.5-VL-3B-Instruct)")
    p.add_argument("--adapter", default=None, help="LoRA adapter 目录 (微调后使用)")
    p.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit", help="量化方式")
    p.add_argument("--prompt", default="这张照片里是什么植物？请给出中文名、学名和科属，并说明识别依据。",
                   help="识别提问")
    p.add_argument("--max-tokens", type=int, default=512)
    return p


def load_model(model_name: str, quant: str, adapter: str | None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[加载] {model_name} (quant={quant}, adapter={adapter or '无'})")
    # 注意: 不要传 dtype=torch.bfloat16! transformers 5.5.0 + 4bit + device_map=auto
    # 在 Windows 上会触发 torch_cpu.dll 原生崩溃 (0xC0000005, 加载约30%处)。
    # 不传 dtype 由 transformers 自行决定 (4bit 走 fp16 路径, 已实测正常)。
    kwargs = dict(device_map="auto", trust_remote_code=False)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def recognize(model, processor, image_path: Path, prompt: str, max_tokens: int) -> str:
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    # 去掉输入部分
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[错误] 缺少 torch, 请先运行 scripts/setup_env.ps1")
        sys.exit(1)

    try:
        model, processor = load_model(args.model, args.quant, args.adapter)
    except ImportError as e:
        print(f"[错误] 依赖不完整: {e}")
        print("提示: Qwen3-VL 需要 transformers>=4.57, 请执行: pip install -U transformers")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        print("可能原因: 磁盘空间不足 / 网络无法访问 HuggingFace (可设置 HF_ENDPOINT=https://hf-mirror.com)")
        sys.exit(1)

    img_path = Path(args.image)
    if img_path.is_dir():
        imgs = sorted(img_path.glob("*"))
        imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
        print(f"[目录] 发现 {len(imgs)} 张图片")
    elif img_path.is_file():
        imgs = [img_path]
    else:
        print(f"[错误] 图片不存在: {img_path}")
        sys.exit(1)

    for i, p in enumerate(imgs, 1):
        t0 = time.time()
        try:
            result = recognize(model, processor, p, args.prompt, args.max_tokens)
            dt = time.time() - t0
            print(f"\n=== [{i}/{len(imgs)}] {p.name} ({dt:.1f}s) ===")
            print(result)
        except Exception as e:
            print(f"\n=== [{i}/{len(imgs)}] {p.name} 识别失败: {e} ===")


if __name__ == "__main__":
    main()
