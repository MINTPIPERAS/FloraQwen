#!/usr/bin/env python
"""FloraQwen QLoRA 微调脚本 (Unsloth 后端, 适配 8GB 显存)。

用法:
    python scripts/train_lora.py --config configs/lora_config.yaml
    python scripts/train_lora.py --config configs/lora_config.yaml --model Qwen/Qwen2.5-VL-3B-Instruct

前置:
    1. 运行 scripts/setup_env.ps1 装好环境 (含 unsloth)
    2. python scripts/prepare_data.py 生成 data/processed/train.json
    3. 首次运行会自动下载基座模型 (~9GB), 请保证磁盘空间

显存要点 (8GB):
    - 4-bit NF4 加载基座 + 只训 LoRA (参数量 ~1%)
    - batch=1 + 梯度累积, 梯度检查点 "unsloth" 模式
    - 冻结视觉编码器: Unsloth 默认不训练 ViT 部分, 如需调整见源码
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="FloraQwen QLoRA 训练")
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--model", default=None, help="覆盖基座模型名")
    parser.add_argument("--data", default=None, help="覆盖训练数据路径")
    parser.add_argument("--out", default=None, help="覆盖输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    if args.data:
        cfg["data_path"] = args.data
    if args.out:
        cfg["output_dir"] = args.out

    data_path = Path(cfg["data_path"])
    if not data_path.is_file():
        print(f"[错误] 训练数据不存在: {data_path}")
        print("请先运行: python scripts/prepare_data.py")
        sys.exit(1)

    try:
        from datasets import load_dataset
        from PIL import Image
        from trl import SFTConfig
        from unsloth import (FastVisionModel, UnslothVisionDataCollator,
                             UnslothVisionTrainer, is_bfloat16_supported)
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        print("请运行 scripts/setup_env.ps1 安装环境 (含 unsloth), 或在 WSL2 中安装 unsloth")
        sys.exit(1)

    # ---------- 1. 加载基座 (4-bit) ----------
    print(f"[1/5] 加载基座模型: {cfg['model_name']} (4-bit)")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg["model_name"],
        load_in_4bit=True,
        max_seq_length=cfg["training"]["max_seq_length"],
    )

    # ---------- 2. 挂载 LoRA ----------
    print("[2/5] 配置 LoRA")
    model = FastVisionModel.get_peft_model(model, **cfg["lora"])

    # ---------- 3. 数据转换 (ShareGPT -> Unsloth vision 格式) ----------
    print(f"[3/5] 加载训练数据: {data_path}")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def convert(sample: dict) -> dict:
        img = Image.open(sample["images"][0]).convert("RGB")
        return {"image": img, "messages": sample["conversations"]}

    dataset = dataset.map(convert, remove_columns=["images"])

    # ---------- 4. 训练器 ----------
    print("[4/5] 初始化训练器 (batch=1 + 梯度累积)")
    t = cfg["training"]
    sft_args = dict(
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        output_dir=cfg["output_dir"],
        seed=t["seed"],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = UnslothVisionTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model=model, tokenizer=tokenizer),
        train_dataset=dataset,
        args=SFTConfig(**sft_args),
    )

    # ---------- 5. 训练 + 保存 ----------
    print("[5/5] 开始训练")
    trainer.train()

    adapter_dir = Path(cfg["output_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    print(f"LoRA adapter 已保存: {adapter_dir}")

    if cfg.get("merge_on_save", False):
        merged_dir = f"{cfg['output_dir']}-merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"合并后模型已保存: {merged_dir} (bf16, 体积较大)")

    print("\n完成! 推理测试: python scripts/infer.py --image <图片> --adapter models/lora-qwen3vl-4b-flora")


if __name__ == "__main__":
    main()
