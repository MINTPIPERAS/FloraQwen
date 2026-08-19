#!/usr/bin/env python
"""FloraQwen QLoRA 训练脚本 (纯 transformers + peft 后端)。

用途: unsloth 在 Windows 上无法安装时的备选训练方案。
用法:
    python scripts/train_lora_trl.py --config configs/lora_config.yaml
    python scripts/train_lora_trl.py --config configs/lora_config.yaml --model Qwen/Qwen2.5-VL-3B-Instruct

与 scripts/train_lora.py (Unsloth) 的区别:
    - 使用 transformers Trainer + peft + bitsandbytes, 不依赖 unsloth
    - 相同显存策略: 4-bit NF4 + batch=1 + 梯度累积 + 梯度检查点

数据格式: data/processed/train.json (ShareGPT: conversations + images)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(data_path: str):
    """ShareGPT JSON -> [{image: PIL, messages: [{role, content}]}]"""
    from PIL import Image

    samples = json.loads(Path(data_path).read_text(encoding="utf-8"))
    out = []
    for s in samples:
        img = Image.open(s["images"][0]).convert("RGB")
        messages = [
            {"role": "user", "content": m["value"].replace("<image>\n", "").replace("<image>", "")}
            if m["from"] == "human"
            else {"role": "assistant", "content": m["value"]}
            for m in s["conversations"]
        ]
        out.append({"image": img, "messages": messages})
    return out


class VLDataCollator:
    """把 (image, messages) 编码为模型输入, prompt 部分 mask 掉 (labels=-100)。"""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        images = [f["image"] for f in features]
        texts = [
            self.processor.apply_chat_template(f["messages"], tokenize=False, add_generation_prompt=False)
            for f in features
        ]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # labels: 只对 assistant 回答计算 loss, prompt 部分 = -100
        labels = batch["input_ids"].clone()
        for i, f in enumerate(features):
            prompt_tokens = self.processor.apply_chat_template(
                f["messages"][:-1], tokenize=True, add_generation_prompt=True
            )
            prompt_len = len(prompt_tokens)
            labels[i, :prompt_len] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="FloraQwen QLoRA 训练 (transformers 后端)")
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--model", default=None, help="覆盖基座模型名")
    parser.add_argument("--data", default=None, help="覆盖训练数据路径")
    parser.add_argument("--out", default=None, help="覆盖输出目录")
    parser.add_argument("--freeze-vision", action="store_true", default=True,
                        help="冻结视觉编码器 (默认开启, 省显存且花草识别不需要微调视觉)")
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
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  BitsAndBytesConfig, Trainer, TrainingArguments)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}\n请运行 scripts/setup_env.ps1 安装环境")
        sys.exit(1)

    # ---------- 1. 4-bit 加载基座 ----------
    print(f"[1/5] 加载基座: {cfg['model_name']} (4-bit NF4)")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # 注意: 不传 torch_dtype/dtype! transformers 5.5.0 + 4bit + device_map=auto
    # 在 Windows 上传 torch_dtype=torch.bfloat16 会原生崩溃 (0xC0000005, 同 infer.py 问题)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",  # 无需 flash-attn
    )
    processor = AutoProcessor.from_pretrained(cfg["model_name"])

    # ---------- 2. 冻结视觉编码器 (省显存) ----------
    if args.freeze_vision:
        vision = getattr(getattr(model, "model", model), "visual", None)
        if vision is not None:
            for p in vision.parameters():
                p.requires_grad = False
            print("  已冻结视觉编码器 (model.visual)")
        else:
            print("  [提示] 未找到视觉编码器属性, 跳过冻结")

    # ---------- 3. LoRA ----------
    print("[2/5] 配置 LoRA")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = cfg.get("lora", {})
    lora = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  可训练参数: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({trainable/total*100:.2f}%)")

    # ---------- 4. 数据 ----------
    print("[3/5] 加载训练数据")
    dataset = build_dataset(str(data_path))
    print(f"  共 {len(dataset)} 条样本")
    collator = VLDataCollator(processor)

    # ---------- 5. 训练 ----------
    print("[4/5] 初始化 Trainer (batch=1 + 梯度累积)")
    t = cfg.get("training", {})
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
        num_train_epochs=t.get("num_train_epochs", 3),
        learning_rate=t.get("learning_rate", 1e-4),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        logging_steps=t.get("logging_steps", 5),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 2),
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        seed=t.get("seed", 42),
        dataloader_pin_memory=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("[5/5] 开始训练")
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"LoRA adapter 已保存: {cfg['output_dir']}")
    print("推理测试: python scripts/infer.py --image <图片> --adapter models/lora-...")


if __name__ == "__main__":
    main()
