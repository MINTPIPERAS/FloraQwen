# FloraQwen

基于千问（Qwen）基座模型的花草识别智能体，可在 **8GB 显存笔记本**（如 RTX 4060 Laptop）上完成 QLoRA 微调与本地推理。

## 技术方案

完整方案见 [`docs/FloraQwen-花草识别智能体技术方案.md`](docs/FloraQwen-花草识别智能体技术方案.md)。

- **基座**：Qwen/Qwen3-VL-4B-Instruct（首选，经 hf-mirror 下载）/ Qwen2.5-VL-3B-Instruct（备选）
- **训练**：QLoRA（4-bit NF4 + 梯度累积 + 梯度检查点），8GB 可行
- **部署**：transformers 4-bit 推理 / GGUF + Ollama

## 环境状态（本机已就绪）

| 项 | 值 |
|---|---|
| 系统 | Windows + conda 环境 `floraqwen`（Python 3.11.15） |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8.0 GB) |
| torch | 2.11.0+cu126（CUDA 12.6, `cuda.is_available()=True`） |
| 依赖 | transformers 5.5.0 / bitsandbytes / peft / accelerate / datasets 5.0.1 等 |
| 模型 | Qwen/Qwen3-VL-4B-Instruct（**ModelScope 下载，8.9GB，2026-08-18 完成**，F 盘） |

**国内网络要点**（已实测）：
- PyPI 走清华镜像：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`
- **模型下载首选 ModelScope（魔搭）**——国内直连、多线程 + 断点续传，8.9GB 约 18 分钟下完：
  ```powershell
  pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
  modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir F:\hf-cache\models\Qwen3-VL-4B-Instruct
  python logs\wire_hf_cache.py   # 校验 sha256 并以硬链接接入 HF 缓存 (F:\hf-cache\huggingface), 零额外磁盘
  ```
- hf-mirror 兜底：`$env:HF_ENDPOINT="https://hf-mirror.com"` + `$env:HF_HUB_DISABLE_XET="1"`（必须禁用 Xet，否则 401）。注意：hf-mirror 大文件走 AWS CDN 签名 URL，速度慢且易中途失效；**曾因 `logs/download_single.ps1` 中 sha256↔文件名映射写反导致"下载正确也校验失败"的死循环（2026-08-18 已修正）**
- Windows 上 PyTorch 的 CUDA wheel 只能从官方源获取（各镜像均 302 回官方）；网络受限时用分片并发 + 代理下载（见 logs/torch_dl/download_parts.ps1 的经验）
- 含中文的 `.ps1` 脚本需保存为 **UTF-8 带 BOM**，否则本机 PowerShell 按 GBK 解析会吞字符串引号

## 快速开始

> 💡 完整操作说明见 [`docs/启动指南.md`](docs/启动指南.md)（conda 初始化、`(base)` 说明、FAQ）。
> 💡 **推荐用一键启动器**（自动设环境变量 + 自动用 floraqwen 的 python，不依赖激活与当前目录）：

```powershell
cd F:\GithubDeskClone\FloraQwen
.\scripts\run.ps1 scripts\check_env.py
.\scripts\run.ps1 scripts\infer.py --image data\raw\images\rose_001.jpg
```

传统方式（手动激活，每个新终端需先设环境变量，否则会去 C 盘重新下载模型）：

```powershell
# 0. (重要) 设置环境变量: 模型缓存放 F 盘, 防止写满 C 盘
#    每个新终端运行一次: . .\scripts\set_env.ps1
#    或手动: $env:HF_HOME="F:\hf-cache\huggingface"; $env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"

# 1. 搭建环境 (若本机尚未执行)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. 检查环境
conda activate floraqwen
python scripts\check_env.py

# 3. 准备数据: 图片放入 data\raw\images\, 标签写入 data\raw\labels.csv
#    (参考 data\raw\labels.example.csv, 当前含示例占位数据)
python scripts\prepare_data.py

# 4. 基线推理 (模型已下载到 F:\hf-cache\huggingface)
python scripts\infer.py --image data\raw\images\rose_001.jpg

# 5. QLoRA 微调 (需真实数据, 示例数据仅用于流程验证)
python scripts\train_lora.py --config configs\lora_config.yaml
```

## 项目结构

```
FloraQwen/
├── docs/                   # 技术方案
├── data/
│   ├── raw/                # 原始图片 + labels.csv (示例模板见 labels.example.csv)
│   └── processed/          # train.json (ShareGPT 格式训练集)
├── configs/lora_config.yaml
├── scripts/
│   ├── setup_env.ps1              # Windows 环境搭建 (conda + CUDA)
│   ├── set_env.ps1                # 每终端运行: HF_HOME/HF_ENDPOINT 等环境变量
│   ├── check_env.py               # 环境检查
│   ├── prepare_data.py            # 图片+标签 → ShareGPT JSON
│   ├── build_labels_from_folders.py  # 按物种文件夹批量生成 labels.csv (iNaturalist/自拍)
│   ├── download_flowers102.py     # Oxford 102 Flowers 一键下载整理 (国际基线)
│   ├── train_lora.py              # QLoRA 微调 (Unsloth 后端)
│   ├── train_lora_trl.py          # QLoRA 微调 (纯 transformers+peft 备选, Windows 可用)
│   ├── evaluate.py                # 识别评估 (按物种目录算准确率)
│   └── infer.py                   # 推理 (基线/微调后, 4-bit)
├── src/floraqwen/          # 智能体核心包 (agent.py: 结构化识别 + 置信度兜底)
├── models/                 # 微调产物
└── notebooks/              # 探索实验
```

## 里程碑进度

| 阶段 | 状态 |
|---|---|
| M1 环境搭建 + 基线推理 | 🚧 环境已就绪；**模型已下载到 F 盘**（ModelScope, 2026-08-18），待跑通 `infer.py` 基线推理 |
| M2 数据收集与清洗 | 🚧 管道已就绪（`prepare_data.py` 实测通过），待真实数据 |
| M3 QLoRA 微调 v1 | ⏳ |
| M4 评估迭代 | ⏳ |
| M5 智能体交互层 + 工具调用 | ⏳ |
| M6 部署 (GGUF/Ollama) | ⏳ |
