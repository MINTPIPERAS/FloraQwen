# FloraQwen

基于千问（Qwen3-VL-4B）的花草识别智能体，可在 **8GB 显存笔记本**（如 RTX 4060 Laptop）上完成 QLoRA 微调与本地推理。本项目已端到端跑通：**数据下载 → 训练集构建 → QLoRA 微调 → 推理**，以下命令均为实测记录。

## 当前进度

| 里程碑 | 状态 |
|---|---|
| M1 环境搭建 + 基线推理 | ✅ 2026-09-17 打通 |
| M2 数据管道（Oxford 102 Flowers, 8189 张） | ✅ 2026-09-17 落盘 |
| M3 QLoRA 微调（子集 915 条 ✅ / 全量 24466 条 ✅） | ✅ 2026-09-17 / 09-20 |
| M4 评估（Top-1 准确率） | 🚧 评估集已就绪（2040 张 / 102 类） |
| M5 智能体交互层 + 工具调用 | ⏳ `src/floraqwen/agent.py` 骨架已写 |
| M6 部署（GGUF/Ollama） | ⏳ |

## 环境要求

| 项 | 要求 | 本机实测 |
|---|---|---|
| GPU | 8GB VRAM（4-bit QLoRA 路线） | RTX 4060 Laptop, 训练显存峰值 ~7.9GB |
| 内存 | 16GB+（训练时关浏览器等大户） | 15.6GB 可跑 |
| 系统 | Windows 10/11 | 已实测 |
| Python | conda 环境 `floraqwen`（Python 3.11 + torch cu126） | 见 `scripts/setup_env.ps1` |

模型与缓存默认放 **F 盘**（`HF_HOME=F:\hf-cache\huggingface`），防止写满 C 盘；基座模型 Qwen3-VL-4B-Instruct (8.9GB) 已通过 ModelScope 下载并硬链接接入 HF 缓存（见 `docs/启动指南.md`）。**如何把下载/读取位置改成别的盘，见下节。**

## 模型缓存位置：默认行为与自定义

**默认行为**：下载与读取走的是**同一个目录**——`HF_HOME` 指向的缓存（本项目默认 `F:\hf-cache\huggingface`，模型实体在 `HF_HOME\hub\` 下）。首次 `from_pretrained` 下载到那里，之后每次加载也从那里读；`HF_HUB_OFFLINE=1` 时纯离线读取、绝不联网。

> ⚠️ `set_env.ps1` 默认 `HF_HUB_OFFLINE=1`。**新机器第一次下载模型前**请勿开启离线模式（不运行该脚本，或设 `$env:HF_HUB_OFFLINE = "0"`），下载完成后再改回 `1`。

### 新人自定义缓存位置（不改项目代码，三选一）

**方法一：环境变量覆盖（推荐）**
不要运行 `set_env.ps1`、不要用 `run.ps1` 启动（两者会把 `HF_HOME` 覆盖回 F 盘），改为每个终端手动设置：

```powershell
$env:HF_HOME = "D:\my-hf-cache"              # 下载 + 读取都在这个目录
# 可选: 只单独指定模型缓存目录 (优先级高于 HF_HOME\hub)
$env:HF_HUB_CACHE = "D:\my-hf-cache\hub"

# 首次下载需要联网 + 国内镜像:
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"                # 不加会 401

# 之后所有命令直接用 python 运行 (不经过 run.ps1):
python scripts\train_lora_trl.py --config configs\lora_config.yaml
python scripts\infer.py --image xxx.jpg
```

`infer.py` 的 F 盘兜底**只在 `HF_HOME` 未设置时触发**——只要你显式设置了 `HF_HOME`，它就完全尊重你的设置。缓存下载完整后建议补上 `$env:HF_HUB_OFFLINE = "1"`，加载零联网、更快更稳。

**方法二：永久生效**

```powershell
setx HF_HOME "D:\my-hf-cache"     # 对之后新开的终端生效 (set_env.ps1/run.ps1 运行时仍会覆盖当前会话)
```

**方法三：完全绕开缓存，直接用本地模型目录**（读取位置自己说了算，适合模型放在移动硬盘/共享盘的场景）

```powershell
# 用 ModelScope 把模型下载到任意目录 (国内直连, 多线程断点续传)
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir D:\models\Qwen3-VL-4B-Instruct

# 训练与推理都支持 --model 传本地路径, 不查任何缓存:
python scripts\train_lora_trl.py --config configs\lora_config.yaml --model D:\models\Qwen3-VL-4B-Instruct
python scripts\infer.py --image xxx.jpg --model D:\models\Qwen3-VL-4B-Instruct
```

> 补充：什么都不设置时，HuggingFace 的默认缓存在 `C:\Users\<用户名>\.cache\huggingface`（需要 C 盘有 ~10GB 空闲）。

## 快速开始（5 步）

> 推荐用一键启动器 `.\scripts\run.ps1 <脚本>`：自动切到项目根目录、设好 HF 环境变量、用 `floraqwen` 的 python 运行，无需激活环境。
> 手动方式：每个新终端先 `. .\scripts\set_env.ps1`，再 `conda activate floraqwen`。

### ① 环境检查（~1 分钟）

```powershell
.\scripts\run.ps1 scripts\check_env.py
```

torch/CUDA 为 [OK] 即环境可用。

### ② 模型加载冒烟测试（~40 秒，必须通过）

```powershell
.\scripts\run.ps1 logs\smoke_load_model.py
```

期望输出 `SMOKE TEST PASSED`。**若在此崩溃 = Windows 加载补丁丢失**（见下文"已知坑"第 1 条），后续训练/推理都会段错误。

### ③ 数据准备（首次约 80 分钟，主要是下载）

```powershell
# 3.1 下载并整理 Oxford 102 Flowers (8189 张 + 官方 train/val/test 划分)
python scripts\download_flowers102.py --out data\raw\flowers102

# 3.2 生成训练标签 (102 类中文名/科属/特征映射; 默认官方 train 划分 1020 行)
python scripts\make_flowers102_labels.py
#    需要全量 8189 张时: python scripts\make_flowers102_labels.py --all --out data\raw\labels.all.csv

# 3.3 生成 ShareGPT 格式训练集
python scripts\prepare_data.py --labels data\raw\labels.csv --images-dir data\raw --out data\processed\train.json --max-per-class 3   # 子集 ~915 条
#    全量: 把 --labels 换成 labels.all.csv、--out 换成 data\processed\train.full.json、去掉 --max-per-class (~24466 条)

# 3.4 (可选) 构建评估集 (官方 test 划分, 硬链接零额外磁盘)
python scripts\build_test_set.py --max-per-class 20
```

### ④ QLoRA 训练

```powershell
# 子集试跑 (915 条, 3 epoch, 实测 ~1.5 小时) —— 先跑这个验证全链路
python scripts\train_lora_trl.py --config configs\lora_config.yaml

# 全量训练 (24466 条, 1 epoch, 实测 ~10-14 小时)
python scripts\train_lora_trl.py --config configs\lora_config_full.yaml
```

- 训练细节（参数说明、时长预估、断点续训思路）见 `docs/2026-09-17-训练操作指南.md`；
- 每 100 步自动存 checkpoint；产出 LoRA adapter（~132MB）到 `models/` 下对应目录；
- 训练健康判据：单步 ~20-32s、loss 从 4.5 持续下降、GPU 利用率 85-99%（`nvidia-smi`；任务管理器默认只显示 3D 引擎，会误显 0%）。

### ⑤ 推理与评估

```powershell
# 单图识别 (基座): 
python scripts\infer.py --image data\raw\images\rose_001.jpg

# 加载微调 adapter:
python scripts\infer.py --image "data\test\向日葵\image_05399.jpg" --adapter models\lora-qwen3vl-4b-flora-full

# 批量评估 Top-1 准确率 (每类 N 张, N=1 约 25 分钟):
python scripts\evaluate.py --test-dir data\test --adapter models\lora-qwen3vl-4b-flora-full --limit 1
```

## Windows 已知坑（重要）

1. **transformers 5.5.0 加载段错误（已修）**：Windows 下 safetensors 的 mmap 后端会耗尽提交内存导致 0xC0000005 崩溃。本机在 `site-packages/transformers/modeling_utils.py` 打了 pread 后端站点补丁（上游 [PR #48341](https://github.com/huggingface/transformers/pull/48341) 的最小落地）。**升级/重装 transformers 会丢失补丁**——用第②步冒烟测试检验；升级到 ≥5.16 版本可彻底替代补丁。原理与诊断过程见 `docs/2026-09-17-数据集与训练工作记录.md`。
2. **HF 缓存环境变量**：`set_env.ps1` / `run.ps1` 会自动设 `HF_HOME` 指向 F 盘并开启离线模式；手动跑脚本前务必先设，否则会去 C 盘重新下载 8.9GB 模型。
3. **Flowers102 标签编号**：`imagelabels.mat` 的 1-102 编号对应官方原始类目顺序，**不是** VGG 官网 `categories.html` 的字母序（直接照抄会 102 类全错位）。`make_flowers102_labels.py` 已内置正确映射，勿用官网页面自行对照。
4. **内存偏紧**：全量训练曾因一次性把 2.4 万张图解码进内存而 OOM（`train_lora_trl.py` 已改为按 batch 惰性解码）；运行大任务前关闭浏览器等内存大户。
5. **控制台编码**：含中文/emoji 的输出建议 UTF-8 终端；含中文的 `.ps1` 需保存为 UTF-8 带 BOM。

## 项目结构

```
FloraQwen/
├── configs/
│   ├── lora_config.yaml          # 子集训练配置 (3 epoch)
│   └── lora_config_full.yaml     # 全量训练配置 (24466 条, 1 epoch)
├── data/
│   ├── raw/                      # 图片 + labels.csv (脚本生成; 示例模板 labels.example.csv)
│   │   └── flowers102/           # Oxford 102 Flowers (下载生成, 不入库)
│   ├── processed/                # train.json (ShareGPT 训练集, 脚本生成)
│   └── test/                     # 评估集 <中文名>/*.jpg (脚本生成, 不入库)
├── docs/
│   ├── 启动指南.md                       # conda 初始化、HF 缓存说明
│   ├── FloraQwen-花草识别智能体技术方案.md  # 总体技术方案
│   ├── 2026-09-17-数据集与训练工作记录.md   # 数据拉取/段错误诊断/训练全记录 (原理+踩坑)
│   └── 2026-09-17-训练操作指南.md          # 训练命令/参数/时长/坑清单
├── logs/                         # 诊断与验证脚本 (smoke_load_model / collator_test 等) + 运行日志(不入库)
├── models/                       # LoRA adapter 产物 (不入库)
├── scripts/
│   ├── setup_env.ps1             # Windows 环境搭建 (conda + CUDA torch)
│   ├── set_env.ps1 / run.ps1     # 环境变量设置 / 一键启动器
│   ├── check_env.py              # 环境检查
│   ├── download_flowers102.py    # Oxford 102 Flowers 一键下载整理
│   ├── make_flowers102_labels.py # 102 类英文→中文/科属/特征映射 → labels.csv
│   ├── prepare_data.py           # labels.csv → ShareGPT train.json
│   ├── build_test_set.py         # 官方 test 划分 → data/test/<物种>/ 评估集
│   ├── train_lora.py             # QLoRA 微调 (Unsloth 后端, 本机未装 unsloth 不可用)
│   ├── train_lora_trl.py         # QLoRA 微调 (transformers+peft, Windows 实测可用) ★
│   ├── evaluate.py               # 按物种目录算 Top-1 准确率
│   └── infer.py                  # 推理 (基座/adapter, 4-bit)
├── src/floraqwen/                # 智能体核心包 (agent.py: 结构化识别 + 置信度兜底)
└── notebooks/                    # 探索实验
```

## 常用命令速查

| 用途 | 命令 |
|---|---|
| 环境检查 | `.\scripts\run.ps1 scripts\check_env.py` |
| 模型加载冒烟测试 | `.\scripts\run.ps1 logs\smoke_load_model.py` |
| 下载数据集 | `python scripts\download_flowers102.py --out data\raw\flowers102` |
| 生成训练集 | 见"快速开始 ③" |
| 子集训练 (~1.5h) | `python scripts\train_lora_trl.py --config configs\lora_config.yaml` |
| 全量训练 (~10-14h) | `python scripts\train_lora_trl.py --config configs\lora_config_full.yaml` |
| 单图识别 | `python scripts\infer.py --image <图片> --adapter <adapter目录>` |
| 评估准确率 | `python scripts\evaluate.py --test-dir data\test --adapter <adapter目录> --limit 1` |
| GPU 状态 | `nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 5` |
