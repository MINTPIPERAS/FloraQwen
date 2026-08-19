# FloraQwen 花草识别智能体技术方案

> 基于 Qwen 基座模型、在 **8GB 显存笔记本**上可训练、可部署的花草识别智能体（VLM Agent）。
> 项目：FloraQwen ｜ 更新日期：见 Git 记录

---

## 1. 项目目标与定位

构建一个**对话式花草识别智能体**：

- 输入：一张花草照片（手机/相机拍摄，可能包含背景杂物）
- 输出：多轮自然语言对话，能够回答——
  - 这是什么花/植物？（中文名、学名、科属）
  - 识别依据（花色、叶形、花序等特征）
  - 置信度/相似物种区分
  - 养护建议、花期、毒性等常识（通过工具检索）

**关键定位**：不是"纯分类器"，而是"能对话、会解释、可调用工具的智能体"，基座为 Qwen 视觉语言模型（VLM），通过轻量微调注入花草领域知识。

---

## 2. 硬件约束分析（8GB 显存）

| 环节 | 显存需求（估算） | 8GB 可行性 |
|---|---|---|
| 推理 3B VLM（4-bit） | ~3–4 GB | ✅ 余量充足 |
| 推理 4B VLM（4-bit） | ~4–5 GB | ✅ 可行 |
| 推理 7B VLM（4-bit） | ~6–8 GB | ⚠️ 勉强，易抖动 |
| QLoRA 微调 3B（4-bit 基座） | ~6–7 GB | ✅ 可行（batch=1） |
| QLoRA 微调 4B（4-bit 基座） | ~7.5–8.5 GB | ⚠️ 紧张，需激进优化 |
| 全参微调任意 ≥3B | 远超 8GB | ❌ 不可行 |

结论：
- **训练**：只能走 **QLoRA（4-bit NF4 + 双重量化）+ 冻结视觉编码器** 的轻量微调路线；
- **推理**：4-bit/GGUF 量化后 3B–4B 模型在 8GB 上非常舒服；
- 显存数字为经验估算，落地时以 `nvidia-smi` 实测为准。

---

## 3. 基座模型选型

### 3.1 候选对比

| 模型 | 参数量 | 4-bit 推理显存 | QLoRA 微调可行性 | 中文/视觉能力 | 生态成熟度 |
|---|---|---|---|---|---|
| **Qwen3-VL-4B-Instruct-2507** | ~4.4B | ~4–5 GB | ⚠️ 紧张可行 | 强，最新基座 | Unsloth 等已支持 |
| Qwen3-VL-2B-Instruct-2507 | ~2.3B | ~2.5–3 GB | ✅ 轻松 | 中上 | 同上 |
| Qwen2.5-VL-3B-Instruct | ~3.75B | ~3–4 GB | ✅ 稳妥 | 好，文档多 | LLaMA-Factory / Swift 支持成熟 |
| Qwen2.5-VL-7B-Instruct | ~8.3B | ~6–8 GB | ❌ 微调基本不可行 | 更强 | — |

### 3.2 推荐

- **首选：`Qwen/Qwen3-VL-4B-Instruct`** —— 能力/显存性价比最高，8GB 可推理、可 QLoRA 微调（hf-mirror / ModelScope 均有托管）；
- **稳妥备选：`Qwen/Qwen2.5-VL-3B-Instruct`** —— 如果 Qwen3-VL 微调工具链遇到兼容性问题（如 LLaMA-Factory 版本滞后），回退到它，训练余量更大；
- **极端兜底：`Qwen3-VL-2B-Instruct-2507`** —— 留给推理更快、显存压力最小的场景。

> 参考：[Qwen3-VL 官方模型](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) ｜ [Unsloth 的 Qwen3-VL 微调指南](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) ｜ [Qwen3-VL-4B vs 8B 显存对比](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/)

### 3.3 国内获取模型的要点（本机实测）

- **镜像**：`HF_ENDPOINT=https://hf-mirror.com`，模型 `Qwen/Qwen3-VL-4B-Instruct` 可用；
- **禁用 Xet**：`HF_HUB_DISABLE_XET=1`（新版 huggingface_hub 的 Xet 下载在国内会 401）；
- **磁盘**：模型约 8.9GB，**务必把 HF 缓存放到非系统盘**（`HF_HOME=F:\hf-cache\huggingface`，见 `scripts/set_env.ps1`），否则 C 盘会被写满；
- **速度**：hf-mirror 大文件下载速度波动较大（0.3~4.5 MB/s），且 AWS CDN 对并发连接限流（多连接会 0 字节），**建议单连接 + 断点续传**（脚本见 `logs/download_single.ps1`，含 sha256 校验；通用分片模板见 `logs/torch_dl/download_parts.ps1`）。

---

## 4. 总体架构

```
用户照片 ──► [图像预处理/压缩] ──► Qwen3-VL 4B（QLoRA 微调，4-bit/GGUF）
                                      │
                          ┌───────────┼──────────────────┐
                          ▼           ▼                  ▼
                    [识别输出]    [多轮对话状态]      [工具调用（可选）]
                  中文名/学名/科属  追问特征、养护       植物数据库/百科检索
                          └───────────┴──────────────────┘
                                      │
                              [结构化结果 + 置信度]
                                      ▼
                            Web / CLI / API 界面
```

智能体三层职责：
1. **感知层**：Qwen3-VL 的视觉编码器理解图像（花、叶、环境）；
2. **知识层**：QLoRA 微调注入的花草领域知识 + 可选工具检索外部数据库；
3. **交互层**：对话管理（追问、澄清、多轮修正），输出 JSON 结构化识别结果。

---

## 5. 技术路线选择

| 路线 | 做法 | 优点 | 缺点 | 推荐度 |
|---|---|---|---|---|
| A | 传统 CNN/CLIP 分类器（ResNet/EfficientNet） | 简单、快、省显存 | 不是"基于 Qwen"，无对话能力 | 仅作基线对比 |
| **B（推荐）** | **Qwen3-VL + QLoRA 微调** | 真正基于 Qwen，保留对话/推理能力，单模型搞定识别+解释 | 8GB 下训练需精打细算 | ⭐⭐⭐⭐⭐ |
| C | B + RAG/工具调用（植物百科检索） | 知识可更新、降低幻觉 | 工程量大一档 | ⭐⭐⭐⭐（二期） |

**本方案落地路线**：一期走 **B**，二期叠加 **C** 的工具调用做成完整智能体。

---

## 6. 数据集方案

### 6.1 公开数据集（起步）

| 数据集 | 内容 | 用途 |
|---|---|---|
| [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | 102 类英伦花卉 | 国际基线、评估集 |
| iNaturalist（花草子集） | 千万级带标签照片 | 大样本扩充 |
| [中国植物图像库 PPBC](http://ppbc.iplant.cn/) | 中文植物照片 | 中文物种覆盖 |
| 中草药/民族药植物图像集 | 药用植物 | 特色能力 |
| 自拍/自采照片 | 手机实拍（含背景） | 抗干扰泛化 |

> 版权注意：爬取 PPBC 等需遵守站点协议，优先用开放许可数据（如 iNaturalist 的 CC 协议子集）。

### 6.2 数据增强（对抗 8GB 小模型）

- 随机裁剪、旋转、亮度/色温扰动（模拟真实拍照）；
- **背景干扰增强**：把花贴到不同背景（草地、手、桌面）上，专治"识别背景"；
- 多对象图：一张图多朵花，训练模型定位主体；
- 合成数据：用 Qwen 生成结构化描述（学名、特征、习性），配图构建图文对。

### 6.3 数据格式（LLaMA-Factory ShareGPT 格式）

```json
[
  {
    "conversations": [
      { "from": "human", "value": "<image>\n这张照片里是什么植物？请给出中文名、学名和科属。" },
      { "from": "gpt", "value": "这是**月季**（Rosa chinensis，蔷薇科）。识别依据：羽状复叶、茎上有皮刺、花为杯状重瓣……" }
    ],
    "images": ["data/train/rose_001.jpg"]
  }
]
```

每类至少 50–200 条对话样本起步（含反问、追问变体），再逐步扩充。

---

## 7. 微调方案（8GB 显存可行版）

### 7.1 优化策略清单（按优先级）

1. **QLoRA**：4-bit NF4 + 双重量化加载基座，只训练 LoRA 适配器（参数量 ~1%）；
2. **冻结视觉编码器（ViT 部分）**，只微调 LLM 层的 LoRA —— 大幅省显存，花草识别对视觉特征微调需求低；
3. **batch_size = 1 + 梯度累积**（gradient_accumulation_steps = 8–16），等效大 batch 但显存不变；
4. **限制最大序列长度**（如 1024–2048，图像 token 会占大头，必要时用图像 token 压缩/降采样）；
5. **AdamW 8-bit 优化器**（bitsandbytes），优化器状态占用再砍一半；
6. 训练中若 OOM：降 `max_seq_len` → 冻结更多层 → 换 3B 模型。

### 7.2 LoRA 超参建议

```yaml
# lora_config.yaml（示例）
lora_rank: 16          # 8~32 均可
lora_alpha: 32
lora_dropout: 0.05
target_modules: all    # 或限定 q_proj,k_proj,v_proj,o_proj
learning_rate: 1e-4
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
optim: paged_adamw_8bit
max_seq_length: 2048
```

### 7.3 工具链选择

| 工具 | 说明 | 备注 |
|---|---|---|
| **Unsloth** | 显存占用最低、速度最快，官方支持 Qwen3-VL 微调 | 首选（[教程](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)） |
| **LLaMA-Factory** | 统一微调框架，WebUI 友好，中文文档全 | 备选；确认版本支持目标模型 |
| **MS-Swift** | 阿里系，对 Qwen 生态贴合好 | 备选 |
| Qwen 官方训练脚本 | 最贴近官方配置 | 参考用 |

---

## 8. 智能体能力设计

### 8.1 识别输出（结构化 JSON）

```json
{
  "中文名": "月季",
  "学名": "Rosa chinensis",
  "科属": "蔷薇科 蔷薇属",
  "置信度": 0.87,
  "识别依据": ["羽状复叶", "茎生皮刺", "重瓣杯状花"],
  "可能混淆物种": ["玫瑰（Rosa rugosa）", "蔷薇"],
  "养护提示": "喜光、耐旱，花期 4–11 月，注意白粉病"
}
```

### 8.2 工具调用（二期）

- `search_plant_db(关键词)`：查植物百科/养护资料；
- `similar_species(学名)`：返回易混淆物种对比；
- `weather_advice(城市)`：结合天气给浇水/遮阳建议。

Qwen 系列原生支持 function calling 格式，微调时可保留少量工具调用样本。

### 8.3 多轮交互流程

1. 首轮：用户传图 → 模型给出结构化识别 + 简短解释；
2. 追问：用户问"怎么区分月季和玫瑰" → 模型基于知识对比特征；
3. 修正：用户说"不对，叶子是针形的" → 模型重新推理，给出备选物种；
4. 兜底：低置信度（<0.5）时主动承认"不确定"，请求换个角度拍照。

---

## 9. 推理与部署

| 方式 | 显存 | 说明 |
|---|---|---|
| **GGUF + llama.cpp / Ollama** | ~3–5 GB | 笔记本首选，Qwen3-VL 官方发布 GGUF（[Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)） |
| **vLLM**（AWQ/GPTQ 量化） | ~5–6 GB | 需要多并发、API 服务时用 |
| Transformers 4-bit 直载 | ~5 GB | 开发调试阶段 |

服务形态：本地 Web（Gradio/Streamlit）→ 拍照上传；进阶可封装 FastAPI 供移动端调用。

---

## 10. 评估方案

- **分类准确率**：Flowers-102 测试集 top-1 / top-5；
- **中文花草集**：自建 20–50 类中文常见花草测试集（含实拍）；
- **对抗样本**：模糊、过曝、多对象、背景干扰图；
- **对话质量**：人工评分（识别正确性、解释合理性、幻觉率——学名/科属是否编造）；
- **资源指标**：单次推理延迟、峰值显存、CPU 内存占用。

---

## 11. 项目目录结构与里程碑

```
FloraQwen/
├── docs/                  # 方案与记录
├── data/
│   ├── raw/               # 原始图片
│   ├── processed/         # 清洗/增强后
│   └── train.json         # ShareGPT 格式训练集
├── configs/               # lora/训练/推理配置
├── scripts/               # 数据清洗、训练、评估脚本
├── src/floraqwen/         # 智能体核心代码
├── models/                # 微调产物（LoRA adapter / GGUF）
├── notebooks/             # 探索实验
└── README.md
```

| 阶段 | 内容 | 预计耗时 |
|---|---|---|
| M1 | 环境搭建（CUDA/PyTorch/Unsloth）+ 基线推理跑通 | 3–5 天 |
| M2 | 数据收集与清洗（公开集 + 自拍 + 增强） | 1–2 周 |
| M3 | QLoRA 微调 v1 + 显存调优 | 1–2 周 |
| M4 | 评估迭代（数据集扩充、超参调优） | 持续 2–4 周 |
| M5 | 智能体交互层 + 工具调用 + Web 界面 | 1–2 周 |
| M6 | 部署（GGUF/Ollama）+ 线上实测迭代 | 持续 |

---

## 12. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| QLoRA 微调 OOM | 训练中断 | 冻结 ViT → 缩短序列 → 换 3B/2B |
| Qwen3-VL 微调工具链不成熟 | 无法训练 | 回退 Qwen2.5-VL-3B（生态最稳） |
| 小模型识别准确率不足 | 效果差 | 数据增强 + 更多标注 + 工具兜底检索 |
| 幻觉（编造学名） | 可信度受损 | 置信度阈值 + 强制 JSON 输出 + 拒绝回答低置信问题 |
| 显存够但内存不足（8GB 独显本常见 16GB 内存） | 加载失败 | 用 GGUF 流式加载、限制图像分辨率 |
| 数据版权 | 合规风险 | 只用开放许可数据，标注来源 |

---

## 13. 参考资料

- [Qwen3-VL 官方模型卡（4B-Instruct-2507）](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-2507)
- [Qwen3-VL GGUF（llama.cpp 直接跑）](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)
- [Unsloth：Qwen3-VL 运行与微调教程](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)
- [Qwen3-VL-4B vs 8B：显存与跑分对比](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/)
- [Datature：在自己的数据集上微调 Qwen3-VL](https://datature.io/blog/how-to-fine-tune-qwen3-vl-on-your-own-dataset)
- [Oxford 102 Flowers 数据集](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [中国植物图像库（PPBC）](http://ppbc.iplant.cn/)
- [LLaMA-Factory 仓库](https://github.com/hiyouga/LLaMA-Factory)

---

*本方案假设：笔记本为 8GB 独显（如 RTX 4060 Laptop / RTX 3050Ti 级别），建议搭配 ≥32GB 内存与 NVMe 硬盘。数字均为估算，落地时以实测为准。*
