# S²-Bench
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2412.14542-B31B1B.svg)](https://arxiv.org/abs/2412.14642)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench)

## Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation

**[English (README.md)](README.md)**

作者：Jiatong Li*, Junxian Li*, Weida Wang, Yunqing Liu, Changmeng Zheng, Xiaoyong Wei, Dongzhan Zhou, and Qing Li (* Equal Contribution)

* 论文：[https://arxiv.org/abs/2412.14642](https://arxiv.org/abs/2412.14642)
* Hugging Face 数据集：[phenixace/S2-TOMG-Bench](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench)（完整版 45k）· [phenixace/S2-TOMG-Bench-mini](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench-mini)（mini 版 4.5k）
* 项目主页：[https://phenixace.github.io/tomgbench/](https://phenixace.github.io/tomgbench/)

## 简介

近年来，大语言模型（LLM）在自然语言驱动的分子发现中展现出巨大潜力。然而，现有分子-文本对齐的数据集与基准多基于一对一映射，衡量的是模型检索单一预设答案的能力，而非其生成多样、同样有效的分子候选的创造能力。为填补这一空白，我们提出 **说即结构（S²-Bench，Speak-to-Structure）**，首个面向开放域自然语言驱动分子生成的评测基准。S²-Bench 针对一对多关系设计，要求模型体现真正的分子理解与开放式生成能力。基准包含三大任务：分子编辑（**MolEdit**）、分子优化（**MolOpt**）、定制分子生成（**MolCustom**），分别考察分子发现的不同维度。我们同时提出 **OpenMolIns**，一个大规模指令微调数据集，使 Llama-3.1-8B 在 S²-Bench 上超越 GPT-4o、Claude-3.5 等强模型。我们对 30 个 LLM 的全面评测将焦点从简单模式复现转向真实分子设计，为自然语言驱动分子发现中更强模型铺平道路。

### 当前排行榜

| 排名 | 模型 | 参数量 (B) | $\overline{S\!R}$ (%) | $\overline{W\!S\!R}$ (%) |
|------|------|------------|------------------------|--------------------------|
| 1 | Llama3.1-8B (OpenMolIns-xlarge) | 8 | 58.79 | 39.33 |
| 2 | Claude-3.5 | - | 51.10 | 35.92 |
| 3 | Gemini-1.5-pro | - | 52.25 | 34.80 |
| 4 | GPT-4-turbo | - | 50.74 | 34.23 |
| 5 | GPT-4o | - | 49.08 | 32.29 |
| 6 | Claude-3 | - | 46.14 | 30.47 |
| 7 | Llama3.1-8B (OpenMolIns-large) | 8 | 43.1 | 27.22 |
| 8 | Galactica-125M (OpenMolIns-xlarge) | 0.125 | 44.48 | 25.73 |
| 9 | Llama3-70B-Instruct (Int4) | 70 | 38.54 | 23.93 |
| 10 | Galactica-125M (OpenMolIns-large) | 0.125 | 39.28 | 23.42 |
| 11 | Galactica-125M (OpenMolIns-medium) | 0.125 | 34.54 | 19.89 |
| 12 | GPT-3.5-turbo | - | 28.93 | 18.58 |
| 13 | Galactica-125M (OpenMolIns-small) | 0.125 | 24.17 | 15.18 |
| 14 | Gemma3-12B | 12 | 26.28 | 15.00 |
| 15 | Deepseek-R1-distill-Qwen-7B | 7 | 25.07 | 14.61 |
| 16 | Llama3.1-8B-Instruct | 8 | 26.26 | 14.09 |
| 17 | Llama3-8B-Instruct | 8 | 26.40 | 13.75 |
| 18 | chatglm-9B | 9 | 18.50 | 13.13(7) |
| 19 | Galactica-125M (OpenMolIns-light) | 0.125 | 20.95 | 13.13(6) |
| 20 | ChemDFM-v1.5-8B | 8 | 18.24 | 12.07 |
| 21 | ChemLLM-20B | 20 | 16.23 | 9.76 |
| 22 | Llama3.2-1B (OpenMolIns-large) | 1 | 14.11 | 8.10 |
| 23 | yi-1.5-9B | 9 | 14.10 | 7.32 |
| 24 | Mistral-7B-Instruct-v0.2 | 7 | 11.17 | 4.81 |
| 25 | BioT5-base | 0.25 | 24.19 | 4.21 |
| 26 | MolT5-large | 0.78 | 23.11 | 2.89 |
| 27 | Llama3.1-1B-Instruct | 1 | 3.95 | 1.99 |
| 28 | MolT5-base | 0.25 | 11.11 | 1.30(0) |
| 29 | MolT5-small | 0.08 | 11.55 | 1.29(9) |
| 30 | Qwen2-7B-Instruct | 7 | 0.18 | 0.15 |

### 数据集划分

本仓库为 S²-Bench 评测代码，用于在开放域自然语言驱动分子生成任务上评估 LLM。基准包含三大任务、各三个子任务。**完整版**（phenixace/S2-TOMG-Bench）每子任务 5,000 条（共 45k），**mini 版**（phenixace/S2-TOMG-Bench-mini）每子任务 500 条（共 4.5k），便于快速实验。

---

## 使用说明

### 1. 推理（统一入口：`run_query.bash`）

使用 **`run_query.bash`**，通过环境变量 **`BACKEND`** 选择推理方式：

| BACKEND | 说明 | 脚本 |
|---------|------|------|
| `hf` | 本地 Hugging Face：CausalLM（chat），可选 LoRA | [query_hf.py](./query_hf.py) |
| `hf_multigpu` | 多卡：`accelerate launch query_hf.py` | [query_hf.py](./query_hf.py) |
| `api` / `vllm` / `openai` / `ollama` | OpenAI 兼容 API | [query_api.py](./query_api.py) |
| `t5` | T5 / decoder-only（MolT5、BioT5、Galactica）；BioT5 需加 `EXTRA=--selfies` | [query_hf.py](./query_hf.py) |

**常用环境变量：**

- `BACKEND=hf` — 本地 CausalLM
- `NAME` — 模型名称（用于输出子目录）
- `MODEL` — Hugging Face 模型 id 或路径（如 `Qwen/Qwen3-0.6B`）
- `BENCHMARK_SCALE=full` 或 `mini` — 使用完整版或 mini 版基准
- `USE_LORA=1`、`LORA_PATH` — 可选 LoRA（`BACKEND=hf` 时）

**query_hf.py** 的 `--model_type`：`causal`（默认）、`t5`、`decoder-only`。BioT5 使用 SELFIES：`BACKEND=t5 MODEL_TYPE=t5 EXTRA=--selfies NAME=BioT5-base ... bash run_query.bash`。

**示例：**

```bash
# 完整基准，本地 CausalLM
BACKEND=hf NAME=Qwen3-0.6B MODEL=Qwen/Qwen3-0.6B bash run_query.bash

# 仅 mini 基准
BACKEND=hf BENCHMARK_SCALE=mini NAME=Qwen3-0.6B MODEL=Qwen/Qwen3-0.6B bash run_query.bash

# 单任务/子任务并保存 JSON
python query_hf.py --benchmark_scale mini --task MolCustom --subtask AtomNum \
  --name Qwen3-0.6B --model Qwen/Qwen3-0.6B --json_check --save_json --output_dir ./predictions/

# OpenAI 兼容 API
BACKEND=openai NAME=GPT-4o bash run_query.bash

# T5 / decoder-only 适配器
BACKEND=t5 NAME=galactica-125M-xlarge ADAPTER_PATH=./ckp/... bash run_query.bash
```

**输出：** 预测结果写在 `./predictions/<NAME>/open_generation/<Task>/<Subtask>.csv`。加 `--save_json` 会同时生成同路径下的 `.<Subtask>.json`，包含 `model_name`、`task`、`subtask`、`num_samples` 及 `outputs`（每项含 `idx`、`instruction`、`output`）。

### 2. 指令微调

使用 **`run_train.bash`**，通过环境变量指定模型与数据规模，详见 [instruction_tuning.py](./instruction_tuning.py)。

### 3. 评测

使用 [evaluate.py](./evaluate.py) 对 `predictions/` 下结果打分：

```bash
python evaluate.py --name <NAME> --task MolCustom --subtask AtomNum \
  --output_dir ./predictions/ [--predictions path/to/AtomNum.csv]
```

- **`--correct`** — 对原始预测做后处理：从 JSON 或 `=>`/`->` 后提取 SMILES，适用于模型直接输出文本或带 `molecule` 字段的 JSON。
- **BioT5（SELFIES）：** 当 `--name` 包含 `biot5`（如 `BioT5-base`）且使用 `--correct` 时，会将提取出的字符串从 SELFIES 解码为 SMILES 再计算指标，需安装 `selfies`。
- **`--benchmark_scale`** — 若预测是用 mini 基准生成的，评测时设为 `mini`，以与 ground truth 条数一致。

汇总多子任务结果可使用 `predictions/collect.bash`。

---

## 提交模型

若您的模型在基准上表现突出并希望更新排行榜，请通过项目页面的联系方式提交结果（含原始预测文件），我们核实后会更新排行榜。

## 引用

若使用本数据集或代码，请引用：

```bibtex
@article{li2024speak,
  title={Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation},
  author={Li, Jiatong and Li, Junxian and Liu, Yunqing and Zheng, Changmeng and Wei, Xiaoyong and Zhou, Dongzhan and Li, Qing},
  journal={arXiv preprint arXiv:2412.14642v3},
  year={2024}
}
```
