# S²-Bench
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2412.14542-B31B1B.svg)](https://arxiv.org/abs/2412.14642)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench)

## Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation

**[中文说明 (README_zh.md)](README_zh.md)**

Authors: Jiatong Li*, Junxian Li*, Weida Wang, Yunqing Liu, Changmeng Zheng, Xiaoyong Wei, Dongzhan Zhou, and Qing Li (* Equal Contribution)

* Arxiv: [https://arxiv.org/abs/2412.14642](https://arxiv.org/abs/2412.14642)
* Hugging Face Datasets: [phenixace/S2-TOMG-Bench](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench) (full, 45k) · [phenixace/S2-TOMG-Bench-mini](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench-mini) (mini, 4.5k)
* Project Page: [https://phenixace.github.io/tomgbench/](https://phenixace.github.io/tomgbench/)

## Introduction

Recently, Large Language Models (LLMs) have demonstrated great potential in natural language-driven molecule discovery. 
However, existing datasets and benchmarks for molecule-text alignment are predominantly built on one-to-one mappings, measuring LLMs' ability to retrieve a single, pre-defined answer, rather than their creative potential to generate diverse, yet equally valid, molecular candidates.
To address this critical gap, we propose \textbf{S}peak-to-\textbf{S}tructure (\textbf{S\textsuperscript{2}-Bench}), 
the first benchmark to evaluate LLMs in open-domain natural language-driven molecule generation.
S\textsuperscript{2}-Bench is specifically designed for one-to-many relationships, challenging LLMs to exhibit genuine molecular understanding and open-ended generation capabilities. 
Our benchmark includes three key tasks: molecule editing (\textbf{MolEdit}), molecule optimization (\textbf{MolOpt}), and customized molecule generation (\textbf{MolCustom}), each probing a different aspect of molecule discovery. 
We also introduce \textbf{OpenMolIns}, a large-scale instruction tuning dataset that enables Llama-3.1-8B to surpass the most powerful LLMs like GPT-4o and Claude-3.5 on S\textsuperscript{2}-Bench. 
Our comprehensive evaluation of 30 LLMs shifts the focus from simple pattern recall to realistic molecular design, paving the way for more capable LLMs in natural language-driven molecule discovery.

### Current Leaderboard

| Rank | Model | #Parameters (B) | $\overline{S\!R}$ (%) | $\overline{W\!S\!R}$ (%) |
|------|-------|-----------------|------------------------|--------------------------|
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


### Dataset Categorization

This repository contains the code for the TOMG-Bench benchmark, which evaluates LLMs on Text-based Open Molecule Generation tasks. The benchmark has three main tasks, each with three subtasks. The **full** (phenixace/S2-TOMG-Bench) dataset has 5,000 samples per subtask (45k total); the **mini** (phenixace/S2-TOMG-Bench-mini) dataset has 500 per subtask (4.5k total) for faster experimentation.


## Usage

### 1. Inference (unified entry via `run_query.bash`)

Use **`run_query.bash`** and set the **`BACKEND`** environment variable to choose the inference backend:

| BACKEND        | Description | Script |
|----------------|-------------|--------|
| `hf`           | Local Hugging Face: CausalLM (chat), optional LoRA | [query_hf.py](./query_hf.py) |
| `hf_multigpu`  | Multi-GPU via `accelerate launch query_hf.py`     | [query_hf.py](./query_hf.py) |
| `api` / `vllm` / `openai` / `ollama` | OpenAI-compatible API | [query_api.py](./query_api.py) |
| `t5`           | T5 / decoder-only (MolT5, BioT5, Galactica); use `EXTRA=--selfies` for BioT5 | [query_hf.py](./query_hf.py) |

**Environment variables (examples):**

- `BACKEND=hf` — local CausalLM
- `NAME` — model name (used for output subdirs)
- `MODEL` — Hugging Face model id or path (e.g. `Qwen/Qwen3-0.6B`)
- `BENCHMARK_SCALE=full` or `mini` — use [phenixace/S2-TOMG-Bench](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench) (S²-Bench full) or [phenixace/S2-TOMG-Bench-mini](https://huggingface.co/datasets/phenixace/S2-TOMG-Bench-mini) (S²-Bench mini)
- `USE_LORA=1`, `LORA_PATH` — optional LoRA for `BACKEND=hf`

**query_hf.py** model types: `causal` (default, chat + optional LoRA), `t5` (MolT5/BioT5), `decoder-only` (e.g. Galactica). For BioT5 use SELFIES: `BACKEND=t5 MODEL_TYPE=t5 EXTRA=--selfies NAME=BioT5-base ... bash run_query.bash`.

**Examples:**

```bash
# Full benchmark, local HF CausalLM
BACKEND=hf NAME=Qwen3-0.6B MODEL=Qwen/Qwen3-0.6B bash run_query.bash

# Mini benchmark only (fewer samples)
BACKEND=hf BENCHMARK_SCALE=mini NAME=Qwen3-0.6B MODEL=Qwen/Qwen3-0.6B bash run_query.bash

# Single task/subtask + save JSON
python query_hf.py --benchmark_scale mini --task MolCustom --subtask AtomNum \
  --name Qwen3-0.6B --model Qwen/Qwen3-0.6B --json_check --save_json --output_dir ./predictions/

# OpenAI-compatible API
BACKEND=openai NAME=GPT-4o bash run_query.bash

# T5 / decoder-only with adapter
BACKEND=t5 NAME=galactica-125M-xlarge ADAPTER_PATH=./ckp/... bash run_query.bash
```

**Output:** Predictions are written under `./predictions/<NAME>/open_generation/<Task>/<Subtask>.csv`. With `--save_json`, a corresponding `.<Subtask>.json` file is also written (same directory), containing `model_name`, `task`, `subtask`, `num_samples`, and `outputs` (list of `{idx, instruction, output}`).

### 2. Instruction tuning

Use **`run_train.bash`** with environment variables for model and data scale; see [instruction_tuning.py](./instruction_tuning.py).

### 3. Evaluation

Use [evaluate.py](./evaluate.py) to score results under `predictions/`:

```bash
python evaluate.py --name <NAME> --task MolCustom --subtask AtomNum \
  --output_dir ./predictions/ [--predictions path/to/AtomNum.csv]
```

- **`--correct`** — Post-process raw predictions: extract SMILES from JSON or from text after `=>`/`->`. Use this when the model outputs raw text or JSON with a `molecule` field.
- **BioT5 (SELFIES):** If `--name` contains `biot5` (e.g. `BioT5-base`) and `--correct` is set, the script decodes each extracted string from SELFIES to SMILES before computing metrics. Ensure the `selfies` package is installed.
- **`--benchmark_scale`** — Set to `mini` when evaluating predictions produced with the mini benchmark so that ground-truth length matches.

Aggregate results across subtasks with `predictions/collect.bash`.

---

## Submit Your Model

If your model achieves strong performance on the benchmark and you want to update the leaderboard, please send your results (including raw prediction files) via the contact given on the project page. We will verify and update the leaderboard accordingly.

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{li2024speak,
  title={Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation},
  author={Li, Jiatong and Li, Junxian and Liu, Yunqing and Zheng, Changmeng and Wei, Xiaoyong and Zhou, Dongzhan and Li, Qing},
  journal={arXiv preprint arXiv:2412.14642v3},
  year={2024}
}
```
