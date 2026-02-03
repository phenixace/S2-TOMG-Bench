#!/usr/bin/env bash
# Unified inference launcher: choose route via BACKEND.
# BACKEND = hf | hf_multigpu | api | vllm | openai | ollama | t5
#   hf          = local Hugging Face CausalLM (query_hf.py), optional LoRA
#   hf_multigpu = same script via accelerate launch (query_hf.py, WORLD_SIZE>1)
#   api         = any OpenAI-compatible API via query_api.py (--backend openai|ollama|vllm)
#   vllm        = vLLM server (query_api.py --backend vllm)
#   openai      = OpenAI API (query_api.py --backend openai)
#   ollama      = Ollama API (query_api.py --backend ollama)
#   t5          = T5 / decoder-only via query_hf.py (--model_type t5|decoder-only; use EXTRA=--selfies for BioT5)
# Edit variables then run: BACKEND=hf bash run_query.bash

BACKEND="${BACKEND:-hf}"
NAME="${NAME:-llama3-8B}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
# Benchmark scale: full = S²-Bench (phenixace/S2-TOMG-Bench), mini = S²-Bench-mini (phenixace/S2-TOMG-Bench-mini)
BENCHMARK_SCALE="${BENCHMARK_SCALE:-full}"
PORT="${PORT:-8002}"
OUTPUT_DIR="${OUTPUT_DIR:-./predictions/}"

# For BACKEND=hf: optional LoRA
USE_LORA="${USE_LORA:-0}"
LORA_PATH="${LORA_PATH:-./ckpt/llama3.1-8B-light/checkpoint-1410}"
MAIN_PORT="${MAIN_PORT:-29503}"

# For BACKEND=t5
MODEL_TYPE="${MODEL_TYPE:-decoder-only}"
BASE_MODEL="${BASE_MODEL:-facebook/galactica-125m}"
ADAPTER_PATH="${ADAPTER_PATH:-./ckp/$NAME/checkpoint-375000/}"
GPU="${GPU:-0}"
EXTRA="${EXTRA:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

run_one() {
  local task=$1
  local subtask=$2
  case "$BACKEND" in
    hf)
      if [ "$USE_LORA" = "1" ]; then
        accelerate launch --main_process_port "$MAIN_PORT" query_hf.py --task "$task" --subtask "$subtask" --benchmark_scale "$BENCHMARK_SCALE" --json_check --name "$NAME" --model "$MODEL" --load_lora True --lora_model_path "$LORA_PATH" --output_dir "$OUTPUT_DIR"
      else
        python query_hf.py --task "$task" --subtask "$subtask" --benchmark_scale "$BENCHMARK_SCALE" --json_check --name "$NAME" --model "$MODEL" --output_dir "$OUTPUT_DIR"
      fi
      ;;
    hf_multigpu)
        accelerate launch --main_process_port "$MAIN_PORT" query_hf.py --task "$task" --subtask "$subtask" --benchmark_scale "$BENCHMARK_SCALE" --json_check --name "$NAME" --model "$MODEL" --output_dir "$OUTPUT_DIR"
      ;;
    api|vllm|openai|ollama)
      python query_api.py --backend "$BACKEND" --task "$task" --subtask "$subtask" --json_check --port "$PORT" --name "$NAME" --model "$MODEL" --output_dir "$OUTPUT_DIR"
      ;;
    t5)
      python query_hf.py --model_type "$MODEL_TYPE" --benchmark_scale "$BENCHMARK_SCALE" --name "$NAME" --base_model "$BASE_MODEL" --adapter_path "$ADAPTER_PATH" --task "$task" --subtask "$subtask" --output_dir "$OUTPUT_DIR" $EXTRA
      ;;
    *)
      echo "Unknown BACKEND=$BACKEND. Use: hf | hf_multigpu | api | vllm | openai | ollama | t5"
      exit 1
      ;;
  esac
}

# MolCustom
run_one MolCustom AtomNum
run_one MolCustom BondNum
run_one MolCustom FunctionalGroup
# MolEdit
run_one MolEdit AddComponent
run_one MolEdit DelComponent
run_one MolEdit SubComponent
# MolOpt
run_one MolOpt LogP
run_one MolOpt MR
run_one MolOpt QED
