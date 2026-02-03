#!/usr/bin/env bash
# OpenMolIns instruction tuning: instruction_tuning.py
# Edit variables then run: bash run_train.bash

GPU="${GPU:-7}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
NAME="${NAME:-llama3.2-1B}"
DATA_SCALE="${DATA_SCALE:-light}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
# Common flags: --disable_lora --train_on_inputs --add_eos
EXTRA="${EXTRA:-}"

export CUDA_VISIBLE_DEVICES="$GPU"

python instruction_tuning.py \
  --model "$MODEL" \
  --name "$NAME" \
  --data_scale "$DATA_SCALE" \
  --num_epochs "$NUM_EPOCHS" \
  --save_interval "$SAVE_INTERVAL" \
  --warm_up_steps "$WARMUP_STEPS" \
  --disable_lora \
  --train_on_inputs \
  --add_eos \
  $EXTRA

# Other examples (uncomment and edit variables):
# DATA_SCALE=large NAME=galactica-125M MODEL=facebook/galactica-125m bash run_train.bash
# DATA_SCALE=xlarge NUM_EPOCHS=2 SAVE_INTERVAL=10000 bash run_train.bash
