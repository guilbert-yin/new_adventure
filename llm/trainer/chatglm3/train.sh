#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=1
LORA_RANK=8
LORA_ALPHA=32
LORA_DROUPOUT=0.1

MAX_SOURCE_LEN=4096
MAX_TARGET_LEN=400
DEV_BATCH_SIZE=4
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=200
SAVE_INTERVAL=10
MAX_SEQ_LEN=4608

RUN_NAME=text
BASE_MODEL_PATH=/root/autodl-fs/models/chatglm3-6b-base
DATASET_PATH=/root/autodl-tmp/new_adventure/mht_dataset_table_str_prompt_glm3format_train_evi_num_top30.jsonl
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=train_output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROUPOUT \
    --max_seq_length $MAX_SEQ_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR  2>&1 | tee ${OUTPUT_DIR}/train.log