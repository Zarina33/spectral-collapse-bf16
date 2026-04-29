#!/usr/bin/env bash
# run_bf16_seed123.sh — Second seed for the C3 BF16 KY baseline.
# Same hyperparameters as run_bf16.sh but seed=123 to provide n=2
# variance estimate for the BF16 control (paper L1).
#
# Output goes to a separate directory so it does not collide with
# the seed=42 run in run_bf16.sh.

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p output_ky_bf16_r16_lr2e4_3ep_seed123

python scripts/train_svd.py \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --kz_file __skip__.jsonl \
    --uz_file __skip__.jsonl \
    --output_dir ./output_ky_bf16_r16_lr2e4_3ep_seed123 \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --learning_rate 2e-4 --num_train_epochs 3 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --max_seq_length 256 --warmup_ratio 0.05 \
    --svd_every_steps 100 --eval_steps 200 --save_steps 200 \
    --ppl_eval_every_steps 200 \
    --no_quantize \
    --gpu_max_memory 36GiB --cpu_max_memory 100GiB \
    --seed 123 \
    2>&1 | tee output_ky_bf16_r16_lr2e4_3ep_seed123/train.log
