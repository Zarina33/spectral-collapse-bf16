#!/usr/bin/env bash
# run_bf16.sh — C3 BF16 KY baseline (no quantization).
# On a >=24 GB GPU (A100 40GB, A100 80GB, H100): this runs cleanly without
# CPU offload. The flag --no_quantize skips the BitsAndBytesConfig path
# and loads the full BF16 model.
#
# On smaller GPUs, set --gpu_max_memory to something below your VRAM
# (e.g. 22GiB on A100 40GB) and the rest will be offloaded to CPU.

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p output_ky_bf16_r16_lr2e4_3ep

python scripts/train_svd.py \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --kz_file __skip__.jsonl \
    --uz_file __skip__.jsonl \
    --output_dir ./output_ky_bf16_r16_lr2e4_3ep \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --learning_rate 2e-4 --num_train_epochs 3 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --max_seq_length 256 --warmup_ratio 0.05 \
    --svd_every_steps 100 --eval_steps 200 --save_steps 200 \
    --ppl_eval_every_steps 200 \
    --no_quantize \
    --gpu_max_memory 36GiB --cpu_max_memory 100GiB \
    --seed 42 \
    2>&1 | tee output_ky_bf16_r16_lr2e4_3ep/train.log
