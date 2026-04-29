#!/usr/bin/env bash
# run_eval_seed123.sh — Evaluate the seed=123 BF16 adapter.

set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:-./output_ky_bf16_r16_lr2e4_3ep_seed123/final_adapter}"
OUTDIR="${2:-./output_ky_bf16_r16_lr2e4_3ep_seed123/eval}"

mkdir -p "$OUTDIR"

python scripts/evaluate.py \
    --adapter_path "$ADAPTER" \
    --output_dir "$OUTDIR" \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --kz_file kazakh_raw.jsonl \
    --uz_file uzbek_final_cyrillic.jsonl \
    2>&1 | tee "$OUTDIR/../eval.log"
