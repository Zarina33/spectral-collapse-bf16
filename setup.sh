#!/usr/bin/env bash
# setup.sh — one-time setup on a fresh vast.ai instance.
# Decompresses the training corpus, creates the output dir, and verifies
# CUDA + dependencies are present.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> Decompressing corpora (KY for training, KZ+UZ for cross-lingual eval)..."
for f in kyrgyz_raw kazakh_raw uzbek_final_cyrillic; do
    gz="data/pretrain/${f}.jsonl.gz"
    out="data/pretrain/${f}.jsonl"
    if [ -f "$gz" ] && [ ! -f "$out" ]; then
        gunzip -k "$gz"
        echo "    -> $out"
    fi
done

echo "==> Creating output directory..."
mkdir -p output_ky_bf16_r16_lr2e4_3ep

echo "==> Verifying environment..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}')"
python -c "import torch; gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}, {gb:.1f} GB')"

echo
echo "==> Setup complete. Run BF16 training with:"
echo "    bash run_bf16.sh"
