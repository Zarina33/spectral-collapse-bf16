# BF16 Control Experiment (C3) — vast.ai Setup

This is a self-contained directory for running the **BF16 KY baseline** control experiment for the *Spectral Collapse in Turkic Languages* paper. Designed to clone-and-run on a vast.ai (or any cloud) GPU instance.

The goal: train Gemma-2-9B with LoRA on Kyrgyz **without 4-bit quantization** (the rest of the paper's experiments use NF4) to confirm that the spectral-energy claims survive in full BF16.

## What's in here

```
experiment/
├── README.md            ← this file
├── requirements.txt     ← pinned pip deps
├── setup.sh             ← one-time: decompress corpus + verify env
├── run_bf16.sh          ← train BF16 LoRA on KY
├── run_eval.sh          ← post-training evaluation
├── scripts/
│   ├── train_svd.py     ← training + SVD monitoring (supports --no_quantize)
│   └── evaluate.py      ← PPL + NER (gen + log-lik) + TUMLU
└── data/pretrain/
    ├── kyrgyz_raw.jsonl.gz             ← 38 MB (KY: training corpus)
    ├── kazakh_raw.jsonl.gz             ← 39 MB (KZ: for cross-lingual PPL)
    └── uzbek_final_cyrillic.jsonl.gz   ← 39 MB (UZ: for cross-lingual PPL)
```

Total repo size: ~116 MB (well under GitHub limits for individual files).

## Step-by-step on vast.ai

### 1. Pick an instance

- **A100 40GB** is enough for full BF16 9B + LoRA at seq_len 256, batch 1 (no CPU offload needed). Cheapest on vast.ai (~$0.5–1/hr).
- **A100 80GB** or **H100** are overkill but work.
- **24 GB GPU** (RTX 4090, A10) works *with* CPU offload; ~5× slower. Edit `run_bf16.sh` and lower `--gpu_max_memory` to e.g. `22GiB`.
- 16 GB GPU **will not work** — use the 4-bit version on the main repo instead.

### 2. SSH in and clone

```bash
git clone https://github.com/<your-username>/<this-repo>.git experiment
cd experiment
```

### 3. Install dependencies

If the image already has CUDA + PyTorch (most vast.ai templates do):

```bash
pip install -r requirements.txt
```

If you start from a bare CUDA image, install PyTorch first matching your CUDA version, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

You will also need to log in to Hugging Face once for Gemma access:

```bash
huggingface-cli login
# paste a HF token with read access to gated google/gemma-2-9b
```

### 4. One-time setup (decompresses data, verifies GPU)

```bash
bash setup.sh
```

Expected output ends with `==> Setup complete...`. If `torch.cuda.is_available()` is `False`, fix CUDA before continuing.

### 5. Train

```bash
bash run_bf16.sh
```

Expected runtime on A100 40GB: ~2–3 hours for 3 epochs on 4.4M Kyrgyz tokens. Logs stream to `output_ky_bf16_r16_lr2e4_3ep/train.log`. The checkpoint and final adapter land in `output_ky_bf16_r16_lr2e4_3ep/`.

If you need to detach from the SSH session:

```bash
nohup bash run_bf16.sh > run.log 2>&1 &
disown
# come back later, check progress:
tail -f run.log
```

### 6. Evaluate

After training completes:

```bash
bash run_eval.sh
```

This produces `output_ky_bf16_r16_lr2e4_3ep/eval/eval_report.json` with PPL, NER F1, NER log-likelihood type accuracy, and TUMLU accuracy across KY/KZ/UZ.

All three corpora (KY/KZ/UZ) are bundled in `data/pretrain/` as `.gz` and auto-decompressed by `setup.sh`, so cross-lingual PPL works out of the box.

### 7. Pull results back to your local repo

```bash
# from your local machine:
scp -r vastai:/path/to/experiment/output_ky_bf16_r16_lr2e4_3ep ./
```

## What gets reported

The `eval_report.json` will contain (per language `ky`/`kz`/`uz`):
- `perplexity`: cross-lingual PPL
- `ner_wikiann`: F1 from generation-based NER
- `ner_loglik`: type accuracy from log-likelihood span typing
- `tumlu_qa`: 5-shot multiple-choice accuracy

Plus `svd_log.jsonl` records spectral metrics (SE, effective rank, Frobenius) every 100 training steps — this is what the paper analyses to compare BF16 vs.\ 4-bit dynamics.

## Hyperparameters

These match the 4-bit `E3` baseline exactly except for quantization:

| | value |
|---|---|
| base model | google/gemma-2-9b |
| precision | bfloat16 (no quantization) |
| LoRA rank `r` | 16 |
| LoRA `alpha` | 32 |
| LoRA dropout | 0.05 |
| target modules | q,k,v,o,gate,up,down |
| LR | 2e-4 |
| epochs | 3 |
| batch size | 1 |
| grad accum | 16 |
| max seq length | 256 |
| warmup ratio | 0.05 |
| seed | 42 |
