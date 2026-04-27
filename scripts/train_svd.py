#!/usr/bin/env python3
"""
train_svd.py — Fine-tuning Gemma-2-9B with LoRA + SVD Monitoring (v3)
======================================================================
Скрипт для тонкой настройки Gemma-2-9B на кыргызском, казахском и узбекском
текстовых корпусах с LoRA-адаптерами и мониторингом сингулярных значений.

Выходные файлы:
    output/
    ├── final_adapter/              # LoRA веса
    ├── config_dump.json            # Полный конфиг для reproducibility
    ├── training_log.jsonl          # Loss, PPL, LR, grad_norm, VRAM, tokens/sec
    ├── svd_log.jsonl               # SVD метрики каждые N шагов
    ├── experiment_dashboard.png    # 4-panel figure для статьи
    ├── spectral_energy.png         # Детальные SVD графики по слоям
    └── train_results.json          # Финальные метрики

Требования:
    pip install torch transformers peft datasets accelerate bitsandbytes matplotlib

Запуск:
    python train_svd.py --data_dir ./data/pretrain \
        --ky_file kyrgyz_raw.jsonl --kz_file kazakh_raw.jsonl --uz_file uzbek_final_cyrillic.jsonl
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


# ════════════════════════════════════════════════════════════════════
# 1.  CLI Arguments
# ════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Gemma-2-9B LoRA fine-tuning with SVD monitoring")

    # Paths
    p.add_argument("--model_name", type=str, default="google/gemma-2-9b")
    p.add_argument("--data_dir", type=str, default="./data/pretrain")
    p.add_argument("--ky_file", type=str, default="kyrgyz_raw.jsonl")
    p.add_argument("--kz_file", type=str, default="kazakh_raw.jsonl")
    p.add_argument("--uz_file", type=str, default="uzbek_final_cyrillic.jsonl")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--svd_log", type=str, default="svd_log.jsonl")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--svd_every_steps", type=int, default=100)
    p.add_argument("--val_split", type=float, default=0.10,
                    help="Fraction of data for validation (default 10%%)")
    p.add_argument("--ppl_eval_every_steps", type=int, default=200,
                    help="Per-language PPL eval frequency (default: 200)")
    p.add_argument("--ppl_max_eval_samples", type=int, default=200,
                    help="Max samples per language for PPL eval (default: 200)")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_quantize", action="store_true", default=False,
                    help="Disable 4-bit quantization (full BF16). "
                         "Requires CPU offload on <24GB GPUs.")
    p.add_argument("--gpu_max_memory", type=str, default="13GiB",
                    help="Max GPU memory for auto device_map when --no_quantize "
                         "(rest goes to CPU). Default 13GiB leaves headroom on 16GB.")
    p.add_argument("--cpu_max_memory", type=str, default="100GiB",
                    help="Max CPU RAM for offload when --no_quantize.")
    p.add_argument("--seed", type=int, default=42)

    # Resume from checkpoint
    p.add_argument("--resume", action="store_true", default=False,
                    help="Resume training from the last checkpoint in output_dir")

    # Cross-lingual transfer (Step 4)
    p.add_argument("--init_adapter", type=str, default=None,
                    help="Path to a pre-trained LoRA adapter for cross-lingual "
                         "transfer (e.g. output_kz/final_adapter for KZ→KY)")
    p.add_argument("--transfer_warmup_steps", type=int, default=100,
                    help="Warmup steps when loading a pre-trained adapter "
                         "(default: 100)")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════
# 2.  Data Loading, Token Stats & Tokenization
# ════════════════════════════════════════════════════════════════════
def compute_tokens_per_word(texts, tokenizer, sample_size=2000):
    """Compute average tokens per word on a sample of texts."""
    total_words = 0
    total_tokens = 0
    for text in texts[:sample_size]:
        words = text.split()
        total_words += len(words)
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
    if total_words == 0:
        return 0.0
    return total_tokens / total_words


# English control sentences for Subword Fragmentation Index comparison
EN_CONTROL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming the way we approach complex problems.",
    "She walked through the ancient marketplace, admiring the colorful displays.",
    "The development of sustainable energy sources remains a critical challenge.",
    "Students gathered in the university library to prepare for final examinations.",
    "Modern transportation systems rely heavily on digital infrastructure networks.",
    "The conference brought together researchers from diverse academic backgrounds.",
    "Climate change continues to affect agricultural production worldwide.",
    "The new hospital was equipped with state-of-the-art medical technology.",
    "International cooperation is essential for addressing global security challenges.",
    "The documentary explored the rich cultural heritage of Central Asian nomads.",
    "Economists predict that inflation will gradually decrease over the next quarter.",
    "The orchestra performed a magnificent rendition of the classical symphony.",
    "Young entrepreneurs are increasingly turning to technology startups for innovation.",
    "The mountain expedition required months of careful planning and preparation.",
    "Renewable energy investments have grown significantly in the past decade.",
    "The ancient manuscript contained valuable insights into medieval trade routes.",
    "Public health officials emphasized the importance of vaccination programs.",
    "The architectural design combined traditional elements with modern aesthetics.",
    "Digital literacy has become an essential skill in the contemporary workforce.",
]


def load_and_tokenize(args, tokenizer):
    """
    Load three JSONL files, compute per-language token statistics
    (including English control baseline), tokenize, split into train/val.
    Returns (train_dataset, val_dataset, per_lang_val_datasets,
             per_lang_stats, total_tokens).
    """
    lang_map = {
        "ky": (args.ky_file, "Kyrgyz"),
        "kz": (args.kz_file, "Kazakh"),
        "uz": (args.uz_file, "Uzbek"),
    }
    loaded = []          # (lang, dataset)
    per_lang_stats = {}

    for lang, (fname, lang_name) in lang_map.items():
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.isfile(fpath):
            print(f"[WARNING] File not found: {fpath}, skipping.")
            continue
        ds = load_dataset("json", data_files=fpath, split="train")
        file_mb = os.path.getsize(fpath) / (1024 * 1024)

        # ── Add language label for per-language evaluation ──
        ds = ds.add_column("lang", [lang] * len(ds))

        # ── Tokens-per-word statistic ──
        sample_texts = ds["text"][:2000]
        tpw = compute_tokens_per_word(sample_texts, tokenizer)

        per_lang_stats[lang] = {
            "language": lang_name,
            "file": os.path.basename(fpath),
            "records": len(ds),
            "file_size_mb": round(file_mb, 1),
            "tokens_per_word": round(tpw, 3),
        }
        print(f"  [{lang}] {fname:30s}  {len(ds):>7,} records | "
              f"{file_mb:6.1f} MB | tokens/word = {tpw:.3f}")
        loaded.append((lang, ds))

    if not loaded:
        sys.exit("[ERROR] No data files found. Check --data_dir and filenames.")

    # ── English control baseline (Subword Fragmentation Index) ──
    en_tpw = compute_tokens_per_word(EN_CONTROL_SENTENCES, tokenizer,
                                     sample_size=len(EN_CONTROL_SENTENCES))
    per_lang_stats["en"] = {
        "language": "English (control)",
        "file": "built-in_control_sentences",
        "records": len(EN_CONTROL_SENTENCES),
        "file_size_mb": 0.0,
        "tokens_per_word": round(en_tpw, 3),
    }

    # ── Concatenate & shuffle ──
    dataset = concatenate_datasets([ds for _, ds in loaded]).shuffle(seed=args.seed)
    print(f"\n[INFO] Total records: {len(dataset):,}")

    # ── Tokenize ──
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "lang"],
        num_proc=os.cpu_count(),
        desc="Tokenizing",
    )

    total_tokens = sum(len(ids) for ids in tokenized["input_ids"])
    print(f"[INFO] Total tokens: {total_tokens:,}")

    # ── Per-language estimated tokens (only for real languages) ──
    real_langs = {k: v for k, v in per_lang_stats.items() if k != "en"}
    total_records = sum(s["records"] for s in real_langs.values())
    for lang, stats in real_langs.items():
        ratio = stats["records"] / total_records if total_records > 0 else 0
        stats["estimated_tokens"] = int(total_tokens * ratio)

    # ── Print Subword Fragmentation Index table ──
    print("\n" + "─" * 72)
    print("  SUBWORD FRAGMENTATION INDEX  (Token-to-Word Ratio)")
    print("─" * 72)
    print(f"  {'Lang':<12} {'Language':<18} {'Records':>10} {'Tok/Word':>10} {'vs EN':>8}")
    print("  " + "─" * 60)
    for lang in ["ky", "kz", "uz", "en"]:
        if lang not in per_lang_stats:
            continue
        s = per_lang_stats[lang]
        ratio_vs_en = (s["tokens_per_word"] / en_tpw) if en_tpw > 0 else 0
        marker = "  (baseline)" if lang == "en" else f"  {ratio_vs_en:.1f}x"
        print(f"  {lang:<12} {s['language']:<18} {s['records']:>10,} "
              f"{s['tokens_per_word']:>10.3f}{marker}")
    print("─" * 72)

    # ── Print token distribution table ──
    print(f"\n  {'Lang':<12} {'Est. Tokens':>14}")
    print("  " + "─" * 30)
    for lang in ["ky", "kz", "uz"]:
        if lang not in per_lang_stats or "estimated_tokens" not in per_lang_stats[lang]:
            continue
        s = per_lang_stats[lang]
        print(f"  {lang:<12} {s['estimated_tokens']:>14,}")
    print()

    # ── Train / Val split ──
    split = tokenized.train_test_split(test_size=args.val_split, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[INFO] Train: {len(train_ds):,} | Val: {len(val_ds):,} "
          f"({args.val_split*100:.0f}% split)")

    # ── Per-language validation subsets (for PerLanguagePPLCallback) ──
    per_lang_val_datasets = {}
    for lang_code in ["ky", "kz", "uz"]:
        lang_subset = val_ds.filter(lambda x: x["lang"] == lang_code)
        if len(lang_subset) > 0:
            per_lang_val_datasets[lang_code] = lang_subset.remove_columns(["lang"])
            print(f"  [EVAL] {lang_code} val subset: {len(lang_subset):,} examples")

    # Remove 'lang' column from main datasets (Trainer doesn't need it)
    train_ds = train_ds.remove_columns(["lang"])
    val_ds = val_ds.remove_columns(["lang"])

    return train_ds, val_ds, per_lang_val_datasets, per_lang_stats, total_tokens


# ════════════════════════════════════════════════════════════════════
# 3.  SVD Monitoring Callback  (extended)
# ════════════════════════════════════════════════════════════════════
class SpectralMonitor(TrainerCallback):
    """
    Every `svd_every_steps` steps, for each lora_B matrix computes:
      - spectral_energy   S₁/ΣSᵢ        (collapse detector, threshold 0.7)
      - effective_rank    # of Sᵢ for 90% cumulative energy
      - svd_entropy       H(S) = −Σpᵢ log pᵢ
      - stable_rank       ||W||²_F / σ²_max
      - frobenius_norm_A / B
      - vram, tokens/sec

    Also tracks first_collapse_layer — the first layer where SE > 0.7.
    """

    COLLAPSE_THRESHOLD = 0.7  # S₁/ΣSᵢ > 0.7 → collapse signal

    def __init__(self, svd_log_path: str, svd_every_steps: int = 100,
                 num_model_layers: int = 42, resume: bool = False):
        super().__init__()
        self.svd_log_path = svd_log_path
        self.svd_every_steps = svd_every_steps
        self.num_model_layers = num_model_layers
        self._layer_boundary = num_model_layers // 2  # lower vs upper split
        self._step_start = None
        self._collapse_detected = {}  # {layer_name: first_step}
        os.makedirs(os.path.dirname(svd_log_path) or ".", exist_ok=True)
        if not resume:
            with open(svd_log_path, "w") as f:
                pass

    @staticmethod
    def _stable_rank(S: torch.Tensor) -> float:
        """||W||²_F / ||W||²_2  =  Σ(s²) / s_max²"""
        s_sq = S ** 2
        return (s_sq.sum() / s_sq[0]).item() if s_sq[0] > 0 else 0.0

    @staticmethod
    def _effective_rank(S: torch.Tensor, threshold: float = 0.9) -> int:
        """Number of singular values needed to capture `threshold` of total energy."""
        s_sq = S ** 2
        total = s_sq.sum().item()
        if total == 0:
            return 0
        cumsum = 0.0
        for i, val in enumerate(s_sq.tolist()):
            cumsum += val
            if cumsum / total >= threshold:
                return i + 1
        return len(S)

    @staticmethod
    def _svd_entropy(S: torch.Tensor) -> float:
        """Shannon entropy of normalised singular-value distribution: H(S) = −Σpᵢ log pᵢ"""
        p = S / S.sum()
        p = p[p > 0]
        return -(p * p.log()).sum().item()

    def get_collapse_summary(self):
        """Return a summary dict describing collapse dynamics across layers.

        Returns:
            dict with keys: total_collapsed, first_layer_name, first_layer_index,
            first_layer_zone, first_step, last_step, by_zone (lower/upper counts),
            layers (sorted list of {name, index, step, zone}).
        """
        if not self._collapse_detected:
            return None

        layers = []
        for name, step in self._collapse_detected.items():
            idx = _extract_layer_idx(name)
            zone = "unknown"
            if idx is not None:
                zone = "lower" if idx < self._layer_boundary else "upper"
            layers.append({"name": name, "index": idx, "step": step, "zone": zone})

        # Sort by step (first collapse first), then by layer index
        layers.sort(key=lambda x: (x["step"], x["index"] if x["index"] is not None else 999))

        first = layers[0]
        by_zone = {"lower": 0, "upper": 0, "unknown": 0}
        for l in layers:
            by_zone[l["zone"]] += 1

        return {
            "total_collapsed": len(layers),
            "first_layer_name": first["name"],
            "first_layer_index": first["index"],
            "first_layer_zone": first["zone"],
            "first_step": first["step"],
            "last_step": layers[-1]["step"],
            "by_zone": by_zone,
            "layers": layers,
        }

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.svd_every_steps != 0:
            return

        step = state.global_step
        elapsed = time.time() - self._step_start if self._step_start else 0
        print(f"\n[SVD] Step {step}: computing SVD decomposition...")

        # ── Performance snapshot ──
        perf = {}
        if torch.cuda.is_available():
            perf["vram_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024**3), 3)
            perf["vram_reserved_gb"] = round(
                torch.cuda.memory_reserved() / (1024**3), 3)
            perf["vram_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3)
        if elapsed > 0:
            tokens_per_step = (args.per_device_train_batch_size
                               * args.gradient_accumulation_steps * 512)
            perf["tokens_per_second"] = round(tokens_per_step / elapsed, 1)

        # ── Pair lora_A ↔ lora_B ──
        lora_params = {}
        for name, param in model.named_parameters():
            if "lora_A" in name and "weight" in name:
                base = name.replace("lora_A", "lora_X")
                lora_params.setdefault(base, {})["A"] = (name, param)
            elif "lora_B" in name and "weight" in name:
                base = name.replace("lora_B", "lora_X")
                lora_params.setdefault(base, {})["B"] = (name, param)

        records = []
        collapse_alerts = []

        for base, pair in lora_params.items():
            if "B" not in pair:
                continue

            name_b, param_b = pair["B"]
            W = param_b.detach().float()

            try:
                _, S, _ = torch.linalg.svd(W, full_matrices=False)
            except Exception as e:
                print(f"  [SVD WARNING] {name_b}: {e}")
                continue

            s_sum = S.sum().item()
            se = (S[0].item() / s_sum) if s_sum > 0 else 0.0

            # Extract layer index from name (e.g., "...layers.12....")
            layer_idx = _extract_layer_idx(name_b)

            record = {
                "step": step,
                "layer_name": name_b,
                "layer_index": layer_idx,
                "singular_values": S.cpu().tolist(),
                "spectral_energy": round(se, 6),
                "effective_rank": self._effective_rank(S, threshold=0.9),
                "svd_entropy": round(self._svd_entropy(S), 6),
                "stable_rank": round(self._stable_rank(S), 4),
                "frobenius_norm_B": round(W.norm().item(), 6),
            }

            if "A" in pair:
                _, param_a = pair["A"]
                record["frobenius_norm_A"] = round(
                    param_a.detach().float().norm().item(), 6)

            # ── Collapse detection ──
            if se > self.COLLAPSE_THRESHOLD and name_b not in self._collapse_detected:
                self._collapse_detected[name_b] = step
                record["collapse_detected_at_step"] = step
                collapse_alerts.append(name_b)

            record.update(perf)
            records.append(record)

        with open(self.svd_log_path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if records:
            energies = [r["spectral_energy"] for r in records]
            eff_ranks = [r["effective_rank"] for r in records]
            sranks = [r["stable_rank"] for r in records]
            vram = perf.get("vram_allocated_gb", "N/A")
            tps = perf.get("tokens_per_second", "N/A")
            print(f"  [SVD] {len(records)} layers | "
                  f"SE: [{min(energies):.4f}–{max(energies):.4f}] | "
                  f"EffRank: [{min(eff_ranks)}–{max(eff_ranks)}] | "
                  f"SR: [{min(sranks):.2f}–{max(sranks):.2f}] | "
                  f"VRAM: {vram} GB | {tps} tok/s")

        if collapse_alerts:
            print(f"  ⚠ COLLAPSE ALERT (SE > {self.COLLAPSE_THRESHOLD}) "
                  f"at step {step} in {len(collapse_alerts)} layer(s):")
            for name in collapse_alerts[:5]:
                idx = _extract_layer_idx(name)
                idx_str = f"layer {idx}" if idx is not None else "layer ?"
                zone = ""
                if idx is not None:
                    zone = " (lower)" if idx < self._layer_boundary else " (upper)"
                print(f"      → [{idx_str}{zone}] {name}")
            if len(collapse_alerts) > 5:
                print(f"      ... and {len(collapse_alerts) - 5} more")


# ════════════════════════════════════════════════════════════════════
# 4.  Training Metrics Logger Callback
# ════════════════════════════════════════════════════════════════════
class TrainingMetricsLogger(TrainerCallback):
    """
    Logs every on_log event: loss, eval_loss, perplexity, learning_rate,
    grad_norm, GPU memory, throughput → training_log.jsonl.
    """

    def __init__(self, log_path: str, resume: bool = False):
        super().__init__()
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not resume:
            with open(log_path, "w") as f:
                pass
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")

        record = {
            "step": step,
            "epoch": round(state.epoch, 4) if state.epoch else None,
            "train_loss": round(loss, 6) if loss is not None else None,
            "train_ppl": round(math.exp(loss), 4) if loss is not None and loss < 20 else None,
            "eval_loss": round(eval_loss, 6) if eval_loss is not None else None,
            "eval_ppl": round(math.exp(eval_loss), 4) if eval_loss is not None and eval_loss < 20 else None,
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": round(logs["grad_norm"], 6) if logs.get("grad_norm") is not None else None,
        }

        if torch.cuda.is_available():
            record["vram_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024**3), 3)
            record["vram_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3)

        if self._step_start is not None and loss is not None:
            elapsed = time.time() - self._step_start
            tokens_per_step = (args.per_device_train_batch_size
                               * args.gradient_accumulation_steps * 512)
            record["step_time_sec"] = round(elapsed, 3)
            record["tokens_per_sec"] = round(tokens_per_step / elapsed, 1) if elapsed > 0 else 0

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════════
# 4b.  Per-Language Perplexity Callback
# ════════════════════════════════════════════════════════════════════
class PerLanguagePPLCallback(TrainerCallback):
    """
    Every `eval_every_steps` steps, compute per-language perplexity
    on small held-out sets. Logs to ppl_per_lang_log.jsonl.

    Tracks which language's PPL degrades first as spectral collapse occurs.
    """

    def __init__(self, per_lang_val_datasets: dict, tokenizer,
                 log_path: str, eval_every_steps: int = 200,
                 max_eval_samples: int = 200, eval_batch_size: int = 4,
                 seed: int = 42, resume: bool = False):
        super().__init__()
        self.per_lang_val_datasets = per_lang_val_datasets
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.eval_every_steps = eval_every_steps
        self.max_eval_samples = max_eval_samples
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not resume:
            with open(log_path, "w") as f:
                pass

    def _compute_ppl(self, model, dataset) -> tuple:
        """Compute average cross-entropy loss and perplexity on a dataset."""
        model.eval()

        if len(dataset) > self.max_eval_samples:
            dataset = dataset.shuffle(seed=self.seed).select(range(self.max_eval_samples))

        cols_to_keep = {"input_ids", "attention_mask", "labels"}
        remove_cols = [c for c in dataset.column_names if c not in cols_to_keep]
        if remove_cols:
            dataset = dataset.remove_columns(remove_cols)

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.eval_batch_size,
            collate_fn=collator, shuffle=False)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                labels = batch["labels"]
                n_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return avg_loss, ppl

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.eval_every_steps != 0:
            return

        step = state.global_step
        print(f"\n[PPL] Step {step}: computing per-language perplexity...")

        record = {"step": step,
                  "epoch": round(state.epoch, 4) if state.epoch else None}

        for lang_code, lang_ds in self.per_lang_val_datasets.items():
            loss, ppl = self._compute_ppl(model, lang_ds)
            record[f"{lang_code}_loss"] = round(loss, 6)
            record[f"{lang_code}_ppl"] = round(ppl, 4) if ppl != float("inf") else None
            print(f"  [{lang_code}] loss={loss:.4f}  ppl={ppl:.2f}")

        model.train()

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════════
# 5.  Experiment Dashboard  (single 2×2 figure)
# ════════════════════════════════════════════════════════════════════
def _read_jsonl(path):
    entries = []
    if not os.path.isfile(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip().strip('\x00')
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _extract_layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def plot_experiment_dashboard(training_log_path: str, svd_log_path: str,
                              output_dir: str, target_layers=(5, 12, 20)):
    """
    Creates experiment_dashboard.png — 2×2 figure:
        (0,0) Loss curve  (Train vs Val)
        (0,1) Perplexity  (Train vs Val)
        (1,0) Spectral Energy vs Stable Rank  (averaged per layer group)
        (1,1) Weight Norms (lora_A vs lora_B averaged across layers)
    """
    t_entries = _read_jsonl(training_log_path)
    s_entries = _read_jsonl(svd_log_path)

    if not t_entries and not s_entries:
        print("[PLOT] No log data found, skipping dashboard.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Colors ──
    C_TRAIN = "#e74c3c"
    C_VAL   = "#3498db"
    C_SE    = "#e74c3c"
    C_SR    = "#2ecc71"
    C_NA    = "#e74c3c"
    C_NB    = "#3498db"

    # ────────────────────────────────────────────────────
    # (0,0) Loss — Train vs Val
    # ────────────────────────────────────────────────────
    ax = axes[0, 0]
    train_loss_steps = [(e["step"], e["train_loss"]) for e in t_entries
                        if e.get("train_loss") is not None]
    eval_loss_steps  = [(e["step"], e["eval_loss"]) for e in t_entries
                        if e.get("eval_loss") is not None]

    if train_loss_steps:
        s, v = zip(*train_loss_steps)
        ax.plot(s, v, linewidth=1.2, color=C_TRAIN, label="Train loss", alpha=0.8)
    if eval_loss_steps:
        s, v = zip(*eval_loss_steps)
        ax.plot(s, v, linewidth=2, color=C_VAL, marker="D", markersize=5,
                label="Val loss")
    ax.set_title("Loss  (Train vs Validation)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────
    # (0,1) Perplexity — Train vs Val
    # ────────────────────────────────────────────────────
    ax = axes[0, 1]
    train_ppl_steps = [(e["step"], e["train_ppl"]) for e in t_entries
                       if e.get("train_ppl") is not None]
    eval_ppl_steps  = [(e["step"], e["eval_ppl"]) for e in t_entries
                       if e.get("eval_ppl") is not None]

    if train_ppl_steps:
        s, v = zip(*train_ppl_steps)
        ax.plot(s, v, linewidth=1.2, color=C_TRAIN, label="Train PPL", alpha=0.8)
    if eval_ppl_steps:
        s, v = zip(*eval_ppl_steps)
        ax.plot(s, v, linewidth=2, color=C_VAL, marker="D", markersize=5,
                label="Val PPL")
    ax.set_title("Perplexity  (Train vs Validation)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity  exp(loss)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────
    # (1,0) Spectral Decay vs Stable Rank  (layer averages)
    # ────────────────────────────────────────────────────
    ax = axes[1, 0]
    if s_entries:
        step_se = {}   # {step: [values]}
        step_sr = {}
        step_er = {}   # effective_rank
        for e in s_entries:
            st = e["step"]
            step_se.setdefault(st, []).append(e["spectral_energy"])
            step_sr.setdefault(st, []).append(e.get("stable_rank", 0))
            step_er.setdefault(st, []).append(e.get("effective_rank", 0))

        steps_sorted = sorted(step_se.keys())
        avg_se = [sum(step_se[s]) / len(step_se[s]) for s in steps_sorted]
        avg_sr = [sum(step_sr[s]) / len(step_sr[s]) for s in steps_sorted]
        avg_er = [sum(step_er[s]) / len(step_er[s]) for s in steps_sorted]

        ax.plot(steps_sorted, avg_se, linewidth=1.8, color=C_SE, marker="o",
                markersize=3, label="Spectral Decay S₁/ΣSᵢ")
        # Collapse threshold line
        ax.axhline(y=0.7, color="#999999", linestyle="--", linewidth=1,
                   label="Collapse threshold (0.7)")

        ax2 = ax.twinx()
        ax2.plot(steps_sorted, avg_sr, linewidth=1.8, color=C_SR, marker="s",
                 markersize=3, label="Stable Rank (avg)")
        ax2.plot(steps_sorted, avg_er, linewidth=1.4, color="#f39c12", marker="^",
                 markersize=3, label="Effective Rank (avg)", alpha=0.8)
        ax2.set_ylabel("Rank", color=C_SR, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=C_SR)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    ax.set_title("Spectral Decay vs Stable Rank",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Spectral Decay  S₁/Σ(S)", color=C_SE, fontsize=11)
    ax.tick_params(axis="y", labelcolor=C_SE)
    ax.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────
    # (1,1) Weight Norms  (lora_A vs lora_B, averaged)
    # ────────────────────────────────────────────────────
    ax = axes[1, 1]
    if s_entries:
        step_na = {}  # {step: [norm_A values]}
        step_nb = {}
        for e in s_entries:
            st = e["step"]
            step_nb.setdefault(st, []).append(e.get("frobenius_norm_B", 0))
            if "frobenius_norm_A" in e:
                step_na.setdefault(st, []).append(e["frobenius_norm_A"])

        steps_sorted = sorted(step_nb.keys())
        avg_nb = [sum(step_nb[s]) / len(step_nb[s]) for s in steps_sorted]
        ax.plot(steps_sorted, avg_nb, linewidth=1.8, color=C_NB, marker="o",
                markersize=3, label="||lora_B||_F  (avg)")

        if step_na:
            steps_a = sorted(step_na.keys())
            avg_na = [sum(step_na[s]) / len(step_na[s]) for s in steps_a]
            ax.plot(steps_a, avg_na, linewidth=1.8, color=C_NA, marker="s",
                    markersize=3, label="||lora_A||_F  (avg)")

    ax.set_title("Weight Norms  (lora_A vs lora_B)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Frobenius Norm (averaged across layers)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment Dashboard — LoRA Fine-tuning with SVD Monitoring",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "experiment_dashboard.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved experiment dashboard → {path}")


# ════════════════════════════════════════════════════════════════════
# 6.  Detailed SVD Per-Layer Plot
# ════════════════════════════════════════════════════════════════════
def plot_spectral_energy_detail(svd_log_path: str, output_dir: str,
                                target_layers=(5, 12, 20)):
    """Detailed per-module: Row 0 = Spectral Energy, Row 1 = Effective Rank."""
    entries = _read_jsonl(svd_log_path)
    if not entries:
        print("[PLOT] svd_log.jsonl empty, skipping detail plot.")
        return

    layer_se = {}   # {idx: {module: {step: val}}}
    layer_er = {}   # effective_rank

    for e in entries:
        idx = _extract_layer_idx(e["layer_name"])
        if idx is None:
            continue
        name = e["layer_name"]

        module_key = None
        if "self_attn" in name:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if proj in name:
                    module_key = f"attn.{proj}"
                    break
        elif "mlp" in name:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if proj in name:
                    module_key = f"mlp.{proj}"
                    break
        if module_key is None:
            module_key = name

        layer_se.setdefault(idx, {}).setdefault(module_key, {})[e["step"]] = e["spectral_energy"]
        if "effective_rank" in e:
            layer_er.setdefault(idx, {}).setdefault(module_key, {})[e["step"]] = e["effective_rank"]

    available = sorted(layer_se.keys())
    plot_layers = [l for l in target_layers if l in available]
    if not plot_layers:
        if len(available) >= 3:
            plot_layers = [available[0], available[len(available)//2], available[-1]]
        else:
            plot_layers = available
        print(f"[PLOT] Fallback layers: {plot_layers}")

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    n = len(plot_layers)
    fig, axes = plt.subplots(2, n, figsize=(7*n, 10), squeeze=False)

    for col, lidx in enumerate(plot_layers):
        # Row 0: Spectral Energy (with collapse threshold)
        ax = axes[0][col]
        for i, (mod, sv) in enumerate(sorted(layer_se.get(lidx, {}).items())):
            steps = sorted(sv)
            ax.plot(steps, [sv[s] for s in steps], marker="o", markersize=3,
                    linewidth=1.4, label=mod, color=colors[i % len(colors)])
        ax.axhline(y=0.7, color="#999", linestyle="--", linewidth=1, alpha=0.7,
                   label="collapse (0.7)")
        ax.set_title(f"Layer {lidx} — Spectral Decay", fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("S₁ / Σ(Sᵢ)")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 1: Effective Rank
        ax = axes[1][col]
        for i, (mod, sv) in enumerate(sorted(layer_er.get(lidx, {}).items())):
            steps = sorted(sv)
            ax.plot(steps, [sv[s] for s in steps], marker="s", markersize=3,
                    linewidth=1.4, label=mod, color=colors[i % len(colors)])
        ax.axhline(y=3.0, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7,
                   label="critical (3.0)")
        ax.set_title(f"Layer {lidx} — Effective Rank", fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("# of Sᵢ for 90% energy")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Layer SVD Analysis (lora_B)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "spectral_energy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved per-layer SVD detail → {path}")


# ════════════════════════════════════════════════════════════════════
# 6b. Per-Language PPL vs Spectral Collapse Plot
# ════════════════════════════════════════════════════════════════════
def plot_ppl_per_lang(ppl_log_path: str, svd_log_path: str, output_dir: str):
    """
    Plot per-language perplexity curves with spectral collapse markers.
    Key visualization: shows which language degrades first at collapse onset.
    """
    ppl_entries = _read_jsonl(ppl_log_path)
    svd_entries = _read_jsonl(svd_log_path)

    if not ppl_entries:
        print("[PLOT] ppl_per_lang_log.jsonl empty, skipping per-lang PPL plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    lang_colors = {"ky": "#2196F3", "kz": "#FF9800", "uz": "#4CAF50"}
    lang_names = {"ky": "Kyrgyz", "kz": "Kazakh", "uz": "Uzbek"}

    # ── (a) Per-language PPL curves ──
    ax = axes[0]
    for lang in ["ky", "kz", "uz"]:
        key = f"{lang}_ppl"
        steps = [e["step"] for e in ppl_entries if e.get(key) is not None]
        vals = [e[key] for e in ppl_entries if e.get(key) is not None]
        if steps:
            ax.plot(steps, vals, linewidth=2, marker="o", markersize=4,
                    color=lang_colors[lang], label=lang_names[lang])

    # Find first collapse step from SVD log
    first_collapse_step = None
    for e in svd_entries:
        if e.get("collapse_detected_at_step") is not None:
            step = e["collapse_detected_at_step"]
            if first_collapse_step is None or step < first_collapse_step:
                first_collapse_step = step

    if first_collapse_step is not None:
        ax.axvline(x=first_collapse_step, color="#e74c3c", linestyle="--",
                   linewidth=1.5, alpha=0.7,
                   label=f"First collapse (step {first_collapse_step})")

    ax.set_title("Per-Language Perplexity", fontweight="bold", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity  exp(loss)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── (b) Per-language loss curves ──
    ax = axes[1]
    for lang in ["ky", "kz", "uz"]:
        key = f"{lang}_loss"
        steps = [e["step"] for e in ppl_entries if e.get(key) is not None]
        vals = [e[key] for e in ppl_entries if e.get(key) is not None]
        if steps:
            ax.plot(steps, vals, linewidth=2, marker="s", markersize=4,
                    color=lang_colors[lang], label=lang_names[lang])

    if first_collapse_step is not None:
        ax.axvline(x=first_collapse_step, color="#e74c3c", linestyle="--",
                   linewidth=1.5, alpha=0.7,
                   label=f"First collapse (step {first_collapse_step})")

    ax.set_title("Per-Language Validation Loss", fontweight="bold", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Language Evaluation vs Spectral Collapse",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "ppl_per_lang.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved per-language PPL plot → {path}")


# ════════════════════════════════════════════════════════════════════
# 7.  Config Dump  (reproducibility)
# ════════════════════════════════════════════════════════════════════
def save_config_dump(args, per_lang_stats, total_tokens, model, output_dir):
    """Save config_dump.json with all hyperparameters + token stats."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    config = {
        "experiment_date": datetime.now().isoformat(),
        "model": {
            "name": args.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "trainable_pct": round(100 * trainable / total_params, 4),
            "quantization": ("none (BF16 + CPU offload)"
                             if getattr(args, "no_quantize", False)
                             else "4-bit NF4 double-quant"),
            "dtype": "bfloat16" if args.bf16 else "float16",
            "attn_implementation": "eager",
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "alpha_over_r": args.lora_alpha / args.lora_r,
            "dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "data": {
            "total_tokens": total_tokens,
            "max_seq_length": args.max_seq_length,
            "val_split": args.val_split,
            "per_language": per_lang_stats,
        },
        "training": {
            "num_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (args.per_device_train_batch_size
                                    * args.gradient_accumulation_steps),
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": "cosine",
            "optimizer": "paged_adamw_32bit",
            "max_grad_norm": 0.3,
            "gradient_checkpointing": True,
            "seed": args.seed,
        },
        "evaluation": {
            "eval_steps": args.eval_steps,
            "val_split_pct": args.val_split * 100,
        },
        "svd_monitoring": {
            "log_every_steps": args.svd_every_steps,
            "collapse_threshold": SpectralMonitor.COLLAPSE_THRESHOLD,
            "metrics": ["singular_values", "spectral_energy", "effective_rank",
                        "svd_entropy", "stable_rank", "frobenius_norm_A",
                        "frobenius_norm_B", "vram_allocated_gb", "tokens_per_second"],
        },
        "hardware": {},
    }

    if torch.cuda.is_available():
        config["hardware"] = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
            "cuda_version": torch.version.cuda or "N/A",
        }

    config["software"] = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import transformers, peft, datasets, accelerate
        config["software"]["transformers"] = transformers.__version__
        config["software"]["peft"] = peft.__version__
        config["software"]["datasets"] = datasets.__version__
        config["software"]["accelerate"] = accelerate.__version__
    except Exception:
        pass

    path = os.path.join(output_dir, "config_dump.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Config dump saved → {path}")
    return config


# ════════════════════════════════════════════════════════════════════
# 8.  Model Setup
# ════════════════════════════════════════════════════════════════════
def setup_model_and_tokenizer(args):
    """Load Gemma-2-9B (4-bit NF4 by default, or full BF16 with CPU offload
    when --no_quantize) and attach LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.no_quantize:
        # ── Full BF16 with CPU offload (C3 control experiment) ──
        print(f"[INFO] Loading model (BF16, CPU offload): {args.model_name}")
        print(f"[INFO] max_memory: GPU={args.gpu_max_memory}, "
              f"CPU={args.cpu_max_memory}")
        import os as _os
        offload_dir = _os.path.join(args.output_dir, "offload")
        _os.makedirs(offload_dir, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: args.gpu_max_memory, "cpu": args.cpu_max_memory},
            offload_folder=offload_dir,
            attn_implementation="eager",
        )
        # No k-bit prep needed; just enable gradient checkpointing and
        # ensure inputs require grad for the LoRA wrapper to work.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        # ── Default: 4-bit NF4 quantization ──
        print(f"[INFO] Loading model (4-bit NF4): {args.model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            attn_implementation="eager",
        )
        model = prepare_model_for_kbit_training(model)

    if args.init_adapter:
        # ── Cross-lingual transfer: load pre-trained LoRA adapter ──
        from peft import PeftModel
        print(f"[INFO] Loading pre-trained adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter,
                                          is_trainable=True)
        print(f"[INFO] Transfer mode: warmup={args.transfer_warmup_steps} steps")
    else:
        # ── Standard: create fresh LoRA adapter ──
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════
# 9.  Main
# ════════════════════════════════════════════════════════════════════
def _find_last_checkpoint(output_dir):
    """Find the last checkpoint-XXXX directory in output_dir."""
    import glob
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                         key=lambda p: int(p.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # ── Resume detection ──
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = _find_last_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"[RESUME] Found checkpoint: {resume_checkpoint}")
        else:
            print("[RESUME] No checkpoint found, starting from scratch.")
            args.resume = False

    print("=" * 70)
    print("  Gemma-2-9B  LoRA Fine-tuning + SVD Monitoring  (v3)")
    print("=" * 70)

    # ── Model & tokenizer ──
    model, tokenizer = setup_model_and_tokenizer(args)

    # ── Data (with per-language token stats) ──
    train_ds, val_ds, per_lang_val_datasets, per_lang_stats, total_tokens = \
        load_and_tokenize(args, tokenizer)

    # ── Config dump (before training for reproducibility) ──
    save_config_dump(args, per_lang_stats, total_tokens, model, args.output_dir)

    # ── Data collator (dynamic padding + labels=-100 on pad positions) ──
    def clm_collator(features):
        """Pad to longest in batch; set labels=-100 on padding positions."""
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            ids = f["input_ids"]
            pad_len = max_len - len(ids)
            batch["input_ids"].append(ids + [pad_id] * pad_len)
            batch["attention_mask"].append([1] * len(ids) + [0] * pad_len)
            batch["labels"].append(ids + [-100] * pad_len)
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch
    data_collator = clm_collator

    # ── Callbacks ──
    svd_log_path = os.path.join(args.output_dir, args.svd_log)
    training_log_path = os.path.join(args.output_dir, "training_log.jsonl")
    ppl_log_path = os.path.join(args.output_dir, "ppl_per_lang_log.jsonl")

    # Detect number of transformer layers for lower/upper zone classification
    num_layers = model.config.num_hidden_layers  # Gemma-2-9B: 42
    svd_cb = SpectralMonitor(svd_log_path=svd_log_path,
                                svd_every_steps=args.svd_every_steps,
                                num_model_layers=num_layers,
                                resume=args.resume)
    metrics_cb = TrainingMetricsLogger(log_path=training_log_path,
                                       resume=args.resume)
    ppl_cb = PerLanguagePPLCallback(
        per_lang_val_datasets=per_lang_val_datasets,
        tokenizer=tokenizer,
        log_path=ppl_log_path,
        eval_every_steps=args.ppl_eval_every_steps,
        max_eval_samples=args.ppl_max_eval_samples,
        eval_batch_size=args.per_device_eval_batch_size,
        seed=args.seed,
        resume=args.resume,
    )

    # ── Training arguments (with eval) ──
    # In transfer mode, use warmup_steps instead of warmup_ratio
    warmup_kwargs = {}
    if args.init_adapter:
        warmup_kwargs["warmup_steps"] = args.transfer_warmup_steps
        print(f"[INFO] Transfer warmup: {args.transfer_warmup_steps} steps "
              f"(overrides warmup_ratio)")
    else:
        warmup_kwargs["warmup_ratio"] = args.warmup_ratio

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        **warmup_kwargs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_32bit",
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[svd_cb, metrics_cb, ppl_cb],
    )

    # ── Train ──
    print("\n" + "=" * 70)
    print("  START TRAINING")
    print("=" * 70 + "\n")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    total_time = time.time() - start_time

    # ── Final evaluation ──
    print("\n[INFO] Running final evaluation on validation set...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    eval_ppl = math.exp(eval_loss) if eval_loss and eval_loss < 20 else None
    if eval_ppl is not None:
        print(f"[INFO] Final eval_loss = {eval_loss:.4f}  |  eval_ppl = {eval_ppl:.2f}")
    else:
        print(f"[INFO] Final eval_loss = {eval_loss if eval_loss is not None else 'N/A'}")

    # ── Save adapter ──
    final_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[INFO] Adapter saved → {final_path}")

    # ── Save training metrics ──
    metrics = train_result.metrics
    metrics["total_training_time_sec"] = round(total_time, 2)
    metrics["total_training_time_min"] = round(total_time / 60, 2)
    metrics["total_tokens"] = total_tokens
    metrics["final_eval_loss"] = eval_loss
    metrics["final_eval_ppl"] = eval_ppl

    # Include collapse summary in saved metrics
    collapse_summary = svd_cb.get_collapse_summary()
    if collapse_summary:
        metrics["collapse_first_layer_index"] = collapse_summary["first_layer_index"]
        metrics["collapse_first_layer_zone"] = collapse_summary["first_layer_zone"]
        metrics["collapse_first_step"] = collapse_summary["first_step"]
        metrics["collapse_total_layers"] = collapse_summary["total_collapsed"]

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ── Generate plots ──
    print("\n[INFO] Generating plots...")
    plot_experiment_dashboard(training_log_path, svd_log_path, args.output_dir)
    plot_spectral_energy_detail(svd_log_path, args.output_dir, target_layers=(5, 12, 20))
    plot_ppl_per_lang(ppl_log_path, svd_log_path, args.output_dir)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time:       {total_time/60:.1f} min ({total_time/3600:.2f} h)")
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A')}")
    if metrics.get("train_loss") and metrics["train_loss"] < 20:
        print(f"  Final train PPL:  {math.exp(metrics['train_loss']):.2f}")
    print(f"  Final eval loss:  {eval_loss if eval_loss is not None else 'N/A'}")
    print(f"  Final eval PPL:   {f'{eval_ppl:.2f}' if eval_ppl is not None else 'N/A'}")
    print(f"  Total tokens:     {total_tokens:,}")

    # ── Collapse summary ──
    collapse_summary = svd_cb.get_collapse_summary()
    if collapse_summary:
        print("─" * 70)
        print("  SPECTRAL COLLAPSE SUMMARY:")
        first = collapse_summary
        print(f"    First collapse:  layer {first['first_layer_index']} "
              f"({first['first_layer_zone']}) at step {first['first_step']}")
        print(f"    First layer:     {first['first_layer_name']}")
        print(f"    Total collapsed: {first['total_collapsed']} layer(s) "
              f"(lower: {first['by_zone']['lower']}, upper: {first['by_zone']['upper']})")
        if first["total_collapsed"] > 1:
            print(f"    Last collapse:   step {first['last_step']}")
            print(f"    Collapse order (first 10):")
            for entry in first["layers"][:10]:
                idx_str = f"layer {entry['index']:2d}" if entry["index"] is not None else "layer  ?"
                print(f"      step {entry['step']:5d}  {idx_str} ({entry['zone']:5s})  {entry['name']}")
            if first["total_collapsed"] > 10:
                print(f"      ... and {first['total_collapsed'] - 10} more")
    else:
        print("─" * 70)
        print("  SPECTRAL COLLAPSE: None detected (all SE < 0.7)")

    print("─" * 70)
    print("  OUTPUT FILES:")
    print(f"    {args.output_dir}/")
    print(f"    ├── final_adapter/")
    print(f"    ├── config_dump.json")
    print(f"    ├── training_log.jsonl")
    print(f"    ├── svd_log.jsonl")
    print(f"    ├── ppl_per_lang_log.jsonl")
    print(f"    ├── train_results.json")
    print(f"    ├── experiment_dashboard.png")
    print(f"    ├── spectral_energy.png")
    print(f"    └── ppl_per_lang.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
