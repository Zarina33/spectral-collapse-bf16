#!/usr/bin/env python3
"""
evaluate.py — Post-training evaluation for Turkic LoRA adapters
================================================================
Evaluates a fine-tuned LoRA adapter on three downstream tasks:

  1. Per-language perplexity (KY, KZ, UZ held-out sets)
  2. NER via WikiANN (few-shot prompting + generation)
  3. TUMLU QA benchmark (log-likelihood multiple-choice)

Output:
    output/
    └── eval_report.json    # Full evaluation results

Usage:
    python scripts/evaluate.py --adapter_path ./output/final_adapter
    python scripts/evaluate.py --skip_ner --skip_tumlu   # PPL only
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)


# ════════════════════════════════════════════════════════════════════
# 1.  CLI Arguments
# ════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Post-training evaluation for Turkic LoRA adapters")

    # Model
    p.add_argument("--model_name", type=str, default="google/gemma-2-9b")
    p.add_argument("--adapter_path", type=str, default="./output/final_adapter")
    p.add_argument("--bf16", action="store_true", default=True)

    # Data for PPL (must match train_svd.py defaults for reproducible split)
    p.add_argument("--data_dir", type=str, default="./data/pretrain")
    p.add_argument("--ky_file", type=str, default="kyrgyz_raw.jsonl")
    p.add_argument("--kz_file", type=str, default="kazakh_raw.jsonl")
    p.add_argument("--uz_file", type=str, default="uzbek_final_cyrillic.jsonl")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    # Evaluation controls
    p.add_argument("--skip_ppl", action="store_true", help="Skip perplexity eval")
    p.add_argument("--skip_ner", action="store_true", help="Skip NER eval")
    p.add_argument("--skip_ner_loglik", action="store_true",
                   help="Skip log-likelihood span-typing NER eval")
    p.add_argument("--skip_tumlu", action="store_true", help="Skip TUMLU eval")
    p.add_argument("--ner_shots", type=int, default=3)
    p.add_argument("--ner_max_samples", type=int, default=100)
    p.add_argument("--tumlu_shots", type=int, default=5)
    p.add_argument("--ppl_max_samples", type=int, default=500)
    p.add_argument("--ppl_batch_size", type=int, default=1)

    # Output
    p.add_argument("--output_dir", type=str, default="./output")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════
# 2.  Model Loading
# ════════════════════════════════════════════════════════════════════
def load_model_and_adapter(model_name: str, adapter_path: str,
                           bf16: bool = True):
    """Load base Gemma-2-9B with 4-bit quantization and merge LoRA adapter."""
    from peft import PeftModel

    print(f"[INFO] Loading base model: {model_name}")
    print(f"[INFO] Loading adapter from: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total:,} total, {trainable:,} trainable "
          f"({100*trainable/total:.2f}%)")

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════
# 3.  Per-Language Perplexity
# ════════════════════════════════════════════════════════════════════
def evaluate_perplexity(model, tokenizer, data_dir: str, lang_files: dict,
                        max_seq_length: int = 512, max_samples: int = 500,
                        batch_size: int = 4, val_split: float = 0.10,
                        seed: int = 42) -> dict:
    """
    Evaluate per-language perplexity on held-out data.
    Uses the same seed and val_split as train_svd.py to get identical splits.
    """
    print("\n" + "─" * 60)
    print("  PERPLEXITY EVALUATION")
    print("─" * 60)

    # Reproduce the exact same data pipeline as train_svd.py
    # IMPORTANT: iteration order must match train_svd.py (ky → kz → uz)
    # to produce identical shuffle + split with the same seed.
    loaded = []
    for lang in ["ky", "kz", "uz"]:
        fname = lang_files.get(lang)
        if fname is None:
            continue
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  [WARN] {fpath} not found, skipping {lang}")
            continue
        ds = load_dataset("json", data_files=fpath, split="train")
        ds = ds.add_column("lang", [lang] * len(ds))
        loaded.append(ds)

    if not loaded:
        print("  [ERROR] No data files found.")
        return {}

    # Same concatenation + shuffle + tokenize + split as training
    dataset = concatenate_datasets(loaded).shuffle(seed=seed)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"], truncation=True,
            max_length=max_seq_length, padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=[c for c in dataset.column_names if c != "lang"],
        num_proc=os.cpu_count(), desc="Tokenizing")

    split = tokenized.train_test_split(test_size=val_split, seed=seed)
    val_ds = split["test"]
    print(f"  Val set size: {len(val_ds):,}")

    pad_id = tokenizer.pad_token_id
    def clm_collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            ids = f["input_ids"]
            pad_len = max_len - len(ids)
            batch["input_ids"].append(ids + [pad_id] * pad_len)
            batch["attention_mask"].append([1] * len(ids) + [0] * pad_len)
            batch["labels"].append(ids + [-100] * pad_len)
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

    results = {}

    for lang in lang_files.keys():
        lang_subset = val_ds.filter(lambda x: x["lang"] == lang)
        if len(lang_subset) == 0:
            continue

        lang_clean = lang_subset.remove_columns(["lang"])
        if len(lang_clean) > max_samples:
            lang_clean = lang_clean.shuffle(seed=seed).select(range(max_samples))

        loader = torch.utils.data.DataLoader(
            lang_clean, batch_size=batch_size,
            collate_fn=clm_collator, shuffle=False)

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

        results[lang] = {
            "loss": round(avg_loss, 6),
            "ppl": round(ppl, 4) if ppl != float("inf") else None,
            "n_samples": len(lang_clean),
            "n_tokens": total_tokens,
        }
        print(f"  [{lang}] loss={avg_loss:.4f}  ppl={ppl:.2f}  "
              f"({len(lang_clean)} samples, {total_tokens:,} tokens)")

    return results


# ════════════════════════════════════════════════════════════════════
# 4.  NER Evaluation (WikiANN, few-shot prompting)
# ════════════════════════════════════════════════════════════════════
TAG_MAP = {1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC"}


def wikiann_to_spans(tokens: list, ner_tags: list) -> list:
    """Convert IOB2 tags to (type, entity_text) tuples."""
    spans = []
    current_type = None
    current_tokens = []

    for token, tag in zip(tokens, ner_tags):
        if tag in (1, 3, 5):  # B-* tags
            if current_type:
                spans.append((current_type, " ".join(current_tokens)))
            current_type = TAG_MAP[tag]
            current_tokens = [token]
        elif tag in (2, 4, 6):  # I-* tags
            if current_type == TAG_MAP.get(tag):
                current_tokens.append(token)
            else:
                if current_type:
                    spans.append((current_type, " ".join(current_tokens)))
                current_type = TAG_MAP.get(tag)
                current_tokens = [token]
        else:  # O tag
            if current_type:
                spans.append((current_type, " ".join(current_tokens)))
                current_type = None
                current_tokens = []

    if current_type:
        spans.append((current_type, " ".join(current_tokens)))
    return spans


def format_ner_example(tokens: list, spans: list) -> str:
    """Format a single NER example for few-shot prompt."""
    text = " ".join(tokens)
    if not spans:
        entities = "NONE"
    else:
        entities = "\n".join(f"[{t}] {e}" for t, e in spans)
    return f"Text: {text}\nEntities:\n{entities}"


def parse_ner_output(text: str) -> list:
    """Parse model output for NER entities."""
    entities = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper() == "NONE":
            break
        # Try [TYPE] entity format
        m = re.match(r"\[(PER|ORG|LOC)\]\s*(.+)", line)
        if m:
            entities.append((m.group(1), m.group(2).strip()))
            continue
        # Try TYPE: entity format
        m = re.match(r"(PER|ORG|LOC):\s*(.+)", line)
        if m:
            entities.append((m.group(1), m.group(2).strip()))
    return entities


def compute_ner_f1(gold_spans: list, pred_spans: list) -> dict:
    """Compute entity-level precision, recall, F1 (exact match)."""
    gold_set = set((t, e.lower().strip()) for t, e in gold_spans)
    pred_set = set((t, e.lower().strip()) for t, e in pred_spans)

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": len(pred_set) - tp, "fn": len(gold_set) - tp}


def evaluate_ner(model, tokenizer, languages=("ky", "kz", "uz"),
                 n_shots: int = 3, max_eval_samples: int = 100,
                 max_new_tokens: int = 128) -> dict:
    """Evaluate NER via few-shot prompting on WikiANN test sets."""
    print("\n" + "─" * 60)
    print("  NER EVALUATION  (WikiANN, few-shot prompting)")
    print("─" * 60)

    # WikiANN uses ISO 639-1 codes; map project codes to WikiANN codes
    WIKIANN_LANG_MAP = {"ky": "ky", "kz": "kk", "uz": "uz"}

    NER_INSTRUCTION = (
        "Extract named entities from the text. "
        "Classify each as PER (person), ORG (organization), or LOC (location).\n"
        "Format: [TYPE] entity_name (one per line). If none, write: NONE\n\n"
    )

    results = {}

    for lang in languages:
        wikiann_lang = WIKIANN_LANG_MAP.get(lang, lang)
        print(f"\n  [{lang}] Loading WikiANN ({wikiann_lang})...")
        try:
            ds = load_dataset("unimelb-nlp/wikiann", wikiann_lang)
        except Exception as e:
            print(f"  [{lang}] WikiANN ({wikiann_lang}) not available: {e}")
            continue

        train_split = ds.get("train")
        test_split = ds.get("test")
        if test_split is None or len(test_split) == 0:
            print(f"  [{lang}] No test split available.")
            continue

        # Build few-shot examples from train
        few_shot_prefix = NER_INSTRUCTION
        if train_split and len(train_split) >= n_shots:
            for i in range(n_shots):
                ex = train_split[i]
                spans = wikiann_to_spans(ex["tokens"], ex["ner_tags"])
                few_shot_prefix += format_ner_example(ex["tokens"], spans) + "\n\n"

        # Evaluate on test
        n_eval = min(max_eval_samples, len(test_split))
        print(f"  [{lang}] Evaluating on {n_eval} test examples...")

        all_gold = []
        all_pred = []
        parse_failures = 0

        for idx in range(n_eval):
            ex = test_split[idx]
            gold_spans = wikiann_to_spans(ex["tokens"], ex["ner_tags"])
            text = " ".join(ex["tokens"])

            prompt = few_shot_prefix + f"Text: {text}\nEntities:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024 - max_new_tokens)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False)

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)

            pred_spans = parse_ner_output(generated)
            if not pred_spans and gold_spans:
                parse_failures += 1

            all_gold.append(gold_spans)
            all_pred.append(pred_spans)

        # Aggregate metrics
        total_tp = total_fp = total_fn = 0
        per_type = {}
        for gold, pred in zip(all_gold, all_pred):
            m = compute_ner_f1(gold, pred)
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]

            for etype in ["PER", "ORG", "LOC"]:
                gold_t = [(t, e) for t, e in gold if t == etype]
                pred_t = [(t, e) for t, e in pred if t == etype]
                m_t = compute_ner_f1(gold_t, pred_t)
                if etype not in per_type:
                    per_type[etype] = {"tp": 0, "fp": 0, "fn": 0}
                per_type[etype]["tp"] += m_t["tp"]
                per_type[etype]["fp"] += m_t["fp"]
                per_type[etype]["fn"] += m_t["fn"]

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        per_type_results = {}
        for etype, counts in per_type.items():
            p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0
            r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0
            f = 2 * p * r / (p + r) if (p + r) else 0
            per_type_results[etype] = {"precision": round(p, 4),
                                       "recall": round(r, 4), "f1": round(f, 4)}

        results[lang] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "n_evaluated": n_eval,
            "parse_failures": parse_failures,
            "per_type": per_type_results,
        }
        print(f"  [{lang}] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"(parse_fail={parse_failures}/{n_eval})")

    return results


# ════════════════════════════════════════════════════════════════════
# 4b. NER via log-likelihood span typing (parse-failure immune)
# ════════════════════════════════════════════════════════════════════
NER_TYPE_LABELS = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
}


def evaluate_ner_loglik(model, tokenizer, languages=("ky", "kz", "uz"),
                        max_eval_samples: int = 100) -> dict:
    """Log-likelihood based NER: classify gold entity spans by type.

    For each gold span, score the continuation "person"/"organization"/"location"
    under a uniform prompt and pick argmax. This decouples type knowledge from
    the model's instruction-following / output-formatting ability, which is the
    dominant failure mode for the r=64 configuration under generation-based NER.
    """
    import torch
    print("\n" + "─" * 60)
    print("  NER EVALUATION  (log-likelihood span typing)")
    print("─" * 60)

    WIKIANN_LANG_MAP = {"ky": "ky", "kz": "kk", "uz": "uz"}
    results = {}

    for lang in languages:
        wikiann_lang = WIKIANN_LANG_MAP.get(lang, lang)
        print(f"\n  [{lang}] Loading WikiANN ({wikiann_lang})...")
        try:
            ds = load_dataset("unimelb-nlp/wikiann", wikiann_lang)
        except Exception as e:
            print(f"  [{lang}] WikiANN ({wikiann_lang}) not available: {e}")
            continue

        test_split = ds.get("test")
        if test_split is None or len(test_split) == 0:
            continue

        n_eval = min(max_eval_samples, len(test_split))
        print(f"  [{lang}] Classifying spans from {n_eval} examples...")

        per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in NER_TYPE_LABELS}
        total = 0
        correct = 0

        for idx in range(n_eval):
            ex = test_split[idx]
            gold_spans = wikiann_to_spans(ex["tokens"], ex["ner_tags"])
            if not gold_spans:
                continue

            text = " ".join(ex["tokens"])
            for gold_type, span_text in gold_spans:
                context = (f"Text: {text}\n"
                           f"The entity '{span_text}' is a")
                scores = {}
                for t, label in NER_TYPE_LABELS.items():
                    scores[t] = compute_choice_loglikelihood(
                        model, tokenizer, context, " " + label)
                pred_type = max(scores, key=scores.get)

                total += 1
                if pred_type == gold_type:
                    correct += 1
                    per_type[gold_type]["tp"] += 1
                else:
                    per_type[gold_type]["fn"] += 1
                    per_type[pred_type]["fp"] += 1

        accuracy = correct / total if total > 0 else 0.0

        per_type_results = {}
        for t, c in per_type.items():
            p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0
            r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
            f = 2 * p * r / (p + r) if (p + r) else 0
            per_type_results[t] = {"precision": round(p, 4),
                                   "recall": round(r, 4), "f1": round(f, 4),
                                   "support": c["tp"] + c["fn"]}

        macro_f1 = sum(v["f1"] for v in per_type_results.values()) / 3.0

        results[lang] = {
            "method": "loglik_span_typing",
            "type_accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "n_spans": total,
            "per_type": per_type_results,
        }
        print(f"  [{lang}] type_acc={accuracy:.3f}  macro_F1={macro_f1:.3f}  "
              f"(n_spans={total})")

    return results


# ════════════════════════════════════════════════════════════════════
# 5.  TUMLU QA Evaluation (log-likelihood multiple-choice)
# ════════════════════════════════════════════════════════════════════
TUMLU_LANG_MAP = {
    "kazakh": "kz",
    "uzbek": "uz",
    "kyrgyz": "ky",
}

TUMLU_PROMPT = {
    "kazakh": "Сұрақ: {question}\n{choices}\n\nЖауап:",
    "uzbek": "Савол: {question}\n{choices}\n\nЖавоб:",
    "kyrgyz": "Суроо: {question}\n{choices}\n\nЖооп:",
}


def compute_choice_loglikelihood(model, tokenizer, context: str,
                                 choice: str) -> float:
    """Compute log P(choice | context) for a causal LM."""
    # Encode separately and concatenate to know exact boundary
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    choice_ids = tokenizer.encode(choice, add_special_tokens=False)

    if not choice_ids:
        return float("-inf")

    full_ids = context_ids + choice_ids
    choice_start = len(context_ids)

    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(choice_start, len(full_ids)):
        token_id = full_ids[i]
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob


def format_tumlu_question(question: str, choices: list, lang: str) -> str:
    """Format a single TUMLU question."""
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{labels[i]}) {c}" for i, c in enumerate(choices) if i < len(labels))
    template = TUMLU_PROMPT.get(lang,
        "Question: {question}\n{choices}\n\nAnswer:")
    return template.format(question=question, choices=choices_text)


def evaluate_tumlu(model, tokenizer,
                   languages=("kazakh", "uzbek", "kyrgyz"),
                   n_shots: int = 5) -> dict:
    """Evaluate TUMLU-mini using log-likelihood comparison."""
    print("\n" + "─" * 60)
    print("  TUMLU QA EVALUATION  (log-likelihood)")
    print("─" * 60)

    results = {}

    for lang in languages:
        local = TUMLU_LANG_MAP.get(lang, lang)
        print(f"\n  [{local}] Loading TUMLU-mini ({lang})...")

        try:
            ds = load_dataset("jafarisbarov/TUMLU-mini", lang, split="test")
        except Exception as e:
            print(f"  [{local}] TUMLU-mini not available for {lang}: {e}")
            continue

        if len(ds) == 0:
            print(f"  [{local}] Empty dataset, skipping.")
            continue

        # Determine column names (TUMLU format varies)
        cols = ds.column_names
        q_col = "question" if "question" in cols else cols[0]
        c_col = "choices" if "choices" in cols else None
        a_col = "answer" if "answer" in cols else None

        if c_col is None or a_col is None:
            print(f"  [{local}] Unexpected columns: {cols}, skipping.")
            continue

        # Few-shot prefix (ensure at least 1 eval example remains)
        few_shot_prefix = ""
        n_fs = min(n_shots, max(len(ds) - 1, 0))
        if n_fs == 0 and len(ds) == 0:
            print(f"  [{local}] Empty test set, skipping.")
            continue

        for i in range(n_fs):
            q = ds[i][q_col]
            c = ds[i][c_col]
            a = ds[i][a_col]
            prompt = format_tumlu_question(q, c, lang)
            few_shot_prefix += prompt + f" {a}\n\n"

        # Evaluate
        eval_start = n_fs
        if eval_start >= len(ds):
            print(f"  [{local}] Only {len(ds)} examples, all used for few-shot. Skipping.")
            continue

        correct = 0
        total = 0
        print(f"  [{local}] Evaluating {len(ds) - eval_start} questions "
              f"({n_fs}-shot)...")

        for idx in range(eval_start, len(ds)):
            q = ds[idx][q_col]
            c = ds[idx][c_col]
            gold = ds[idx][a_col]

            context = few_shot_prefix + format_tumlu_question(q, c, lang)

            choice_labels = ["A", "B", "C", "D"]
            log_probs = {}
            for i, label in enumerate(choice_labels[:len(c)]):
                # Compute log P(choice_text | context), not log P(label_letter)
                ll = compute_choice_loglikelihood(
                    model, tokenizer, context, f" {c[i]}")
                log_probs[label] = ll

            predicted = max(log_probs, key=log_probs.get)
            if predicted == gold:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        results[lang] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "n_questions": total,
        }
        print(f"  [{local}] Accuracy: {accuracy:.1%} ({correct}/{total})")

    return results


# ════════════════════════════════════════════════════════════════════
# 6.  Report Generation
# ════════════════════════════════════════════════════════════════════
def generate_report(ppl_results: dict, ner_results: dict,
                    tumlu_results: dict, adapter_path: str,
                    output_dir: str,
                    ner_loglik_results: dict = None) -> dict:
    """Generate JSON evaluation report and print summary table."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "adapter_path": adapter_path,
        "perplexity": ppl_results,
        "ner_wikiann": ner_results,
        "ner_loglik": ner_loglik_results or {},
        "tumlu_qa": tumlu_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print("\n" + "=" * 72)
    print("  EVALUATION REPORT")
    print("=" * 72)

    if ppl_results:
        print("\n  Per-Language Perplexity:")
        print(f"  {'Lang':<8} {'Loss':>10} {'PPL':>10} {'Samples':>10}")
        print("  " + "─" * 40)
        for lang in ["ky", "kz", "uz"]:
            if lang in ppl_results:
                r = ppl_results[lang]
                ppl_str = f"{r['ppl']:.2f}" if r['ppl'] else "inf"
                print(f"  {lang:<8} {r['loss']:>10.4f} {ppl_str:>10} "
                      f"{r['n_samples']:>10}")

    if ner_results:
        print("\n  NER (WikiANN) — Entity-level F1:")
        print(f"  {'Lang':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}")
        print("  " + "─" * 40)
        for lang in ["ky", "kz", "uz"]:
            if lang in ner_results:
                r = ner_results[lang]
                print(f"  {lang:<8} {r['precision']:>8.1%} {r['recall']:>8.1%} "
                      f"{r['f1']:>8.1%} {r['n_evaluated']:>6}")

    if ner_loglik_results:
        print("\n  NER (log-likelihood span typing) — parse-failure immune:")
        print(f"  {'Lang':<8} {'TypeAcc':>10} {'MacroF1':>10} {'N':>6}")
        print("  " + "─" * 40)
        for lang in ["ky", "kz", "uz"]:
            if lang in ner_loglik_results:
                r = ner_loglik_results[lang]
                print(f"  {lang:<8} {r['type_accuracy']:>10.1%} "
                      f"{r['macro_f1']:>10.3f} {r['n_spans']:>6}")

    if tumlu_results:
        print("\n  TUMLU QA — Accuracy:")
        print(f"  {'Lang':<8} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print("  " + "─" * 40)
        for tumlu_lang, local_lang in TUMLU_LANG_MAP.items():
            if tumlu_lang in tumlu_results:
                r = tumlu_results[tumlu_lang]
                print(f"  {local_lang:<8} {r['accuracy']:>10.1%} "
                      f"{r['correct']:>10} {r['n_questions']:>10}")

    print("\n" + "=" * 72)
    print(f"  Full report saved → {report_path}")
    print("=" * 72)

    return report


# ════════════════════════════════════════════════════════════════════
# 7.  Main
# ════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 72)
    print("  Post-Training Evaluation — Turkic LoRA")
    print("=" * 72)

    # ── Load model ──
    model, tokenizer = load_model_and_adapter(
        args.model_name, args.adapter_path, args.bf16)

    ppl_results = {}
    ner_results = {}
    ner_loglik_results = {}
    tumlu_results = {}

    # ── 1. Per-language perplexity ──
    if not args.skip_ppl:
        try:
            lang_files = {"ky": args.ky_file, "kz": args.kz_file, "uz": args.uz_file}
            ppl_results = evaluate_perplexity(
                model, tokenizer, args.data_dir, lang_files,
                max_seq_length=args.max_seq_length,
                max_samples=args.ppl_max_samples,
                batch_size=args.ppl_batch_size,
                val_split=args.val_split,
                seed=args.seed,
            )
        except Exception as e:
            print(f"\n  [ERROR] PPL evaluation failed: {e}")

    # ── 2. NER (few-shot generation) ──
    if not args.skip_ner:
        try:
            ner_results = evaluate_ner(
                model, tokenizer,
                languages=["ky", "kz", "uz"],
                n_shots=args.ner_shots,
                max_eval_samples=args.ner_max_samples,
            )
        except Exception as e:
            print(f"\n  [ERROR] NER evaluation failed: {e}")

    # ── 2b. NER log-likelihood span typing ──
    if not args.skip_ner_loglik:
        try:
            ner_loglik_results = evaluate_ner_loglik(
                model, tokenizer,
                languages=["ky", "kz", "uz"],
                max_eval_samples=args.ner_max_samples,
            )
        except Exception as e:
            print(f"\n  [ERROR] NER log-likelihood evaluation failed: {e}")

    # ── 3. TUMLU QA ──
    if not args.skip_tumlu:
        try:
            tumlu_results = evaluate_tumlu(
                model, tokenizer,
                languages=["kazakh", "uzbek", "kyrgyz"],
                n_shots=args.tumlu_shots,
            )
        except Exception as e:
            print(f"\n  [ERROR] TUMLU evaluation failed: {e}")

    # ── Report ──
    generate_report(ppl_results, ner_results, tumlu_results,
                    args.adapter_path, args.output_dir,
                    ner_loglik_results=ner_loglik_results)

    elapsed = time.time() - start_time
    print(f"\n  Total evaluation time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
