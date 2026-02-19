"""
Quality-check syntax/semantics materials in two steps:

  Step 1 (--step llm):
    LLM evaluates each variant; failing ones are auto-regenerated (up to 2 attempts).
    Outputs: check_results.csv, updated paraphrase_dict.pkl / syntactic_dict.pkl.

  Step 2 (--step overlap):
    Computes Jaccard word overlap between original sentences and their variants,
    then runs a Wilcoxon signed-rank test comparing the two conditions.
    Outputs: overlap_stats.csv.

  --step all runs both in sequence.

Usage:
    python check_materials.py --dataset pereira --step all
    python check_materials.py --dataset control  --step llm
    python check_materials.py --dataset pereira  --step overlap
"""

import argparse
import json
import os
import pickle
import re
import string
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from together import Together
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY    = "MASKED" # set your Together API key here
MODEL      = "moonshotai/Kimi-K2-Instruct-0905"
N_VARIANTS = 10
TEMPERATURE = 0
MAX_TOKENS  = 2000
RETRY_MAX   = 3
RETRY_DELAY = 2.0
REGEN_MAX   = 2          # max re-generation attempts per failing variant
BATCH_SIZE  = 5          # sentences per LLM quality-check call
DEFAULT_WORKERS = 20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Download NLTK stopwords once
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Quality-check prompt
# ---------------------------------------------------------------------------

CHECK_SYSTEM_MSG = (
    "You are a linguistics expert evaluating AI-generated sentence variants. "
    "Output ONLY valid JSON – no prose, no markdown fences."
)

CHECK_PROMPT_TEMPLATE = """\
Evaluate whether each generated variant meets its criteria.

Criteria for PARAPHRASE:
  (a) meaning_preserved  – conveys the same meaning as the original
  (b) syntax_changed     – uses a substantially different grammatical construction
  (c) fluent             – natural, grammatically correct English

Criteria for SYNTACTIC_ALTERNATIVE:
  (a) meaning_different  – meaning is clearly different from the original
  (b) syntax_preserved   – mirrors the syntactic template of the original
  (c) fluent             – natural, grammatically correct English

Items to evaluate:
{items_block}

Return a JSON array, one object per variant:
[
  {{
    "sentence_idx": <int>,
    "condition": "paraphrase" | "syntactic",
    "variant_idx": <int>,
    "passes": true | false,
    "failed_criteria": [],
    "note": "<brief explanation if fails, else empty string>"
  }},
  ...
]
"""

REGEN_PROMPT_TEMPLATE = """\
A previous attempt to generate a {condition_label} for the sentence below failed \
the quality check.

Original sentence: <<{original}>>

Failed criteria: {failed_criteria}
Reviewer note: {note}

Please generate ONE replacement that satisfies all the criteria.

Criteria for PARAPHRASE:
  • Same meaning, substantially different syntactic structure, minimal lexical overlap.
  • Proper names may be kept. Must be natural English.

Criteria for SYNTACTIC_ALTERNATIVE:
  • Same syntactic template (same phrase order, tense, voice), completely different \
meaning and content words. Must be natural English.

Output ONLY the replacement sentence, nothing else.\
"""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def out_dir_for(dataset: str) -> str:
    return os.path.join(SCRIPT_DIR, dataset, "materials")


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pkl_to_csv(data: dict, path: str) -> None:
    rows = []
    for idx, entry in data.items():
        row = {"sentence_idx": idx, "original_sentence": entry["sentence"]}
        for vi, v in enumerate(entry["variants"], start=1):
            row[f"v{vi}"] = v
        rows.append(row)
    cols = ["sentence_idx", "original_sentence"] + [f"v{i}" for i in range(1, N_VARIANTS + 1)]
    pd.DataFrame(rows)[cols].to_csv(path, index=False)


def load_checkpoint(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, ck: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def call_api(client: Together, messages: list[dict]) -> str:
    for attempt in range(RETRY_MAX):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < RETRY_MAX - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [API error attempt {attempt+1}/{RETRY_MAX}] {e}. Retry in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise


def parse_json_array(text: str) -> list[dict]:
    """Extract JSON array from model response, tolerating markdown fences."""
    text = re.sub(r'```(?:json)?', '', text).strip()
    # find first [ ... ] span
    start = text.find('[')
    end   = text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response:\n{text[:500]}")
    return json.loads(text[start:end+1])


# ---------------------------------------------------------------------------
# STEP 1 — LLM quality check + auto-regeneration
# ---------------------------------------------------------------------------

def build_check_items_block(batch: list[dict]) -> str:
    """
    batch: list of {sent_idx, original, condition, variants: [str]}
    Returns the formatted block inserted into CHECK_PROMPT_TEMPLATE.
    """
    lines = []
    for item in batch:
        cond_label = "PARAPHRASE" if item["condition"] == "paraphrase" else "SYNTACTIC_ALTERNATIVE"
        lines.append(f'Original (sentence_idx={item["sent_idx"]}): "{item["original"]}"')
        lines.append(f'Condition: {cond_label}')
        lines.append("Variants:")
        for vi, v in enumerate(item["variants"], start=1):
            lines.append(f"  {vi}. {v}")
        lines.append("")
    return "\n".join(lines)


def run_llm_check(client: Together, batch: list[dict]) -> list[dict]:
    items_block = build_check_items_block(batch)
    prompt = CHECK_PROMPT_TEMPLATE.format(items_block=items_block)
    messages = [
        {"role": "system", "content": CHECK_SYSTEM_MSG},
        {"role": "user",   "content": prompt},
    ]
    raw = call_api(client, messages)
    return parse_json_array(raw)


def regenerate_variant(
    client: Together, original: str, condition: str, failed_criteria: list[str], note: str
) -> str:
    cond_label = "paraphrase" if condition == "paraphrase" else "syntactic alternative"
    prompt = REGEN_PROMPT_TEMPLATE.format(
        condition_label=cond_label,
        original=original,
        failed_criteria=", ".join(failed_criteria) if failed_criteria else "unspecified",
        note=note or "no specific note",
    )
    messages = [{"role": "user", "content": prompt}]
    result = call_api(client, messages).strip()
    result = re.sub(r'^\s*\d+[.):\-]\s*', '', result).strip()
    return result


def _regen_one_variant(
    client: Together,
    sent_idx: int,
    vi: int,
    original: str,
    cond_name: str,
    initial_variant: str,
    initial_failed: list[str],
    initial_note: str,
) -> dict:
    """
    Attempt up to REGEN_MAX re-generations for a single failing variant.
    Returns a result dict with final_text, passes, regen_attempts, failed_criteria, note.
    """
    passes   = False
    failed   = initial_failed
    note     = initial_note
    final_text = initial_variant
    regen_attempts = 0

    for _ in range(REGEN_MAX):
        try:
            new_variant = regenerate_variant(client, original, cond_name, failed, note)
        except Exception as e:
            tqdm.write(f"  [Regen error] sent {sent_idx} v{vi+1}: {e}")
            break

        regen_attempts += 1

        mini_batch = [{
            "sent_idx": sent_idx,
            "original": original,
            "condition": cond_name,
            "variants": [new_variant],
        }]
        try:
            mini_results = run_llm_check(client, mini_batch)
            mr = mini_results[0] if mini_results else None
        except Exception:
            mr = None

        final_text = new_variant
        if mr and mr.get("passes", False):
            passes = True
            failed = []
            note   = ""
            break
        elif mr:
            failed = mr.get("failed_criteria", failed)
            note   = mr.get("note", note)

    return {
        "passes": passes,
        "failed_criteria": failed,
        "note": note,
        "regen_attempts": regen_attempts,
        "final_text": final_text,
        "needs_manual_review": not passes,
    }


def step_llm(dataset: str, workers: int = DEFAULT_WORKERS) -> None:
    out_dir = out_dir_for(dataset)
    ckpt_path = os.path.join(out_dir, "check_checkpoint.json")
    checkpoint = load_checkpoint(ckpt_path)
    ck_lock = threading.Lock()

    para_pkl = os.path.join(out_dir, "paraphrase_dict.pkl")
    synt_pkl = os.path.join(out_dir, "syntactic_dict.pkl")

    if not os.path.isfile(para_pkl) or not os.path.isfile(synt_pkl):
        raise FileNotFoundError(
            f"Missing pkl files in {out_dir}. Run generate_materials.py first."
        )

    para_data = load_pkl(para_pkl)
    synt_data = load_pkl(synt_pkl)

    client = Together(api_key=API_KEY)

    conditions = [
        ("paraphrase", para_data),
        ("syntactic",  synt_data),
    ]

    for cond_name, data in conditions:
        print(f"\n=== LLM check: {cond_name} | dataset: {dataset} | workers: {workers} ===")
        sent_indices = sorted(data.keys())

        # Build list of sentences needing checking
        to_check = [
            sent_idx for sent_idx in sent_indices
            if any(
                str(vi) not in checkpoint.get(str(sent_idx), {}).get(cond_name, {})
                for vi in range(len(data[sent_idx].get("variants", [])))
            )
        ]
        print(f"  Sentences needing first-pass check: {len(to_check)}/{len(sent_indices)}")

        # ---------------------------------------------------------------
        # PHASE 1: parallel first-pass batch checks
        # ---------------------------------------------------------------
        batches = []
        for batch_start in range(0, len(to_check), BATCH_SIZE):
            batch_indices = to_check[batch_start: batch_start + BATCH_SIZE]
            batch = [
                {
                    "sent_idx": idx,
                    "original": data[idx]["sentence"],
                    "condition": cond_name,
                    "variants": data[idx].get("variants", []),
                }
                for idx in batch_indices
            ]
            batches.append((batch_indices, batch))

        # Maps (sent_idx, vi) → first-pass eval result
        first_pass_results: dict[tuple, dict] = {}

        def _check_batch(batch_indices_and_batch):
            bidxs, bat = batch_indices_and_batch
            try:
                results = run_llm_check(client, bat)
                return bidxs, results, None
            except Exception as e:
                return bidxs, [], e

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_check_batch, b): b for b in batches}
            with tqdm(total=len(batches), desc="Check batches") as pbar:
                for future in as_completed(futures):
                    batch_indices, eval_results, error = future.result()
                    if error:
                        tqdm.write(f"  [Error] batch {batch_indices[:2]}: {error}")
                    else:
                        for r in eval_results:
                            key = (r.get("sentence_idx"), r.get("variant_idx", 1) - 1)
                            first_pass_results[key] = r
                    pbar.update(1)

        # ---------------------------------------------------------------
        # Populate checkpoint with first-pass results; collect failing variants
        # ---------------------------------------------------------------
        failing_tasks = []  # (sent_idx, vi, original, variant_text, failed, note)

        for sent_idx in to_check:
            key = str(sent_idx)
            entry = data[sent_idx]
            variants = entry.get("variants", [])
            with ck_lock:
                ck_entry = checkpoint.setdefault(key, {}).setdefault(cond_name, {})

            for vi, variant_text in enumerate(variants):
                vi_key = str(vi)
                if vi_key in ck_entry:
                    continue

                r = first_pass_results.get((sent_idx, vi))
                if r is None:
                    with ck_lock:
                        ck_entry[vi_key] = {
                            "passes": None,
                            "failed_criteria": ["response_missing"],
                            "note": "No eval returned by model",
                            "regen_attempts": 0,
                            "final_text": variant_text,
                            "needs_manual_review": True,
                        }
                    continue

                passes = r.get("passes", False)
                failed = r.get("failed_criteria", [])
                note   = r.get("note", "")

                if passes:
                    with ck_lock:
                        ck_entry[vi_key] = {
                            "passes": True, "failed_criteria": [], "note": "",
                            "regen_attempts": 0, "final_text": variant_text,
                            "needs_manual_review": False,
                        }
                else:
                    # Defer regeneration
                    failing_tasks.append((sent_idx, vi, entry["sentence"], variant_text, failed, note))

        with ck_lock:
            save_checkpoint(ckpt_path, checkpoint)

        # ---------------------------------------------------------------
        # PHASE 2: parallel regeneration across all failing variants
        # ---------------------------------------------------------------
        print(f"  Failing variants to regenerate: {len(failing_tasks)}")

        def _regen_task(args):
            sent_idx, vi, original, variant_text, failed, note = args
            result = _regen_one_variant(
                client, sent_idx, vi, original, cond_name, variant_text, failed, note
            )
            return sent_idx, vi, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_regen_task, t): t for t in failing_tasks}
            with tqdm(total=len(failing_tasks), desc="Regenerating") as pbar:
                for future in as_completed(futures):
                    sent_idx, vi, result = future.result()
                    key = str(sent_idx)
                    # Update the variant text in data
                    data[sent_idx]["variants"][vi] = result["final_text"]
                    with ck_lock:
                        checkpoint.setdefault(key, {}).setdefault(cond_name, {})[str(vi)] = result
                    pbar.update(1)

        with ck_lock:
            save_checkpoint(ckpt_path, checkpoint)

        # Persist updated pkl + CSV
        pkl_path = os.path.join(out_dir, f"{cond_name}_dict.pkl")
        csv_path = os.path.join(out_dir, f"{cond_name}s.csv")
        save_pkl(data, pkl_path)
        pkl_to_csv(data, csv_path)
        print(f"  Updated {pkl_path}")

    # -----------------------------------------------------------------
    # Save check_results.csv
    # -----------------------------------------------------------------
    rows = []
    for sent_idx in sorted(para_data.keys()):
        key = str(sent_idx)
        original = para_data[sent_idx]["sentence"]
        for cond_name, data in conditions:
            variants = data[sent_idx].get("variants", [])
            ck_cond  = checkpoint.get(key, {}).get(cond_name, {})
            for vi, variant_text in enumerate(variants):
                vi_key = str(vi)
                ev = ck_cond.get(vi_key, {})
                rows.append({
                    "sentence_idx":     sent_idx,
                    "original":         original,
                    "condition":        cond_name,
                    "variant_idx":      vi + 1,
                    "variant_text":     ev.get("final_text", variant_text),
                    "final_passes":     ev.get("passes", None),
                    "regen_attempts":   ev.get("regen_attempts", 0),
                    "failed_criteria":  "|".join(ev.get("failed_criteria", [])),
                    "note":             ev.get("note", ""),
                    "needs_manual_review": ev.get("needs_manual_review", False),
                })

    results_path = os.path.join(out_dir, "check_results.csv")
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    needs_review = sum(1 for r in rows if r["needs_manual_review"])
    print(f"Variants needing manual review: {needs_review}/{len(rows)}")


# ---------------------------------------------------------------------------
# STEP 2 — Word overlap statistical check
# ---------------------------------------------------------------------------

def tokenize(text: str, remove_stopwords: bool = False) -> set[str]:
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def compute_sentence_overlap(sentence: str, variants: list[str], remove_sw: bool) -> float:
    orig_toks = tokenize(sentence, remove_stopwords=remove_sw)
    jaccards  = [jaccard(orig_toks, tokenize(v, remove_stopwords=remove_sw)) for v in variants if v]
    return float(np.mean(jaccards)) if jaccards else np.nan


def step_overlap(dataset: str) -> None:
    out_dir = out_dir_for(dataset)

    para_pkl = os.path.join(out_dir, "paraphrase_dict.pkl")
    synt_pkl = os.path.join(out_dir, "syntactic_dict.pkl")

    if not os.path.isfile(para_pkl) or not os.path.isfile(synt_pkl):
        raise FileNotFoundError(
            f"Missing pkl files in {out_dir}. Run generate_materials.py (and optionally check_materials.py --step llm) first."
        )

    para_data = load_pkl(para_pkl)
    synt_data = load_pkl(synt_pkl)

    rows = []
    common_indices = sorted(set(para_data.keys()) & set(synt_data.keys()))

    for sent_idx in common_indices:
        sentence      = para_data[sent_idx]["sentence"]
        para_variants = para_data[sent_idx].get("variants", [])
        synt_variants = synt_data[sent_idx].get("variants", [])

        if not para_variants or not synt_variants:
            continue

        row = {"sentence_idx": sent_idx, "original": sentence}
        for remove_sw, suffix in [(False, "all"), (True, "content")]:
            row[f"para_jaccard_{suffix}"]  = compute_sentence_overlap(sentence, para_variants, remove_sw)
            row[f"synt_jaccard_{suffix}"]  = compute_sentence_overlap(sentence, synt_variants, remove_sw)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Statistical tests
    print(f"\n=== Word overlap check | dataset: {dataset} ===\n")
    print(f"Sentences compared: {len(df)}\n")

    for suffix, label in [("all", "All words"), ("content", "Content words (no stopwords)")]:
        col_p = f"para_jaccard_{suffix}"
        col_s = f"synt_jaccard_{suffix}"
        valid = df[[col_p, col_s]].dropna()

        if valid.empty:
            print(f"[{label}] No valid pairs.")
            continue

        p_vals = valid[col_p].values
        s_vals = valid[col_s].values

        print(f"[{label}]")
        print(f"  Paraphrase  Jaccard: mean={p_vals.mean():.4f}  SD={p_vals.std(ddof=1):.4f}")
        print(f"  Syntactic   Jaccard: mean={s_vals.mean():.4f}  SD={s_vals.std(ddof=1):.4f}")

        if len(valid) >= 10 and not np.all(p_vals - s_vals == 0):
            stat, pval = wilcoxon(p_vals, s_vals)
            print(f"  Wilcoxon signed-rank: W={stat:.1f}, p={pval:.4f}")
            if pval < 0.05:
                print(f"  *** Overlap distributions differ significantly (p<0.05) – inspect manually. ***")
            else:
                print(f"  Distributions not significantly different (p≥0.05). ✓")
        else:
            print(f"  (Too few pairs for Wilcoxon test, or zero differences)")
        print()

    # Append stat columns to df for output
    overlap_path = os.path.join(out_dir, "overlap_stats.csv")
    df.to_csv(overlap_path, index=False)
    print(f"Saved: {overlap_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality-check syntax/semantics materials.")
    parser.add_argument(
        "--dataset", required=True, choices=["pereira", "control"],
        help="Which dataset to check."
    )
    parser.add_argument(
        "--step", required=True, choices=["llm", "overlap", "all"],
        help="Which check step to run."
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel API requests (default: {DEFAULT_WORKERS})."
    )
    args = parser.parse_args()

    if args.step in ("llm", "all"):
        step_llm(args.dataset, args.workers)
    if args.step in ("overlap", "all"):
        step_overlap(args.dataset)
