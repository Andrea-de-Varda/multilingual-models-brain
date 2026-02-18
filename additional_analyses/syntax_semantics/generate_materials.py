"""
Generate syntax/semantics dissociation materials for Pereira2018 and Tuckute2024.

For each sentence, produce:
  - 10 paraphrases:            same meaning, different syntactic structure
  - 10 syntactic alternatives:  same syntactic template, different meaning

Usage:
    python generate_materials.py --dataset pereira
    python generate_materials.py --dataset control
    python generate_materials.py --dataset pereira --max_sentences 5   # dry run
"""

import argparse
import json
import os
import pickle
import re
import time
import traceback

import pandas as pd
from together import Together
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = "MASKED"
MODEL   = "moonshotai/Kimi-K2.5"
N_VARIANTS = 10
TEMPERATURE = 0
MAX_TOKENS  = 1200
RETRY_MAX   = 3
RETRY_DELAY = 2.0   # seconds, doubled on each retry
CALL_DELAY  = 0.5   # seconds between calls

ROIS = [
    'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG',
    'lang_LH_AntTemp', 'lang_LH_PostTemp'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPT = """\
You are given a sentence. Generate {n} paraphrases that preserve the EXACT MEANING \
but use a DIFFERENT SYNTACTIC STRUCTURE each time.

Rules:
1. Preserve meaning completely – same events, same participants, same relations.
2. Change the grammatical construction substantially each time. Use a variety of \
constructions across the {n} paraphrases: active→passive, topicalization, cleft \
sentences ("It was X that..."), nominalization, existential constructions, relative \
clauses as main clause, etc.
3. Replace content words (nouns, verbs, adjectives, adverbs) with synonyms or \
equivalent expressions – minimize lexical overlap with the original.
4. Keep proper names (people, places) as they are – they are entity references, \
not arbitrary words.
5. Each sentence must be natural, fluent English.
6. Target length: within ±4 words of the original.

Examples of the syntactic variety expected:
  Original: "John broke the vase."
  → Passive:         "The vase was shattered by John."
  → Topicalization:  "The ceramic vessel, John ended up smashing."
  → Cleft:           "It was John who destroyed the vase."
  → Nominalization:  "John's shattering of the ceramic container was accidental."
  → Existential:     "There was an incident where John smashed the vessel."

Original sentence: <<{sentence}>>

Output ONLY a numbered list with exactly {n} items, one per line, no other text:
1.
2.
...
{n}.\
"""

SYNTACTIC_PROMPT = """\
You are given a sentence. Generate {n} new sentences that EXACTLY MIRROR ITS \
SYNTACTIC TEMPLATE but have COMPLETELY DIFFERENT MEANING.

Rules:
1. Mirror the syntactic template precisely:
   – Preserve the sentence type (active/passive, declarative/question, affirmative/negative).
   – Preserve the sequence and type of phrasal constituents: subject NP, main verb, \
object NP, prepositional phrases, subordinate/relative clauses, adverbs – in the same positions.
   – Preserve tense, aspect, and voice of the main verb.
   – Preserve function words (determiners, prepositions, conjunctions, auxiliaries) \
in the same structural roles.
2. Replace ALL content words (nouns, verbs, adjectives, adverbs) with completely \
different words referring to entirely different entities, actions, or properties.
3. Replace proper names with different proper names.
4. The {n} sentences must be natural, plausible, fluent English.
5. Target length: within ±2 words of the original.
6. The {n} sentences should differ from each other in content.

Examples:
  Original: "John saw a cat."
    Template: [ProperN] + [past transitive V] + [Det] + [concrete N]
  ✓ Good: "Maria lost a ring.",  "Tom built a shed.",  "Sara found a coin."
  ✗ Bad:  "A cat was seen by John."   (passive – wrong structure)
  ✗ Bad:  "John has seen many cats."  (different tense/number)

  Original: "The scientists discovered that the new compound was highly effective."
    Template: [Det+N] [past V] [that-clause: Det+Adj+N + past copula + Adv+Adj]
  ✓ Good: "The engineers confirmed that the old bridge was structurally unsound."
  ✓ Good: "The doctors noted that the experimental treatment was surprisingly successful."

Original sentence: <<{sentence}>>

Output ONLY a numbered list with exactly {n} items, one per line, no other text:
1.
2.
...
{n}.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sentences(dataset: str) -> list[str]:
    if dataset == "pereira":
        csv_path = os.path.join(SCRIPT_DIR, "../pereira/pereira_averaged.csv")
        df = pd.read_csv(csv_path)
        return df["Sentence"].tolist()
    elif dataset == "control":
        csv_path = os.path.join(
            SCRIPT_DIR,
            "../control/data/brain-lang-data_participant_20230728.csv"
        )
        control = pd.read_csv(csv_path)
        avg_1 = (
            control[control["roi"].isin(ROIS)]
            .groupby(["sentence", "target_UID"])
            .agg({"response_target": "mean", "cond": "first", "sentence": "first"})
            .reset_index(drop=True)
        )
        df = avg_1.groupby("sentence").agg({"response_target": "mean", "cond": "first"})
        df = df[df["cond"] == "B"]
        return df.index.tolist()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'pereira' or 'control'.")


def parse_numbered_list(text: str, n: int = N_VARIANTS) -> list[str]:
    """Extract items from a numbered list response."""
    pattern = r'^\s*\d+[.):\-]\s*(.+)$'
    items = re.findall(pattern, text, re.MULTILINE)
    # strip any residual markdown/quotes
    items = [re.sub(r'^["\']|["\']$', '', item.strip()) for item in items]
    return items[:n]


def call_api(client: Together, prompt: str) -> str:
    """Call Together API with retry logic."""
    for attempt in range(RETRY_MAX):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < RETRY_MAX - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    [API error, attempt {attempt+1}/{RETRY_MAX}] {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise


def load_checkpoint(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, ck: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, indent=2)
    os.replace(tmp, path)


def save_outputs(data: dict, out_dir: str, condition: str) -> None:
    """Save dict as pickle and CSV."""
    pkl_path = os.path.join(out_dir, f"{condition}_dict.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for idx, entry in data.items():
        row = {"sentence_idx": idx, "original_sentence": entry["sentence"]}
        for vi, v in enumerate(entry["variants"], start=1):
            row[f"v{vi}"] = v
        rows.append(row)

    csv_path = os.path.join(out_dir, f"{condition}s.csv")
    variant_cols = [f"v{i}" for i in range(1, N_VARIANTS + 1)]
    pd.DataFrame(rows)[["sentence_idx", "original_sentence"] + variant_cols].to_csv(
        csv_path, index=False
    )
    print(f"  Saved: {pkl_path}")
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(dataset: str, max_sentences: int | None = None) -> None:
    out_dir = os.path.join(SCRIPT_DIR, dataset, "materials")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "gen_checkpoint.json")
    checkpoint = load_checkpoint(ckpt_path)
    # checkpoint structure: {str(sent_idx): {"sentence": ..., "paraphrase": [...], "syntactic": [...]}}

    sentences = load_sentences(dataset)
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    print(f"Dataset: {dataset} | Sentences: {len(sentences)} | Model: {MODEL}")
    done = sum(
        1 for v in checkpoint.values()
        if len(v.get("paraphrase", [])) == N_VARIANTS
        and len(v.get("syntactic", [])) == N_VARIANTS
    )
    print(f"Already complete: {done}/{len(sentences)}")

    client = Together(api_key=API_KEY)

    conditions = {
        "paraphrase": PARAPHRASE_PROMPT,
        "syntactic":  SYNTACTIC_PROMPT,
    }

    for sent_idx, sentence in enumerate(tqdm(sentences, desc="Sentences")):
        key = str(sent_idx)

        # Check what's already done for this sentence
        ck_entry = checkpoint.get(key, {"sentence": sentence, "paraphrase": [], "syntactic": []})

        any_missing = False
        for cond_name, prompt_template in conditions.items():
            if len(ck_entry.get(cond_name, [])) == N_VARIANTS:
                continue  # already done

            any_missing = True
            prompt = prompt_template.format(sentence=sentence, n=N_VARIANTS)

            try:
                raw = call_api(client, prompt)
                variants = parse_numbered_list(raw, N_VARIANTS)

                if len(variants) < N_VARIANTS:
                    print(
                        f"\n  [Warning] sent {sent_idx} ({cond_name}): "
                        f"got {len(variants)}/{N_VARIANTS} items. Raw response:\n{raw[:300]}"
                    )

                ck_entry[cond_name] = variants
                ck_entry["sentence"] = sentence

            except Exception as e:
                print(f"\n  [Error] sent {sent_idx} ({cond_name}): {e}")
                traceback.print_exc()
                ck_entry[cond_name] = ck_entry.get(cond_name, [])

            time.sleep(CALL_DELAY)

        if any_missing:
            checkpoint[key] = ck_entry
            save_checkpoint(ckpt_path, checkpoint)

    # -----------------------------------------------------------------------
    # Save final outputs
    # -----------------------------------------------------------------------
    print("\nSaving final outputs...")
    para_data = {}
    synt_data = {}
    incomplete = []

    for sent_idx, sentence in enumerate(sentences):
        key = str(sent_idx)
        entry = checkpoint.get(key, {})

        para_vars = entry.get("paraphrase", [])
        synt_vars = entry.get("syntactic", [])

        if len(para_vars) < N_VARIANTS or len(synt_vars) < N_VARIANTS:
            incomplete.append(sent_idx)

        para_data[sent_idx] = {"sentence": sentence, "variants": para_vars}
        synt_data[sent_idx] = {"sentence": sentence, "variants": synt_vars}

    save_outputs(para_data, out_dir, "paraphrase")
    save_outputs(synt_data, out_dir, "syntactic")

    if incomplete:
        print(
            f"\n[Warning] {len(incomplete)} sentences have incomplete variants "
            f"(indices: {incomplete[:20]}{'...' if len(incomplete) > 20 else ''}). "
            f"Re-run to fill them in."
        )
    else:
        print(f"\nAll {len(sentences)} sentences complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate syntax/semantics materials.")
    parser.add_argument(
        "--dataset", required=True, choices=["pereira", "control"],
        help="Which dataset to process."
    )
    parser.add_argument(
        "--max_sentences", type=int, default=None,
        help="Cap number of sentences (for dry runs)."
    )
    args = parser.parse_args()
    generate(args.dataset, args.max_sentences)
