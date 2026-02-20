"""
Prepares sentence-level feature matrices for INLP residualization analysis.

Output:
  features/features_semantics.csv  — 7 semantic rating features
  features/features_syntax.csv     — 7 syntactic/form features

Run locally with conda env 'analysis'.
"""

import os
import pandas as pd
import numpy as np
import string
import spacy
import wordfreq
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("features", exist_ok=True)

# ──────────────────────────────────────────────
# Load and filter control data (same filter as perturbation_encoding.py)
# ──────────────────────────────────────────────
CONTROL_CSV = "../control/data/brain-lang-data_participant_20230728.csv"
ROIS = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp']

control = pd.read_csv(CONTROL_CSV)
avg_1 = (control[control["roi"].isin(ROIS)]
         .groupby(["sentence", "target_UID"])
         .agg({"response_target": "mean", "cond": "first", "sentence": "first"})
         .reset_index(drop=True))
df = avg_1.groupby("sentence").agg({"response_target": "mean", "cond": "first"})
df = df[df["cond"] == "B"]
sentences = df.index.tolist()
print(f"Number of training sentences (cond=B): {len(sentences)}")

# ──────────────────────────────────────────────
# Aggregate ratings at sentence level from raw CSV
# (take mean across participants/ROIs for rating columns)
# ──────────────────────────────────────────────
SEMANTIC_COLS = [
    "rating_imageability_mean",
    "rating_others_thoughts_mean",
    "rating_physical_mean",
    "rating_places_mean",
    "rating_valence_mean",
    "rating_arousal_mean",
    "rating_sense_mean",
]

SYNTAX_CSV_COLS = [
    "log-prob-gpt2-xl_mean",
    "rating_gram_mean",
    "rating_frequency_mean",
    "rating_conversational_mean",
]

all_rating_cols = SEMANTIC_COLS + SYNTAX_CSV_COLS

# Aggregate at sentence level: these columns vary by sentence (not by participant),
# so taking mean is equivalent to taking any value, but mean is safe.
df_ratings = (control[control["cond"] == "B"]
              .groupby("sentence")[all_rating_cols]
              .mean())

# Align to our sentence list
df_ratings = df_ratings.loc[sentences]
assert list(df_ratings.index) == sentences, "Sentence alignment mismatch in ratings"
print("Rating columns aligned. NaN counts:")
print(df_ratings.isna().sum())

# ──────────────────────────────────────────────
# Compute spaCy + wordfreq features per sentence
# ──────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

dep_lengths = []
avg_word_freqs = []
avg_word_lengths = []

for sent in tqdm(sentences, desc="spaCy + wordfreq"):
    doc = nlp(sent)

    # Dependency length: mean |token.i - token.head.i| per sentence
    if len(doc) > 0:
        dl = np.mean([abs(tok.i - tok.head.i) for tok in doc])
    else:
        dl = 0.0
    dep_lengths.append(dl)

    # Word-level features: exclude punctuation tokens
    words = [tok.text for tok in doc if not tok.is_punct and tok.text.strip()]
    if words:
        freqs = [wordfreq.zipf_frequency(w, 'en') for w in words]
        avg_wf = np.mean(freqs)
        avg_wl = np.mean([len(w) for w in words])
    else:
        avg_wf = 0.0
        avg_wl = 0.0
    avg_word_freqs.append(avg_wf)
    avg_word_lengths.append(avg_wl)

computed_syntax = pd.DataFrame({
    "dep_length_mean":     dep_lengths,
    "avg_word_freq_zipf":  avg_word_freqs,
    "avg_word_length":     avg_word_lengths,
}, index=sentences)

print("\nComputed syntax feature stats:")
print(computed_syntax.describe())

# ──────────────────────────────────────────────
# Build and save feature matrices
# ──────────────────────────────────────────────
features_semantics = df_ratings[SEMANTIC_COLS].copy()
features_semantics.index.name = "sentence"

features_syntax = pd.concat([
    df_ratings[SYNTAX_CSV_COLS],
    computed_syntax,
], axis=1)
features_syntax.index.name = "sentence"

# Sanity checks
assert features_semantics.shape == (len(sentences), 7), \
    f"Unexpected shape: {features_semantics.shape}"
assert features_syntax.shape == (len(sentences), 7), \
    f"Unexpected shape: {features_syntax.shape}"

print(f"\nSemantics features: {features_semantics.shape}")
print(features_semantics.head())
print(f"\nSyntax features: {features_syntax.shape}")
print(features_syntax.head())

print("\nNaN in semantics:", features_semantics.isna().sum().to_dict())
print("NaN in syntax:   ", features_syntax.isna().sum().to_dict())

features_semantics.to_csv("features/features_semantics.csv")
features_syntax.to_csv("features/features_syntax.csv")
print("\nSaved features/features_semantics.csv and features/features_syntax.csv")
