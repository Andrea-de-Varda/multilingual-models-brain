"""
Evaluation script for INLP Residualization Analysis
====================================================
Mirrors perturbation_encoding_Pereira_Tuckute.py in structure.

Evaluates trained encoders on:
  a) Zero-shot transfer to 9 new languages (confirmatory fMRI data)
  b) 5-fold CV on Tuckute2024 training data (from saved cv_results/)

Normalizes both by the "intact" condition from this same analysis.

Outputs:
  results/test_results.csv        — transfer results, same format as perturbation
  results/cv_results_summary.csv  — CV summary (mean r, SE, normalized r)
  Plots comparable to perturbation bar charts
"""

import os, pickle, warnings, math
import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings("ignore", message="Mean of empty slice.")
np.seterr(invalid="warn", divide="warn")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIRM_BASE = "/home/andrea/Documents/PhD/Alice/confirmatory"
EMB_DIR      = os.path.join(CONFIRM_BASE, "embeddings")
TRANS_DIR    = os.path.join(CONFIRM_BASE, "transcribed")
FMRI_DICT    = os.path.join(CONFIRM_BASE, "data/dict_fMRI")

REG_BASE  = os.path.join(SCRIPT_DIR, "registered_models")
NORM_BASE = os.path.join(SCRIPT_DIR, "registered_models/normaliz_params")
PROJ_BASE = os.path.join(SCRIPT_DIR, "registered_models/projections")
CV_BASE   = os.path.join(SCRIPT_DIR, "cv_results")
RES_DIR   = os.path.join(SCRIPT_DIR, "results")
PLOT_DIR  = os.path.join(SCRIPT_DIR, "plots")

for d in [RES_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

os.chdir(CONFIRM_BASE)

# ──────────────────────────────────────────────
# Constants (identical to perturbation scripts)
# ──────────────────────────────────────────────
dict_bestlayer = {
    "nllb200_distilled_600M": 9,  "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7,  "infoxlm_base": 7,  "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9,  "xlmr_large": 14,   "distilmbert": 4,    "bert_base": 6,
    "mdeberta": 9,   "mt5_small": 2,     "mt5_base": 9,       "mt5_large": 17,
    "mgpt": 14,      "xglm_small": 10,   "xglm_med": 16,      "xglm_large": 40,
    "xglm_xl": 48,
}

passages   = ["Passage_1", "Passage_2", "Passage_3"]
languages  = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_map   = {c: n for c, n in zip(lang_codes, languages)}
keep = {'ar': [1,2,3], 'de': [2,3], 'hi': [2,3], 'it': [1,2,3],
        'ko': [1],     'pt': [1,3], 'ru': [1,2,3], 'zh': [1,2,3], 'pl': [1,2,3]}

CONDITIONS        = ["intact", "semantics_ablated", "syntax_ablated"]
CONDITION_LABELS  = {
    "intact":            "Intact",
    "semantics_ablated": "Semantics ablated",
    "syntax_ablated":    "Syntax ablated",
}
GROUP_COLORS = {
    "intact":            "tab:blue",
    "semantics_ablated": "tab:red",
    "syntax_ablated":    "tab:purple",
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_pickle(path):
    with open(path, "rb") as h:
        return pickle.load(h)

def load_embeddings(passage, model_key, lang_code):
    return load_pickle(os.path.join(EMB_DIR, passage, f"{model_key}_{lang_code}"))

def compute_nullspace_projection(W_stack):
    """Recompute P_null from saved W_stack via SVD (Ben-Israel formula)."""
    U, S, Vt = np.linalg.svd(W_stack, full_matrices=False)
    rank = int(np.sum(S > 1e-10 * S[0]))
    if rank == 0:
        return np.eye(W_stack.shape[1], dtype=np.float32)
    P_rowspace = Vt[:rank].T @ Vt[:rank]
    return (np.eye(W_stack.shape[1], dtype=np.float32) - P_rowspace.astype(np.float32))

def apply_projection(X, W_stack):
    """Project X onto the nullspace of W_stack."""
    P_null = compute_nullspace_projection(W_stack)
    return (X @ P_null).astype(np.float32)

def embed_words_to_bins(embeddings, words_id):
    """Align word embeddings to 2-second TR bins (130 bins, 0–258 s)."""
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)  # 130 bins
    nD = embeddings.shape[1]
    out = []
    for i in range(time.shape[0]):
        sel = (ids == i)
        if not np.any(sel):
            out.append(np.full((nD,), np.nan, dtype=float))
        else:
            out.append(np.mean(embeddings[sel], axis=0))
    out = np.array(out, dtype=float)
    # Mean imputation for empty bins
    arr = ma.array(out, mask=np.isnan(out))
    col_mean = arr.mean(axis=0)
    return np.where(np.isnan(out), col_mean, out)

def preproc_align(lang_code, passage, embeddings):
    """Load transcription, align word embeddings to TR bins."""
    df = pd.read_csv(os.path.join(TRANS_DIR, passage, f"{lang_code}.csv"))
    df = df[(df["end"] <= 260) & (df["text"] != " ")]
    time = np.arange(0, 260, 2)
    words_id = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        words_id[i] = np.where(df["end"].iloc[i] > time)[0][-1]
    return embed_words_to_bins(embeddings, words_id)

# ──────────────────────────────────────────────
# Load fMRI data
# ──────────────────────────────────────────────
d_fmri = load_pickle(FMRI_DICT)

# ──────────────────────────────────────────────
# Discover available models and conditions
# ──────────────────────────────────────────────
model_keys = [m for m in os.listdir(REG_BASE)
              if os.path.isdir(os.path.join(REG_BASE, m))
              and m not in ("normaliz_params", "projections")]
available = {}
for mk in model_keys:
    reg_dir  = os.path.join(REG_BASE, mk)
    norm_dir = os.path.join(NORM_BASE, mk)
    proj_dir = os.path.join(PROJ_BASE, mk)
    if not os.path.isdir(norm_dir):
        continue
    conds = sorted([
        c for c in CONDITIONS
        if (os.path.isfile(os.path.join(reg_dir, c)) and
            os.path.isfile(os.path.join(norm_dir, c)))
    ])
    if conds:
        available[mk] = conds
print(f"Available models: {list(available.keys())}")

# ──────────────────────────────────────────────
# Transfer evaluation (zero-shot to new languages)
# ──────────────────────────────────────────────
CACHE_PATH = os.path.join(RES_DIR, "test_results.csv")
if os.path.exists(CACHE_PATH):
    results = pd.read_csv(CACHE_PATH)
else:
    results = pd.DataFrame(columns=["condition", "model", "language",
                                     "r1", "r2", "r3", "r_mean", "r_sd"])

skip_keys = set(results[["condition", "model", "language"]].itertuples(index=False, name=None))

new_rows = []
for model_key, conds in available.items():
    best_layer = dict_bestlayer.get(model_key)
    if best_layer is None:
        continue

    # Load projection matrices (once per model)
    proj_dir  = os.path.join(PROJ_BASE, model_key)
    W_sem, W_syn = None, None
    if os.path.isfile(os.path.join(proj_dir, "semantics_ablated")):
        W_sem = load_pickle(os.path.join(proj_dir, "semantics_ablated"))
    if os.path.isfile(os.path.join(proj_dir, "syntax_ablated")):
        W_syn = load_pickle(os.path.join(proj_dir, "syntax_ablated"))

    for lang in lang_codes:
        kept = keep[lang]
        lang_label = lang_map[lang]

        for cond in conds:
            if (cond, model_key, lang_label) in skip_keys:
                continue

            reg     = load_pickle(os.path.join(REG_BASE,  model_key, cond))
            Xsc, Ysc = load_pickle(os.path.join(NORM_BASE, model_key, cond))

            rs = []
            ok_any = False
            for passage in passages:
                try:
                    emb_dict = load_embeddings(passage, model_key, lang)
                except FileNotFoundError:
                    rs.append(np.nan)
                    continue

                if best_layer not in emb_dict:
                    rs.append(np.nan)
                    continue

                # Apply projection to test embeddings (same as training)
                X_raw_test = emb_dict[best_layer]  # (T × d)
                if cond == "semantics_ablated" and W_sem is not None:
                    X_raw_test = apply_projection(X_raw_test, W_sem)
                elif cond == "syntax_ablated" and W_syn is not None:
                    X_raw_test = apply_projection(X_raw_test, W_syn)
                # "intact": no projection

                X_aligned = preproc_align(lang, passage, X_raw_test)
                X = Xsc.transform(X_aligned)

                y_raw = d_fmri[passage][lang_label].reshape(-1, 1).flatten()
                y = Ysc.transform(y_raw.reshape(-1, 1)).flatten()

                n = min(X.shape[0], y.shape[0])
                X, y = X[:n], y[:n]

                pred = reg.predict(X)
                if (np.nanstd(pred) == 0 or np.nanstd(y) == 0 or
                        not np.isfinite(pred).all() or not np.isfinite(y).all()):
                    rs.append(np.nan)
                    continue

                r = pearsonr(pred, y)[0]
                rs.append(r)
                ok_any = True

            if not ok_any:
                continue

            r_keep = [rs[i-1] for i in kept if i <= len(rs) and not np.isnan(rs[i-1])]
            new_rows.append({
                "condition": cond,
                "model":     model_key,
                "language":  lang_label,
                "r1":        rs[0] if len(rs) > 0 else np.nan,
                "r2":        rs[1] if len(rs) > 1 else np.nan,
                "r3":        rs[2] if len(rs) > 2 else np.nan,
                "r_mean":    np.nanmean(r_keep) if r_keep else np.nan,
                "r_sd":      np.nanstd(r_keep)  if r_keep else np.nan,
            })

if new_rows:
    results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)

results.to_csv(CACHE_PATH, index=False)
print(f"Transfer results saved to {CACHE_PATH}")
print(results.groupby("condition")["r_mean"].describe())

# ──────────────────────────────────────────────
# Aggregate transfer results per model
# ──────────────────────────────────────────────
_old = np.seterr(invalid="raise", divide="raise")
try:
    per_model = (results.groupby(["condition", "model"])
                 .agg(r=("r_mean", "mean"),
                      SE=("r_mean", lambda x: (x.std(ddof=1) / np.sqrt(x.notna().sum())
                                               if x.notna().sum() > 1 else np.nan)))
                 .reset_index())
finally:
    np.seterr(**_old)

# ──────────────────────────────────────────────
# Normalization by intact condition (transfer)
# ──────────────────────────────────────────────
intact_r = (per_model[per_model["condition"] == "intact"]
            .set_index("model")["r"])

per_model["r_norm"] = per_model.apply(
    lambda row: row["r"] / intact_r.get(row["model"], np.nan)
    if row["condition"] != "intact" else 1.0,
    axis=1
)

per_model.to_csv(os.path.join(RES_DIR, "test_results_per_model.csv"), index=False)
print("\nTransfer per-model summary:")
print(per_model.groupby("condition")[["r", "r_norm"]].mean().round(3))

# ──────────────────────────────────────────────
# CV results summary
# ──────────────────────────────────────────────
cv_rows = []
for mk in available.keys():
    cv_path = os.path.join(CV_BASE, f"{mk}_cv.pkl")
    if not os.path.isfile(cv_path):
        continue
    cv_data = load_pickle(cv_path)
    r_intact_mean = np.nanmean(cv_data.get("intact", [np.nan]))
    for cond, rs in cv_data.items():
        rs = [r for r in rs if np.isfinite(r)]
        if not rs:
            continue
        mean_r = np.mean(rs)
        se_r   = np.std(rs, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.nan
        norm_r = mean_r / r_intact_mean if r_intact_mean > 0 else np.nan
        cv_rows.append({
            "model":     mk,
            "condition": cond,
            "mean_r":    mean_r,
            "se_r":      se_r,
            "r_norm":    norm_r if cond != "intact" else 1.0,
        })

cv_summary = pd.DataFrame(cv_rows)
cv_summary.to_csv(os.path.join(RES_DIR, "cv_results_summary.csv"), index=False)
print("\nCV results summary:")
print(cv_summary.groupby("condition")[["mean_r", "r_norm"]].mean().round(3))

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
def bar_means_se(pm, cond_order):
    means = {}
    ses   = {}
    for c in cond_order:
        vals = pm.loc[pm["condition"] == c, "r"].to_numpy()
        vals = vals[np.isfinite(vals)]
        means[c] = np.mean(vals) if vals.size else np.nan
        ses[c]   = (np.std(vals, ddof=1) / np.sqrt(vals.size)
                    if vals.size > 1 else np.nan)
    return means, ses

cond_order = CONDITIONS

# ── Figure 1: Transfer (raw R) ──
means, ses = bar_means_se(per_model, cond_order)

fig, ax = plt.subplots(dpi=400, figsize=(6, 3.5))
for i, cond in enumerate(cond_order):
    yerr = ses[cond] if np.isfinite(ses[cond]) else 0.0
    ax.bar(i, means[cond], yerr=yerr, capsize=5,
           color=GROUP_COLORS[cond], edgecolor="black", linewidth=1.1, zorder=2)
    vals = per_model.loc[per_model["condition"] == cond, "r"].to_numpy()
    jitter = np.random.normal(0, 0.08, size=vals.size)
    ax.scatter(i + jitter, vals, color="black", alpha=0.35, s=18, zorder=3)

ax.set_xticks(range(len(cond_order)))
ax.set_xticklabels([CONDITION_LABELS[c] for c in cond_order],
                   fontsize=11, rotation=20, ha="right")
ax.set_ylabel("R (transfer to new languages)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "transfer_raw.svg"), format="svg", bbox_inches="tight")
plt.show()

# ── Figure 2: Transfer (normalized by intact) ──
means_n = {}
ses_n   = {}
for c in cond_order:
    vals = per_model.loc[per_model["condition"] == c, "r_norm"].to_numpy()
    vals = vals[np.isfinite(vals)]
    means_n[c] = np.mean(vals) if vals.size else np.nan
    ses_n[c]   = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan

fig, ax = plt.subplots(dpi=400, figsize=(6, 3.5))
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, zorder=1)
for i, cond in enumerate(cond_order):
    yerr = ses_n[cond] if np.isfinite(ses_n[cond]) else 0.0
    ax.bar(i, means_n[cond], yerr=yerr, capsize=5,
           color=GROUP_COLORS[cond], edgecolor="black", linewidth=1.1, zorder=2)
    vals = per_model.loc[per_model["condition"] == cond, "r_norm"].to_numpy()
    jitter = np.random.normal(0, 0.08, size=vals.size)
    ax.scatter(i + jitter, vals, color="black", alpha=0.35, s=18, zorder=3)

ax.set_xticks(range(len(cond_order)))
ax.set_xticklabels([CONDITION_LABELS[c] for c in cond_order],
                   fontsize=11, rotation=20, ha="right")
ax.set_ylabel("Normalized R (÷ intact)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "transfer_normalized.svg"), format="svg", bbox_inches="tight")
plt.show()

# ── Figure 3: CV (normalized) ──
if not cv_summary.empty:
    cv_means_n = {}
    cv_ses_n   = {}
    for c in cond_order:
        sub = cv_summary[cv_summary["condition"] == c]
        vals = sub["r_norm"].to_numpy()
        vals = vals[np.isfinite(vals)]
        cv_means_n[c] = np.mean(vals) if vals.size else np.nan
        cv_ses_n[c]   = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan

    fig, ax = plt.subplots(dpi=400, figsize=(6, 3.5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, zorder=1)
    for i, cond in enumerate(cond_order):
        yerr = cv_ses_n[cond] if np.isfinite(cv_ses_n.get(cond, np.nan)) else 0.0
        ax.bar(i, cv_means_n.get(cond, np.nan), yerr=yerr, capsize=5,
               color=GROUP_COLORS[cond], edgecolor="black", linewidth=1.1, zorder=2)
        sub = cv_summary[cv_summary["condition"] == cond]
        vals = sub["r_norm"].to_numpy()
        jitter = np.random.normal(0, 0.08, size=vals.size)
        ax.scatter(i + jitter, vals, color="black", alpha=0.35, s=18, zorder=3)

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in cond_order],
                       fontsize=11, rotation=20, ha="right")
    ax.set_ylabel("Normalized R, CV (÷ intact)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cv_normalized.svg"), format="svg", bbox_inches="tight")
    plt.show()

print("\nDone. Plots saved to", PLOT_DIR)
