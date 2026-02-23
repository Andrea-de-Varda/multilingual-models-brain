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
from scipy.stats import pearsonr, norm as scipy_norm
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
# Aggregate transfer results per model (mean over languages)
# ──────────────────────────────────────────────
_old = np.seterr(invalid="raise", divide="raise")
try:
    per_model = (results.groupby(["condition", "model"])
                 .agg(r=("r_mean", "mean"))
                 .reset_index())
finally:
    np.seterr(**_old)

per_model.to_csv(os.path.join(RES_DIR, "test_results_per_model.csv"), index=False)
print("\nTransfer per-model summary:")
print(per_model.groupby("condition")["r"].describe().round(3))

# ──────────────────────────────────────────────
# CV results summary
# ──────────────────────────────────────────────
cv_rows = []
for mk in available.keys():
    cv_path = os.path.join(CV_BASE, f"{mk}_cv.pkl")
    if not os.path.isfile(cv_path):
        continue
    cv_data = load_pickle(cv_path)
    for cond, rs in cv_data.items():
        rs_finite = [r for r in rs if np.isfinite(r)]
        if not rs_finite:
            continue
        mean_r = np.mean(rs_finite)
        se_r   = np.std(rs_finite, ddof=1) / np.sqrt(len(rs_finite)) if len(rs_finite) > 1 else np.nan
        cv_rows.append({"model": mk, "condition": cond, "mean_r": mean_r, "se_r": se_r})

cv_summary = pd.DataFrame(cv_rows)
cv_summary.to_csv(os.path.join(RES_DIR, "cv_results_summary.csv"), index=False)
print("\nCV results summary:")
print(cv_summary.groupby("condition")["mean_r"].describe().round(3))

# ──────────────────────────────────────────────
# Statistical significance (Stouffer's method)
# ──────────────────────────────────────────────
def r_to_z(r1, r2, n=130):
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(2.0 / (n - 3))
    z = (z1 - z2) / se
    p_two = 2 * (1 - scipy_norm.cdf(abs(z)))
    return z, p_two

def combine_z_statistics(z_stats):
    z_stats = np.asarray(z_stats, dtype=float)
    z_comb = z_stats.sum() / np.sqrt(len(z_stats))
    p_two = 2 * scipy_norm.cdf(-abs(z_comb))
    p_lower = scipy_norm.cdf(z_comb)
    return z_comb, p_two, p_lower

def bh_correction(p_array):
    p = np.asarray(p_array, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out

COMPARISONS = [
    ("semantics_ablated", "intact"),
    ("syntax_ablated",    "intact"),
    ("semantics_ablated", "syntax_ablated"),
]

def build_pairs_transfer(results_df, comparisons=COMPARISONS, n=130):
    """For each (model, language), compute z for each (cond_test, cond_ref) pair."""
    pairs = []
    for (model, language), grp in results_df.groupby(["model", "language"]):
        r_by_cond = {}
        for cond, sub in grp.groupby("condition"):
            r_row = sub["r_mean"]
            if not r_row.empty and np.isfinite(r_row.iloc[0]):
                r_by_cond[cond] = float(r_row.iloc[0])

        for cond_test, cond_ref in comparisons:
            if cond_test not in r_by_cond or cond_ref not in r_by_cond:
                continue
            z, p = r_to_z(r_by_cond[cond_test], r_by_cond[cond_ref], n=n)
            pairs.append({
                "model": model, "language": language,
                "comparison": f"{cond_test} vs {cond_ref}",
                "r_test": r_by_cond[cond_test], "r_ref": r_by_cond[cond_ref],
                "z": z, "p_two": p
            })
    return pd.DataFrame(pairs)

def stouffer_summary_residualize(pairs_df, label="condition"):
    rows = []
    for cond, sub in pairs_df.groupby(label):
        zc, p2, pl = combine_z_statistics(sub["z"].values)
        rows.append({label: cond, "N": len(sub), "z_comb": zc, "p_two": p2, "p_lower": pl})
    overall = pd.DataFrame(rows).sort_values("p_two").reset_index(drop=True)
    m = overall.shape[0]
    overall["p_two_bonf"] = np.minimum(1.0, overall["p_two"] * m)
    overall["q_lower_BH"] = bh_correction(overall["p_lower"].values)
    return overall

# ── Transfer stats ──
pairs_transfer = build_pairs_transfer(results, n=130)
stats_transfer = stouffer_summary_residualize(pairs_transfer, label="comparison")

print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE — TRANSFER (Stouffer's method, n=130 TRs)")
print("=" * 60)
print(stats_transfer.to_string(index=False))
print(f"\n  Pairs per comparison: {pairs_transfer.groupby('comparison').size().to_dict()}")

stats_transfer.to_csv(os.path.join(RES_DIR, "stats_transfer.csv"), index=False)
pairs_transfer.to_csv(os.path.join(RES_DIR, "stats_transfer_pairs.csv"), index=False)

# ── CV stats (fold-level pairs, n ≈ 48 sentences per fold) ──
cv_fold_rows = []
for mk in available.keys():
    cv_path = os.path.join(CV_BASE, f"{mk}_cv.pkl")
    if not os.path.isfile(cv_path):
        continue
    cv_data = load_pickle(cv_path)
    if "intact" not in cv_data:
        continue
    r_by_cond_folds = {cond: folds for cond, folds in cv_data.items()}
    for cond_test, cond_ref in COMPARISONS:
        if cond_test not in r_by_cond_folds or cond_ref not in r_by_cond_folds:
            continue
        for fold_i, (r_test_f, r_ref_f) in enumerate(
                zip(r_by_cond_folds[cond_test], r_by_cond_folds[cond_ref])):
            if not (np.isfinite(r_test_f) and np.isfinite(r_ref_f)):
                continue
            cv_fold_rows.append({
                "model": mk, "fold": fold_i,
                "comparison": f"{cond_test} vs {cond_ref}",
                "r_test": r_test_f, "r_ref": r_ref_f
            })

if cv_fold_rows:
    pairs_cv = pd.DataFrame(cv_fold_rows)
    n_fold = 48  # ~240 training sentences / 5 folds
    zp = pairs_cv.apply(
        lambda row: pd.Series(r_to_z(row["r_test"], row["r_ref"], n=n_fold),
                              index=["z", "p_two"]), axis=1
    )
    pairs_cv = pd.concat([pairs_cv, zp], axis=1)
    stats_cv = stouffer_summary_residualize(pairs_cv, label="comparison")

    print("\n" + "=" * 60)
    print(f"STATISTICAL SIGNIFICANCE — CV (Stouffer's method, n≈{n_fold} sentences/fold)")
    print("=" * 60)
    print(stats_cv.to_string(index=False))
    print(f"\n  Pairs per comparison: {pairs_cv.groupby('comparison').size().to_dict()}")

    stats_cv.to_csv(os.path.join(RES_DIR, "stats_cv.csv"), index=False)
    pairs_cv.to_csv(os.path.join(RES_DIR, "stats_cv_pairs.csv"), index=False)

print(f"\nStats saved to {RES_DIR}")

# ──────────────────────────────────────────────
# Plotting — boxplots, two-panel figure
# ──────────────────────────────────────────────
cond_order       = ["intact", "syntax_ablated", "semantics_ablated"]
cond_order_ablated = ["syntax_ablated", "semantics_ablated"]
SPINE_LW = 2.0
BOX_LW   = 1.1

def style_boxplot(bp, color, alpha=1.0):
    for box in bp['boxes']:
        box.set(facecolor=color, alpha=alpha, edgecolor="black", linewidth=BOX_LW)
    for item in bp['whiskers'] + bp['caps']:
        item.set(color="black", linewidth=1.0)
    for med in bp['medians']:
        med.set(color="black", linewidth=1.2)

def add_boxplot_panel(ax, data_dict, order, ylabel, xticklabels):
    """
    data_dict : {condition: 1-d array of values}
    order     : list of condition keys to plot (left to right)
    """
    for i, cond in enumerate(order):
        vals = data_dict.get(cond, np.array([]))
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        bp = ax.boxplot(
            [vals], positions=[i], widths=0.55,
            showfliers=False, patch_artist=True,
            whis=(5, 95), manage_ticks=False, zorder=2
        )
        style_boxplot(bp, GROUP_COLORS[cond])
        jitter = np.random.default_rng(42).normal(0, 0.07, size=vals.size)
        ax.scatter(i + jitter, vals, color="black", alpha=0.25, s=16, zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xticklabels, fontsize=10, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", width=1.1, length=4)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.margins(x=0)

xlabels_all     = [CONDITION_LABELS[c] for c in cond_order]
xlabels_ablated = [CONDITION_LABELS[c] for c in cond_order_ablated]

# ── Build data dicts ──
transfer_data = {
    cond: per_model.loc[per_model["condition"] == cond, "r"].to_numpy()
    for cond in cond_order
}
cv_data_plot = {
    cond: cv_summary.loc[cv_summary["condition"] == cond, "mean_r"].to_numpy()
    for cond in cond_order
} if not cv_summary.empty else {}

# ── Per-model normalized values (ablated / intact, model-matched) ──
pm_pivot = per_model.pivot(index="model", columns="condition", values="r")
transfer_norm = {}
for c in cond_order_ablated:
    if c in pm_pivot.columns and "intact" in pm_pivot.columns:
        transfer_norm[c] = (pm_pivot[c] / pm_pivot["intact"]).to_numpy()

cv_norm = {}
if not cv_summary.empty:
    cv_pivot = cv_summary.pivot(index="model", columns="condition", values="mean_r")
    for c in cond_order_ablated:
        if c in cv_pivot.columns and "intact" in cv_pivot.columns:
            cv_norm[c] = (cv_pivot[c] / cv_pivot["intact"]).to_numpy()

# ── Figure 1: Transfer + CV (raw R), side-by-side ──
fig, axes = plt.subplots(1, 2, dpi=400, figsize=(9 * 0.6, 3.7 * 0.8))
add_boxplot_panel(axes[0], transfer_data, cond_order,
                  ylabel="R", xticklabels=xlabels_all)
axes[0].set_title("Transfer to new languages", fontsize=11, pad=6)
if cv_data_plot:
    add_boxplot_panel(axes[1], cv_data_plot, cond_order,
                      ylabel="R", xticklabels=xlabels_all)
    axes[1].set_title("Cross-validation (Tuckute2024)", fontsize=11, pad=6)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "residualize.svg"), format="svg", bbox_inches="tight")
# plt.show()

# ── Figure 2: Transfer only (standalone) ──
fig, ax = plt.subplots(dpi=400, figsize=(4.5, 3.7 * 0.9))
add_boxplot_panel(ax, transfer_data, cond_order,
                  ylabel="R (zero-shot transfer)", xticklabels=xlabels_all)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "residualize_transfer.svg"), format="svg", bbox_inches="tight")
# plt.show()

# ── Figure 3: CV only (standalone) ──
if cv_data_plot:
    fig, ax = plt.subplots(dpi=400, figsize=(4.5, 3.7 * 0.9))
    add_boxplot_panel(ax, cv_data_plot, cond_order,
                      ylabel="R (5-fold CV)", xticklabels=xlabels_all)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residualize_cv.svg"), format="svg", bbox_inches="tight")
    # plt.show()

# ── Figure 4: Normalized (÷ intact per model), transfer + CV, side-by-side ──
fig, axes = plt.subplots(1, 2, dpi=400, figsize=(7 * 0.9, 3.7 * 0.9))
ax = axes[0]
add_boxplot_panel(ax, transfer_norm, cond_order_ablated,
                  ylabel="R / R(intact)", xticklabels=xlabels_ablated)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
ax.set_title("Transfer (normalized)", fontsize=11, pad=6)

ax = axes[1]
if cv_norm:
    add_boxplot_panel(ax, cv_norm, cond_order_ablated,
                      ylabel="R / R(intact)", xticklabels=xlabels_ablated)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_title("CV (normalized)", fontsize=11, pad=6)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "residualize_normalized.svg"), format="svg", bbox_inches="tight")
# plt.show()

# ── Figure 5: Normalized transfer only (standalone) ──
fig, ax = plt.subplots(dpi=400, figsize=(3.5, 3.7 * 0.9))
add_boxplot_panel(ax, transfer_norm, cond_order_ablated,
                  ylabel="R / R(intact)", xticklabels=xlabels_ablated)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "residualize_transfer_normalized.svg"), format="svg", bbox_inches="tight")
# plt.show()

# ── Figure 6: Normalized CV only (standalone) ──
if cv_norm:
    fig, ax = plt.subplots(dpi=400, figsize=(3.5, 3.7 * 0.9))
    add_boxplot_panel(ax, cv_norm, cond_order_ablated,
                      ylabel="R / R(intact)", xticklabels=xlabels_ablated)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residualize_cv_normalized.svg"), format="svg", bbox_inches="tight")
    # plt.show()


# ──────────────────────────────────────────────
# Diagnostics plot — R² decodability across INLP steps
# ──────────────────────────────────────────────
DIAG_DIR = os.path.join(SCRIPT_DIR, "diagnostics")

FEAT_LABELS_SEM = {
    "rating_imageability_mean":    "Imageability",
    "rating_others_thoughts_mean": "Others' thoughts",
    "rating_physical_mean":        "Physical",
    "rating_places_mean":          "Places",
    "rating_valence_mean":         "Valence",
    "rating_arousal_mean":         "Arousal",
    "rating_sense_mean":           "Plausibility",
}
FEAT_LABELS_SYN = {
    "log-prob-gpt2-xl_mean":      "Surprisal",
    "rating_gram_mean":           "Grammaticality",
    "rating_frequency_mean":      "Frequency (overall)",
    "rating_conversational_mean": "Frequency (conv)",
    "dep_length_mean":            "Dep. length",
    "avg_word_freq_zipf":         "Word freq.",
    "avg_word_length":            "Word length",
}

def load_diagnostics(condition_suffix):
    frames = []
    for mk in available.keys():
        path = os.path.join(DIAG_DIR, f"{mk}_{condition_suffix}_diagnostics.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df["model"] = mk
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def plot_diagnostics_panel(ax, diag_df, feat_labels, title, n_feats=7):
    if diag_df.empty:
        ax.set_visible(False)
        return
    features = list(feat_labels.keys())
    palette  = plt.cm.tab10.colors
    colors   = {f: palette[i % len(palette)] for i, f in enumerate(features)}

    # Only plot at full-cycle boundaries (step 0, n_feats, 2*n_feats, ...)
    # to avoid within-cycle sawtooth artifacts
    cycle_steps = set(diag_df.loc[diag_df["step"] % n_feats == 0, "step"].unique())

    for feat in features:
        sub = diag_df[diag_df["feature_name"] == feat]
        sub = sub[sub["step"].isin(cycle_steps)]
        color = colors[feat]
        for _, grp in sub.groupby("model"):
            grp_s = grp.sort_values("step")
            ax.plot(grp_s["step"], grp_s["r2"],
                    color=color, alpha=0.12, linewidth=0.7, zorder=2)
        # invisible line just to register the label in the legend
        ax.plot([], [], color=color, linewidth=1.5, label=feat_labels[feat])

    ax.set_xlabel("INLP step", fontsize=10)
    ax.set_ylabel("R² (decodability)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", width=1.1, length=4, labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              handlelength=1.5, labelspacing=0.35)

diag_sem = load_diagnostics("semantics")
diag_syn = load_diagnostics("syntax")

if not diag_sem.empty or not diag_syn.empty:
    fig, axes = plt.subplots(1, 2, dpi=400, figsize=(11 * 0.6, 3.8 * 0.7))
    plot_diagnostics_panel(axes[0], diag_sem, FEAT_LABELS_SEM, "Semantics ablation")
    plot_diagnostics_panel(axes[1], diag_syn, FEAT_LABELS_SYN, "Syntax ablation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residualize_diagnostics.svg"),
                format="svg", bbox_inches="tight")
    # plt.show()

# ──────────────────────────────────────────────
# Combined diagnostics plot — within vs across domain
# ──────────────────────────────────────────────
# Within (red): features being removed  |  Across (blue): features that should survive
cross_sem = load_diagnostics("cross_semantics")  # syntax feats on sem-ablated embs
cross_syn = load_diagnostics("cross_syntax")      # semantic feats on syn-ablated embs

COLOR_WITHIN  = "tab:red"
COLOR_ACROSS  = "tab:blue"

def plot_within_across_panel(ax, within_df, across_df, title, n_feats=7):
    """Plot within-domain (red) and across-domain (blue) R² lines."""
    if within_df.empty and across_df.empty:
        ax.set_visible(False)
        return

    # Filter within-domain to cycle boundaries to remove sawtooth
    within_filtered = within_df[within_df["step"] % n_feats == 0] if not within_df.empty else within_df

    # Thin lines: each (feature, model) combination
    for df, color in [(within_filtered, COLOR_WITHIN),
                       (across_df, COLOR_ACROSS)]:
        if df.empty:
            continue
        for (feat, mk), grp in df.groupby(["feature_name", "model"]):
            grp_s = grp.sort_values("step")
            ax.plot(grp_s["step"], grp_s["r2"],
                    color=color, alpha=0.10, linewidth=0.6, zorder=2)

    # Bold average lines (mean across features and models at each step)
    # Only include steps with data from at least half the models
    # (avoids spikes from per-model endpoint steps)
    for df, color, label in [(within_filtered, COLOR_WITHIN, "Within-domain"),
                              (across_df, COLOR_ACROSS, "Across-domain")]:
        if df.empty:
            continue
        n_models = df["model"].nunique()
        step_model_counts = df.groupby("step")["model"].nunique()
        valid_steps = set(step_model_counts[step_model_counts >= n_models * 0.5].index)
        df_valid = df[df["step"].isin(valid_steps)]
        mean_curve = df_valid.groupby("step")["r2"].mean().reset_index().sort_values("step")
        ax.plot(mean_curve["step"], mean_curve["r2"],
                color=color, linewidth=2.2, label=label, zorder=4)

    ax.set_xlabel("INLP step", fontsize=10)
    ax.set_ylabel("R² (decodability)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", width=1.1, length=4, labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

has_cross = not cross_sem.empty or not cross_syn.empty
if has_cross:
    fig, axes = plt.subplots(1, 2, dpi=400, figsize=(11 * 0.6, 3.8 * 0.7))
    plot_within_across_panel(axes[0], diag_sem, cross_sem, "Semantics ablation")
    plot_within_across_panel(axes[1], diag_syn, cross_syn, "Syntax ablation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residualize_cross_diagnostics.svg"),
                format="svg", bbox_inches="tight")
    # plt.show()

print("\nDone. Plots saved to", PLOT_DIR)
