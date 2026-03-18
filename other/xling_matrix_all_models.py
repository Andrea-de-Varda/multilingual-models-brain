#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr


mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

sys.modules['numpy._core.numeric'] = np.core.numeric

# ── config ────────────────────────────────────────────────────────────────────
model_names = [
    "nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B",
    "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm",
    "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta",
    "mt5_small", "mt5_base", "mt5_large",
    "mgpt", "xglm_small", "xglm_med", "xglm_large", "xglm_xl",
]

langs       = ['fr', 'ta', 'es', 'tr', 'vi', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice  = ['French', 'Tamil', 'Spanish', 'Turkish', 'Vietnamese', 'Marathi',
               'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']
lang_dict   = dict(zip(langs, langs_nice))
pretty_labels = [lang_dict[l] for l in langs]

FROI = "all"

# ── data loading ──────────────────────────────────────────────────────────────
def patched_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load(model_prefix, froi="all", multitrain=True):
    mtpfx = "multitrain_" if multitrain else ""
    filename = f"results/multilingual_{mtpfx}{model_prefix}_{froi}"
    return patched_load(filename)

def get_best_layer_matrix(model):
    res_dict = load(model, froi=FROI, multitrain=False)
    layer_means = {k: v["r"].mean() for k, v in res_dict.items()}
    best_layer  = max(layer_means, key=layer_means.get)
    df_best     = res_dict[best_layer]

    M = pd.DataFrame(np.nan, index=langs, columns=langs, dtype=float)
    test_cols = [c for c in df_best.columns if c not in ("target_lang", "r")]
    for _, row in df_best.iterrows():
        tr = row["target_lang"]
        if tr not in langs:
            continue
        for t in test_cols:
            if t in langs:
                try:
                    M.at[tr, t] = float(row[t])
                except Exception:
                    pass
    return M

# ── average across all models ─────────────────────────────────────────────────
mats = []
for model in model_names:
    try:
        M = get_best_layer_matrix(model)
        mats.append(M.to_numpy())
        print(f"[OK]   {model}")
    except Exception as e:
        print(f"[SKIP] {model}: {e}")

stacked = np.stack(mats, axis=2)           # (n_langs, n_langs, n_models)
A = np.nanmean(stacked, axis=2)

# symmetrize
n = A.shape[0]
for i in range(n):
    for j in range(i + 1, n):
        m = np.nanmean([A[i, j], A[j, i]])
        A[i, j] = m
        A[j, i] = m

sym_df = pd.DataFrame(A, index=langs, columns=langs)

# ── heatmap helper ────────────────────────────────────────────────────────────
def plot_heatmap(mat, savepath):
    mask_upper = np.triu(np.ones_like(mat, dtype=bool), k=1)
    absmax     = np.nanmax(np.abs(mat))

    plt.figure(figsize=(10.5 * 0.7, 8.2 * 0.7), dpi=300)
    sns.set_context("talk")
    df = pd.DataFrame(mat, index=langs, columns=langs)
    ax = sns.heatmap(
        df,
        mask=mask_upper,
        cmap="RdBu_r",
        vmin=-absmax, vmax=absmax, center=0,
        square=True,
        cbar_kws={"label": "r"},
        linewidths=0.3, linecolor="white",
    )
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(pretty_labels, rotation=0,  ha="right", fontsize=16)

    for i in range(len(langs)):
        for j in range(i + 1):  # lower triangle including diagonal
            if not np.isnan(mat[i, j]) and mat[i, j] < 0:
                ax.plot(j + 0.5, i + 0.5, 'o', color='gray', markersize=4, zorder=3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(savepath, format="svg", bbox_inches="tight")
    plt.show()

# ── plot: average across all models ──────────────────────────────────────────
plot_heatmap(A, "plots/xling_matrix_all_models.svg")

# ── plot: infoxlm_large only ──────────────────────────────────────────────────
M_info = get_best_layer_matrix("infoxlm_large").to_numpy()
n = M_info.shape[0]
for i in range(n):
    for j in range(i + 1, n):
        m = np.nanmean([M_info[i, j], M_info[j, i]])
        M_info[i, j] = m
        M_info[j, i] = m
plot_heatmap(M_info, "plots/xling_matrix_infoxlm_large.svg")

# ── reliability vs. average transfer ─────────────────────────────────────────

rois = ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp']
data = pd.read_csv("data/Alice_Story_TimeSeries.csv")

reliability = {}
for l in set(data["Language"]):
    temp = data[data.Language == l]
    if len(temp) == 24:
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois))]
        ts1 = list(part1.iloc[:, 7:].mean())[9:-3]
        ts2 = list(part2.iloc[:, 7:].mean())[9:-3]
        r, _ = pearsonr(ts1, ts2)
        reliability[l] = r  # keyed by full language name

def _p_label(p):
    if p < 0.001: return "p < 0.001"
    if p < 0.01:  return "p < 0.01"
    if p < 0.05:  return "p < 0.05"
    return f"p = {p:.2f}"

def _annotate_stats(ax, r_val, p_val):
    txt = f"r = {r_val:.2f}\n{_p_label(p_val)}"
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85))

def plot_reliability_vs_avg_transfer(mat, rel, savepath, title):
    """One dot per language: x=reliability, y=avg off-diagonal transfer."""
    lang_nice_to_code = {v: k for k, v in lang_dict.items()}
    xs, ys, lbls = [], [], []
    for nice_name, code in lang_nice_to_code.items():
        if nice_name not in rel:
            continue
        idx = langs.index(code)
        row = mat[idx, :].copy()
        row[idx] = np.nan
        xs.append(rel[nice_name])
        ys.append(np.nanmean(row))
        lbls.append(nice_name)

    r_val, p_val = pearsonr(xs, ys)
    m, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs), max(xs), 100)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    ax.scatter(xs, ys, s=45, color='steelblue', zorder=3)
    ax.plot(x_line, m * x_line + b, color='firebrick', linewidth=1.5, zorder=2)
    for x, y, lbl in zip(xs, ys, lbls):
        ax.annotate(lbl, (x, y), fontsize=9.5, xytext=(4, 3), textcoords='offset points')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xlabel("Split-half reliability ($r$)", fontsize=14)
    ax.set_ylabel("Avg. cross-lingual transfer ($r$)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=11)
    _annotate_stats(ax, r_val, p_val)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savepath, format="svg", bbox_inches="tight")
    plt.show()
    return r_val, p_val, xs, ys, lbls

def plot_reliability_vs_pairwise_transfer(mat, rel, savepath, title):
    """One dot per language pair: x=avg reliability of l_i & l_j, y=transfer value."""
    lang_nice_to_code = {v: k for k, v in lang_dict.items()}
    xs, ys = [], []
    lang_list = [l for l in langs_nice if l in rel]
    for i, ni in enumerate(lang_list):
        for j, nj in enumerate(lang_list):
            if j >= i:
                continue
            ci, cj = lang_nice_to_code[ni], lang_nice_to_code[nj]
            ii, jj = langs.index(ci), langs.index(cj)
            val = mat[ii, jj]
            if np.isnan(val):
                continue
            xs.append((rel[ni] + rel[nj]) / 2)
            ys.append(val)

    r_val, p_val = pearsonr(xs, ys)
    m, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs), max(xs), 100)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    ax.scatter(xs, ys, s=20, color='steelblue', alpha=0.6, zorder=3)
    ax.plot(x_line, m * x_line + b, color='firebrick', linewidth=1.5, zorder=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xlabel(r"Avg. split-half reliability ($L_i$, $L_j$)", fontsize=16)
    ax.set_ylabel(r"Pairwise transfer ($L_i \to L_j$)", fontsize=16)
    if title:
        ax.set_title(title, fontsize=11)
    _annotate_stats(ax, r_val, p_val)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savepath, format="svg", bbox_inches="tight")
    plt.show()
    return r_val, p_val, len(xs)

print("\n" + "="*60)
print("RELIABILITY vs. TRANSFER CORRELATIONS")
print("="*60)

r, p, xs, ys, lbls = plot_reliability_vs_avg_transfer(
    A, reliability, "plots/reliability_vs_transfer_all_models.svg", "")
print(f"  [avg-transfer, all models]   r = {r:.3f}, p = {p:.4f}  (N={len(xs)} languages)")
for lbl, x, y in sorted(zip(lbls, xs, ys), key=lambda t: t[1]):
    print(f"    {lbl:<15}  reliability={x:.3f}  avg_transfer={y:.4f}")

r, p, xs, ys, lbls = plot_reliability_vs_avg_transfer(
    M_info, reliability, "plots/reliability_vs_transfer_infoxlm_large.svg", "")
print(f"\n  [avg-transfer, infoxlm_large] r = {r:.3f}, p = {p:.4f}  (N={len(xs)} languages)")
for lbl, x, y in sorted(zip(lbls, xs, ys), key=lambda t: t[1]):
    print(f"    {lbl:<15}  reliability={x:.3f}  avg_transfer={y:.4f}")

r, p, n = plot_reliability_vs_pairwise_transfer(
    A, reliability, "plots/reliability_vs_pairwise_all_models.svg", "")
print(f"\n  [pairwise-transfer, all models]   r = {r:.3f}, p = {p:.4f}  (N={n} pairs)")

r, p, n = plot_reliability_vs_pairwise_transfer(
    M_info, reliability, "plots/reliability_vs_pairwise_infoxlm_large.svg", "")
print(f"  [pairwise-transfer, infoxlm_large] r = {r:.3f}, p = {p:.4f}  (N={n} pairs)")

print("="*60 + "\n")

# ── within vs. across scatter (infoxlm_large) ────────────────────────────────
lang_nice_to_code = {v: k for k, v in lang_dict.items()}

within_vals, across_vals, scatter_labels = [], [], []
for nice_name, code in lang_nice_to_code.items():
    idx = langs.index(code)
    w = M_info[idx, idx]
    row = M_info[idx, :].copy()
    row[idx] = np.nan
    a = np.nanmean(row)
    within_vals.append(w)
    across_vals.append(a)
    scatter_labels.append(nice_name)

within_vals = np.array(within_vals)
across_vals = np.array(across_vals)

fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)

lo = min(within_vals.min(), across_vals.min()) - 0.06
hi = max(within_vals.max(), across_vals.max()) + 0.06
ax.fill_between([lo, hi], [lo, hi], [hi, hi], color='#ffe0e0', alpha=0.5, zorder=0)
ax.fill_between([lo, hi], [lo, hi], [lo, lo], color='#e0e8ff', alpha=0.5, zorder=0)
ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=1, zorder=1)

ax.scatter(within_vals, across_vals, s=55, color='steelblue', edgecolor='black',
           linewidth=0.5, zorder=3)
for x, y, lbl in zip(within_vals, across_vals, scatter_labels):
    if lbl == "Romanian":
        ax.annotate(lbl, (x-.12, y+.02), fontsize=11, xytext=(5, 4), textcoords='offset points')

ax.set_xlabel("Within-language transfer ($r$)", fontsize=15)
ax.set_ylabel("Avg. across-language transfer ($r$)", fontsize=15)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.text(0.06, 0.92, "across > within", transform=ax.transAxes,
        ha='left', va='top', fontsize=11, color='#b03030', fontstyle='italic')
ax.text(0.97, 0.06, "within > across", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=11, color='#3050a0', fontstyle='italic')
sns.despine()
plt.tight_layout()
plt.savefig("plots/within_vs_across_infoxlm.svg", format="svg", bbox_inches="tight")
plt.show()

# ── permutation test: are the observed flips (across > within) unusual? ───────
# Null: for each language's row, randomly pick one of the 12 cells as "within"
# and average the other 11 as "across." Count how many languages flip.
# If the observed count falls within or below the null, the flips are just noise.
n_langs_mat = len(langs)
n_perm = 10000
rng = np.random.default_rng(42)

obs_diffs = np.array([
    M_info[i, i] - np.nanmean(np.concatenate([M_info[i, :i], M_info[i, i+1:]]))
    for i in range(n_langs_mat)
])
obs_n_flips = int(np.sum(obs_diffs < 0))

perm_n_flips = np.zeros(n_perm, dtype=int)
for p in range(n_perm):
    n_flips = 0
    for i in range(n_langs_mat):
        row = M_info[i, :].copy()
        valid = np.where(~np.isnan(row))[0]
        fake_diag_idx = rng.choice(valid)
        fake_within = row[fake_diag_idx]
        row_off = np.delete(row, fake_diag_idx)
        fake_across = np.nanmean(row_off)
        if fake_within - fake_across < 0:
            n_flips += 1
    perm_n_flips[p] = n_flips

null_mean   = perm_n_flips.mean()
null_std    = perm_n_flips.std()
p_val_exact = np.mean(perm_n_flips == obs_n_flips)
p_val_leq   = np.mean(perm_n_flips <= obs_n_flips)
n_exact     = int(np.sum(perm_n_flips == obs_n_flips))

# ── stats printout for reviewer response ─────────────────────────────────────
print("\n" + "="*60)
print("PERMUTATION TEST STATS (infoxlm_large, within vs. across)")
print("="*60)
print(f"  N languages in matrix           : {n_langs_mat}")
print(f"  Observed flips (across > within): {obs_n_flips} / {n_langs_mat}")
print(f"  Languages with across > within  : {[langs_nice[i] for i in range(n_langs_mat) if obs_diffs[i] < 0]}")
print(f"  Languages with within > across  : {[langs_nice[i] for i in range(n_langs_mat) if obs_diffs[i] >= 0]}")
print()
print(f"  Null distribution ({n_perm} permutations):")
print(f"    Mean  : {null_mean:.2f}")
print(f"    SD    : {null_std:.2f}")
print(f"    Range : [{perm_n_flips.min()}, {perm_n_flips.max()}]")
print()
print(f"  Simulations with exactly {obs_n_flips} flip(s): {n_exact} / {n_perm}  (p = {p_val_exact:.4f})")
print(f"  Simulations with <= {obs_n_flips} flip(s)     : {int(np.sum(perm_n_flips <= obs_n_flips))} / {n_perm}  (one-tailed p = {p_val_leq:.4f})")
print()
print(f"  {'Language':<15} {'within':>8} {'across_mean':>12} {'diff (w-a)':>12} {'flip?':>6}")
print("  " + "-"*55)
for i in range(n_langs_mat):
    w = M_info[i, i]
    a = np.nanmean(np.concatenate([M_info[i, :i], M_info[i, i+1:]]))
    d = w - a
    print(f"  {langs_nice[i]:<15} {w:>8.4f} {a:>12.4f} {d:>12.4f} {'YES' if d < 0 else '':>6}")
print("="*60 + "\n")

# ── plot ──────────────────────────────────────────────────────────────────────
counts = np.bincount(perm_n_flips, minlength=n_langs_mat + 1)
probs  = counts / n_perm
active = np.where(probs > 0)[0]

fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
colors_bar = ['#b03030' if x == obs_n_flips else 'lightsteelblue' for x in active]
ax.bar(active, probs[active], color=colors_bar, edgecolor='white', linewidth=0.5, zorder=2)
ax.axvline(obs_n_flips, color='firebrick', linewidth=2, linestyle='--', zorder=3)
ymax = max(probs[active])
ax.text(obs_n_flips + 0.2, ymax * 0.9, f"observed = {obs_n_flips}",
        color='firebrick', fontsize=10, va='top', ha='left')
p_str = f"p = {p_val_leq:.3f}" if p_val_leq >= 0.001 else "p < 0.001"
ax.text(0.97, 0.95, p_str, transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85))
ax.set_xlabel("# languages with across > within", fontsize=14)
ax.set_ylabel("Proportion", fontsize=14)
ax.set_xticks(active)
sns.despine()
plt.tight_layout()
plt.savefig("plots/permutation_within_vs_across.svg", format="svg", bbox_inches="tight")
plt.show()
