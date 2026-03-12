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
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(pretty_labels, rotation=0,  ha="right", fontsize=14)

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
    ax.scatter(xs, ys, s=40, color='steelblue', zorder=3)
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
    ax.scatter(xs, ys, s=15, color='steelblue', alpha=0.6, zorder=3)
    ax.plot(x_line, m * x_line + b, color='firebrick', linewidth=1.5, zorder=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xlabel(r"Avg. split-half reliability ($L_i$, $L_j$)", fontsize=14)
    ax.set_ylabel(r"Pairwise transfer ($L_i \to L_j$)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=11)
    _annotate_stats(ax, r_val, p_val)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savepath, format="svg", bbox_inches="tight")
    plt.show()

plot_reliability_vs_avg_transfer(A,      reliability, "plots/reliability_vs_transfer_all_models.svg",    "")
plot_reliability_vs_avg_transfer(M_info, reliability, "plots/reliability_vs_transfer_infoxlm_large.svg", "")

plot_reliability_vs_pairwise_transfer(A,      reliability, "plots/reliability_vs_pairwise_all_models.svg",    "")
plot_reliability_vs_pairwise_transfer(M_info, reliability, "plots/reliability_vs_pairwise_infoxlm_large.svg", "")

