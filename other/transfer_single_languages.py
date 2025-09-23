#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import lang2vec.lang2vec as l2v
from statsmodels.stats.multitest import multipletests

TARGET_MODEL = "infoxlm_large" 
FROI         = "all"
langs_matrix = langs[:]
pretty_labels = [lang_dict[l] for l in langs_matrix]

def get_best_layer_matrix(model):
    res_dict = load(model, froi=FROI, monol=False, random=False,
                    md=False, rh=False, native=False, multitrain=False)
    layer_means = {k: v["r"].mean() for k, v in res_dict.items()}
    best_layer = max(layer_means, key=layer_means.get)
    df_best = res_dict[best_layer]

    M = pd.DataFrame(np.nan, index=langs_matrix, columns=langs_matrix, dtype=float)
    test_cols = [c for c in df_best.columns if c not in ("target_lang", "r")]
    for _, row in df_best.iterrows():
        tr = row["target_lang"]
        if tr not in langs_matrix:
            continue
        for t in test_cols:
            if t in langs_matrix:
                try:
                    M.at[tr, t] = float(row[t])
                except Exception:
                    pass
    return M, best_layer

def build_symmetric_matrix():
    if TARGET_MODEL == "ALL":
        mats = []
        for model in model_names:
            try:
                M, _ = get_best_layer_matrix(model)
                mats.append(M)
            except Exception as e:
                print(f"[WARN] Skipping {model}: {e}")
        stacked = np.stack([m.to_numpy() for m in mats], axis=2)
        A = np.nanmean(stacked, axis=2)
        title = "Average across models"
    else:
        M, best_layer = get_best_layer_matrix(TARGET_MODEL)
        A = M.to_numpy()
        title = f"{TARGET_MODEL} (best layer={best_layer})"

    # symmetrize
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            m = np.nanmean([A[i, j], A[j, i]])
            A[i, j] = m
            A[j, i] = m
    return A, title

def plot_heatmap(sym_df, title):
    A = sym_df.to_numpy()
    mask_upper = np.triu(np.ones_like(A, dtype=bool), k=1)
    absmax = np.nanmax(np.abs(A))

    plt.figure(figsize=(10.5*.7, 8.2*.7), dpi=300)
    sns.set_context("talk")
    ax = sns.heatmap(
        sym_df,
        mask=mask_upper,
        cmap="RdBu_r",
        vmin=-absmax, vmax=absmax, center=0,
        square=True,
        cbar_kws={"label": "r"},
        linewidths=0.3, linecolor="white"
    )
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(pretty_labels, rotation=0, ha="right", fontsize=14)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/xling_matrix.svg", format="svg", bbox_inches="tight")
    plt.show()

def make_comparison_df(sym_df):
    langs = sym_df.index.tolist()
    pairs, values = [], []
    for i in range(len(langs)):
        for j in range(i+1, len(langs)):
            pairs.append((langs[i], langs[j]))
            values.append(sym_df.iloc[i, j])
    comparison_df = pd.DataFrame(pairs, columns=["Lang1", "Lang2"])
    comparison_df["transfer"] = values
    return comparison_df

def add_lang2vec_distances(comparison_df):
    iso_codes = {
        "fa": "fas", "mr": "mar", "es": "spa", "ro": "ron", "fr": "fra", "lt": "lit",
        "af": "afr", "nl": "nld", "no": "nob", "tr": "tur", "vi": "vie", "ta": "tam",
    }
    comparison_df["iso1"] = comparison_df["Lang1"].map(iso_codes)
    comparison_df["iso2"] = comparison_df["Lang2"].map(iso_codes)

    syn, geo, pho, gen, inv, feat = [], [], [], [], [], []
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df)):
        syn.append(l2v.syntactic_distance(row["iso1"], row["iso2"]))
        geo.append(l2v.geographic_distance(row["iso1"], row["iso2"]))
        pho.append(l2v.phonological_distance(row["iso1"], row["iso2"]))
        gen.append(l2v.genetic_distance(row["iso1"], row["iso2"]))
        inv.append(l2v.inventory_distance(row["iso1"], row["iso2"]))
        feat.append(l2v.featural_distance(row["iso1"], row["iso2"]))

    comparison_df["syn"] = syn
    comparison_df["geo"] = geo
    comparison_df["pho"] = pho
    comparison_df["gen"] = gen
    comparison_df["inv"] = inv
    comparison_df["feat"] = feat
    return comparison_df

def run_correlations(comparison_df):
    results = []
    for col in ["syn", "geo", "pho", "gen", "inv", "feat"]:
        r, p = pearsonr(comparison_df["transfer"], comparison_df[col])
        results.append((col, r, p))

    cols, rs, ps = zip(*results)
    reject, pvals_corr, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")

    for c, r, p_raw, p_corr, rej in zip(cols, rs, ps, pvals_corr, reject):
        print(f"{c:>5}: r={r:.3f}, p_raw={p_raw:.3g}, p_corr={p_corr:.3g}, significant={rej}")

if __name__ == "__main__":
    A, title = build_symmetric_matrix()
    sym_df = pd.DataFrame(A, index=langs_matrix, columns=langs_matrix)

    diag_vals = np.diag(A)
    off_vals  = A[~np.eye(len(A), dtype=bool)]
    print(f"[diag mean]     {np.nanmean(diag_vals):.4f}")
    print(f"[off-diag mean] {np.nanmean(off_vals):.4f}")
    print(f"[overall mean]  {np.nanmean(A):.4f}")

    plot_heatmap(sym_df, title)

    comparison_df = make_comparison_df(sym_df)
    comparison_df = add_lang2vec_distances(comparison_df)
    run_correlations(comparison_df)

#   syn: r=-0.202, p_raw=0.103, p_corr=0.31, significant=False
#   geo: r=0.169, p_raw=0.175, p_corr=0.35, significant=False
#   pho: r=0.083, p_raw=0.506, p_corr=0.609, significant=False
#   gen: r=0.063, p_raw=0.613, p_corr=0.613, significant=False
#   inv: r=0.275, p_raw=0.0253, p_corr=0.152, significant=False
#  feat: r=-0.083, p_raw=0.508, p_corr=0.609, significant=False
 
# [diag mean]     0.3035
# [off-diag mean] 0.1263
# [overall mean]  0.1411