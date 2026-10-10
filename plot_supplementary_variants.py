"""
Study I supplementary variants of Figure 2A (reads the results written by fit_encoding.py):
  - Supplementary Figure 3 (SI S5): median-performing layer instead of best layer, WITHIN and ACROSS;
  - Supplementary Figure 4 (SI S7): fROIs defined with the native-language localizer, WITHIN and ACROSS;
  - Supplementary Figure 10A (SI S14): encoding models trained on a single language and tested on each of the
    others (best layer).
For each variant it prints the network-level and fROI-level means and the significance against the
circular-shift baselines (same procedure as plot_encoding_mono.py / plot_encoding_multi.py: Fisher r-to-z per
language, Stouffer across languages and then across the four shifts, with the layer selected in the same way for
the real and the shifted results). No shifted (null) runs were computed for the native-language fROIs, so no
significance is reported for them.
Figures: plots/SI_median_layer.svg, plots/SI_native_frois.svg, plots/SI_single_language_training.svg
"""
import os
import pickle
import sys
from os import chdir

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"
if "__file__" in globals():  # run as a script: resolve paths relative to the repository root
    chdir(os.path.dirname(os.path.abspath(__file__)))
# when run interactively, set the working directory to the repository root before running

MODELS = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large",
          "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base",
          "mt5_large", "mgpt", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
NICE = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$",
        "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$",
        "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
FAMILY = ["NLLB"] * 3 + ["XLM-Align", "InfoXLM", "InfoXLM"] + ["XLM-R"] * 3 + ["BERT", "BERT", "DeBERTa"] + \
         ["mT5"] * 3 + ["mGPT"] + ["XGLM"] * 4
PALETTE = {"BERT": "steelblue", "DeBERTa": "teal", "InfoXLM": "firebrick", "NLLB": "tomato", "XGLM": "forestgreen",
           "XLM-Align": "firebrick", "XLM-R": "lightsteelblue", "mGPT": "yellowgreen", "mT5": "darkorange"}
POSITIONS = [1, 2, 3, 4.5, 5.5, 6.5, 9, 10, 11, 12, 13, 14, 15.5, 16.5, 17.5, 19, 20, 21, 22, 23]
FROIS = ["Lang_LH_AntTemp", "Lang_LH_IFG", "Lang_LH_IFGorb", "Lang_LH_MFG", "Lang_LH_PostTemp"]
MARKERS = {"Lang_LH_AntTemp": "s", "Lang_LH_IFG": "o", "Lang_LH_IFGorb": "^", "Lang_LH_MFG": "v", "Lang_LH_PostTemp": "X"}
SHIFTS = [26, 52, 78, 104]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def best_layer(res, col):
    means = [res[k][col].mean() for k in res]
    return list(res)[int(np.argmax(means))]


def median_layer(res, col):
    # as get_median_layerwise in plot_encoding_mono.py: the layer with the median mean score (lower middle if even)
    means = [res[k][col].mean() for k in res]
    srt = sorted(means)
    med = srt[len(means) // 2] if len(means) % 2 == 1 else srt[len(means) // 2 - 1]
    return list(res)[means.index(med)]


def stouffer(z):
    z = np.sum(z) / np.sqrt(len(z))
    return z, 2 * norm.cdf(-abs(z))


def significance(res, null, col, select):
    real = res[select(res, col)][col].values
    z_shift = []
    for s in SHIFTS:
        rnd = null[s][select(null[s], col)][col].values
        z = (np.arctanh(real) - np.arctanh(rnd)) / np.sqrt(2 / (130 - 3))
        z_shift.append(stouffer(z)[0])
    return stouffer(z_shift)[1]


def summarise(pattern, col, select, null_pattern=None):
    """Score of each model (selected layer, mean across languages), SE, and p for 'all' and each fROI."""
    out = {}
    for froi in ["all"] + FROIS:
        rows = []
        for m in MODELS:
            res = load(pattern.format(m=m, f=froi))
            df = res[select(res, col)]
            p = significance(res, load(null_pattern.format(m=m, f=froi)), col, select) if null_pattern else np.nan
            rows.append([df[col].mean(), df[col].std() / np.sqrt(len(df)), p])
        out[froi] = pd.DataFrame(rows, columns=["r", "se", "p"], index=MODELS)
    return out


def report(name, out):
    print(f"\n=== {name} ===")
    for froi, df in out.items():
        sig = "n/a (no null runs)" if df.p.isna().all() else f"{(df.p < .05).sum()}/20 p<.05, {(df.p < .001).sum()}/20 p<.001"
        print(f"  {froi:18s} mean r = {df.r.mean():.3f} (range {df.r.min():.3f}-{df.r.max():.3f}, SE {df.se.min():.3f}-{df.se.max():.3f}); {sig}")


def dotplot(panels, path):
    fig, axes = plt.subplots(len(panels), 1, figsize=(24 * .7, 6 * .7 * len(panels)), dpi=300, sharex=True)
    for ax, (label, out) in zip(np.atleast_1d(axes), panels):
        main = out["all"]
        for i, (m, pos) in enumerate(zip(MODELS, POSITIONS)):
            alpha = 0.9 if (np.isnan(main.p[m]) or main.p[m] < .05) else 0.3  # semi-transparent: n.s.
            ax.errorbar(pos, main.r[m], yerr=main.se[m], fmt="o", color=PALETTE[FAMILY[i]], markersize=14,
                        alpha=alpha, lw=3, capsize=5)
            for froi in FROIS:
                ax.plot(pos, out[froi].r[m], marker=MARKERS[froi], color=PALETTE[FAMILY[i]], markersize=8, alpha=0.4, lw=0)
        ax.set_ylabel("R")
        ax.text(1.01, 0.5, label, transform=ax.transAxes, rotation=90, va="center", fontsize=16)
        ax.grid(axis="y", linestyle="--", linewidth=1.5, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(POSITIONS, NICE, rotation=45, ha="right")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(path, format="svg", bbox_inches="tight")
    plt.close()


# Supplementary Figure 3 / SI S5: median layer
med_w = summarise("results/monolingual_{m}_{f}", "m", median_layer, "results/monolingual_{m}_{f}_circshift")
med_a = summarise("results/multilingual_multitrain_{m}_{f}", "r", median_layer,
                  "results/multilingual_multitrain_{m}_{f}_circshift")
report("Median layer, WITHIN (SI S5)", med_w)
report("Median layer, ACROSS (SI S5)", med_a)
dotplot([("WITHIN", med_w), ("ACROSS", med_a)], "plots/SI_median_layer.svg")

# Supplementary Figure 4 / SI S7: native-language fROIs (best layer; no null runs available)
nat_w = summarise("results/monolingual_native_{m}_{f}", "m", best_layer)
nat_a = summarise("results/multilingual_multitrain_native_{m}_{f}", "r", best_layer)
report("Native-language fROIs, WITHIN (SI S7)", nat_w)
report("Native-language fROIs, ACROSS (SI S7)", nat_a)
dotplot([("WITHIN", nat_w), ("ACROSS", nat_a)], "plots/SI_native_frois.svg")

# Supplementary Figure 10A / SI S14: training on a single language, testing on each of the others (best layer)
single = summarise("results/multilingual_{m}_{f}", "r", best_layer, "results/multilingual_{m}_{f}_circshift")
report("Single-language training (SI S14)", single)
dotplot([("single-language training", single)], "plots/SI_single_language_training.svg")
