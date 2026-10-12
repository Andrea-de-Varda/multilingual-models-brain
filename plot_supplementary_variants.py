"""
Study I supplementary variants of Figure 2A (reads the results written by fit_encoding.py):
  - Supplementary Figure 3 (SI S5): median-performing layer instead of best layer, WITHIN and ACROSS;
  - Supplementary Figure 4 (SI S7): fROIs defined with the native-language localizer, WITHIN and ACROSS;
  - Supplementary Figure 10A (SI S14): encoding models trained on a single language and tested on each of the
    others (best layer).
For each variant it prints the network-level and fROI-level means and the significance against the
circular-shift baselines (same procedure as plot_encoding_mono.py / plot_encoding_multi.py: Fisher r-to-z per
language, Stouffer across languages and then across the four shifts, with the layer selected in the same way for
the real and the shifted results). For the native-language fROIs, significance is reported once the circular-shift
baselines have been computed with `python fit_encoding.py --mode native-random`.
Figures (drawn exactly like the Figure 2A panels, one file per condition): plots/SI_median_layer_{within,across}.svg,
plots/SI_native_frois_{within,across}.svg, plots/SI_single_language_training.svg
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


def figure_2a_panel(out, path, brackets):
    """One panel drawn exactly like Figure 2A (plot_encoding_mono.py / plot_encoding_multi.py, "DEFINITIVE FINAL
    PLOT"): network-level score per model with SE across languages (large dots) and the five fROIs (small markers).
    The model-type brackets are drawn on the WITHIN (top) panel only, as in Figure 2A."""
    import seaborn as sns
    main = out["all"]
    plt.figure(figsize=(24 * .7, 11.5 * .7), dpi=300)
    sns.set_context("talk")
    ax = plt.gca()
    for i, (m, pos) in enumerate(zip(MODELS, POSITIONS)):
        ax.errorbar(pos, main.r[m], yerr=main.se[m], fmt='o', color=PALETTE[FAMILY[i]],
                    markersize=16, alpha=0.9, lw=3, capsize=5)
    for froi in FROIS:
        for i, (m, pos) in enumerate(zip(MODELS, POSITIONS)):
            ax.plot(pos, out[froi].r[m], marker=MARKERS[froi], color=PALETTE[FAMILY[i]], markersize=9, alpha=0.4, lw=0)
    if brackets:
        def add_bracket(pos1, pos2, text, y_offset=0.05, weight="normal"):
            y = main.r.max() + y_offset
            ax.plot([pos1, pos1, pos2, pos2], [y, y + 0.02, y + 0.02, y], color='black', lw=2)
            ax.text((pos1 + pos2) / 2, y + 0.03, text, ha='center', va='bottom', fontsize=19, weight=weight)
        P = POSITIONS
        add_bracket(P[0], P[2], 'translation', y_offset=-.07)
        add_bracket(P[3], P[5], 'contrastive', y_offset=.02)
        add_bracket(P[0], P[5], 'explicit', y_offset=.12, weight="bold")
        add_bracket(P[6], P[11], 'masked LM', y_offset=-.03)
        add_bracket(P[12], P[14], 'span corr.', y_offset=-.09)
        add_bracket(P[15], P[19], 'causal LM', y_offset=.12)
        add_bracket(P[6], P[19], 'implicit', y_offset=.21, weight="bold")
    ax.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.3)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(0, 0.75)
    plt.xticks(POSITIONS, labels=NICE, rotation=45, ha='right', fontsize=22)
    plt.yticks([.1, .2, .3, .4, .5, .6, .7], fontsize=23)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(path, format="svg", bbox_inches="tight")
    plt.close()


# Supplementary Figure 3 / SI S5: median layer
med_w = summarise("results/monolingual_{m}_{f}", "m", median_layer, "results/monolingual_{m}_{f}_circshift")
med_a = summarise("results/multilingual_multitrain_{m}_{f}", "r", median_layer,
                  "results/multilingual_multitrain_{m}_{f}_circshift")
report("Median layer, WITHIN (SI S5)", med_w)
report("Median layer, ACROSS (SI S5)", med_a)
figure_2a_panel(med_w, "plots/SI_median_layer_within.svg", brackets=True)
figure_2a_panel(med_a, "plots/SI_median_layer_across.svg", brackets=False)

# Supplementary Figure 4 / SI S7: native-language fROIs (best layer); significance only if the circular-shift
# baselines have been computed (python fit_encoding.py --mode native-random)
has_null = os.path.isfile("results/monolingual_native_distilmbert_all_circshift")
nat_w = summarise("results/monolingual_native_{m}_{f}", "m", best_layer,
                  "results/monolingual_native_{m}_{f}_circshift" if has_null else None)
nat_a = summarise("results/multilingual_multitrain_native_{m}_{f}", "r", best_layer,
                  "results/multilingual_multitrain_native_{m}_{f}_circshift" if has_null else None)
report("Native-language fROIs, WITHIN (SI S7)", nat_w)
report("Native-language fROIs, ACROSS (SI S7)", nat_a)
figure_2a_panel(nat_w, "plots/SI_native_frois_within.svg", brackets=True)
figure_2a_panel(nat_a, "plots/SI_native_frois_across.svg", brackets=False)

# Supplementary Figure 10A / SI S14: training on a single language, testing on each of the others (best layer)
single = summarise("results/multilingual_{m}_{f}", "r", best_layer, "results/multilingual_{m}_{f}_circshift")
report("Single-language training (SI S14)", single)
figure_2a_panel(single, "plots/SI_single_language_training.svg", brackets=False)
