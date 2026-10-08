"""
Network-summary bar plot: per approach (mono / multi), one bar per
{Random baseline, LH language, RH language, MD network}, with one dot per
model and SEM errorbars on the bar height.

Note: circshift (random) result files only exist for the LH language network,
so the random bar is the LH-lang circshift score (best layer per shift,
averaged across the 4 shifts, matching `compute_significance_for_froi`).

Output: 2 PNGs in this folder (one per approach).
"""

import os
import sys
import pickle
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

# numpy 2.x pickle compat shim (matches plot_encoding_*.py)
sys.modules["numpy._core.numeric"] = np.core.numeric

ROOT = "/home/andrea/Documents/PhD/Alice"
OUTDIR = os.path.join(ROOT, "other", "network_summary_plot")
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# Model list (matches plot_encoding_multi.py / plot_encoding_mono.py)
# ---------------------------------------------------------------------------
model_names = [
    "nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B",
    "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm",
    "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta",
    "mt5_small", "mt5_base", "mt5_large", "mgpt",
    "xglm_small", "xglm_med", "xglm_large", "xglm_xl",
]

# ---------------------------------------------------------------------------
# Loading + best-layer helpers (match plot_encoding_multi.py semantics)
# ---------------------------------------------------------------------------
def patched_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load(model_prefix, froi="all", monol=False, random=False,
         md=False, rh=False, multitrain=True):
    if monol:
        if md:
            fn = f"results/monolingual_md_{model_prefix}_{froi}"
        elif rh:
            fn = f"results/monolingual_rh_{model_prefix}_{froi}"
        elif random:
            fn = f"results/monolingual_{model_prefix}_{froi}_circshift"
        else:
            fn = f"results/monolingual_{model_prefix}_{froi}"
    else:
        mtpfx = "multitrain_" if multitrain else ""
        if md:
            fn = f"results/multilingual_{mtpfx}md_{model_prefix}_{froi}"
        elif rh:
            fn = f"results/multilingual_{mtpfx}rh_{model_prefix}_{froi}"
        elif random:
            fn = f"results/multilingual_{mtpfx}{model_prefix}_{froi}_circshift"
        else:
            fn = f"results/multilingual_{mtpfx}{model_prefix}_{froi}"
    return patched_load(fn)


def best_layer_score(res_dict, colname):
    """Mean (across languages/folds) of the best-performing layer."""
    means = [v[colname].mean() for v in res_dict.values()]
    return means[int(np.argmax(means))]


# ---------------------------------------------------------------------------
# Compute the four per-model score vectors for a given approach
# ---------------------------------------------------------------------------
def collect_scores(monol):
    """Returns dict: network -> list of per-model scores (length = n models)."""
    colname = "m" if monol else "r"
    out = {"Random": [], "LH lang": [], "RH lang": [], "MD"     : []}

    for model in model_names:
        # actual scores for each network at the "all" aggregate
        out["LH lang"].append(
            best_layer_score(load(model, froi="all", monol=monol), colname)
        )
        out["RH lang"].append(
            best_layer_score(load(model, froi="all", rh=True, monol=monol), colname)
        )
        out["MD"].append(
            best_layer_score(load(model, froi="all", md=True, monol=monol), colname)
        )

        # random baseline: LH-lang circshift "all"; avg across the 4 shifts
        rand_dict = load(model, froi="all", random=True, monol=monol)
        per_shift = [
            best_layer_score(rand_dict[shift], colname)
            for shift in [26, 52, 78, 104]
        ]
        out["Random"].append(float(np.mean(per_shift)))

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COLORS = {
    "Random":  "#9e9e9e",   # gray
    "LH lang": "#d62728",   # red
    "RH lang": "#d62728",   # red but lower alpha (set per-call)
    "MD":      "#1f77b4",   # blue
}
ALPHAS = {"Random": 0.85, "LH lang": 0.95, "RH lang": 0.45, "MD": 0.95}


def _bar_with_dots(ax, position, values, color, alpha, label, jitter_seed):
    mean_v = float(np.mean(values))
    sem_v  = float(np.std(values, ddof=1) / sqrt(len(values)))
    ax.bar(position, mean_v, width=0.7, color=color, alpha=alpha,
           linewidth=0, zorder=2, label=label)
    ax.errorbar(position, mean_v, yerr=sem_v, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5,
                zorder=4)
    rng = np.random.default_rng(jitter_seed)
    jitter = rng.uniform(-0.18, 0.18, size=len(values))
    ax.scatter(np.full_like(values, position, dtype=float) + jitter,
               values, s=28, color="black", alpha=0.55, zorder=5,
               edgecolor="white", linewidth=0.4)
    return mean_v, sem_v


DISPLAY_LABELS = {
    "Random":  "Random",
    "LH lang": "Language\n(left)",
    "RH lang": "Language\n(right)",
    "MD":      "Formal reasoning",
}


def plot_four_bars(scores, savepath, ylim=(0, 0.42)):
    networks = ["Random", "LH lang", "RH lang", "MD"]
    positions = [0, 1.4, 2.4, 3.8]  # gap between Random and the network bars

    plt.figure(figsize=(8.5, 6), dpi=200)
    sns.set_context("talk")
    ax = plt.gca()

    for pos, net, seed in zip(positions, networks, range(4)):
        _bar_with_dots(ax, pos, np.asarray(scores[net]),
                       COLORS[net], ALPHAS[net], net, jitter_seed=seed)

    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_LABELS[n] for n in networks], fontsize=16)
    ax.set_ylabel("brain prediction", fontsize=18, labelpad=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(*ylim)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savepath, format="png", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for label, monol in [("mono", True), ("multi", False)]:
        print(f"\n=== {label} ===")
        scores = collect_scores(monol=monol)
        for net, vals in scores.items():
            print(f"  {net:8s}  mean={np.mean(vals):.4f}  "
                  f"sem={np.std(vals, ddof=1)/sqrt(len(vals)):.4f}  "
                  f"n={len(vals)}")

        # save the per-model table for reproducibility
        pd.DataFrame({"model": model_names, **scores}).to_csv(
            os.path.join(OUTDIR, f"{label}_scores.csv"), index=False
        )

        plot_four_bars(
            scores,
            savepath=os.path.join(OUTDIR, f"{label}_4bars.png"),
        )
        print(f"  saved -> {label}_4bars.png, {label}_scores.csv")

    print("\nDone.")
