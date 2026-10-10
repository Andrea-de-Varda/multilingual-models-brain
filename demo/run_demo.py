"""
Replication demo: Study I encoding analyses (WITHIN and ACROSS) for one model, DistilmBERT, from the raw
transcriptions and fMRI time series in this repository, compared with the values reported in the paper
(Figure 2A/2B; Source Data Fig. 2).

    python demo/run_demo.py           # quick: all layers WITHIN (+ significance), ACROSS at the paper's best layer
    python demo/run_demo.py --full    # all layers in both conditions, with significance (GPU recommended)

Steps: (1) word embeddings of the 12 Study I passages from DistilmBERT (downloaded from Hugging Face;
cached in demo/cache/), (2) alignment of the embeddings with the 130 fMRI volumes, (3) cross-participant
encoding models within each language (WITHIN) and zero-shot across languages (ACROSS), (4) significance
against circularly shifted fMRI responses (Fisher r-to-z + Stouffer). Runtime: about 5 minutes for the
embeddings on a laptop CPU, then about 1 minute (GPU) to 10 minutes (CPU) in quick mode.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_utils as du  # noqa: E402

# values reported in the paper for DistilmBERT, language network ('all'); Source Data Fig. 2 (Fig2B_layerwise, Fig2A_summary)
PAPER = {
    "WITHIN": {"layers": [0.030445, 0.083529, 0.109169, 0.131199, 0.142502, 0.155845, 0.173093],
               "best_layer": 6, "z": 3.321278, "p": 8.960637e-04},
    "ACROSS": {"layers": [0.001822, 0.005299, 0.039226, 0.081957, 0.139847, 0.114869, 0.097520],
               "best_layer": 4, "z": 6.903273, "p": 5.081786e-12},
}
TOL = 0.005  # tolerance on r (embeddings recomputed with a different library version can differ slightly)


def run_condition(name, X_layers, d, layers, col, fn):
    print(f"\n=== {name} ===")
    real, shifted = {}, {s: {} for s in du.SHIFTS}
    for layer in layers:
        t = time.time()
        real[layer] = fn(X_layers[layer], d)
        for s in du.SHIFTS:
            shifted[s][layer] = fn(X_layers[layer], d, shift=s)
        paper = PAPER[name]["layers"][layer]
        ok = abs(real[layer][col].mean() - paper) < TOL
        print(f"  layer {layer}: mean r = {real[layer][col].mean():.4f}  (paper {paper:.4f}) "
              f"{'OK' if ok else 'DIFFERENT'}   null (mean of 4 shifts) = "
              f"{np.mean([shifted[s][layer][col].mean() for s in du.SHIFTS]):+.4f}   [{time.time() - t:.0f} s]")
    best = max(real, key=lambda l: real[l][col].mean())
    z, p = du.significance(real, shifted, col)
    print(f"  best layer: {best} (paper {PAPER[name]['best_layer']}), r = {real[best][col].mean():.4f} "
          f"SE = {real[best][col].std(ddof=1) / np.sqrt(len(real[best])):.4f}")
    if len(layers) == 1:
        print("  (only one layer evaluated: the null uses the same layer; the paper picks the best layer of each null run)")
    print(f"  Stouffer z = {z:.2f}, two-sided p = {p:.2e}   (paper: z = {PAPER[name]['z']:.2f}, p = {PAPER[name]['p']:.2e})")
    return real


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="run all layers in the ACROSS condition too")
    ap.add_argument("--backend", default=None, choices=["torch_cuda", "torch", "numpy"],
                    help="himalaya backend (default: GPU if available, otherwise torch on CPU)")
    args = ap.parse_args()

    print("himalaya backend:", du.set_backend(args.backend))
    d = du.load_fmri()
    print("1. Word embeddings (DistilmBERT, 7 layers incl. the input embedding layer)")
    t = time.time()
    emb = du.get_all_embeddings(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
    print(f"   done in {time.time() - t:.0f} s")
    print("2. Aligning word embeddings with the 130 fMRI volumes")
    X_layers = {layer: {lang: du.preproc_align(lang, emb[lang][layer]) for lang in du.LANGS} for layer in range(7)}

    print("3. Encoding models (network-level response of the left-hemisphere language network)")
    run_condition("WITHIN", X_layers, d, range(7), "m", du.within_encoding)
    across_layers = range(7) if args.full else [PAPER["ACROSS"]["best_layer"]]
    run_condition("ACROSS", X_layers, d, across_layers, "r", du.across_encoding)


if __name__ == "__main__":
    main()
