# Network summary bar plot

Self-contained plotting script that produces network-level summary bars
across all 20 models, separately for the monolingual and multilingual
encoding analyses. Reads from `../../results/` and writes PNGs into this
folder. Touches nothing else in the repo.

## Run

```
source /home/andrea/miniforge3/etc/profile.d/conda.sh && conda activate analysis
python other/network_summary_plot/network_summary_plot.py
```

## Outputs

- `mono_4bars.png`, `multi_4bars.png` — four bars per panel labelled
  Random | Language left | Language right | Formal reasoning. Each bar
  is the mean across the 20 models on the `"all"` aggregate fROI of
  that network; dots are individual models; errorbars are SEM across
  models.
- `mono_scores.csv`, `multi_scores.csv` — per-model scores feeding the
  plots (one row per model, columns: Random, LH lang, RH lang, MD).

## Per-model score definition (matches existing scripts)

For each model and each fROI = `"all"` of a network, the score is the
mean over the cross-validation result of the best-performing layer
(`get_best_layerwise` in `plot_encoding_*.py`). Mono uses column `"m"`
(per-language mean), multi uses column `"r"`.

## Random baseline

Circular-shift result files (`*_circshift`) only exist for the LH language
network. The random bar therefore uses the LH-lang `"all"` circshift
results, taking the best-layer score for each of the four shifts
(26, 52, 78, 104) and averaging across shifts (matching the convention
used inside `compute_significance_for_froi`).

The **mono** random baseline is high (R ≈ 0.14) because `get_best_layerwise`
picks the highest-performing layer of the shifted data — a selection step
that upward-biases the null. The Fisher-z significance test in
`plot_encoding_mono.py` / `plot_encoding_multi.py` controls for this on a
per-model basis (paired comparison), but the *visual* baseline still ends
up large. The **multi** baseline (R ≈ 0.04) is much smaller because
results are averaged across many cross-lingual fits before the best-layer
selection.
