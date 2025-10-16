# Single plot: one bar per language (avg across models), one semi-transparent dot per model×language,
# where each dot is averaged across the four training conditions: main, natstor, control, pereira.
# Assumes `transf_results` with columns: train, condition, language, model, r_mean, ...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CORPORA = ['main', 'natstor', 'control', 'pereira']

# Optional: enforce a specific language order (leave as None to auto-order by mean desc)
# LANG_ORDER = ['Italian', 'German', 'Russian', ...]
LANG_ORDER = None

# 1) Filter and average across the 4 conditions for each (model, language)
df = transf_results[
    (transf_results['condition'] == 'experimental') &
    (transf_results['train'].isin(CORPORA))
][['language', 'model', 'r_mean']].copy()

df_avg = df.groupby(['model', 'language'], as_index=False)['r_mean'].mean()

# 2) Matrix: rows=models, cols=languages (values = mean across the 4 conditions)
values = df_avg.pivot(index='model', columns='language', values='r_mean')

# 3) Per-language summary across models
means = values.mean(axis=0, skipna=True)
n     = values.count(axis=0)
sem   = values.std(axis=0, ddof=1, skipna=True) / np.sqrt(np.maximum(n, 1))

# 4) Language order
if LANG_ORDER is not None:
    ordered = [l for l in LANG_ORDER if l in values.columns]
    extras  = [l for l in values.columns if l not in LANG_ORDER]
    lang_list = ordered + extras
else:
    # fallback: sort by mean (high → low)
    lang_list = means.sort_values(ascending=False).index.tolist()

values = values[lang_list]
means  = means[lang_list].values
sem    = sem[lang_list].values

# 5) Plot: bar per language (mean across models) + jittered dots per model×language
x = np.arange(len(lang_list))
fig, ax = plt.subplots(figsize=(0.6*len(lang_list), 5.5), dpi=300)

# Bars (mean ± SEM across models)
ax.bar(
    x, means, yerr=sem, capsize=4,
    color='lightgray', edgecolor='black', linewidth=1.0, zorder=2
)

# Dots (each model×language; averaged across the 4 conditions)
for j, lang in enumerate(lang_list):
    y_j = values[lang].values
    mask = ~np.isnan(y_j)
    if mask.any():
        jitter = np.random.uniform(-0.18, 0.18, size=mask.sum())
        ax.scatter(
            np.full(mask.sum(), x[j]) + jitter,
            y_j[mask],
            s=30, alpha=0.5,
            facecolor='indianred', edgecolor='black', linewidth=0.4,
            zorder=3
        )
    else:
        ax.text(x[j], 0, 'NA', ha='center', va='bottom', fontsize=11, color='black')

ax.axhline(0, color='black', lw=1.5, zorder=1)
ax.set_xlim(-0.5, len(lang_list)-0.5)
ax.set_xticks(x)
ax.set_xticklabels(lang_list, rotation=45, ha='right')
ax.set_ylabel('R')

ymin = np.nanmin(values.values) if np.isfinite(np.nanmin(values.values)) else 0
ax.set_ylim(-.25, .66)

plt.tight_layout()
plt.show()
