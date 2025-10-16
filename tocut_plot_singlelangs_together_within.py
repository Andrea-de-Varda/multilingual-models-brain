# One figure, single axes:
# Left block = Study 1 (from VALUES_S1), right block = Study 2 (from VALUES_S2).
# Each block: bars sorted independently by mean R (desc), with semi-transparent dots per model×language.
# No saving.

lang_dict = {'fr': 'French',
 'ta': 'Tamil',
 'es': 'Spanish',
 'tr': 'Turkish',
 'vi': 'Vietnamese',
 'mr': 'Marathi',
 'af': 'Afrikaans',
 'nl': 'Dutch',
 'no': 'Norwegian',
 'fa': 'Farsi',
 'ro': 'Romanian',
 'lt': 'Lithuanian'}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If not already in memory, load Study 2 matrix (rows=models, cols=languages)
VALUES_S1 = pd.read_csv("/home/dev/Downloads/values_s1.csv", index_col=0)
VALUES_S2 = pd.read_csv("/home/dev/Downloads/values_s2.csv", index_col=0)

# Combined plot: Study 1 (VALUES_S1) on the left, Study 2 (VALUES_S2) on the right.
# Each block is sorted independently by mean R (desc).
# Bars = mean across models per language (±SEM); dots = each model×language (jittered).
# Study 1 columns are language *codes*; we map them to pretty names via `lang_dict`.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assumes VALUES_S1 is in memory; load Study 2:
VALUES_S2 = pd.read_csv("/home/dev/Downloads/values_s2.csv", index_col=0)

# Provide mapping for Study 1 codes -> pretty names. If you already have lang_dict, reuse it.
# Example:
# lang_dict = {"en": "English", "it": "Italian", "de": "German", "ta": "Tamil", ...}

def per_lang_stats(M):
    means = M.mean(axis=0, skipna=True)
    n     = M.count(axis=0)
    sem   = M.std(axis=0, ddof=1, skipna=True) / np.sqrt(np.maximum(n, 1))
    order = means.sort_values(ascending=False).index.tolist()
    return M[order], means[order].values, sem[order].values, order

# Sort each study independently
S1_sorted, means_s1, sem_s1, langs_s1_codes = per_lang_stats(VALUES_S1)
S2_sorted, means_s2, sem_s2, langs_s2_names = per_lang_stats(VALUES_S2)

# Display labels: map Study 1 codes -> pretty names if possible
def map_labels(codes, mapping):
    if mapping is None:
        return codes
    return [mapping.get(c, c) for c in codes]

# Use existing lang_dict if defined; else fallback identity
try:
    S1_labels = map_labels(langs_s1_codes, lang_dict)
except NameError:
    S1_labels = langs_s1_codes  # no mapping available; show codes

S2_labels = list(langs_s2_names)  # study 2 already in full names

# X positions: two groups with a gap
gap = 2.0
x1 = np.arange(len(langs_s1_codes))
x2 = np.arange(len(langs_s2_names)) + (len(langs_s1_codes) + gap)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Bars
bar_kw = dict(capsize=4, edgecolor='black', linewidth=1.0, zorder=2)
ax.bar(x1, means_s1, yerr=sem_s1, color='lightgray',     label='Study 1', **bar_kw)
ax.bar(x2, means_s2, yerr=sem_s2, color='lightsteelblue', label='Study 2', **bar_kw)

# Dots (jittered)
rng = np.random.default_rng(0)
for j, lang in enumerate(langs_s1_codes):
    y = S1_sorted[lang].values
    m = ~np.isnan(y)
    if m.any():
        ax.scatter(x1[j] + rng.uniform(-0.18, 0.18, m.sum()),
                   y[m], s=30, alpha=0.5,
                   facecolor='indianred', edgecolor='black', linewidth=0.4, zorder=3)

for j, lang in enumerate(langs_s2_names):
    y = S2_sorted[lang].values
    m = ~np.isnan(y)
    if m.any():
        ax.scatter(x2[j] + rng.uniform(-0.18, 0.18, m.sum()),
                   y[m], s=30, alpha=0.5,
                   facecolor='darkslateblue', edgecolor='black', linewidth=0.4, zorder=3)

# Axes
ax.axhline(0, color='black', lw=1.5, zorder=1)
xmin = -0.5
xmax = (x2[-1] + 0.5) if len(x2) else (len(x1) - 0.5)
ax.set_xlim(xmin, xmax)
ax.set_ylabel('R')
ax.set_ylim(-0.25, 0.66)

# X ticks: Study 1 (mapped), then Study 2 (as-is)
xticks  = np.concatenate([x1, x2])
xlabels = S1_labels + S2_labels
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=45, ha='right')

# Group headers
y_top = ax.get_ylim()[1]
ax.text(x1.mean(), y_top, "Study 1", ha='center', va='bottom', fontsize=20)
ax.text(x2.mean(), y_top, "Study 2", ha='center', va='bottom', fontsize=20)

plt.tight_layout()
plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load
VALUES_S1 = pd.read_csv("/home/dev/Downloads/values_s1.csv", index_col=0)
VALUES_S2 = pd.read_csv("/home/dev/Downloads/values_s2.csv", index_col=0)

# Map Study 1 codes → names
lang_dict = {'fr':'French','ta':'Tamil','es':'Spanish','tr':'Turkish','vi':'Vietnamese',
             'mr':'Marathi','af':'Afrikaans','nl':'Dutch','no':'Norwegian',
             'fa':'Farsi','ro':'Romanian','lt':'Lithuanian'}
S1 = VALUES_S1.rename(columns=lambda c: lang_dict.get(c, c))
S2 = VALUES_S2.copy()

# Use the UNION of languages (avoids empty intersection)
all_langs = sorted(set(S1.columns) | set(S2.columns))
S1 = S1.reindex(columns=all_langs)
S2 = S2.reindex(columns=all_langs)

# Combine models from both studies
combined = pd.concat([S1, S2], axis=0, ignore_index=True)

# Guard: ensure we have at least one finite value
if not np.isfinite(np.nanmin(combined.values)):
    raise ValueError("No finite values found. Check column names after mapping; print(S1.columns, S2.columns).")

# Stats across all models (both studies)
means = combined.mean(axis=0, skipna=True)
n     = combined.count(axis=0)
sem   = combined.std(axis=0, ddof=1, skipna=True) / np.sqrt(np.maximum(n, 1))

# Global sort by mean (desc)
lang_order = means.sort_values(ascending=False).index.tolist()
combined = combined[lang_order]
means = means[lang_order].values
sem = sem[lang_order].values

# X + figure
x = np.arange(len(lang_order))
fig, ax = plt.subplots(figsize=(0.6*max(len(lang_order), 1), 5.5), dpi=300)

# Set finite limits BEFORE axhline to avoid singular transforms
# y-lims from data with a small floor
data_min = np.nanmin(combined.values)
data_max = np.nanmax(combined.values)
ymin = min(-0.05, data_min if np.isfinite(data_min) else -0.05)
ymax = max(0.66, data_max if np.isfinite(data_max) else 0.66)
ax.set_ylim(ymin, ymax)
ax.set_xlim(-0.5, len(lang_order)-0.5 if len(lang_order) else 0.5)

# Bars
ax.bar(x, means, yerr=sem, capsize=4,
       color='lightsteelblue', edgecolor='black', linewidth=1.4, zorder=2)

# Dots
rng = np.random.default_rng(0)
for j, lang in enumerate(lang_order):
    y = combined[lang].values
    m = np.isfinite(y)
    if m.any():
        jitter = rng.uniform(-0.18, 0.18, size=m.sum())
        ax.scatter(np.full(m.sum(), x[j]) + jitter, y[m],
                   s=30, alpha=0.4, facecolor='blue',
                   edgecolor='black', linewidth=0.4, zorder=3)
    else:
        ax.text(x[j], 0, 'NA', ha='center', va='bottom', fontsize=11, color='black')

# Now safe to draw baseline
ax.axhline(0, color='black', lw=1.5, zorder=1)

# Ticks/labels
ax.set_xticks(x)
ax.set_xticklabels(lang_order, rotation=45, ha='right')
ax.set_ylabel('R')

plt.tight_layout()
plt.show()

