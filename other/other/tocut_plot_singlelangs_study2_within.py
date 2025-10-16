# One bar per language (avg across models) + semi-transparent dots per model×language
# Re-rank languages by average performance (descending). No saving.

sns.set_context("talk")

langs = list(within_all["language"].unique())
r_lang = {lang: [] for lang in langs}

# Build matrix: rows=models, cols=languages (values=r_mean)
for model in model_names:
    sub = within_all[within_all.model == model]
    for lang in langs:
        therow = sub[sub.language == lang]
        r = therow["r_mean"].values[0] if not therow.empty else np.nan
        r_lang[lang].append(r)

values = pd.DataFrame(r_lang)  # shape: (n_models, n_langs)

# Stats across models
means = values.mean(axis=0, skipna=True)
n = values.count(axis=0)
sem = values.std(axis=0, ddof=1, skipna=True) / np.sqrt(np.maximum(n, 1))

# Sort languages by mean performance (high → low)
lang_order = means.sort_values(ascending=False).index.tolist()
values = values[lang_order]
means = means[lang_order].values
sem = sem[lang_order].values

# Plot
x = np.arange(len(lang_order))
fig, ax = plt.subplots(figsize=(0.6*len(lang_order), 5.5), dpi=300)

# Bars = average across models per language (with SEM)
ax.bar(
    x, means, yerr=sem, capsize=4,
    color='lightgray', edgecolor='black', linewidth=1.0, zorder=2
)

# Dots = each model×language value (semi-transparent, jittered)
for j, lang in enumerate(lang_order):
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
ax.set_xlim(-0.5, len(lang_order)-0.5)
ax.set_xticks(x)
ax.set_xticklabels(lang_order, rotation=45, ha='right')
ax.set_ylabel('R')
bottom_val = np.nanmin(values.values) if np.isfinite(np.nanmin(values.values)) else 0
ax.set_ylim(-.25, .66)

plt.tight_layout()
plt.show()

VALUES_S2 = values.copy()               # models × languages matrix (before/after reordering—order won’t matter)
VALUES_S2.to_csv("/home/dev/Downloads/values_s2.csv")
