LANG_ORDER = ['ta', 'fr', 'fa', 'mr', 'vi', 'lt', 'af', 'nl', 'tr', 'es', 'ro', 'no']


# Build the table from your new pipeline
values = data.iloc[1:].replace(0, np.nan)  # drop 'Mean'; treat 0 as missing

# Reorder columns to match the first plot (keep only those present; extras go after)
ordered = [l for l in LANG_ORDER if l in values.columns]
extras  = [l for l in values.columns if l not in LANG_ORDER]
values  = values[ordered + extras]

labels = list(values.columns)
x = np.arange(len(labels))

# Stats across models
means = values.mean(axis=0, skipna=True).values
n = values.count(axis=0).values
sem = values.std(axis=0, ddof=1, skipna=True).values / np.sqrt(np.maximum(n, 1))

# Plot: 1 bar per language + semi-transparent dots for each model×language
fig, ax = plt.subplots(figsize=(0.6*len(labels), 5.5), dpi=300)

ax.bar(
    x, means, yerr=sem, capsize=4,
    color='lightgray', edgecolor='black', linewidth=1.0, zorder=2
)

for j, lang in enumerate(labels):
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
ax.set_ylim(-.25, .66)
ax.set_xticks(x)
ax.set_xticklabels([lang_dict[l] for l in labels], rotation=45, ha='right')
ax.set_ylabel('R')
ax.set_ylim(bottom=min(-0.05, np.nanmin(values.values) if np.isfinite(np.nanmin(values.values)) else 0))

plt.tight_layout()
plt.show()  # no saving