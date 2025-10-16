# Compute per-language stats from your existing `data` (rows: models, columns: languages)
values = data.iloc[1:].replace(0, np.nan)  # drop the 'Mean' row; treat 0 as missing
labels = list(values.columns)
x = np.arange(len(labels))

means = values.mean(axis=0, skipna=True).values
n = values.count(axis=0).values
sem = values.std(axis=0, ddof=1, skipna=True).values / np.sqrt(n)

# Sort languages by encoding performance (mean across models), descending
sort_idx = np.argsort(means)[::-1]
values = values.iloc[:, sort_idx]
means = means[sort_idx]
sem = sem[sort_idx]
labels = [labels[i] for i in sort_idx]
x = np.arange(len(labels))

# Plot: 1 bar per language (avg across models) + semi-transparent dots (each model×language)
fig, ax = plt.subplots(figsize=(0.6*len(labels), 5.5), dpi=300)

bars = ax.bar(
    x, means, yerr=sem, capsize=4,
    color='lightgray', edgecolor='black', linewidth=1.0, zorder=2
)

# Overlay one dot per model×language
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
ax.set_xlim(-0.5, len(labels)-0.5)
ax.set_xticks(x)
ax.set_xticklabels([lang_dict[l] for l in labels], rotation=45, ha='right')
ax.set_ylabel('R')
ax.set_ylim(-.25, .66)
plt.tight_layout()
plt.show()

lang_order = labels
print("LANG_ORDER =", lang_order)


LANG_ORDER_S1 = labels                  # the sorted order from Study I
VALUES_S1 = values.copy()               # models × languages matrix (after reordering)
print("LANG_ORDER_S1 =", LANG_ORDER_S1)

VALUES_S1.to_csv("/home/dev/Downloads/values_s1.csv")
