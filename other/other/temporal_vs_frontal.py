# =========================
# LH spatial plot: WITHIN vs ACROSS (hierarchy + brackets)
# =========================

# --- config ---
roi_short_order = list(reversed(['IFGorb', 'IFG', 'MFG', 'AntTemp', 'PostTemp']))  # ['PostTemp','AntTemp','MFG','IFG','IFGorb']
roi_full_map = {s: f'Lang_LH_{s}' for s in ['IFGorb','IFG','MFG','AntTemp','PostTemp']}
roi_labels = {s: s for s in roi_short_order}  # tick labels as the short names

# group structure (display order)
groups = [
    ("WITHIN", "Native",        None,               dict(monol=True,  native=True,  multitrain=True)),
    ("WITHIN", "English",       None,               dict(monol=True,  native=False, multitrain=True)),
    ("ACROSS", "Native",  "Train N-1",             dict(monol=False, native=True,  multitrain=True)),
    ("ACROSS", "Native",  "Train 1",               dict(monol=False, native=True,  multitrain=False)),
    ("ACROSS", "English", "Train N-1",             dict(monol=False, native=False, multitrain=True)),
    ("ACROSS", "English", "Train 1",               dict(monol=False, native=False, multitrain=False)),
]

# colors per subgroup
subgroup_color = {
    ("WITHIN","Native",None):          "navy",
    ("WITHIN","English",None):         "cornflowerblue",
    ("ACROSS","Native","Train N-1"):   "tomato",
    ("ACROSS","Native","Train 1"):     "lightsalmon",
    ("ACROSS","English","Train N-1"):  "forestgreen",
    ("ACROSS","English","Train 1"):    "yellowgreen",
}

def _best_for_combo(model, froi, opts):
    # choose correct column name depending on monolingual vs across-languages
    col = "m" if opts.get("monol", False) else "r"
    res = load(model, froi=froi,
               monol=opts.get("monol", False),
               split_context=False,
               random=False,
               md=False,
               rh=False,
               native=opts.get("native", False),
               multitrain=opts.get("multitrain", True))
    return get_best_layerwise(res, colname=col, give_mean=True, give_all=False)

# assemble rows
records = []
for top, sub, subsub, opts in groups:
    for roi_short in roi_short_order:
        froi = roi_full_map[roi_short]
        points = []
        for model in model_names:
            try:
                points.append(_best_for_combo(model, froi, opts))
            except Exception:
                # if a model/froi is missing for a combo, skip that model
                continue
        if len(points) == 0:
            mean_r, se_r = np.nan, np.nan
        else:
            mean_r = float(np.mean(points))
            se_r   = float(np.std(points) / np.sqrt(len(points)))
        records.append({
            "top": top, "sub": sub, "subsub": subsub,
            "roi_short": roi_short, "froi": froi,
            "r": mean_r, "se": se_r,
            "all_points": points,
            "color": subgroup_color[(top, sub, subsub)]
        })

df_lh = pd.DataFrame(records)

# ---- lay out x positions in hierarchical blocks ----
spacing = 1.0            # within-subgroup spacing between adjacent ROIs
gap_between_sub = 1.2    # extra gap between subgroups (e.g., WITHIN: Native vs English)
gap_between_top = 2.0    # extra gap between WITHIN and ACROSS

positions = []
tick_labels = []
x = 0.0

# store index ranges for brackets
ranges_top = {}      # {top: (xmin, xmax)}
ranges_sub = {}      # {(top,sub): (xmin,xmax)}
ranges_subsub = {}   # {(top,sub,subsub): (xmin,xmax)}

for top in ["WITHIN","ACROSS"]:
    top_start = None
    # iterate subgroups in the order defined above
    for (t, sub, subsub, _) in [g for g in groups if g[0] == top]:
        subgroup_start = x
        if (top, sub) not in ranges_sub:
            # remember the first time we see this (top, sub)
            ranges_sub[(top, sub)] = [np.inf, -np.inf]
        # place the 5 ROIs
        for roi_short in roi_short_order:
            # handle None (stored as NaN) correctly
            if subsub is None:
                mask_subsub = df_lh["subsub"].isna()
            else:
                mask_subsub = df_lh["subsub"].eq(subsub)
            mask = (
                df_lh["top"].eq(top)
                & df_lh["sub"].eq(sub)
                & mask_subsub
                & df_lh["roi_short"].eq(roi_short)
            )
            row = df_lh[mask]
            if len(row) != 1:
                raise ValueError(f"Missing or duplicate row for {(top, sub, subsub, roi_short)}; got {len(row)}")
            idx = row.index[0]
            positions.append(x)
            tick_labels.append(roi_labels[roi_short])
            df_lh.loc[idx, "pos"] = x
            x += spacing
        subgroup_end = x - spacing
        # update (top,sub) range
        ranges_sub[(top, sub)][0] = min(ranges_sub[(top, sub)][0], subgroup_start)
        ranges_sub[(top, sub)][1] = max(ranges_sub[(top, sub)][1], subgroup_end)
        # record (top,sub,subsub)
        ranges_subsub[(top, sub, subsub)] = (subgroup_start, subgroup_end)
        # gap between sub-subgroups
        x += gap_between_sub
        if top_start is None:
            top_start = subgroup_start
    top_end = x - gap_between_sub
    ranges_top[top] = (top_start, top_end)
    # gap between WITHIN and ACROSS
    x += gap_between_top


df_lh["pos"] = df_lh["pos"].astype(float)

# ---- plot ----
plt.figure(figsize=(16*.8, 7*.8), dpi=300)
ax = plt.gca()
sns.set_context("talk")

# light grid
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.3)

# scatter all model points per cell (vertical stacks)
for _, row in df_lh.iterrows():
    xs = [row["pos"]]*len(row["all_points"])
    ax.plot(xs, row["all_points"], marker='o', markersize=5, lw=0,
            alpha=0.15, color=row["color"], zorder=2)

# mean ± SE (black error + colored mean marker)
ax.errorbar(df_lh["pos"], df_lh["r"], yerr=df_lh["se"], fmt='none',
            ecolor='black', elinewidth=2.3, capsize=3, capthick=1.1, zorder=1)
ax.scatter(df_lh["pos"], df_lh["r"], s=90, c=df_lh["color"], edgecolor='none', zorder=1)

# ticks/labels
ax.set_xticks(positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
ax.set_ylabel("R", fontsize=16)
ax.set_ylim(-0.05, 0.55)
plt.yticks(fontsize=12)
sns.despine()
plt.tight_layout()

# ---- brackets ----
def add_bracket(ax, x0, x1, text, y, height=0.018, lw=1.4, fontsize=12, weight='normal'):
    """Draw a bracket from x0 to x1 with label above."""
    ax.plot([x0, x0, x1, x1], [y, y+height, y+height, y], color='black', lw=lw, clip_on=False)
    ax.text((x0+x1)/2, y+height*1.35, text, ha='center', va='bottom', fontsize=fontsize, weight=weight)

y_base = df_lh['r'].max() + 0.06
step   = 0.075  # vertical separation between hierarchy levels

# Level 1: sub-subgroups (ACROSS has Train N-1 / Train 1; WITHIN has only one level deep—still label Native/English directly at sub level)
for (top, sub, subsub), (x0, x1) in ranges_subsub.items():
    if top == "ACROSS":  # draw only the Train brackets
        add_bracket(ax, x0, x1, subsub, y=y_base)

# Level 2: subgroups (Native / English) for both WITHIN and ACROSS
y2 = y_base + step
for (top, sub), (x0, x1) in ranges_sub.items():
    add_bracket(ax, x0, x1, sub, y=y2, weight='bold')

# Level 3: top groups (WITHIN vs ACROSS)
y3 = y2 + step
for top, (x0, x1) in ranges_top.items():
    add_bracket(ax, x0, x1, top, y=y3, weight='bold')

# plt.savefig("plots/spatial_LH_WITHIN_ACROSS.svg", format="svg", bbox_inches="tight")
plt.show()
