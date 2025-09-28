import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import chdir
from scipy.stats import pearsonr
from math import atanh, tanh, sqrt
import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

dirName = "/home/dev/Documents/PhD/Alice"
chdir(dirName)

data = pd.read_csv("data/Alice_Story_TimeSeries.csv")

rois = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp'] # 'Lang_LH_AngG'

lang_dict = {}
for l in set(data["Language"]):
    temp = data[data.Language == l]
    if len(temp) == 24:
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois))]
        
        ts1 = list(part1.iloc[:, 7:].mean())[9:-3] # 12 sec of silence at the beginning and end (--> 130 TRs)
        ts2 = list(part2.iloc[:, 7:].mean())[9:-3] # [6:-6]; then, shift by 3 TRs (lag in HRF)
        lang_dict[l] = [ts1, ts2]

corrs = []
for key, value in lang_dict.items():
    corr, p = pearsonr(value[0], value[1])
    print(key, corr)
    corrs.append([key, corr, p])
    
corrs = pd.DataFrame(corrs, columns = ["lang", "r", "p"])
corrs = corrs.sort_values(by="r")
#corrs.to_csv("results/correlations/correlation_participants.csv", index=False)

labels = corrs["lang"]
values = corrs["r"]
p_values = corrs["p"]

colors = ['blue' if val < 0 else 'red' for val in values]

fig, ax = plt.subplots(figsize=(13*.75, 6*.75), dpi = 300)
bars = plt.bar(labels, values, color=colors)
for bar, p_value in zip(bars, p_values):
    if bar.get_height() > 0:
        if p_value < 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "**", ha='center', va='bottom')
        elif p_value < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*", ha='center', va='bottom')
    elif bar.get_height() < 0:
        if p_value < 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - .06, "**", ha='center', va='bottom')
        elif p_value < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - .06, "*", ha='center', va='bottom')
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(base=0.05))
plt.xticks(rotation=45, ha='right')
plt.ylim(None, 0.55)
plt.ylabel("r")
plt.xlabel("Language (* p < .05; ** p < .005)")
plt.title("Time series correlation")
plt.tight_layout()
plt.show()

# create outcome variable to use in encoding
fmri_d = {}
for key, value in lang_dict.items():
    try:
        a = np.array(value[0])
        b = np.array(value[1])
        fmri_d[key] = np.mean([a, b], axis=0)
    except IndexError:
        pass
    
with open("data/dict_fMRI", 'wb') as handle:
    pickle.dump(fmri_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(corrs[(corrs["r"] > 0) & (corrs["p"] < 0.05)])
print(corrs[(corrs["r"] > 0) & (corrs["p"] < 0.05)]["r"].mean())

##########################################################################################
# asked by reviewers: fROI-based analyses

froi_d = {}
for l in set(data["Language"]):
    temp = data[data.Language == l]
    if len(temp) == 24:
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois))]

        froi_d.setdefault(l, {})

        # participant 1
        p1_data = {}
        for roi in rois:
            roi_data = part1[part1.ROI == roi].iloc[:, 7:]
            p1_data[roi] = roi_data.values[0][9:-3]  # row per ROI
        all_avg = np.mean(np.stack(list(p1_data.values())), axis=0)
        p1_data["all"] = all_avg
        froi_d[l][sub1] = p1_data

        # participant 2
        p2_data = {}
        for roi in rois:
            roi_data = part2[part2.ROI == roi].iloc[:, 7:]
            p2_data[roi] = roi_data.values[0][9:-3]
        all_avg = np.mean(np.stack(list(p2_data.values())), axis=0)
        p2_data["all"] = all_avg
        froi_d[l][sub2] = p2_data

with open("data/dict_fROI", 'wb') as handle:
    pickle.dump(froi_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# check correlations across fROIs within parts
roi_corrs_within = []
for lang in froi_d:
    for participant in froi_d[lang]:
        rois_only = [roi for roi in froi_d[lang][participant] if roi != "all"]
        for roi1, roi2 in combinations(rois_only, 2):
            ts1 = froi_d[lang][participant][roi1]
            ts2 = froi_d[lang][participant][roi2]
            r, p = pearsonr(ts1, ts2)
            roi_corrs_within.append((lang, participant, roi1, roi2, r, p))
df_within_participant = pd.DataFrame(roi_corrs_within, columns=["Language", "Participant", "ROI_1", "ROI_2", "r", "p"])

###############################################################################
###############################################################################
###############################################################################

rois = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp']
froi_markers = {
    'Lang_LH_AntTemp': 's',
    'Lang_LH_IFG': 'o',
    'Lang_LH_IFGorb': '^',
    'Lang_LH_MFG': 'v',
    'Lang_LH_PostTemp': 'X'
}

def corr_ci(r, n, alpha=0.05):
    # fisher z CI, back-transformed to r
    if n <= 3 or np.isclose(abs(r), 1.0):
        return (np.nan, np.nan)
    z = atanh(r)
    se = 1.0 / sqrt(n - 3)
    zcrit = 1.96 if np.isclose(alpha, 0.05) else None
    if zcrit is None:
        from scipy.stats import norm
        zcrit = norm.ppf(1 - alpha/2)
    lo = tanh(z - zcrit * se)
    hi = tanh(z + zcrit * se)
    return (lo, hi)

records = []
for lang, parts in froi_d.items():
    if len(parts) != 2:
        continue
    uid1, uid2 = list(parts.keys())
    rois_plus_all = rois + ['all']

    for roi in rois_plus_all:
        ts1 = np.array(parts[uid1][roi], dtype=float)
        ts2 = np.array(parts[uid2][roi], dtype=float)
        # Drop any NaNs pairwise
        mask = np.isfinite(ts1) & np.isfinite(ts2)
        ts1, ts2 = ts1[mask], ts2[mask]
        if ts1.size < 4:
            r, p = np.nan, np.nan
            lo, hi = np.nan, np.nan
            n = ts1.size
        else:
            r, p = pearsonr(ts1, ts2)
            n = ts1.size
            lo, hi = corr_ci(r, n, alpha=0.05)

        records.append({
            'lang': lang,
            'roi': roi,
            'r': r,
            'p': p,
            'n': n,
            'r_lo': lo,
            'r_hi': hi
        })

df = pd.DataFrame.from_records(records)
order = (
    df[df['roi'] == 'all']
    .sort_values('r', na_position='first')
    ['lang']
    .tolist()
)

fig, ax = plt.subplots(figsize=(12.1*.75, 6*.75), dpi=300)
for roi in rois:
    sub = df[(df['roi'] == roi) & (df['lang'].isin(order))].copy()
    sub['x'] = sub['lang'].map({lang:i for i,lang in enumerate(order)})
    y = sub['r'].values
    yerr = np.vstack([
        y - sub['r_lo'].values,
        sub['r_hi'].values - y
    ])
    colors = ['red' if r > 0 else 'blue' for r in y]
    # ax.errorbar(
    #     sub['x'].values, y, yerr=yerr,
    #     fmt=froi_markers[roi],
    #     markersize=4,
    #     elinewidth=0.7,
    #     capsize=2,
    #     alpha=0.6,
    #     linestyle='none',
    #     color='gray'
    # )
    ax.scatter(
        sub['x'].values, y,
        marker=froi_markers[roi],
        c=colors,
        s=25,
        alpha=0.4,
        label=roi
    )
sub_all = df[(df['roi'] == 'all') & (df['lang'].isin(order))].copy()
sub_all['x'] = sub_all['lang'].map({lang:i for i,lang in enumerate(order)})
y_all = sub_all['r'].values
yerr_all = np.vstack([
    y_all - sub_all['r_lo'].values,
    sub_all['r_hi'].values - y_all
])
colors_all = ['red' if r > 0 else 'blue' for r in y_all]
ax.errorbar(
    sub_all['x'].values, y_all, yerr=yerr_all,
    fmt='o',
    markersize=8,
    elinewidth=1.2,
    capsize=3,
    linestyle='none',
    color='black'  # error bars for 'all'
)
ax.scatter(
    sub_all['x'].values, y_all,
    marker='o',
    c=colors_all,
    s=60,
    edgecolor='black',
    zorder=3,
    label='all'
)
for _, row in sub_all.iterrows():
    if pd.notna(row['r']) and pd.notna(row['p']):
        star = "**" if row['p'] < 0.005 else ("*" if row['p'] < 0.05 else "")
        if star:
            offset = 0.15 if row['r'] >= 0 else -0.23
            ax.text(
                row['x'], row['r'] + offset,
                star, ha='center', va='bottom', fontsize=11
            )

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha='right')
ax.set_ylabel("R")
# ax.set_xlabel("Language (* p < .05; ** p < .005)")
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
# ax.set_ylim(-0.15, 0.55)
# ax.legend(loc='upper left', ncol=2, frameon=False)
plt.savefig("plots/time-series-corr.svg", format="svg", bbox_inches="tight")
plt.tight_layout()
plt.show()

roi_summary = (
    df.groupby("roi")["r"]
      .agg(
          mean="mean",
          std="std",
          count="count",
          min="min",
          max="max",
          n_pos=lambda x: (x > 0).sum(),
          frac_pos=lambda x: (x > 0).mean()
      )
)
roi_summary['sem'] = roi_summary['std'] / roi_summary['count']**0.5
roi_summary = roi_summary[['mean', 'sem', 'frac_pos']] #  'min', 'max'
print(roi_summary.sort_values(by="mean", ascending = False))

#                       mean       sem       min       max
# roi                                                     
# Lang_LH_PostTemp  0.141140  0.023694 -0.175632  0.422637
# Lang_LH_AntTemp   0.121991  0.031412 -0.391922  0.475791
# all               0.080869  0.033318 -0.662467  0.461557
# Lang_LH_IFG       0.075657  0.023776 -0.263456  0.401316
# Lang_LH_MFG       0.056307  0.037940 -0.750040  0.455420
# Lang_LH_IFGorb    0.029402  0.029987 -0.606524  0.317150

##########################################################################################

langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']

corrs_across = []
for language in langs:
    temp = []
    t1, t2 = lang_dict[language]
    for language1 in langs:
        if language == language1:
            corr = pearsonr(t1, t2)[0]
            temp.append(corr)
        else:
            t3, t4 = lang_dict[language1]
            c1 = pearsonr(t1, t3)[0]
            c2 = pearsonr(t1, t4)[0]
            c3 = pearsonr(t2, t3)[0]
            c4 = pearsonr(t2, t4)[0]
            corr = np.mean([c1, c2, c3, c4])
            temp.append(corr)
    corrs_across.append(temp)
corrs_across = pd.DataFrame(corrs_across)
corrs_across.index = langs
corrs_across.columns = langs 

num_rows = 2
num_cols = 6
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 2), dpi=300) 
axes_flat = axes.flatten()
for i, (index, row) in enumerate(corrs_across.iterrows()):
    avg_value = row.drop(index).mean()
    std = row.drop(index).std() / np.sqrt(len(row.drop(index)))
    diagonal_value = row[index]
    axes_flat[i].errorbar(
        [0, 1],
        [avg_value, diagonal_value],
        yerr=[std, np.nan],
        fmt='o',
        mfc='skyblue',
        mec='black',
        mew=1,
        markersize=4,
        ecolor='black',
        elinewidth=1.2,
        capsize=3,
        zorder=10
    )
    axes_flat[i].axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    axes_flat[i].set_title(f"{index}", fontsize=10)
    axes_flat[i].set_ylim(-.17, 0.53)
    axes_flat[i].set_xticks([0, 1])
    axes_flat[i].set_xticklabels(['Avg', 'Within'], fontsize=8)
    axes_flat[i].set_xlim(-0.5, 1.5)
for j in range(i+1, num_rows * num_cols):
    fig.delaxes(axes_flat[j])
plt.tight_layout()
plt.show()


###############################################################################
###############################################################################

###############################################################################
###############################################################################

##########################
# CONTROL DATA (RH & MD) #
##########################

rois_rh = ['Lang_RH_AntTemp','Lang_RH_IFG','Lang_RH_IFGorb','Lang_RH_MFG','Lang_RH_PostTemp'] # 'Lang_RH_AngG'

lang_dict_rh = {}
for l in set(data["Language"]):
    temp = data[data.Language == l]
    if len(temp) == 24:
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois_rh))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois_rh))]
        
        ts1 = list(part1.iloc[:, 7:].mean())[9:-3]
        ts2 = list(part2.iloc[:, 7:].mean())[9:-3]
        lang_dict_rh[l] = [ts1, ts2]


# create outcome variable (RH) to use in encoding
fmri_d_rh = {}
for key, value in lang_dict_rh.items():
    if key in fmri_d.keys():
        a = np.array(value[0])
        b = np.array(value[1])
        fmri_d_rh[key] = np.mean([a, b], axis=0)
    
with open("data/dict_fMRI_rh", 'wb') as handle:
    pickle.dump(fmri_d_rh, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
################################################################################
# asked by reviewers: fROIs

froi_d_rh = {}
for l in set(data["Language"]):
    temp = data[data.Language == l]
    if len(temp) == 24:
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois_rh))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois_rh))]

        froi_d_rh.setdefault(l, {})

        # participant 1
        p1_data = {}
        for roi in rois_rh:
            roi_data = part1[part1.ROI == roi].iloc[:, 7:]
            p1_data[roi] = roi_data.values[0][9:-3]
        all_avg = np.mean(np.stack(list(p1_data.values())), axis=0)
        p1_data["all"] = all_avg
        froi_d_rh[l][sub1] = p1_data

        # participant 2
        p2_data = {}
        for roi in rois_rh:
            roi_data = part2[part2.ROI == roi].iloc[:, 7:]
            p2_data[roi] = roi_data.values[0][9:-3]
        all_avg = np.mean(np.stack(list(p2_data.values())), axis=0)
        p2_data["all"] = all_avg
        froi_d_rh[l][sub2] = p2_data

with open("data/dict_fROI_rh", 'wb') as handle:
    pickle.dump(froi_d_rh, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# plot correlations with right hem

corrs_rh = []
for l in langs:
    r, p = pearsonr(fmri_d[l], fmri_d_rh[l])
    corrs_rh.append([l, r, p])
corrs_rh = pd.DataFrame(corrs_rh, columns = ["lang", "r", "p"])
corrs_rh = corrs_rh.sort_values(by="r")
corrs_rh.mean()

labels = corrs_rh["lang"]
values = corrs_rh["r"]
p_values = corrs_rh["p"]

colors = ['blue' if val < 0 else 'red' for val in values]

fig, ax = plt.subplots(figsize=(7*.65, 6*.65), dpi = 300)
bars = plt.bar(labels, values, color=colors)
for bar, p_value in zip(bars, p_values):
    if bar.get_height() > 0:
        if p_value < 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "**", ha='center', va='bottom')
        elif p_value < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*", ha='center', va='bottom')
    elif bar.get_height() < 0:
        if p_value < 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - .06, "**", ha='center', va='bottom')
        elif p_value < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - .06, "*", ha='center', va='bottom')
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(base=0.05))
plt.xticks(rotation=45, ha='right')
plt.ylabel("R")
plt.xlabel("Language (* p < .05; ** p < .005)")
plt.title("Left vs. Right correlation")
plt.ylim(-.3, 1)
plt.tight_layout()
plt.show()

##############
# MD NETWORK #
##############

data_md = pd.read_csv("data/Alice_Story_TimeSeries_MD.csv")

lang_code = {row["UID"] : row["Language"] for index, row in data.iterrows()} # lang from UID in lang time series
data_md["Language"] = data_md["UID"].map(lang_code)

rois_md = ['MD_LH_Precentral_A_PrecG', 'MD_LH_Precentral_B_IFGop', 'MD_LH_antParietal', 'MD_LH_insula', 'MD_LH_medialFrontal', 'MD_LH_midFrontal', 'MD_LH_midFrontalOrb', 'MD_LH_midParietal', 'MD_LH_postParietal', 'MD_LH_supFrontal', 'MD_RH_Precentral_A_PrecG', 'MD_RH_Precentral_B_IFGop', 'MD_RH_antParietal', 'MD_RH_insula', 'MD_RH_medialFrontal', 'MD_RH_midFrontal', 'MD_RH_midFrontalOrb', 'MD_RH_midParietal', 'MD_RH_postParietal', 'MD_RH_supFrontal']

lang_dict_md = {}
for l in set(data_md["Language"]):
    temp = data_md[data_md.Language == l]
    if len(temp) == 40: # 20 rois X 2 partic
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois_md))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois_md))]
        
        ts1 = list(part1.iloc[:, 6:].mean())[9:-3] # 12 sec of silence at the beginning and end (--> 130 TRs)
        ts2 = list(part2.iloc[:, 6:].mean())[9:-3]
        lang_dict_md[l] = [ts1, ts2]

# create outcome variable (MD) to use in encoding
fmri_d_md = {}
for key, value in lang_dict_md.items():
    if key in fmri_d.keys():
        a = np.array(value[0])
        b = np.array(value[1])
        fmri_d_md[key] = np.mean([a, b], axis=0)
    
with open("data/dict_fMRI_md", 'wb') as handle:
    pickle.dump(fmri_d_md, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################
# asked by reviewers

froi_d_md = {}
for l in set(data_md["Language"]):
    temp = data_md[data_md.Language == l]
    if len(temp) == 40:  # 20 ROIs x 2 participants
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois_md))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois_md))]

        froi_d_md.setdefault(l, {})

        # participant 1
        p1_data = {}
        for roi in rois_md:
            roi_data = part1[part1.ROI == roi].iloc[:, 6:]
            p1_data[roi] = roi_data.values[0][9:-3]
        all_avg = np.mean(np.stack(list(p1_data.values())), axis=0)
        p1_data["all"] = all_avg
        froi_d_md[l][sub1] = p1_data

        # participant 2
        p2_data = {}
        for roi in rois_md:
            roi_data = part2[part2.ROI == roi].iloc[:, 6:]
            p2_data[roi] = roi_data.values[0][9:-3]
        all_avg = np.mean(np.stack(list(p2_data.values())), axis=0)
        p2_data["all"] = all_avg
        froi_d_md[l][sub2] = p2_data

with open("data/dict_fROI_md", 'wb') as handle:
    pickle.dump(froi_d_md, handle, protocol=pickle.HIGHEST_PROTOCOL)

################################
# correlation between MD and L #
################################

corrs_netw = []
for l in langs:
    r, p = pearsonr(fmri_d[l], fmri_d_md[l])
    corrs_netw.append([l, r, p])
corrs_netw = pd.DataFrame(corrs_netw, columns = ["lang", "r", "p"])
corrs_netw = corrs_netw.sort_values(by="r")
corrs_netw.mean()

labels = corrs_netw["lang"]
values = corrs_netw["r"]
p_values = corrs_netw["p"]

colors = ['blue' if val < 0 else 'red' for val in values]

fig, ax = plt.subplots(figsize=(7*.65, 6*.65), dpi = 300)
bars = plt.bar(labels, values, color=colors)
for bar, p_value in zip(bars, p_values):
    if bar.get_height() > 0:
        if p_value < 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "**", ha='center', va='bottom')
        elif p_value < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*", ha='center', va='bottom')
    elif bar.get_height() < 0:
        if p_value < 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - .06, "**", ha='center', va='bottom')
        elif p_value < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - .06, "*", ha='center', va='bottom')
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(base=0.05))
plt.xticks(rotation=45, ha='right')
plt.ylabel("R")
plt.xlabel("Language (* p < .05; ** p < .005)")
plt.title("MD - L correlation")
plt.ylim(-.25, 0.75)
plt.tight_layout()
plt.show()

####################
# additional plots ##################
# corr with L and corr between part #
#####################################

corrs_all = []
for l in langs:
    r_md_l = pearsonr(fmri_d_md[l], fmri_d[l])[0]
    r_rh_l = pearsonr(fmri_d_rh[l], fmri_d[l])[0]
    r_l_within = pearsonr(lang_dict[l][0], lang_dict[l][1])[0]
    r_md_within = pearsonr(lang_dict_md[l][0], lang_dict_md[l][1])[0]
    r_rh_within = pearsonr(lang_dict_rh[l][0], lang_dict_rh[l][1])[0]
    corrs_all.append([l, r_md_l, r_rh_l, r_l_within, r_md_within, r_rh_within])
    
corrs_all = pd.DataFrame(corrs_all, columns = ["lang", "MD_L", "RH_L", "rel_L", "rel_MD", "rel_RH"])

plt.figure(figsize=(4.2,4), dpi=300)
sns.regplot(y=corrs_all["MD_L"], x=corrs_all["rel_MD"], ci=95, scatter_kws={'s': 20})
plt.xlabel("Reliability MD signal")
plt.ylabel("Correlation MD - L")
for idx, row in corrs_all.iterrows():
    plt.annotate(row["lang"], (row["rel_MD"], row["MD_L"]), fontsize=10, alpha=1, color='black')
plt.show()

plt.figure(figsize=(4.2,4), dpi=300)
sns.regplot(y=corrs_all["RH_L"], x=corrs_all["rel_RH"], ci=95, scatter_kws={'s': 20})
plt.xlabel("Reliability RH signal")
plt.ylabel("Correlation RH - LH")
for idx, row in corrs_all.iterrows():
    plt.annotate(row["lang"], (row["rel_RH"], row["RH_L"]), fontsize=10, alpha=1, color='black')
plt.show()

# correlations across (left and right) and (left and MD)
md_all_corr, rh_all_corr = [], []
for language in ["French", "Romanian", "Tamil", "Farsi", "Norwegian", "Afrikaans", "Turkish", "Lithuanian", "Vietnamese", "Dutch", "Spanish", "Marathi"]:
    left = fmri_d[language]
    right = fmri_d_rh[language]
    md = fmri_d_md[language]
    r_hem, _ = pearsonr(left, right)
    r_md, _ = pearsonr(left, md)
    md_all_corr.append(r_md)
    rh_all_corr.append(r_hem)
print(np.mean(md_all_corr))  
print(np.mean(rh_all_corr))    

###############################################################################
# lastly, we want the MAIN time-series (language network), but using the ROIs #
# identified in the participants' first language                              #
# (also asked by reviewers)                                                   #
###############################################################################

data_native = pd.read_csv("data/Alice_Story_TimeSeries_native.csv") # langloc in native language
rois = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp']

# language data is missing from this--but it'll be the same as in original data (same participants)
part_dict_lang = {row["UID"] : row["Language"] for index, row in data.iterrows()}
data_native["Language"] = data_native["UID"].map(part_dict_lang)

froi_d_native = {}
for l in set(data_native["Language"]):
    temp = data_native[data_native.Language == l]
    if len(temp) == 24:
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois))]

        froi_d_native.setdefault(l, {})

        # participant 1
        p1_data = {}
        for roi in rois:
            roi_data = part1[part1.ROI == roi].iloc[:, 7:]
            p1_data[roi] = roi_data.values[0][9:-3]  # row per ROI
        all_avg = np.mean(np.stack(list(p1_data.values())), axis=0)
        p1_data["all"] = all_avg
        froi_d_native[l][sub1] = p1_data

        # participant 2
        p2_data = {}
        for roi in rois:
            roi_data = part2[part2.ROI == roi].iloc[:, 7:]
            p2_data[roi] = roi_data.values[0][9:-3]
        all_avg = np.mean(np.stack(list(p2_data.values())), axis=0)
        p2_data["all"] = all_avg
        froi_d_native[l][sub2] = p2_data

with open("data/dict_fROI_native", 'wb') as handle:
    pickle.dump(froi_d_native, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# check correlations (from author response letter)

languages_of_interest = ["Afrikaans", "Dutch", "Farsi", "French", "Lithuanian", "Marathi", "Norwegian", "Romanian", "Spanish", "Tamil", "Turkish", "Vietnamese"]

corr_list = {}
for lang in languages_of_interest:
    part1, part2 = froi_d[lang].keys()
    for froi in froi_d[lang][part1].keys():
        ts_std = froi_d[lang][part1][froi]
        ts_ntv = froi_d_native[lang][part1][froi]
        r, _ = pearsonr(ts_std, ts_ntv)
        try:
            corr_list[froi].append(r)
        except KeyError:
            corr_list[froi] = [r]

labels = list(corr_list.keys())
nice_labels = [label.replace('Lang_LH_', '').replace('_', ' ') if label != 'all' else 'All Lang fROIs' for label in labels]

means = [np.mean(corr_list[k]) for k in labels]
ses = [np.std(corr_list[k], ddof=1) / np.sqrt(len(corr_list[k])) for k in labels]

plt.figure(figsize=(5, 3), dpi = 300)
x = np.arange(len(labels))
plt.bar(x, means, yerr=ses, capsize=5, color='lightblue', edgecolor='black')
plt.xticks(x, nice_labels, rotation=45, ha='right')
plt.ylabel('R')
plt.title('Time series correlations (English vs. native localizer)')
plt.tight_layout()
plt.show()

# for lang, mean in zip(nice_labels, means):
#     print(lang, round(mean, 2))
