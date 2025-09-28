import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import chdir
from scipy.stats import pearsonr, beta
import pickle
from itertools import combinations
from math import atanh, tanh, sqrt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

dirName = "/home/dev/Documents/PhD/Alice/confirmatory"
chdir(dirName)

def calculate_p_value(r, N):
    # SciPy's original calculation, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    dist = beta(N/2 - 1, N/2 - 1, loc=-1, scale=2)
    p_value = 2 * dist.cdf(-abs(r))
    return p_value

#uid_info = pd.read_csv("data/run_uid_passage.csv")[["UID", "Language", "Run", "Passage"]]
#uid_lang = {row["UID"] : row["Language"] for index, row in uid_info.iterrows()}

data = pd.read_csv("data/confirmatory_time_series.csv")

rois = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp'] # 'Lang_LH_AngG'

data = data[data.ROI.isin(rois)]
# data = data[data.Language != "Polish"]

TR_columns = [f"T_{str(n)}" for n in range(1, 143)]

lang_dict = {"Passage_1" : {lang : [] for lang in data["Language"].unique()}, 
             "Passage_2" : {lang : [] for lang in data["Language"].unique()}, 
             "Passage_3" : {lang : [] for lang in data["Language"].unique()}}

for l in data["Language"].unique():
    temp = data[data.Language == l]
    for part in temp["UID"].unique():
        temp_part = temp[temp.UID == part]
        for passage_n, run in zip(["Passage_1", "Passage_2", "Passage_3"],temp_part["Passage"].unique()):
            temp_run = temp_part[temp_part.Passage == run].loc[:,TR_columns].iloc[:,9:-3].mean(axis=0).to_numpy()
            lang_dict[passage_n][l].append(temp_run) # 9:-3

corrs = []
all_corrs = []
d_corr_keep = {"Passage_1" : {lang : [] for lang in data["Language"].unique()}, 
               "Passage_2" : {lang : [] for lang in data["Language"].unique()}, 
               "Passage_3" : {lang : [] for lang in data["Language"].unique()}}
for passage, passagedata in lang_dict.items():
    print(passage)
    for lang, responses in passagedata.items():
        n_resp = len(responses)
        if n_resp == 2:
            r, p = pearsonr(responses[0], responses[1])
            print(f"{n_resp} {lang:10} {r:.4f}, {p:.4f}")
            corrs.append([passage, lang, r, p, n_resp])
            d_corr_keep[passage][lang] = [0, 1]
            all_corrs.append(r)
        else:
            r_0, p_0 = pearsonr(responses[0], responses[1])
            r_1, p_1 = pearsonr(responses[0], responses[2])
            r_2, p_2 = pearsonr(responses[1], responses[2])
            #print(r_0, r_1, r_2)
            r = np.max([r_0, r_1, r_2])
            # find which participants to keep based on max corr
            idx = np.argmax([r_0, r_1, r_2])
            p = [p_0, p_1, p_2][idx]
            # count significant, positive correlations
            n_sig_corrs = 0
            for the_r, the_p in zip([r_0, r_1, r_2], [p_0, p_1, p_2]):
                if the_r > 0 and the_p < 0.05:
                    n_sig_corrs += 1
            if n_sig_corrs >= 2:
                d_corr_keep[passage][lang] = [0, 1, 2]
            else:
                if idx == 0:
                    d_corr_keep[passage][lang] = [0, 1]
                elif idx == 1:
                    d_corr_keep[passage][lang] = [0, 2]
                elif idx == 2:
                    d_corr_keep[passage][lang] = [1, 2]
            #p = calculate_p_value(r, 130)
            print(f"{n_resp} {lang:10} {r:.4f}, {p:.4f}")
            corrs.append([passage, lang, r, p, n_resp])
            all_corrs.extend([r_0, r_1, r_2])
    print("\n")
corrs = pd.DataFrame(corrs, columns = ["passage", "lang", "r", "p", "n_resp"])
print(corrs.r.mean())
print(np.mean(all_corrs))

########
# plot #
########

plt.figure(figsize=(7, 4), dpi=300)
average_r = corrs.groupby('lang')['r'].mean().sort_values(ascending=True)  # for sorting
languages = average_r.index
passages = corrs['passage'].unique()
x = np.arange(len(languages))
width = 0.25
bar_padding = 0.02  # slight space between bars

def add_significance(ax, bars, significance):
    for bar, sign in zip(bars, significance):
        height = bar.get_height()
        y_pos = height + (0.02 if height > 0 else -0.07)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, sign, ha='center', va='bottom', color='black', fontsize=8)

for i, passage in enumerate(passages):
    offset = (i * width) + (i * bar_padding) - (width / 2)
    subset = corrs[corrs['passage'] == passage]
    subset = subset.set_index('lang').reindex(languages).reset_index()  # Reorder based on average 'r'
    r_values = subset['r'].values
    colors = ['red' if val > 0 else 'blue' for val in r_values]
    bars = plt.bar(x + offset, r_values, width, label=passage, color=colors)
    # significance labels
    p_values = subset['p'].values
    sig_labels = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in p_values]
    add_significance(plt.gca(), bars, sig_labels)

ax = plt.gca()
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(base=0.05))
plt.xticks(x + width/2, languages, rotation=45, ha='right')
plt.ylabel("r")
plt.xlabel("Language (* p < .05; ** p < .01; *** p < .001)")
plt.title("Time series correlation")
plt.ylim(-.1, .65)
plt.tight_layout()
plt.show()

######################################
# another plot to check for outliers #
######################################

for passage_name, passage_data in lang_dict.items():
    for lang_name, lang_data in passage_data.items():
        mean_ts = np.mean(lang_data, axis=0)    
        plt.figure(figsize = (15*.5, 5*.5), dpi = 300)
        for ts in lang_data:
            plt.plot(range(len(ts)), ts, linewidth = 1)
            plt.scatter(range(len(ts)), ts, s = 4)
        plt.plot(range(len(mean_ts)), mean_ts, linewidth = 2.5, color="black")
        plt.plot(range(len(mean_ts)), mean_ts, linewidth = 1.5, color="gray")
        pname = " ".join(passage_name.split("_"))
        plt.title(f"{lang_name} - {pname}")
        #plt.savefig(f"data/plot_ts/{lang_name}_{passage_name}.png")
        plt.show()
        
for passage_name, passage_data in lang_dict.items():
    for lang_name, lang_data in passage_data.items():
        mean_ts = np.mean(lang_data, axis=0)    
        plt.figure(figsize = (15*.5, 5*.5), dpi = 300)
        for ts in lang_data:
            plt.plot(range(len(ts)), ts)
        plt.title(f"{lang_name} - {passage_name}")
        plt.show()

##########################################
# plot corrs acros fROIs per participant #
##########################################

corr_rois = []
for l in data["Language"].unique():
    temp = data[data.Language == l]
    for part in temp["UID"].unique():
        temp_part = temp[temp.UID == part]
        for passage_n, run in zip(["Passage_1", "Passage_2", "Passage_3"],temp_part["Passage"].unique()):
            temp_run = temp_part[temp_part.Passage == run].loc[:,TR_columns].iloc[:,9:-3].to_numpy()
            corr_part = []
            for c1, c2 in combinations(temp_run, 2):
                r, _ = pearsonr(c1, c2)
                corr_part.append(r)
            corr_rois.append([l, part, passage_n, np.mean(corr_part), np.std(corr_part)])
corr_rois = pd.DataFrame(corr_rois, columns = ["language", "part", "passage", "r", "sd"])

corr_rois['Language_Part'] = corr_rois['language'] + ' ' + corr_rois['part'].astype(str)

language_order = ['Hindi', 'Arabic', 'Portuguese', 'Italian', 'Russian', 'Mandarin', 'Korean', 'German', 'Polish']

corr_rois = corr_rois[corr_rois['language'].isin(language_order)]
corr_rois['language'] = pd.Categorical(corr_rois['language'], categories=language_order, ordered=True)
corr_rois = corr_rois.sort_values(['language', 'part', 'passage'])

unique_languages = corr_rois['language'].unique()
expanded_data = []
for language in unique_languages:
    language_data = corr_rois[corr_rois['language'] == language]
    expanded_data.append(language_data)
    # Add NaN row for spacing
    if language != unique_languages[-1]:  # Avoid adding a gap after the last language
        expanded_data.append(pd.DataFrame({'Language_Part': [f'Gap {language}'], 'r': [np.nan], 'passage': ['Gap']}))

corr_rois_expanded = pd.concat(expanded_data, ignore_index=True)

plt.figure(figsize=(7, 3), dpi=300)
bar_plot = sns.barplot(x='Language_Part', y='r', hue='passage', data=corr_rois_expanded, palette="Reds", dodge=True)

language_midpoints = []
last_idx = 0
for language in unique_languages:
    parts = corr_rois['Language_Part'][corr_rois['language'] == language].unique()
    mid_point = last_idx + len(parts) / 2 - 0.5
    language_midpoints.append((mid_point, language))
    last_idx += len(parts) + 1  # Adjusting for gaps

plt.xticks([x[0] for x in language_midpoints], [x[1] for x in language_midpoints], rotation=45, ha='right')
plt.ylabel('r')
plt.xlabel('')
plt.title('Correlations among language fROIs')
plt.legend().set_visible(False)
ax = plt.gca()
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
plt.ylim(None, 1)
plt.xlim(-.75, None)
plt.tight_layout()
plt.show()

#################################
# plot data with 3 participants #
#################################

def plain_avg_ci(rs, alpha=0.05):
    rs = np.array(rs, dtype=float)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return np.nan, np.nan, np.nan
    mean = rs.mean()
    se = np.std(rs) / np.sqrt(len(rs))
    return mean, mean-se, mean+se

rows = []
for passage, passagedata in lang_dict.items():
    for lang, responses in passagedata.items():
        if len(responses) == 3:
            r_0, p_0 = pearsonr(responses[0], responses[1])
            r_1, p_1 = pearsonr(responses[0], responses[2])
            r_2, p_2 = pearsonr(responses[1], responses[2])
            (r0_lo, r0_hi) = corr_ci(r_0, 130)
            (r1_lo, r1_hi) = corr_ci(r_1, 130)
            (r2_lo, r2_hi) = corr_ci(r_2, 130)
            rows.append([passage, lang,
                         r_0, p_0, r0_lo, r0_hi,
                         r_1, p_1, r1_lo, r1_hi,
                         r_2, p_2, r2_lo, r2_hi])

corrs_3 = pd.DataFrame(rows, columns=[
    "passage","lang",
    "r1","p1","r1_lo","r1_hi",
    "r2","p2","r2_lo","r2_hi",
    "r3","p3","r3_lo","r3_hi"
])

average_r = corrs_3.groupby('lang')[['r1','r2','r3']].mean().mean(axis=1).sort_values()
languages = average_r.index.tolist()
passages = sorted(corrs_3['passage'].unique())
xcenters  = np.arange(len(languages))

triplet_offsets = [-0.20, 0.00, 0.20]
jitter_small    = [-0.05, 0.00, 0.05]
big_marker_size = 99
small_marker_sz = 28

plt.figure(figsize=(13*.65, 6*.65), dpi=300)
lang_to_idx = {lang:i for i,lang in enumerate(languages)}
for p_i, passage in enumerate(passages):
    sub = (
        corrs_3[corrs_3['passage'] == passage]
        .set_index('lang')
        .reindex(languages)
        .reset_index()
    )
    pair_cols = [('r1','p1','r1_lo','r1_hi'),
                 ('r2','p2','r2_lo','r2_hi'),
                 ('r3','p3','r3_lo','r3_hi')]
    for li, lang in enumerate(languages):
        x0 = xcenters[li] + triplet_offsets[p_i]
        r_triplet = []
        for (rcol, pcol, lcol, hcol), joff in zip(pair_cols, jitter_small):
            r = sub.loc[li, rcol]
            p = sub.loc[li, pcol]
            if not np.isfinite(r):
                continue
            x_small = x0 + joff
            if np.isfinite(p) and p < 0.05:
                color = 'red' if r > 0 else 'blue'
                alpha = 0.55
            else:
                color = 'gray'
                alpha = 0.15
            plt.scatter([x_small], [r], s=small_marker_sz, c=[color], alpha=alpha, zorder=3)
            r_triplet.append(r)
        if len(r_triplet) >= 1:
            rbar, rbar_lo, rbar_hi = plain_avg_ci(r_triplet, alpha=0.05)
            yerr_big = np.array([[rbar - rbar_lo], [rbar_hi - rbar]])
            c_big = 'red' if rbar > 0 else 'blue'
            plt.errorbar(x0, rbar, yerr=yerr_big, fmt='D', markersize=6,
                         color='black', ecolor='black', elinewidth=1.2, capsize=4, alpha=0.95, zorder=4)
            plt.scatter([x0], [rbar], s=big_marker_size, c=[c_big], edgecolors='black', linewidths=1.1, zorder=5)
for i in range(len(languages)-1):
    plt.axvline(x=(xcenters[i] + xcenters[i+1]) / 2, color='0.85', linewidth=0.8, zorder=0)
ax = plt.gca()
import matplotlib.ticker as mticker
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.grid(True, which='major', linestyle='dashed', linewidth=0.5)
ax.set_xlim(-0.6, len(languages)-0.4)
plt.xticks(xcenters, languages, rotation=45, ha='right')
plt.ylabel("R")
plt.ylim(-0.43, 0.65)
plt.tight_layout()
plt.savefig("../plots/time-series-corr-confirmatory.svg", format="svg", bbox_inches="tight")
plt.show()

#########################################################
#########################################################
#########################################################

# create outcome variable to use in encoding
fmri_d = {"Passage_1" : {lang : [] for lang in data["Language"].unique()}, 
          "Passage_2" : {lang : [] for lang in data["Language"].unique()}, 
          "Passage_3" : {lang : [] for lang in data["Language"].unique()}}

for passage, passagedata in lang_dict.items():
    for lang, responses in passagedata.items():
        idx_keep = d_corr_keep[passage][lang]
        resp = np.array([responses[idx] for idx in idx_keep]).mean(axis=0)
        fmri_d[passage][lang] = resp
    
with open("data/dict_fMRI", 'wb') as handle:
    pickle.dump(fmri_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#########################################################
#########################################################
#########################################################
# asked by reviewer -- save by fROI


froi_d_expt2 = {
    "Passage_1": {lang: {} for lang in data["Language"].unique()},
    "Passage_2": {lang: {} for lang in data["Language"].unique()},
    "Passage_3": {lang: {} for lang in data["Language"].unique()},
}

for l in data["Language"].unique():
    temp = data[data.Language == l]
    for uid in temp["UID"].unique():
        temp_part = temp[temp.UID == uid]
        for passage_n, run in zip(["Passage_1", "Passage_2", "Passage_3"], temp_part["Passage"].unique()):
            roi_series = {}
            per_roi = []
            for roi in rois:
                row = temp_part[(temp_part.Passage == run) & (temp_part.ROI == roi)]
                if row.empty:
                    continue
                ts = row.loc[:, TR_columns].iloc[:, 9:-3].values.squeeze()
                roi_series[roi] = ts
                per_roi.append(ts)
            if roi_series:
                roi_series["all"] = np.mean(np.stack(per_roi, axis=0), axis=0)
                froi_d_expt2[passage_n][l][uid] = roi_series

with open("data/dict_fROI", "wb") as handle:
    pickle.dump(froi_d_expt2, handle, protocol=pickle.HIGHEST_PROTOCOL)


#########################################################
# reliability (only on data that I'll use for encoding) #
#########################################################

remove_passages = [["German", "Passage_1"],
                   ["Hindi", "Passage_1"],
                   ["Korean", "Passage_2"],
                   ["Korean", "Passage_3"],
                   ["Portuguese", "Passage_2"]]

all_rs = []
for passage, passagedata in lang_dict.items():
    for lang, responses in passagedata.items():
        if [lang, passage] not in remove_passages:
            idx_keep = d_corr_keep[passage][lang]
            resp = [responses[idx] for idx in idx_keep]
            mean_r = []
            for c1, c2 in combinations(resp, 2):
                r, _ = pearsonr(c1, c2)
                mean_r.append(r)
            all_rs.append(np.mean(mean_r))
print(np.mean(all_rs))

######################################################################
# figure for SI (corr across participants vs. corr across languages) #
######################################################################

def correlate_nonmatching(ts1, ts2):
    n_dims = ts1.shape[0]
    correlations = []
    for i in range(n_dims):
        other_dim = (i + 1) % n_dims  # non-matching dimension
        corr = pearsonr(ts1[i, :], ts2[other_dim, :])[0]  # corr ts1[i] with ts2[other_dim]
        correlations.append(corr)
    print(correlations)
    return np.mean(correlations)

d_passages_keep = {'Arabic' : [1, 2, 3], 'German' : [2, 3], 'Hindi' : [2, 3], 'Italian' : [1, 2, 3], 'Korean' : [1], 'Portuguese' : [1, 3], 'Russian' : [1, 2, 3], 'Mandarin' : [1, 2, 3], 'Polish' : [1, 2, 3]}

d_all = {"Passage_1" : {lang : [] for lang in data["Language"].unique()}, 
          "Passage_2" : {lang : [] for lang in data["Language"].unique()}, 
          "Passage_3" : {lang : [] for lang in data["Language"].unique()}}

for passage, passagedata in lang_dict.items():
    for lang, responses in passagedata.items():
        idx_keep = d_corr_keep[passage][lang]
        resp = np.array([responses[idx] for idx in idx_keep])
        d_all[passage][lang] = resp
        

all_corrs_across = []
for passage in passages:
    passage_n = int(passage[-1])
    corrs_across = []
    for language in data["Language"].unique():
        temp = []
        ts = d_all[passage][language]
        for language1 in data["Language"].unique():
            ts1 = d_all[passage][language1]
            if language == language1:
                if passage_n in d_passages_keep[language]:
                    corr = correlate_nonmatching(ts1, ts)
                    temp.append(corr)
                else:
                    temp.append(np.nan)
            else:
                if (passage_n in d_passages_keep[language]) and (passage_n in d_passages_keep[language1]):
                    across_corrs = []
                    for the_ts in ts:
                        for the_ts1 in ts1:
                            corr = pearsonr(the_ts, the_ts1)[0]
                            across_corrs.append(corr)
                    corr = np.mean([across_corrs])
                    temp.append(corr)
                else:
                    temp.append(np.nan)
        corrs_across.append(temp)
    corrs_across = pd.DataFrame(corrs_across)
    corrs_across.index = data["Language"].unique()
    corrs_across.columns = data["Language"].unique()
    all_corrs_across.append(corrs_across)
    
averaged_df = pd.DataFrame(np.nanmean(all_corrs_across, axis=0), columns = all_corrs_across[0].columns, index = all_corrs_across[0].index)

num_rows = 2
num_cols = 6
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 2), dpi=300) 
axes_flat = axes.flatten()
for i, (index, row) in enumerate(averaged_df.iterrows()):
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
