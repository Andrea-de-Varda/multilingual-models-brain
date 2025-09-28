import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from os import chdir
from tqdm import tqdm
import pickle

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/NaturalStories")

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

ns = pd.read_csv("NaturalStories_TimeSeries_20220615.csv")

#####################
# some basic checks #
#####################

for s in ns["Story"].unique():
    temp = ns[ns.Story == s]
    n = len(set(temp.UID))
    print(f"{s:<30} {n}")

d_story_groups = {"kingofbirds" : ["StoriesAud_4_kingofbirds_N", "Stories_ToM_4_kingofbirds_N", "Stories_4_kingofbirds_N"],
                  "elvis" : ["StoriesAud_5_elvis_T", "Stories_ToM_5_elvis_T", "Stories_5_elvis_T", "ISCmega_elvis"], # ISCmega_elvis removed (different n° of TRs)
                  "aqua" : ["Stories_ToM_2_aqua_T", "Stories_2_aqua_T", "StoriesAud_2_aqua_T"],
                  "tulips" : ["Stories_9_tulips_T"],
                  "boar" : ["Stories_ToM_1_boar_T", "Stories_1_boar_T", "StoriesAud_1_boar_T"],
                  "tree" : ["ISCmega_tree"], 
                  "highschool" : ["Stories_ToM_7_highschool_N", "Stories_7_highschool_N"],
                  "matchstickseller" : ["Stories_3_matchstickseller_N"],
                  "mrsticky" : ["Stories_6_mrsticky_N"]}

ns_tr_columns = ["T_" + str(n) for n in range(1, 227)] # T_1, T_2, ... T_226

# check that same story groups have the same n° of TRs
for story_group, story_list in d_story_groups.items():
    print("\n", story_group)
    temp = ns[ns.Story.isin(story_list)]
    for single_story in story_list:
        temp_single = temp[temp.Story == single_story].filter(ns_tr_columns).copy().dropna(axis=1, how="all")
        print(f"{single_story:<30} {temp_single.shape}")
# NOTE: ISCmega_elvis had additional silence at the end of the story. Correcting that later (trimming). 

###############################
# extracting reliability info #
###############################

def split_half(data, total=1000):
    r = []
    num_cols = data.shape[1]
    all_columns = np.arange(num_cols)
    for n in tqdm(range(total), total=total):
        np.random.seed(n)  # seed for reproducibility
        # split columns into two halves
        cols_random = np.random.choice(all_columns, num_cols // 2, replace=False)
        cols_not_random = np.setdiff1d(all_columns, cols_random, assume_unique=True)
        # mean for each half
        part1 = np.mean(data[:, cols_random], axis=1)
        part2 = np.mean(data[:, cols_not_random], axis=1)
        r_temp, _ = pearsonr(part1, part2)
        r.append(r_temp)
    # Compute stats for uncorrected R
    mean_r = np.mean(r)
    sd_r = np.std(r)
    print(f"\nR (uncorrected) = {mean_r}, SD = {sd_r}")
    return mean_r, sd_r

# def split_half(timeseries, n_splits=1000):
#     n_rows, n_cols = timeseries.shape
#     correlations = np.zeros(n_splits)

#     for i in range(n_splits):
#         np.random.seed(i)
#         indices = np.random.permutation(n_rows)
#         half = n_rows // 2
#         group1_indices = indices[:half]
#         group2_indices = indices[half:]

#         # get mean for two groups
#         mean1 = np.mean(timeseries[group1_indices, :], axis=0)
#         mean2 = np.mean(timeseries[group2_indices, :], axis=0)

#         corr, _ = pearsonr(mean1, mean2)
#         correlations[i] = corr

#     return np.mean(correlations), np.std(correlations)

lang_rois = ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp']

d_timeseries = {group : [] for group in d_story_groups.keys()} # all ts of individual subjects
for story_group, story_list in d_story_groups.items():
    temp = ns[(ns.Story.isin(story_list)) & (ns.ROI.isin(lang_rois))].copy().dropna(axis=1, how="all")
    story_parts = temp.UID.unique()
    # print(len(story_parts))
    for part in story_parts:
        if story_group == "elvis": # ISCmega_elvis had different N° of TRs
            elvis_columns = ["T_" + str(n) for n in range(1, 152)]
            thepart = temp[temp.UID == part].filter(elvis_columns).mean(axis=0).tolist()
            d_timeseries[story_group].append(thepart)
        else:
            thepart = temp[temp.UID == part].filter(ns_tr_columns).mean(axis=0).tolist()
            d_timeseries[story_group].append(thepart)

d_timeseries = {k : np.array(v) for k, v in d_timeseries.items()}

df_reliab = []
for k, v in tqdm(d_timeseries.items(), total = len(d_timeseries.keys())):
    thesplit = split_half(v.T)
    n = v.shape[0]
    n1 = v.shape[1]
    df_reliab.append([k, thesplit[0], thesplit[1], n, n1])
df_reliab = pd.DataFrame(df_reliab, columns = ["story", "r", "sd", "n", "n1"]).sort_values(by="r")
# df_reliab["story_n"] = df_reliab['story'] + " (N = " + df_reliab['n'].astype(str) + ")"

d_story_nice = {"tulips" : "Tulips", 
                "mrsticky" : "Mr. Sticky", 
                "tree" : "Tree", 
                "highschool" : "High school",
                "matchstickseller" : "Matchstick seller", 
                "aqua" : "Aqua", 
                "boar" : "Boar", 
                "kingofbirds" : "King of birds", 
                "elvis" : "Elvis"}
df_reliab["story_nice"] = df_reliab["story"].map(d_story_nice)

fig, ax = plt.subplots(figsize=(5*.85, 3*.85), dpi=300)
ax.errorbar(
    x=df_reliab['story_nice'],
    y=df_reliab['r'],
    yerr=df_reliab['sd'],
    fmt='o',
    mfc='skyblue',
    mec='black',
    mew=1,
    markersize=7,
    ecolor='black',
    elinewidth=2,
    capsize=5,
    zorder=12)
for x, y, n in zip(df_reliab["story_nice"], df_reliab["r"], df_reliab["n"]):
    ax.annotate(
        str(n),
        (x, 0.01),
        ha='center',
        zorder=14)
ax.set_ylabel('R', fontsize=12)
plt.ylim(0, None)
plt.xticks(rotation=35, ha='right')
ax.yaxis.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.tight_layout()
plt.show()


print(f'Overall reliability = {(df_reliab["r"] * df_reliab["n1"]).sum() / df_reliab["n1"].sum()}')

d_natstor = {k : v.mean(axis=0) for k, v in d_timeseries.items()}

#############################################
# check that transcripts and TRs correspond #
#############################################
    
# note: this part assumes whisper_NatStories.py has already been run
story_n_dict = {"1" : "boar",
                "2" : "aqua",
                "3" : "matchstickseller",
                "4" : "kingofbirds", 
                "5" : "elvis", 
                "6" : "mrsticky",
                "7" : "highschool",
                "10" : "tree",
                "9" : "tulips"}

###################################
# 16s of silence before and after ######################
# trying with different Deltas (shift = 0, 1, 2, 3, 4) #
########################################################

d_shift_0, d_shift_1, d_shift_2, d_shift_3, d_shift_4 = {}, {}, {}, {}, {}
for n, name in story_n_dict.items():
    df = pd.read_csv(f"transcribed/{n}.csv")
    d_shift_0[name] = d_natstor[name][ 8 : -8]
    d_shift_1[name] = d_natstor[name][ 9 : -7]
    d_shift_2[name] = d_natstor[name][10 : -6]
    d_shift_3[name] = d_natstor[name][11 : -5]
    d_shift_4[name] = d_natstor[name][12 : -4]

dicts = [d_shift_0, d_shift_1, d_shift_2, d_shift_3, d_shift_4]
names = ["d_shift_0", "d_shift_1", "d_shift_2", "d_shift_3", "d_shift_4"]

for d, n in zip(dicts, names):
    save(d, f"response/{n}")
