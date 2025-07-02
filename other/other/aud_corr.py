import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import chdir
from scipy.stats import pearsonr
from math import sqrt
import pickle

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
plt.ylim(-.7, 0.8)
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
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 2), dpi = 300) 
axes_flat = axes.flatten()
for i, (index, row) in enumerate(corrs_across.iterrows()):
    avg_value = row.drop(index).mean()
    std = row.drop(index).std() / sqrt(len(row.drop(index)))
    diagonal_value = row[index]
    axes_flat[i].bar(['Avg', 'Within'], [avg_value, diagonal_value], yerr = [std, np.nan], capsize = 3)
    axes_flat[i].set_title(f"{index}", fontsize = 10)
    axes_flat[i].set_ylim(-.17, 0.53)
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

data_md = pd.read_csv("data/Alice_Story_TimeSeries_SpeechRegions.csv")

lang_code = {row["UID"] : row["Language"] for index, row in data.iterrows()} # lang from UID in lang time series
data_md["Language"] = data_md["UID"].map(lang_code)

rois_md = ['LH_TE11']

lang_dict_md = {}
for l in set(data_md["Language"]):
    temp = data_md[data_md.Language == l]
    if len(temp) == 24: # 20 rois X 2 partic
        sub1, sub2 = list(set(temp.UID))
        part1 = temp[(temp.UID == sub1) & (temp.ROI.isin(rois_md))]
        part2 = temp[(temp.UID == sub2) & (temp.ROI.isin(rois_md))]
        
        ts1 = list(part1.iloc[:, 6:].mean())[9:-3] # 12 sec of silence at the beginning and end (--> 130 TRs)
        ts2 = list(part2.iloc[:, 6:].mean())[9:-3]
        lang_dict_md[l] = [ts1, ts2]

corrs_md = []
for key, value in lang_dict_md.items():
    corr, p = pearsonr(value[0], value[1])
    print(key, corr)
    corrs_md.append([key, corr, p])
    
corrs_md = pd.DataFrame(corrs_md, columns = ["lang", "r", "p"])
corrs_md = corrs_md.sort_values(by="r")
#corrs.to_csv("results/correlations/correlation_participants.csv", index=False)

labels = corrs_md["lang"]
values = corrs_md["r"]
p_values = corrs_md["p"]

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
plt.ylim(-.7, 0.8)
plt.ylabel("r")
plt.xlabel("TE11 and TE12 (* p < .05; ** p < .005)")
plt.title("Time series correlation - auditory regions")
plt.tight_layout()
plt.show()


merged = pd.merge(corrs, corrs_md, on = "lang")
pearsonr(merged["r_x"], merged["r_y"])

len(corrs_md[corrs_md["r"] > 0]) / len(corrs_md)
    
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
