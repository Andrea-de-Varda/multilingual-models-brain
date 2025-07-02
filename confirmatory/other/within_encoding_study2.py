import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, norm, ttest_rel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import sqrt
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice.")

chdir("/home/dev/Documents/PhD/Alice/confirmatory")

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, passage, embeddings):
    df = pd.read_csv(f"transcribed/{passage}/{lang}.csv")
    df = df[df["end"] <= 260]
    df = df[df["text"] != " "] # !IMPORTANT! change from Exp1 analysis -- there was a whitespace in mandarin messing up the alignment between embeddings and timeseries, so removing it (and using iloc in time_words.iloc[i] bc now the indexes are not aligned - alternatively could do reset_index())
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words.iloc[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

def test_model_Ridge(X, y, n=10, random = False, shuffle=False):
    if shuffle:
        kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_pred = []; y_tot = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()
        if random:
            np.random.seed(0)  # seed for reproducibility
            np.random.shuffle(y_train)  # shuffling y
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_pred.extend(y_pred.tolist())
        y_tot.extend(y_test.tolist())
    r_tot = pearsonr(out_pred, y_tot)[0]
    print(round(r_tot, 4))
    return r_tot

passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_code_dict = {k : v for k, v in zip(lang_codes, languages)}
lang_code_dict_inv = {v : k for k, v in lang_code_dict.items()}

with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)
    
dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 14, 
                  "nllb200_1B" : 15, 
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 10, 
                  "xlmr_large" : 15, 
                  "distilmbert" : 4, 
                  "bert_base" : 5, 
                  "mdeberta" : 9, 
                  "mt5_small" : 5, 
                  "mt5_base" : 11,
                  "mt5_large" : 14, 
                  "mgpt" : 14, 
                  "xglm_small" : 15, 
                  "xglm_med" : 10, 
                  "xglm_large" : 11, 
                  "xglm_xl" : 45}

model_names = dict_bestlayer.keys()
        
#################################
# TEST encoding models "within" #
#################################

# Passages to remove (no sig. correlation in fMRI time-series):
# Korean [2, 3]
# German [1]
# Portuguese [2]
# Hindi [1]

d_passages_keep = {'ar' : [1, 2, 3], 'de' : [2, 3], 'hi' : [2, 3], 'it' : [1, 2, 3], 'ko' : [1], 'pt' : [1, 3], 'ru' : [1, 2, 3], 'zh' : [1, 2, 3], 'pl' : [1, 2, 3]}

###########################################
# crossval within passage (as in Study I) #
###########################################

within_results_cv = []
for lang in lang_codes:
    passages_keep = [f"Passage_{str(n)}" for n in d_passages_keep[lang]]
    print("\n", lang_code_dict[lang])
    for modelname, best_layer in dict_bestlayer.items():
        rs = []
        for test_passage in passages_keep:
            X = preproc_align(lang, test_passage, load(f"{test_passage}/{modelname}_{lang}")[best_layer])
            y = d[test_passage][lang_code_dict[lang]].reshape(-1, 1).flatten()
            r = test_model_Ridge(X, y)
            rs.append(r)
        print(f"Processed with {modelname} -- {rs}")
        within_results_cv.append(["experimental", modelname, lang_code_dict[lang], rs, np.mean(rs), np.std(rs) / np.sqrt(len(rs))])

within_results_cv = pd.DataFrame(within_results_cv, columns = ["condition", "model", "language", "r", "r_mean", "r_se"])
within_results_cv.groupby("model").agg({"r_mean" : "mean"})

within_results_cv_random = []
for lang in lang_codes:
    passages_keep = [f"Passage_{str(n)}" for n in d_passages_keep[lang]]
    print("\n", lang_code_dict[lang])
    for modelname, best_layer in dict_bestlayer.items():
        rs = []
        for test_passage in passages_keep:
            X = preproc_align(lang, test_passage, load(f"{test_passage}/{modelname}_{lang}")[best_layer])
            y = d[test_passage][lang_code_dict[lang]].reshape(-1, 1).flatten()
            r = test_model_Ridge(X, y, random = True)
            rs.append(r)
        print(f"Processed with {modelname} -- {rs}")
        within_results_cv_random.append(["experimental", modelname, lang_code_dict[lang], rs, np.mean(rs), np.std(rs) / np.sqrt(len(rs))])

within_results_cv_random = pd.DataFrame(within_results_cv_random, columns = ["condition", "model", "language", "r", "r_mean", "r_se"])
within_all = pd.merge(within_results_cv, within_results_cv_random, suffixes = ("", "_random"), on = ["condition", "model", "language"])

# within_all.to_csv("other/confirmatory_within_all.csv", index=False)
within_all = pd.read_csv("other/confirmatory_within_all.csv")

###############################
# checking stats significance #
###############################

def r_to_z(r1, r2, n = 130):    
    # fisher r-to-z transformation
    z_1 = np.arctanh(r1)
    z_2 = np.arctanh(r2)
    z_diff = z_1 - z_2
    # standard error of difference
    se_diff = np.sqrt(2*(1/(n-3)))
    z_stat = z_diff / se_diff
    p = 2 * (1 - norm.cdf(np.abs(z_stat))) # two tailed
    return z_stat, p

def combine_z_statistics(z_stats, return_z = False):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    if return_z:
        out = z_combined, combined_pvalue
    else:
        out = combined_pvalue
    return out

compare_z, compare_p = [], []
for scores, scores_random in zip(within_all["r"], within_all["r_random"]):
    zs = []
    for the_score, the_score_random in zip(scores, scores_random):
        z, p = r_to_z(the_score, the_score_random)
        zs.append(z)
    z_combined, p_combined = combine_z_statistics(zs, return_z = True)
    compare_z.append(z_combined)
    compare_p.append(p_combined)

within_all["z"] = compare_z
within_all["p"] = compare_p

# within_all.to_csv("other/confirmatory_within_all.csv", index=False)
within_all = pd.read_csv("other/confirmatory_within_all.csv")

within_all_grouped = within_all.groupby("model").agg(Score=("r_mean", "mean"), sd=("r_mean", "std"), p=("z", combine_z_statistics))
within_all_grouped = within_all_grouped.reset_index()

print(within_all_grouped["p"].max()) # 1.5386293701472894e-07

############
# PLOTTING #
############

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

df = within_all_grouped
df["model"] = pd.Categorical(df["model"], categories=model_names, ordered=True)
df = df.sort_values("model")
df = df.reset_index(drop=True)
df["Model"] = df["model"].map(names_nice_dict)
df["Family"] = df["model"].map(class_dict)
df["n"] = 9
df["se"] = df["sd"] / np.sqrt(df["n"])
title = ""
ylimstart = 0
ylim = 0.75

study1 = pd.read_csv("../results/mono_multi.csv")
studies_merged = pd.merge(study1[["Model", "Score_mono", "se_mono"]], df, on = "Model")
df = studies_merged

def add_bracket(ax, pos1, pos2, text, y_offset=0.05, weight = "normal"):
    """Adds a bracket and text annotation on the plot"""
    mid = (pos1 + pos2) / 2
    y = max(df['Score']) + y_offset  # Adjust based on your data
    ax.plot([pos1, pos1, pos2, pos2], [y, y + 0.02, y + 0.02, y], color='black', lw=2)
    ax.text(mid, y + 0.03, text, ha='center', va='bottom', fontsize=19, weight=weight)

plt.figure(figsize=(24*.7, 11.5*.7), dpi = 300)
sns.set_context("talk")
palette = sns.color_palette("tab20", n_colors = 9)
palette_d = {'BERT' : "steelblue",
             'DeBERTa' : "teal",
             'InfoXLM' : "firebrick",
             'NLLB' : "tomato",
             'XGLM' : "forestgreen",
             'XLM-Align' : "firebrick",
             'XLM-R' : "lightsteelblue",
             'mGPT' : "yellowgreen",
             'mT5' : "darkorange"}
df["color"] = df["Family"].map(palette_d)
ax = plt.gca()
bar_positions = [1,2,3,
                 4.5, 5.5, 6.5,
                 9, 10, 11, 12, 13, 14, 15.5, 16.5, 17.5, 19, 20, 21, 22, 23]
bars = ax.bar(bar_positions, df['Score'], yerr=[df['sd'][i] / sqrt(df['n'][i]) for i in range(len(df))],
              capsize=5, color=df["color"], edgecolor='.2', alpha = 0.8, lw = 3)
ax.hlines(df['Score_mono'], xmin=[x - 0.3 for x in bar_positions], xmax=[x + 0.3 for x in bar_positions],
          colors='black', linestyles=(0, (1, 1)), linewidth=2) # linestyles=(0, (1, 1))
# add_bracket(ax, bar_positions[0], bar_positions[2], 'translation', y_offset = .01)
# add_bracket(ax, bar_positions[3], bar_positions[5], 'contrastive', y_offset = .01)
# add_bracket(ax, bar_positions[0], bar_positions[5], 'explicit', y_offset = .12, weight = "bold")
# add_bracket(ax, bar_positions[6], bar_positions[11], 'masked LM', y_offset = -.0)
# add_bracket(ax, bar_positions[12], bar_positions[14], 'span corr.', y_offset = -.0)
# add_bracket(ax, bar_positions[15], bar_positions[19], 'causal LM', y_offset = +.1)
# add_bracket(ax, bar_positions[6], bar_positions[19], 'implicit', y_offset = +.19, weight = "bold")
plt.title(title, fontsize=30, weight='bold', pad=20)
plt.xlabel('Model', fontsize=27, labelpad=20)
plt.ylabel('R', fontsize=27, labelpad=20)
plt.ylim(ylimstart, ylim)
plt.xticks(bar_positions, labels = df["Model"], rotation=45, ha='right', fontsize=22)
plt.yticks([.1, .2, .3, .4, .5, .6, .7], fontsize=23)
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

###########################
# Relationship w/ study I #
###########################

pearsonr(studies_merged["Score_mono"], studies_merged["Score"])

r, p = pearsonr(studies_merged['Score_mono'], studies_merged['Score'])

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(studies_merged['Score_mono'], studies_merged['Score'], color=studies_merged['color'], alpha=1, s = 200)
coefficients = np.polyfit(studies_merged['Score_mono'], studies_merged['Score'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(studies_merged['Score_mono'])-.05, max(studies_merged['Score_mono'])+.05, 100)
y_values = polynomial(x_values)
plt.plot(x_values, y_values, ls='--', c='gray')
for i in range(len(studies_merged)):
    plt.errorbar(studies_merged['Score_mono'][i], studies_merged['Score'][i],
                  xerr=studies_merged['se_mono'][i], yerr=studies_merged['se'][i],
                  fmt='o', color=studies_merged['color'][i], zorder = 5)

plt.text(0.03, 0.98, f"r = {round(r, 2)}, p < 0.0001", 
          fontsize=15, ha='left', va='top', alpha=1, 
          bbox=dict(facecolor='white', alpha=0.7), 
          transform=plt.gca().transAxes)

plt.xlabel('Study I WITHIN encoding (R)', fontsize = 17)
plt.ylabel('Study II WITHIN encoding (R)', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.xlim(-90, 45)
#plt.ylim(0.15, 0.6)
#plt.yticks([0.2, 0.3, 0.4, 0.5])
plt.grid(True)
plt.show()

print(studies_merged.mean())
print(studies_merged.std() / np.sqrt(20))
print(ttest_rel(studies_merged["Score_mono"], studies_merged["Score"]))

#########################
# plot SINGLE LANGUAGES #
#########################

# r_lang = {lang : [] for lang in within_all["language"].unique()}
# for model in model_names:
#     for lang in within_all["language"].unique():
#         therow = within_all[(within_all.language == lang) & (within_all.model == model)]
#         r = therow["r_mean"].values[0]
#         r_lang[lang].append(r)

# data = pd.DataFrame(r_lang)

# fig, axes = plt.subplots(20, 1, figsize=(10*.9, 24*.9), dpi = 300)
# yticks = [0, .5, 1]
# for i, ax in enumerate(axes):
#     ax.bar(data.columns, data.iloc[i], color='indianred')
#     for j, value in enumerate(data.iloc[i]):
#         if value == 0:
#             ax.text(j, 0, 'NA', ha='center', va='bottom', fontsize=15, color='black')
#     ax.set_ylim(-.45, 1)
#     ax.set_yticks(yticks)  # Set the y-ticks to [0, 0.3, 0.6, 1]
#     ax.axhline(y=0, color='black',lw=2)
#     # grid
#     ax.xaxis.grid(False)
#     ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
#     if i < len(axes) - 1:
#         ax.set_xticklabels([])  # Hide x-tick labels
#     else:
#         ax.set_xticklabels(data.columns, rotation=45, ha="right")
# plt.suptitle('', y=.97, fontsize=26, weight="bold")
# plt.tight_layout()
# plt.show()

sns.set_context("talk")
r_lang = {lang: [] for lang in within_all["language"].unique()}
r_se_lang = {lang: [] for lang in within_all["language"].unique()}

for model in model_names:
    for lang in within_all["language"].unique():
        therow = within_all[(within_all.language == lang) & (within_all.model == model)]
        r = therow["r_mean"].values[0]
        r_se = therow["r_se"].values[0]
        r_lang[lang].append(r)
        r_se_lang[lang].append(r_se)

data = pd.DataFrame(r_lang)
data_se = pd.DataFrame(r_se_lang)

avg_across_models = data.mean(axis=0)
se_across_models = data.sem(axis=0)

data_with_avg = pd.concat([pd.DataFrame([avg_across_models], index=['Mean']), data]) # append avg row
data_se_with_avg = pd.concat([pd.DataFrame([se_across_models], index=['Mean']), data_se])

fig_height = 26 * 0.9
fig_width = 10 * 0.9

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
gs = gridspec.GridSpec(len(data_with_avg), 1, height_ratios=[1.5] + [1] * (len(data_with_avg) - 1), hspace=0.5)
yticks = [0, 0.5, 1]
axes = [fig.add_subplot(gs[i]) for i in range(len(data_with_avg))]
for i, ax in enumerate(axes):
    bar_color = 'steelblue' if i == 0 else 'indianred'  # avg row is blue, others are red
    ax.bar(
        data_with_avg.columns, 
        data_with_avg.iloc[i], 
        yerr=data_se_with_avg.iloc[i], 
        color=bar_color, 
        capsize=5
    )
    ax.set_ylim(-0.45, 1)
    ax.set_yticks(yticks)
    ax.axhline(y=0, color='black', lw=2)
    ax.xaxis.grid(False)
    if i == 0:
        ax.set_ylabel('Average', rotation=0, ha='right', va='center')
        ax.set_xticklabels([])
    else:
        ax.set_ylabel(names_formatted[i - 1], rotation=0, ha='right', va='center')
        ax.set_xticklabels([])  
axes[-1].set_xticklabels(data_with_avg.columns, rotation=45, ha="right")
plt.suptitle('', y=0.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()