import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr, norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice.")

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

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

##########################
# load registered models #
##########################

passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_code_dict = {k : v for k, v in zip(lang_codes, languages)}
lang_code_dict_inv = {v : k for k, v in lang_code_dict.items()}

with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

# NEW LAYERS FOR REVISION -- new CV approach caused (slightly) different peaks
dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 15,
                  "nllb200_1B" : 17,
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 9,
                  "xlmr_large" : 14,
                  "distilmbert" : 4, 
                  "bert_base" : 6,
                  "mdeberta" : 9, 
                  "mt5_small" : 2, 
                  "mt5_base" : 9,
                  "mt5_large" : 17, 
                  "mgpt" : 14, 
                  "xglm_small" : 10, 
                  "xglm_med" : 16, 
                  "xglm_large" : 40, 
                  "xglm_xl" : 48}

model_names = dict_bestlayer.keys()

d_models = {}
for train_name in ["main", "pereira", "control", "natstor"]:
    norm_params = {}
    models = {}
    models_random = {}
    for modelname in model_names:
        with open(f"registered_models/{train_name}/normaliz_params/{modelname}", 'rb') as handle:
            file_norm = pickle.load(handle)
            norm_params[modelname] = file_norm
        with open(f"registered_models/{train_name}/{modelname}", 'rb') as handle:
            file = pickle.load(handle)
            models[modelname] = file
        for random_idx in [0, 1, 2, 3]: # new in revision, different seeds
            with open(f"registered_models/{train_name}/{modelname}_random_{random_idx}", 'rb') as handle:
                file_random = pickle.load(handle)
                try:
                    models_random[modelname][random_idx] = file_random
                except KeyError:
                    models_random[modelname] = {random_idx : file_random}
    d_models[train_name] = {"norm" : norm_params, "models" : models, "random" : models_random}
        
##################################
# TEST encoding models zero-shot #
##################################

# Passages to remove (no sig. correlation in fMRI time-series):
# Korean [2, 3]
# German [1]
# Portuguese [2]
# Hindi [1]

d_passages_keep = {'ar' : [1, 2, 3], 'de' : [2, 3], 'hi' : [2, 3], 'it' : [1, 2, 3], 'ko' : [1], 'pt' : [1, 3], 'ru' : [1, 2, 3], 'zh' : [1, 2, 3], 'pl' : [1, 2, 3]}

transf_results = []
# main results
for lang in lang_codes:
    passages_keep = d_passages_keep[lang]
    print("\n", lang_code_dict[lang])
    for train_name in d_models.keys():
        print("\n", train_name)
        norm_params = d_models[train_name]["norm"]
        models = d_models[train_name]["models"]
        for regname, reg in models.items():
            X_scaler, y_scaler = norm_params[regname] # previous norm parameters
            X = [X_scaler.transform(preproc_align(lang, passage, load(f"{passage}/{regname}_{lang}")[dict_bestlayer[regname]])) for passage in passages]
            predictions = [reg.predict(the_x) for the_x in X]
            responses = [y_scaler.transform(d[passage][lang_code_dict[lang]].reshape(-1, 1)).flatten() for passage in passages]
            rs = [pearsonr(pred, resp)[0] for pred, resp in zip(predictions, responses)]
            print(regname, rs)
            transf_results.append([train_name, "experimental", lang_code_dict[lang], regname, rs[0], rs[1], rs[2], np.mean([rs[i-1] for i in passages_keep]), np.std([rs[i-1] for i in passages_keep])])

#random results
for lang in lang_codes:
    passages_keep = d_passages_keep[lang]
    print("\n", lang_code_dict[lang])
    for train_name in d_models.keys():
        print("\n", train_name)
        norm_params = d_models[train_name]["norm"]
        models_random = d_models[train_name]["random"]
        for regname, reg in models_random.items():
            for random_idx in [0, 1, 2, 3]:
                X_scaler, y_scaler = norm_params[regname] # previous norm parameters
                X = [X_scaler.transform(preproc_align(lang, passage, load(f"{passage}/{regname}_{lang}")[dict_bestlayer[regname]])) for passage in passages]
                predictions = [reg[random_idx].predict(the_x) for the_x in X]
                responses = [y_scaler.transform(d[passage][lang_code_dict[lang]].reshape(-1, 1)).flatten() for passage in passages]
                rs = [pearsonr(pred, resp)[0] for pred, resp in zip(predictions, responses)]
                print(regname, rs)
                transf_results.append([train_name, f"random_{random_idx}", lang_code_dict[lang], regname, rs[0], rs[1], rs[2], np.mean([rs[i-1] for i in passages_keep]), np.std([rs[i-1] for i in passages_keep])])

transf_results = pd.DataFrame(transf_results, columns = ["train", "condition", "language", "model", "r1", "r2", "r3", "r_mean", "r_sd"])
#transf_results.to_csv("confirmatory_results.csv", index=False)
transf_results = pd.read_csv("confirmatory_results.csv")

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

def combine_z_statistics(z_stats, return_z = True):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    if return_z:
        out = z_combined, combined_pvalue
    else:
        out = combined_pvalue
    return out

results_sig = []
for pair in transf_results.groupby(["language", "train", "model"]):
    language, corpus, model = pair[0]
    lang_code = lang_code_dict_inv[language]
    keep_passages = d_passages_keep[lang_code]
    df = pair[1]
    rs = [df[df["condition"] == "experimental"][["r1", "r2", "r3"]].to_numpy()[0][idx-1] for idx in keep_passages]
    full_rs_random = [] # aggregating over passages and random seeds / circular shifts
    full_zs = [] # this is aggregating over random seeds
    for random_idx in [0, 1, 2, 3]:
        rs_random = [df[df["condition"] == f"random_{random_idx}"][["r1", "r2", "r3"]].to_numpy()[0][idx-1] for idx in keep_passages]
        full_rs_random.extend(rs_random)
        zs = [] # this is aggregating over passages
        for r_exp, r_rand in zip(rs, rs_random):
            z, p = r_to_z(r_exp, r_rand)
            zs.append(z)
        z_combined, p_combined = combine_z_statistics(zs, return_z = True)
        full_zs.append(z_combined)
    z_combined_full, p_combined_full = combine_z_statistics(full_zs, return_z = True)
    results_sig.append([language, corpus, model, np.mean(rs), np.mean(full_rs_random), z_combined_full, p_combined_full])
results_sig = pd.DataFrame(results_sig, columns = ["lang", "training", "model", "r_mean", "r_random", "z", "p"])

combined_df = []
for (model, corpus), df in results_sig.groupby(["model", "training"]):
    #print(df)
    z_combined_full, p_combined_full = combine_z_statistics(df["z"])
    print(f"{corpus.upper()} -- {z_combined_full:.3}, {p_combined_full:.5}")
    if p_combined_full < 0.001:
        asterisk = "***"
    elif p_combined_full < 0.01:
        asterisk = "**"
    elif p_combined_full < 0.05:
        asterisk = "*"
    # elif p_combined_full < 0.1:
    #     asterisk = "."
    else:
        asterisk = ""
    combined_df.append([corpus, model, df["r_mean"].mean(), df["r_random"].mean(), df["r_mean"].std() / np.sqrt(len(df)), z_combined_full, p_combined_full, asterisk])
combined_df = pd.DataFrame(combined_df, columns = ["train", "model", "r", "r_random", "SE", "z", "p", "sig"])
# combined_df.to_csv("confirmatory_results_aggregated.csv")

combined_df = pd.read_csv("confirmatory_results_aggregated.csv")
    
############
# PLOTTING #
############

# base barplot

label_map = {"main": "Study I","natstor": "NatStories","control": "Tuckute2024","pereira": "Pereira2018"}
label_order = ["Study I", "NatStories", "Tuckute2024", "Pereira2018"]
colors = {"Study I": "darkslateblue","NatStories": "lightsteelblue","Tuckute2024": "tab:red","Pereira2018": "lightsalmon"}
combined_df["train_label"] = combined_df["train"].map(label_map)
group_stats = combined_df.groupby("train_label").agg(
    mean_r=("r", "mean"),
    se_r=("SE", "mean")
).loc[label_order]
fig, ax = plt.subplots(dpi = 400, figsize=(4*.7, 3*.7))
for i, label in enumerate(label_order):
    bar = ax.bar(i, group_stats.loc[label, "mean_r"],
                 yerr=group_stats.loc[label, "se_r"],
                 capsize=5, color=colors[label], edgecolor="black",
                 linewidth=1.2, zorder=2)
for i, label in enumerate(label_order):
    y_vals = combined_df[combined_df["train_label"] == label]["r"]
    jitter = np.random.normal(loc=0, scale=0.08, size=len(y_vals))
    x_vals = i + jitter
    ax.scatter(x_vals, y_vals, color="black", alpha=0.4, s=20, zorder=3)
ax.set_ylabel("R", fontsize=13)
# ax.set_xlabel("Training dataset", fontsize=13)
ax.set_xticks(range(len(label_order)))
ax.set_xticklabels(label_order, fontsize=12, rotation = 25, ha="right")
ax.tick_params(axis="y", labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
ax.set_ylim(None, group_stats["mean_r"].max() + 0.1)
plt.tight_layout()
plt.savefig("../plots/study2_summary.svg", format="svg", bbox_inches="tight")
plt.show()


model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# palette_d = {'BERT' : "steelblue",
#              'DeBERTa' : "teal",
#              'InfoXLM' : "firebrick",
#              'NLLB' : "tomato",
#              'XGLM' : "forestgreen",
#              'XLM-Align' : "firebrick",
#              'XLM-R' : "lightsteelblue",
#              'mGPT' : "yellowgreen",
#              'mT5' : "darkorange"}
# combined_df["Family"] = combined_df["model"].map(class_dict)
# combined_df["color"] = combined_df["Family"].map(palette_d)

bar_positions = [1,2,3,
                 4.5, 5.5, 6.5,
                 9, 10, 11, 12, 13, 14, 15.5, 16.5, 17.5, 19, 20, 21, 22, 23]
###########
# STORIES #
###########

train_names = ["main", "natstor"]
train_names_nice = ["Study 1", "NatStor"]
colors = ["darkslateblue", "lightsteelblue"]

plt.figure(figsize=(24*.7, 11.5*.7), dpi=300)
sns.set_context("talk")
ax = plt.gca()
c = -.2
for train_name, title, color in zip(train_names, train_names_nice, colors):
    
    df = combined_df[combined_df["train"] == train_name]
    df['model'] = pd.Categorical(df['model'], categories=model_names, ordered=True)
    df = df.sort_values('model').reset_index(drop=True)

    x_vals = [pos + c for pos in bar_positions]

    for i, row in df.iterrows():
        if row["sig"] in [np.nan]:   
            marker_color = "gray"
            alpha = 0.3
        else:
            marker_color = color
            alpha = 0.9

        ax.errorbar(
            x_vals[i],
            row['r'],
            yerr=row['SE'],
            fmt='o',
            color=marker_color,
            ecolor=marker_color,
            elinewidth=3,
            capsize=5,
            markersize=17,
            alpha=alpha,
            markeredgecolor=".2"
        )

    c += .4

for spine in ax.spines.values():
    spine.set_linewidth(3.5)
ax.grid(axis="y", linestyle="--", color="gray", alpha=0.5, linewidth = 3.5)
plt.axhline(y=0, color='black', lw=3.5)
plt.xlabel('Model', fontsize=33, labelpad=20)
plt.ylabel('R', fontsize=33, labelpad=20)
plt.ylim(-.07, .35)
plt.xticks(bar_positions, labels=names_formatted, rotation=45, ha='right', fontsize=24)
plt.yticks([0, .1, .2, .3], fontsize=27)
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("../plots/sudy2_main_natstor.svg", format="svg", bbox_inches="tight")
plt.show()

#############
# SENTENCES #
#############
    
train_names = ["control", "pereira"]
train_names_nice = ["Tuckute 2024", "Pereira 2018"]
colors = ["tab:red", "lightsalmon"]

plt.figure(figsize=(24*.7, 11.5*.7), dpi=300)
sns.set_context("talk")
ax = plt.gca()
c = -.2
for train_name, title, color in zip(train_names, train_names_nice, colors):
    
    df = combined_df[combined_df["train"] == train_name]
    df['model'] = pd.Categorical(df['model'], categories=model_names, ordered=True)
    df = df.sort_values('model').reset_index(drop=True)

    x_vals = [pos + c for pos in bar_positions]

    for i, row in df.iterrows():
        if row["sig"] in [np.nan]:   
            marker_color = "gray"
            alpha = 0.3
        else:
            marker_color = color
            alpha = 0.9

        ax.errorbar(
            x_vals[i],
            row['r'],
            yerr=row['SE'],
            fmt='o',
            color=marker_color,
            ecolor=marker_color,
            elinewidth=3,
            capsize=5,
            markersize=17,
            alpha=alpha,
            markeredgecolor=".2"
        )

    c += .4

for spine in ax.spines.values():
    spine.set_linewidth(3.5)
ax.grid(axis="y", linestyle="--", color="gray", alpha=0.5, linewidth = 3.5)
plt.axhline(y=0, color='black', lw=3.5)
plt.xlabel('Model', fontsize=33, labelpad=20)
plt.ylabel('R', fontsize=33, labelpad=20)
plt.ylim(-.07, .35)
plt.xticks(bar_positions, labels=names_formatted, rotation=45, ha='right', fontsize=24)
plt.yticks([0, .1, .2, .3], fontsize=27)
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("../plots/sudy2_control_pereira.svg", format="svg", bbox_inches="tight")
plt.show()

train_names = ["main", "natstor", "control", "pereira"]
for train_name in train_names:
    print(train_name)
    df = combined_df[combined_df["train"] == train_name]
    idx = df["r"].argmax()
    model = df[["model", "r", "p", "SE"]].iloc[idx]
    print(model)
    print(((df["p"]<.05) & (df["z"] > 0)).sum(), "\n\n")

########################
# INDIVIDUAL LANGUAGES #
########################

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

train_names = ["control", "pereira"]
train_names_nice = ["Tuckute 2024", "Pereira 2018"]
colors = ["tab:red", "lightsalmon"]
x_offsets = [-.2, .2]
yticks = [-.5, 0, .5]

# fig, axes = plt.subplots(20, 1, figsize=(10*.9, 24*.9), dpi=300)
# for train_name, title, color, x_offset in zip(train_names, train_names_nice, colors, x_offsets):
#     for i, model in enumerate(model_names):
#         data = transf_results[(transf_results["condition"] == "experimental") & (transf_results["train"] == train_name) & (transf_results["model"] == model)][["language", "r_mean"]]
#         data.index = data["language"]
#         del data["language"]
#         data = data.T
#         ax = axes[i]
        
#         x_positions = np.arange(len(data.columns)) + x_offset  # Set x positions with offset
#         ax.bar(x_positions, data.iloc[0], width = 0.37, color=color)
        
#         ax.set_xticks(np.arange(len(data.columns)))  
#         ax.set_xticklabels(data.columns, rotation=45, ha="right")
#         ax.set_ylim(-.5, .6)
#         ax.set_yticks(yticks)
#         ax.axhline(y=0, color='black',lw=2)
#         # grid
#         ax.xaxis.grid(False)
#         ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
#         if i < len(axes) - 1:
#             ax.set_xticks([])  
#             ax.set_xticklabels([])
#         else:
#             ax.set_xticks(np.arange(len(data.columns)))  
#             ax.set_xticklabels(data.columns, rotation=45, ha="right")
# plt.suptitle('', y=.97, fontsize=26, weight="bold")
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(len(model_names) + 1, 1, figsize=(10 * .9, (25.2) * .9), dpi=300)
# for train_name, title, color, x_offset in zip(train_names, train_names_nice, colors, x_offsets):
#     r_lang = {lang: [] for lang in transf_results["language"].unique()}
#     # results for each model
#     for i, model in enumerate(model_names):
#         data = transf_results[(transf_results["condition"] == "experimental") & (transf_results["train"] == train_name) & (transf_results["model"] == model)][["language", "r_mean"]]
#         data.index = data["language"]
#         del data["language"]
#         data = data.T
#         r_lang.update({col: r_lang[col] + [data[col].iloc[0]] for col in data.columns})

#         ax = axes[i]
#         x_positions = np.arange(len(data.columns)) + x_offset
#         ax.bar(x_positions, data.iloc[0], width=0.37, color=color)
#         ax.set_xticks(np.arange(len(data.columns)))
#         ax.set_xticklabels(data.columns, rotation=45, ha="right")
#         ax.set_ylim(-.5, .6)
#         ax.set_yticks(yticks)
#         ax.axhline(y=0, color='black', lw=2)
#         ax.xaxis.grid(False)
#         ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
#         if i < len(axes) - 1:
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#         else:
#             ax.set_xticks(np.arange(len(data.columns)))
#             ax.set_xticklabels(data.columns, rotation=45, ha="right")
    
#     # mean encoding performance
#     r_mean = pd.DataFrame(r_lang).mean()
#     yerr = pd.DataFrame(r_lang).std() / np.sqrt(20)
    
#     # mean row at the bottom
#     ax_mean = axes[len(model_names)]  # last subplot
#     x_positions = np.arange(len(r_mean.index)) + x_offset
#     ax_mean.bar(x_positions, r_mean, width=0.37, color=color)  # same color for mean
#     plt.errorbar(x_positions, r_mean, yerr = yerr, fmt='none', capsize=5, capthick=1, color='black') 
#     ax_mean.set_xticks(np.arange(len(r_mean.index)))
#     ax_mean.set_xticklabels(r_mean.index, rotation=45, ha="right")
#     ax_mean.set_ylim(-.5, .6)
#     ax_mean.set_yticks(yticks)
#     ax_mean.axhline(y=0, color='black', lw=2)
#     ax_mean.xaxis.grid(False)
#     ax_mean.set_ylabel("Average", rotation=0, ha='right', va='center')

# plt.suptitle('', y=.97, fontsize=26, weight="bold")
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(10 * 0.9, 25.2 * 0.9), dpi=300)
gs = gridspec.GridSpec(len(model_names) + 1, 1, height_ratios=[1.65] + [1] * len(model_names), hspace=0.5)
axes = [fig.add_subplot(gs[i]) for i in range(len(model_names) + 1)]
for train_name, title, color, x_offset in zip(train_names, train_names_nice, colors, x_offsets):
    r_lang = {lang: [] for lang in transf_results["language"].unique()}
    # results for each model
    for i, model in enumerate(model_names):
        data = transf_results[
            (transf_results["condition"] == "experimental") & 
            (transf_results["train"] == train_name) & 
            (transf_results["model"] == model)
        ][["language", "r_mean"]]
        data.index = data["language"]
        del data["language"]
        data = data.T
        r_lang.update({col: r_lang[col] + [data[col].iloc[0]] for col in data.columns})
        # model results
        ax = axes[i + 1]  # shift all individual model plots down by 1
        x_positions = np.arange(len(data.columns)) + x_offset
        ax.bar(x_positions, data.iloc[0], width=0.37, color=color)
        ax.set_ylim(-0.5, 0.6)
        ax.set_yticks(yticks)
        ax.axhline(y=0, color='black', lw=2)
        ax.xaxis.grid(False)
        ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
        ax.set_xticks([])
        ax.set_xticklabels([])
    r_mean = pd.DataFrame(r_lang).mean()
    yerr = pd.DataFrame(r_lang).std() / np.sqrt(len(model_names))
    ax_avg = axes[0]
    x_positions = np.arange(len(r_mean.index)) + x_offset
    ax_avg.bar(x_positions, r_mean, width=0.37, color=color)
    ax_avg.errorbar(x_positions, r_mean, yerr=yerr, fmt='none', capsize=5, capthick=1, color='black')
    ax_avg.set_ylim(-0.5, 0.6)
    ax_avg.set_yticks(yticks)
    ax_avg.axhline(y=0, color='black', lw=2)
    ax_avg.xaxis.grid(False)
    ax_avg.set_ylabel("Average", rotation=0, ha='right', va='center')
    ax_avg.set_xticks([])
    ax_avg.set_xticklabels([])
axes[-1].set_xticks(np.arange(len(r_mean.index)))
axes[-1].set_xticklabels(r_mean.index, rotation=45, ha="right")
plt.suptitle('', y=0.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()



train_names = ["main", "natstor"]
train_names_nice = ["Study 1", "NatStor"]
colors = ["darkslateblue", "lightsteelblue"]

fig = plt.figure(figsize=(10 * 0.9, 25.2 * 0.9), dpi=300)
gs = gridspec.GridSpec(len(model_names) + 1, 1, height_ratios=[1.65] + [1] * len(model_names), hspace=0.5)
axes = [fig.add_subplot(gs[i]) for i in range(len(model_names) + 1)]
for train_name, title, color, x_offset in zip(train_names, train_names_nice, colors, x_offsets):
    r_lang = {lang: [] for lang in transf_results["language"].unique()}
    # results for each model
    for i, model in enumerate(model_names):
        data = transf_results[
            (transf_results["condition"] == "experimental") & 
            (transf_results["train"] == train_name) & 
            (transf_results["model"] == model)
        ][["language", "r_mean"]]
        data.index = data["language"]
        del data["language"]
        data = data.T
        r_lang.update({col: r_lang[col] + [data[col].iloc[0]] for col in data.columns})
        # model results
        ax = axes[i + 1]  # shift all individual model plots down by 1
        x_positions = np.arange(len(data.columns)) + x_offset
        ax.bar(x_positions, data.iloc[0], width=0.37, color=color)
        ax.set_ylim(-0.5, 0.6)
        ax.set_yticks(yticks)
        ax.axhline(y=0, color='black', lw=2)
        ax.xaxis.grid(False)
        ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
        ax.set_xticks([])
        ax.set_xticklabels([])
    r_mean = pd.DataFrame(r_lang).mean()
    yerr = pd.DataFrame(r_lang).std() / np.sqrt(len(model_names))
    ax_avg = axes[0]
    x_positions = np.arange(len(r_mean.index)) + x_offset
    ax_avg.bar(x_positions, r_mean, width=0.37, color=color)
    ax_avg.errorbar(x_positions, r_mean, yerr=yerr, fmt='none', capsize=5, capthick=1, color='black')
    ax_avg.set_ylim(-0.5, 0.6)
    ax_avg.set_yticks(yticks)
    ax_avg.axhline(y=0, color='black', lw=2)
    ax_avg.xaxis.grid(False)
    ax_avg.set_ylabel("Average", rotation=0, ha='right', va='center')
    ax_avg.set_xticks([])
    ax_avg.set_xticklabels([])
axes[-1].set_xticks(np.arange(len(r_mean.index)))
axes[-1].set_xticklabels(r_mean.index, rotation=45, ha="right")
plt.suptitle('', y=0.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()