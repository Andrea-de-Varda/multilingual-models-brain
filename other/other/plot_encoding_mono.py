#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:14:23 2024

@author: dev
"""
import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.stats import pearsonr, norm
from math import sqrt
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns
from adjustText import adjust_text
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import seaborn as sns
import copy
import itertools

chdir("/home/dev/Documents/PhD/Alice")

def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_predictions(name):
    with open(f"results/predictions/{name}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def fullseries_r(langs, model_prefix, n_layers, random = False, random_prefix = ""):
    layerwise_dict = {}
    for n in range(n_layers+1):
        print(f"Processing layer {n}")
        m = []
        for idx, lang in enumerate(langs):
            if random:
                # IMPORTANT : CORRECT THIS ONCE WE HAVE RANDOM DATA
                y_tot, pred = load_predictions(f"sequential_{model_prefix}_{lang}_{n}")
            else:
                y_tot, pred = load_predictions(f"sequential_{model_prefix}_{lang}_{n}")
            r = pearsonr(pred, y_tot)[0]
            m.append(r)
        #########################
        sd = [0] * len(m)
        df = pd.DataFrame(zip(langs, m, sd), columns=["lang", "m", "sd"])
        layerwise_dict[n] = df
    save(layerwise_dict, f"results/sanity_check/{random_prefix}monolingual_{model_prefix}")
    return layerwise_dict

all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
lang_code_d_reversed = {v : k for k, v in lang_code_dict.items()}


xglm_langs = ["ca", "ja", "en", "es", "vi", "ta", "tr", "fr"]

xglm_small  = fullseries_r(xglm_langs, "xglm_small", 24)
xglm_med    = fullseries_r(xglm_langs, "xglm_med", 24)
xglm_large  = fullseries_r(xglm_langs, "xglm_large", 48)
xglm_xl     = fullseries_r(xglm_langs, "xglm_xl", 48)
mbert       = fullseries_r(all_codes, "bert_base", 12)
distilmbert = fullseries_r(all_codes, "distilmbert", 6)
xlmr_base   = fullseries_r(all_codes, "xlmr_base", 12)
xlmr_large  = fullseries_r(all_codes, "xlmr_large", 24)
mt5_small   = fullseries_r(all_codes, "mt5_small", 8)
mt5_base    = fullseries_r(all_codes, "mt5_base", 12)
mt5_large   = fullseries_r(all_codes, "mt5_large", 24)

# random
xglm_small  = fullseries_r(xglm_langs, "xglm_small", 24, random = True, random_prefix = "random_")
xglm_med    = fullseries_r(xglm_langs, "xglm_med", 24, random = True, random_prefix = "random_")
xglm_large  = fullseries_r(xglm_langs, "xglm_large", 48, random = True, random_prefix = "random_")
xglm_xl     = fullseries_r(xglm_langs, "xglm_xl", 48, random = True, random_prefix = "random_")
mbert       = fullseries_r(all_codes, "bert_base", 12, random = True, random_prefix = "random_")
distilmbert = fullseries_r(all_codes, "distilmbert", 6, random = True, random_prefix = "random_")
xlmr_base   = fullseries_r(all_codes, "xlmr_base", 12, random = True, random_prefix = "random_")
xlmr_large  = fullseries_r(all_codes, "xlmr_large", 24, random = True, random_prefix = "random_")
mt5_small   = fullseries_r(all_codes, "mt5_small", 8, random = True, random_prefix = "random_")
mt5_base    = fullseries_r(all_codes, "mt5_base", 12, random = True, random_prefix = "random_")
mt5_large   = fullseries_r(all_codes, "mt5_large", 24, random = True, random_prefix = "random_")

###############################################################################
# plot ########################################################################
###############################################################################

def load(model_prefix, monol = True, split_context = False, random = False):
    if monol:
        if split_context:
            with open(f"results/sanity_check/split_context_monolingual_{model_prefix}", 'rb') as handle:
                file = pickle.load(handle)
        else:
            if random:
                with open(f"results/sanity_check/random_monolingual_{model_prefix}", 'rb') as handle:
                    file = pickle.load(handle)
            else:
                with open(f"results/sanity_check/monolingual_{model_prefix}", 'rb') as handle:
                    file = pickle.load(handle)
    else:
        raise ValueError('No multilingual sequential split')
    return file

def find_median_index(lst):
    sorted_lst = sorted(lst)
    mid_point = len(lst) // 2
    if len(lst) % 2 == 1:  # If odd, take the middle value
        median = sorted_lst[mid_point]
    else:  # If even, take the lower of the two middle values
        median = sorted_lst[mid_point - 1]
    return lst.index(median)

def get_best_layerwise(res_dict, colname = "m", give_mean = True, give_all = False):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    print(f"Best layer is {idx_max}")
    if give_all:
        best = res_dict[idx_max][colname].tolist()
    else:
        if give_mean:
            best = mean_results[idx_max]
        else:
            best = sd_results[idx_max] # so the SD is the cross-lingual variation for the best layer
    return best

def get_median_layerwise(res_dict, colname = "m", give_mean = True, give_all = False):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_median = find_median_index(mean_results)
    if give_all:
        median = res_dict[idx_median][colname].tolist()
    else:
        if give_mean:
            median = mean_results[idx_median]
        else:
            median = sd_results[idx_median] # so the SD is the cross-lingual variation for the best layer
    return median

def plot_aggregate(df, title, ylim=None, ylimstart=None, sig=[]):
    plt.figure(figsize=(14*.7, 11.5*.7), dpi = 300)
    sns.set_context("talk")
    palette = sns.color_palette("tab10")  # Colorblind-friendly palette
    ax = sns.barplot(x='Model', y='Score', hue='Family', data=df,
                     dodge=False, palette=palette, edgecolor='.2')
    for i in range(len(df['Score'])):
        plt.errorbar(i, df['Score'][i], yerr=df['sd'][i]/sqrt(df["n"][i]), fmt='none', capsize=5, ecolor='black', capthick=2)

    # significance asterisks
    for i, value in enumerate(df['Score']):
        if i < len(sig):
            y = value + df['sd'][i]/sqrt(df["n"][i]) + 0.02
            plt.text(i, y, sig[i], ha='center', va='bottom', color='black', fontsize=20, weight='bold')

    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=23)
    # leg = plt.legend(title='Model Family', title_fontsize='20', fontsize='18', loc='upper left', bbox_to_anchor=(1, 1))
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(4.0)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    ax.get_legend().remove()
    plt.show()

# Specifying model names

model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["XLM-R$_{base}$", "XLM-R$_{large}$", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "DistilmBERT", "mBERT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
model_family = ["XLM-R", "XLM-R", "mT5", "mT5", "mT5", "BERT", "BERT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [15, 15, 15, 15, 15, 15, 15, 8, 8, 8, 8] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# Specifying language names

langs = ['ja', 'fr', 'ta', 'ca', 'es', 'tr', 'vi', 'en', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice = ['Japanese', 'French', 'Tamil', 'Catalan', 'Spanish', 'Turkish', 'Vietnamese', 'English', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

lang_dict = {k : v for k, v in zip(langs, langs_nice)}

###############################################################################
########################
# SIGNIFICANCE TESTING #
########################

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

# stouffer method
def combine_z_statistics(z_stats):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    return combined_pvalue

# asterisks for plotting
def add_significance_asterisks(data):
    significance_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]
    for item in data:
        model, p_value = item
        asterisk = '' # default non-sig
        for threshold, symbol in significance_levels:
            if p_value < threshold:
                asterisk = symbol
                break
        item.append(asterisk)
    return data


p_values_best = []
for model in model_names:
    test = get_best_layerwise(load(model), give_all = True)
    rand = get_best_layerwise(load(model, random=True), give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, rand):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    p_values_best.append([model, p])
p_values_best = add_significance_asterisks(p_values_best)
p_values_best = pd.DataFrame(p_values_best, columns = ["model", "p", "asterisk"])
    
p_values_median = []
for model in model_names:
    test = get_median_layerwise(load(model), give_all = True)
    rand = get_median_layerwise(load(model, random=True), give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, rand):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    p_values_median.append([model, p])
p_values_median = add_significance_asterisks(p_values_median)
p_values_median = pd.DataFrame(p_values_median, columns = ["model", "p", "asterisk"])

###########################
# barplot with best layer #
###########################

# monolingual, best layer #####################################################
best_monol = [get_best_layerwise(load(model)) for model in model_names]
best_monol_sd = [get_best_layerwise(load(model), give_mean = False) for model in model_names]

best_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol,
    'Family': model_family,
    'sd' : best_monol_sd,
    'n' : n_langs
})
plot_aggregate(best_layer, "",  ylim = .85, ylimstart = -.4)
plot_aggregate(best_layer, "",  ylim = .85, ylimstart = -.1, sig = p_values_best["asterisk"])

# monolingual, median layer ###################################################
median_monol = [get_median_layerwise(load(model)) for model in model_names]
median_monol_sd = [get_median_layerwise(load(model), give_mean = False) for model in model_names]

median_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol,
    'Family': model_family,
    'sd' : median_monol_sd,
    'n' : n_langs
})
plot_aggregate(median_layer, "", ylim = .85, ylimstart = -.4)
plot_aggregate(median_layer, "", ylim = .85, ylimstart = -.1, sig = p_values_median["asterisk"])

# random, best layer ##########################################################
best_monol_random = [get_best_layerwise(load(model, random=True)) for model in model_names]
best_monol_sd_random = [get_best_layerwise(load(model, random=True), give_mean = False) for model in model_names]

best_layer_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol_random,
    'Family': model_family,
    'sd' : best_monol_sd_random,
    'n' : n_langs
})

plot_aggregate(best_layer_random, "",  ylim = .85, ylimstart = -.2)

# random, median layer ########################################################
median_monol_random = [get_median_layerwise(load(model, random=True)) for model in model_names]
median_monol_sd_random = [get_median_layerwise(load(model, random=True), give_mean = False) for model in model_names]

median_layer_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol_random,
    'Family': model_family,
    'sd' : median_monol_sd_random,
    'n' : n_langs
})

plot_aggregate(median_layer_random, "",  ylim = .85, ylimstart = -.2)

###############################################################################

colname = "m"
out_dfs = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi = 300)
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)
ax = sns.lineplot(
    data=out_dfs,
    x='l', 
    y='m', 
    hue='Class',  # Color by class
    style='Model',  # Different markers for each name
    markers=True, 
    dashes=False, 
    palette=palette
)
ax.set_xlabel('Layer position', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
ax.set_title('Monolingual encoding by layer', fontsize=30, weight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(title_fontsize='22', fontsize='18', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
ax.set_xlim([out_dfs['l'].min()-.02, out_dfs['l'].max()+.02])
ax.set_ylim(-.1, .4)
plt.show()

###############################################################################

colname = "m"
sequential = True # change accordingly
out_dfs = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)
model_colors = {model: color for model, color in zip(['BERT', 'XGLM', 'XLM-R', 'mT5'], sns.color_palette("tab10", n_colors=4))}
model_markers = {model: marker for model, marker in zip(names_nice_dict.values(), ['>', 'v', '^', 's', 'o', '<', 'p', '*', 'h', 'H', 'D'])}
out_dfs['color'] = out_dfs['Class'].map(model_colors)
out_dfs['marker'] = out_dfs['Model'].map(model_markers)

sns.set_style('whitegrid')
sns.set_context('talk')
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)

n_cols = 2
n_rows = (n_classes + n_cols - 1) // n_cols

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*1.2, 8*1.2), dpi=300)
axes = axes.flatten() 

class_order = ['XLM-R', 'BERT', 'mT5', 'XGLM']
for i, model_class in enumerate(class_order):
    ax = axes[i]
    group_data = out_dfs[out_dfs['Class'] == model_class]
    for model in group_data['Model'].unique():
        model_data = group_data[group_data['Model'] == model]
        sns.lineplot(
            data=model_data,
            x='l', 
            y='m',
            color=model_data['color'].iloc[0],
            marker=model_data['marker'].iloc[0],
            label=model,
            ax=ax
        )
    #ax.legend(title='Model', fontsize=13, title_fontsize=20, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title(model_class, fontsize=25)
    ax.set_xlabel('')
    ax.set_xticks([.2, .4, .6, .8])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel('')
    ax.set_xlim([group_data['l'].min()-.02, group_data['l'].max()+.02])
    ax.set_ylim(-.1, .4)
    ax.get_legend().remove()
fig.text(0.56, 0.04, 'Layer position', ha='center', va='center', fontsize=23)
fig.text(0.04, 0.5, 'R', ha='center', va='center', rotation='vertical', fontsize=23)

legend_handles = [Line2D([0], [0], color=model_colors[class_dict_nice[model]], marker=model_markers[model], label=model, linestyle='-', markersize=10) for model in out_dfs['Model'].unique()]

fig.legend(handles=legend_handles, loc='upper right', fontsize=18, title_fontsize=24, title='Model', bbox_to_anchor=(1.25, .95))
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()

################################
# Encoding results by language #
################################

colname = "m"
sequential = True
r_lang = {lang : [] for lang in langs}
sd_lang = {lang : [] for lang in langs}
for model in model_names:
    res_dict = load(model)
    # first selecting best layer
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    df = res_dict[idx_max]
    for lang in langs:
        try:
            therow = df[df.lang == lang]
            r = therow.values[0][1]
            sd = therow.values[0][2]
            r_lang[lang].append(r)
            sd_lang[lang].append(sd)
        except IndexError: # xglm models miss some languages
            r_lang[lang].append(0)
            sd_lang[lang].append(0)

data = pd.DataFrame(r_lang)
data1 = pd.DataFrame(sd_lang) # IMPORTANT: here the SD is the variation across languages

fig, axes = plt.subplots(11, 1, figsize=(10*.9, 15*.9), dpi=300)
yticks = [0, .5, 1]
for i, ax in enumerate(axes):
    ax.bar(data.columns, data.iloc[i], color='indianred', yerr=data1.iloc[i])
    ax.set_ylim(-.45, 1)
    ax.set_yticks(yticks)  # Set the y-ticks to [0, 0.3, 0.6, 1]
    ax.axhline(y=0, color='black',lw=2)
    # grid
    ax.xaxis.grid(False)
    ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
    if i < len(axes) - 1:
        ax.set_xticklabels([])  # Hide x-tick labels
    else:
        ax.set_xticklabels([lang_dict[l] for l in data.columns], rotation=45, ha="right")
plt.suptitle('', y=.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()


###############################################################################
###############################################################################
###############################################################################

# SPLIT CONTEXT

def embed_words(embeddings, words_id, start, end):
    ids = words_id.astype(int)
    time = np.arange(start, end, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, chunks):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    new_chunks = {}
    for chunk_num, n in enumerate(range(0, 260, 26)):
        embeddings = chunks[chunk_num]
        start, end = n, n+26
        subset = df[(df["end"] > start) & (df["end"] < end)]
        time = np.arange(start, end, 2) # sampled each 2 sec
        time_words = subset["end"]
        words_id = np.zeros([len(time_words)])
        for i in range(len(time_words)):
            words_id[i] = np.where(time_words.tolist()[i]> time)[0][-1]
        embedded_words = embed_words(embeddings, words_id, start, end)
        new_chunks[chunk_num] = embedded_words
    return new_chunks

def fullseries_r_splitcontext(langs, model_prefix, n_layers):
    layerwise_dict = {}
    for layer_n in range(n_layers+1):
        print(f"Processing layer {layer_n}")
        fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}", layer_n)) for lang in langs]
        #########################
        m = []; sd = []
        for idx, lang in enumerate(langs):
            chunks = fmri_data[idx]
            y = d[lang_code_dict[lang]]
            weights = load_Ridge_weights(f"chunked_{model_prefix}_{lang}_{layer_n}")
            preds = []
            for chunk_num in range(10):
                X_test = chunks[chunk_num]
                w_test = weights[chunk_num]
                pred = np.dot(X_test, w_test)
                preds.append(pred)
            preds = np.concatenate(preds)
            r = pearsonr(preds, y)[0]
            m.append(r)
        #########################
        sd = [0] * len(m)
        df = pd.DataFrame(zip(langs, m, sd), columns=["lang", "m", "sd"])
        layerwise_dict[layer_n] = df
    save(layerwise_dict, f"results/sanity_check/split_context_monolingual_{model_prefix}")
    
    
xglm_small  = fullseries_r_splitcontext(xglm_langs, "xglm_small", 24)
xglm_med    = fullseries_r_splitcontext(xglm_langs, "xglm_med", 24)
xglm_large  = fullseries_r_splitcontext(xglm_langs, "xglm_large", 48)
xglm_xl     = fullseries_r_splitcontext(xglm_langs, "xglm_xl", 48)
mbert       = fullseries_r_splitcontext(all_codes, "bert_base", 12)
distilmbert = fullseries_r_splitcontext(all_codes, "distilmbert", 6)
xlmr_base   = fullseries_r_splitcontext(all_codes, "xlmr_base", 12)
xlmr_large  = fullseries_r_splitcontext(all_codes, "xlmr_large", 24)
mt5_small   = fullseries_r_splitcontext(all_codes, "mt5_small", 8)
mt5_base    = fullseries_r_splitcontext(all_codes, "mt5_base", 12)
mt5_large   = fullseries_r_splitcontext(all_codes, "mt5_large", 24)

# monolingual, best layer #####################################################
best_monol = [get_best_layerwise(load(model, split_context = True)) for model in model_names]
best_monol_sd = [get_best_layerwise(load(model, split_context = True), give_mean = False) for model in model_names]

best_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol,
    'Family': model_family,
    'sd' : best_monol_sd,
    'n' : n_langs
})

plot_aggregate(best_layer, "",  ylim = .85, ylimstart = -.1)

# monolingual, median layer ###################################################
median_monol = [get_median_layerwise(load(model, split_context = True)) for model in model_names]
median_monol_sd = [get_median_layerwise(load(model), give_mean = False) for model in model_names]

median_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol,
    'Family': model_family,
    'sd' : median_monol_sd,
    'n' : n_langs
})
plot_aggregate(median_layer, "", ylim = .85, ylimstart = -.1)
