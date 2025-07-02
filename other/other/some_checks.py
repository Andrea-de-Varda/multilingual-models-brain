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
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns
from adjustText import adjust_text
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

chdir("/home/dev/Documents/PhD/Alice")


# load fMRI data
with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

def load_Ridge_weights(name):
    with open(f"results/coefficients/{name}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_Ridge_weights_sequential(name):
    with open(f"results/coefficients/sequential_{name}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_embeddings(name, layer):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)[layer]
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

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

# mBERT

all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
lang_code_d_reversed = {v : k for k, v in lang_code_dict.items()}

all_codes = [lang_code_d_reversed[l] for l in all_langs]

fmri_data = [preproc_align(lang, load_embeddings(f"bert_base_{lang}", 6)) for lang in all_codes]
fmri_response = [d[lang_code_dict[lang]] for lang in all_codes]


random = {}
for lang in all_codes:
    weights = load_Ridge_weights(f"bert_base_{lang}_6")
    data = np.array_split(preproc_align(lang, load_embeddings(f"bert_base_{lang}", 6)), 10)   
    pred0 = [np.dot(d, w) for w, d in zip(weights, data)]
    pred = np.concatenate(pred0)
    y = d[lang_code_dict[lang]]
    y_split = np.array_split(y, 10)
    R0 = round(pearsonr(pred, y)[0], 2)
    R1 = round(np.mean([pearsonr(thepred, they)[0] for thepred, they in zip(pred0, y_split)]), 2)
    random[lang] = {"pred" : pred, "r0" : R0, "r1" : R1}
    plt.figure(dpi = 300)
    plt.plot(range(130), y)
    plt.plot(range(130), pred)
    plt.title(f"RANDOM {lang}         {R0}   {R1}")
    plt.show()
    

seq = {}
for lang in all_codes:
    weights = load_Ridge_weights_sequential(f"bert_base_{lang}_6")
    data = np.array_split(preproc_align(lang, load_embeddings(f"bert_base_{lang}", 12)), 10)   
    pred0 = [np.dot(d, w) for w, d in zip(weights, data)]
    pred = np.concatenate(pred0)
    y = d[lang_code_dict[lang]]
    y_split = np.array_split(y, 10)
    R0 = round(pearsonr(pred, y)[0], 2)
    R1 = round(np.mean([pearsonr(thepred, they)[0] for thepred, they in zip(pred0, y_split)]), 2)
    seq[lang] = {"pred" : pred, "r0" : R0, "r1" : R1}
    plt.figure(dpi = 300)
    plt.plot(range(130), y)
    plt.plot(range(130), pred)
    plt.title(f"SEQUENTIAL {lang}         {R0}   {R1}")
    plt.show()

for lang in all_codes:
    r = pearsonr(seq[lang]["pred"], random[lang]["pred"])[0]
    print(round(r, 4))
    
np.mean([seq[lang]["r0"] for lang in all_codes])
np.mean([seq[lang]["r1"] for lang in all_codes])

np.mean([random[lang]["r0"] for lang in all_codes])
np.mean([random[lang]["r1"] for lang in all_codes])



###############################################################################
###############################################################################
###############################################################################

def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_Ridge_weights(name):
    with open(f"results/coefficients/{name}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def fullseries_r(langs, model_prefix, n_layers):
    layerwise_dict = {}
    for n in range(n_layers+1):
        print(f"Processing layer {n}")
        m = []
        for idx, lang in enumerate(langs):
            weights = load_Ridge_weights(f"sequential_{model_prefix}_{lang}_{n}")
            data = np.array_split(preproc_align(lang, load(f"{model_prefix}_{lang}")[n]), 10)  
            pred0 = [np.dot(d, w) for w, d in zip(weights, data)]
            pred = np.concatenate(pred0)
            y = d[lang_code_dict[lang]]
            r = pearsonr(pred, y)[0]
            m.append(r)
        #########################
        sd = [0] * len(m)
        df = pd.DataFrame(zip(langs, m, sd), columns=["lang", "m", "sd"])
        layerwise_dict[n] = df
    save(layerwise_dict, f"results/sanity_check/monolingual_{model_prefix}")
    return layerwise_dict

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




# plot

def load(model_prefix, monol = True, reset_context = False):
    if reset_context:
        if monol:
            with open(f"results/split_context/monolingual_{model_prefix}", 'rb') as handle:
                file = pickle.load(handle)
        else:
            raise ValueError('No multilingual sequential split')
    else:
        if monol:
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

def get_best_layerwise(res_dict, colname = "m", give_mean = True):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    print(f"Best layer is {idx_max}")
    if give_mean:
        best = mean_results[idx_max]
    else:
        best = sd_results[idx_max] # so the SD is the cross-lingual variation for the best layer
    return best

def get_median_layerwise(res_dict, colname = "m", give_mean = True):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_median = find_median_index(mean_results)
    if give_mean:
        median = mean_results[idx_median]
    else:
        median = sd_results[idx_median] # so the SD is the cross-lingual variation for the best layer
    return median

def plot_aggregate(df, title, ylim = None, ylimstart = None):
    plt.figure(figsize=(14, 10), dpi = 300)  
    sns.set_context("talk")
    palette = sns.color_palette("tab10")  # Colorblind-friendly palette
    ax = sns.barplot(x='Model', y='Score', hue='Family', data=df,
                     dodge=False, palette=palette, edgecolor='.2')
    for i in range(len(df['Score'])):
        plt.errorbar(i, df['Score'][i], yerr=df['sd'][i], fmt='none', capsize=5, ecolor='black', capthick=2)
    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=23)
    plt.yticks(fontsize=23)
    leg = plt.legend(title='Model Family', title_fontsize='20', fontsize='18', loc='upper left', bbox_to_anchor=(1, 1))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# Specifying model names

model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["XLM-R$_{base}$", "XLM-R$_{large}$", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "DistilmBERT", "mBERT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
model_family = ["XLM-R", "XLM-R", "mT5", "mT5", "mT5", "BERT", "BERT", "XGLM", "XGLM", "XGLM", "XGLM"]

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}

# Specifying language names

langs = ['ja', 'fr', 'ta', 'ca', 'es', 'tr', 'vi', 'en', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice = ['Japanese', 'French', 'Tamil', 'Catalan', 'Spanish', 'Turkish', 'Vietnamese', 'English', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

lang_dict = {k : v for k, v in zip(langs, langs_nice)}

###########################
# barplot with best layer #
###########################

# monolingual, random split ###################################################
best_monol = [get_best_layerwise(load(model)) for model in model_names]
best_monol_sd = [get_best_layerwise(load(model), give_mean = False) for model in model_names]

best_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol,
    'Family': model_family,
    'sd' : best_monol_sd
})

plot_aggregate(best_layer, "Monolingual encoding",  ylim = .85)
