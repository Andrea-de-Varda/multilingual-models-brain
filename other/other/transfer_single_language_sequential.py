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

chdir("/home/dev/Documents/PhD/Alice")

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

###############################################################################

# load fMRI data
with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)
    
all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
lang_code_d_reversed = {v : k for k, v in lang_code_dict.items()}

# Lithuanian baltic
# Marathi, Farsi Indo-Iranian
# Vietnamese Austro-asiatic (Vietic)
# Tamil Dravidian

###########################################################################
# predict data in ALL LANGUAGES with ridge weights from a single language #
###########################################################################

# XGLMxl best layer is 16; mBERT best layer is 6 (best in transfer)

def load_Ridge_weights(name):
    with open(f"results/coefficients/sequential_{name}", 'rb') as handle:
        file = pickle.load(handle)
        mean_weights = np.mean(file, axis=0) # averaging weights over folds
    return mean_weights

def load_embeddings(name, layer):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)[layer]
    return file

# mBERT

all_langs = ["Farsi", "Marathi", "Catalan", "Spanish", "Romanian", "French", "Lithuanian", "Afrikaans", "Dutch", "English", "Norwegian", "Turkish", "Vietnamese", "Tamil", "Japanese"]

all_codes = [lang_code_d_reversed[l] for l in all_langs]

fmri_data = [preproc_align(lang, load_embeddings(f"bert_base_{lang}", 6)) for lang in all_codes]
fmri_response = [d[lang_code_dict[lang]] for lang in all_codes]

out = []
out_p = []
for lang in all_codes:
    weights = load_Ridge_weights(f"bert_base_{lang}_6")
    pred = [np.dot(data, weights) for data in fmri_data]
    rs = [pearsonr(thepred, theresponse)[0] for thepred, theresponse in zip(pred, fmri_response)]
    ps = [pearsonr(thepred, theresponse)[1] for thepred, theresponse in zip(pred, fmri_response)]
    out.append(rs)
    out_p.append(ps)
corr = pd.DataFrame(out, columns=all_codes)
p_values = pd.DataFrame(out_p, columns=all_codes)

# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.01
mask = mask_non_sig
corr[mask] = np.nan

corr.index = all_langs
corr.columns = all_langs
df = corr

plt.figure(figsize=(7, 7), dpi=150)
ax = sns.heatmap(df, cmap='jet', center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .82}, linewidths=0.1)

# Manually annotate the heatmap
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        text = df.iloc[i, j]
        if pd.notna(text):
            ax.text(j+0.5, i+0.5, f'{text:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=9)
        else:
            ax.text(j+0.5, i+0.5, '', horizontalalignment='center', verticalalignment='center', fontsize=9)

plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16, rotation=0, ha="right")
plt.show()

##############################################################
# plot transfer performance against avg encoding performance #
##############################################################

# Some monolingual encoding models transfer very well to other languages (e.g., Turkish), and some language get transfered very well

def load_results(name):
    with open(f"results/out_reg/{name}", 'rb') as handle:
        file = pickle.load(handle)
        mean_r = np.mean(file, axis=0) # averaging weights over folds
    return mean_r

monol_res = [load_results(f"bert_base_{lang}_6") for lang in all_codes]

out = []
out_p = []
for lang in all_codes:
    weights = load_Ridge_weights(f"bert_base_{lang}_6")
    pred = [np.dot(data, weights) for data in fmri_data]
    rs = [pearsonr(thepred, theresponse)[0] for thepred, theresponse in zip(pred, fmri_response)]
    ps = [pearsonr(thepred, theresponse)[1] for thepred, theresponse in zip(pred, fmri_response)]
    out.append(rs)
    out_p.append(ps)
corr = pd.DataFrame(out, columns = all_codes)

np.fill_diagonal(corr.values, np.nan)
colmean = corr.mean(axis=0).tolist()
rowmean = corr.mean(axis=1).tolist()


df_langs = pd.DataFrame({"transfer_to" : colmean,
                         "transfer_from" : rowmean,
                         "monol" : monol_res,
                         "lang" : all_langs})

# TRANSFER TO
plt.figure(figsize=(23/3.5, 14/3.5), dpi = 300)
scatter = sns.scatterplot(data=df_langs, 
                          x="monol", 
                          y="transfer_to", 
                          #hue="modelkind", 
                          #style="modelkind", 
                          palette="deep",
                          s=100)
sns.regplot(data=df_langs, 
            x="monol", 
            y="transfer_to", 
            scatter=False,
            color="gray")
texts = []
for line in range(0, df_langs.shape[0]):
    texts.append(scatter.text(df_langs.monol[line], 
                              df_langs.transfer_to[line], 
                              df_langs.lang[line], 
                              horizontalalignment='left', 
                              size='medium', 
                              color='black'))
adjust_text(texts)
plt.xlabel("Monolingual encoding performance", fontsize=12.2)
plt.ylabel("Transfer to L", fontsize=12.2)
#plt.title("Bidirectional models", fontsize=17, weight="bold")
#plt.legend(title="Model Kind", loc="upper left")
#plt.ylim((0, 30))
plt.xlim((-0.2, .75))
plt.tick_params(axis='both', labelsize=12)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Model Kind")
plt.text(0.05, 0.95, f'R = {round(pearsonr(df_langs["monol"], df_langs["transfer_to"])[0], 2)}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.5))
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# TRANSFER FROM
plt.figure(figsize=(23/3.5, 14/3.5), dpi = 300)
scatter = sns.scatterplot(data=df_langs, 
                          x="monol", 
                          y="transfer_from", 
                          #hue="modelkind", 
                          #style="modelkind", 
                          palette="deep",
                          s=100)
sns.regplot(data=df_langs, 
            x="monol", 
            y="transfer_from", 
            scatter=False,
            color="gray")
texts = []
for line in range(0, df_langs.shape[0]):
    texts.append(scatter.text(df_langs.monol[line], 
                              df_langs.transfer_from[line], 
                              df_langs.lang[line], 
                              ha='center', 
                              va="center",
                              size='medium', 
                              color='black'))
adjust_text(texts)
plt.xlabel("Monolingual encoding performance", fontsize=12.2)
plt.ylabel("Transfer from L", fontsize=12.2)
#plt.title("Bidirectional models", fontsize=17, weight="bold")
#plt.legend(title="Model Kind", loc="upper left")
#plt.ylim((0, 30))
plt.xlim((-0.2, .75))
plt.tick_params(axis='both', labelsize=12)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Model Kind")
plt.text(0.05, 0.95, f'R = {round(pearsonr(df_langs["monol"], df_langs["transfer_from"])[0], 2)}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.5))
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

