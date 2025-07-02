import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
from researchpy import corr_pair
from math import sqrt
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns
from adjustText import adjust_text
import lang2vec.lang2vec as l2v
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, rankdata

# IMPORTANT NOTE: lang2vec needs to be installed from source (pip has older version)

chdir("/home/dev/Documents/PhD/Alice")

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
    
# all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian', 'Italian']
# all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro", "ita"]

all_langs = ['Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

# xglm_langs = ["ca", "ja", "en", "es", "vi", "ta", "tr", "fr", "ita"]
xglm_langs = ["es", "vi", "ta", "tr", "fr"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
lang_code_d_reversed = {v : k for k, v in lang_code_dict.items()}

###########################################################################
# predict data in ALL LANGUAGES with ridge weights from a single language #
###########################################################################

# xlmr_large best layer is 15 (best in transfer)
# previous coefficients are obtained on the various folds, now need the coefficients obtained with the full data in a single language

def load_embeddings(name, layer):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)[layer]
    return file

# mBERT
# new ordering to make the plot better for genetic clustering (which is added manually afterwards)
# all_langs = ["Farsi", "Marathi", "Catalan", "Spanish", "Italian", "Romanian", "French", "Lithuanian", "Afrikaans", "Dutch", "English", "Norwegian", "Turkish", "Vietnamese", "Tamil", "Japanese"]
all_langs = ["Farsi", "Marathi", "Spanish", "Romanian", "French", "Lithuanian", "Afrikaans", "Dutch", "Norwegian", "Turkish", "Vietnamese", "Tamil"]

all_codes = [lang_code_d_reversed[l] for l in all_langs]
#all_codes = xglm_langs

fmri_data = [preproc_align(lang, load_embeddings(f"xlmr_large_{lang}", 15)) for lang in all_codes]
fmri_response = [d[lang_code_dict[lang]] for lang in all_codes]

out = []
out_p = []
for lang in all_codes:
    # set data
    X = preproc_align(lang, load_embeddings(f"xlmr_large_{lang}", 15))
    y = d[lang_code_dict[lang]]
    # scaling
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X, y)
    weights = reg.coef_#; print(coefs)
    pred = [reg.predict(X_scaler.transform(data)) for data in fmri_data]
    rs = [pearsonr(thepred, y_scaler.transform(theresponse.reshape(-1, 1)).flatten())[0] for thepred, theresponse in zip(pred, fmri_response)]
    ps = [pearsonr(thepred, y_scaler.transform(theresponse.reshape(-1, 1)).flatten())[1] for thepred, theresponse in zip(pred, fmri_response)]
    out.append(rs)
    out_p.append(ps)
corr = pd.DataFrame(out, columns = all_codes)
p_values = pd.DataFrame(out_p, columns = all_codes)
corr.index = all_langs
corr.columns = all_langs

df = corr.copy()
# Create a mask for non-significant correlations
mask_non_sig = p_values > 0.05
#mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask = mask_non_sig #| mask_upper
df[mask.values] = np.nan

plt.figure(figsize=(7, 7), dpi=150)
# sns.heatmap(df, annot=True, fmt=".2f", cmap='jet', center = 0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .82}, linewidths=0.1, annot_kws={"size": 10})
sns.heatmap(df, annot=False, fmt=".2f", cmap='jet', center = 0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .82}, linewidths=0.1)
plt.xticks(fontsize=16, rotation=45, ha="right")
plt.yticks(fontsize=16, rotation = 0, ha="right")
plt.show()

#########################
# interacting with WALS #
#########################

df_stacked = corr.stack()
df_stacked = df_stacked[df_stacked.index.get_level_values(0) != df_stacked.index.get_level_values(1)] # filter out the diagonal


pairs = [(min(lang1, lang2), max(lang1, lang2)) for lang1, lang2 in df_stacked.index] # separate the upper and lower triangular values
unique_pairs = pd.unique(pairs)
# prepare data
data = {
    'Lang1': [],
    'Lang2': [],
    '1to2': [],
    '2to1': []
}

for lang1, lang2 in unique_pairs:
    if (lang1, lang2) in df_stacked.index:
        data['Lang1'].append(lang1)
        data['Lang2'].append(lang2)
        data['1to2'].append(df_stacked[(lang1, lang2)])
        data['2to1'].append(df_stacked[(lang2, lang1)])
    else:  # if missing pairs (though there shouldn't be any)
        data['Lang1'].append(lang1)
        data['Lang2'].append(lang2)
        data['1to2'].append(np.nan)
        data['2to1'].append(np.nan)

comparison_df = pd.DataFrame(data) # 15 x 14 / 2 values
print("correlation = ", pearsonr(comparison_df["1to2"], comparison_df["2to1"])) # statistic=0.6901676684866376, pvalue=1.4489189831199006e-10
comparison_df["transfer"] = comparison_df[["1to2", "2to1"]].mean(axis=1) # there is a high correlation, we can average 1to2 and 2to1

# ISO 693-3 codes
iso = ["fas", "mar", "cat", "spa", "ita", "ron", "fra", "lit", "afr", "nld", "eng", "nob", "tur", "vie", "tam", "jpn"]
iso_codes = {lang : code for lang, code in zip(all_langs, iso)}
comparison_df["iso1"] = comparison_df["Lang1"].map(iso_codes)
comparison_df["iso2"] = comparison_df["Lang2"].map(iso_codes)

# first trying with aggregate measures 
syn, geo, pho, gen, inv, feat = [], [], [], [], [], []
for index, row in tqdm(comparison_df.iterrows(), total = len(comparison_df)):
    syn.append(l2v.syntactic_distance(row["iso1"], row["iso2"]))
    geo.append(l2v.geographic_distance(row["iso1"], row["iso2"]))
    pho.append(l2v.phonological_distance(row["iso1"], row["iso2"]))
    gen.append(l2v.genetic_distance(row["iso1"], row["iso2"]))
    inv.append(l2v.inventory_distance(row["iso1"], row["iso2"]))
    feat.append(l2v.featural_distance(row["iso1"], row["iso2"]))

comparison_df["syn"] = syn
comparison_df["geo"] = geo
comparison_df["pho"] = pho
comparison_df["gen"] = gen
comparison_df["inv"] = inv
comparison_df["feat"] = feat

for colname in ["syn", "geo", "pho", "gen", "inv", "feat"]:
    r, _ = pearsonr(comparison_df["transfer"], comparison_df[colname])
    print(colname, r, _) # nothing is significant here

# comparison_df.to_csv("results/comparison_df.csv", index=False)
comparison_df = pd.read_csv("results/comparison_df.csv")

####################
# compare with MRR #
####################

def load(name):
    with open(f"other/synonyms/embeddings/{name}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def cosine_distance_matrix(vectors1, vectors2):
    v1_norm = vectors1 / np.linalg.norm(vectors1, axis=1)[:, np.newaxis]
    v2_norm = vectors2 / np.linalg.norm(vectors2, axis=1)[:, np.newaxis]
    cos_sim_matrix = np.dot(v1_norm, v2_norm.T)
    return 1-cos_sim_matrix

def mean_reciprocal_rank(matching_ranks):
    return np.mean(1 / matching_ranks)

def evaluate_language_pair(language1, language2, results, layer):
    l1 = np.array([vec[layer] for vec in results[language1]])
    l2 = np.array([vec[layer] for vec in results[language2]])
    cos_sim_matrix = cosine_distance_matrix(l1, l2)
    # rank similarities across rows
    ranks = np.apply_along_axis(rankdata, 1, cos_sim_matrix, method='ordinal')
    matching_ranks = np.diag(ranks)  # diagonal (matching words) ranks
    # get retrieval metrics
    mrr = mean_reciprocal_rank(matching_ranks) # mean reciprocal rank
    return mrr

res = load("xlmr_large")
mrr_res = []
for index, row in tqdm(comparison_df.iterrows(), total = len(comparison_df)):
    l1, l2 = lang_code_d_reversed[row["Lang1"]], lang_code_d_reversed[row["Lang2"]]
    mrr = evaluate_language_pair(l1, l2, res, 15)
    mrr_res.append(mrr)
comparison_df["mrr"] = mrr_res
print(pearsonr(comparison_df["transfer"], comparison_df["mrr"]))

corr_df = comparison_df.corr()
