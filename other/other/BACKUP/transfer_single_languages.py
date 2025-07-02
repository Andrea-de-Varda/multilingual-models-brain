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
from scipy.stats import ttest_ind

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

# mBERT best layer is 6 (best in transfer)
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

fmri_data = [preproc_align(lang, load_embeddings(f"mt5_large_{lang}", 15)) for lang in all_codes]
fmri_response = [d[lang_code_dict[lang]] for lang in all_codes]

out = []
out_p = []
for lang in all_codes:
    # set data
    X = preproc_align(lang, load_embeddings(f"mt5_large_{lang}", 15))
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
print("correlation = ", pearsonr(comparison_df["1to2"], comparison_df["2to1"])) # 0.8806475303421969, 4.329380438968545e-40
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
# We could make a case about this: no effect of syntactic, genetic, phonetic similarity etc, but as
# we know from Kauf, LLM-brain alignment is mainly due by semantics, and languages do not differ mas-
# -sively in the way they encode meaning

# comparison_df.to_csv("results/comparison_df.csv", index=False)
comparison_df = pd.read_csv("results/comparison_df.csv")

################################################
# now considering individual relevant features #
################################################

features = l2v.get_features("eng", "syntax_wals", header=True)

lang_dict_wals = {lang : l2v.get_features(lang, "syntax_wals", header=True)[lang] for lang in tqdm(iso)}

results_wals = []
for n in tqdm(list(range(103))):
    feature_d = {lang : lang_dict_wals[lang][n] for lang in iso}
    comparison_df["feat1"] = comparison_df["iso1"].map(feature_d)
    comparison_df["feat2"] = comparison_df["iso2"].map(feature_d)
    comparison_df["feat_"]  = comparison_df["feat1"] == comparison_df["feat2"]
    t, p = ttest_ind(comparison_df[comparison_df["feat_"] == True]["transfer"], comparison_df[comparison_df["feat_"] == False]["transfer"])
    results_wals.append([features["CODE"][n], t, p])
results_wals = pd.DataFrame(results_wals, columns = ["feat", "t", "p"])
print(results_wals[results_wals["p"] < 0.01])

# indoeuropean vs. non-indoeuropean
is_indo = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
indo    = {lang : ind for lang, ind in zip(all_langs, is_indo)}

comparison_df["indo1"] = comparison_df["Lang1"].map(indo)
comparison_df["indo2"] = comparison_df["Lang1"].map(indo)
comparison_df["indo"]  = comparison_df["indo1"] + comparison_df["indo1"] == 2 # both indo
t, p = ttest_ind(comparison_df[comparison_df["indo"] == False]["transfer"], comparison_df[comparison_df["indo"] == True]["transfer"])
print(f"t = {t}, p = {p}")


###############################################################################

################################
# ANALYSIS OF SINGLE LANGUAGES #
############################

# this presupposes that the script plot_encoding_mono.py has been already run
def load_monol(model_prefix):
    with open(f"results/sanity_check/monolingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_multi(model_prefix):
    with open(f"results/multilingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

monol_results = load_monol("xlmr_large")[16]
monol_results["Language"] = monol_results["lang"].map(lang_code_dict)

transf_dict = {}
for language in all_langs:
    temp = comparison_df[(comparison_df['Lang1'] == language) | (comparison_df['Lang2'] == language)]
    tr = temp["transfer"].mean()
    transf_dict[language] = tr
    
monol_results["transfer"] = monol_results["Language"].map(transf_dict)

plt.scatter(monol_results["transfer"], monol_results["m"])
pearsonr(monol_results["transfer"], monol_results["m"])

#####################
# all languages now #
#####################

def best_layer(res_dict, colname = "r"):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    return idx_max

model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

out_dfs = []
for model in model_names:
    multi_data = load_multi(model)
    layer = best_layer(multi_data)
    print(f"\n\nBest layer for {model} is {layer}")
    
    if "xglm" in model:
        languages = xglm_langs
    else:
        languages = all_codes
    
    fmri_data = [preproc_align(lang, load_embeddings(f"{model}_{lang}", layer)) for lang in languages]
    fmri_response = [d[lang_code_dict[lang]] for lang in languages]
    
    out = []
    out_p = []
    for lang in languages:
        X = preproc_align(lang, load_embeddings(f"{model}_{lang}", layer))
        y = d[lang_code_dict[lang]]
        # scaling
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X = X_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X, y)
        pred = [reg.predict(X_scaler.transform(data)) for data in fmri_data]
        rs = [pearsonr(thepred, y_scaler.transform(theresponse.reshape(-1, 1)).flatten())[0] for thepred, theresponse in zip(pred, fmri_response)]
        ps = [pearsonr(thepred, y_scaler.transform(theresponse.reshape(-1, 1)).flatten())[1] for thepred, theresponse in zip(pred, fmri_response)]
        out.append(rs)
        out_p.append(ps)
    corr = pd.DataFrame(out, columns = languages)
    corr.index = languages
    corr.columns = languages
    
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
    
    comparison_df = pd.DataFrame(data)
    comparison_df["transfer"] = comparison_df[["1to2", "2to1"]].mean(axis=1)
    comparison_df["model"] = model
    out_dfs.append(comparison_df)
    
out_dfs = pd.concat(out_dfs)
# out_dfs.to_csv("results/monol_and_transfer.csv")
out_dfs = pd.read_csv("results/monol_and_transfer.csv")

###################################################
# lang sim and transfer performance across models #
###################################################

out_dfs["Lang1"] = out_dfs["Lang1"].map(lang_code_dict)
out_dfs["Lang2"] = out_dfs["Lang2"].map(lang_code_dict)

lang_sim = pd.read_csv("results/comparison_df.csv")
lang_sim = lang_sim[['Lang1', 'Lang2', 'syn', 'geo', 'pho', 'gen', 'inv', 'feat']]

lang_sim[['Lang1', 'Lang2']] = lang_sim.apply(lambda x: sorted([x['Lang1'], x['Lang2']]), axis=1, result_type='expand')
out_dfs[['Lang1', 'Lang2']] = out_dfs.apply(lambda x: sorted([x['Lang1'], x['Lang2']]), axis=1, result_type='expand')

merged = pd.merge(out_dfs, lang_sim, how="left")

print(corr_pair(merged[['transfer', 'syn', 'geo', 'pho', 'gen', 'inv', 'feat']]))

for model in set(merged["model"]):
    print("\n", model)
    temp = merged[merged["model"] == model]
    for colname in ["syn", "geo", "pho", "gen", "inv", "feat"]:
        r, _ = pearsonr(temp["transfer"], temp[colname])
        print(colname, r, _)
        
# merged.to_csv("results/transfer_and_lin_features.csv")
merged = pd.read_csv("results/transfer_and_lin_features.csv")
