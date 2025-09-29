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
from scipy.stats import pearsonr, norm
from researchpy import corr_pair
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

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
    
all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian', 'Italian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro", "ita"]

xglm_langs = ["ca", "ja", "en", "es", "vi", "ta", "tr", "fr", "ita"]

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
def combine_z_statistics(z_stats, return_z = False):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    if return_z:
        return z_combined
    else:
        return combined_pvalue

rom = ["ita", "ca", "ro", "es"] # Romance
ger = ["af", "nl", "en", "no"]  # Germanic
oth = ["tr", "vi", "ta", "ja"]  # Other (all from diff families)

# get embeddings by fam
emb_rom = {lang : preproc_align(lang, load_embeddings(f"xlmr_large_{lang}", 15)) for lang in rom} # layer 16, best in transfer
emb_ger = {lang : preproc_align(lang, load_embeddings(f"xlmr_large_{lang}", 15)) for lang in ger}
emb_oth = {lang : preproc_align(lang, load_embeddings(f"xlmr_large_{lang}", 15)) for lang in oth}

# get response by fam
resp_rom = {lang : d[lang_code_dict[lang]] for lang in rom}
resp_ger = {lang : d[lang_code_dict[lang]] for lang in ger}
resp_oth = {lang : d[lang_code_dict[lang]] for lang in oth}

results = []
for test_idx in range(4):
    rom_, ger_ = rom.copy(), ger.copy()
    rom_test_l, ger_test_l = rom[test_idx], ger[test_idx]
    rom_.remove(rom_test_l), ger_.remove(ger_test_l)
    #print(rom_, ger_)
    print(rom_test_l, ger_test_l)
    
    # training data
    X_rom = np.vstack([emb_rom[lang] for lang in rom_])
    X_ger = np.vstack([emb_ger[lang] for lang in ger_])
    
    # fMRI response
    y_rom = np.hstack([resp_rom[lang] for lang in rom_])
    y_ger = np.hstack([resp_ger[lang] for lang in ger_])
    
    #################
    # fitting (rom) #
    #################
    
    X_scaler_rom = StandardScaler()
    y_scaler_rom = StandardScaler()
    X_rom = X_scaler_rom.fit_transform(X_rom)
    y_rom = y_scaler_rom.fit_transform(y_rom.reshape(-1, 1)).flatten()
    reg_rom = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg_rom.fit(X_rom, y_rom)
    
    # test within
    pred_rom_within = reg_rom.predict(X_scaler_rom.transform(emb_rom[rom_test_l]))
    y_rom_within    = y_scaler_rom.transform(resp_rom[rom_test_l].reshape(-1, 1)).flatten()
    r_rom_within, p_rom_within = pearsonr(pred_rom_within, y_rom_within)
    
    # test across (weak) - all germanic
    rs_across_weak_rom = []
    for l in ger:
        pred_rom_across1 = reg_rom.predict(X_scaler_rom.transform(emb_ger[l]))
        y_rom_across1    = y_scaler_rom.transform(resp_ger[l].reshape(-1, 1)).flatten()
        r_rom_across1, p_rom_across1 = pearsonr(pred_rom_across1, y_rom_across1)
        rs_across_weak_rom.append(r_rom_across1)
    zs_rom = [r_to_z(r_rom_within, r)[0] for r in rs_across_weak_rom]
    z_rom  = combine_z_statistics(zs_rom, return_z = True)
    
    
    #################
    # fitting (ger) #
    #################
    
    X_scaler_ger = StandardScaler()
    y_scaler_ger = StandardScaler()
    X_ger = X_scaler_ger.fit_transform(X_ger)
    y_ger = y_scaler_ger.fit_transform(y_ger.reshape(-1, 1)).flatten()
    reg_ger = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg_ger.fit(X_ger, y_ger)
    
    # test within
    pred_ger_within = reg_ger.predict(X_scaler_ger.transform(emb_ger[ger_test_l]))
    y_ger_within    = y_scaler_ger.transform(resp_ger[ger_test_l].reshape(-1, 1)).flatten()
    r_ger_within, p_ger_within = pearsonr(pred_ger_within, y_ger_within)
    
    # test across (weak)
    rs_across_weak_ger = []
    for l in rom:
        pred_ger_across1 = reg_ger.predict(X_scaler_ger.transform(emb_rom[l]))
        y_ger_across1    = y_scaler_ger.transform(resp_rom[l].reshape(-1, 1)).flatten()
        r_ger_across1, p_ger_across1 = pearsonr(pred_ger_across1, y_ger_across1)
        rs_across_weak_ger.append(r_ger_across1)
    zs_ger = [r_to_z(r_ger_within, r)[0] for r in rs_across_weak_ger]
    z_ger  = combine_z_statistics(zs_ger, return_z = True)
    
    results.append([[rom_test_l, ger_test_l], r_rom_within, r_ger_within, rs_across_weak_rom, rs_across_weak_ger, z_rom, z_ger])


# AGGREGATE

df = []
for res in results:
    #print(res[0], res[1], res[2], np.mean(res[3]), np.mean(res[4]), np.mean(res[5]), np.mean(res[6]))
    df.append(["rom", res[1], np.mean(res[3]), res[5]])
    df.append(["ger", res[2], np.mean(res[4]), res[6]])
    print(f"ROM = {res[1]}, weak = {np.mean(res[3])}")
    print(f"GER = {res[2]}, weak = {np.mean(res[4])}")

df = pd.DataFrame(df, columns=["condition", "within", "across", "z"])

results = df.groupby('condition').agg({
    'within': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))],  # mean and SE
    'across': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))],
    'z': combine_z_statistics  # aggregatr z
})
results.columns = ['within_mean', 'within_se', 'across_mean', 'across_se', 'z_pvalue']


# PLOT #

fig, ax = plt.subplots(figsize=(4, 2.5), dpi = 300)
positions = range(len(results))
bar_width = 0.35

error_config = {'capsize': 5, 'elinewidth': 2, 'markeredgewidth': 2}
bars1 = ax.bar(positions, results['within_mean'], bar_width, yerr=results['within_se'],
               error_kw=error_config, label='Within')
bars2 = ax.bar([p + bar_width for p in positions], results['across_mean'], bar_width, 
               yerr=results['across_se'], error_kw=error_config, label='Across')

y_max = results[['within_mean', 'across_mean']].values.flatten().max() * 1.1

for pos, cond in enumerate(results.index):
    x1, x2 = pos, pos + bar_width
    y, h, col = results.loc[cond, ['within_mean', 'across_mean']].max() + 0.05, 0.02, 'black'
    ax.plot([x1, x1, x2, x2], [y + h, y + h * 2, y + h * 2, y + h], lw=1.5, c=col)

ax.text(bar_width / 2, 0.33, "n.s.",
        ha='center', va='bottom', fontsize=12)

ax.text(1+ bar_width / 2, 0.59, "***",
        ha='center', va='bottom', fontsize=12)

ax.set_xlabel('Condition')
ax.set_ylabel('R')
ax.set_xticks([p + bar_width / 2 for p in positions])
ax.set_xticklabels(["Germanic", "Romance"])
#ax.set_ylim(0, 0.7)
ax.legend()
plt.show()

