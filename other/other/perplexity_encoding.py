import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr, norm
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import copy
import itertools
import glob
from researchpy import corr_pair

chdir("/home/dev/Documents/PhD/Alice")

ppx = glob.glob("other/perplex/*.pkl")

out = []; out_random = []
for filename in ppx:
    with open(filename, 'rb') as file:
        loaded = pickle.load(file)
    cleaned = re.sub(r"(other\/perplex\/perplexity\_)(.*)(.pkl)", r"\2", filename)
    if "random" in cleaned:
        cl1 = re.sub("\_random", "", cleaned)
        lang = cl1[-2:]
        mod  = cl1[:-3]
        out_random.append([lang, mod, np.mean(loaded), np.std(loaded)])
    else:
        lang = cleaned[-2:]
        mod  = cleaned[:-3]
        out.append([lang, mod, np.mean(loaded), np.std(loaded)])
out = pd.DataFrame(out, columns = ["lang", "mod", "ppx", "sd"])
out_random = pd.DataFrame(out_random, columns = ["lang", "mod", "ppx", "sd"])

out_all = pd.merge(out, out_random, on = ["lang", "mod"], suffixes=(None, '_random'))
out_all["ppx_diff"] = out_all["ppx"] - out_all["ppx_random"]
print(corr_pair(out_all[["ppx", "ppx_random"]])) # highly correlated so need to control for that

out_all.replace({'mod':{'mbert':'bert_base'}, 'lang':{'it':'ita'}}, inplace = True) # different name

xglm_langs = out_all[out_all["mod"] == "xglm_small"]["lang"].tolist() # compare on same languages

# check at the model level
out_filt = out_all[out_all["lang"].isin(xglm_langs)]
out_all.groupby("mod").agg({"ppx" : "mean"}) 

#########################
# load encoding results #
#########################

def load(model_prefix):
    with open(f"results/sanity_check/monolingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_multi(model_prefix):
    with open(f"results/multilingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def get_best_layerwise(res_dict, colname = "m"):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    print(f"Best layer is {idx_max}")
    best = res_dict[idx_max][["lang", colname]]
    return best

model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["XLM-R$_{base}$", "XLM-R$_{large}$", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "DistilmBERT", "mBERT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
model_family = ["XLM-R", "XLM-R", "mT5", "mT5", "mT5", "BERT", "BERT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# get all encoding (monol)
out_enc = []
for model in model_names:
    df = get_best_layerwise(load(model))
    df["mod"] = model
    out_enc.append(df)
out_enc = pd.concat(out_enc)

out_enc_multi = []
for model in model_names:
    df = get_best_layerwise(load_multi(model), colname = "r")
    df["mod"] = model
    out_enc_multi.append(df)
out_enc_multi = pd.concat(out_enc_multi)

# merge, check corr
merged = pd.merge(out_all, out_enc, on=["lang", "mod"])
merged = pd.merge(merged, out_enc_multi, on=["lang", "mod"])

for model in model_names:
    temp = merged[merged["mod"] == model]
    print(model, "-", len(temp))
    print(f"Monolingual = {round(pearsonr(temp['ppx'], temp['m'])[0], 4)}")
    print(f"Multilingual = {round(pearsonr(temp['ppx'], temp['r'])[0], 4)}")
    print(f"Scores = {round(pearsonr(temp['m'], temp['r'])[0], 4)}")

############
# plotting #
############

sns.set_style('whitegrid')
sns.set_context('talk')

colname = "m"
n_rows = 3
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), dpi=300)
axes = axes.flatten()

r_mono_all = []
for idx, model in enumerate(model_names):
    ax = axes[idx]
    temp = merged[merged["mod"] == model]
    r_monolingual_ = pearsonr(temp['ppx'], temp[colname])[0]
    r_monolingual = round(r_monolingual_, 2)
    r_mono_all.append(r_monolingual_)
    sns.regplot(x=temp["ppx"], y=temp[colname], ci=95, scatter_kws={'s': 20, 'color': 'gray'}, line_kws={'color': 'gray'}, ax=ax)
    
    ax.annotate(f"r = {r_monolingual}", (0.5*max(temp["ppx"]), 0.75), fontsize=17, alpha=1, color='black')
    ax.set_ylim(-0.25, 0.9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(names_nice_dict[model])

for ax in axes[len(model_names):]:
    ax.axis('off')

#fig.suptitle('Encoding performance and next-word prediction ability', fontsize=20)
fig.text(0.5, 0, 'Perplexity', ha='center', va='center', fontsize=18)
fig.text(0, 0.5, 'Encoding performance', ha='center', va='center', rotation='vertical', fontsize=18)

plt.tight_layout()
plt.show()

print(np.mean(r_mono_all))

###################
# add reliab info #
#################à#

langs = ['ita', 'ja', 'fr', 'ta', 'ca', 'es', 'tr', 'vi', 'en', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice = ['Italian', 'Japanese', 'French', 'Tamil', 'Catalan', 'Spanish', 'Turkish', 'Vietnamese', 'English', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

lang_code_d_reversed = {k : v for k, v in zip(langs_nice, langs)}

corrs = pd.read_csv("results/correlations/correlation_participants_new.csv")
corrs = corrs[["lang", "r", "p"]]
corrs = corrs[(corrs["r"] > 0) & (corrs["p"] < .05)]
corrs["code"] = corrs.lang.map(lang_code_d_reversed)
corr_dict = {row["code"] : row["r"] for index, row in corrs.iterrows()}
merged["part_corr"] = merged["lang"].map(corr_dict)

merged.to_csv("results/perplexity_results.csv")

# IDEA: redo the linguistic analysis (transfer_single_languages), but this time, with perplexity as covariate (both in L1 and L2)

