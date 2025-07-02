import numpy as np
import pandas as pd
import re
from os import chdir
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import pearsonr
from researchpy import corr_pair

chdir("/home/dev/Documents/PhD/Alice")

ppx = glob.glob("perplexity/*.pkl")

out = []; out_random = []
for filename in ppx:
    with open(filename, 'rb') as file:
        loaded = pickle.load(file)
    mod = re.sub(r"(perplexity\/PERPLEXITY\_results\_)(.*)(.pkl)", r"\2", filename)
    for lang, v in loaded.items():
        out_random.append([lang, mod, v["random"]])
        out.append([lang, mod, v["result"]])
out = pd.DataFrame(out, columns = ["lang", "mod", "ppx"])
out_random = pd.DataFrame(out_random, columns = ["lang", "mod", "ppx"])

out_all = pd.merge(out, out_random, on = ["lang", "mod"], suffixes=(None, '_random'))
out_all["ppx_diff"] = out_all["ppx"] - out_all["ppx_random"]
out_all.replace({'mod':{'mbert':'bert_base'}, 'lang':{'it':'ita'}}, inplace = True)
print(corr_pair(out_all[["ppx", "ppx_random"]])) # highly correlated so need to control for that

xglm_langs = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

# check at the model level
out_filt = out_all[out_all["lang"].isin(xglm_langs)]
ppx_modelwise = out_all.groupby("mod").agg({"ppx" : "mean"}).sort_values(by="ppx", ascending=False)

#########################
# load encoding results #
#########################

def load(model_prefix):
    with open(f"results/monolingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_multi(model_prefix):
    with open(f"results/multilingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def get_best_layerwise(res_dict, colname = "m"):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    #sd_results   = [value[colname].std() for key, value in res_dict.items()]
    #idx_max = len(mean_results)-1#np.argmax(mean_results)
    idx_max = np.argmax(mean_results)
    print(f"Best layer is {idx_max}")
    best = res_dict[idx_max][["lang", colname]]
    return best

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 5, 5, 5, 5] # n langs by model

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
    #print(model, "-", len(temp))
    print(f"Monolingual = {round(pearsonr(temp['ppx'], temp['m'])[0], 4)}   {round(pearsonr(temp['ppx'], temp['m'])[1], 4)}")
    print(f"Multilingual = {round(pearsonr(temp['ppx'], temp['r'])[0], 4)}")
    #print(f"Scores = {round(pearsonr(temp['m'], temp['r'])[0], 4)}")

grouped_all = merged.groupby("mod").agg({"ppx" : "mean", "m" : "mean", "r" : "mean", "mod" : "max"})
pearsonr(grouped_all["ppx"], grouped_all["r"])
pearsonr(grouped_all["ppx"], grouped_all["m"])

pearsonr(merged["ppx"], merged["m"])
pearsonr(merged["ppx"], merged["r"])

plt.scatter(merged["ppx"], merged["m"])

plt.scatter(grouped_all["ppx"], grouped_all["m"])

len(grouped_all)




############
# plotting #
############

sns.set_style('whitegrid')
sns.set_context('talk')

colname = "m"
n_rows = 5
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12), dpi=300)
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

