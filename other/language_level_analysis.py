"""
SI S15 (Language-level analyses): does next-word prediction ability (residual perplexity, from
perplexity/plot_perplexity.py) predict differences in encoding performance across languages?
For each language, encoding performance (best layer of each model, network-level 'all' fROI) and residual
perplexity are averaged across the models that support that language, and the two are correlated (n = 12).
"""
import os
import pickle
from os import chdir

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

if "__file__" in globals():  # run as a script: resolve paths relative to the repository
    chdir(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
# when run interactively, set the working directory to the repository root before running

def patched_load(path):
    import sys
    import numpy
    sys.modules['numpy._core.numeric'] = numpy.core.numeric
    with open(path, 'rb') as f:
        return pickle.load(f)

def load(model_prefix, froi="all", monol=False, multitrain=True):
    if monol:
        filename = f"results/monolingual_{model_prefix}_{froi}"
    else:
        mtpfx = "multitrain_" if multitrain else ""
        filename = f"results/multilingual_{mtpfx}{model_prefix}_{froi}"
    return patched_load(filename)

def get_best_layerwise(res_dict, colname = "m", give_mean = True, give_all = False):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    print(f"Best layer is {idx_max}")
    if give_all:
        best = res_dict[idx_max]
    else:
        if give_mean:
            best = mean_results[idx_max]
        else:
            best = sd_results[idx_max] # so the SD is the cross-lingual variation for the best layer
    return best

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}

######################
# load previous data #
######################

ppx_res = pd.read_csv("perplexity/perplexity_results_by_language.csv")[["lang", "mod", "ppx_residuals"]] # per-language residual perplexity (from perplexity/plot_perplexity.py)

within_res = []  # WITHIN condition (results/monolingual_*)
for model in model_names:
    df = get_best_layerwise(load(model, froi="all", monol=True), colname = "m", give_all = True)[["lang", "m"]]
    df["mod"] = model
    df.columns = ["lang", "r_within", "mod"]
    within_res.append(df)

across_res = []  # ACROSS condition (results/multilingual_multitrain_*)
for model in model_names:
    df = get_best_layerwise(load(model, froi="all", monol=False), colname = "r", give_all = True)
    df["mod"] = model
    df.columns = ["lang", "r_across", "mod"]
    across_res.append(df)

within_df = pd.concat(within_res, ignore_index=True)
across_df = pd.concat(across_res, ignore_index=True)

encoding_res = pd.merge(within_df, across_df, on = ["lang", "mod"])
all_df = pd.merge(encoding_res, ppx_res, on = ["lang", "mod"])

##########################################################
# Language level: average across models, then correlate #
##########################################################

tmp = all_df.drop_duplicates(subset=["lang", "mod"])
avg_lang = (tmp.groupby("lang", as_index=False)
              .agg(r_within_mean=("r_within", "mean"),
                   r_across_mean=("r_across", "mean"),
                   ppx_residuals=("ppx_residuals", "mean"),
                   n_models=("mod", "nunique")))
print(avg_lang)
r, p = pearsonr(avg_lang["ppx_residuals"], avg_lang["r_within_mean"])
print(f"WITHIN: r = {r:.3f}, p = {p:.3f} (n = {len(avg_lang)} languages)")   # SI S15: r = -0.11, p = 0.73
r, p = pearsonr(avg_lang["ppx_residuals"], avg_lang["r_across_mean"])
print(f"ACROSS: r = {r:.3f}, p = {p:.3f} (n = {len(avg_lang)} languages)")   # SI S15: r = -0.07, p = 0.84
