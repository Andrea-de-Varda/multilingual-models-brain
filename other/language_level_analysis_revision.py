import pandas as pd
from scipy.stats import pearsonr, norm
import statsmodels.api as sm
from os import chdir
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

chdir("/home/dev/Documents/PhD/Alice")

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

ppx_res = pd.read_csv("other/synonyms/perplexity_results.csv")[["lang", "mod", "ppx_residuals"]] # perplexity-encoding

multi_res = []
for model in model_names:
    df = get_best_layerwise(load(model, froi="all", monol=True), colname = "m", give_all = True)[["lang", "m"]]
    df["mod"] = model
    df.columns = ["lang", "r_multi", "mod"]
    multi_res.append(df)
    
mono_res = []
for model in model_names:
    df = get_best_layerwise(load(model, froi="all", monol=False), colname = "r", give_all = True)
    df["mod"] = model
    df.columns = ["lang", "r_mono", "mod"]
    mono_res.append(df)

mono_df = pd.concat(mono_res, ignore_index=True)
multi_df = pd.concat(multi_res, ignore_index=True)
    
encoding_res = pd.merge(mono_df, multi_df, on = ["lang", "mod"])
all_df = pd.merge(encoding_res, ppx_res, on = ["lang", "mod"])

########
# TEST #
########

def r_to_z_single(r, n):
    z = np.arctanh(r)  # Fisher transformation
    z_stat = z * np.sqrt(n - 3)  # Multiply by sqrt(n-3) directly
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))  # Two-tailed test
    return z_stat, p_value

# Combine z-statistics using Stouffer's method
def combine_z_statistics(z_stats):
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    return z_combined, combined_pvalue


res = []
for modelname, df in all_df.groupby("mod"):
    r_ppx_mono, _ = pearsonr(df["ppx_residuals"], df["r_mono"])
    r_ppx_multi, _ = pearsonr(df["ppx_residuals"], df["r_multi"])
    # z stats
    n_langs = len(df)
    z_ppx_mono, _ = r_to_z_single(r_ppx_mono, n_langs)
    z_ppx_multi, _ = r_to_z_single(r_ppx_multi, n_langs)
    res.append([modelname, r_ppx_mono, r_ppx_multi, z_ppx_mono, z_ppx_multi, n_langs])
res = pd.DataFrame(res, columns = ["model", "ppx_mono", "ppx_multi", "z_ppx_mono", "z_ppx_multi", "n_langs"])
print(res)

print(combine_z_statistics(res["z_ppx_mono"]))
print(combine_z_statistics(res["z_ppx_multi"]))