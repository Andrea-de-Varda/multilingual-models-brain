import pandas as pd
from scipy.stats import pearsonr, norm
import statsmodels.api as sm
from os import chdir
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

chdir("/home/dev/Documents/PhD/Alice")

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}

######################
# load previous data #
######################

ppx_res = pd.read_csv("other/synonyms/perplexity_results.csv")[["lang", "mod", "ppx_residuals"]] # perplexity-encoding	
sem_multi = pd.read_csv('other/synonyms/singlelangs_multilingual_results.csv')[["model", "language", "mrr", "r"]] # sentence translat., multi
sem_mono  = pd.read_csv('other/synonyms/singlelangs_monolingual_results.csv')[["model", "language", "mrr", "r"]] # sentence translat., monol

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

multi = pd.merge(ppx_res, sem_multi, left_on = ["mod", "lang"], right_on = ["model", "language"])
mono  = pd.merge(ppx_res, sem_mono, left_on = ["mod", "lang"], right_on = ["model", "language"])

res_lang_mono = []
for modelname, df in mono.groupby("mod"):
    r_ppx, _ = pearsonr(df["ppx_residuals"], df["r"])
    r_mrr, _ = pearsonr(df["mrr"], df["r"])
    # z stats
    n_langs = len(df)
    z_ppx, _ = r_to_z_single(r_ppx, n_langs)
    z_mrr, _ = r_to_z_single(r_mrr, n_langs)
    res_lang_mono.append([modelname, r_ppx, r_mrr, z_ppx, z_mrr, n_langs])
res_lang_mono = pd.DataFrame(res_lang_mono, columns = ["model", "ppx", "mrr", "z_ppx", "z_mrr", "n_langs"])
print(res_lang_mono)
print(combine_z_statistics(res_lang_mono["z_ppx"]))
print(combine_z_statistics(res_lang_mono["z_mrr"]))

res_lang_multi = []
for modelname, df in multi.groupby("mod"):
    r_ppx, _ = pearsonr(df["ppx_residuals"], df["r"])
    r_mrr, _ = pearsonr(df["mrr"], df["r"])
    # z stats
    n_langs = len(df)
    z_ppx, _ = r_to_z_single(r_ppx, n_langs)
    z_mrr, _ = r_to_z_single(r_mrr, n_langs)
    res_lang_multi.append([modelname, r_ppx, r_mrr, z_ppx, z_mrr, n_langs])
res_lang_multi = pd.DataFrame(res_lang_multi, columns = ["model", "ppx", "mrr", "z_ppx", "z_mrr", "n_langs"])
print(res_lang_multi)
print(combine_z_statistics(res_lang_multi["z_ppx"]))
print(combine_z_statistics(res_lang_multi["z_mrr"]))


mono_agg = mono.groupby("lang").agg({"mrr" : "mean", "r" : "mean", "ppx_residuals" : "mean"})
print(pearsonr(mono_agg["ppx_residuals"], mono_agg["r"]))
print(pearsonr(mono_agg["mrr"], mono_agg["r"]))

multi_agg = multi.groupby("lang").agg({"mrr" : "mean", "r" : "mean", "ppx_residuals" : "mean"})
print(pearsonr(multi_agg["ppx_residuals"], multi_agg["r"]))
print(pearsonr(multi_agg["mrr"], multi_agg["r"]))

##########
# checks #
##########

def agg_res(df):
    df_agg = df.groupby("mod").agg({"mrr" : "mean", "r" : "mean", "ppx_residuals" : "mean"})
    #print(df_agg)
    print(f"\nPPX: {pearsonr(df_agg['ppx_residuals'], df_agg['r'])}")
    print(f"MRR: {pearsonr(df_agg['mrr'], df_agg['r'])}\n")

agg_res(mono)
agg_res(multi)

############################################
# RELIABILITY on both axes (models, langs) #
############################################

def splithalf(data, total=100, verbose = False):
    data = [np.array(rt) for rt in data if len(rt) > 2]
    r = []
    for n in range(total):
        a, b = [], []
        for rt_idx, rt in enumerate(data):
            all_parts = np.arange(len(rt))
            np.random.seed(n+rt_idx) # set seed based on idx
            parts_random = np.random.choice(all_parts, len(all_parts) // 2, replace=False)
            parts_not_random = np.setdiff1d(all_parts, parts_random, assume_unique=True)
            
            part1 = np.mean(rt[parts_random])
            part2 = np.mean(rt[parts_not_random])
            
            a.append(part1)
            b.append(part2)
        
        r_temp, _ = pearsonr(a, b)
        r.append(r_temp)

    # Compute stats for uncorrected R
    mean_r = np.mean(r)
    sd_r = np.std(r, ddof=1)  # ddof=1 for sample standard deviation
    ci = norm.interval(0.95, loc=mean_r, scale=sd_r/np.sqrt(len(r)))  # 95% CI
    if verbose:
        print(f"\nR (uncorrected) = {mean_r}, SD = {sd_r}, CI = {ci}")

def reliab_df(df, group = "mod"):
    df_agg = df.groupby(group).agg({"mrr" : list, "r" : list, "ppx_residuals" : list})
    for col in ["mrr", "ppx_residuals", "r"]:
        print(f"\n\n{col.upper()}")
        data = df_agg[col]
        res = splithalf(data, verbose=True)
    
reliab_df(mono)
reliab_df(mono, group="lang")
reliab_df(multi)
reliab_df(multi, group="lang")

#########################################
# corr encoding mono and encoding multi #
#########################################

all_langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_langs, all_codes)}

df_agg_mono  = mono.groupby("lang").agg({"mrr" : "mean", "r" : "mean", "ppx_residuals" : "mean"})
df_agg_multi = multi.groupby("lang").agg({"mrr" : "mean", "r" : "mean", "ppx_residuals" : "mean"})
agg_merged = pd.merge(df_agg_mono, df_agg_multi, left_index=True, right_index=True, suffixes=('_mono', '_multi'))
print(pearsonr(agg_merged["r_mono"], agg_merged["r_multi"])) # statistic=0.6145174116644082, pvalue=0.033488998685826395


reliab = pd.read_csv("results/correlations/correlation_participants.csv")
reliab.columns = ["lang", "reliab", "p_reliab"]
reliab = reliab[reliab["lang"].isin(all_langs)]
reliab.index = reliab["lang"].map(lang_code_dict)
realiab_merged = pd.merge(agg_merged, reliab, left_index=True, right_index=True)
print(pearsonr(realiab_merged["r_mono"], realiab_merged["reliab"]))
print(pearsonr(realiab_merged["r_multi"], realiab_merged["reliab"]))

model_mono = LinearRegression()
model_mono.fit(realiab_merged[["reliab"]], realiab_merged["r_mono"])
realiab_merged["r_residual_mono"] = realiab_merged["r_mono"] - model_mono.predict(realiab_merged[["reliab"]])
print(pearsonr(realiab_merged["r_residual_mono"], realiab_merged["ppx_residuals_mono"]))
print(pearsonr(realiab_merged["r_residual_mono"], realiab_merged["mrr_mono"]))

model_multi = LinearRegression()
model_multi.fit(realiab_merged[["reliab"]], realiab_merged["r_multi"])
realiab_merged["r_residual_multi"] = realiab_merged["r_multi"] - model_multi.predict(realiab_merged[["reliab"]])
print(pearsonr(realiab_merged["r_residual_multi"], realiab_merged["ppx_residuals_multi"]))
print(pearsonr(realiab_merged["r_residual_multi"], realiab_merged["mrr_multi"]))

res_lang_mono = []
for modelname, df in mono.groupby("mod"):
    # residualize reliability
    df.index = df["lang"]
    df = pd.merge(df, reliab, left_index=True, right_index=True)
    model = LinearRegression()
    model.fit(df[["reliab"]], df["r"])
    df["r_residual"] = df["r"] - model.predict(df[["reliab"]])
    
    r_ppx, _ = pearsonr(df["ppx_residuals"], df["r_residual"])
    r_mrr, _ = pearsonr(df["mrr"], df["r_residual"])
    # z stats
    n_langs = len(df)
    z_ppx, _ = r_to_z_single(r_ppx, n_langs)
    z_mrr, _ = r_to_z_single(r_mrr, n_langs)
    res_lang_mono.append([modelname, r_ppx, r_mrr, z_ppx, z_mrr, n_langs])
res_lang_mono = pd.DataFrame(res_lang_mono, columns = ["model", "ppx", "mrr", "z_ppx", "z_mrr", "n_langs"])
print(res_lang_mono)
print(combine_z_statistics(res_lang_mono["z_ppx"]))
print(combine_z_statistics(res_lang_mono["z_mrr"]))

res_lang_multi = []
for modelname, df in multi.groupby("mod"):
    # residualize reliability
    df.index = df["lang"]
    df = pd.merge(df, reliab, left_index=True, right_index=True)
    model = LinearRegression()
    model.fit(df[["reliab"]], df["r"])
    df["r_residual"] = df["r"] - model.predict(df[["reliab"]])
    
    r_ppx, _ = pearsonr(df["ppx_residuals"], df["r_residual"])
    r_mrr, _ = pearsonr(df["mrr"], df["r_residual"])
    # z stats
    n_langs = len(df)
    z_ppx, _ = r_to_z_single(r_ppx, n_langs)
    z_mrr, _ = r_to_z_single(r_mrr, n_langs)
    res_lang_multi.append([modelname, r_ppx, r_mrr, z_ppx, z_mrr, n_langs])
res_lang_multi = pd.DataFrame(res_lang_multi, columns = ["model", "ppx", "mrr", "z_ppx", "z_mrr", "n_langs"])
print(res_lang_multi)
print(combine_z_statistics(res_lang_multi["z_ppx"]))
print(combine_z_statistics(res_lang_multi["z_mrr"]))

######################################################
# data reliab once partialing out signal reliab info #
######################################################

multi = pd.merge(ppx_res, sem_multi, left_on = ["mod", "lang"], right_on = ["model", "language"])
mono  = pd.merge(ppx_res, sem_mono, left_on = ["mod", "lang"], right_on = ["model", "language"])

mono.index = mono["lang"]
mono = pd.merge(mono, reliab, left_index=True, right_index=True)
model_mono = LinearRegression()
model_mono.fit(mono[["reliab"]], mono["r"])
mono["r_residual"] = mono["r"] - model_mono.predict(mono[["reliab"]])

multi.index = multi["lang"]
multi = pd.merge(multi, reliab, left_index=True, right_index=True)
model_multi = LinearRegression()
model_multi.fit(multi[["reliab"]], multi["r"])
multi["r_residual"] = multi["r"] - model_multi.predict(multi[["reliab"]])

def reliab_df_res(df, group = "mod"):
    df_agg = df.groupby(group).agg({"mrr" : list, "r_residual" : list, "ppx_residuals" : list})
    for col in ["mrr", "ppx_residuals", "r_residual"]:
        print(f"\n\n{col.upper()}")
        data = df_agg[col]
        res = splithalf(data, verbose=True)
        
def reliab_df_res(df, group = "mod", colname = "r_residual"):
    df_agg = df.groupby(group).agg({colname : list})
    data = df_agg[colname]
    res = splithalf(data, verbose=True)

def compute_residuals(group):
    model = LinearRegression(fit_intercept=False)
    X = group[["reliab"]]
    y = group["r"]
    model.fit(X, y)
    group["r_residual"] = y - model.predict(X)
    return group

mono  = mono.groupby("mod", group_keys=False).apply(compute_residuals)
multi = multi.groupby("mod", group_keys=False).apply(compute_residuals)

for _, temp in mono.groupby("mod", group_keys=False):
    print(pearsonr(temp["r"], temp["r_residual"]))
    
pearsonr(mono["r"], mono["r_residual"])

def reliab_df_res(df, group = "mod", colname = "r_residual"):
    df_agg = df.groupby(group).agg({colname : list})
    data = df_agg[colname]
    res = splithalf(data, verbose=True)
    
reliab_df_res(mono, group="mod")
reliab_df_res(mono, group="lang")
reliab_df_res(mono, group="mod", colname = "r")


reliab_df_res(multi, group="mod")
reliab_df_res(multi, group="lang")

# within models
df_agg = df.groupby("lang").agg({colname : list})
splithalf(df_agg["r_residual"], verbose=True)


mono  = mono.groupby("lang", group_keys=False).apply(compute_residuals)
multi = multi.groupby("lang", group_keys=False).apply(compute_residuals)

df_agg_mono_  = mono.groupby("lang").agg({"mrr" : "mean", "r_residual" : "mean", "ppx_residuals" : "mean"})
df_agg_multi_ = multi.groupby("lang").agg({"mrr" : "mean", "r_residual" : "mean", "ppx_residuals" : "mean"})
agg_merged_ = pd.merge(df_agg_mono_, df_agg_multi_, left_index=True, right_index=True, suffixes=('_mono', '_multi'))
print(pearsonr(agg_merged_["r_residual_mono"], agg_merged_["r_residual_multi"])) # statistic=0.541536584167886, pvalue=0.06898854224133832





df = multi
group = "lang"
df_agg = df.groupby(group).agg({"mrr" : list, "r_residual" : list, "ppx_residuals" : list})
for col in ["mrr", "ppx_residuals", "r_residual"]:
    print(f"\n\n{col.upper()}")
    data = df_agg[col]
    res = splithalf(data, verbose=True)

for name, group in mono.groupby("lang", group_keys=False):
    print(group)
    model = LinearRegression()
    X = group[["reliab"]]
    y = group["r"]
    model.fit(X, y)
    group["r_residual"] = y - model.predict(X)
    pearsonr(group["r_residual"], group["r"])
