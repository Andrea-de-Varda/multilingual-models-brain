import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from os import chdir

chdir("/home/dev/Documents/PhD/Alice")

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}

model_languages = [["NLLB$_{d-small}$", 200,  615],
                   ["NLLB$_{d-large}$", 200, 1371],
                   ["NLLB$_{large}$",   200, 1371],
                   ["XLM-Align",         94,  278],
                   ["InfoXLM$_{small}$", 94,  278], 
                   ["InfoXLM$_{large}$", 94,  560],
                   ["mMiniLM",          100,  118],
                   ["XLM-R$_{base}$",   100,  278],
                   ["XLM-R$_{large}$",  100,  560],
                   ["DistilmBERT",      104,  135],
                   ["mBERT",            104,  178],
                   ["mDeBERTa",         100,  278],
                   ["mT5$_{small}$",    101,  172],
                   ["mT5$_{base}$",     101,  390],
                   ["mT5$_{large}$",    101,  973],
                   ["mGPT",              60, 1418],
                   ["XGLM$_{small}$",    30,  564],
                   ["XGLM$_{med}$",      30, 1733],
                   ["XGLM$_{large}$",    30, 2942],
                   ["XGLM$_{xl}$",       30, 4552]]

model_languages = pd.DataFrame(model_languages, columns = ["Model", "n_languages", "n_params"])

encoding = pd.read_csv("results/mono_multi.csv")
df_langs = pd.merge(model_languages, encoding)
df_langs.columns

print("N languages")
print("WITHIN", pearsonr(df_langs["n_languages"], df_langs["Score_mono"]))
print("ACROSS", pearsonr(df_langs["n_languages"], df_langs["Score_multi"]))

print("N params")
print("WITHIN", pearsonr(df_langs["n_params"], df_langs["Score_mono"]))
print("ACROSS", pearsonr(df_langs["n_params"], df_langs["Score_multi"]))

#####################
# LINEAR REGRESSION #
#####################

# load previous data
ppx_res = pd.read_csv("results/perplexity_results.csv")[["Model", "ppx_mean"]] # perplexity-encoding
sem_multi = pd.read_csv('other/synonyms/multilingual_results.csv')[["model", "mrr", "encod"]] # sentence translat., multi
sem_mono  = pd.read_csv('other/synonyms/monolingual_results.csv')[["model", "mrr", "encod"]]  # sentence translat., monol

# merge
sem_multi["Model"] = sem_multi["model"].map(names_nice_dict)
sem_mono["Model"]  = sem_mono["model"].map(names_nice_dict)
sem = pd.merge(sem_mono, sem_multi, on="Model", suffixes=('_mono', '_multi'))
covar = pd.merge(ppx_res, sem)
all_data = pd.merge(df_langs, covar)
corr_df = all_data[["Score_mono", "Score_multi", "n_languages", "n_params", "mrr_multi", "mrr_mono", "ppx_mean"]].corr()

X_within = all_data[["n_languages", "n_params", "mrr_mono", "ppx_mean"]]
y_within = all_data["Score_mono"]
X_within = sm.add_constant(X_within)
model_within = sm.OLS(y_within, X_within).fit()
print("WITHIN-LANGUAGE RESULTS")
print(model_within.summary())

X_across = all_data[["n_languages", "n_params", "mrr_multi", "ppx_mean"]]
y_across = all_data["Score_multi"]
X_across = sm.add_constant(X_across)
model_across = sm.OLS(y_across, X_across).fit()
print("\nACROSS-LANGUAGE RESULTS")
print(model_across.summary())



