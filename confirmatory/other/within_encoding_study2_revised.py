import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import ast
from os import chdir
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend, get_backend
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, norm, ttest_rel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import sqrt
import warnings
import torch

warnings.filterwarnings("ignore", message="Mean of empty slice.")

chdir("/home/dev/Documents/PhD/Alice/confirmatory")

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BACKEND = "torch_cuda" if DEVICE == "cuda" else "numpy"
set_backend(BACKEND)
print(f"[INFO] Himalaya backend set to {get_backend()} ({DEVICE})", flush=True)

def _to_backend(a):
    if BACKEND.startswith("torch"):
        if isinstance(a, np.ndarray):
            if a.dtype != np.float32:
                a = a.astype(np.float32, copy=False)
            return torch.from_numpy(a).to(DEVICE)
        if torch.is_tensor(a):
            return a.to(dtype=torch.float32, device=DEVICE)
    return a

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

def preproc_align(lang, passage, embeddings):
    df = pd.read_csv(f"transcribed/{passage}/{lang}.csv")
    df = df[df["end"] <= 260]
    df = df[df["text"] != " "]  # keep in sync with Exp1 fix
    time = np.arange(0, 260, 2)
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words.iloc[i] > time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

def test_model_Ridge_crosspart(X, y_train, y_test, n=10):
    kf = KFold(n_splits=n, shuffle=False)
    X_scaler = StandardScaler()
    preds, golds = [], []
    for train_index, test_index in kf.split(X):
        X_train = X_scaler.fit_transform(X[train_index])
        X_test  = X_scaler.transform(X[test_index])
        ytr = y_train.reshape(-1, 1)
        yte = y_test.reshape(-1, 1)
        y_scaler = StandardScaler().fit(ytr[train_index])
        ytr_train = y_scaler.transform(ytr[train_index]).ravel()
        yte_test  = y_scaler.transform(yte[test_index]).ravel()
        reg = KernelRidgeCV(alphas=(1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1_000,10_000))
        reg.fit(_to_backend(X_train), _to_backend(ytr_train[:, None]))
        y_pred = reg.predict(_to_backend(X_test))
        if BACKEND.startswith("torch"):
            y_pred = y_pred.squeeze().detach().cpu().numpy()
        else:
            y_pred = np.asarray(y_pred).squeeze()
        preds.extend(y_pred.tolist())
        golds.extend(yte_test.tolist())
    return pearsonr(preds, golds)[0]

passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_code_dict = {k : v for k, v in zip(lang_codes, languages)}
lang_code_dict_inv = {v : k for k, v in lang_code_dict.items()}

with open("data/dict_fROI", "rb") as handle:
    froi_expt2 = pickle.load(handle)

d_corr_keep = {'Passage_1': {'Hindi': [0, 2],
  'Arabic': [1, 2],
  'Portuguese': [0, 1, 2],
  'Italian': [0, 1, 2],
  'Russian': [0, 2],
  'Mandarin': [0, 1, 2],
  'Korean': [0, 1, 2],
  'Polish': [0, 2],
  'German': [0, 2]},
 'Passage_2': {'Hindi': [0, 1, 2],
  'Arabic': [0, 1],
  'Portuguese': [0, 2],
  'Italian': [1, 2],
  'Russian': [0, 1],
  'Mandarin': [0, 1],
  'Korean': [0, 2],
  'Polish': [0, 1, 2],
  'German': [0, 2]},
 'Passage_3': {'Hindi': [0, 1, 2],
  'Arabic': [0, 1],
  'Portuguese': [1, 2],
  'Italian': [1, 2],
  'Russian': [0, 2],
  'Mandarin': [1, 2],
  'Korean': [0, 2],
  'Polish': [0, 1, 2],
  'German': [0, 2]}}

dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 15,
                  "nllb200_1B" : 17,
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 9,
                  "xlmr_large" : 14,
                  "distilmbert" : 4, 
                  "bert_base" : 6,
                  "mdeberta" : 9, 
                  "mt5_small" : 2, 
                  "mt5_base" : 9,
                  "mt5_large" : 17, 
                  "mgpt" : 14, 
                  "xglm_small" : 10, 
                  "xglm_med" : 16, 
                  "xglm_large" : 40, 
                  "xglm_xl" : 48}

model_names = dict_bestlayer.keys()

#################################
# TEST encoding models "within" #
#################################

# Passages to remove (no sig. correlation in fMRI time-series):
# Korean [2, 3] ; German [1] ; Portuguese [2] ; Hindi [1]
d_passages_keep = {'ar' : [1, 2, 3], 'de' : [2, 3], 'hi' : [2, 3], 'it' : [1, 2, 3],
                   'ko' : [1], 'pt' : [1, 3], 'ru' : [1, 2, 3], 'zh' : [1, 2, 3], 'pl' : [1, 2, 3]}

def kept_passages(lang_code):
    return [f"Passage_{i}" for i in d_passages_keep[lang_code]]

def ordered_uid_pairs(passage, lang_name):
    uids = list(froi_expt2.get(passage, {}).get(lang_name, {}).keys())
    if len(uids) < 2:
        return []
    if d_corr_keep is not None:
        keep_idx = d_corr_keep.get(passage, {}).get(lang_name, [])
        kept = [uids[i] for i in keep_idx if i < len(uids)]
        if len(kept) < 2:
            kept = uids[:2]
    else:
        kept = uids
    pairs = []
    for i in range(len(kept)):
        for j in range(len(kept)):
            if i == j:
                continue
            pairs.append((kept[i], kept[j]))
    return pairs

within_results_cv = []
for lang in lang_codes:
    passages_keep = kept_passages(lang)
    print("\n", lang_code_dict[lang])
    for modelname, best_layer in dict_bestlayer.items():
        rs = []
        for test_passage in passages_keep:
            X = preproc_align(lang, test_passage, load(f"{test_passage}/{modelname}_{lang}")[best_layer])
            pair_scores = []
            uid_pairs = ordered_uid_pairs(test_passage, lang_code_dict[lang])
            for uid_a, uid_b in uid_pairs:
                ya = froi_expt2[test_passage][lang_code_dict[lang]][uid_a].get("all", None)
                yb = froi_expt2[test_passage][lang_code_dict[lang]][uid_b].get("all", None)
                if ya is None or yb is None:
                    continue
                r_ab = test_model_Ridge_crosspart(X, ya, yb, n=10)
                pair_scores.append(r_ab)
            r_passage = np.nan if len(pair_scores) == 0 else float(np.nanmean(pair_scores))
            rs.append(r_passage)
        print(f"Processed with {modelname} -- {rs}")
        finite_rs = [x for x in rs if np.isfinite(x)]
        r_mean = np.nan if len(finite_rs) == 0 else np.mean(finite_rs)
        r_se = np.nan if len(finite_rs) == 0 else np.std(finite_rs) / np.sqrt(len(finite_rs))
        within_results_cv.append([
            "experimental", modelname, lang_code_dict[lang], rs, r_mean, r_se
        ])

within_results_cv = pd.DataFrame(within_results_cv,
                                 columns=["condition", "model", "language", "r", "r_mean", "r_se"])
_ = within_results_cv.groupby("model").agg({"r_mean":"mean"})
# within_results_cv.to_csv("other/within_results_cv.csv", index=False)
within_results_cv = pd.read_csv("other/within_results_cv.csv")


# Baseline: circular-shift random 
shift_vals = (26, 52, 78, 104)

within_results_cv_random = []
for lang in lang_codes:
    passages_keep = kept_passages(lang)
    print("\n", lang_code_dict[lang], "(circular-shift baseline)")
    for modelname, best_layer in dict_bestlayer.items():
        rs = []
        for test_passage in passages_keep:
            X = preproc_align(lang, test_passage, load(f"{test_passage}/{modelname}_{lang}")[best_layer])
            uid_pairs = ordered_uid_pairs(test_passage, lang_code_dict[lang])

            shift_scores = []
            for shift in shift_vals:
                pair_scores = []
                for uid_a, uid_b in uid_pairs:
                    ya = froi_expt2[test_passage][lang_code_dict[lang]][uid_a].get("all", None)
                    yb = froi_expt2[test_passage][lang_code_dict[lang]][uid_b].get("all", None)
                    if ya is None or yb is None:
                        continue
                    ya_shift = np.roll(ya, shift)
                    yb_shift = np.roll(yb, shift)
                    r_ab = test_model_Ridge_crosspart(X, ya_shift, yb_shift, n=10)
                    pair_scores.append(r_ab)
                if len(pair_scores) > 0:
                    shift_scores.append(np.nanmean(pair_scores))
            r_passage_shift = np.nan if len(shift_scores) == 0 else float(np.nanmean(shift_scores))
            rs.append(r_passage_shift)
        print(f"Processed with {modelname} -- {rs}")
        finite_rs = [x for x in rs if np.isfinite(x)]
        r_mean = np.nan if len(finite_rs) == 0 else np.mean(finite_rs)
        r_se = np.nan if len(finite_rs) == 0 else np.std(finite_rs) / np.sqrt(len(finite_rs))
        within_results_cv_random.append([
            "experimental", modelname, lang_code_dict[lang], rs, r_mean, r_se
        ])

within_results_cv_random = pd.DataFrame(within_results_cv_random,
                                        columns=["condition", "model", "language", "r", "r_mean", "r_se"])

within_all = pd.merge(within_results_cv, within_results_cv_random,
                      suffixes=("", "_random"),
                      on=["condition", "model", "language"])

within_all.to_csv("other/confirmatory_within_all.csv", index=False)
# within_all = pd.read_csv("other/confirmatory_within_all.csv")

###############################
# checking stats significance #
###############################

def parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s in {"", "[", "]"}:
            return []
        return ast.literal_eval(s)
    return x

within_all["r"] = within_all["r"].apply(parse_list)
within_all["r_random"] = within_all["r_random"].apply(parse_list)

def r_to_z(r1, r2, n=130):
    z_1 = np.arctanh(r1)
    z_2 = np.arctanh(r2)
    z_diff = z_1 - z_2
    se_diff = np.sqrt(2 * (1 / (n - 3)))
    z_stat = z_diff / se_diff
    p = 2 * (1 - norm.cdf(np.abs(z_stat)))
    return z_stat, p

def combine_z_statistics(z_stats, return_z=False):
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    return (z_combined, combined_pvalue) if return_z else combined_pvalue

compare_z, compare_p = [], []
for scores, scores_random in zip(within_all["r"], within_all["r_random"]):
    zs = []
    for the_score, the_score_random in zip(scores, scores_random):
        z, p = r_to_z(the_score, the_score_random)
        zs.append(z)
    z_combined, p_combined = combine_z_statistics(zs, return_z=True)
    compare_z.append(z_combined)
    compare_p.append(p_combined)

within_all["z"] = compare_z
within_all["p"] = compare_p

# Optionally persist again to match your later line:
# within_all.to_csv("other/confirmatory_within_all.csv", index=False)
# within_all = pd.read_csv("other/confirmatory_within_all.csv")

within_all_grouped = within_all.groupby("model").agg(
    Score=("r_mean", "mean"),
    sd=("r_mean", "std"),
    p=("z", combine_z_statistics)
).reset_index()

print(within_all_grouped["p"].max())  # should mirror your original end-of-block print

########
# plot #
########

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B",
               "xlm_align",
               "infoxlm_base", "infoxlm_large",
               "multiminilm",
               "xlmr_base", "xlmr_large",
               "distilmbert", "bert_base", "mdeberta",
               "mt5_small", "mt5_base", "mt5_large",
               "mgpt",
               "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$",
                   "XLM-Align",
                   "InfoXLM$_{small}$", "InfoXLM$_{large}$",
                   "mMiniLM",
                   "XLM-R$_{base}$", "XLM-R$_{large}$",
                   "DistilmBERT", "mBERT", "mDeBERTa",
                   "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$",
                   "mGPT",
                   "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB",
                "XLM-Align",
                "InfoXLM", "InfoXLM",
                "XLM-R",
                "XLM-R", "XLM-R",
                "BERT", "BERT", "DeBERTa",
                "mT5", "mT5", "mT5",
                "mGPT",
                "XGLM", "XGLM", "XGLM", "XGLM"]

names_nice_dict = {name: nice for name, nice in zip(model_names, names_formatted)}
class_dict      = {name: fam  for name, fam  in zip(model_names, model_family)}

df_main = within_all_grouped.copy()
df_main["model"] = pd.Categorical(df_main["model"], categories=model_names, ordered=True)
df_main = df_main.sort_values("model").reset_index(drop=True)
df_main["Model"]  = df_main["model"].map(names_nice_dict)
df_main["Family"] = df_main["model"].map(class_dict)

df_main["n"]  = 9
df_main["se"] = df_main["sd"] / np.sqrt(df_main["n"])

palette_d = {
    'BERT': "steelblue",
    'DeBERTa': "teal",
    'InfoXLM': "firebrick",
    'NLLB': "tomato",
    'XGLM': "forestgreen",
    'XLM-Align': "firebrick",
    'XLM-R': "lightsteelblue",
    'mGPT': "yellowgreen",
    'mT5': "darkorange"
}
df_main["color"] = df_main["Family"].map(palette_d)

bar_positions = [1,2,3,
                 4.5, 5.5, 6.5,
                 9, 10, 11, 12, 13, 14,
                 15.5, 16.5, 17.5,
                 19, 20, 21, 22, 23]

title = ""
ylimstart = 0
ylim = 0.75

plt.figure(figsize=(24 * .7, 11.5 * .7), dpi=300)
sns.set_context("talk")
ax = plt.gca()
for i, pos in enumerate(bar_positions):
    row = df_main.iloc[i]
    err = row['sd'] / sqrt(row['n'])  # same as df_main["se"][i]
    ax.errorbar(pos, row['Score'], yerr=err, fmt='o', color=row['color'],
                markersize=16, alpha=0.9, lw=3, capsize=5)

ax.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.3)
plt.title(title, fontsize=30, weight='bold', pad=20)
plt.xlabel('Model', fontsize=27, labelpad=20)
plt.ylabel('R', fontsize=27, labelpad=20)
plt.ylim(ylimstart, ylim)
plt.xticks(bar_positions, labels=df_main["Model"], rotation=45, ha='right', fontsize=22)
plt.yticks([.1, .2, .3, .4, .5, .6, .7], fontsize=23)
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("../plots/study2_multi_multitrain.svg", format="svg", bbox_inches="tight")
plt.show()
