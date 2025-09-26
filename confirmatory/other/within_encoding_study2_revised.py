import os
import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, norm, ttest_rel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import sqrt
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice.")

chdir("/home/dev/Documents/PhD/Alice/confirmatory")

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
        ytr_train = StandardScaler().fit_transform(ytr[train_index]).flatten()
        yte_scaler = StandardScaler().fit(yte[train_index])
        yte_test   = yte_scaler.transform(yte[test_index]).flatten()
        reg = RidgeCV(alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, ytr_train)
        y_pred = reg.predict(X_test)
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

# Optionally persist exactly as before:
# within_all.to_csv("other/confirmatory_within_all.csv", index=False)
# within_all = pd.read_csv("other/confirmatory_within_all.csv")

###############################
# checking stats significance #
###############################

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
