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
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

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

def loo_uid_splits(passage, lang_name):
    # if 3 participants, train on 2, test on 1
    # if 2, train on 1, test on 1
    uids = list(froi_expt2.get(passage, {}).get(lang_name, {}).keys())
    if len(uids) < 2:
        return []
    keep_idx = d_corr_keep.get(passage, {}).get(lang_name, None)
    if keep_idx is not None and len(keep_idx) >= 2:
        kept = [uids[i] for i in keep_idx if i < len(uids)]
    else:
        kept = uids[:2]  # safest fallback
    splits = []
    if len(kept) >= 3:
        a, b, c = kept[:3]
        splits.append(([a, b], c))
        splits.append(([a, c], b))
        splits.append(([b, c], a))
    else:
        a, b = kept[:2]
        splits.append(([a], b))
        splits.append(([b], a))
    return splits

within_results_cv = []
for lang in lang_codes:
    passages_keep = kept_passages(lang)
    lang_name = lang_code_dict[lang]
    print("\n", lang_name)
    for modelname, best_layer in dict_bestlayer.items():
        rs = []
        for test_passage in passages_keep:
            X = preproc_align(lang, test_passage, load(f"{test_passage}/{modelname}_{lang}")[best_layer])
            split_scores = []
            for train_uids, test_uid in loo_uid_splits(test_passage, lang_name):
                ys_train = []
                for uid_tr in train_uids:
                    y_tr = froi_expt2[test_passage][lang_name][uid_tr].get("all", None)
                    if y_tr is not None:
                        ys_train.append(np.asarray(y_tr))
                y_te = froi_expt2[test_passage][lang_name][test_uid].get("all", None)
                if len(ys_train) == 0 or y_te is None:
                    continue
                y_train_avg = ys_train[0] if len(ys_train) == 1 else np.mean(np.stack(ys_train, axis=0), axis=0)
                r = test_model_Ridge_crosspart(X, y_train_avg, y_te, n=10)
                split_scores.append(r)
            r_passage = np.nan if len(split_scores) == 0 else float(np.nanmean(split_scores))
            rs.append(r_passage)
        print(f"Processed with {modelname} -- {rs}")
        finite_rs = [x for x in rs if np.isfinite(x)]
        r_mean = np.nan if len(finite_rs) == 0 else float(np.mean(finite_rs))
        r_se   = np.nan if len(finite_rs) == 0 else float(np.std(finite_rs) / np.sqrt(len(finite_rs)))
        within_results_cv.append([
            "experimental", modelname, lang_name, rs, r_mean, r_se
        ])

within_results_cv = pd.DataFrame(
    within_results_cv,
    columns=["condition", "model", "language", "r", "r_mean", "r_se"]
)


_ = within_results_cv.groupby("model").agg({"r_mean":"mean"})
# within_results_cv.to_csv("other/within_results_cv.csv", index=False)
within_results_cv = pd.read_csv("other/within_results_cv.csv")

###################
# RANDOM baseline #
###################

shift_vals = (26, 52, 78, 104)
within_results_cv_random = []
for lang in lang_codes:
    passages_keep = kept_passages(lang)
    lang_name = lang_code_dict[lang]
    print("\n", f"{lang_name} (circular-shift baseline)")
    for modelname, best_layer in dict_bestlayer.items():
        rs = []
        for test_passage in passages_keep:
            X = preproc_align(lang, test_passage, load(f"{test_passage}/{modelname}_{lang}")[best_layer])
            shift_scores = []
            for shift in shift_vals:
                split_scores = []
                for train_uids, test_uid in loo_uid_splits(test_passage, lang_name):
                    ys_train = []
                    for uid_tr in train_uids:
                        y_tr = froi_expt2[test_passage][lang_name][uid_tr].get("all", None)
                        if y_tr is not None:
                            ys_train.append(np.asarray(y_tr))
                    y_te = froi_expt2[test_passage][lang_name][test_uid].get("all", None)
                    if len(ys_train) == 0 or y_te is None:
                        continue
                    y_train_avg = ys_train[0] if len(ys_train) == 1 else np.mean(np.stack(ys_train, axis=0), axis=0)
                    ytr_shift = np.roll(y_train_avg, shift)
                    yte_shift = np.roll(y_te, shift)
                    r_shift = test_model_Ridge_crosspart(X, ytr_shift, yte_shift, n=10)
                    split_scores.append(r_shift)
                if len(split_scores) > 0:
                    shift_scores.append(float(np.nanmean(split_scores)))
            r_passage_shift = np.nan if len(shift_scores) == 0 else float(np.nanmean(shift_scores))
            rs.append(r_passage_shift)
        print(f"Processed with {modelname} -- {rs}")
        finite_rs = [x for x in rs if np.isfinite(x)]
        r_mean = np.nan if len(finite_rs) == 0 else float(np.mean(finite_rs))
        r_se   = np.nan if len(finite_rs) == 0 else float(np.std(finite_rs) / np.sqrt(len(finite_rs)))
        within_results_cv_random.append([
            "experimental", modelname, lang_name, rs, r_mean, r_se
        ])

within_results_cv_random = pd.DataFrame(within_results_cv_random, columns=["condition", "model", "language", "r", "r_mean", "r_se"])

within_all = pd.merge(within_results_cv, within_results_cv_random, suffixes=("", "_random"), on=["condition", "model", "language"])

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

within_all_grouped = within_all.groupby("model").agg(
    Score=("r_mean", "mean"),
    sd=("r_mean", "std"),
    p=("z", combine_z_statistics)
).reset_index()

print(within_all_grouped["p"])

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

################
# load study 1 #
################

study1 = pd.read_csv("../results/mono_multi.csv")
studies_merged = pd.merge(study1[["Model", "Score_mono", "se_mono"]], df_main, on = "Model")
df = studies_merged

r, p = pearsonr(studies_merged['Score_mono'], studies_merged['Score'])

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(studies_merged['Score_mono'], studies_merged['Score'], color=studies_merged['color'], alpha=1, s = 200)
coefficients = np.polyfit(studies_merged['Score_mono'], studies_merged['Score'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(studies_merged['Score_mono'])-.05, max(studies_merged['Score_mono'])+.05, 100)
y_values = polynomial(x_values)
plt.plot(x_values, y_values, ls='--', c='gray')
for i in range(len(studies_merged)):
    plt.errorbar(studies_merged['Score_mono'][i], studies_merged['Score'][i],
                  xerr=studies_merged['se_mono'][i], yerr=studies_merged['se'][i],
                  fmt='o', color=studies_merged['color'][i], zorder = 5)

plt.text(0.03, 0.98, f"r = {round(r, 2)}, p < 0.0001", 
          fontsize=15, ha='left', va='top', alpha=1, 
          bbox=dict(facecolor='white', alpha=0.7), 
          transform=plt.gca().transAxes)

plt.xlabel('Study I WITHIN encoding (R)', fontsize = 17)
plt.ylabel('Study II WITHIN encoding (R)', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.xlim(-90, 45)
#plt.ylim(0.15, 0.6)
#plt.yticks([0.2, 0.3, 0.4, 0.5])
plt.savefig("../plots/study2_study1_corr.svg", format="svg", bbox_inches="tight")
plt.grid(True)
plt.show()

# single langs
sns.set_context("talk")
r_lang = {lang: [] for lang in within_all["language"].unique()}
r_se_lang = {lang: [] for lang in within_all["language"].unique()}

for model in model_names:
    for lang in within_all["language"].unique():
        therow = within_all[(within_all.language == lang) & (within_all.model == model)]
        r = therow["r_mean"].values[0]
        r_se = therow["r_se"].values[0]
        r_lang[lang].append(r)
        r_se_lang[lang].append(r_se)

data = pd.DataFrame(r_lang)
data_se = pd.DataFrame(r_se_lang)
avg_across_models = data.mean(axis=0)
se_across_models  = data.sem(axis=0)
data_with_avg     = pd.concat([pd.DataFrame([avg_across_models], index=['Mean']), data])
data_se_with_avg  = pd.concat([pd.DataFrame([se_across_models],  index=['Mean']), data_se])
fig_height = 26 * 0.9
fig_width  = 6 * 0.9

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
gs = gridspec.GridSpec(len(data_with_avg), 1, height_ratios=[1.5] + [1] * (len(data_with_avg) - 1), hspace=0.5)
axes = [fig.add_subplot(gs[i]) for i in range(len(data_with_avg))]
labels = list(data_with_avg.columns)
x = np.arange(len(labels))  # FIXED numeric positions for all rows
yticks = [0, 0.5]
for i, ax in enumerate(axes):
    color = 'steelblue' if i == 0 else 'indianred'
    ax.errorbar(
        x,
        data_with_avg.iloc[i].values,
        yerr=data_se_with_avg.iloc[i].values,
        fmt='o',
        mfc=color,
        mec='black',
        mew=1,
        markersize=6,
        ecolor='black',
        elinewidth=1.2,
        capsize=3,
        zorder=10
    )
    ax.axhline(0, color='black', lw=2, zorder=0)
    ax.set_ylim(-0.35, 0.5)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(x)
    if i == 0:
        ax.set_ylabel('Average', rotation=0, ha='right', va='center')
        ax.set_xticklabels([])
    else:
        ax.set_ylabel(names_formatted[i - 1], rotation=0, ha='right', va='center')
        ax.set_xticklabels([])
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(labels, rotation=45, ha="right")
plt.suptitle('', y=0.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.savefig("../plots/study2_study1_singlelangs.svg", format="svg", bbox_inches="tight")
plt.show()
