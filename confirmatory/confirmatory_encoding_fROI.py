import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr, norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice.")

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

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
    df = df[df["text"] != " "]
    time = np.arange(0, 260, 2)
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words.iloc[i] > time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_code_dict = {k: v for k, v in zip(lang_codes, languages)}
lang_code_dict_inv = {v: k for k, v in lang_code_dict.items()}

d_passages_keep = {'ar':[1,2,3],'de':[2,3],'hi':[2,3],'it':[1,2,3],'ko':[1],'pt':[1,3],'ru':[1,2,3],'zh':[1,2,3],'pl':[1,2,3]}

# NEW LAYERS (revision)
dict_bestlayer = {"nllb200_distilled_600M":9,"nllb200_distilled_1B":15,"nllb200_1B":17,
                  "xlm_align":7,"infoxlm_base":7,"infoxlm_large":14,"multiminilm":9,
                  "xlmr_base":9,"xlmr_large":14,"distilmbert":4,"bert_base":6,"mdeberta":9,
                  "mt5_small":2,"mt5_base":9,"mt5_large":17,"mgpt":14,"xglm_small":10,
                  "xglm_med":16,"xglm_large":40,"xglm_xl":48}
model_names = list(dict_bestlayer.keys())

# fROIs  markers
rois_long = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp']
roi_short = {
    'Lang_LH_AntTemp':'AntTemp',
    'Lang_LH_IFG':'IFG',
    'Lang_LH_IFGorb':'IFGorb',
    'Lang_LH_MFG':'MFG',
    'Lang_LH_PostTemp':'PostTemp',
}
roi_markers = {
    'Lang_LH_AntTemp':'s',
    'Lang_LH_IFG':'o',
    'Lang_LH_IFGorb':'^',
    'Lang_LH_MFG':'v',
    'Lang_LH_PostTemp':'X',
}


# -----------------

with open("data/dict_fROI", "rb") as handle:
    froi_d_expt2 = pickle.load(handle)

def get_resp_roi(lang_code, passage, roi_long):
    lang_name = lang_code_dict[lang_code]
    subj_dict = froi_d_expt2[passage][lang_name]  # {uid: {roi_long: ts, 'all': ts}}
    # average across available subjects (usually 2)
    ts_list = []
    for uid, per_roi in subj_dict.items():
        if roi_long in per_roi:
            ts_list.append(per_roi[roi_long])
    if len(ts_list) == 0:
        return None
    return np.mean(np.stack(ts_list, axis=0), axis=0)


train_names_all = ["main", "pereira", "control", "natstor"]

d_models_froi = {}
for train_name in train_names_all:
    norm_params = {}
    models = {}
    models_random = {}
    for modelname in model_names:
        for roi_long in rois_long:
            rshort = roi_short[roi_long]
            reg_key = f"{modelname}_{rshort}"
            # normalizers
            with open(f"registered_models/{train_name}/normaliz_params/{reg_key}", 'rb') as handle:
                norm_params[reg_key] = pickle.load(handle)
            # main model
            with open(f"registered_models/{train_name}/{reg_key}", 'rb') as handle:
                models[reg_key] = pickle.load(handle)
            # random models
            for random_idx in [0,1,2,3]:
                with open(f"registered_models/{train_name}/{reg_key}_random_{random_idx}", 'rb') as handle:
                    file_random = pickle.load(handle)
                models_random.setdefault(reg_key, {})[random_idx] = file_random
    d_models_froi[train_name] = {"norm": norm_params, "models": models, "random": models_random}

#######################################
# TEST: per-ROI only (no cross-ROI!!) #
#######################################

def r_to_z(r1, r2, n=130):
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(2*(1/(n-3)))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return z, p

def combine_z(zs):
    zc = np.sum(zs) / np.sqrt(len(zs))
    p = 2 * norm.cdf(-abs(zc))
    return zc, p

# evaluate experimental + random per-ROI
transf_results_froi = []
for lang in lang_codes:
    passages_keep = d_passages_keep[lang]
    print("\n", lang_code_dict[lang])
    for train_name in d_models_froi.keys():
        print("\n", train_name)
        packs = d_models_froi[train_name]
        norms = packs["norm"]
        models = packs["models"]

        # EXPERIMENTAL
        for reg_key, reg in models.items():
            modelname, rshort = reg_key.rsplit("_", 1)
            X_list = [
                norms[reg_key][0].transform(
                    preproc_align(lang, passage,
                                  load(f"{passage}/{modelname}_{lang}")[dict_bestlayer[modelname]])
                ) for passage in passages
            ]
            # build Y for matching ROI
            # map short back to long
            roi_long = [k for k,v in roi_short.items() if v == rshort][0]
            y_list = []
            for passage in passages:
                resp = get_resp_roi(lang, passage, roi_long)
                if resp is None:
                    y_list.append(np.full(130, np.nan))  # keep length, will drop later
                else:
                    y_list.append(norms[reg_key][1].transform(resp.reshape(-1,1)).flatten())
            # per-passage r
            rs = []
            for Xp, yp in zip(X_list, y_list):
                mask = ~np.isnan(yp)
                rs.append(pearsonr(reg.predict(Xp)[mask], yp[mask])[0] if mask.sum() > 3 else np.nan)
            transf_results_froi.append([train_name, "experimental", lang_code_dict[lang], modelname, rshort,
                                        rs[0], rs[1], rs[2],
                                        np.nanmean([rs[i-1] for i in passages_keep]),
                                        np.nanstd([rs[i-1] for i in passages_keep])])

        # RANDOM
        models_rand = packs["random"]
        for reg_key, rdict in models_rand.items():
            modelname, rshort = reg_key.rsplit("_", 1)
            X_list = [
                norms[reg_key][0].transform(
                    preproc_align(lang, passage,
                                  load(f"{passage}/{modelname}_{lang}")[dict_bestlayer[modelname]])
                ) for passage in passages
            ]
            roi_long = [k for k,v in roi_short.items() if v == rshort][0]
            y_list = []
            for passage in passages:
                resp = get_resp_roi(lang, passage, roi_long)
                if resp is None:
                    y_list.append(np.full(130, np.nan))
                else:
                    y_list.append(norms[reg_key][1].transform(resp.reshape(-1,1)).flatten())
            for random_idx in [0,1,2,3]:
                rs = []
                for Xp, yp in zip(X_list, y_list):
                    mask = ~np.isnan(yp)
                    pred = rdict[random_idx].predict(Xp)
                    rs.append(pearsonr(pred[mask], yp[mask])[0] if mask.sum() > 3 else np.nan)
                transf_results_froi.append([train_name, f"random_{random_idx}", lang_code_dict[lang], modelname, rshort,
                                            rs[0], rs[1], rs[2],
                                            np.nanmean([rs[i-1] for i in d_passages_keep[lang]]),
                                            np.nanstd([rs[i-1] for i in d_passages_keep[lang]])])

transf_results_froi = pd.DataFrame(transf_results_froi, columns=["train","condition","language","model","roi","r1","r2","r3","r_mean","r_sd"])

# save 
# transf_results_froi.to_csv("confirmatory_results_froi.csv", index=False)
# transf_results_froi = pd.read_csv("confirmatory_results_froi.csv")


# significance per ROI 
results_sig_froi = []
for (language, train, model, roi), df in transf_results_froi.groupby(["language","train","model","roi"]):
    lang_code = lang_code_dict_inv[language]
    keep_passages = d_passages_keep[lang_code]
    exp_row = df[df["condition"]=="experimental"][["r1","r2","r3"]].to_numpy()
    if exp_row.size == 0:
        continue
    rs_exp = [exp_row[0][i-1] for i in keep_passages]
    full_rs_random, full_zs = [], []
    for random_idx in [0,1,2,3]:
        rand_row = df[df["condition"]==f"random_{random_idx}"][["r1","r2","r3"]].to_numpy()
        if rand_row.size == 0:
            continue
        rs_rand = [rand_row[0][i-1] for i in keep_passages]
        full_rs_random.extend(rs_rand)
        zs = []
        for r_e, r_r in zip(rs_exp, rs_rand):
            if np.isnan(r_e) or np.isnan(r_r): 
                continue
            z,_ = r_to_z(r_e, r_r, n=130)
            zs.append(z)
        if len(zs)>0:
            zc,_ = combine_z(zs)
            full_zs.append(zc)
    if len(full_zs)==0:
        continue
    z_full, p_full = combine_z(full_zs)
    results_sig_froi.append([language, train, model, roi, np.nanmean(rs_exp), np.nanmean(full_rs_random), z_full, p_full])

results_sig_froi = pd.DataFrame(results_sig_froi, columns=["lang","training","model","roi","r_mean","r_random","z","p"])

# aggregate across languages per (model, train, roi)
combined_froi = []
for (model, train, roi), df in results_sig_froi.groupby(["model","training","roi"]):
    zc, pc = combine_z(df["z"].values)
    if pc < .001: sig="***"
    elif pc < .01: sig="**"
    elif pc < .05: sig="*"
    else: sig=""
    combined_froi.append([train, model, roi, df["r_mean"].mean(),
                          df["r_random"].mean(),
                          df["r_mean"].std()/np.sqrt(len(df)),
                          zc, pc, sig])
combined_froi = pd.DataFrame(combined_froi,
                             columns=["train","model","roi","r","r_random","SE","z","p","sig"])
# save if desired:
# combined_froi.to_csv("confirmatory_results_froi_aggregated.csv", index=False)

# -----------------
# PLOTTING with different markers per ROI
# -----------------
names_formatted = ["NLLB$_{d-small}$","NLLB$_{d-large}$","NLLB$_{large}$","XLM-Align",
                   "InfoXLM$_{small}$","InfoXLM$_{large}$","mMiniLM","XLM-R$_{base}$",
                   "XLM-R$_{large}$","DistilmBERT","mBERT","mDeBERTa","mT5$_{small}$",
                   "mT5$_{base}$","mT5$_{large}$","mGPT","XGLM$_{small}$","XGLM$_{med}$",
                   "XGLM$_{large}$","XGLM$_{xl}$"]
model_names = ["nllb200_distilled_600M","nllb200_distilled_1B","nllb200_1B","xlm_align",
               "infoxlm_base","infoxlm_large","multiminilm","xlmr_base","xlmr_large",
               "distilmbert","bert_base","mdeberta","mt5_small","mt5_base","mt5_large",
               "mgpt","xglm_small","xglm_med","xglm_large","xglm_xl"]

bar_positions = [1,2,3, 4.5,5.5,6.5, 9,10,11,12,13,14, 15.5,16.5,17.5, 19,20,21,22,23]
name_map = {m:n for m,n in zip(model_names, names_formatted)}

# GROUPED scatter with ROI markers — STORIES (main, natstor)
train_groups = [("main","Study 1","darkslateblue"), ("natstor","NatStories","lightsteelblue")]
plt.figure(figsize=(24*.7, 11.5*.7), dpi=300)
sns.set_context("talk")
ax = plt.gca()
offsets = [-.25, 0, .25, .5, .75]  # one per ROI
for (train_name, title, color) in train_groups:
    df_t = combined_froi[combined_froi["train"]==train_name].copy()
    df_t['model'] = pd.Categorical(df_t['model'], categories=model_names, ordered=True)
    df_t = df_t.sort_values('model')
    for j, roi_long in enumerate(rois_long):
        mark = roi_markers[roi_long]
        sub = df_t[df_t["roi"]==roi_short[roi_long]]
        x_vals = [pos + offsets[j] for pos in bar_positions]
        # align lengths
        sub = sub.set_index("model").reindex(model_names).reset_index()
        for i, row in sub.iterrows():
            y = row['r']; se = row['SE']; sig = row['sig']
            if pd.isna(y): 
                continue
            marker_color = color if sig != "" else "gray"
            alpha = 0.95 if sig != "" else 0.35
            ax.errorbar(
                x_vals[i], y, yerr=se,
                fmt=mark, mfc=marker_color, mec='black', mew=1.2, markersize=11,
                ecolor=marker_color, elinewidth=2.2, capsize=5, alpha=alpha, zorder=4
            )

for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.grid(axis="y", linestyle="--", color="gray", alpha=0.5, linewidth=2.5)
plt.axhline(y=0, color='black', lw=3)
plt.xlabel('Model', fontsize=24, labelpad=16)
plt.ylabel('R', fontsize=24, labelpad=16)
plt.ylim(-.07, .35)
plt.xticks(bar_positions, [name_map[m] for m in model_names], rotation=45, ha='right', fontsize=16)
plt.yticks([0, .1, .2, .3], fontsize=18)
sns.despine()
# legend
handles = [plt.Line2D([0],[0], marker=roi_markers[r], color='w', markerfacecolor='black', markersize=10, markeredgecolor='black', label=roi_short[r]) for r in rois_long]
plt.legend(handles=handles, title="fROI", bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig("../plots/study2_froi_main_natstor.svg", format="svg", bbox_inches="tight")
plt.show()

# GROUPED scatter with ROI markers — SENTENCES (control, pereira)
train_groups = [("control","Tuckute 2024","tab:red"), ("pereira","Pereira 2018","lightsalmon")]
plt.figure(figsize=(24*.7, 11.5*.7), dpi=300)
sns.set_context("talk")
ax = plt.gca()
for (train_name, title, color) in train_groups:
    df_t = combined_froi[combined_froi["train"]==train_name].copy()
    df_t['model'] = pd.Categorical(df_t['model'], categories=model_names, ordered=True)
    df_t = df_t.sort_values('model')
    for j, roi_long in enumerate(rois_long):
        mark = roi_markers[roi_long]
        sub = df_t[df_t["roi"]==roi_short[roi_long]]
        x_vals = [pos + offsets[j] for pos in bar_positions]
        sub = sub.set_index("model").reindex(model_names).reset_index()
        for i, row in sub.iterrows():
            y = row['r']; se = row['SE']; sig = row['sig']
            if pd.isna(y): 
                continue
            marker_color = color if sig != "" else "gray"
            alpha = 0.95 if sig != "" else 0.35
            ax.errorbar(
                x_vals[i], y, yerr=se,
                fmt=mark, mfc=marker_color, mec='black', mew=1.2, markersize=11,
                ecolor=marker_color, elinewidth=2.2, capsize=5, alpha=alpha, zorder=4
            )

for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.grid(axis="y", linestyle="--", color="gray", alpha=0.5, linewidth=2.5)
plt.axhline(y=0, color='black', lw=3)
plt.xlabel('Model', fontsize=24, labelpad=16)
plt.ylabel('R', fontsize=24, labelpad=16)
plt.ylim(-.07, .35)
plt.xticks(bar_positions, [name_map[m] for m in model_names], rotation=45, ha='right', fontsize=16)
plt.yticks([0, .1, .2, .3], fontsize=18)
sns.despine()
handles = [plt.Line2D([0],[0], marker=roi_markers[r], color='w', markerfacecolor='black', markersize=10, markeredgecolor='black', label=roi_short[r]) for r in rois_long]
plt.legend(handles=handles, title="fROI", bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig("../plots/study2_froi_control_pereira.svg", format="svg", bbox_inches="tight")
plt.show()

# quick “best model count” per train, per ROI (optional)
for train_name in ["main","natstor","control","pereira"]:
    print(train_name)
    df_t = combined_froi[combined_froi["train"]==train_name]
    for roi in [roi_short[r] for r in rois_long]:
        sub = df_t[df_t["roi"]==roi]
        idx = sub["r"].argmax()
        row = sub.iloc[idx][["model","r","p","SE"]]
        print(f"  {roi}: {row.to_dict()}")
    print()
