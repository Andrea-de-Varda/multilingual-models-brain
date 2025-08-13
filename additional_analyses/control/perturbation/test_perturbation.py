import numpy as np
import numpy.ma as ma
import pandas as pd
import os, pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="numpy.core.numeric is deprecated"
)
warnings.filterwarnings("ignore", message="Mean of empty slice.")

# ---------- PATHS ----------
CONFIRM_BASE = "/home/dev/Documents/PhD/Alice/confirmatory"
REG_BASE     = "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models"
NORM_BASE    = "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models/normaliz_params"
EMB_DIR      = os.path.join(CONFIRM_BASE, "embeddings")      # embeddings/Passage_1/<model>_<lang>
TRANS_DIR    = os.path.join(CONFIRM_BASE, "transcribed")     # transcribed/Passage_1/it.csv
FMRI_DICT    = os.path.join(CONFIRM_BASE, "data/dict_fMRI")  # pickled dict
os.chdir(CONFIRM_BASE)

# ---------- LAYER PICKS ----------
dict_bestlayer = {
    "nllb200_distilled_600M": 9,  "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48
}

# ---------- HELPERS ----------
def load_pickle(path):
    with open(path, "rb") as h:
        return pickle.load(h)

def load_new_embeddings(passage, model_key, lang_code):
    # expects: embeddings/{passage}/{model_key}_{lang_code}
    return load_pickle(os.path.join(EMB_DIR, passage, f"{model_key}_{lang_code}"))

def imputate_na(arr):
    return np.where(np.isnan(arr), ma.array(arr, mask=np.isnan(arr)).mean(axis=0), arr)

def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)
    out = [np.mean(embeddings[ids == i], axis=0) for i in range(time.shape[0])]
    return imputate_na(np.array(out))

def preproc_align(lang_code, passage, embeddings):
    df = pd.read_csv(os.path.join(TRANS_DIR, passage, f"{lang_code}.csv"))
    df = df[(df["end"] <= 260) & (df["text"] != " ")]
    time = np.arange(0, 260, 2)
    words_id = np.zeros(len(df))
    for i in range(len(df)):
        words_id[i] = np.where(df["end"].iloc[i] > time)[0][-1]
    return embed_words(embeddings, words_id)

# ---------- METADATA ----------
d = load_pickle(FMRI_DICT)
passages   = ["Passage_1", "Passage_2", "Passage_3"]
languages  = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_map   = {c: n for c, n in zip(lang_codes, languages)}
keep = {'ar':[1,2,3],'de':[2,3],'hi':[2,3],'it':[1,2,3],'ko':[1],'pt':[1,3],'ru':[1,2,3],'zh':[1,2,3],'pl':[1,2,3]}

# ---------- LOAD ENCODERS ----------
# directory structure produced by training: registered_models/<model_key>/<perturb_type>
model_keys = [m for m in os.listdir(REG_BASE) if os.path.isdir(os.path.join(REG_BASE, m))]
# collect available perturb types per model (intersection with norms)
available = {}
for mk in model_keys:
    reg_dir  = os.path.join(REG_BASE, mk)
    norm_dir = os.path.join(NORM_BASE, mk)
    if not os.path.isdir(norm_dir): 
        continue
    perts_reg  = {p for p in os.listdir(reg_dir)  if os.path.isfile(os.path.join(reg_dir, p))}
    perts_norm = {p for p in os.listdir(norm_dir) if os.path.isfile(os.path.join(norm_dir, p))}
    perts = sorted(list(perts_reg & perts_norm))
    if perts:
        available[mk] = perts

# ---------- EVALUATION ----------
rows = []  # per (perturb_type, model_key, language)
for model_key, perturbs in available.items():
    best_layer = dict_bestlayer.get(model_key, None)
    if best_layer is None:
        continue
    for lang in lang_codes:
        kept = keep[lang]
        for pert in tqdm(perturbs, desc=f"{model_key} | {lang_map[lang]}", leave=False):
            reg  = load_pickle(os.path.join(REG_BASE,  model_key, pert))
            Xsc, Ysc = load_pickle(os.path.join(NORM_BASE, model_key, pert))
            rs = []
            ok_any = False
            for passage in passages:
                try:
                    emb_dict = load_new_embeddings(passage, model_key, lang)  # dict[layer] -> (T x D)
                except FileNotFoundError:
                    rs.append(np.nan)
                    continue
                X = preproc_align(lang, passage, emb_dict[best_layer])
                X = Xsc.transform(X)
                y = d[passage][lang_map[lang]].reshape(-1, 1)
                y = Ysc.transform(y).flatten()
                y = y[:X.shape[0]]
                pred = reg.predict(X)
                rs.append(pearsonr(pred, y)[0])
                ok_any = True
            if not ok_any:
                continue
            r_keep = [rs[i-1] for i in kept if not np.isnan(rs[i-1])]
            rows.append([pert, model_key, lang_map[lang],
                         rs[0] if len(rs)>0 else np.nan,
                         rs[1] if len(rs)>1 else np.nan,
                         rs[2] if len(rs)>2 else np.nan,
                         np.nanmean(r_keep) if len(r_keep)>0 else np.nan,
                         np.nanstd(r_keep)  if len(r_keep)>0 else np.nan])

results = pd.DataFrame(rows, columns=["perturb_type","model","language","r1","r2","r3","r_mean","r_sd"])
# results.to_csv("perturbation_confirmatory_results_all_models.csv", index=False)

# ---------- AGGREGATION (per perturbation x model over languages) ----------
per_model = (results.groupby(["perturb_type","model"])
             .agg(r=("r_mean","mean"),
                  SE=("r_mean", lambda x: x.std()/np.sqrt(x.notna().sum())))
             .reset_index())

# ---------- PLOT (bars = perturbation means, dots = models) ----------
import matplotlib.pyplot as plt
import itertools
np.random.seed(0)

np.random.seed(0)

# bar stats
group_stats = (per_model.groupby("perturb_type")
               .agg(mean_r=("r","mean"),
                    se_r=("SE","mean"))
               .sort_values("mean_r", ascending=False))

pert_order = list(group_stats.index)

# dynamic color map covering any number of perturbations
palette = list(plt.cm.Set3.colors) + list(plt.cm.tab20.colors) + list(plt.cm.Pastel1.colors) + list(plt.cm.Accent.colors)
colors = {p: c for p, c in zip(pert_order, itertools.islice(itertools.cycle(palette), len(pert_order)))}

fig, ax = plt.subplots(dpi=400, figsize=(max(4, 0.9*len(pert_order)), 2.6))
for i, pert in enumerate(pert_order):
    ax.bar(i,
           group_stats.loc[pert, "mean_r"],
           yerr=group_stats.loc[pert, "se_r"],
           capsize=5,
           color=colors[pert],
           edgecolor="black",
           linewidth=1.2,
           zorder=2)

for i, pert in enumerate(pert_order):
    y_vals = per_model.loc[per_model["perturb_type"] == pert, "r"].to_numpy()
    jitter = np.random.normal(loc=0, scale=0.08, size=len(y_vals))
    x_vals = i + jitter
    ax.scatter(x_vals, y_vals, color="black", alpha=0.45, s=18, zorder=3)

ax.set_ylabel("R", fontsize=12)
ax.set_xticks(range(len(pert_order)))
ax.set_xticklabels(pert_order, fontsize=10, rotation=30, ha="right")
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
ax.set_ylim(None, group_stats["mean_r"].max() + 0.1)
plt.tight_layout()
plt.show()








import itertools
np.random.seed(0)

df_mgpt = results[results["model"] == "mgpt"].copy()

group_stats_mgpt = (df_mgpt.groupby("perturb_type")
                    .agg(mean_r=("r_mean","mean"),
                         se_r=("r_mean", lambda x: x.std()/np.sqrt(x.notna().sum())))
                    .sort_values("mean_r", ascending=False))

pert_order = list(group_stats_mgpt.index)
palette = (list(plt.cm.Set3.colors) + list(plt.cm.tab20.colors) +
           list(plt.cm.Pastel1.colors) + list(plt.cm.Accent.colors))
colors = {p: c for p, c in zip(pert_order, itertools.islice(itertools.cycle(palette), len(pert_order)))}

fig, ax = plt.subplots(dpi=400, figsize=(max(4, 0.9*len(pert_order)), 2.6))
for i, pert in enumerate(pert_order):
    ax.bar(i,
           group_stats_mgpt.loc[pert, "mean_r"],
           yerr=group_stats_mgpt.loc[pert, "se_r"],
           capsize=5,
           color=colors[pert],
           edgecolor="black",
           linewidth=1.2,
           zorder=2)

for i, pert in enumerate(pert_order):
    y_vals = df_mgpt.loc[df_mgpt["perturb_type"] == pert, "r_mean"].to_numpy()  # per-language points
    jitter = np.random.normal(loc=0, scale=0.08, size=len(y_vals))
    x_vals = i + jitter
    ax.scatter(x_vals, y_vals, color="black", alpha=0.45, s=18, zorder=3)

ax.set_title("mGPT", fontsize=12)
ax.set_ylabel("R", fontsize=12)
ax.set_xticks(range(len(pert_order)))
ax.set_xticklabels(pert_order, fontsize=10, rotation=30, ha="right")
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
ax.set_ylim(None, group_stats_mgpt["mean_r"].max() + 0.1)
plt.tight_layout()
plt.show()

