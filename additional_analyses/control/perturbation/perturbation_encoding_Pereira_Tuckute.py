import numpy as np
import numpy.ma as ma
import pandas as pd
import os, pickle
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
import itertools
import sys
sys.modules['numpy._core.numeric'] = np.core.numeric

warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="numpy.core.numeric is deprecated")
np.seterr(invalid="warn", divide="warn")

CONFIRM_BASE = "/home/dev/Documents/PhD/Alice/confirmatory"
EMB_DIR      = os.path.join(CONFIRM_BASE, "embeddings")      # embeddings/Passage_1/<model>_<lang>
TRANS_DIR    = os.path.join(CONFIRM_BASE, "transcribed")     # transcribed/Passage_1/it.csv
FMRI_DICT    = os.path.join(CONFIRM_BASE, "data/dict_fMRI")  # pickled dict
os.chdir(CONFIRM_BASE)

DATASETS = {
    "Tuckute2024": {
        "REG_BASE":  "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models",
        "NORM_BASE": "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models/normaliz_params",
    },
    "Pereira2018": {
        "REG_BASE":  "/home/dev/Documents/PhD/Alice/additional_analyses/pereira/perturbation/perturbation/registered_models",
        "NORM_BASE": "/home/dev/Documents/PhD/Alice/additional_analyses/pereira/perturbation/perturbation/registered_models/normaliz_params",
    },
}

dict_bestlayer = {
    "nllb200_distilled_600M": 9,  "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48
}

def load_pickle(path):
    with open(path, "rb") as h:
        return pickle.load(h)

def load_new_embeddings(passage, model_key, lang_code):
    return load_pickle(os.path.join(EMB_DIR, passage, f"{model_key}_{lang_code}"))

def _summ_arr(a):
    a = np.asarray(a)
    return dict(shape=tuple(a.shape),
                n_nan=int(np.isnan(a).sum()),
                n_inf=int(np.isinf(a).sum()),
                std=float(np.nanstd(a)) if a.size else np.nan)

def embed_words_instrumented(embeddings, words_id, *, model_key=None, pert=None, lang=None, passage=None,
                             empty_bin_log=None, allnan_col_log=None):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)  # 130 bins
    nD = embeddings.shape[1]
    out = []
    empty_idx = []

    for i in range(time.shape[0]):
        sel = (ids == i)
        if not np.any(sel):
            empty_idx.append(i)
            out.append(np.full((nD,), np.nan, dtype=float))
        else:
            out.append(np.mean(embeddings[sel], axis=0))

    out = np.array(out, dtype=float)

    if empty_bin_log is not None and len(empty_idx) > 0:
        empty_bin_log.append({
            "model": model_key, "perturb": pert, "lang": lang, "passage": passage,
            "n_empty_bins": len(empty_idx), "sample_empty_bins": empty_idx[:10]
        })

    arr = ma.array(out, mask=np.isnan(out))
    col_mean = arr.mean(axis=0)
    out_imputed = np.where(np.isnan(out), col_mean, out)

    if allnan_col_log is not None:
        allnan_cols = np.where(np.all(np.isnan(out), axis=0))[0]
        if allnan_cols.size > 0:
            allnan_col_log.append({
                "model": model_key, "perturb": pert, "lang": lang, "passage": passage,
                "n_allnan_cols": int(allnan_cols.size)
            })

    return out_imputed

def preproc_align(lang_code, passage, embeddings, *, model_key=None, pert=None, lang_label=None,
                  empty_bin_log=None, allnan_col_log=None):
    df = pd.read_csv(os.path.join(TRANS_DIR, passage, f"{lang_code}.csv"))
    df = df[(df["end"] <= 260) & (df["text"] != " ")]
    time = np.arange(0, 260, 2)
    words_id = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        words_id[i] = np.where(df["end"].iloc[i] > time)[0][-1]
    return embed_words_instrumented(
        embeddings, words_id,
        model_key=model_key, pert=pert, lang=lang_label, passage=passage,
        empty_bin_log=empty_bin_log, allnan_col_log=allnan_col_log
    )

d = load_pickle(FMRI_DICT)
passages   = ["Passage_1", "Passage_2", "Passage_3"]
languages  = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_map   = {c: n for c, n in zip(lang_codes, languages)}
keep = {'ar':[1,2,3],'de':[2,3],'hi':[2,3],'it':[1,2,3],'ko':[1],'pt':[1,3],'ru':[1,2,3],'zh':[1,2,3],'pl':[1,2,3]}

def evaluate_dataset(REG_BASE, NORM_BASE, dataset_label, DEBUG=True):
    empty_bin_log, allnan_col_log, inner_loop_log = [], [], []

    model_keys = [m for m in os.listdir(REG_BASE) if os.path.isdir(os.path.join(REG_BASE, m))]
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

    rows = []
    for model_key, perturbs in available.items():
        best_layer = dict_bestlayer.get(model_key, None)
        if best_layer is None:
            continue
        for lang in lang_codes:
            kept = keep[lang]
            lang_label = lang_map[lang]
            for pert in tqdm(perturbs, desc=f"[{dataset_label}] {model_key} | {lang_label}", leave=False):
                reg  = load_pickle(os.path.join(REG_BASE,  model_key, pert))
                Xsc, Ysc = load_pickle(os.path.join(NORM_BASE, model_key, pert))
                rs = []
                ok_any = False
                for passage in passages:
                    try:
                        emb_dict = load_new_embeddings(passage, model_key, lang)
                    except FileNotFoundError:
                        rs.append(np.nan)
                        if DEBUG:
                            inner_loop_log.append(dict(dataset=dataset_label, model=model_key, perturb=pert, lang=lang_label,
                                                       passage=passage, issue="missing_embeddings_file"))
                        continue

                    if best_layer not in emb_dict:
                        rs.append(np.nan)
                        if DEBUG:
                            inner_loop_log.append(dict(dataset=dataset_label, model=model_key, perturb=pert, lang=lang_label,
                                                       passage=passage, issue="missing_best_layer", best_layer=best_layer))
                        continue

                    X_raw = preproc_align(
                        lang, passage, emb_dict[best_layer],
                        model_key=model_key, pert=pert, lang_label=lang_label,
                        empty_bin_log=empty_bin_log, allnan_col_log=allnan_col_log
                    )

                    X = Xsc.transform(X_raw)
                    y_raw = d[passage][lang_label].reshape(-1, 1).flatten()
                    y = Ysc.transform(y_raw.reshape(-1, 1)).flatten()

                    n = min(X.shape[0], y.shape[0])
                    if X.shape[0] != y.shape[0] and DEBUG:
                        inner_loop_log.append(dict(dataset=dataset_label, model=model_key, perturb=pert, lang=lang_label,
                                                   passage=passage, issue="length_mismatch",
                                                   X_len=int(X.shape[0]), y_len=int(y.shape[0]), used=n))
                    X = X[:n]
                    y = y[:n]

                    pred = reg.predict(X)

                    if (np.nanstd(pred) == 0) or (np.nanstd(y) == 0) or (not np.isfinite(pred).all()) or (not np.isfinite(y).all()):
                        rs.append(np.nan)
                        if DEBUG:
                            inner_loop_log.append(dict(dataset=dataset_label, model=model_key, perturb=pert, lang=lang_label,
                                                       passage=passage, issue="pearson_skipped",
                                                       pred=_summ_arr(pred), y=_summ_arr(y)))
                        continue

                    r = pearsonr(pred, y)[0]
                    rs.append(r)
                    ok_any = True

                if not ok_any:
                    continue

                r_keep = [rs[i-1] for i in kept if not np.isnan(rs[i-1])]
                rows.append([
                    dataset_label, pert, model_key, lang_label,
                    rs[0] if len(rs)>0 else np.nan,
                    rs[1] if len(rs)>1 else np.nan,
                    rs[2] if len(rs)>2 else np.nan,
                    np.nanmean(r_keep) if len(r_keep)>0 else np.nan,
                    np.nanstd(r_keep)  if len(r_keep)>0 else np.nan
                ])

    results = pd.DataFrame(
        rows,
        columns=["dataset","perturb_type","model","language","r1","r2","r3","r_mean","r_sd"]
    )

    _old = np.seterr(invalid="raise", divide="raise")
    try:
        per_model = (results.groupby(["dataset","perturb_type","model"])
                     .agg(r=("r_mean","mean"),
                          SE=("r_mean", lambda x: x.std(ddof=1)/np.sqrt(x.notna().sum()) if x.notna().sum()>1 else np.nan))
                     .reset_index())
    finally:
        np.seterr(**_old)

    return results, per_model

all_results = []
all_per_model = []
for label, cfg in DATASETS.items():
    res, pm = evaluate_dataset(cfg["REG_BASE"], cfg["NORM_BASE"], label, DEBUG=True)
    all_results.append(res)
    all_per_model.append(pm)

results = pd.concat(all_results, ignore_index=True)
per_model = pd.concat(all_per_model, ignore_index=True)

group_stats = (per_model.groupby("perturb_type")
               .agg(mean_r=("r","mean"),
                    se_r=("SE", lambda s: (pd.Series(s).dropna().std(ddof=1)/np.sqrt(pd.Series(s).dropna().shape[0])
                                           if pd.Series(s).dropna().shape[0]>1 else np.nan)))
               .sort_values("mean_r", ascending=False))

pert_order = list(group_stats.index)
palette = list(plt.cm.Set3.colors) + list(plt.cm.tab20.colors) + list(plt.cm.Pastel1.colors) + list(plt.cm.Accent.colors)
colors = {p: c for p, c in zip(pert_order, itertools.islice(itertools.cycle(palette), len(pert_order)))}

fig, ax = plt.subplots(dpi=400, figsize=(max(5, 0.65*len(pert_order)), 2.8))
yerr = group_stats["se_r"].fillna(0).to_numpy()
for i, pert in enumerate(pert_order):
    ax.bar(i, group_stats.loc[pert, "mean_r"],
           yerr=yerr[i], capsize=5, color=colors[pert],
           edgecolor="black", linewidth=1.2, zorder=2)

marker_map = {"Pereira2018": "s", "Tuckute2024": "o"}  # square vs circle
for i, pert in enumerate(pert_order):
    sub = per_model[per_model["perturb_type"] == pert]
    for dataset_label, marker in marker_map.items():
        vals = sub.loc[sub["dataset"] == dataset_label, "r"].to_numpy()
        if len(vals) == 0:
            continue
        jitter = np.random.normal(loc=0, scale=0.08, size=len(vals))
        x_vals = i + jitter
        ax.scatter(x_vals, vals, marker=marker, color="black", alpha=0.25, s=18, zorder=3, label=dataset_label)

handles, labels = ax.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(), title="Dataset", frameon=False, loc="best")
ax.set_ylabel("R", fontsize=12)
ax.set_xticks(range(len(pert_order)))
ax.set_xticklabels(pert_order, fontsize=10, rotation=30, ha="right")
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
ax.set_ylim(None, None)
plt.tight_layout()
plt.show()
