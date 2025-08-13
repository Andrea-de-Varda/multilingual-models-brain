import numpy as np
import numpy.ma as ma
import pandas as pd
import os, pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
import itertools

# ----------------- WARNINGS -----------------
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="numpy.core.numeric is deprecated")
np.seterr(invalid="warn", divide="warn")  # surface where NaNs/Infs are created

# ----------------- PATHS -----------------
CONFIRM_BASE = "/home/dev/Documents/PhD/Alice/confirmatory"
REG_BASE     = "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models"
NORM_BASE    = "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models/normaliz_params"
EMB_DIR      = os.path.join(CONFIRM_BASE, "embeddings")      # embeddings/Passage_1/<model>_<lang>
TRANS_DIR    = os.path.join(CONFIRM_BASE, "transcribed")     # transcribed/Passage_1/it.csv
FMRI_DICT    = os.path.join(CONFIRM_BASE, "data/dict_fMRI")  # pickled dict
os.chdir(CONFIRM_BASE)

# ----------------- LAYER PICKS -----------------
dict_bestlayer = {
    "nllb200_distilled_600M": 9,  "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48
}

# ----------------- HELPERS -----------------
def load_pickle(path):
    with open(path, "rb") as h:
        return pickle.load(h)

def load_new_embeddings(passage, model_key, lang_code):
    # expects: embeddings/{passage}/{model_key}_{lang_code}
    return load_pickle(os.path.join(EMB_DIR, passage, f"{model_key}_{lang_code}"))

# ----------------- DIAGNOSTIC LOGGING -----------------
DEBUG = True
empty_bin_log = []   # each item: dict(model, perturb, lang, passage, n_empty_bins, sample_empty_bins)
allnan_col_log = []  # each item: dict(model, perturb, lang, passage, n_allnan_cols)
inner_loop_log = []  # optional detailed per-passage checks

def _summ_arr(a):
    a = np.asarray(a)
    return dict(shape=tuple(a.shape),
                n_nan=int(np.isnan(a).sum()),
                n_inf=int(np.isinf(a).sum()),
                std=float(np.nanstd(a)) if a.size else np.nan)

# ----------------- ALIGNMENT + EMBEDDING BINNING (INSTRUMENTED) -----------------
def embed_words_instrumented(embeddings, words_id, *, model_key=None, pert=None, lang=None, passage=None):
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

    if DEBUG and len(empty_idx) > 0:
        empty_bin_log.append({
            "model": model_key, "perturb": pert, "lang": lang, "passage": passage,
            "n_empty_bins": len(empty_idx), "sample_empty_bins": empty_idx[:10]
        })

    # column-wise mean imputation (same logic as original imputate_na)
    arr = ma.array(out, mask=np.isnan(out))
    col_mean = arr.mean(axis=0)  # can be NaN if a column is entirely NaN across time
    out_imputed = np.where(np.isnan(out), col_mean, out)

    # log all-NaN columns that could not be imputed
    if DEBUG:
        allnan_cols = np.where(np.all(np.isnan(out), axis=0))[0]
        if allnan_cols.size > 0:
            allnan_col_log.append({
                "model": model_key, "perturb": pert, "lang": lang, "passage": passage,
                "n_allnan_cols": int(allnan_cols.size)
            })

    return out_imputed

def preproc_align(lang_code, passage, embeddings, *, model_key=None, pert=None, lang_label=None):
    df = pd.read_csv(os.path.join(TRANS_DIR, passage, f"{lang_code}.csv"))
    df = df[(df["end"] <= 260) & (df["text"] != " ")]
    time = np.arange(0, 260, 2)
    words_id = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        words_id[i] = np.where(df["end"].iloc[i] > time)[0][-1]
    return embed_words_instrumented(embeddings, words_id, model_key=model_key, pert=pert, lang=lang_label, passage=passage)

# ----------------- METADATA -----------------
d = load_pickle(FMRI_DICT)
passages   = ["Passage_1", "Passage_2", "Passage_3"]
languages  = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_map   = {c: n for c, n in zip(lang_codes, languages)}
keep = {'ar':[1,2,3],'de':[2,3],'hi':[2,3],'it':[1,2,3],'ko':[1],'pt':[1,3],'ru':[1,2,3],'zh':[1,2,3],'pl':[1,2,3]}

# ----------------- LOAD ENCODERS -----------------
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

# ----------------- EVALUATION (INSTRUMENTED) -----------------
rows = []  # per (perturb_type, model_key, language)
for model_key, perturbs in available.items():
    best_layer = dict_bestlayer.get(model_key, None)
    if best_layer is None:
        continue
    for lang in lang_codes:
        kept = keep[lang]
        lang_label = lang_map[lang]
        for pert in tqdm(perturbs, desc=f"{model_key} | {lang_label}", leave=False):
            reg  = load_pickle(os.path.join(REG_BASE,  model_key, pert))
            Xsc, Ysc = load_pickle(os.path.join(NORM_BASE, model_key, pert))
            rs = []
            ok_any = False
            for passage in passages:
                try:
                    emb_dict = load_new_embeddings(passage, model_key, lang)  # dict[layer] -> (T x D)
                except FileNotFoundError:
                    rs.append(np.nan)
                    if DEBUG:
                        inner_loop_log.append(dict(model=model_key, perturb=pert, lang=lang_label,
                                                   passage=passage, issue="missing_embeddings_file"))
                    continue

                if best_layer not in emb_dict:
                    rs.append(np.nan)
                    if DEBUG:
                        inner_loop_log.append(dict(model=model_key, perturb=pert, lang=lang_label,
                                                   passage=passage, issue="missing_best_layer", best_layer=best_layer))
                    continue

                X_raw = preproc_align(lang, passage, emb_dict[best_layer],
                                      model_key=model_key, pert=pert, lang_label=lang_label)

                # scale features
                X = Xsc.transform(X_raw)

                # target
                y_raw = d[passage][lang_label].reshape(-1, 1).flatten()
                y = Ysc.transform(y_raw.reshape(-1, 1)).flatten()

                # align lengths
                n = min(X.shape[0], y.shape[0])
                if X.shape[0] != y.shape[0] and DEBUG:
                    inner_loop_log.append(dict(model=model_key, perturb=pert, lang=lang_label,
                                               passage=passage, issue="length_mismatch",
                                               X_len=int(X.shape[0]), y_len=int(y.shape[0]), used=n))
                X = X[:n]
                y = y[:n]

                # predict
                pred = reg.predict(X)

                # guard pearson inputs
                if (np.nanstd(pred) == 0) or (np.nanstd(y) == 0) or (not np.isfinite(pred).all()) or (not np.isfinite(y).all()):
                    rs.append(np.nan)
                    if DEBUG:
                        inner_loop_log.append(dict(model=model_key, perturb=pert, lang=lang_label,
                                                   passage=passage, issue="pearson_skipped",
                                                   pred=_summ_arr(pred), y=_summ_arr(y)))
                    continue

                r = pearsonr(pred, y)[0]
                rs.append(r)
                ok_any = True

            if not ok_any:
                continue

            r_keep = [rs[i-1] for i in kept if not np.isnan(rs[i-1])]
            if DEBUG and len(r_keep) == 0:
                inner_loop_log.append(dict(model=model_key, perturb=pert, lang=lang_label,
                                           passage="ALL_KEPT_PASSAGES", issue="r_keep_empty",
                                           kept=str(kept), rs=str(rs)))

            rows.append([pert, model_key, lang_label,
                         rs[0] if len(rs)>0 else np.nan,
                         rs[1] if len(rs)>1 else np.nan,
                         rs[2] if len(rs)>2 else np.nan,
                         np.nanmean(r_keep) if len(r_keep)>0 else np.nan,
                         np.nanstd(r_keep)  if len(r_keep)>0 else np.nan])

results = pd.DataFrame(rows, columns=["perturb_type","model","language","r1","r2","r3","r_mean","r_sd"])

# ----------------- DIAGNOSTICS AFTER EVAL -----------------
if DEBUG:
    # per-group coverage
    counts = (results.groupby(["perturb_type","model"])["r_mean"]
              .apply(lambda s: s.notna().sum())
              .reset_index(name="n_non_na"))
    print("\n[diag] Groups with ZERO valid r_mean (n=0):")
    print(counts[counts["n_non_na"] == 0].to_string(index=False))
    print("\n[diag] Groups with only ONE valid r_mean (n=1):")
    print(counts[counts["n_non_na"] == 1].to_string(index=False))

    # missing cells
    missing_cells = results[results["r_mean"].isna()][["perturb_type","model","language"]]
    print("\n[diag] Missing per-language cells (first 40 shown):")
    print(missing_cells.head(40).to_string(index=False))
    print(f"[diag] Total missing cells: {len(missing_cells)}")

    # dump detailed logs
    if empty_bin_log:
        pd.DataFrame(empty_bin_log).to_csv("debug_empty_bins.csv", index=False)
        print("[diag] Wrote debug_empty_bins.csv")
    if allnan_col_log:
        pd.DataFrame(allnan_col_log).to_csv("debug_allnan_cols.csv", index=False)
        print("[diag] Wrote debug_allnan_cols.csv")
    if inner_loop_log:
        pd.DataFrame(inner_loop_log).to_csv("debug_inner_loop.csv", index=False)
        print("[diag] Wrote debug_inner_loop.csv")

# ----------------- AGGREGATION (pinpoint divide-by-zero) -----------------
_old = np.seterr(invalid="raise", divide="raise")
try:
    per_model = (results.groupby(["perturb_type","model"])
                 .agg(r=("r_mean","mean"),
                      SE=("r_mean", lambda x: x.std(ddof=1)/np.sqrt(x.notna().sum()) if x.notna().sum()>1 else np.nan))
                 .reset_index())
except FloatingPointError as e:
    print("\n[diag] FloatingPointError during aggregation:", repr(e))
    # fallback with warnings to continue
    np.seterr(**_old)
    per_model = (results.groupby(["perturb_type","model"])
                 .agg(r=("r_mean","mean"),
                      SE=("r_mean", lambda x: x.std(ddof=1)/np.sqrt(x.notna().sum()) if x.notna().sum()>1 else np.nan))
                 .reset_index())
else:
    np.seterr(**_old)

# ----------------- PLOT (bars = perturbation means, dots = models) -----------------
group_stats = (per_model.groupby("perturb_type")
               .agg(mean_r=("r","mean"),
                    se_r=("SE", lambda s: (pd.Series(s).dropna().std(ddof=1)/np.sqrt(pd.Series(s).dropna().shape[0])
                                           if pd.Series(s).dropna().shape[0]>1 else np.nan)))
               .sort_values("mean_r", ascending=False))

pert_order = list(group_stats.index)
palette = list(plt.cm.Set3.colors) + list(plt.cm.tab20.colors) + list(plt.cm.Pastel1.colors) + list(plt.cm.Accent.colors)
colors = {p: c for p, c in zip(pert_order, itertools.islice(itertools.cycle(palette), len(pert_order)))}

fig, ax = plt.subplots(dpi=400, figsize=(max(4, 0.9*len(pert_order)), 2.6))
yerr = group_stats["se_r"].fillna(0).to_numpy()
for i, pert in enumerate(pert_order):
    ax.bar(i, group_stats.loc[pert, "mean_r"],
           yerr=yerr[i], capsize=5, color=colors[pert],
           edgecolor="black", linewidth=1.2, zorder=2)

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
ax.set_ylim(None, np.nanmax(group_stats["mean_r"]) + 0.1 if len(group_stats) else 0.3)
plt.tight_layout()
plt.show()
