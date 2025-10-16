import numpy as np
import numpy.ma as ma
import pandas as pd
import os, pickle
from tqdm import tqdm
from scipy.stats import pearsonr, norm
import matplotlib.pyplot as plt
import warnings
import itertools
import sys
from matplotlib.lines import Line2D
sys.modules['numpy._core.numeric'] = np.core.numeric

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="numpy.core.numeric is deprecated")
np.seterr(invalid="warn", divide="warn")

CONFIRM_BASE = "/home/dev/Documents/PhD/Alice/confirmatory"
EMB_DIR      = os.path.join(CONFIRM_BASE, "embeddings")
TRANS_DIR    = os.path.join(CONFIRM_BASE, "transcribed")
FMRI_DICT    = os.path.join(CONFIRM_BASE, "data/dict_fMRI")
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

def evaluate_dataset(REG_BASE, NORM_BASE, dataset_label, DEBUG=True, skip_keys=None):
    skip_keys = set() if skip_keys is None else set(skip_keys)

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
                # --- NEW: skip combos already present in cached results ---
                if (dataset_label, pert, model_key, lang_label) in skip_keys:
                    continue
                # ----------------------------------------------------------

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


cache_results_path = "perturbation_results.csv"
cache_per_model_path = "perturbation_results_per_model.csv"

if os.path.exists(cache_results_path):
    results = pd.read_csv(cache_results_path)
else:
    results = pd.DataFrame(columns=["dataset","perturb_type","model","language","r1","r2","r3","r_mean","r_sd"])

skip_keys = set(
    results[["dataset","perturb_type","model","language"]]
    .itertuples(index=False, name=None)
)

new_results_list = []
for label, cfg in DATASETS.items():
    res_new, _ = evaluate_dataset(cfg["REG_BASE"], cfg["NORM_BASE"], label, DEBUG=True, skip_keys=skip_keys)
    if len(res_new):
        new_results_list.append(res_new)

if new_results_list:
    results = pd.concat([results] + new_results_list, ignore_index=True)

results.to_csv(cache_results_path, index=False)

_old = np.seterr(invalid="raise", divide="raise")
try:
    per_model = (results.groupby(["dataset","perturb_type","model"])
                 .agg(r=("r_mean","mean"),
                      SE=("r_mean", lambda x: x.std(ddof=1)/np.sqrt(x.notna().sum()) if x.notna().sum()>1 else np.nan))
                 .reset_index())
finally:
    np.seterr(**_old)

per_model.to_csv(cache_per_model_path, index=False)

################

pert_labels = {
    "intact": "Intact",
    "contentwords": "Content words",
    "nounsverbsadj": "N + V + Adj",
    "nounsverbs": "N + V",
    "nouns": "N",
    "verbs": "V",
    "functionwords": "Function words",
    "paraphrase": "Paraphrase",
    "1LocalWordSwap": "1 local swap",
    "2LocalWordSwap": "2 local swaps",
    "3LocalWordSwaps": "3 local swaps",
    "4LocalWordSwaps": "4 local swaps",
    "5LocalWordSwaps": "5 local swaps",
    "Reversed": "Reversed word order",
}


pert_order = [
    'intact',
    'paraphrase',
    'contentwords', 'nounsverbsadj', 'nounsverbs', 'nouns', 'verbs', 'functionwords',
    '1LocalWordSwap', '2LocalWordSwap', '3LocalWordSwaps', '4LocalWordSwaps', '5LocalWordSwaps', 'Reversed'
]

groups = [
    ("Intact",        ['intact']),
    ("Paraphrase",    ['paraphrase']),
    ("Information loss", ['contentwords','nounsverbsadj','nounsverbs','nouns','verbs','functionwords']),
    ("Word order",    ['1LocalWordSwap','2LocalWordSwap','3LocalWordSwaps','4LocalWordSwaps','5LocalWordSwaps','Reversed'])
]

############################
# STATISTICAL SIGNIFICANCE #
############################

def r_to_z(r1, r2, n=130):
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(2.0 / (n - 3))
    z = (z1 - z2) / se
    p_two = 2 * (1 - norm.cdf(abs(z)))
    return z, p_two

def combine_z_statistics(z_stats):
    z_stats = np.asarray(z_stats, dtype=float)
    z_comb = z_stats.sum() / np.sqrt(len(z_stats))
    p_two = 2 * norm.cdf(-abs(z_comb))
    p_lower = norm.cdf(z_comb)
    return z_comb, p_two, p_lower

inv_lang_map = {v: k for k, v in lang_map.items()}

def build_pairs(results_df, level="language", n=130):
    pairs = []
    if level == "language":
        for (dataset, model, language), grp in results_df.groupby(["dataset", "model", "language"]):
            r_int = grp.loc[grp["perturb_type"] == "intact", "r_mean"]
            if r_int.empty or not np.isfinite(r_int.iloc[0]): 
                continue
            r_int = float(r_int.iloc[0])

            for pert, sub in grp.groupby("perturb_type"):
                if pert == "intact":
                    continue
                r_pert = sub["r_mean"]
                if r_pert.empty or not np.isfinite(r_pert.iloc[0]):
                    continue
                r_pert = float(r_pert.iloc[0])
                z, p = r_to_z(r_pert, r_int, n=n)
                pairs.append({
                    "dataset": dataset, "model": model, "language": language,
                    "perturb_type": pert, "level": "language", "z": z, "p_two": p
                })

    elif level == "passage":
        long = results_df.melt(
            id_vars=["dataset","perturb_type","model","language"],
            value_vars=["r1","r2","r3"],
            var_name="passage", value_name="r"
        ).dropna(subset=["r"])
        long["passage_idx"] = long["passage"].str.extract(r"r(\d)").astype(int)
        def keep_passage(row):
            code = inv_lang_map.get(row["language"], None)
            if code is None: 
                return True
            allowed = set(keep.get(code, []))
            return row["passage_idx"] in allowed if allowed else True
        long = long[long.apply(keep_passage, axis=1)]
        for keys, grp in long.groupby(["dataset","model","language","passage_idx"]):
            dataset, model, language, passage_idx = keys
            r_int = grp.loc[grp["perturb_type"] == "intact", "r"]
            if r_int.empty or not np.isfinite(r_int.iloc[0]):
                continue
            r_int = float(r_int.iloc[0])
            for pert, sub in grp.groupby("perturb_type"):
                if pert == "intact":
                    continue
                r_pert = sub["r"]
                if r_pert.empty or not np.isfinite(r_pert.iloc[0]):
                    continue
                r_pert = float(r_pert.iloc[0])
                z, p = r_to_z(r_pert, r_int, n=n)
                pairs.append({
                    "dataset": dataset, "model": model, "language": language,
                    "passage": passage_idx, "perturb_type": pert,
                    "level": "passage", "z": z, "p_two": p
                })
    else:
        raise ValueError("level must be 'language' or 'passage'")
    return pd.DataFrame(pairs)

def stouffer_summary(pairs_df):
    rows = []
    for pert, sub in pairs_df.groupby("perturb_type"):
        zc, p2, pl = combine_z_statistics(sub["z"].values)
        rows.append({"perturb_type": pert, "N": len(sub), "z_comb": zc, "p_two": p2, "p_lower": pl})
    overall = pd.DataFrame(rows).sort_values("p_two")
    m_overall = overall.shape[0]
    overall["p_two_bonf"] = np.minimum(1.0, overall["p_two"] * m_overall)
    rows = []
    for (pert, dataset), sub in pairs_df.groupby(["perturb_type","dataset"]):
        zc, p2, pl = combine_z_statistics(sub["z"].values)
        rows.append({"perturb_type": pert, "dataset": dataset, "N": len(sub), "z_comb": zc, "p_two": p2, "p_lower": pl})
    by_dataset = pd.DataFrame(rows)
    by_dataset = by_dataset.sort_values(["dataset","p_two"]).reset_index(drop=True)
    by_dataset["p_two_bonf"] = by_dataset.groupby("dataset")["p_two"].transform(
        lambda s: np.minimum(1.0, s * s.size)
    )
    def bh(p):
        p = np.asarray(p, float)
        m = len(p)
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty_like(q)
        out[order] = q
        return out
    overall = overall.sort_values("p_two")
    overall["q_lower_BH"] = bh(overall["p_lower"].values)
    by_dataset = by_dataset.sort_values(["perturb_type","p_lower"])
    by_dataset["q_lower_BH"] = by_dataset.groupby("dataset")["p_lower"].transform(bh)
    return overall, by_dataset

def summarize_by_group(pairs_df):
    df = pairs_df.copy()
    df["group"] = df["perturb_type"].map(pert_to_group)
    rows = []
    for g, sub in df.groupby("group"):
        zc, p2, pl = combine_z_statistics(sub["z"].values)
        rows.append({"group": g, "N": len(sub), "z_comb": zc, "p_two": p2, "p_lower": pl})
    out = pd.DataFrame(rows).sort_values("p_two").reset_index(drop=True)
    m_groups = out.shape[0]
    out["p_two_bonf"] = np.minimum(1.0, out["p_two"] * m_groups)
    return out

pairs_lang = build_pairs(results, level="language", n=130)
overall_lang, by_dataset_lang = stouffer_summary(pairs_lang)
print(overall_lang)

pert_to_group = {}
for gname, items in groups:
    for p in items:
        pert_to_group[p] = gname

def summarize_by_group(pairs_df):
    df = pairs_df.copy()
    df["group"] = df["perturb_type"].map(pert_to_group)
    rows = []
    for g, sub in df.groupby("group"):
        zc, p2, pl = combine_z_statistics(sub["z"].values)
        rows.append({"group": g, "N": len(sub), "z_comb": zc, "p_two": p2, "p_lower": pl})
    out = pd.DataFrame(rows).sort_values("p_lower")
    return out

by_group_lang = summarize_by_group(pairs_lang)
print(by_group_lang)

# alpha_sig = 0.05
# sig_map = {
#     row.perturb_type: (np.isfinite(row.p_two_bonf) and (row.p_two_bonf < alpha_sig))
#     for _, row in overall_lang.iterrows()
# }

###############################################################################

group_color = {
    "Intact": "tab:blue",
    "Information loss": "tab:red",
    "Paraphrase": "tab:green",
    "Word order": "tab:purple",
}

alpha_schedules = {
    "Intact":        [1.0],
    "Information loss": np.linspace(1.0, 0.6, num=len([*groups[1][1]])).tolist(),
    "Paraphrase":    [1.0],
    "Word order":    np.linspace(1.0, 0.5, num=len([*groups[3][1]])).tolist(),
}

bar_means = {}
bar_se    = {}
for pert in pert_order:
    vals = per_model.loc[per_model["perturb_type"] == pert, "r"].to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        bar_means[pert] = np.nan
        bar_se[pert]    = np.nan
    else:
        bar_means[pert] = np.mean(vals)
        bar_se[pert]    = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan

gap = 0.8  # horizontal space between groups
x_positions = {}
x = 0.0
group_boundaries = []
for g_name, g_items in groups:
    for item in g_items:
        x_positions[item] = x
        x += 1.0
    group_boundaries.append(x - 0.5)  # boundary after this group
    x += gap

total_width = x - gap


fig, ax = plt.subplots(dpi=400, figsize=(max(8, 0.5*len(pert_order) + 2.5), 3))
for g_name, g_items in groups:
    base_c = group_color[g_name]
    alphas = alpha_schedules[g_name]
    for i, pert in enumerate(g_items):
        xpos = x_positions[pert]
        mean = bar_means.get(pert, np.nan)
        se   = bar_se.get(pert, np.nan)
        ax.bar(xpos, mean, yerr=0 if not np.isfinite(se) else se, capsize=5,
               color=base_c, alpha=alphas[i] if i < len(alphas) else 1.0,
               edgecolor="black", linewidth=1.1, zorder=2)
marker_map = {"Pereira2018": ("s", 0.15, "black"), "Tuckute2024": ("o", 0.15, "black")}
legend_handles = {}
for pert in pert_order:
    xpos = x_positions[pert]
    sub = per_model[per_model["perturb_type"] == pert]
    for dataset_label, (marker, alpha, color) in marker_map.items():
        vals = sub.loc[sub["dataset"] == dataset_label, "r"].to_numpy()
        if vals.size == 0: 
            continue
        jitter = np.random.normal(loc=0, scale=0.08, size=vals.size)
        ax.scatter(xpos + jitter, vals, marker=marker, color=color, alpha=alpha, s=18, zorder=3,
                   label=dataset_label if dataset_label not in legend_handles else None)
        if dataset_label not in legend_handles:
            legend_handles[dataset_label] = True

xticks = [x_positions[p] for p in pert_order]
ax.set_xticks(xticks)
# ax.set_xticklabels(pert_order, fontsize=10, rotation=30, ha="right")
ax.set_xticklabels([pert_labels[p] for p in pert_order],
                   fontsize=10, rotation=30, ha="right")
y_top = np.nanmax([v for v in bar_means.values() if np.isfinite(v)]) if len(bar_means) else 0.3
y_top = y_top + 0.12
for g_name, g_items in groups:
    x_start = x_positions[g_items[0]] - 0.45
    x_end   = x_positions[g_items[-1]] + 0.45
    y = y_top
    ax.plot([x_start, x_end], [y, y], color="black", linewidth=1.0)
    ax.plot([x_start, x_start], [y, y-0.02], color="black", linewidth=1.0)
    ax.plot([x_end,   x_end],   [y, y-0.02], color="black", linewidth=1.0)
    ax.text((x_start + x_end)/2, y + 0.02, g_name, ha="center", va="bottom", fontsize=11)
ax.legend(title="Encoding models trained on", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.set_ylabel("R", fontsize=12)
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
ax.set_ylim(None, None)
plt.tight_layout()
plt.savefig("../plots/perturb.svg", format="svg", bbox_inches="tight")
plt.show()


# legend for multipanel
legend_elements = [
    Line2D([0], [0], marker='o', color='black', linestyle='',
           markersize=6, label='Tuckute2024'),
    Line2D([0], [0], marker='s', color='black', linestyle='',
           markersize=6, label='Pereira2018')
]

fig, ax = plt.subplots(figsize=(2, 1), dpi=400)
ax.axis('off')
ax.legend(handles=legend_elements,
          loc='center',
          frameon=False,
          ncol=2)
plt.savefig("../plots/legend_perturb.svg", format="svg", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(dpi=400, figsize=(9.5*.9, 3.7*.9))

for g_name, g_items in groups:
    base_c = group_color[g_name]
    alphas = alpha_schedules[g_name]
    for i, pert in enumerate(g_items):
        xpos = x_positions[pert]
        vals = per_model.loc[per_model["perturb_type"] == pert, "r"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        bp = ax.boxplot(
            [vals],
            positions=[xpos],
            widths=0.66,
            showfliers=False,
            patch_artist=True,
            whis=(5, 95),
            manage_ticks=False,
            zorder=2
        )


        fc_alpha = alphas[i] if i < len(alphas) else 1.0
        for b in bp['boxes']:
            b.set(facecolor=base_c, alpha=fc_alpha, edgecolor="black", linewidth=1.1)
        for ln in bp['whiskers'] + bp['caps']:
            ln.set(color="black", linewidth=1.0)
        for med in bp['medians']:
            med.set(color="black", linewidth=1.2)

marker_map = {"Pereira2018": ("s", 0.15, "black"), "Tuckute2024": ("o", 0.15, "black")}
legend_handles = {}
for pert in pert_order:
    xpos = x_positions[pert]
    sub = per_model[per_model["perturb_type"] == pert]
    for dataset_label, (marker, alpha, color) in marker_map.items():
        vals = sub.loc[sub["dataset"] == dataset_label, "r"].to_numpy()
        if vals.size == 0: 
            continue
        jitter = np.random.normal(loc=0, scale=0.08, size=vals.size)
        ax.scatter(xpos + jitter, vals, marker=marker, color=color, alpha=alpha, s=18, zorder=3,
                   label=dataset_label if dataset_label not in legend_handles else None)
        if dataset_label not in legend_handles:
            legend_handles[dataset_label] = True
xticks = [x_positions[p] for p in pert_order]
ax.set_xticks(xticks)
min_pos = min(x_positions[p] for p in pert_order)
max_pos = max(x_positions[p] for p in pert_order)
ax.set_xlim(min_pos - 1, max_pos + 1)
ax.margins(x=0)                            

ax.set_xticklabels([pert_labels[p] for p in pert_order],
                   fontsize=10, rotation=30, ha="right")
y_top = np.nanmax([v for v in bar_means.values() if np.isfinite(v)]) if len(bar_means) else 0.3
y_top = y_top + 0.12
for g_name, g_items in groups:
    x_start = x_positions[g_items[0]] - 0.45
    x_end   = x_positions[g_items[-1]] + 0.45
    y = y_top
    ax.plot([x_start, x_end], [y, y], color="black", linewidth=1.0)
    ax.plot([x_start, x_start], [y, y-0.02], color="black", linewidth=1.0)
    ax.plot([x_end,   x_end],   [y, y-0.02], color="black", linewidth=1.0)
    ax.text((x_start + x_end)/2, y + 0.02, g_name, ha="center", va="bottom", fontsize=11)
ax.legend(title="Encoding models trained on", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.set_ylabel("R", fontsize=12)
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2.25)
ax.spines["bottom"].set_linewidth(2.25)
ax.tick_params(axis="both", width=1.1, length=4)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
ax.set_ylim(None, None)
plt.tight_layout()
plt.savefig("../plots/perturb.svg", format="svg", bbox_inches="tight")
plt.show()

# =======================
# similarity vs encoding
# ======================

SIMILARITY_DIRS = {
    "Pereira2018": "/home/dev/Documents/PhD/Alice/additional_analyses/pereira/perturbation/similarity_outputs_pereira",
    "Tuckute2024": "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/similarity_outputs", 
}

pert_use = [p for p in pert_order if p != "intact"]
def load_similarity_summaries(sim_dirs):
    rows = []
    for dataset_label, base_dir in sim_dirs.items():
        if (not os.path.isdir(base_dir)):
            continue
        for mk in os.listdir(base_dir):
            mdir = os.path.join(base_dir, mk)
            if not os.path.isdir(mdir):
                continue
            summ_files = [f for f in os.listdir(mdir) if f.endswith("_summary.csv")]
            for sf in summ_files:
                path = os.path.join(mdir, sf)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if {"model","perturbation","mean_similarity","sem_similarity"}.issubset(df.columns):
                    df = df[["model","perturbation","mean_similarity","sem_similarity"]].copy()
                    df["dataset"] = dataset_label
                    rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["dataset","model","perturbation","mean_similarity","sem_similarity"])
    out = pd.concat(rows, ignore_index=True)
    out = out[out["perturbation"].isin(pert_use)]
    out["model"] = out["model"].astype(str)
    out.rename(columns={"perturbation":"perturb_type",
                        "mean_similarity":"sim_mean",
                        "sem_similarity":"sim_SE"}, inplace=True)
    return out

sim_df = load_similarity_summaries(SIMILARITY_DIRS)

enc_df = per_model.copy()
enc_df = enc_df[enc_df["perturb_type"].isin(pert_use)]
enc_df["model"] = enc_df["model"].astype(str)

merged = pd.merge(
    sim_df,
    enc_df[["dataset","perturb_type","model","r","SE"]],
    on=["dataset","perturb_type","model"],
    how="inner"
)

def _sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)

agg = (merged.groupby(["dataset","perturb_type"])
              .agg(sim_x=("sim_mean","mean"),
                   sim_x_SE=("sim_mean", _sem),
                   enc_y=("r","mean"),
                   enc_y_SE=("r", _sem),
                   n_models=("model","nunique"))
              .reset_index())

pert_to_group = {}
for gname, items in groups:
    for p in items:
        pert_to_group[p] = gname
agg["group"] = agg["perturb_type"].map(pert_to_group)
agg["label"] = agg["perturb_type"].map(pert_labels)

# Order points by your pert_order
agg["pert_order_idx"] = agg["perturb_type"].apply(lambda p: pert_order.index(p) if p in pert_order else 1e9)
agg = agg.sort_values(["dataset","pert_order_idx"]).reset_index(drop=True)


agg2 = (agg.groupby("perturb_type")
          .agg(sim_x=("sim_x","mean"),
               sim_x_SE=("sim_x", _sem),
               enc_y=("enc_y","mean"),
               enc_y_SE=("enc_y", _sem),
               n_datasets=("dataset","nunique"))
          .reset_index())

agg2["group"] = agg2["perturb_type"].map(pert_to_group)
agg2["label"] = agg2["perturb_type"].map(pert_labels)
agg2["pert_order_idx"] = agg2["perturb_type"].apply(lambda p: pert_order.index(p) if p in pert_order else 1e9)
agg2 = agg2.sort_values("pert_order_idx").reset_index(drop=True)

fig, ax = plt.subplots(dpi=400, figsize=(6.4, 4.2))
ax.grid(axis="both", linestyle="--", alpha=0.35, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for _, row in agg2.iterrows():
    base_c = group_color.get(row["group"], "gray")
    ax.errorbar(
        row["sim_x"], row["enc_y"],
        xerr=(0.0 if not np.isfinite(row["sim_x_SE"]) else row["sim_x_SE"]),
        yerr=(0.0 if not np.isfinite(row["enc_y_SE"]) else row["enc_y_SE"]),
        fmt="o", ms=6, mfc=base_c, mec="black",
        ecolor="black", elinewidth=0.9, capsize=3, alpha=0.95, zorder=3
    )
    ax.text(row["sim_x"], row["enc_y"] + 0.003, row["label"],
            ha="center", va="bottom", fontsize=8, color="black", alpha=0.9)

mask = np.isfinite(agg2["sim_x"]) & np.isfinite(agg2["enc_y"])
x = agg2.loc[mask, "sim_x"].to_numpy()
y = agg2.loc[mask, "enc_y"].to_numpy()

if x.size >= 2:
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    yline = slope * xline + intercept
    ax.plot(xline, yline, linewidth=1.5, zorder=2)

    r, p = pearsonr(x, y)
    ax.text(
        0.02, 0.98,
        f"r = {r:.2f},  p = {p:.3f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.8)
    )


group_handles = [Line2D([0],[0], marker="o", linestyle="",
                        color="black", markerfacecolor=group_color[g], label=g)
                 for g,_ in groups]
ax.legend(handles=group_handles, title="Perturbation group",
          frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

ax.set_xlabel("Sentence-level similarity to Intact (cosine)")
ax.set_ylabel("Encoding performance (R)")

if np.isfinite(x).any():
    ax.set_xlim(max(0.0, x.min()-0.02), min(1.0, x.max()+0.02))
if np.isfinite(y).any():
    ax.set_ylim(y.min()-0.02, y.max()+0.02)

plt.tight_layout()
plt.savefig("../plots/similarity_vs_encoding.svg", format="svg", bbox_inches="tight")
plt.show()
