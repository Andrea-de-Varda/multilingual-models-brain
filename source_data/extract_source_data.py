"""
Regenerates, from the files in this repository, the values in the Source Data workbooks published with the paper
(source_data/Source Data Fig. 1-3 and Extended Data Fig. 1 .xlsx). Writes one CSV per panel to source_data/csv/.
Usage (from anywhere): python source_data/extract_source_data.py [all|fig2|fig3|ed1|fig1c]
The xlsx workbooks were assembled from these CSVs (plus the descriptives of Figure 1C, see other/dataset_descriptives.py).
"""
# Mirrors the loaders/aggregations of plot_encoding_mono.py, plot_encoding_multi.py, plot_layerwise.py,
# time_series_corr.py, confirmatory_get_timeseries.py, perturbation_encoding_Pereira_Tuckute.py, test_residualize.py.
import os, sys, pickle, glob, warnings
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, pandas as pd
from scipy.stats import pearsonr, norm
from math import atanh, tanh, sqrt
warnings.filterwarnings("ignore")
sys.modules['numpy._core.numeric'] = np.core.numeric
OUT = "source_data/csv"; os.makedirs(OUT, exist_ok=True)
part = sys.argv[1] if len(sys.argv) > 1 else "all"

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["NLLB_d-small", "NLLB_d-large", "NLLB_large", "XLM-Align", "InfoXLM_small", "InfoXLM_large", "mMiniLM", "XLM-R_base", "XLM-R_large", "DistilmBERT", "mBERT", "mDeBERTa", "mT5_small", "mT5_base", "mT5_large", "mGPT", "XGLM_small", "XGLM_med", "XGLM_large", "XGLM_xl"]
model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
nice = dict(zip(model_names, names_formatted)); fam = dict(zip(model_names, model_family))
lang_full = {"es":"Spanish","mr":"Marathi","af":"Afrikaans","vi":"Vietnamese","ta":"Tamil","lt":"Lithuanian","tr":"Turkish","nl":"Dutch","no":"Norwegian","fa":"Farsi","fr":"French","ro":"Romanian"}

def pload(path):
    with open(path, 'rb') as f: return pickle.load(f)

def fname(model, froi, cond, kind=None):
    pre = "monolingual" if cond == "WITHIN" else "multilingual_multitrain"
    if kind == "md": return f"results/{pre}_md_{model}_{froi}"
    if kind == "rh": return f"results/{pre}_rh_{model}_{froi}"
    if kind == "circshift": return f"results/{pre}_{model}_{froi}_circshift"
    return f"results/{pre}_{model}_{froi}"

def col(cond): return "m" if cond == "WITHIN" else "r"
def langcol(df): return "lang" if "lang" in df.columns else "target_lang"

def r_to_z(r1, r2, n=130):
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(2*(1/(n-3))); return z
def stouffer_p(zs):
    zc = np.sum(zs)/np.sqrt(len(zs)); return zc, 2*norm.cdf(-abs(zc))

def best_idx(res, c):
    means = [res[k][c].mean() for k in sorted(res)]
    return int(np.argmax(means)), len(res)

LH = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp','all']
RH = ['Lang_RH_AntTemp','Lang_RH_IFG','Lang_RH_IFGorb','Lang_RH_MFG','Lang_RH_PostTemp','all']
MD = ['MD_LH_Precentral_A_PrecG','MD_LH_Precentral_B_IFGop','MD_LH_antParietal','MD_LH_insula','MD_LH_medialFrontal','MD_LH_midFrontal','MD_LH_midFrontalOrb','MD_LH_midParietal','MD_LH_postParietal','MD_LH_supFrontal','MD_RH_Precentral_A_PrecG','MD_RH_Precentral_B_IFGop','MD_RH_antParietal','MD_RH_insula','MD_RH_medialFrontal','MD_RH_midFrontal','MD_RH_midFrontalOrb','MD_RH_midParietal','MD_RH_postParietal','MD_RH_supFrontal','all']

if part in ("all", "fig2a"):
    long, summ = [], []
    for cond in ["WITHIN", "ACROSS"]:
        c = col(cond)
        for froi in LH:
            for m in model_names:
                res = pload(fname(m, froi, cond)); bi, nl = best_idx(res, c)
                df = res[bi]; lc = langcol(df)
                vals = df[c].to_numpy(); langs = df[lc].tolist()
                for l, v in zip(langs, vals):
                    long.append([cond, froi, nice[m], fam[m], bi, nl, lang_full.get(l, l), v])
                # significance vs circular shifts (as in compute_significance_for_froi)
                rnd = pload(fname(m, froi, cond, "circshift"))
                zt = []
                for shift in [26, 52, 78, 104]:
                    rr = rnd[shift]; rbi, _ = best_idx(rr, c)
                    rv = rr[rbi][c].to_numpy()
                    zs = [r_to_z(t, r) for t, r in zip(vals, rv)]
                    zt.append(np.sum(zs)/np.sqrt(len(zs)))
                zc, p = stouffer_p(zt)
                summ.append([cond, froi, nice[m], fam[m], bi, nl, len(vals), vals.mean(), vals.std(ddof=1), vals.std(ddof=1)/sqrt(len(vals)), zc, p])
    pd.DataFrame(long, columns=["condition","fROI","model","family","best_layer","n_layers","language","r"]).to_csv(f"{OUT}/fig2a_per_language.csv", index=False)
    pd.DataFrame(summ, columns=["condition","fROI","model","family","best_layer","n_layers","n_languages","mean_r","sd_across_languages","sem_across_languages","stouffer_z","p_two_sided"]).to_csv(f"{OUT}/fig2a_summary.csv", index=False)
    print("fig2a done")

if part in ("all", "fig2c"):
    rows, pts = [], []
    for cond in ["WITHIN", "ACROSS"]:
        c = col(cond)
        for net, frois, kind in [("LH", LH, None), ("RH", RH, "rh"), ("MD", MD, "md")]:
            for froi in frois:
                best = []
                for m in model_names:
                    res = pload(fname(m, froi, cond, kind)); bi, nl = best_idx(res, c)
                    v = res[bi][c].mean(); best.append(v)
                    pts.append([cond, net, froi, nice[m], bi, v])
                best = np.array(best)
                rows.append([cond, net, froi, len(best), best.mean(), best.std(ddof=0)/sqrt(len(best))])
    pd.DataFrame(rows, columns=["condition","network","fROI","n_models","mean_r","sem_across_models"]).to_csv(f"{OUT}/fig2c_summary.csv", index=False)
    pd.DataFrame(pts, columns=["condition","network","fROI","model","best_layer","best_layer_r_mean_across_languages"]).to_csv(f"{OUT}/fig2c_per_model.csv", index=False)
    print("fig2c done")

if part in ("all", "fig2b"):
    rows = []
    for cond in ["WITHIN", "ACROSS"]:
        c = col(cond)
        for m in model_names:
            res = pload(fname(m, "all", cond)); L = max(res.keys())
            for k in sorted(res):
                rows.append([cond, nice[m], fam[m], k, L, k/L, res[k][c].mean()])
    lw = pd.DataFrame(rows, columns=["condition","model","family","layer","last_layer_index","layer_position_normalized","mean_r_across_languages"])
    lw.to_csv(f"{OUT}/fig2b_layerwise.csv", index=False)
    # averaged curve (plot_layerwise.py: 7 bins over normalized depth, mean per model per bin, then mean and sd/sqrt(n_models))
    w_ = lw[lw.condition=="WITHIN"].rename(columns={"mean_r_across_languages":"m_mono"})
    a_ = lw[lw.condition=="ACROSS"].rename(columns={"mean_r_across_languages":"m_multi"})
    both = pd.merge(a_[["model","layer_position_normalized","m_multi"]], w_[["model","layer_position_normalized","m_mono"]], on=["model","layer_position_normalized"])
    both["l_bin"] = pd.cut(both["layer_position_normalized"], bins=np.linspace(0,1,8), include_lowest=True, right=True)
    agg = []
    for name, df in both.groupby("l_bin"):
        g = df.groupby("model").agg({"m_multi":"mean","m_mono":"mean"}); div = np.sqrt(len(g))
        agg.append([str(name), len(g), g.m_mono.mean(), g.m_mono.std()/div, g.m_multi.mean(), g.m_multi.std()/div])
    pd.DataFrame(agg, columns=["layer_bin","n_models","mean_r_WITHIN","sem_WITHIN","mean_r_ACROSS","sem_ACROSS"]).to_csv(f"{OUT}/fig2b_averaged.csv", index=False)
    print("fig2b done")

if part in ("all", "fig2d"):
    ppx = pd.read_csv("results/perplexity_results.csv"); enc = pd.read_csv("results/mono_multi.csv")
    enc["mod"] = model_names  # same order as in plot scripts (verified below by Model label)
    assert list(enc["Model"]) == ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
    mm = enc.merge(ppx[["mod","ppx_mean","ppx_se"]], on="mod")
    out = pd.DataFrame({"model": mm["mod"].map(nice), "family": mm["Family"], "n_languages": mm["n"],
                        "perplexity_residual_mean": mm["ppx_mean"], "perplexity_residual_sem_across_languages": mm["ppx_se"],
                        "r_WITHIN_best_layer": mm["Score_mono"], "sem_WITHIN": mm["se_mono"], "r_ACROSS_best_layer": mm["Score_multi"], "sem_ACROSS": mm["se_multi"]})
    out.to_csv(f"{OUT}/fig2d_per_model.csv", index=False)
    rw = pearsonr(out["perplexity_residual_mean"], out["r_WITHIN_best_layer"]); ra = pearsonr(out["perplexity_residual_mean"], out["r_ACROSS_best_layer"])
    pd.DataFrame([["WITHIN", len(out), rw[0], rw[1]], ["ACROSS", len(out), ra[0], ra[1]]], columns=["condition","n_models","pearson_r","p_two_sided"]).to_csv(f"{OUT}/fig2d_stats.csv", index=False)
    print("fig2d done", rw, ra)

def corr_ci(r, n, alpha=0.05):
    if n <= 3 or np.isclose(abs(r), 1.0): return (np.nan, np.nan)
    z = atanh(r); se = 1.0/sqrt(n-3); return (tanh(z-1.96*se), tanh(z+1.96*se))

if part in ("all", "ed1"):
    data = pd.read_csv("data/Alice_Story_TimeSeries.csv")
    rois = ['Lang_LH_AntTemp','Lang_LH_IFG','Lang_LH_IFGorb','Lang_LH_MFG','Lang_LH_PostTemp']
    rows = []
    for l in sorted(set(data["Language"])):
        temp = data[data.Language == l]
        if len(temp) != 24: continue
        sub1, sub2 = list(set(temp.UID))
        parts = {}
        for sub in (sub1, sub2):
            pd_ = temp[(temp.UID == sub) & (temp.ROI.isin(rois))]
            d = {roi: pd_[pd_.ROI == roi].iloc[:, 7:].values[0][9:-3].astype(float) for roi in rois}
            d["all"] = np.mean(np.stack(list(d.values())), axis=0); parts[sub] = d
        for roi in rois + ["all"]:
            t1, t2 = parts[sub1][roi], parts[sub2][roi]; mask = np.isfinite(t1) & np.isfinite(t2)
            r, p = pearsonr(t1[mask], t2[mask]); lo, hi = corr_ci(r, int(mask.sum()))
            rows.append([l, roi, int(mask.sum()), r, p, lo, hi])
    pd.DataFrame(rows, columns=["language","fROI","n_volumes","pearson_r","p_two_sided","ci95_low","ci95_high"]).to_csv(f"{OUT}/ed1a_studyI_timeseries_correlations.csv", index=False)
    # Study II
    data = pd.read_csv("confirmatory/data/confirmatory_time_series.csv"); data = data[data.ROI.isin(rois)]
    TRc = [f"T_{n}" for n in range(1, 143)]
    rows = []
    for l in data["Language"].unique():
        temp = data[data.Language == l]
        per_passage = {f"Passage_{i}": [] for i in (1,2,3)}
        for part_ in temp["UID"].unique():
            tp = temp[temp.UID == part_]
            for pn, run in zip(["Passage_1","Passage_2","Passage_3"], tp["Passage"].unique()):
                per_passage[pn].append((part_, tp[tp.Passage == run].loc[:, TRc].iloc[:, 9:-3].mean(axis=0).to_numpy()))
        for pn, lst in per_passage.items():
            if len(lst) == 3:
                for (a, ta), (b, tb) in [(lst[0], lst[1]), (lst[0], lst[2]), (lst[1], lst[2])]:
                    r, p = pearsonr(ta, tb); lo, hi = corr_ci(r, 130)
                    rows.append([l, pn, f"{a}-{b}", 130, r, p, lo, hi])
    df = pd.DataFrame(rows, columns=["language","passage","participant_pair","n_volumes","pearson_r","p_two_sided","ci95_low","ci95_high"])
    df.to_csv(f"{OUT}/ed1b_studyII_pair_correlations.csv", index=False)
    s = df.groupby(["language","passage"])["pearson_r"].agg(mean_r="mean", n_pairs="count", sd_pop=lambda x: np.std(x)).reset_index()
    s["sem_as_plotted_sd_over_sqrt_n"] = s["sd_pop"]/np.sqrt(s["n_pairs"]); s.to_csv(f"{OUT}/ed1b_studyII_summary.csv", index=False)
    print("ed1 done")

if part in ("all", "fig3"):
    import shutil
    shutil.copy("confirmatory/confirmatory_results_aggregated.csv", f"{OUT}/fig3a_confirmatory_results_aggregated.csv")
    shutil.copy("confirmatory/perturbation_results_per_model.csv", f"{OUT}/fig3b_perturbation_per_model.csv")
    shutil.copy("additional_analyses/residualize_info/results/test_results_per_model.csv", f"{OUT}/fig3c_ablation_per_model.csv")
    shutil.copy("additional_analyses/residualize_info/results/stats_transfer.csv", f"{OUT}/fig3c_stats.csv")
    # Fig 3B statistics, exactly as in perturbation_encoding_Pereira_Tuckute.py (language level, n = 130, Bonferroni over perturbation types)
    res = pd.read_csv("confirmatory/perturbation_results.csv"); pairs = []
    for (d, m, l), g in res.groupby(["dataset","model","language"]):
        ri = g.loc[g.perturb_type=="intact","r_mean"]
        if ri.empty or not np.isfinite(ri.iloc[0]): continue
        ri = float(ri.iloc[0])
        for pt, s in g.groupby("perturb_type"):
            if pt == "intact": continue
            rp = s.r_mean.iloc[0]
            if not np.isfinite(rp): continue
            r1 = np.clip(rp, -0.999999, 0.999999); r2 = np.clip(ri, -0.999999, 0.999999)
            pairs.append([pt, r_to_z(r1, r2)])
    pairs = pd.DataFrame(pairs, columns=["perturb_type","z"]); rows = []
    for pt, s in pairs.groupby("perturb_type"):
        zc, p = stouffer_p(s.z.values); rows.append([pt, len(s), zc, p])
    st = pd.DataFrame(rows, columns=["perturbation","n_comparisons","stouffer_z","p_two_sided"]); st["p_bonferroni"] = np.minimum(1, st.p_two_sided*len(st))
    st.sort_values("p_two_sided").to_csv(f"{OUT}/fig3b_stats.csv", index=False)
    # Fig 3C decodability curves
    rows = []
    for cond in ["semantics", "syntax"]:
        for m in model_names:
            f = f"additional_analyses/residualize_info/diagnostics/{m}_{cond}_diagnostics.csv"; fc = f"additional_analyses/residualize_info/diagnostics/{m}_cross_{cond}_diagnostics.csv"
            for path, kind in [(f, "targeted"), (fc, "non-targeted")]:
                if os.path.exists(path):
                    d = pd.read_csv(path); d["ablation"] = "meaning" if cond == "semantics" else "form"; d["feature_set"] = kind; d["model"] = nice[m]; rows.append(d)
    pd.concat(rows).to_csv(f"{OUT}/fig3c_decodability_curves.csv", index=False)
    print("fig3 done")

if part in ("all", "fig1c"):
    all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]
    rows = []
    for lang in all_codes:
        df = pd.read_csv("transcribed/"+lang+".csv"); df = df[df["end"] <= 260]
        rows.append([lang_full[lang], len(df["text"]), len(df["text"].unique())])
    d = pd.DataFrame(rows, columns=["language","tokens","types"]); d.to_csv(f"{OUT}/fig1c_studyI_tokens_types.csv", index=False)
    print("fig1c", d.tokens.mean(), d.types.mean())
