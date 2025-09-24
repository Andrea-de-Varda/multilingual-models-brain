frois = ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp', 'all']

def get_best_all_langs(model, froi):
    """Returns per-language best-layer correlations (list) for given model & fROI."""
    return get_best_layerwise(load(model, froi=froi), give_all=True)

def p_to_ast(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def pairwise_froi_tests_for_model(model, frois, n_obs=130):
    """
    For a single model, compare each fROI against each other fROI.
    Uses Fisher z per language, then Stouffer across languages (two-tailed).
    Returns a tidy DataFrame and square matrices (z, p).
    """
    records = []
    froi_vecs = {f: get_best_all_langs(model, f) for f in frois}
    
    for i, fA in enumerate(frois):
        for j, fB in enumerate(frois):
            if i == j:
                continue
            a = froi_vecs[fA]
            b = froi_vecs[fB]
            z_langs = []
            for ra, rb in zip(a, b):
                z_ab, _ = r_to_z(ra, rb, n=n_obs)
                z_langs.append(z_ab)
            z_model = combine_z_statistics(z_langs, return_z=True)
            p_model = 2 * norm.cdf(-abs(z_model))
            records.append({
                "model": model,
                "froi_A": fA,
                "froi_B": fB,
                "z_model": z_model,
                "p_model": p_model
            })
    
    df = pd.DataFrame.from_records(records)
    df["pair"] = df["froi_A"] + " vs " + df["froi_B"]
    df["asterisk"] = df["p_model"].apply(p_to_ast)
    
    zmat = df.pivot(index="froi_A", columns="froi_B", values="z_model").reindex(index=frois, columns=frois)
    pmat = df.pivot(index="froi_A", columns="froi_B", values="p_model").reindex(index=frois, columns=frois)
    
    return df, zmat, pmat

def combine_across_models(pairwise_df):
    """
    Combine model-level z’s across models for each fROI pair using Stouffer.
    Returns a tidy DF and square matrices (z_combined, p_combined).
    """
    recs = []
    for (fA, fB), grp in pairwise_df.groupby(["froi_A", "froi_B"]):
        zs = grp["z_model"].tolist()
        z_comb = np.sum(zs) / np.sqrt(len(zs))
        p_comb = 2 * norm.cdf(-abs(z_comb))
        recs.append({
            "froi_A": fA,
            "froi_B": fB,
            "z_combined": z_comb,
            "p_combined": p_comb
        })
    out = pd.DataFrame(recs)
    out["pair"] = out["froi_A"] + " vs " + out["froi_B"]
    out["asterisk"] = out["p_combined"].apply(p_to_ast)
    
    zmat = out.pivot(index="froi_A", columns="froi_B", values="z_combined").reindex(index=frois, columns=frois)
    pmat = out.pivot(index="froi_A", columns="froi_B", values="p_combined").reindex(index=frois, columns=frois)
    return out, zmat, pmat

# -------- run per model --------
per_model_results = {}
per_model_zmats = {}
per_model_pmats = {}

for model in model_names:
    df_m, z_m, p_m = pairwise_froi_tests_for_model(model, frois, n_obs=130)
    per_model_results[model] = df_m
    per_model_zmats[model] = z_m
    per_model_pmats[model] = p_m

# -------- combine across models (Stouffer) --------
all_models_pairwise = pd.concat(per_model_results.values(), ignore_index=True)
combined_df, combined_zmat, combined_pmat = combine_across_models(all_models_pairwise)

def matrix_asterisks(pmat):
    ast = pmat.copy()
    ast[:] = ""
    for i in range(len(ast.index)):
        for j in range(len(ast.columns)):
            if i == j or pd.isna(pmat.iat[i, j]):
                continue
            p = pmat.iat[i, j]
            ast.iat[i, j] = p_to_ast(p)
    return ast

combined_asterisks = matrix_asterisks(combined_pmat)
