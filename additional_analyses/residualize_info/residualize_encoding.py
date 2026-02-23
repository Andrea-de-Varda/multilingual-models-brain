"""
INLP Residualization Encoding Analysis
=======================================
For each multilingual transformer model:
  1. Extract sentence embeddings (Tuckute2024, cond=B) at the best layer
  2. Run INLP (regression variant) to remove semantic or syntactic features
  3. Train RidgeCV encoding models on: intact / semantics_ablated / syntax_ablated
  4. 5-fold CV for all three conditions
  5. Save models, scalers, projection matrices (W_stack), and diagnostics

INLP implementation follows Ravfogel et al. (2020) §4 using the Ben-Israel
numerically stable formula: collect all weight vectors into W_stack, recompute
the combined null-space projection via SVD from original X at each step.

Run on cluster (GPU). Mirrors perturbation_encoding.py in model list / layers.
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import torch
from transformers import (
    XGLMTokenizer, XGLMForCausalLM,
    BertTokenizer, BertForMaskedLM,
    AutoTokenizer, AutoModelForMaskedLM, AutoModel,
    MT5EncoderModel, T5Tokenizer,
    DistilBertModel, DistilBertTokenizer,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CONTROL_DATA = os.path.join(SCRIPT_DIR, "../control/data/brain-lang-data_participant_20230728.csv")
FEAT_SEM     = os.path.join(SCRIPT_DIR, "features/features_semantics.csv")
FEAT_SYN     = os.path.join(SCRIPT_DIR, "features/features_syntax.csv")

REG_OUT   = os.path.join(SCRIPT_DIR, "registered_models")
NORM_OUT  = os.path.join(SCRIPT_DIR, "registered_models/normaliz_params")
PROJ_OUT  = os.path.join(SCRIPT_DIR, "registered_models/projections")
CV_OUT    = os.path.join(SCRIPT_DIR, "cv_results")
DIAG_OUT  = os.path.join(SCRIPT_DIR, "diagnostics")

for d in [REG_OUT, NORM_OUT, PROJ_OUT, CV_OUT, DIAG_OUT]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Training data (Tuckute2024, same filter as perturbation_encoding.py)
# ──────────────────────────────────────────────
ROIS = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp']
control = pd.read_csv(CONTROL_DATA)
avg_1 = (control[control["roi"].isin(ROIS)]
         .groupby(["sentence", "target_UID"])
         .agg({"response_target": "mean", "cond": "first", "sentence": "first"})
         .reset_index(drop=True))
df_brain = avg_1.groupby("sentence").agg({"response_target": "mean", "cond": "first"})
df_brain = df_brain[df_brain["cond"] == "B"]
y = df_brain["response_target"].to_numpy()
sentences = df_brain.index.tolist()
print(f"Training sentences: {len(sentences)}, y shape: {y.shape}")

# ──────────────────────────────────────────────
# Load feature matrices for INLP
# ──────────────────────────────────────────────
feat_sem = pd.read_csv(FEAT_SEM, index_col="sentence").loc[sentences]
feat_syn = pd.read_csv(FEAT_SYN, index_col="sentence").loc[sentences]

sem_feature_names = list(feat_sem.columns)
syn_feature_names = list(feat_syn.columns)
print(f"Semantic features: {sem_feature_names}")
print(f"Syntactic features: {syn_feature_names}")

# Fill any NaNs with column mean (should be rare)
feat_sem = feat_sem.fillna(feat_sem.mean())
feat_syn = feat_syn.fillna(feat_syn.mean())

Y_sem = feat_sem.to_numpy()   # (n, 7)
Y_syn = feat_syn.to_numpy()   # (n, 7)

# ──────────────────────────────────────────────
# Models (identical to perturbation_encoding.py)
# ──────────────────────────────────────────────
dict_bestlayer = {
    "nllb200_distilled_600M": 9,  "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7,  "infoxlm_base": 7,  "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9,  "xlmr_large": 14,   "distilmbert": 4,    "bert_base": 6,
    "mdeberta": 9,   "mt5_small": 2,     "mt5_base": 9,       "mt5_large": 17,
    "mgpt": 14,      "xglm_small": 10,   "xglm_med": 16,      "xglm_large": 40,
    "xglm_xl": 48,
}

MODEL_SPECS = [
    ("xlmr_large",  AutoTokenizer,        AutoModelForMaskedLM, "xlm-roberta-large",                      "▁",  1, -1),
    ("xglm_xl",     XGLMTokenizer,        XGLMForCausalLM,      "facebook/xglm-4.5B",                    "▁",  1, None),
    ("xglm_large",  XGLMTokenizer,        XGLMForCausalLM,      "facebook/xglm-2.9B",                    "▁",  1, None),
    ("xglm_med",    XGLMTokenizer,        XGLMForCausalLM,      "facebook/xglm-1.7B",                    "▁",  1, None),
    ("xglm_small",  XGLMTokenizer,        XGLMForCausalLM,      "facebook/xglm-564M",                    "▁",  1, None),
    ("bert_base",   BertTokenizer,        BertForMaskedLM,      "bert-base-multilingual-cased",           "##", 1, -1),
    ("distilmbert", DistilBertTokenizer,  DistilBertModel,      "distilbert-base-multilingual-cased",     "##", 1, -1),
    ("xlmr_base",   AutoTokenizer,        AutoModelForMaskedLM, "xlm-roberta-base",                       "▁",  1, -1),
    ("mt5_small",   T5Tokenizer,          MT5EncoderModel,      "google/mt5-small",                       "▁",  0, -1),
    ("mt5_base",    T5Tokenizer,          MT5EncoderModel,      "google/mt5-base",                        "▁",  0, -1),
    ("mt5_large",   T5Tokenizer,          MT5EncoderModel,      "google/mt5-large",                       "▁",  0, -1),
    ("mdeberta",    AutoTokenizer,        AutoModel,            "microsoft/mdeberta-v3-base",              "▁",  1, -1),
    ("xlm_align",   AutoTokenizer,        AutoModel,            "microsoft/xlm-align-base",               "▁",  1, -1),
    ("infoxlm_base",  AutoTokenizer,      AutoModel,            "microsoft/infoxlm-base",                 "▁",  1, -1),
    ("infoxlm_large", AutoTokenizer,      AutoModel,            "microsoft/infoxlm-large",                "▁",  1, -1),
    ("multiminilm", AutoTokenizer,        AutoModel,            "microsoft/Multilingual-MiniLM-L12-H384", "▁",  1, -1),
    ("mgpt",        AutoTokenizer,        AutoModel,            "ai-forever/mGPT",                        "Ġ",  0, None),
    ("nllb200_distilled_600M", AutoTokenizer, AutoModel,        "facebook/nllb-200-distilled-600M",       "▁",  1, -1),
    ("nllb200_distilled_1B",   AutoTokenizer, AutoModel,        "facebook/nllb-200-distilled-1.3B",       "▁",  1, -1),
    ("nllb200_1B",             AutoTokenizer, AutoModel,        "facebook/nllb-200-1.3B",                 "▁",  1, -1),
]

CONDITIONS = ["intact", "semantics_ablated", "syntax_ablated"]

# ──────────────────────────────────────────────
# Tokenization helpers (identical to perturbation_encoding.py)
# ──────────────────────────────────────────────
def tok_maker(a, sep, toker, cased=True):
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out

@torch.no_grad()
def get_word_embeddings(sentence, tokenizer, model):
    if getattr(model.config, "is_encoder_decoder", False):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        outputs = model.get_encoder()(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
    else:
        input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
    return [h[0].detach().cpu().numpy() for h in hidden_states]

def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start=0, emb_end=None):
    layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs]
    toks = tok_maker(tokens, sep, tokenizer, cased)
    len_emb = layer_embs[0].shape[0]
    len_toks = sum(len(t) for t in toks)
    if len_emb != len_toks:
        raise ValueError("Token/embedding length mismatch. Check special tokens.")
    out_dict = {}
    for li, embs in enumerate(layer_embs):
        rows, idx = [], 0
        for word in toks:
            if len(word) == 1:
                rows.append(embs[idx]); idx += 1
            else:
                seg = embs[idx:idx + len(word)]; idx += len(word)
                rows.append(np.mean(seg, axis=0))
        out_dict[li] = np.vstack(rows)
    return out_dict

# ──────────────────────────────────────────────
# INLP (regression variant, Ben-Israel numerically stable)
# ──────────────────────────────────────────────
def compute_nullspace_projection(W_stack):
    """
    Given W_stack (n_steps × d), compute the orthogonal projection matrix
    onto the nullspace of W (i.e., remove the row-space of W).
    Uses SVD for numerical stability (Ben-Israel formula).
    Returns P_null (d × d).
    """
    U, S, Vt = np.linalg.svd(W_stack, full_matrices=False)
    rank = int(np.sum(S > 1e-10 * S[0]))
    if rank == 0:
        return np.eye(W_stack.shape[1], dtype=np.float32)
    P_rowspace = Vt[:rank].T @ Vt[:rank]   # (d, d)
    return np.eye(W_stack.shape[1], dtype=np.float32) - P_rowspace.astype(np.float32)

def r2_score_fast(X, y_col):
    """Simple R² using Ridge(alpha=1) for speed."""
    reg = Ridge(alpha=1.0)
    reg.fit(X, y_col)
    y_pred = reg.predict(X)
    ss_res = np.sum((y_col - y_pred) ** 2)
    ss_tot = np.sum((y_col - y_col.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def run_inlp(X_raw, Y_features, feature_names, model_key, condition_name,
             max_steps=10000, r2_threshold=0.01):
    """
    Runs INLP (regression variant) to remove all linear information about
    Y_features from X_raw.

    Returns:
      W_stack     : numpy array (n_steps × d) — accumulated weight vectors
      diagnostics : list of (step, feature_name, r2) tuples
    """
    # Standardize embeddings for the INLP step
    X_scaler = StandardScaler()
    X_std = X_scaler.fit_transform(X_raw).astype(np.float32)

    # Standardize each feature independently
    Y_scalers = []
    Y_std = np.zeros_like(Y_features, dtype=np.float64)
    for j in range(Y_features.shape[1]):
        sc = StandardScaler()
        Y_std[:, j] = sc.fit_transform(Y_features[:, j].reshape(-1, 1)).flatten()
        Y_scalers.append(sc)

    n, d = X_std.shape
    n_feats = Y_features.shape[1]

    weight_vectors = []  # list of (d,) arrays
    diagnostics = []     # list of (step, feature_name, r2)

    # ── Step 0: decodability from original embeddings ──
    print(f"  [INLP {condition_name}] Step 0 (no removal):")
    for j, fname in enumerate(feature_names):
        r2 = r2_score_fast(X_std, Y_std[:, j])
        diagnostics.append((0, fname, float(r2)))
        print(f"    {fname}: R²={r2:.4f}")

    step = 0
    last_r2s = [1.0] * n_feats  # initialise high to enter loop

    while step < max_steps:
        for j, fname in enumerate(feature_names):
            step += 1

            # Recompute current projected X from original X_std using all w so far
            if len(weight_vectors) == 0:
                X_curr = X_std
            else:
                W = np.vstack(weight_vectors)  # (step-1, d)
                P_null = compute_nullspace_projection(W)
                X_curr = X_std @ P_null

            # Train regressor for feature j
            reg = Ridge(alpha=1.0)
            reg.fit(X_curr, Y_std[:, j])
            w_j = reg.coef_.astype(np.float32)
            weight_vectors.append(w_j)

            # Evaluate decodability of ALL features after this step
            W_full = np.vstack(weight_vectors)
            P_null_full = compute_nullspace_projection(W_full)
            X_eval = X_std @ P_null_full

            step_r2s = []
            for jj, fname2 in enumerate(feature_names):
                r2 = r2_score_fast(X_eval, Y_std[:, jj])
                diagnostics.append((step, fname2, float(r2)))
                step_r2s.append(r2)
            last_r2s = step_r2s

            print(f"  [INLP {condition_name}] Step {step:3d} | removed: {fname:35s} "
                  f"| max_R²={max(step_r2s):.4f}")

            if step >= max_steps:
                break

        # Check convergence after full cycle through all features
        if max(last_r2s) < r2_threshold:
            print(f"  [INLP {condition_name}] Converged at step {step} "
                  f"(max R²={max(last_r2s):.4f} < {r2_threshold})")
            break

    if step == max_steps and max(last_r2s) >= r2_threshold:
        still_decodable = [feature_names[j] for j, r2 in enumerate(last_r2s)
                           if r2 >= r2_threshold]
        print(f"\n  WARNING [{model_key} | {condition_name}]: reached MAX_STEPS={max_steps}. "
              f"Max R²={max(last_r2s):.4f} still >= {r2_threshold}.")
        print(f"  Features still decodable: {still_decodable}")

    W_stack = np.vstack(weight_vectors)  # (n_steps, d)
    return W_stack, diagnostics


def compute_cross_diagnostics(X_raw, W_stack, Y_cross, cross_feature_names,
                               model_key, condition_name):
    """
    After INLP removed features in one condition, evaluate how well the
    *other* set of features can still be decoded at each INLP step.

    E.g., after semantics ablation, check syntax decodability (and vice versa).
    Uses W_stack[:k] to reconstruct the projection at step k.
    """
    X_scaler = StandardScaler()
    X_std = X_scaler.fit_transform(X_raw).astype(np.float32)

    Y_std = np.zeros_like(Y_cross, dtype=np.float64)
    for j in range(Y_cross.shape[1]):
        sc = StandardScaler()
        Y_std[:, j] = sc.fit_transform(Y_cross[:, j].reshape(-1, 1)).flatten()

    n_steps = W_stack.shape[0]
    diagnostics = []

    # Step 0: no removal
    for j, fname in enumerate(cross_feature_names):
        r2 = r2_score_fast(X_std, Y_std[:, j])
        diagnostics.append((0, fname, float(r2)))

    # Evaluate at full-cycle boundaries (every 7th step = one complete round)
    steps_to_eval = list(range(7, n_steps + 1, 7))

    for k in steps_to_eval:
        P_null = compute_nullspace_projection(W_stack[:k])
        X_proj = X_std @ P_null
        for j, fname in enumerate(cross_feature_names):
            r2 = r2_score_fast(X_proj, Y_std[:, j])
            diagnostics.append((k, fname, float(r2)))

    print(f"  [Cross-diag {condition_name}] Computed R² for {len(cross_feature_names)} "
          f"cross-features at {len(steps_to_eval)+1} steps")
    return diagnostics


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
for model_key, Tok, Mdl, pid, sep, emb_start, emb_end in MODEL_SPECS:

    out_dir   = os.path.join(REG_OUT,  model_key)
    norm_dir  = os.path.join(NORM_OUT, model_key)
    proj_dir  = os.path.join(PROJ_OUT, model_key)
    cv_path   = os.path.join(CV_OUT,   f"{model_key}_cv.pkl")
    diag_sem  = os.path.join(DIAG_OUT, f"{model_key}_semantics_diagnostics.csv")
    diag_syn  = os.path.join(DIAG_OUT, f"{model_key}_syntax_diagnostics.csv")
    cross_diag_sem = os.path.join(DIAG_OUT, f"{model_key}_cross_semantics_diagnostics.csv")
    cross_diag_syn = os.path.join(DIAG_OUT, f"{model_key}_cross_syntax_diagnostics.csv")
    for d in [out_dir, norm_dir, proj_dir]:
        os.makedirs(d, exist_ok=True)

    # Skip if all outputs already exist
    cond_files_done = all(
        os.path.isfile(os.path.join(out_dir, c)) and
        os.path.isfile(os.path.join(norm_dir, c))
        for c in CONDITIONS
    )
    proj_done = (
        os.path.isfile(os.path.join(proj_dir, "semantics_ablated")) and
        os.path.isfile(os.path.join(proj_dir, "syntax_ablated"))
    )
    cross_diag_done = (
        os.path.isfile(cross_diag_sem) and os.path.isfile(cross_diag_syn)
    )
    if cond_files_done and proj_done and os.path.isfile(cv_path) and cross_diag_done:
        print(f"=== {model_key}: all outputs exist. Skipping.")
        continue

    # Fast path: only cross-diagnostics missing → reload projections, skip INLP + training
    if cond_files_done and proj_done and os.path.isfile(cv_path) and not cross_diag_done:
        print(f"\n=== {model_key} ({pid}): computing cross-diagnostics only ===")
        tok = (Tok.from_pretrained(pid, add_prefix_space=True)
               if model_key == "mgpt" else Tok.from_pretrained(pid))
        mdl = Mdl.from_pretrained(pid).to(device).eval()
        best_layer = dict_bestlayer[model_key]
        X_rows = []
        for sent in tqdm(sentences, desc=f"{model_key} embeddings", leave=False):
            emb_dict = get_embeddings_tokens(
                sent.split(), sep, tok, mdl,
                cased=True, emb_start=emb_start, emb_end=emb_end
            )
            X_rows.append(emb_dict[best_layer].mean(axis=0))
        X_raw = np.vstack(X_rows).astype(np.float32)
        del mdl
        torch.cuda.empty_cache()

        with open(os.path.join(proj_dir, "semantics_ablated"), "rb") as h:
            W_sem = pickle.load(h)
        with open(os.path.join(proj_dir, "syntax_ablated"), "rb") as h:
            W_syn = pickle.load(h)

        cross_sem_rows = compute_cross_diagnostics(
            X_raw, W_sem, Y_syn, syn_feature_names, model_key, "semantics→syntax"
        )
        pd.DataFrame(cross_sem_rows, columns=["step", "feature_name", "r2"]).to_csv(
            cross_diag_sem, index=False
        )
        cross_syn_rows = compute_cross_diagnostics(
            X_raw, W_syn, Y_sem, sem_feature_names, model_key, "syntax→semantics"
        )
        pd.DataFrame(cross_syn_rows, columns=["step", "feature_name", "r2"]).to_csv(
            cross_diag_syn, index=False
        )
        continue

    print(f"\n=== {model_key} ({pid}) ===")

    # ── Extract embeddings ──
    tok = (Tok.from_pretrained(pid, add_prefix_space=True)
           if model_key == "mgpt" else Tok.from_pretrained(pid))
    mdl = Mdl.from_pretrained(pid).to(device).eval()
    best_layer = dict_bestlayer[model_key]

    X_rows = []
    for sent in tqdm(sentences, desc=f"{model_key} embeddings", leave=False):
        emb_dict = get_embeddings_tokens(
            sent.split(), sep, tok, mdl,
            cased=True, emb_start=emb_start, emb_end=emb_end
        )
        X_rows.append(emb_dict[best_layer].mean(axis=0))
    X_raw = np.vstack(X_rows).astype(np.float32)
    print(f"  Embeddings shape: {X_raw.shape}")

    del mdl
    torch.cuda.empty_cache()

    # ── Run INLP ──
    print("\n  Running INLP for semantics...")
    W_sem, diag_sem_rows = run_inlp(
        X_raw, Y_sem, sem_feature_names, model_key, "semantics"
    )
    pd.DataFrame(diag_sem_rows, columns=["step", "feature_name", "r2"]).to_csv(diag_sem, index=False)
    with open(os.path.join(proj_dir, "semantics_ablated"), "wb") as h:
        pickle.dump(W_sem, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved semantics W_stack shape: {W_sem.shape}")

    print("\n  Running INLP for syntax...")
    W_syn, diag_syn_rows = run_inlp(
        X_raw, Y_syn, syn_feature_names, model_key, "syntax"
    )
    pd.DataFrame(diag_syn_rows, columns=["step", "feature_name", "r2"]).to_csv(diag_syn, index=False)
    with open(os.path.join(proj_dir, "syntax_ablated"), "wb") as h:
        pickle.dump(W_syn, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved syntax W_stack shape: {W_syn.shape}")

    # ── Cross-condition diagnostics ──
    # After removing semantics, how well can syntax still be decoded? (and vice versa)
    cross_sem_rows = compute_cross_diagnostics(
        X_raw, W_sem, Y_syn, syn_feature_names, model_key, "semantics→syntax"
    )
    pd.DataFrame(cross_sem_rows, columns=["step", "feature_name", "r2"]).to_csv(
        cross_diag_sem, index=False
    )
    cross_syn_rows = compute_cross_diagnostics(
        X_raw, W_syn, Y_sem, sem_feature_names, model_key, "syntax→semantics"
    )
    pd.DataFrame(cross_syn_rows, columns=["step", "feature_name", "r2"]).to_csv(
        cross_diag_syn, index=False
    )

    # ── Compute ablated embeddings ──
    P_null_sem = compute_nullspace_projection(W_sem)
    P_null_syn = compute_nullspace_projection(W_syn)

    X_conditions = {
        "intact":            X_raw,
        "semantics_ablated": (X_raw @ P_null_sem).astype(np.float32),
        "syntax_ablated":    (X_raw @ P_null_syn).astype(np.float32),
    }

    # ── Train encoding models ──
    for cond, X_cond in X_conditions.items():
        if (os.path.isfile(os.path.join(out_dir, cond)) and
                os.path.isfile(os.path.join(norm_dir, cond))):
            print(f"  >> {cond}: already trained, skipping")
            continue

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train  = X_scaler.fit_transform(X_cond)
        y_train  = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        reg = RidgeCV(alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4))
        reg.fit(X_train, y_train)

        with open(os.path.join(out_dir, cond), "wb") as h:
            pickle.dump(reg, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(norm_dir, cond), "wb") as h:
            pickle.dump([X_scaler, y_scaler], h, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  >> {cond}: trained (alpha={reg.alpha_:.2e})")

    # ── 5-fold CV ──
    if not os.path.isfile(cv_path):
        print("\n  Running 5-fold CV...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {c: [] for c in CONDITIONS}

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_raw)):
            y_train_fold = y[train_idx]
            y_test_fold  = y[test_idx]

            for cond, X_cond in X_conditions.items():
                X_tr = X_cond[train_idx]
                X_te = X_cond[test_idx]

                X_sc = StandardScaler()
                y_sc = StandardScaler()
                X_tr_s = X_sc.fit_transform(X_tr)
                y_tr_s = y_sc.fit_transform(y_train_fold.reshape(-1, 1)).flatten()
                X_te_s = X_sc.transform(X_te)
                y_te_s = y_sc.transform(y_test_fold.reshape(-1, 1)).flatten()

                reg_cv = RidgeCV(alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4))
                reg_cv.fit(X_tr_s, y_tr_s)
                pred = reg_cv.predict(X_te_s)

                if (np.nanstd(pred) == 0 or np.nanstd(y_te_s) == 0 or
                        not np.isfinite(pred).all() or not np.isfinite(y_te_s).all()):
                    cv_results[cond].append(np.nan)
                else:
                    r = pearsonr(pred, y_te_s)[0]
                    cv_results[cond].append(float(r))

            print(f"    Fold {fold_i+1}/5 done. r_intact={cv_results['intact'][-1]:.3f}, "
                  f"r_sem={cv_results['semantics_ablated'][-1]:.3f}, "
                  f"r_syn={cv_results['syntax_ablated'][-1]:.3f}")

        with open(cv_path, "wb") as h:
            pickle.dump(cv_results, h, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  CV results saved to {cv_path}")
        for cond, rs in cv_results.items():
            print(f"    {cond}: mean_r={np.nanmean(rs):.3f} ± {np.nanstd(rs):.3f}")

    print(f"=== {model_key} complete ===\n")

print("\nAll models processed.")
