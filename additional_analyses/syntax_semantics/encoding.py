"""
Train Ridge brain-encoding models on syntax/semantics variant embeddings.

For each sentence we have 10 paraphrase variants and 10 syntactic-alternative variants.
We extract embeddings for each variant (best layer, mean-pooled across tokens), then
average those vectors across the 10 variants to obtain one representation per sentence
per condition.  Ridge regression is then trained on those averaged representations
against the English brain responses – identical procedure to the perturbation encoding.

Usage:
    python encoding.py --dataset pereira
    python encoding.py --dataset control
"""

import argparse
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    MT5EncoderModel,
    T5Tokenizer,
    XGLMForCausalLM,
    XGLMTokenizer,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Embedding helpers (verbatim from perturbation_encoding.py)
# ---------------------------------------------------------------------------

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
        outputs = model.get_encoder()(
            **inputs, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states
    else:
        input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
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


# ---------------------------------------------------------------------------
# Model specs and best-layer dict (verbatim from perturbation_encoding.py)
# ---------------------------------------------------------------------------

dict_bestlayer = {
    "nllb200_distilled_600M": 9, "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48,
}

MODEL_SPECS = [
    ("xlmr_large",  AutoTokenizer,       AutoModelForMaskedLM, "xlm-roberta-large",                     "▁", 1, -1),
    ("xglm_xl",     XGLMTokenizer,       XGLMForCausalLM,      "facebook/xglm-4.5B",                    "▁", 1, None),
    ("xglm_large",  XGLMTokenizer,       XGLMForCausalLM,      "facebook/xglm-2.9B",                    "▁", 1, None),
    ("xglm_med",    XGLMTokenizer,       XGLMForCausalLM,      "facebook/xglm-1.7B",                    "▁", 1, None),
    ("xglm_small",  XGLMTokenizer,       XGLMForCausalLM,      "facebook/xglm-564M",                    "▁", 1, None),
    ("bert_base",   BertTokenizer,       BertForMaskedLM,      "bert-base-multilingual-cased",           "##", 1, -1),
    ("distilmbert", DistilBertTokenizer, DistilBertModel,      "distilbert-base-multilingual-cased",     "##", 1, -1),
    ("xlmr_base",   AutoTokenizer,       AutoModelForMaskedLM, "xlm-roberta-base",                      "▁", 1, -1),
    ("mt5_small",   T5Tokenizer,         MT5EncoderModel,      "google/mt5-small",                      "▁", 0, -1),
    ("mt5_base",    T5Tokenizer,         MT5EncoderModel,      "google/mt5-base",                       "▁", 0, -1),
    ("mt5_large",   T5Tokenizer,         MT5EncoderModel,      "google/mt5-large",                      "▁", 0, -1),
    ("mdeberta",    AutoTokenizer,       AutoModel,            "microsoft/mdeberta-v3-base",             "▁", 1, -1),
    ("xlm_align",   AutoTokenizer,       AutoModel,            "microsoft/xlm-align-base",              "▁", 1, -1),
    ("infoxlm_base",  AutoTokenizer,     AutoModel,            "microsoft/infoxlm-base",                "▁", 1, -1),
    ("infoxlm_large", AutoTokenizer,     AutoModel,            "microsoft/infoxlm-large",               "▁", 1, -1),
    ("multiminilm", AutoTokenizer,       AutoModel,            "microsoft/Multilingual-MiniLM-L12-H384","▁", 1, -1),
    ("mgpt",        AutoTokenizer,       AutoModel,            "ai-forever/mGPT",                       "Ġ", 0, None),
    ("nllb200_distilled_600M", AutoTokenizer, AutoModel,       "facebook/nllb-200-distilled-600M",      "▁", 1, -1),
    ("nllb200_distilled_1B",   AutoTokenizer, AutoModel,       "facebook/nllb-200-distilled-1.3B",      "▁", 1, -1),
    ("nllb200_1B",             AutoTokenizer, AutoModel,       "facebook/nllb-200-1.3B",                "▁", 1, -1),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ROIS = ["lang_LH_IFGorb", "lang_LH_IFG", "lang_LH_MFG", "lang_LH_AntTemp", "lang_LH_PostTemp"]


def load_brain_data(dataset: str):
    """Return (sentences, y) matching the ordering used by generate_materials.py."""
    if dataset == "pereira":
        csv_path = os.path.join(SCRIPT_DIR, "../pereira/pereira_averaged.csv")
        df = pd.read_csv(csv_path)
        sentences = df["Sentence"].tolist()
        y = df["EffectSize"].to_numpy()
    elif dataset == "control":
        csv_path = os.path.join(
            SCRIPT_DIR,
            "../control/data/brain-lang-data_participant_20230728.csv",
        )
        control = pd.read_csv(csv_path)
        avg_1 = (
            control[control["roi"].isin(ROIS)]
            .groupby(["sentence", "target_UID"])
            .agg({"response_target": "mean", "cond": "first", "sentence": "first"})
            .reset_index(drop=True)
        )
        df = avg_1.groupby("sentence").agg({"response_target": "mean", "cond": "first"})
        df = df[df["cond"] == "B"]
        sentences = df.index.tolist()
        y = df["response_target"].to_numpy()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return sentences, y


# ---------------------------------------------------------------------------
# Main encoding loop
# ---------------------------------------------------------------------------

def encode(dataset: str) -> None:
    sentences, y = load_brain_data(dataset)
    print(f"Dataset: {dataset} | Sentences: {len(sentences)}")

    mat_dir = os.path.join(SCRIPT_DIR, dataset, "materials")
    with open(os.path.join(mat_dir, "paraphrase_dict.pkl"), "rb") as f:
        para_dict = pickle.load(f)
    with open(os.path.join(mat_dir, "syntactic_dict.pkl"), "rb") as f:
        synt_dict = pickle.load(f)
    conditions = {"paraphrase": para_dict, "syntactic": synt_dict}

    REG_OUT  = os.path.join(mat_dir, "registered_models")
    NORM_OUT = os.path.join(REG_OUT,  "normaliz_params")
    os.makedirs(REG_OUT,  exist_ok=True)
    os.makedirs(NORM_OUT, exist_ok=True)

    for model_key, Tok, Mdl, pid, sep, emb_start, emb_end in MODEL_SPECS:
        out_dir  = os.path.join(REG_OUT,  model_key)
        norm_dir = os.path.join(NORM_OUT, model_key)
        os.makedirs(out_dir,  exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)

        missing = [
            cond for cond in conditions
            if not (
                os.path.isfile(os.path.join(out_dir,  cond)) and
                os.path.isfile(os.path.join(norm_dir, cond))
            )
        ]

        if not missing:
            print(f"=== {model_key}: all conditions already trained. Skipping.")
            continue

        print(f"\n=== {model_key} ({pid}) === | pending: {missing}")
        tok = (
            Tok.from_pretrained(pid, add_prefix_space=True)
            if model_key == "mgpt"
            else Tok.from_pretrained(pid)
        )
        mdl = Mdl.from_pretrained(pid).to(device).eval()
        best_layer = dict_bestlayer[model_key]

        for cond_name in missing:
            var_dict = conditions[cond_name]
            print(f">> Training encoder for condition: {cond_name}")

            X_rows = []
            for sent_idx in tqdm(range(len(sentences)), leave=False):
                entry = var_dict[sent_idx]
                variants = [v for v in entry["variants"] if v.strip()]

                variant_embs = []
                for variant in variants:
                    emb_dict = get_embeddings_tokens(
                        variant.split(), sep, tok, mdl,
                        cased=True, emb_start=emb_start, emb_end=emb_end,
                    )
                    variant_embs.append(emb_dict[best_layer].mean(axis=0))

                X_rows.append(np.mean(variant_embs, axis=0))

            X = np.vstack(X_rows)

            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X)
            y_train = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

            reg = RidgeCV(alphas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4))
            reg.fit(X_train, y_train)

            with open(os.path.join(out_dir,  cond_name), "wb") as h:
                pickle.dump(reg, h, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(norm_dir, cond_name), "wb") as h:
                pickle.dump([X_scaler, y_scaler], h, protocol=pickle.HIGHEST_PROTOCOL)

        del mdl
        torch.cuda.empty_cache()

    print(f"\nDone. Outputs in: {REG_OUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syntax/semantics encoding models.")
    parser.add_argument(
        "--dataset", required=True, choices=["pereira", "control"],
        help="Which dataset to process.",
    )
    args = parser.parse_args()
    encode(args.dataset)
