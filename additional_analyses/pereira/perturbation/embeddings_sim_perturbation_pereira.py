import os, re, gc, math, json, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from contextlib import nullcontext
from transformers import (
    XGLMTokenizer, XGLMForCausalLM,
    BertTokenizer, BertForMaskedLM,
    AutoTokenizer, AutoModelForMaskedLM, AutoModel,
    MT5EncoderModel, T5Tokenizer,
    DistilBertModel, DistilBertTokenizer
)
from transformers.utils import logging as hf_logging

if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"Some weights .* pooler")
warnings.filterwarnings("ignore", message=r"You are using the default legacy behaviour of the T5Tokenizer")
warnings.filterwarnings("ignore", message=r"The sentencepiece tokenizer.*byte fallback")
warnings.filterwarnings("ignore", message=r"Mean of empty slice")
warnings.filterwarnings("ignore", message=r"Using `TRANSFORMERS_CACHE` is deprecated")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

dict_bestlayer = {
    "nllb200_distilled_600M": 9, "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48
}

MODEL_SPECS = [
    ("xlmr_large", AutoTokenizer, AutoModelForMaskedLM, "xlm-roberta-large", "▁", 1, -1),
    ("xglm_xl",   XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-4.5B", "▁", 1, None),
    ("xglm_large",XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-2.9B", "▁", 1, None),
    ("xglm_med",  XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-1.7B", "▁", 1, None),
    ("xglm_small",XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-564M", "▁", 1, None),
    ("bert_base", BertTokenizer,  BertForMaskedLM,      "bert-base-multilingual-cased", "##", 1, -1),
    ("distilmbert", DistilBertTokenizer, DistilBertModel,"distilbert-base-multilingual-cased", "##", 1, -1),
    ("xlmr_base", AutoTokenizer,  AutoModelForMaskedLM, "xlm-roberta-base", "▁", 1, -1),
    ("mt5_small", T5Tokenizer,    MT5EncoderModel,      "google/mt5-small", "▁", 0, -1),
    ("mt5_base",  T5Tokenizer,    MT5EncoderModel,      "google/mt5-base",  "▁", 0, -1),
    ("mt5_large", T5Tokenizer,    MT5EncoderModel,      "google/mt5-large", "▁", 0, -1),
    ("mdeberta",  AutoTokenizer,  AutoModel,            "microsoft/mdeberta-v3-base", "▁", 1, -1),
    ("xlm_align", AutoTokenizer,  AutoModel,            "microsoft/xlm-align-base", "▁", 1, -1),
    ("infoxlm_base", AutoTokenizer, AutoModel,          "microsoft/infoxlm-base", "▁", 1, -1),
    ("infoxlm_large",AutoTokenizer, AutoModel,          "microsoft/infoxlm-large", "▁", 1, -1),
    ("multiminilm", AutoTokenizer, AutoModel,           "microsoft/Multilingual-MiniLM-L12-H384", "▁", 1, -1),
    ("mgpt",      AutoTokenizer,  AutoModel,            "ai-forever/mGPT", "Ġ", 0, None),
    ("nllb200_distilled_600M", AutoTokenizer, AutoModel,"facebook/nllb-200-distilled-600M", "▁", 1, -1),
    ("nllb200_distilled_1B",   AutoTokenizer, AutoModel,"facebook/nllb-200-distilled-1.3B", "▁", 1, -1),
    ("nllb200_1B",             AutoTokenizer, AutoModel,"facebook/nllb-200-1.3B",           "▁", 1, -1),
]

def lengths_fast(tokenizer, texts):
    try:
        enc = tokenizer(texts, return_length=True, add_special_tokens=True, truncation=False)
        if isinstance(enc, dict) and "length" in enc:
            return enc["length"]
        if hasattr(enc, "encodings") and enc.encodings:
            return [len(e.ids) for e in enc.encodings]
    except Exception:
        pass
    return [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]

def tokenize_batch(tokenizer, texts):
    enc = tokenizer(
        [t.split() for t in texts],
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    word_ids = [None] * len(texts)
    encs = getattr(enc, "encodings", None)
    if encs:
        try:
            cand = [e.word_ids() for e in encs]
            seq_len = enc["input_ids"].shape[1]
            ok = all((w is not None) and (len(w) == seq_len) for w in cand)
            if ok:
                word_ids = cand
        except Exception:
            pass
    return enc, word_ids

def pool_words_numpy(layer_hidden, word_ids, attn_mask):
    B, S, D = layer_hidden.shape
    out = np.zeros((B, D), dtype=layer_hidden.dtype)
    for b in range(B):
        wid = word_ids[b]
        use_fallback = (
            wid is None or not isinstance(wid, list) or len(wid) != S or all(w is None for w in wid)
        )
        if use_fallback:
            v = layer_hidden[b, attn_mask[b]]
            out[b] = v.mean(axis=0)
            continue
        max_w = max([w for w in wid if w is not None], default=-1)
        if max_w < 0:
            v = layer_hidden[b, attn_mask[b]]
            out[b] = v.mean(axis=0)
            continue
        sums = np.zeros((max_w + 1, D), dtype=layer_hidden.dtype)
        cnts = np.zeros((max_w + 1,), dtype=np.int32)
        for i, w in enumerate(wid):
            if w is not None and attn_mask[b, i]:
                sums[w] += layer_hidden[b, i]
                cnts[w] += 1
        cnts[cnts == 0] = 1
        words = sums / cnts[:, None]
        out[b] = words.mean(axis=0)
    return out

def forward_hidden_states_encoder_only(model, enc):
    if getattr(model.config, "is_encoder_decoder", False):
        encoder = model.get_encoder()
        return encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
    return model(**enc, output_hidden_states=True, return_dict=True).hidden_states

def batched_sentence_embeddings(
    sentences, tokenizer, model, layer, emb_start, emb_end,
    token_budget=6000, use_fp16=True
):
    model.eval().to(device)

    lens = lengths_fast(tokenizer, sentences)
    order = np.argsort(lens).tolist()
    sentences_sorted = [sentences[i] for i in order]
    lens_sorted = [lens[i] for i in order]

    D = model.config.hidden_size
    embs = np.zeros((len(sentences_sorted), D), dtype=np.float32)

    i = 0
    while i < len(sentences_sorted):
        total = 0
        j = i
        while j < len(sentences_sorted) and total + lens_sorted[j] <= token_budget:
            total += lens_sorted[j]
            j += 1
        if j == i:  # extremely long item
            j = i + 1

        batch_texts = sentences_sorted[i:j]
        enc, word_ids = tokenize_batch(tokenizer, batch_texts)
        enc = {k: v.to(device) for k, v in enc.items() if k in ["input_ids", "attention_mask"]}

        ctx = torch.autocast(device_type=device, dtype=torch.float16) if (use_fp16 and device == "cuda") else nullcontext()
        with torch.inference_mode(), ctx:
            hs = forward_hidden_states_encoder_only(model, enc)

        layer_tensor = hs[layer] if len(hs) > layer else hs[-1]
        if emb_end is not None:
            layer_tensor = layer_tensor[:, emb_start:emb_end, :]
            attn = enc["attention_mask"][:, emb_start:emb_end]
        else:
            attn = enc["attention_mask"]

        layer_np = layer_tensor.detach().to("cpu").numpy()
        attn_np = attn.detach().to("cpu").numpy().astype(bool)
        S = layer_np.shape[1]
        word_ids = [w if (w is not None and isinstance(w, list) and len(w) == S) else None for w in word_ids]

        batch_embs = pool_words_numpy(layer_np, word_ids, attn_np)
        embs[i:j] = batch_embs.astype(np.float32)

        del hs, layer_tensor, enc, layer_np, attn_np, batch_embs
        torch.cuda.empty_cache(); gc.collect()
        i = j

    return embs, order

OUT_DIR = "similarity_outputs_pereira"
os.makedirs(OUT_DIR, exist_ok=True)

with open("perturbation_dict", "rb") as handle:
    perturbed = pickle.load(handle)  # {perturb_type: [sentences]}

if "intact" not in perturbed:
    raise ValueError("Missing 'intact' key in perturbation_dict.")

lens = {k: len(v) for k, v in perturbed.items()}
if len(set(lens.values())) != 1:
    raise ValueError(f"Inconsistent lengths across perturbations: {lens}")
N = next(iter(lens.values()))

def model_paths(model_key):
    mdir = os.path.join(OUT_DIR, model_key)
    os.makedirs(mdir, exist_ok=True)
    return {
        "dir": mdir,
        "per_sentence": os.path.join(mdir, f"{model_key}_per_sentence.csv"),
        "summary": os.path.join(mdir, f"{model_key}_summary.csv"),
        "ckpt": os.path.join(mdir, f"{model_key}_checkpoint.json"),
        "sort_idx": os.path.join(mdir, f"{model_key}_intact_sort_idx.npy"),
    }

def load_checkpoint(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"done": [], "meta": {}}

def save_checkpoint(path, ck):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f)
    os.replace(tmp, path)

def append_rows_csv(path, rows, header_cols):
    df = pd.DataFrame(rows)
    header = not os.path.isfile(path)
    df.to_csv(path, mode="a", index=False, header=header, columns=header_cols)

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("gpu:", torch.cuda.get_device_name(0))
    except Exception:
        pass

SUMMARY_COLS = ["model","hf_id","layer","perturbation","n","mean_similarity","std_similarity","sem_similarity"]
PER_SENT_COLS = ["model","hf_id","layer","perturbation","sentence_idx","similarity_cosine"]

for model_key, Tok, Mdl, pid, sep, emb_start, emb_end in MODEL_SPECS:
    print(f"\n=== {model_key} ({pid}) ===")
    paths = model_paths(model_key)
    ckpt = load_checkpoint(paths["ckpt"])
    done = set(ckpt.get("done", []))

    if all((p in done) for p in perturbed.keys() if p != "intact") and \
       os.path.isfile(paths["summary"]) and os.path.isfile(paths["per_sentence"]):
        print(f"{model_key}: already complete. Skipping.")
        continue

    tok = Tok.from_pretrained(pid, add_prefix_space=True) if model_key == "mgpt" else Tok.from_pretrained(pid)
    mdl = Mdl.from_pretrained(pid)
    best_layer = dict_bestlayer[model_key]
    if os.path.isfile(paths["sort_idx"]):
        sort_idx = np.load(paths["sort_idx"])
        intact_embs, _ = batched_sentence_embeddings(
            [perturbed["intact"][i] for i in sort_idx], tok, mdl, best_layer, emb_start, emb_end,
            token_budget=6000, use_fp16=True
        )
    else:
        intact_embs, sort_idx = batched_sentence_embeddings(
            perturbed["intact"], tok, mdl, best_layer, emb_start, emb_end,
            token_budget=6000, use_fp16=True
        )
        np.save(paths["sort_idx"], np.array(sort_idx))

    intact_norm = intact_embs / (np.linalg.norm(intact_embs, axis=1, keepdims=True) + 1e-9)
    inv = np.empty_like(sort_idx)
    inv[sort_idx] = np.arange(len(sort_idx))

    for perturb_type, sents in perturbed.items():
        if perturb_type == "intact" or (perturb_type in done):
            continue

        print(f"  -> {perturb_type}")
        s_sorted = [sents[i] for i in sort_idx]
        pert_embs, _ = batched_sentence_embeddings(
            s_sorted, tok, mdl, best_layer, emb_start, emb_end,
            token_budget=6000, use_fp16=True
        )
        pert_norm = pert_embs / (np.linalg.norm(pert_embs, axis=1, keepdims=True) + 1e-9)

        sims_sorted = np.sum(pert_norm * intact_norm, axis=1)
        sims = sims_sorted[inv]

        # per-sentence incremental write
        rows_sentence = [{
            "model": model_key, "hf_id": pid, "layer": best_layer,
            "perturbation": perturb_type, "sentence_idx": int(i),
            "similarity_cosine": float(s)
        } for i, s in enumerate(sims)]
        append_rows_csv(paths["per_sentence"], rows_sentence, PER_SENT_COLS)

        # summary write
        sd = float(np.std(sims, ddof=1)) if N > 1 else 0.0
        sem = float(sd / math.sqrt(N)) if N > 1 else 0.0
        row_sum = [{
            "model": model_key, "hf_id": pid, "layer": best_layer,
            "perturbation": perturb_type, "n": int(N),
            "mean_similarity": float(np.mean(sims)),
            "std_similarity": sd, "sem_similarity": sem
        }]
        append_rows_csv(paths["summary"], row_sum, SUMMARY_COLS)

        # checkpoint update
        done.add(perturb_type)
        ckpt["done"] = sorted(list(done))
        ckpt["meta"] = {"hf_id": pid, "layer": best_layer}
        save_checkpoint(paths["ckpt"], ckpt)

        torch.cuda.empty_cache(); gc.collect()

    del mdl
    torch.cuda.empty_cache(); gc.collect()

print("\nAll requested models processed (or skipped if already complete).")
