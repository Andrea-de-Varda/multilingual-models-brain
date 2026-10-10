"""
Functions used by the replication demo (encoding_demo.ipynb and run_demo.py).

The functions are copied from the analysis scripts of the paper, so that the demo runs exactly the same
procedure on a single model (DistilmBERT):
  - extract_embeddings: from get_model_embeddings.py (embeddings_chunked, get_embeddings_tokens, tok_maker)
  - preproc_align / embed_words / imputate_na: from fit_encoding.py
  - within_encoding / across_encoding / circshift: from fit_encoding.py (monolingual_encoding,
    multilingual_encoding_multitrain, monolingual_encoding_circshift, multilingual_encoding_multitrain_circshift)
  - r_to_z / stouffer: from plot_encoding_mono.py and plot_encoding_multi.py
Only file handling (paths, caching, progress messages) has been simplified.
"""
import os
import pickle
import re

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.stats import norm, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHAS = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)
SHIFTS = (26, 52, 78, 104)  # circular shifts (in TRs) used for the significance baselines

LANGS = {"af": "Afrikaans", "nl": "Dutch", "fa": "Farsi", "fr": "French", "lt": "Lithuanian", "mr": "Marathi",
         "no": "Norwegian", "ro": "Romanian", "es": "Spanish", "ta": "Tamil", "tr": "Turkish", "vi": "Vietnamese"}
# characters that the DistilmBERT tokenizer does not handle; replaced exactly as in get_model_embeddings.py
SPECIAL_REPLACE = {"fr": [("…", "...")], "lt": [("–", "-")], "ro": [("–", "-")]}


# ---------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------
def load_fmri():
    """Per-participant fROI time series of Study I (130 TRs, already shifted by 3 TRs = 6 s).
    Structure: d[language][participant_id][fROI] -> array of 130 values; fROI 'all' = network average."""
    with open(os.path.join(REPO_ROOT, "data", "dict_fROI"), "rb") as f:
        return pickle.load(f)


def load_transcription(lang):
    df = pd.read_csv(os.path.join(REPO_ROOT, "transcribed", f"{lang}.csv"))
    return df[df["end"] <= 260]  # words after 260 s were not scanned


# ---------------------------------------------------------------------------------------------------------
# Embeddings (get_model_embeddings.py)
# ---------------------------------------------------------------------------------------------------------
def tok_maker(a, sep, toker, cased=True):
    # matches each word with its sub-word tokens (credit: https://stackoverflow.com/questions/74458282)
    plainseq = " ".join(a)
    b = [re.sub(sep, "", item) for item in toker.tokenize(plainseq)]
    c = []
    for element in a:
        temp_list = []
        while "".join(temp_list) != (element if cased else element.lower()):
            temp_list.append(b.pop(0))
        c.append(temp_list)
    return c


def get_embeddings_tokens(tokens, sep, tokenizer, model, emb_start=1, emb_end=-1):
    import torch
    input_ids = tokenizer.encode(" ".join(tokens), return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    layer_embs = [h[0].cpu().numpy()[emb_start:emb_end] for h in outputs.hidden_states]  # drop [CLS] and [SEP]
    toks = tok_maker(tokens, sep, tokenizer)
    if layer_embs[0].shape[0] != sum(len(t) for t in toks):
        raise ValueError("number of embeddings and number of sub-word tokens differ")
    out = {}
    for layer, embs in enumerate(layer_embs):
        words, i = [], 0
        for word in toks:  # average the sub-word tokens of each word
            words.append(embs[i] if len(word) == 1 else np.mean(embs[i:i + len(word)], axis=0))
            i += len(word)
        out[layer] = np.vstack(words)
    return out


def extract_embeddings(lang, tokenizer, model, sep="##", ctx_size=100, progress=None):
    """Word embeddings for one language. Bidirectional models only see the preceding context: a window of
    (at most) 100 words is slid over the text and only the embedding of the last word of each window is kept."""
    text = load_transcription(lang)["text"].str.cat(sep=" ")
    for old, new in SPECIAL_REPLACE.get(lang, []):
        text = text.replace(old, new)
    words = text.split()
    chunks = [words[0:i] for i in range(1, ctx_size)] + \
             [words[i:i + ctx_size] for i in range(0, len(words) - ctx_size + 1)]
    if progress is None:
        import sys
        progress = sys.stderr.isatty()
    if progress:
        from tqdm.auto import tqdm
        chunks_iter = tqdm(chunks, desc=f"embeddings {lang}", leave=False)
    else:
        chunks_iter = chunks
    final = None
    for chunk in chunks_iter:
        embs = get_embeddings_tokens(chunk, sep, tokenizer, model)
        if final is None:
            final = {k: [] for k in embs}
        for layer in embs:
            final[layer].append(embs[layer][-1])
    return {k: np.array(v) for k, v in final.items()}


def get_all_embeddings(cache_dir, model_name="distilbert-base-multilingual-cased"):
    """Extract (or load from cache_dir) DistilmBERT embeddings for the 12 Study I languages."""
    os.makedirs(cache_dir, exist_ok=True)
    out, tokenizer, model = {}, None, None
    for lang in LANGS:
        path = os.path.join(cache_dir, f"distilmbert_{lang}")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                out[lang] = pickle.load(f)
            continue
        if model is None:
            from transformers import DistilBertModel, DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertModel.from_pretrained(model_name).eval()
        out[lang] = extract_embeddings(lang, tokenizer, model)
        with open(path, "wb") as f:
            pickle.dump(out[lang], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {LANGS[lang]:<11s} {out[lang][0].shape[0]} words")
    return out


# ---------------------------------------------------------------------------------------------------------
# Alignment of word embeddings with TRs (fit_encoding.py)
# ---------------------------------------------------------------------------------------------------------
def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)


def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)  # 130 TRs of 2 s
    emb_words = np.array([np.mean(embeddings[ids == i], axis=0) for i in range(time.shape[0])])
    return imputate_na(emb_words)  # TRs without words get the mean embedding


def preproc_align(lang, embeddings):
    """TR-level embeddings: average of the words whose offset falls within each 2-s TR."""
    df = load_transcription(lang).reset_index(drop=True)
    time = np.arange(0, 260, 2)
    words_id = np.array([np.where(t > time)[0][-1] for t in df["end"]], dtype=float)
    return embed_words(embeddings, words_id)


# ---------------------------------------------------------------------------------------------------------
# Encoding models (fit_encoding.py)
# ---------------------------------------------------------------------------------------------------------
BACKEND = "numpy"


def set_backend(name=None):
    """Choose the himalaya backend: 'torch_cuda' (GPU, as used for the paper), 'torch' (CPU, ~2x faster
    than numpy) or 'numpy'. With no argument, the fastest available one is used."""
    global BACKEND
    from himalaya.backend import set_backend as _set
    if name is None:
        try:
            import torch
            name = "torch_cuda" if torch.cuda.is_available() else "torch"
        except ImportError:
            name = "numpy"
    _set(name)
    BACKEND = name
    return name


def _fit_predict(Xtr, ytr, Xte):
    """Ridge regression with alpha chosen by nested 5-fold CV on the training folds (himalaya). As in
    fit_encoding.py, the kernel (dual) solver is used when there are fewer samples than features and the
    primal solver otherwise (same solution), and data are passed to torch backends as float32."""
    if Xtr.shape[0] > Xtr.shape[1]:
        from himalaya.ridge import RidgeCV as Est
    else:
        from himalaya.kernel_ridge import KernelRidgeCV as Est
    if BACKEND.startswith("torch"):
        import torch
        dev = "cuda" if BACKEND == "torch_cuda" else "cpu"
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
        pred = Est(alphas=ALPHAS).fit(t(Xtr), t(ytr[:, None])).predict(t(Xte))
        return pred.detach().cpu().numpy().ravel()
    return np.asarray(Est(alphas=ALPHAS).fit(Xtr, ytr[:, None]).predict(Xte)).ravel()


def cross_participant_cv(X, y_train_participant, y_test_participant, n_splits=10):
    """WITHIN: 10-fold sequential CV, train on one participant, test on the other; returns Pearson r of the
    concatenated held-out predictions and the predictions themselves."""
    preds, ys = [], []
    for tr, te in KFold(n_splits=n_splits, shuffle=False).split(X):
        xs, ysc = StandardScaler(), StandardScaler()
        Xtr, Xte = xs.fit_transform(X[tr]), xs.transform(X[te])
        ytr = ysc.fit_transform(y_train_participant[tr].reshape(-1, 1)).ravel()
        yte = ysc.transform(y_test_participant[te].reshape(-1, 1)).ravel()
        preds.extend(_fit_predict(Xtr, ytr, Xte).tolist())
        ys.extend(yte.tolist())
    return pearsonr(preds, ys)[0], np.array(preds), np.array(ys)


def within_encoding(X_by_lang, d, froi="all", shift=0):
    """WITHIN condition for one layer: mean of the two training directions (A->B and B->A) per language."""
    rows = []
    for lang, X in X_by_lang.items():
        p1, p2 = d[LANGS[lang]].keys()
        y1, y2 = np.roll(d[LANGS[lang]][p1][froi], shift), np.roll(d[LANGS[lang]][p2][froi], shift)
        r1 = cross_participant_cv(X, y1, y2)[0]
        r2 = cross_participant_cv(X, y2, y1)[0]
        rows.append([lang, r1, r2, 0.5 * (r1 + r2)])
    return pd.DataFrame(rows, columns=["lang", "m1", "m2", "m"])


def across_encoding(X_by_lang, d, froi="all", shift=0, return_predictions=False):
    """ACROSS condition for one layer: for each held-out language, train on 90% of the data of all other
    languages (both participants) and predict the corresponding held-out 10% of the target language."""
    langs = list(X_by_lang)
    X = [X_by_lang[l] for l in langs]
    Y = {l: [np.roll(d[LANGS[l]][p][froi], shift) for p in d[LANGS[l]]] for l in langs}
    rows, all_preds = [], {}
    for i, test_lang in enumerate(langs):
        pred, t1, t2 = [], [], []
        for tr, te in KFold(n_splits=10, shuffle=False).split(X[0]):
            Xtr_ = np.concatenate([X[j][tr] for j in range(len(langs)) if j != i])
            ytr1 = np.concatenate([Y[langs[j]][0][tr] for j in range(len(langs)) if j != i])
            ytr2 = np.concatenate([Y[langs[j]][1][tr] for j in range(len(langs)) if j != i])
            Xtr, ytr = np.concatenate([Xtr_, Xtr_]), np.concatenate([ytr1, ytr2])  # two participants per language
            xs, ysc = StandardScaler(), StandardScaler()
            Xtr = xs.fit_transform(Xtr)
            ytr = ysc.fit_transform(ytr.reshape(-1, 1)).ravel()
            pred.extend(_fit_predict(Xtr, ytr, xs.transform(X[i][te])).tolist())
            t1.extend(ysc.transform(Y[test_lang][0][te].reshape(-1, 1)).ravel().tolist())
            t2.extend(ysc.transform(Y[test_lang][1][te].reshape(-1, 1)).ravel().tolist())
        r = 0.5 * (pearsonr(pred, t1)[0] + pearsonr(pred, t2)[0])  # evaluated on each participant, averaged
        rows.append([test_lang, r])
        all_preds[test_lang] = (np.array(pred), np.array(t1), np.array(t2))
    df = pd.DataFrame(rows, columns=["target_lang", "r"])
    return (df, all_preds) if return_predictions else df


# ---------------------------------------------------------------------------------------------------------
# Statistics (plot_encoding_mono.py / plot_encoding_multi.py)
# ---------------------------------------------------------------------------------------------------------
def r_to_z(r1, r2, n=130):
    """Fisher r-to-z difference between two correlations computed on n = 130 TRs."""
    return (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(2 * (1 / (n - 3)))


def stouffer(zs):
    z = np.sum(zs) / np.sqrt(len(zs))
    return z, 2 * norm.cdf(-abs(z))


def best_layer(layerwise, col):
    layers = sorted(layerwise)
    return layers[int(np.argmax([layerwise[l][col].mean() for l in layers]))]


def significance(layerwise, layerwise_shifted, col):
    """As in the paper: per-language Fisher z differences between the best layer of the real model and the
    best layer of each circularly shifted model, combined across languages and then across the four shifts
    with Stouffer's method."""
    test = layerwise[best_layer(layerwise, col)][col].values
    z_per_shift = []
    for shift in SHIFTS:
        rnd = layerwise_shifted[shift][best_layer(layerwise_shifted[shift], col)][col].values
        z_per_shift.append(stouffer(r_to_z(test, rnd))[0])
    return stouffer(z_per_shift)
