import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# paths
CONFIRM_BASE = "/home/dev/Documents/PhD/Alice/confirmatory"
REG_BASE = "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models"
NORM_BASE = "/home/dev/Documents/PhD/Alice/additional_analyses/control/perturbation/registered_models/normaliz_params"
EMB_DIR = os.path.join(CONFIRM_BASE, "embeddings")      # embeddings/Passage_1/mgpt_it (dict of layers)
TRANS_DIR = os.path.join(CONFIRM_BASE, "transcribed")    # transcribed/Passage_1/it.csv
FMRI_DICT = os.path.join(CONFIRM_BASE, "data/dict_fMRI") # pickled dict
os.chdir(CONFIRM_BASE)

LAYERNUM = 14  # mGPT layer

def load_pickle(path):
    with open(path, "rb") as h:
        return pickle.load(h)

def load_new_embeddings(passage, lang_code):
    return load_pickle(os.path.join(EMB_DIR, passage, f"mgpt_{lang_code}"))

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

d = load_pickle(FMRI_DICT)
passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_map = {c: n for c, n in zip(lang_codes, languages)}
keep = {'ar':[1,2,3],'de':[2,3],'hi':[2,3],'it':[1,2,3],'ko':[1],'pt':[1,3],'ru':[1,2,3],'zh':[1,2,3],'pl':[1,2,3]}

perturb_types = [p for p in os.listdir(REG_BASE) if os.path.isfile(os.path.join(REG_BASE, p))]
models = {p: load_pickle(os.path.join(REG_BASE, p)) for p in perturb_types}
norms  = {p: load_pickle(os.path.join(NORM_BASE, p)) for p in perturb_types}

rows = []
for lang in lang_codes:
    kept = keep[lang]
    for pert in tqdm(perturb_types, desc=f"{lang_map[lang]}"):
        reg = models[pert]
        X_scaler, y_scaler = norms[pert]
        rs = []
        for passage in passages:
            emb_dict = load_new_embeddings(passage, lang)
            X = preproc_align(lang, passage, emb_dict[LAYERNUM])
            X = X_scaler.transform(X)
            y = d[passage][lang_map[lang]].reshape(-1, 1)
            y = y_scaler.transform(y).flatten()
            y = y[:X.shape[0]]
            pred = reg.predict(X)
            rs.append(pearsonr(pred, y)[0])
        r_keep = [rs[i-1] for i in kept]
        rows.append([pert, lang_map[lang], rs[0], rs[1], rs[2], np.mean(r_keep), np.std(r_keep)])

results = pd.DataFrame(rows, columns=["perturb_type","language","r1","r2","r3","r_mean","r_sd"])
# results.to_csv("perturbation_confirmatory_results.csv", index=False)

# aggregation
agg = (results.groupby("perturb_type")
               .agg(r_mean=("r_mean","mean"),
                    se=("r_mean", lambda x: x.std()/np.sqrt(len(x)))))
# agg.to_csv("perturbation_confirmatory_results_aggregated.csv", index=False)
