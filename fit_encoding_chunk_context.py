import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from time import sleep

chdir("/home/dev/Documents/PhD/Alice")

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name, layernum):
    with open("embeddings/split_context/"+name, 'rb') as handle:
        file = pickle.load(handle)
    file = {idx : vec[layernum] for idx, vec in file.items()}
    return file

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id, start, end):
    ids = words_id.astype(int)
    time = np.arange(start, end, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, chunks):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    new_chunks = {}
    for chunk_num, n in enumerate(range(0, 260, 26)):
        embeddings = chunks[chunk_num]
        start, end = n, n+26
        subset = df[(df["end"] > start) & (df["end"] < end)]
        time = np.arange(start, end, 2) # sampled each 2 sec
        time_words = subset["end"]
        words_id = np.zeros([len(time_words)])
        for i in range(len(time_words)):
            words_id[i] = np.where(time_words.tolist()[i]> time)[0][-1]
        embedded_words = embed_words(embeddings, words_id, start, end)
        new_chunks[chunk_num] = embedded_words
    return new_chunks
    
def monolingual_encoding_chunked(langs, model_prefix, n_layers, overwrite = False):
    layerwise_dict = {}
    if os.path.isfile(f"results/split_context/monolingual_{model_prefix}") and overwrite == False:
        print(f"Encoding for {model_prefix} already done")
    else:
        for layer_n in range(n_layers+1):
            print(f"Processing layer {layer_n}")
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}", layer_n)) for lang in langs]
            #########################
            m = []
            for idx, lang in enumerate(langs):
                chunks = fmri_data[idx]
                y = d[lang_code_dict[lang]]
                y_tot = []
                y_pred_all = []
                for chunk_num, n in tqdm(enumerate(range(0, 130, 13)), total = 10):
                    X_test = chunks[chunk_num]
                    X_train = np.vstack([chunks[chunk_idx] for chunk_idx in range(10) if chunk_idx != chunk_num])
                    test_indices = range(n, n+13)
                    y_train = np.array([y[i] for i in range(len(y)) if i not in test_indices])
                    y_test = y[n:n+13]
                    # scale
                    X_scaler = StandardScaler()
                    y_scaler = StandardScaler()
                    X_train = X_scaler.fit_transform(X_train)
                    X_test = X_scaler.transform(X_test)
                    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
                    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    y_pred_all.extend(y_pred.tolist())
                    y_tot.extend(y_test.tolist())
                    
                #print(f"{lang} = {the_r}")
                r_tot = pearsonr(y_pred_all, y_tot)[0]
                print(f"{lang} = {round(r_tot, 4)}")
                m.append(r_tot)
            #########################
            mean_r = np.mean(m)
            print(f"Mean r = {mean_r} ({model_prefix} - {layer_n})")
            #########################
            df = pd.DataFrame(zip(langs, m), columns=["lang", "m"])
            layerwise_dict[layer_n] = df
            #print(layerwise_dict)
            sleep(10)
        save(layerwise_dict, f"results/split_context/monolingual_{model_prefix}")
    return layerwise_dict

###############################################################################

# load fMRI data
with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)
    
all_langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
    

# XGLM langs 
# xglm_langs = ["ca", "ja", "en", "es", "vi", "ta", "tr", "fr", "ita"]
xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

#############################
# Non-shuffled (monol only) #
#############################

xglm_small    = monolingual_encoding_chunked(xglm_langs, "xglm_small", 24)
xglm_med      = monolingual_encoding_chunked(xglm_langs, "xglm_med", 24)
xglm_large    = monolingual_encoding_chunked(xglm_langs, "xglm_large", 48)
xglm_xl       = monolingual_encoding_chunked(xglm_langs, "xglm_xl", 48)
mbert         = monolingual_encoding_chunked(all_codes, "bert_base", 12)
distilmbert   = monolingual_encoding_chunked(all_codes, "distilmbert", 6)
xlmr_base     = monolingual_encoding_chunked(all_codes, "xlmr_base", 12)
xlmr_large    = monolingual_encoding_chunked(all_codes, "xlmr_large", 24)
mt5_small     = monolingual_encoding_chunked(all_codes, "mt5_small", 8)
mt5_base      = monolingual_encoding_chunked(all_codes, "mt5_base", 12)
mt5_large     = monolingual_encoding_chunked(all_codes, "mt5_large", 24)
mdeberta      = monolingual_encoding_chunked(all_codes, "mdeberta", 12)
xlm_align     = monolingual_encoding_chunked(all_codes, "xlm_align", 12)
infoxlm_base  = monolingual_encoding_chunked(all_codes, "infoxlm_base", 12)
infoxlm_large = monolingual_encoding_chunked(all_codes, "infoxlm_large", 24)
multiminilm   = monolingual_encoding_chunked(all_codes, "multiminilm", 12)
nllb_d_600m   = monolingual_encoding_chunked(all_codes, "nllb200_distilled_600M", 12)
nllb_d_1b     = monolingual_encoding_chunked(all_codes, "nllb200_distilled_1B", 24)
nllb_1b       = monolingual_encoding_chunked(all_codes, "nllb200_1B", 24)
mgpt          = monolingual_encoding_chunked(mgpt_langs, "mgpt", 24)
