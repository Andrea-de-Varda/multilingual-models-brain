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
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

def test_model_Ridge_random(X, y, n, saveto, save_results = True, shuffle=True, prefix = ""):
    if shuffle:
        kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_reg = []
    out_coefs = []
    out_pred = []; y_tot = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()
        np.random.seed(0)  # seed for reproducibility
        np.random.shuffle(y_train)  # shuffling y
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_pred.extend(y_pred.tolist())
        y_tot.extend(y_test.tolist())
        coefs = reg.coef_#; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(r)
    #print(round(np.mean(out_reg), 4))
    r_tot = round(pearsonr(out_pred, y_tot)[0], 4)
    print(r_tot)
    out_predictions = [y_tot, out_pred]
    if save_results:
        save(out_reg, f"results/out_reg/random_{prefix}{saveto}") # saving all rs and coefficients for later use
        save(out_coefs, f"results/coefficients/random_{prefix}{saveto}")
        save(out_predictions, f"results/predictions/random_{prefix}{saveto}")
    return np.mean(out_reg), np.std(out_reg)
    
def monolingual_encoding_random(langs, model_prefix, n_layers, shuffle=True, prefix = ""):
    layerwise_dict = {}
    if os.path.isfile(f"results/random/monolingual_{prefix}{model_prefix}"):
        print(f"Encoding for {model_prefix} already done")
    else:
        for n in range(n_layers+1):
            print(f"Processing layer {n}")
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
            #########################
            m = []; sd = []
            for idx, lang in enumerate(langs):
                the_r, the_sd = test_model_Ridge_random(fmri_data[idx], d[lang_code_dict[lang]], 10, saveto = f"{model_prefix}_{lang}_{n}", shuffle=shuffle, prefix = prefix)
                m.append(the_r)
                sd.append(the_sd)
            #########################
            mean_r = np.mean(m)
            print(f"Mean r = {mean_r} ({model_prefix} - {n})")
            #########################
            df = pd.DataFrame(zip(langs, m, sd), columns=["lang", "m", "sd"])
            layerwise_dict[n] = df
        save(layerwise_dict, f"results/random/monolingual_{prefix}{model_prefix}")
    return layerwise_dict

def multilingual_encoding_random(langs, model_prefix, n_layers):
    layerwise_dict = {}
    if os.path.isfile(f"results/random/multilingual_{model_prefix}"):
        print(f"Encoding for {model_prefix} already done")
    else:
        for n in range(n_layers+1):
            print(f"Processing layer {n}")
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
            ############################
            out_predictions = []
            for i in tqdm(range(len(langs))):
                X_data = fmri_data[:i] + fmri_data[i+1:] # exclude lang_i
                X_train = np.concatenate(X_data)
                y_names = langs[:i] + langs[i+1:]
                y_train = np.concatenate([d[lang_code_dict[name]] for name in y_names])
                X_test = fmri_data[i]#.reshape(1, -1)
                y_test = d[lang_code_dict[langs[i]]]
                #print(f"{langs[i]} --- {y_names}")
                # scaling
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
                np.random.seed(0)  # seed for reproducibility
                np.random.shuffle(y_train)  # shuffling y
                # fitting
                reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                r, p = pearsonr(y_test, y_pred)
                print(langs[i], r)
                out_predictions.append([langs[i], r])
            ############################
            out_predictions = pd.DataFrame(out_predictions, columns = ["lang", "r"])
            layerwise_dict[n] = out_predictions
            mean_r = np.mean(out_predictions["r"])
            print(f"Mean r = {mean_r} ({model_prefix} - {n})")
        save(layerwise_dict, f"results/random/multilingual_{model_prefix}")
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

xglm_small  = monolingual_encoding_random(xglm_langs, "xglm_small", 24)
xglm_med    = monolingual_encoding_random(xglm_langs, "xglm_med", 24)
xglm_large  = monolingual_encoding_random(xglm_langs, "xglm_large", 48)
xglm_xl     = monolingual_encoding_random(xglm_langs, "xglm_xl", 48)

# all langs
mbert       = monolingual_encoding_random(all_codes, "bert_base", 12)
distilmbert = monolingual_encoding_random(all_codes, "distilmbert", 6)

xlmr_base   = monolingual_encoding_random(all_codes, "xlmr_base", 12)
xlmr_large  = monolingual_encoding_random(all_codes, "xlmr_large", 24)

mt5_small   = monolingual_encoding_random(all_codes, "mt5_small", 8)
mt5_base    = monolingual_encoding_random(all_codes, "mt5_base", 12)
mt5_large   = monolingual_encoding_random(all_codes, "mt5_large", 24)

##################
# MODEL TRANSFER #
##################

xglm_small_multi    = multilingual_encoding_random(xglm_langs, "xglm_small", 24)
xglm_med_multi      = multilingual_encoding_random(xglm_langs, "xglm_med", 24)
xglm_large_multi    = multilingual_encoding_random(xglm_langs, "xglm_large", 48)
xglm_xl_multi       = multilingual_encoding_random(xglm_langs, "xglm_xl", 48)
mbert_multi         = multilingual_encoding_random(all_codes, "bert_base", 12)
distilmbert_multi   = multilingual_encoding_random(all_codes, "distilmbert", 6)
xlmr_base_multi     = multilingual_encoding_random(all_codes, "xlmr_base", 12)
xlmr_large_multi    = multilingual_encoding_random(all_codes, "xlmr_large", 24)
mt5_small_multi     = multilingual_encoding_random(all_codes, "mt5_small", 8)
mt5_base_multi      = multilingual_encoding_random(all_codes, "mt5_base", 12)
mt5_large_multi     = multilingual_encoding_random(all_codes, "mt5_large", 24)
mdeberta_multi      = multilingual_encoding_random(all_codes, "mdeberta", 12)
xlm_align_multi     = multilingual_encoding_random(all_codes, "xlm_align", 12)
infoxlm_base_multi  = multilingual_encoding_random(all_codes, "infoxlm_base", 12)
infoxlm_large_multi = multilingual_encoding_random(all_codes, "infoxlm_large", 24)
multiminilm_multi   = multilingual_encoding_random(all_codes, "multiminilm", 12)
nllb_d_600m_multi   = multilingual_encoding_random(all_codes, "nllb200_distilled_600M", 12)
nllb_d_1b_multi     = multilingual_encoding_random(all_codes, "nllb200_distilled_1B", 24)
nllb_1b_multi       = multilingual_encoding_random(all_codes, "nllb200_1B", 24)
mgpt_multi          = multilingual_encoding_random(mgpt_langs, "mgpt", 24)

#############################
# Non-shuffled (monol only) #
#############################

xglm_small    = monolingual_encoding_random(xglm_langs, "xglm_small", 24, shuffle=False, prefix = "sequential_")
xglm_med      = monolingual_encoding_random(xglm_langs, "xglm_med", 24, shuffle=False, prefix = "sequential_")
xglm_large    = monolingual_encoding_random(xglm_langs, "xglm_large", 48, shuffle=False, prefix = "sequential_")
xglm_xl       = monolingual_encoding_random(xglm_langs, "xglm_xl", 48, shuffle=False, prefix = "sequential_")
mbert         = monolingual_encoding_random(all_codes, "bert_base", 12, shuffle=False, prefix = "sequential_")
distilmbert   = monolingual_encoding_random(all_codes, "distilmbert", 6, shuffle=False, prefix = "sequential_")
xlmr_base     = monolingual_encoding_random(all_codes, "xlmr_base", 12, shuffle=False, prefix = "sequential_")
xlmr_large    = monolingual_encoding_random(all_codes, "xlmr_large", 24, shuffle=False, prefix = "sequential_")
mt5_small     = monolingual_encoding_random(all_codes, "mt5_small", 8, shuffle=False, prefix = "sequential_")
mt5_base      = monolingual_encoding_random(all_codes, "mt5_base", 12, shuffle=False, prefix = "sequential_")
mt5_large     = monolingual_encoding_random(all_codes, "mt5_large", 24, shuffle=False, prefix = "sequential_")
mdeberta      = monolingual_encoding_random(all_codes, "mdeberta", 12, shuffle=False, prefix = "sequential_")
xlm_align     = monolingual_encoding_random(all_codes, "xlm_align", 12, shuffle=False, prefix = "sequential_")
infoxlm       = monolingual_encoding_random(all_codes, "infoxlm_base", 12, shuffle=False, prefix = "sequential_")
infoxlm_large = monolingual_encoding_random(all_codes, "infoxlm_large", 24, shuffle=False, prefix = "sequential_")
multiminilm   = monolingual_encoding_random(all_codes, "multiminilm", 12, shuffle=False, prefix = "sequential_")
nllb_d_600m   = monolingual_encoding_random(all_codes, "nllb200_distilled_600M", 12, shuffle=False, prefix = "sequential_")
nllb_d_1b     = monolingual_encoding_random(all_codes, "nllb200_distilled_1B", 24, shuffle=False, prefix = "sequential_")
nllb_1b       = monolingual_encoding_random(all_codes, "nllb200_1B", 24, shuffle=False, prefix = "sequential_") 
mgpt          = monolingual_encoding_random(mgpt_langs, "mgpt", 24, shuffle=False, prefix = "sequential_")

