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

def multilingual_encoding_registered(langs, model_prefix, prefix = "", overwrite = False, random = False): # best layer (XGLM-small, cross-validated) is 15
    n = dict_bestlayer[model_prefix]
    print(f"Processing layer {n}")
    fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
    ############################
    X_data = fmri_data # complete data
    X_train = np.concatenate(X_data)
    y_names = langs
    y_train = np.concatenate([d[lang_code_dict[name]] for name in y_names])
    X_test = X_train
    y_test = y_train
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    norm_params = [X_scaler, y_scaler]
    save(norm_params, f"confirmatory/registered_models/main/normaliz_params/{model_prefix}")
    # fitting
    if random:
        for random_idx, shift_val in enumerate([26, 52, 78, 104]):
            y_train_random = np.roll(y_train, shift_val)
            reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
            reg.fit(X_train, y_train_random)
            y_pred = reg.predict(X_test)
            r, p = pearsonr(y_test, y_pred)
            print(f"r = {round(r, 4)}")
            save(reg, f"confirmatory/registered_models/main/{model_prefix}_random_{random_idx}")
    else:
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, p = pearsonr(y_test, y_pred)
        print(f"r = {round(r, 4)}")
        save(reg, f"confirmatory/registered_models/main/{model_prefix}")

###############################################################################

dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 15,
                  "nllb200_1B" : 17,
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 9,
                  "xlmr_large" : 14,
                  "distilmbert" : 4, 
                  "bert_base" : 6,
                  "mdeberta" : 9, 
                  "mt5_small" : 2, 
                  "mt5_base" : 9,
                  "mt5_large" : 17, 
                  "mgpt" : 14, 
                  "xglm_small" : 10, 
                  "xglm_med" : 16, 
                  "xglm_large" : 40, 
                  "xglm_xl" : 48}

# load fMRI data
with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

all_langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
    
# XGLM langs 
xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

##################
# MODEL TRANSFER #
##################

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

model_langs_dict = {}
for model in model_names:
    if "mgpt" in model:
        model_langs_dict[model] = mgpt_langs
    elif "xglm" in model:
        model_langs_dict[model] = xglm_langs
    else:
        model_langs_dict[model] = all_codes

for model in model_names:
    langs = model_langs_dict[model] # retrieve languages
    # fit encoding
    multilingual_encoding_registered(langs, model)
    multilingual_encoding_registered(langs, model, random = True)
