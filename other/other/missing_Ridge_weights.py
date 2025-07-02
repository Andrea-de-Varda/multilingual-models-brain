import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
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

def test_model_Ridge_single(X, y, n, saveto, save_results = True):
    kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    out_reg = []
    out_coefs = []
    out_predictions = []
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_predictions.append([y_test, y_pred])
        coefs = reg.coef_#; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(r)
    print(round(np.mean(out_reg), 4), round(np.std(out_reg), 4))
    if save_results:
        save(out_coefs, f"results/coefficients/{saveto}")
    return np.mean(out_reg), np.std(out_reg)
    
# I had forgotten to extract coefficients from Ridge regression, which I need for a plot
# I'll just redo a single layer for XLMRxl and mBERT

def monolingual_encoding_singlelayer(langs, model_prefix, layer):
    fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[layer]) for lang in langs]
    #########################
    m = []; sd = []
    for idx, lang in enumerate(langs):
        the_r, the_sd = test_model_Ridge_single(fmri_data[idx], d[lang_code_dict[lang]], 10, saveto = f"{model_prefix}_{lang}_{layer}")
        m.append(the_r)
        sd.append(the_sd)
    #########################
    mean_r = np.mean(m)
    print(f"Mean r = {mean_r}")

###############################################################################

# load fMRI data
with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)
    
all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
    

# XGLM langs 

xglm_langs = ["ca", "ja", "en", "es", "vi", "ta", "tr", "fr"]

# XGLM best layer is 16, mBERT 6
# mBERT best layer (monol) is 12
monolingual_encoding_singlelayer(xglm_langs, "xglm_xl", 16)
monolingual_encoding_singlelayer(all_codes, "bert_base", 6)

