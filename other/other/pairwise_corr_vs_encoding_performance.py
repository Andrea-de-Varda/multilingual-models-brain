import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
from glob import glob
import re
from scipy.stats import pearsonr
import pickle
import scipy.stats as stats

dirName = "/home/dev/Documents/PhD/Alice"
chdir(dirName)

corrs = pd.read_csv("results/correlations/correlation_participants_new.csv")

with open("results/monolingual_sequential_xglm_xl", 'rb') as handle:
    layerwise_dict = pickle.load(handle)
    
all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

xglm_langs = ["ca", "ja", "en", "es", "vi", "ta", "tr", "fr"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
lang_code_d_reversed = {v : k for k, v in lang_code_dict.items()}

corrs["lang"] = corrs["lang"].map(lang_code_d_reversed)



model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

for model in model_names:
    with open(f"results/monolingual_sequential_{model}", 'rb') as handle:
        layerwise_dict = pickle.load(handle)
    if "xglm" in model:
        langs = xglm_langs
    else:
        langs = all_codes
    outd = {lang : [] for lang in langs}
    for n in range(max(layerwise_dict.keys())):
        lw = layerwise_dict[n]
        lw.index = langs
        for lang in langs:
            enc = lw[lw.index == lang].m.values[0]
            outd[lang].append(enc)
    outd = {key : np.mean(value) for key, value in outd.items()} # avg across layers
    
    out = []
    for l in langs:
        r = corrs[corrs.lang == l].r.values[0]
        enc = outd[l]
        out.append([l, r, enc])
    out = pd.DataFrame(out, columns=["lang", "r_subj", "r_encoding"])
    
    print(model, pearsonr(out["r_subj"], out["r_encoding"]))

