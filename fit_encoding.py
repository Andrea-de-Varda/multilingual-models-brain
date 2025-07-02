import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr

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
    time = np.arange(0, 260, 2) # fMRI data is composed by 130 images, each acquired every 2 sec (total 260 sec)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0) # average embeddings for the 2 sec time window
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv") # load transcription obtained with whisper
    df = df[df["end"] <= 260]   # remove words after 260 sec (not scanned)
    time = np.arange(0, 260, 2) # acquired every 2 sec (see embed words)
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    for i in range(len(time_words)): # find the words corresponding to each TR (pronounced between the start and the end of a TR)
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

def test_model_Ridge(X, y, n, saveto, save_results = True, shuffle=False, prefix = ""):
    if shuffle: # note that shuffling might artificially increase the encoding scores. Default is non-shuffled. All the analyses now are w/o shuffling.
        kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_reg = []
    out_coefs = []
    out_pred = []; y_tot = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for train_index, test_index in tqdm(kf.split(X), total=n):
        # Normalizing embedding and responses. The normalization parameters are always estimated on the training data and transferred to the test data
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)) # Alpha values log spaced. Alpha chosen with leave-one-out nested CV
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_pred.extend(y_pred.tolist())
        y_tot.extend(y_test.tolist())
        coefs = reg.coef_#; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(r)
    #print(round(np.mean(out_reg), 4))
    r_tot = pearsonr(out_pred, y_tot)[0]
    print(round(r_tot, 4))
    out_predictions = [y_tot, out_pred]
    if save_results:
        save(r_tot, f"results/rs/{prefix}{saveto}")
        save(out_reg, f"results/out_reg/{prefix}{saveto}") # saving all rs and coefficients for later use
        save(out_coefs, f"results/coefficients/{prefix}{saveto}")
        save(out_predictions, f"results/predictions/{prefix}{saveto}")
    return r_tot

def monolingual_encoding(langs, model_prefix, n_layers, d, shuffle=False, prefix = "", overwrite = False):
    # This simply repeats the process for (a) all the languages in the sample and (b) all the layers of a given model from which embeddings are available
    layerwise_dict = {}
    if os.path.isfile(f"results/monolingual_{prefix}{model_prefix}") and overwrite == False:
        print(f"Encoding for {model_prefix} already done")
    else:
        # Encoding evaluated layer by layer
        for n in range(n_layers+1):
            print(f"Processing layer {n}")
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
            #########################
            m = []
            for idx, lang in enumerate(langs):
                the_r = test_model_Ridge(fmri_data[idx], d[lang_code_dict[lang]], 10, saveto = f"{model_prefix}_{lang}_{n}", shuffle=shuffle, prefix = prefix)
                m.append(the_r)
            #########################
            mean_r = np.mean(m)
            #print(f"All rs = {m}")
            print(f"Mean r = {round(mean_r, 4)} ({model_prefix} - {n})")
            #########################
            df = pd.DataFrame(zip(langs, m), columns=["lang", "m"])
            layerwise_dict[n] = df
        save(layerwise_dict, f"results/monolingual_{prefix}{model_prefix}")
    return layerwise_dict

def multilingual_encoding(langs, model_prefix, n_layers, d, prefix = "", overwrite = False):
    # This function trains encoding models in a set of languages (all but one) and evaluates the encoding in the left-out language
    layerwise_dict = {}
    if os.path.isfile(f"results/multilingual_{prefix}{model_prefix}") and overwrite == False:
        print(f"Encoding for {model_prefix} already done")
    else:
        for n in range(n_layers+1):
            print(f"Processing layer {n}")
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
            ############################
            out_predictions = []
            for i in tqdm(range(len(langs))):
                X_data = fmri_data[:i] + fmri_data[i+1:] # Training data (X) excludes lang_i
                X_train = np.concatenate(X_data)
                y_names = langs[:i] + langs[i+1:]
                y_train = np.concatenate([d[lang_code_dict[name]] for name in y_names])
                X_test = fmri_data[i]
                y_test = d[lang_code_dict[langs[i]]]
                #print(f"{langs[i]} --- {y_names}")
                # scaling (normalization parameters are estimated on training data only)
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
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
        save(layerwise_dict, f"results/multilingual_{prefix}{model_prefix}")
    return layerwise_dict

###############################################################################

# load fMRI data
with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

all_langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}

xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

##################
# MODEL TRANSFER #
##################

xglm_small_multi  = multilingual_encoding(xglm_langs, "xglm_small", 24, d)
xglm_med_multi    = multilingual_encoding(xglm_langs, "xglm_med", 24, d)
xglm_large_multi  = multilingual_encoding(xglm_langs, "xglm_large", 48, d)
xglm_xl_multi     = multilingual_encoding(xglm_langs, "xglm_xl", 48, d)
mbert_multi       = multilingual_encoding(all_codes, "bert_base", 12, d)
distilmbert_multi = multilingual_encoding(all_codes, "distilmbert", 6, d)
xlmr_base_multi   = multilingual_encoding(all_codes, "xlmr_base", 12, d)
xlmr_large_multi  = multilingual_encoding(all_codes, "xlmr_large", 24, d)
mt5_small_multi   = multilingual_encoding(all_codes, "mt5_small", 8, d)
mt5_base_multi    = multilingual_encoding(all_codes, "mt5_base", 12, d)
mt5_large_multi   = multilingual_encoding(all_codes, "mt5_large", 24, d)
mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d)
xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d)
infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d)
infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d)
multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d)
nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d)
nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d)
nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d)
mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d)

#############################
# Non-shuffled (monol only) #
#############################

xglm_small  = monolingual_encoding(xglm_langs, "xglm_small", 24, d)
xglm_med    = monolingual_encoding(xglm_langs, "xglm_med", 24, d)
xglm_large  = monolingual_encoding(xglm_langs, "xglm_large", 48, d)
xglm_xl     = monolingual_encoding(xglm_langs, "xglm_xl", 48, d)
mbert       = monolingual_encoding(all_codes, "bert_base", 12, d)
distilmbert = monolingual_encoding(all_codes, "distilmbert", 6, d)
xlmr_base   = monolingual_encoding(all_codes, "xlmr_base", 12, d)
xlmr_large  = monolingual_encoding(all_codes, "xlmr_large", 24, d)
mt5_small   = monolingual_encoding(all_codes, "mt5_small", 8, d)
mt5_base    = monolingual_encoding(all_codes, "mt5_base", 12, d)
mt5_large   = monolingual_encoding(all_codes, "mt5_large", 24, d)
mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d)
xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d)
infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d)
infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d)
multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d)
nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d)
nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d)
nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d) 
mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d)

###############################################################################

####################
# Right hemisphere #
####################

with open("data/dict_fMRI_rh", 'rb') as handle:
    d_rh = pickle.load(handle)

# sequential split
xglm_small    = monolingual_encoding(xglm_langs, "xglm_small", 24, d_rh, prefix = "rh_")
xglm_med      = monolingual_encoding(xglm_langs, "xglm_med", 24, d_rh, prefix = "rh_")
xglm_large    = monolingual_encoding(xglm_langs, "xglm_large", 48, d_rh, prefix = "rh_")
xglm_xl       = monolingual_encoding(xglm_langs, "xglm_xl", 48, d_rh, prefix = "rh_")
mbert         = monolingual_encoding(all_codes, "bert_base", 12, d_rh, prefix = "rh_")
distilmbert   = monolingual_encoding(all_codes, "distilmbert", 6, d_rh, prefix = "rh_")
xlmr_base     = monolingual_encoding(all_codes, "xlmr_base", 12, d_rh, prefix = "rh_")
xlmr_large    = monolingual_encoding(all_codes, "xlmr_large", 24, d_rh, prefix = "rh_")
mt5_small     = monolingual_encoding(all_codes, "mt5_small", 8, d_rh, prefix = "rh_")
mt5_base      = monolingual_encoding(all_codes, "mt5_base", 12, d_rh, prefix = "rh_")
mt5_large     = monolingual_encoding(all_codes, "mt5_large", 24, d_rh, prefix = "rh_")
mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d_rh, prefix = "rh_")
xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d_rh, prefix = "rh_")
infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d_rh, prefix = "rh_")
infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d_rh, prefix = "rh_")
multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d_rh, prefix = "rh_")
nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_rh, prefix = "rh_")
nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_rh, prefix = "rh_")
nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d_rh, prefix = "rh_") 
mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d_rh, prefix = "rh_")


# multilingual
xglm_small_multi    = multilingual_encoding(xglm_langs, "xglm_small", 24, d_rh, prefix = "rh_")
xglm_med_multi      = multilingual_encoding(xglm_langs, "xglm_med", 24, d_rh, prefix = "rh_")
xglm_large_multi    = multilingual_encoding(xglm_langs, "xglm_large", 48, d_rh, prefix = "rh_")
xglm_xl_multi       = multilingual_encoding(xglm_langs, "xglm_xl", 48, d_rh, prefix = "rh_")
mbert_multi         = multilingual_encoding(all_codes, "bert_base", 12, d_rh, prefix = "rh_")
distilmbert_multi   = multilingual_encoding(all_codes, "distilmbert", 6, d_rh, prefix = "rh_")
xlmr_base_multi     = multilingual_encoding(all_codes, "xlmr_base", 12, d_rh, prefix = "rh_")
xlmr_large_multi    = multilingual_encoding(all_codes, "xlmr_large", 24, d_rh, prefix = "rh_")
mt5_small_multi     = multilingual_encoding(all_codes, "mt5_small", 8, d_rh, prefix = "rh_")
mt5_base_multi      = multilingual_encoding(all_codes, "mt5_base", 12, d_rh, prefix = "rh_")
mt5_large_multi     = multilingual_encoding(all_codes, "mt5_large", 24, d_rh, prefix = "rh_")
mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d_rh, prefix = "rh_")
xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d_rh, prefix = "rh_")
infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d_rh, prefix = "rh_")
infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d_rh, prefix = "rh_")
multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d_rh, prefix = "rh_")
nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_rh, prefix = "rh_")
nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_rh, prefix = "rh_")
nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d_rh, prefix = "rh_")
mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d_rh, prefix = "rh_")

##############
# MD NETWORK #
##############

with open("data/dict_fMRI_md", 'rb') as handle:
    d_md = pickle.load(handle)

xglm_small  = monolingual_encoding(xglm_langs, "xglm_small", 24, d_md, prefix = "md_")
xglm_med    = monolingual_encoding(xglm_langs, "xglm_med", 24, d_md, prefix = "md_")
xglm_large  = monolingual_encoding(xglm_langs, "xglm_large", 48, d_md, prefix = "md_")
xglm_xl     = monolingual_encoding(xglm_langs, "xglm_xl", 48, d_md, prefix = "md_")
mbert       = monolingual_encoding(all_codes, "bert_base", 12, d_md, prefix = "md_")
distilmbert = monolingual_encoding(all_codes, "distilmbert", 6, d_md, prefix = "md_")
xlmr_base   = monolingual_encoding(all_codes, "xlmr_base", 12, d_md, prefix = "md_")
xlmr_large  = monolingual_encoding(all_codes, "xlmr_large", 24, d_md, prefix = "md_")
mt5_small   = monolingual_encoding(all_codes, "mt5_small", 8, d_md, prefix = "md_")
mt5_base    = monolingual_encoding(all_codes, "mt5_base", 12, d_md, prefix = "md_")
mt5_large   = monolingual_encoding(all_codes, "mt5_large", 24, d_md, prefix = "md_")
mdeberta      = monolingual_encoding(all_codes, "mdeberta", 12, d_md, prefix = "md_")
xlm_align     = monolingual_encoding(all_codes, "xlm_align", 12, d_md, prefix = "md_")
infoxlm       = monolingual_encoding(all_codes, "infoxlm_base", 12, d_md, prefix = "md_")
infoxlm_large = monolingual_encoding(all_codes, "infoxlm_large", 24, d_md, prefix = "md_")
multiminilm   = monolingual_encoding(all_codes, "multiminilm", 12, d_md, prefix = "md_")
nllb_d_600m   = monolingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_md, prefix = "md_")
nllb_d_1b     = monolingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_md, prefix = "md_")
nllb_1b       = monolingual_encoding(all_codes, "nllb200_1B", 24, d_md, prefix = "md_") 
mgpt          = monolingual_encoding(mgpt_langs, "mgpt", 24, d_md, prefix = "md_")

# multilingual
xglm_small_multi    = multilingual_encoding(xglm_langs, "xglm_small", 24, d_md, prefix = "md_")
xglm_med_multi      = multilingual_encoding(xglm_langs, "xglm_med", 24, d_md, prefix = "md_")
xglm_large_multi    = multilingual_encoding(xglm_langs, "xglm_large", 48, d_md, prefix = "md_")
xglm_xl_multi       = multilingual_encoding(xglm_langs, "xglm_xl", 48, d_md, prefix = "md_")
mbert_multi         = multilingual_encoding(all_codes, "bert_base", 12, d_md, prefix = "md_")
distilmbert_multi   = multilingual_encoding(all_codes, "distilmbert", 6, d_md, prefix = "md_")
xlmr_base_multi     = multilingual_encoding(all_codes, "xlmr_base", 12, d_md, prefix = "md_")
xlmr_large_multi    = multilingual_encoding(all_codes, "xlmr_large", 24, d_md, prefix = "md_")
mt5_small_multi     = multilingual_encoding(all_codes, "mt5_small", 8, d_md, prefix = "md_")
mt5_base_multi      = multilingual_encoding(all_codes, "mt5_base", 12, d_md, prefix = "md_")
mt5_large_multi     = multilingual_encoding(all_codes, "mt5_large", 24, d_md, prefix = "md_")
mdeberta_multi      = multilingual_encoding(all_codes, "mdeberta", 12, d_md, prefix = "md_")
xlm_align_multi     = multilingual_encoding(all_codes, "xlm_align", 12, d_md, prefix = "md_")
infoxlm_base_multi  = multilingual_encoding(all_codes, "infoxlm_base", 12, d_md, prefix = "md_")
infoxlm_large_multi = multilingual_encoding(all_codes, "infoxlm_large", 24, d_md, prefix = "md_")
multiminilm_multi   = multilingual_encoding(all_codes, "multiminilm", 12, d_md, prefix = "md_")
nllb_d_600m_multi   = multilingual_encoding(all_codes, "nllb200_distilled_600M", 12, d_md, prefix = "md_")
nllb_d_1b_multi     = multilingual_encoding(all_codes, "nllb200_distilled_1B", 24, d_md, prefix = "md_")
nllb_1b_multi       = multilingual_encoding(all_codes, "nllb200_1B", 24, d_md, prefix = "md_")
mgpt_multi          = multilingual_encoding(mgpt_langs, "mgpt", 24, d_md, prefix = "md_")
