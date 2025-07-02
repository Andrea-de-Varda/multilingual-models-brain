import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import wordfreq
from scipy.stats import pearsonr
from tqdm import tqdm

chdir("/home/dev/Documents/PhD/Alice")

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id, func = np.mean):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = func(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def get_fl(lang, func = np.mean):
    # baseline with Zipf frequency and Length
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    
    text = df["text"].str.cat(sep=' ')
    words = text.split()
    length = [len(w) for w in words]
    freq   = [wordfreq.zipf_frequency(w, lang) for w in words]
    fl = np.array([[f, l] for f, l in zip(freq, length)])
    
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(fl, words_id, func = func)
    return embedded_words

def test_model_Ridge(X, y, n, shuffle=False):
    if shuffle:
        kf = KFold(n_splits=n, shuffle=True, random_state = 0)
    else:
        kf = KFold(n_splits=n, shuffle=False)
    out_reg = []
    out_coefs = []
    out_pred = []; y_tot = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for train_index, test_index in kf.split(X):
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()
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
    r_tot = pearsonr(out_pred, y_tot)[0]
    return r_tot

# ! Afrikaans and Marathi are missing from WordFreq !
all_langs = ['Dutch', 'Farsi', 'French', 'Lithuanian', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["nl", "fa", "fr", "lt", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}

###########
# testing #
###########

with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

fmri_data = [get_fl(lang) for lang in all_codes]
m = []
for idx, lang in enumerate(all_codes):
    the_r = test_model_Ridge(fmri_data[idx], d[lang_code_dict[lang]], 10)
    m.append(the_r)
mean_r_within = np.mean(m)
print(f"Mean r = {mean_r_within}")
# np.std(m)

out_predictions = []
for i in tqdm(range(len(all_codes))):
    X_data = fmri_data[:i] + fmri_data[i+1:] # exclude lang_i
    X_train = np.concatenate(X_data)
    y_names = all_codes[:i] + all_codes[i+1:]
    y_train = np.concatenate([d[lang_code_dict[name]] for name in y_names])
    X_test = fmri_data[i]#.reshape(1, -1)
    y_test = d[lang_code_dict[all_codes[i]]]
    # scaling
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
    out_predictions.append([all_codes[i], r])
out_predictions = pd.DataFrame(out_predictions, columns = ["lang", "r"])
mean_r_between = np.mean(out_predictions["r"])
print(f"Mean r = {mean_r_between}")

############
# Study II #
############

# first: train FL models on training datasets

# Study I

study1 = np.vstack(fmri_data)
study1_y = np.concatenate([d[lang_code_dict[lang]] for lang in all_codes])

# Control

rois = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp']
control = pd.read_csv("additional_analyses/control/data/brain-lang-data_participant_20230728.csv")
avg_1 = control[control["roi"].isin(rois)].groupby(["sentence", "target_UID"]).agg({"response_target" : "mean", "cond" : "first", "sentence" : "first"}).reset_index(drop=True) # first average across fROIs
df = avg_1.groupby("sentence").agg({"response_target" : "mean", "cond" : "first"}) # then average across participants
is_baseline = (df["cond"] == "B").to_numpy()
sentences_control = np.array(df.index.tolist())[is_baseline].tolist()
y_control = df["response_target"].to_numpy()[is_baseline]

control = []
for sent in sentences_control:
    words = sent.split()
    length = [len(w) for w in words]
    freq   = [wordfreq.zipf_frequency(w, "en") for w in words]
    control.append([np.mean(length), np.mean(freq)])
control = np.array(control)

# Pereira

pereira = pd.read_csv("additional_analyses/pereira/pereira_averaged.csv")
y_pereira = pereira["EffectSize"].to_numpy()
sentences_pereira = pereira["Sentence"].tolist()

pereira = []
for sent in sentences_pereira:
    words = sent.split()
    length = [len(w) for w in words]
    freq   = [wordfreq.zipf_frequency(w, "en") for w in words]
    pereira.append([np.mean(length), np.mean(freq)])
pereira = np.array(pereira)

# NatStories

def embed_words_natstor(embeddings, words_id, total_duration, func = np.mean):
    ids = words_id.astype(int)
    time = np.arange(0, total_duration, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = func(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

story_n_dict = {"1" : "boar",
                "2" : "aqua",
                "3" : "matchstickseller",
                "4" : "kingofbirds", 
                "5" : "elvis", 
                "6" : "mrsticky",
                "7" : "highschool",
                "10" : "tree",
                "9" : "tulips"}

stories = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]

with open("additional_analyses/NaturalStories/response/d_shift_3", 'rb') as handle:
    d3 = pickle.load(handle)
    

y_natstor = []
natstor = []
for story in stories:
    df = pd.read_csv("additional_analyses/NaturalStories/transcribed/"+story+".csv")
    total_duration =  len(d3[story_n_dict[story]]) * 2
    
    text = df["text"].str.cat(sep=' ')
    words = text.split()
    length = [len(w) for w in words]
    freq   = [wordfreq.zipf_frequency(w, "en") for w in words]
    fl = np.array([[f, l] for f, l in zip(freq, length)])
    
    time = np.arange(0, total_duration, 2)
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words_natstor(fl, words_id, total_duration, func = np.mean)
    y_natstor.append(d3[story_n_dict[story]])
    natstor.append(embedded_words)
y_natstor = np.concatenate(y_natstor)
natstor = np.concatenate(natstor)

###################################
# FITTING frequency-length models #
###################################

training_data = {"study1" : [study1, study1_y],
                 "control" : [control, y_control],
                 "pereira" : [pereira, y_pereira],
                 "natstor" : [natstor, y_natstor]}

model_dict = {}
for k, v in training_data.items():
    X, y = v
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X)
    y_train = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    # fitting
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X, y)
    model_dict[k] = [reg, X_scaler, y_scaler]

########################
# predict study 2 data #
########################

with open("confirmatory/data/dict_fMRI", 'rb') as handle:
    d_2 = pickle.load(handle)

def get_fl_study2(lang, passage, func = np.mean):
    # baseline with Zipf frequency and Length
    df = pd.read_csv(f"confirmatory/transcribed/{passage}/{lang}.csv")
    df = df[df["end"] <= 260]
    df = df[df["text"] != " "]
    
    text = df["text"].str.cat(sep=' ')
    words = text.split()
    length = [len(w) for w in words]
    freq   = [wordfreq.zipf_frequency(w, lang) for w in words]
    fl = np.array([[f, l] for f, l in zip(freq, length)])
    
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words.iloc[i]> time)[0][-1]
    embedded_words = embed_words(fl, words_id, func = func)
    return embedded_words

d_passages_keep = {'ar' : [1, 2, 3], 'de' : [2, 3], 'hi' : [2, 3], 'it' : [1, 2, 3], 'ko' : [1], 'pt' : [1, 3], 'ru' : [1, 2, 3], 'zh' : [1, 2, 3], 'pl' : [1, 2, 3]}

lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_code_dict = {k : v for k, v in zip(lang_codes, languages)}
lang_code_dict_inv = {v : k for k, v in lang_code_dict.items()}

passages = ["Passage_1", "Passage_2", "Passage_3"]

all_res = {k : [] for k in model_dict.keys()}
for idx, lang in enumerate(lang_codes):
    passages_keep = d_passages_keep[lang]
    lang_res = {k : [] for k in model_dict.keys()}
    for passage in passages_keep:
        thepassage = f"Passage_{str(passage)}"
        fmri_data_lang = get_fl_study2(lang, thepassage)
        y = d_2[thepassage][lang_code_dict[lang]]
        for train_name, values in model_dict.items():
            reg, X_scaler, y_scaler = values
            X = X_scaler.transform(fmri_data_lang)
            y = y_scaler.transform(y.reshape(-1, 1)).flatten()
            y_pred = reg.predict(X)
            r, _ = pearsonr(y_pred, y)
            lang_res[train_name].append(r)
    lang_res = {k : np.mean(v) for k, v in lang_res.items()}
    for k, v in lang_res.items():
        all_res[k].append(v)
all_res = pd.DataFrame(all_res)
all_res.mean()
        