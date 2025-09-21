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

def embed_words(embeddings, words_id, func=np.mean):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)
    emb_words = []
    for i in range(time.shape[0]):
        emb = func(embeddings[ids == i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def get_fl(lang, func=np.mean, extra_vars=True):
    # zipf, len + (optional) word onset / rate
    df = pd.read_csv(f"transcribed/{lang}.csv")
    df = df[df["end"] <= 260]
    text = df["text"].str.cat(sep=" ")
    words = text.split()
    length = [len(w) for w in words]
    freq = [wordfreq.zipf_frequency(w, lang) for w in words]
    fl = np.array(list(zip(freq, length)))
    # assign each word to TR by end time
    time = np.arange(0, 260, 2)
    time_words = df["end"].to_numpy()
    words_id = np.zeros(len(time_words))
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i] > time)[0][-1]
    X = embed_words(fl, words_id, func=func)
    if extra_vars:
        # word rate per TR (by start bin)
        df["bin"] = (df["start"] // 2).astype(int)
        counts = np.bincount(df["bin"], minlength=130)[:130]
        # onset of first word within TR
        first_onsets = df.groupby("bin")["start"].min().reindex(range(130)).to_numpy()
        X = imputate_na(np.column_stack([X, counts, first_onsets]))
    return X

def ridge_cv_corr(X, y_train_part, y_test_part, n_splits=10, alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)):
    kf = KFold(n_splits=n_splits, shuffle=False)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    all_pred, all_true = [], []
    for tr_idx, te_idx in kf.split(X):
        X_tr = X_scaler.fit_transform(X[tr_idx])
        X_te = X_scaler.transform(X[te_idx])
        y_tr = y_scaler.fit_transform(y_train_part[tr_idx].reshape(-1, 1)).ravel()
        y_te = y_scaler.transform(y_test_part[te_idx].reshape(-1, 1)).ravel()
        reg = RidgeCV(alphas=alphas)
        reg.fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
        all_pred.extend(y_hat.tolist())
        all_true.extend(y_te.tolist())
    return pearsonr(all_pred, all_true)[0]

def baseline_within(langs, lang_code_dict, d_froi, froi, extra_vars=True, n_splits=10):
    results = {}
    # TR features
    X_feats = {lc: get_fl(lc, extra_vars=extra_vars) for lc in langs}
    for lc in langs:
        lang_name = lang_code_dict[lc]
        part1, part2 = list(d_froi[lang_name].keys())
        y1 = d_froi[lang_name][part1][froi]
        y2 = d_froi[lang_name][part2][froi]
        r12 = ridge_cv_corr(X_feats[lc], y1, y2, n_splits=n_splits)
        r21 = ridge_cv_corr(X_feats[lc], y2, y1, n_splits=n_splits)
        results[lc] = {"m1": r12, "m2": r21, "m": 0.5 * (r12 + r21)}
        print(f"WITHIN-BASE | lang={lc} | fROI={froi} | r12={r12:.3f} r21={r21:.3f} mean={results[lc]['m']:.3f}")
    return pd.DataFrame([
        {"lang": lc, **vals} for lc, vals in results.items()
    ])

def baseline_across(langs, lang_code_dict, d_froi, froi, extra_vars=True, n_splits=10, alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)):
    X_feats = {lc: get_fl(lc, extra_vars=extra_vars) for lc in langs}
    kf = KFold(n_splits=n_splits, shuffle=False)
    out_rows = []
    for i, train_lc in enumerate(langs):
        train_name = lang_code_dict[train_lc]
        p1, p2 = list(d_froi[train_name].keys())
        y_train_p1_full = d_froi[train_name][p1][froi]
        y_train_p2_full = d_froi[train_name][p2][froi]
        X_train_full = X_feats[train_lc]
        agg = {lc: {"pred": [], "t1": [], "t2": []} for lc in langs}
        for tr_idx, te_idx in kf.split(X_train_full):
            X_tr = np.concatenate([X_train_full[tr_idx], X_train_full[tr_idx]])
            y_tr = np.concatenate([y_train_p1_full[tr_idx], y_train_p2_full[tr_idx]])
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_tr = X_scaler.fit_transform(X_tr)
            y_tr = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
            reg = RidgeCV(alphas=alphas)
            reg.fit(X_tr, y_tr)
            for tgt_lc in langs:
                tgt_name = lang_code_dict[tgt_lc]
                tp1, tp2 = list(d_froi[tgt_name].keys())
                X_te = X_scaler.transform(X_feats[tgt_lc][te_idx])  # scale with TRAIN scaler
                y_t1 = y_scaler.transform(d_froi[tgt_name][tp1][froi][te_idx].reshape(-1, 1)).ravel()
                y_t2 = y_scaler.transform(d_froi[tgt_name][tp2][froi][te_idx].reshape(-1, 1)).ravel()
                y_hat = reg.predict(X_te)
                agg[tgt_lc]["pred"].extend(y_hat.tolist())
                agg[tgt_lc]["t1"].extend(y_t1.tolist())
                agg[tgt_lc]["t2"].extend(y_t2.tolist())
        per_tgt = {}
        for tgt_lc, dct in agg.items():
            r1 = pearsonr(dct["pred"], dct["t1"])[0]
            r2 = pearsonr(dct["pred"], dct["t2"])[0]
            per_tgt[tgt_lc] = 0.5 * (r1 + r2)
        mean_other = np.mean([v for k, v in per_tgt.items() if k != train_lc])
        out_rows.append({"target_lang": train_lc, "r": mean_other})
        print(f"ACROSS-BASE | train={train_lc} | fROI={froi} | mean r over others={mean_other:.3f}")
    return pd.DataFrame(out_rows)

def summarize_within(df):
    mean_ = df["m"].mean()
    se_   = df["m"].std(ddof=1) / np.sqrt(len(df))
    return mean_, se_

def summarize_across(df):
    mean_ = df["r"].mean()
    se_   = df["r"].std(ddof=1) / np.sqrt(len(df))
    return mean_, se_

froi = "all"
all_langs = ['Dutch','Farsi','French','Lithuanian','Norwegian','Romanian','Spanish','Tamil','Turkish','Vietnamese']
all_codes = ["nl","fa","fr","lt","no","ro","es","ta","tr","vi"]
lang_code_dict = {lc: name for lc, name in zip(all_codes, all_langs)}

with open("data/dict_fROI","rb") as handle:
    d_froi = pickle.load(handle)

df_within  = baseline_within(all_codes, lang_code_dict, d_froi, froi=froi, extra_vars=True, n_splits=10)
df_across  = baseline_across(all_codes, lang_code_dict, d_froi, froi=froi, extra_vars=True, n_splits=10)

mean_w, se_w = summarize_within(df_within)
mean_a, se_a = summarize_across(df_across)

print(f"[WITHIN]  mean r = {mean_w:.3f}, SE = {se_w:.3f}")
print(f"[ACROSS]  mean r = {mean_a:.3f}, SE = {se_a:.3f}")

############
# Study II #
############

# first: train FL models on training datasets

# Study I

with open("data/dict_fMRI", 'rb') as handle: # for study II, avg across fROIs
    d = pickle.load(handle)

fmri_data = [get_fl(lc, extra_vars=False) for lc in all_codes] # no onset in study II (meaningless for two datasets)

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
    control.append([np.mean(freq), np.mean(length)])
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
    pereira.append([np.mean(freq), np.mean(length)])
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
    reg.fit(X_train, y_train)
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

all_res.std() / np.sqrt(len(all_res))
