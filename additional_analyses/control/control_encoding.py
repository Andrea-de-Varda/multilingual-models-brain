import pandas as pd
import numpy as np
import numpy.ma as ma
from os import chdir
import re
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel, MT5EncoderModel, T5Tokenizer, XLMModel, XLMTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer
import math
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/control")

############
# GET DATA #
############

control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
df = control[control["roi"] == "lang_LH_netw"].groupby("sentence").agg({"response_target" : "mean", "cond" : "max"})

sentences = df.index.tolist()
y = df["response_target"].to_numpy()

##################
# GET EMBEDDINGS #
##################

def save(file, name):
    with open("embeddings/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def tok_maker(a, sep, toker, cased = True):
    plainseq = " ".join(a)
    b = [re.sub(sep, "", item) for item in toker.tokenize(plainseq)]
    c = []
    if cased:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element:
                temp_list.append(b.pop(0))
            c.append(temp_list)
    else:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element.lower():
                temp_list.append(b.pop(0))
            c.append(temp_list)
    return c

def get_word_embeddings(sentence, tokenizer, model):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy() for hidden_state in hidden_states]
    return layer_embeddings


def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start = 0, emb_end = None):
    layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs] # discard embeddings for special chars
    toks = tok_maker(tokens, sep, tokenizer, cased)
    len_emb = layer_embs[0].shape[0]
    len_toks = len([t for tok in toks for t in tok])
    if len_emb != len_toks:
        raise ValueError("The number of embeddings does not correspond to the number of tokens. Check the special characters added by the tokenizer.")
    out_dict = {}
    for idx, embs in enumerate(layer_embs):
        out = []
        theindex = 0
        for index, word in enumerate(toks):
            if len(word) == 1:
                emb = embs[theindex]
                theindex += 1
                out.append(emb)
            else:
                emb = embs[theindex:theindex+len(word)]
                theindex += len(word)
                out.append(np.mean(emb, axis=0))
        out = np.vstack(out)
        out_dict[idx] = out
    return out_dict

# train on control data, test on new Alice data
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

xglm_small_embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = None)
    xglm_small_embeddings.append(emb)
save(xglm_small_embeddings, "xglm_small")
xglm_small_embeddings = load("xglm_small")

# tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
# model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")


# tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
# model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")


# tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
# model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")


# tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
# model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")


# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
# model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")


# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")


# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
# model = MT5EncoderModel.from_pretrained("google/mt5-small")


# tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
# model = MT5EncoderModel.from_pretrained("google/mt5-base")


# tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
# model = MT5EncoderModel.from_pretrained("google/mt5-large")


# focusing on XLM-R large for now
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

xlmr_large_embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    xlmr_large_embeddings.append(emb)
# save(xlmr_large_embeddings, "xlmr_large")
xlmr_large_embeddings = load("xlmr_large")

#############################
# fit and transfer encoding #
#############################

def load_Alice(name):
    with open("../../embeddings/"+name, 'rb') as handle:
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
    df = pd.read_csv("../../transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

with open("../../data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)
    
# all_langs = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian', 'Italian']
# all_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro", 'ita']

all_langs = ['Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}

all_results = []
for layernum in range(25):
    X = np.vstack([vec[layernum].mean(axis = 0) for vec in xlmr_large_embeddings])
    ###################
    # k-fold encoding #
    ###################
    kf = KFold(n_splits=10, shuffle=True, random_state = 0)
    y_tot = []; out_pred = []
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    rs = []
    for train_index, test_index in tqdm(kf.split(X), total=10):
        X_train = X_scaler.fit_transform(X[train_index])
        X_test = X_scaler.transform(X[test_index])
        y_train = y_scaler.fit_transform(y[train_index].reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y[test_index].reshape(-1, 1)).flatten()
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        out_pred.extend(y_pred.tolist())
        y_tot.extend(y_test)
        rs.append(pearsonr(y_test, y_pred)[0])
    r_tot, _ = pearsonr(out_pred, y_tot)
    print(f"Monolingual encoding (layer {layernum})")
    print(f"R = {round(r_tot, 4)}")
    ############
    # transfer #
    ############
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X, y)
    # random baseline
    np.random.seed(0)  # seed for reproducibility
    y_random = y.copy()
    np.random.shuffle(y_random)  # shuffling y
    reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg_random.fit(X, y_random)
    #############################
    # predict held-out language #
    #############################
    for lang in all_codes:
        X_transf = preproc_align(lang, load_Alice(f"xlmr_large_{lang}")[layernum])
        y_transf = d[lang_code_dict[lang]]
        y_pred = reg.predict(X_transf)
        y_pred_random = reg_random.predict(X_transf)
        r_transf, _ = pearsonr(y_transf, y_pred)
        r_transf_random, _ = pearsonr(y_transf, y_pred_random)
        print(f"{lang:<5} {round(r_transf, 4):>+7} {round(r_transf_random, 4):>+7}")
        all_results.append([layernum, r_tot, np.mean(rs), np.std(rs), lang, r_transf, r_transf_random])
all_results = pd.DataFrame(all_results, columns = ["layer", "r_tot", "r_mean", "r_std", "lang", "r_transf", "r_random"])

# all_results.to_csv("results/xlmr_large.csv", index = False)
all_results = pd.read_csv("results/xlmr_large.csv")
all_results_filtered = all_results[all_results.lang != "en"]

###################
# plot BEST LAYER #
###################

# best layer is 14 (monol)

best_l = all_results[all_results.layer == 14].sort_values(by="r_transf")
best_l["language"] = best_l["lang"].map(lang_code_dict)
best_l.loc[best_l.index.max() + 1] = pd.Series({'language': ' '}) # add empty entry for 

plt.figure(figsize=(5, 4), dpi=300)
plt.barh(best_l['language'], best_l['r_transf'], color='navy', label='r_transf')
plt.axvline(0, color="black", lw=1)
plt.barh("Control", best_l['r_tot'].mean(), color='tomato', label='Total R')
plt.xlabel('R')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.tick_params(axis='y', which='both', left=False)
plt.xlim(-.04, None)
plt.show()

# random
plt.figure(figsize=(5, 4), dpi=300)
plt.barh(best_l['language'], best_l['r_random'], color='gray', label='r_random')
plt.axvline(0, color="black", lw=1)
plt.barh("Control", best_l['r_tot'].mean(), color='tomato', label='Total R')
plt.xlabel('R')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.tick_params(axis='y', which='both', left=False)
#plt.xlim(-.04, None)
plt.show()

##################
# PLOT LAYERWISE #
##################

layerwise = all_results_filtered.groupby("layer").agg({"layer" : "mean", "r_mean" : "mean", "r_std" : "mean", "r_transf" : "mean", "r_random" : "mean"})

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi=300)
ax = sns.lineplot(
    data=layerwise,
    x='layer', 
    y='r_mean', 
    markers=True, 
    dashes=False,
    color="blue",
    linewidth=3.5, 
    label="Control"
)
ax = sns.lineplot(
    data=layerwise,
    x='layer', 
    y='r_transf', 
    markers=True, 
    dashes=False,
    color="red",
    linewidth=3.5,
    label="Transfer"
)
ax = sns.lineplot(
    data=layerwise,
    x='layer', 
    y='r_random', 
    markers=True, 
    dashes=False,
    color="gray",
    linewidth=3.5,
    label="Random"
)

plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Layer', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(title_fontsize='22', fontsize='22', loc='upper right', bbox_to_anchor=(1.35, 1.03), frameon=True)
ax.set_xlim([layerwise['layer'].min()-.02, layerwise['layer'].max()+.02])
ax.set_ylim(-.1, .5)
plt.show()






