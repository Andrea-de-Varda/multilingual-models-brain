import numpy as np
import numpy.ma as ma
import pandas as pd
import re
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from os import chdir
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from time import sleep
import torch.nn.functional as F
import torch
import scipy.io as io
from scipy.special import softmax

chdir("/home/dev/Documents/PhD/Alice")

def save(file, name):
    with open("embeddings/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def tok_maker(a, sep, toker, cased = True):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
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
    # sentence_embedding = outputs[0]
    # return sentence_embedding.cpu().numpy()[0][1:]
    hidden_states = outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy() for hidden_state in hidden_states]
    return layer_embeddings


def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True):
    layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    toks = tok_maker(tokens, sep, tokenizer, cased)
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

def embeddings(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    text = df["text"].str.cat(sep=' ')
    if replace:
        for repl in special_replace:
            text = text.replace(repl[0], repl[1])
    embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model)
    save(embs, saveto+"_"+lang)
    return embs


def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    tmp = delay_mat(emb_words, np.arange(1,5))
    tmp = imputate_na(tmp)
    return tmp

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id) # using pre-computed pca parameters
    return embedded_words

def test_model_Ridge(X, y, n):
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
    print(np.mean(out_reg), np.std(out_reg))
    return np.mean(out_reg), np.std(out_reg)

# surprisal
def get_surprisal(prompt, toker, model):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(a, sep, toker, cased = False):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
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

def get_surprisal_tokens(tokens, sep, toker, model, cased=False):
    s = get_surprisal(" ".join(tokens), toker, model)
    toks = tok_maker(tokens, sep, toker, cased)
    theindex = 0
    out = []
    for index, word in enumerate(toks[1:]):
        if len(word) == 1:
            surp = s[theindex]
            theindex += 1
            out.append(surp)
        else:
            surp = s[theindex:theindex+len(word)]
            theindex += len(word)
            out.append(sum(surp))
    return out
    
#######################################
# GPT2-xl English-only (sanity check) #
#######################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 2048-d, 24 layers (+ emb layer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

en = embeddings("en", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "gpt2_xl")

with open("data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

results = []
for n in range(49):
    print(f"Processing layer {n}")
    en = load("gpt2_xl_en")[n]
    en = preproc_align("en", en)

    en_r, en_sd = test_model_Ridge(en, d["English"], 10)
    results.append([n, en_r, en_sd])

results = pd.DataFrame(results, columns = ["l", "r", "sd"])
#results.to_csv("other/english_GPT2_random.csv")

results = pd.read_csv("other/english_GPT2_random.csv")

colors = ['blue' if val < 0 else 'red' for val in results["r"]]
fig, ax = plt.subplots(figsize=(14*.85, 6*.85), dpi = 300)
bars = plt.bar(results["l"], results["r"], yerr=np.array(results["sd"])/sqrt(10),capsize=2, color=colors) # error bars as SE
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(base=0.05))
plt.xticks(results["l"])
plt.ylabel("Encoding R")
plt.xlabel("Layer")
plt.title("GPT2-xl")
plt.ylim(top=.65)
plt.ylim(bottom=-.05)
plt.tight_layout()
plt.show()


# a = get_embeddings_tokens("I like pizza".split(), sep = "Ġ", tokenizer = tokenizer, model = model)
# b = get_embeddings_tokens("I like pasta".split(), sep = "Ġ", tokenizer = tokenizer, model = model)
# c = get_embeddings_tokens("I like cats".split(), sep = "Ġ", tokenizer = tokenizer, model = model)
# d = get_embeddings_tokens("I like dogs".split(), sep = "Ġ", tokenizer = tokenizer, model = model)
# e = get_embeddings_tokens("I like cars".split(), sep = "Ġ", tokenizer = tokenizer, model = model)
# f = get_embeddings_tokens("I like bikes".split(), sep = "Ġ", tokenizer = tokenizer, model = model)

# a[48][2].shape # 1600, *4

# pearsonr(a[48][2],b[48][2])
# pearsonr(c[48][2],d[48][2])
# pearsonr(e[48][2],f[48][2])

#########################################################
# check how surprisal, freq, len do on the same dataset #
#########################################################

df = pd.read_csv("transcribed/en.csv")
df = df[df["end"] <= 260]
text = df["text"].str.cat(sep=' ')

# length
length = [len(w) for w in text.split()]

# frequency
freq = pd.read_excel("/home/dev/Documents/Datasets/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr

freq = [get_f(w) for w in text.split()]

# surprisal
s_gpt2 = get_surprisal_tokens(text.split(), sep = "Ġ", toker = tokenizer, model = model, cased = True)
s_gpt2 = [0] + s_gpt2

sfl = np.array(list(zip(length, freq, s_gpt2)))
df = pd.read_csv("transcribed/en.csv")
df = df[df["end"] <= 260]
time = np.arange(0, 260, 2) # sampled each 2 sec
time_words = df["end"]
words_id = np.zeros([len(time_words)])
for i in range(len(time_words)):
    words_id[i] = np.where(time_words[i]> time)[0][-1]
embedded_words = embed_words(sfl, words_id) # using pre-computed pca parameters

en_r, en_sd = test_model_Ridge(embedded_words, d["English"], 10)


# plot XGLM english

with open("results/layerwise_within_xglm_small_randomized", 'rb') as handle:
    layerwise_dict = pickle.load(handle)

to_plot = []
for layer in range(25):
    m, sd = layerwise_dict[layer].loc[0, :]
    to_plot.append([layer, m, sd])
    
results = pd.DataFrame(to_plot, columns = ["l", "r", "sd"])
colors = ['blue' if val < 0 else 'red' for val in results["r"]]
fig, ax = plt.subplots(figsize=(10*.85, 6*.85), dpi = 300)
bars = plt.bar(results["l"], results["r"], yerr=np.array(results["sd"])/sqrt(10),capsize=2, color=colors) # error bars as SE
ax.yaxis.grid(True, which='both', linestyle='dashed', linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(base=0.05))
plt.xticks(results["l"])
plt.ylabel("Encoding R")
plt.xlabel("Layer")
plt.title("XGLM, small (English only)")
plt.ylim(top=.72)
plt.ylim(bottom=-.05)
plt.tight_layout()
plt.show()
