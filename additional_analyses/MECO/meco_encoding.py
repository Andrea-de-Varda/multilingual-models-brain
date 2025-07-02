import pandas as pd
import numpy as np
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

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/MECO")

meco = pd.read_csv("Meco_l1.csv", index_col=None)
meco = meco.iloc[: , 1:]; print("N (with NA) =", len(meco))
meco = meco[meco['ia'].notna()]; print("N (no NA) =",len(meco))
meco["trialid"] = meco.trialid.astype(int)
meco["itemid"] = meco.itemid.astype(int)
meco["ianum"] = meco.ianum.astype(int)
meco["unique"] = meco.trialid.astype(str)+"-"+meco.ianum.astype(str)+"-"+meco.lang
meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])] # wrong itemid assignment (as in 1.2 release)

# languages = ['du', 'it', 'en', 'no','sp', 'tr'] # 13 languages --  Dutch, English, Italian, Norwegian, Spanish, Turkish shared with Alice
languages = ['du', 'no', 'sp', 'tr']
mgpt_languages = ['sp', 'tr']
xglm_languages = ['sp', 'tr']

##################################
# get clean passages and fp data #
##################################

out = {lang:{} for lang in languages}
for language in languages: 
    print("\n\n", language.upper())
    for textid in set(meco.trialid):
        print(textid)
        temp = meco[(meco["lang"] == language) & (meco["trialid"] == textid)]
        therange = range(1, max(temp.ianum)+1)
        temp_out = []
        temp_out_unique = []
        fp = []
        for itemindex in therange:
            try:
                of_interest = temp[temp.ianum == itemindex]
                token = list(set(of_interest.ia))
                unique = list(set(of_interest.unique))
                if len(token) > 1:
                    print(textid, itemindex, token)
                temp_out_unique.append(unique[0])
                temp_out.append(token[0])
                of_interest['firstrun.dur'] = of_interest['firstrun.dur'].fillna(0)
                rt = np.nanmean(of_interest["firstrun.dur"])
                fp.append(rt)
            except IndexError:
                print(language, textid, itemindex)
        out[language][textid] = {"words" : temp_out, "fp" : fp, "unique" : temp_out_unique}

##################
# get embeddings #
##################

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open(name, 'rb') as handle:
        file = pickle.load(handle)
    return file
               
def tok_maker(a, sep, toker, cased = True):
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out

def get_word_embeddings(sentence, tokenizer, model, emb_start = 0, emb_end = None, split_words = False):
    #input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')
    if split_words: # for some models, we need to force word splitting
        input_ids = tokenizer.encode(sentence.split(), add_special_tokens = True, is_split_into_words=True, return_tensors="pt")
    else:
        input_ids = tokenizer.encode(sentence, add_special_tokens = True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_embeddings = hidden_states[-1][0].cpu().numpy()
    return layer_embeddings[emb_start:emb_end]

def get_encoder_embeddings(sentence, tokenizer, model, split_words = False):
    if split_words:
        inputs = tokenizer(sentence.split(), add_special_tokens=True, is_split_into_words=True, return_tensors="pt")
    else:
        inputs = tokenizer(sentence, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(**inputs, output_hidden_states=True) # output of the encoder only
    hidden_states = encoder_outputs.hidden_states
    layer_embeddings = hidden_states[-1][0].cpu().numpy()
    return layer_embeddings

def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start = 0, emb_end = None, split_words = False, is_seq2seq = False):
    if is_seq2seq:
        layer_embs = get_encoder_embeddings(" ".join(tokens), tokenizer, model, split_words = split_words)
    else:
        layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model, split_words = split_words)
    # layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model, split_words)
    #print(f"Layer embs shape = {layer_embs.shape}")
    toks = tok_maker(tokens, sep, tokenizer, cased)
    out_ = []
    theindex = 0
    for index, word in enumerate(toks):
        if len(word) == 1:
            emb = layer_embs[theindex]#; print(emb.shape)
            theindex += 1
            out_.append(emb)
        else:
            emb = layer_embs[theindex:theindex+len(word)].mean(axis=0)#; print(emb.shape)
            theindex += len(word)
            out_.append(emb)
    print("finished")
    return out_

# xglm_small
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

xglm_small_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = None) 
                           for docnum in range(1, 13)] for lang in languages}
save(xglm_small_embeddings, "embeddings/xglm_small")

# xglm_med
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

xglm_med_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = None) 
                           for docnum in range(1, 13)] for lang in languages}
save(xglm_med_embeddings, "embeddings/xglm_med")

# xglm_large
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")

xglm_large_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = None) 
                           for docnum in range(1, 13)] for lang in languages}
save(xglm_large_embeddings, "embeddings/xglm_large")

# xglm_xl
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")

xglm_xl_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = None) 
                           for docnum in range(1, 13)] for lang in languages}
save(xglm_xl_embeddings, "embeddings/xglm_xl")

# mbert
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

bert_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "##", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(bert_embeddings, "embeddings/bert_base")

# distilmbert
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

distilmbert_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "##", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(distilmbert_embeddings, "embeddings/distilmbert")

# xlm-roberta-base
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

xlmr_base_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(xlmr_base_embeddings, "embeddings/xlmr_base")

# xlm-roberta-large
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

xlmr_large_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(xlmr_large_embeddings, "embeddings/xlmr_large")

# mT5, small
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

mt5_small_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 0, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(mt5_small_embeddings, "embeddings/mt5_small")

# mT5, base
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

mt5_base_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 0, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(mt5_base_embeddings, "embeddings/mt5_base")

# mT5, small
tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

mt5_large_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 0, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(mt5_large_embeddings, "embeddings/mt5_large")

##############
# NEW MODELS #
##############

# mdeberta
tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")

mdeberta_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(mdeberta_embeddings, "embeddings/mdeberta")

# xlm-align
tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
model = AutoModel.from_pretrained("microsoft/xlm-align-base")

xlmalign_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(xlmalign_embeddings, "embeddings/xlm_align")

# infoxlm (base)
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base")

infoxlm_base_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(infoxlm_base_embeddings, "embeddings/infoxlm_base")

# infoxlm (large)
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
model = AutoModel.from_pretrained("microsoft/infoxlm-large")

infoxlm_large_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(infoxlm_large_embeddings, "embeddings/infoxlm_large")

# multiminilm
tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

multiminilm_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], 
                                                 sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1) 
                           for docnum in range(1, 13)] for lang in languages}
save(multiminilm_embeddings, "embeddings/multiminilm")

# nllm distilled 600M
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")

nllb200_distilled_600M_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1, is_seq2seq = True) for docnum in range(1, 13)] for lang in languages}
save(nllb200_distilled_600M_embeddings, "embeddings/nllb200_distilled_600M")

# nllm distilled 1B
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")

nllb200_distilled_1B_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1, is_seq2seq = True) for docnum in range(1, 13)] for lang in languages}
save(nllb200_distilled_1B_embeddings, "embeddings/nllb200_distilled_1B")

# nllm 1B
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")

nllb200_1B_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], sep = "▁", tokenizer = tokenizer, model = model, emb_start = 1, emb_end = -1, is_seq2seq = True) for docnum in range(1, 13)] for lang in languages}
save(nllb200_1B_embeddings, "embeddings/nllb200_1B")

# mGPT
tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", add_prefix_space=True)
model = AutoModel.from_pretrained("ai-forever/mGPT")

mgpt_embeddings = {lang : [get_embeddings_tokens(out[lang][docnum]["words"], sep = "Ġ", tokenizer = tokenizer, model = model, split_words = True) for docnum in range(1, 13)] for lang in mgpt_languages}
save(mgpt_embeddings, "embeddings/mgpt")

############
# ENCODING #
############

def meco_encoding(languages, model_embeddings, modelname):
    out_predictions = []
    out_predictions_multi = []
    for language in languages:
        fp = [out[language][n]["fp"] for n in range(1, 13)] # all fp in lang
        embeddings = [model_embeddings[language][n] for n in range(0, 12)] # all embs in lang
        for i in range(12):
            X_data = embeddings[:i] + embeddings[i+1:] # exclude lang_i
            X_train = np.concatenate(X_data)
            y_train = np.concatenate(fp[:i] + fp[i+1:])
            X_test = embeddings[i]#.reshape(1, -1)
            y_test = fp[i]
            #print(f"{langs[i]} --- {y_names}")
            # scaling
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = y_scaler.transform(np.array(y_test).reshape(-1, 1)).flatten()
            # fitting
            reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
            reg.fit(X_train, y_train)
            # predict held-out story
            y_pred = reg.predict(X_test)
            r, p = pearsonr(y_test, y_pred)
            out_predictions.append([language, i, r])
            print(f"MONO {language:<5} {round(r,4)}")
            # predict held-out languages
            for language_test in languages:
                if language_test != language:
                    X_test_multi = X_scaler.transform(model_embeddings[language_test][i])
                    y_test_multi = y_scaler.transform(np.array(out[language_test][i+1]["fp"]).reshape(-1, 1)).flatten()
                    y_pred_multi = reg.predict(X_test_multi)
                    r_multi, p_multi = pearsonr(y_test_multi, y_pred_multi)
                    out_predictions_multi.append([language, language_test, r_multi])
                    print(f"MULTI - train: {language:<5} test: {language_test:<5} R = {round(r_multi,4)}")
            ############################
    out_predictions = pd.DataFrame(out_predictions, columns = ["language", "story", "r"])
    out_predictions_multi = pd.DataFrame(out_predictions_multi, columns = ["language", "language_test", "r"])
    out_predictions.to_csv(f"results/mono_{modelname}.csv")
    out_predictions_multi.to_csv(f"results/multi_{modelname}.csv")
    return out_predictions, out_predictions_multi

xglm_small_embeddings = load("embeddings/xglm_small")
xglm_med_embeddings = load("embeddings/xglm_med")
xglm_large_embeddings = load("embeddings/xglm_large")
xglm_xl_embeddings = load("embeddings/xglm_xl")
bert_embeddings = load("embeddings/bert_base")
distilmbert_embeddings = load("embeddings/distilmbert")
xlmr_base_embeddings = load("embeddings/xlmr_base")
xlmr_large_embeddings = load("embeddings/xlmr_large")
mt5_small_embeddings = load("embeddings/mt5_small")
mt5_base_embeddings = load("embeddings/mt5_base")
mt5_large_embeddings = load("embeddings/mt5_large")
mt5_large_embeddings = load("embeddings/mt5_large")

mdeberta_embeddings = load("embeddings/mdeberta")
xlm_align_embeddings = load("embeddings/xlm_align")
infoxlm_base_embeddings = load("embeddings/infoxlm_base")
infoxlm_large_embeddings = load("embeddings/infoxlm_large")
multiminilm_embeddings = load("embeddings/multiminilm")
nllb200_distilled_600M_embeddings = load("embeddings/nllb200_distilled_600M")
nllb200_distilled_1BM_embeddings = load("embeddings/nllb200_distilled_1B")
nllb200_1B_embeddings = load("embeddings/nllb200_1B")
mgpt_embeddings = load("embeddings/mgpt")



meco_encoding(languages, xglm_small_embeddings, "xglm_small")
meco_encoding(languages, xglm_med_embeddings, "xglm_med")
meco_encoding(languages, xglm_large_embeddings, "xglm_large")
meco_encoding(languages, xglm_xl_embeddings, "xglm_xl")
meco_encoding(languages, bert_embeddings, "bert_base")
meco_encoding(languages, distilmbert_embeddings, "distilmbert")
meco_encoding(languages, xlmr_base_embeddings, "xlmr_base")
meco_encoding(languages, xlmr_large_embeddings, "xlmr_large")
meco_encoding(languages, mt5_small_embeddings, "mt5_small")
meco_encoding(languages, mt5_base_embeddings, "mt5_base")
meco_encoding(languages, mt5_large_embeddings, "mt5_large")

############
# PLOTTING #
############

def plot_aggregate(df, title, ylim=None, ylimstart=None, sig=[]):
    plt.figure(figsize=(14*.7, 11.5*.7), dpi = 300)
    sns.set_context("talk")
    palette = sns.color_palette("tab10")  # Colorblind-friendly palette
    ax = sns.barplot(x='Model', y='Score', hue='Family', data=df,
                     dodge=False, palette=palette, edgecolor='.2')
    for i in range(len(df['Score'])):
        plt.errorbar(i, df['Score'][i], yerr=df['sd'][i]/sqrt(df["n"][i]), fmt='none', capsize=5, ecolor='black', capthick=2)

    # significance asterisks
    for i, value in enumerate(df['Score']):
        if i < len(sig):
            y = value + df['sd'][i]/sqrt(df["n"][i]) + 0.02
            plt.text(i, y, sig[i], ha='center', va='bottom', color='black', fontsize=20, weight='bold')

    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=23)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    ax.get_legend().remove()
    plt.show()

model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["XLM-R$_{base}$", "XLM-R$_{large}$", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "DistilmBERT", "mBERT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
model_family = ["XLM-R", "XLM-R", "mT5", "mT5", "mT5", "BERT", "BERT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [16, 16, 16, 16, 16, 16, 16, 9, 9, 9, 9] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# aggregate monolingual results
monol_results = []
for model in model_names:
    monol = pd.read_csv(f"results/mono_{model}.csv")
    m = monol.groupby("language").agg({"r" : "mean"}).mean()
    sd = monol.groupby("language").agg({"r" : "mean"}).std()
    n = len(monol["language"].unique())
    monol_results.append([model, m.item(), sd.item(), n])
monol_results = pd.DataFrame(monol_results, columns = ["Model", "Score", "sd", "n"])
monol_results["Model"] = monol_results["Model"].map(names_nice_dict)
monol_results["Family"] = monol_results["Model"].map(class_dict_nice)

plot_aggregate(monol_results, "",  ylim = .85, ylimstart = -.1,)

# aggregate multilingual results
multi_results = []
for model in model_names:
    multi = pd.read_csv(f"results/multi_{model}.csv")
    m = multi.groupby("language").agg({"r" : "mean"}).mean()
    sd = multi.groupby("language").agg({"r" : "mean"}).std()
    n = len(multi["language"].unique())
    multi_results.append([model, m.item(), sd.item(), n])
multi_results = pd.DataFrame(multi_results, columns = ["Model", "Score", "sd", "n"])
multi_results["Model"] = multi_results["Model"].map(names_nice_dict)
multi_results["Family"] = multi_results["Model"].map(class_dict_nice)

plot_aggregate(multi_results, "",  ylim = .85, ylimstart = -.1,)

###############################################
# relationship neural and behavioral encoding #
###############################################

d_rename_langs = {lang : lang for lang in languages}
d_rename_langs["it"] = "ita"


def load_neural(model_prefix):
    with open(f"../../results/sanity_check/monolingual_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

def get_best_layerwise(res_dict, colname = "m"):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    best = res_dict[idx_max][["lang", colname]]
    return best

all_dfs = []
for model in model_names:
    monol = pd.read_csv(f"results/mono_{model}.csv")
    monol = monol.groupby("language").agg({"r" : "mean"})
    monol["lang"] = monol.index.map(d_rename_langs)
    neuro = get_best_layerwise(load_neural(model))
    merged = pd.merge(monol, neuro)
    all_dfs.append(merged)
    print(f"{model:<15} {round(pearsonr(merged['m'], merged['r'])[0], 4)}")
        
all_dfs = pd.concat(all_dfs)

pearsonr(all_dfs["r"], all_dfs["m"])


# def embeddings_chunked(words, sep, tokenizer, model, n_splits = 5, overlap = 50, emb_start = 0, emb_end = None):
#         chunk_size = math.ceil(len(words) / n_splits)
#         n_splits = n_splits + math.ceil(overlap*n_splits / chunk_size) # add splits needed (lost because of overlap size)
#         chunks = []
#         for split in range(n_splits):
#             if split == 0:
#                 start = split * chunk_size
#                 end = start + chunk_size
#                 chunks.append(" ".join(words[start : end]))
#             elif split == n_splits -1:
#                 start = split * chunk_size - overlap * split
#                 end = start + chunk_size
#                 chunks.append(" ".join(words[start : ]))
#             else:
#                 start = split * chunk_size - overlap * split
#                 end = start + chunk_size
#                 chunks.append(" ".join(words[start : end]))
            
#         all_embs = [get_embeddings_tokens(tokens = chunk.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end) for chunk in chunks]
#         final_embeddings = []
        
#         for i in range(len(all_embs) - 1):
#             if i == 0:
#                 non_overlap = all_embs[i]
#                 final_embeddings.extend(non_overlap)
#             else:
#                 non_overlap = all_embs[i][overlap:]
#                 final_embeddings.extend(non_overlap)
        
#         final_embeddings.extend(all_embs[-1][overlap:])
        
#         return final_embeddings