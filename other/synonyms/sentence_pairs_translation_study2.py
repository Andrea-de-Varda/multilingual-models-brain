import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy.stats import rankdata, pearsonr, norm
from itertools import combinations
from os import chdir
import os
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel, MT5EncoderModel, T5Tokenizer, XLMModel, XLMTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer
from deep_translator import GoogleTranslator
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

chdir("/home/dev/Documents/PhD/Alice")

folder_path = "other/synonyms/"

##############################
# CREATE TRANSLATION DATASET #
##############################

pereira = pd.read_csv("/home/dev/Documents/PhD/Alice/additional_analyses/pereira/pereira_averaged.csv")
sentences = pereira["Sentence"].tolist()

names_languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
all_languages = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh-CN", "pl"]
    
lang_name_dict = {code : name for code, name in zip(all_languages, names_languages)}
lang_name_dict_rev = {name : code for code, name in zip(all_languages, names_languages)}

translations = {lang: [] for lang in all_languages}

# translate each sentence to each language
for lang in all_languages:
    if translations[lang] == []:
        print(f"Translating to {lang}...")
        for sent in tqdm(sentences):
            trans = GoogleTranslator(source="en", target=lang).translate(sent)
            translations[lang].append(trans)
    else:
        print(f"{lang} already done")

translations_df = pd.DataFrame(translations, index=sentences)
# translations_df.to_csv(folder_path+'translated_sentences_exp2.csv', index_label='English Sentence')
translations_df = pd.read_csv(folder_path+'translated_sentences_exp2.csv') 

##################
# GET EMBEDDINGS #
##################

def save(file, name):
    with open(f"{folder_path}embeddings_2/{name}", 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open(f"{folder_path}embeddings_2/{name}", 'rb') as handle:
        file = pickle.load(handle)
    return file
        
def get_word_embeddings(sentence, tokenizer, model):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy().mean(axis=0) for hidden_state in hidden_states]
    return {idx : emb for idx, emb in enumerate(layer_embeddings)}

def get_encoder_embeddings(sentence, tokenizer, model):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model.get_encoder()(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy().mean(axis=0) for hidden_state in hidden_states]
    return {idx : emb for idx, emb in enumerate(layer_embeddings)}

def get_all_embeddings(languages, df, tokenizer, model, is_seq2seq = False):
    d_embeddings = {}
    for l in languages:
        print(f"Processing language {l}")
        sentences = df[l].tolist()
        if is_seq2seq:
            embeddings = [get_encoder_embeddings(s, tokenizer, model) for s in tqdm(sentences)]
        else:
            embeddings = [get_word_embeddings(s, tokenizer, model) for s in tqdm(sentences)]
        d_embeddings[l] = embeddings
    return d_embeddings

def get_all_embeddings(languages, df, tokenizer, model, is_seq2seq=False, save_intermediate=False, intermediate_modelname = None):
    d_embeddings = {}
    for l in languages:
        print(f"Processing language {l}")
        sentences = df[l].tolist()
        file_path = f"{folder_path}embeddings/{l}_embeddings.pkl"
        # check if intermediate file exists and load it (in case code crashed)
        if save_intermediate and os.path.isfile(file_path):
            print(f"Loading embeddings for {l} (existing)...")
            d_embeddings[l] = load(f"{intermediate_modelname}_intermediate_{l}.pkl")
        else:
            # get embeddings if not already saved
            if is_seq2seq:
                embeddings = [get_encoder_embeddings(s, tokenizer, model) for s in tqdm(sentences)]
            else:
                embeddings = [get_word_embeddings(s, tokenizer, model) for s in tqdm(sentences)]
            d_embeddings[l] = embeddings
            
            # save embeddings if save_intermediate is True
            if save_intermediate:
                print(f"Saving embeddings for language {l}...")
                save(embeddings, f"{intermediate_modelname}_intermediate_{l}.pkl")
    return d_embeddings

all_languages = ["English Sentence", "ar", "de", "hi", "it", "ko", "pt", "ru", "zh-CN", "pl"]

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
xglm_small = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(xglm_small, "xglm_small")
del model, tokenizer

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")
xglm_med = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(xglm_med, "xglm_med")
del model, tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
bert = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(bert, "bert_base")
del model, tokenizer, bert

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
distilbert = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(distilbert, "distilmbert")
del model, tokenizer, distilbert

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")
xlmr_large = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(xlmr_large, "xlmr_large")
del model, tokenizer, xlmr_large

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
xlmr_base = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(xlmr_base, "xlmr_base")
del model, tokenizer, xlmr_base

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base")
infoxlm_base = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(infoxlm_base, "infoxlm_base")
del model, tokenizer, infoxlm_base

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
model = AutoModel.from_pretrained("microsoft/infoxlm-large")
infoxlm_large = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(infoxlm_large, "infoxlm_large")
del model, tokenizer, infoxlm_large

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
mminilm = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(mminilm, "multiminilm")
del model, tokenizer, mminilm

tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base") # TO RUN
model = AutoModel.from_pretrained("microsoft/xlm-align-base")
xlm_align = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(xlm_align, "xlm_align")
del model, tokenizer, xlm_align

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
mdeberta = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(mdeberta, "mdeberta")
del model, tokenizer, mdeberta

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")
mt5_small = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(mt5_small, "mt5_small")
del model, tokenizer, mt5_small

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")
mt5_base = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(mt5_base, "mt5_base")
del model, tokenizer, mt5_base

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")
mt5_large = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(mt5_large, "mt5_large")
del model, tokenizer, mt5_large

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")
nllb200_distilled_600M = get_all_embeddings(all_languages, translations_df, tokenizer, model, is_seq2seq = True)
save(nllb200_distilled_600M, "nllb200_distilled_600M")
del model, tokenizer, nllb200_distilled_600M

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")
nllb200_distilled_1B = get_all_embeddings(all_languages, translations_df, tokenizer, model, is_seq2seq = True)
save(nllb200_distilled_1B, "nllb200_distilled_1B")
del model, tokenizer, nllb200_distilled_1B

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
model = AutoModel.from_pretrained("ai-forever/mGPT")
mgpt = get_all_embeddings(all_languages, translations_df, tokenizer, model)
save(mgpt, "mgpt")
del model, tokenizer, mgpt

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")
xglm_large = get_all_embeddings(all_languages, translations_df, tokenizer, model, save_intermediate=True, intermediate_modelname = "xglm_large")
save(xglm_large, "xglm_large")
del model, tokenizer, xglm_large

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")
xglm_xl = get_all_embeddings(all_languages, translations_df, tokenizer, model, save_intermediate=True, intermediate_modelname = "xglm_xl")
save(xglm_xl, "xglm_xl")
del model, tokenizer, xglm_xl

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")
nllb200_1B = get_all_embeddings(all_languages, translations_df, tokenizer, model, is_seq2seq = True)
save(nllb200_1B, "nllb200_1B")
del model, tokenizer, nllb200_1B

###############################################################################
###############################################################################
###############################################################################

###########################################
# first, transfer encoding like in exp. 1 #
###########################################

def load_confirm(name):
    with open(name, 'rb') as handle:
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

def preproc_align(lang, passage, embeddings):
    df = pd.read_csv(f"confirmatory/transcribed/{passage}/{lang}.csv")
    df = df[df["end"] <= 260]
    df = df[df["text"] != " "] # !IMPORTANT! change from Exp1 analysis -- there was a whitespace in mandarin messing up the alignment between embeddings and timeseries, so removing it (and using iloc in time_words.iloc[i] bc now the indexes are not aligned - alternatively could do reset_index())
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words.iloc[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
lang_code_dict = {k : v for k, v in zip(lang_codes, languages)}
lang_code_dict_inv = {v : k for k, v in lang_code_dict.items()}

with open("confirmatory/data/dict_fMRI", 'rb') as handle:
    d = pickle.load(handle)

d_passages_keep = {'ar' : [1, 2, 3], 'de' : [2, 3], 'hi' : [2, 3], 'it' : [1, 2, 3], 'ko' : [1], 'pt' : [1, 3], 'ru' : [1, 2, 3], 'zh' : [1, 2, 3], 'pl' : [1, 2, 3]}

def test_transfer(modelname):
    n_layers = max(load_confirm(f"confirmatory/embeddings/Passage_1/{modelname}_ar").keys())
    layerwise_dict = {}
    for layernum in tqdm(range(n_layers+1), total=n_layers+1):
        # prepare data
        all_X = {}
        all_y = {}
        for lang in lang_codes:
            passages_keep = [f"Passage_{str(p)}" for p in d_passages_keep[lang]]
            X = {passage : preproc_align(lang, passage, load_confirm(f"confirmatory/embeddings/{passage}/{modelname}_{lang}")[layernum]) for passage in passages_keep}
            y = {passage : d[passage][lang_code_dict[lang]] for passage in passages_keep}
            all_X[lang] = X
            all_y[lang] = y
        # training and testing
        res = []
        for test_lang in lang_codes:
            res_passage = []
            for test_passage in all_X[test_lang].keys(): # restrict to reliable passages
                X_train = []
                y_train = []
                for train_lang in lang_codes:
                    if train_lang != test_lang: # held-out language
                        train_lang_passages = all_X[train_lang].keys()
                        if test_passage in train_lang_passages:
                            X_train.append(all_X[train_lang][test_passage])
                            y_train.append(all_y[train_lang][test_passage])
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)
                X_test = all_X[test_lang][test_passage]
                y_test = all_y[test_lang][test_passage]
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
                reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                r, p = pearsonr(y_test, y_pred)
                res_passage.append(r)
            res.append([test_lang, np.mean(res_passage)])
        res = pd.DataFrame(res, columns=["lang", "r"])
        print(f"Mean R = {res['r'].mean()}")
        layerwise_dict[layernum] = res
    return layerwise_dict

all_model_res = {}
for modelname in dict_bestlayer.keys():
    print(f"\nProcessing with {modelname}")
    res_model = test_transfer(modelname)
    all_model_res[modelname] = res_model

# with open(f"{folder_path}encoding_study2_mrr.pkl", 'wb') as handle:
#     pickle.dump(all_model_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

model_res_agg = {}
model_se = {}
model_bestlayer = {}
for model, res in all_model_res.items():
    modelres = {l : r.mean().iloc[0] for l, r in res.items()}
    best_l = max(modelres, key=modelres.get)
    model_res_agg[model] = modelres[best_l]
    model_se[model] = res[best_l]["r"].std() / np.sqrt(9)
    model_bestlayer[model] = best_l

################
# MRR analysis #
################

def cosine_distance_matrix(vectors1, vectors2):
    v1_norm = vectors1 / np.linalg.norm(vectors1, axis=1)[:, np.newaxis]
    v2_norm = vectors2 / np.linalg.norm(vectors2, axis=1)[:, np.newaxis]
    cos_sim_matrix = np.dot(v1_norm, v2_norm.T)
    return 1-cos_sim_matrix

def mean_reciprocal_rank(matching_ranks):
    return np.mean(1 / matching_ranks)

def evaluate_language_pair(language1, language2, results, layer):
    l1 = np.array([vec[layer] for vec in results[language1]])
    l2 = np.array([vec[layer] for vec in results[language2]])
    cos_sim_matrix = cosine_distance_matrix(l1, l2)
    # rank similarities across rows
    ranks = np.apply_along_axis(rankdata, 1, cos_sim_matrix, method='ordinal')
    matching_ranks = np.diag(ranks)  # diagonal (matching words) ranks
    mrr = mean_reciprocal_rank(matching_ranks) # mean reciprocal rank
    return mrr

###############################################################################

directory_path = folder_path+"embeddings_2"
modelnames = os.listdir(directory_path)
print(f"Data from {len(modelnames)} models")

########################
# multilingual results #
########################

results = []
for model in modelnames:
    res = load(model)
    layer = model_bestlayer[model] 
    print(f"Loaded {model}")
    out = []
    for l1, l2 in combinations(['ar', 'de', 'hi', 'it', 'ko', 'pt', 'ru', 'pl', 'zh-CN'], 2):
        res_ = evaluate_language_pair(l1, l2, res, layer)
        out.append(res_)
    # mean and SE for each metric across all pairs
    mrr = np.mean(out)
    mrr_se = np.std(out) / np.sqrt(9)
    results.append([model, mrr, mrr_se])
    print(f"Done {model} (mrr: {mrr})")
mrr_res = pd.DataFrame(results, columns = ["model", "mrr", "mrr_se"])
mrr_res["r"] = mrr_res["model"].map(model_res_agg)
mrr_res["encod_se"] = mrr_res["model"].map(model_se)
print(mrr_res.sort_values(by="r"))

print(pearsonr(mrr_res["mrr"], mrr_res["r"]))


############
# PLOTTING #
############

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

palette_d = {'BERT' : "steelblue",
             'DeBERTa' : "teal",
             'InfoXLM' : "firebrick",
             'NLLB' : "tomato",
             'XGLM' : "forestgreen",
             'XLM-Align' : "firebrick",
             'XLM-R' : "lightsteelblue",
             'mGPT' : "yellowgreen",
             'mT5' : "darkorange"}

#########
# MULTI #
#########

mrr_res["Family"] = mrr_res["model"].map(class_dict)
mrr_res["color"] = mrr_res["Family"].map(palette_d)

r, p = pearsonr(mrr_res["mrr"], mrr_res["r"])

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(mrr_res['mrr'], mrr_res['r'], color=mrr_res['color'], alpha=1, s = 200)

coefficients = np.polyfit(mrr_res['mrr'], mrr_res['r'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(mrr_res['mrr'])-.03, max(mrr_res['mrr'])+.03, 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, ls='--', c='gray')

for i in range(len(mrr_res)):
    plt.errorbar(mrr_res['mrr'][i], mrr_res['r'][i],
                 xerr=mrr_res['mrr_se'][i], yerr=mrr_res['encod_se'][i],
                 fmt='o', color=mrr_res['color'][i], zorder = 5)

plt.text(0.98, 0.98, f"r = {round(r, 2)}, p = {round(p, 4)}", 
         fontsize=15, ha='right', va='top', alpha=1, 
         bbox=dict(facecolor='white', alpha=0.7), 
         transform=plt.gca().transAxes)
plt.xlabel('Mean Reciprocal Rank', fontsize = 17)
plt.ylabel('R across-languages', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.xlim(0, 1)
#plt.ylim(0.18, 0.426)
#plt.yticks([0.2, 0.3, 0.4])
plt.grid(True)
plt.show()
