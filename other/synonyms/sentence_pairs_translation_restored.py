import pandas as pd
import numpy as np
from scipy.stats import rankdata, pearsonr, norm
from itertools import combinations
from os import chdir
import os
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel, MT5EncoderModel, T5Tokenizer, XLMModel, XLMTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer
from deep_translator import GoogleTranslator
from tqdm import tqdm
from time import sleep
import pickle
import matplotlib.pyplot as plt

chdir("/home/dev/Documents/PhD/Alice")

folder_path = "other/synonyms/"

##############################
# CREATE TRANSLATION DATASET #
##############################

pereira = pd.read_csv("/home/dev/Documents/PhD/Alice/additional_analyses/pereira/pereira_averaged.csv")
sentences = pereira["Sentence"].tolist()

for sentence in sentences:
    print(sentence)

names_languages = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese', 'Marathi']
all_languages = ["af", "nl", "fa", "fr", "lt", "no", "ro", "es", "ta", "tr", "vi", "mr"]
    
lang_name_dict = {code : name for code, name in zip(all_languages, names_languages)}
lang_name_dict_rev = {name : code for code, name in zip(all_languages, names_languages)}

translations = {lang: [] for lang in all_languages}

# translate each sentence to each language
for lang in all_languages:
    print(f"Translating to {lang}...")
    for sent in tqdm(sentences):
        trans = GoogleTranslator(source="en", target=lang).translate(sent)
        translations[lang].append(trans)

translations_df = pd.DataFrame(translations, index=sentences)
# translations_df.to_csv(folder_path+'translated_sentences.csv', index_label='English Sentence')
translations_df = pd.read_csv(folder_path+'translated_sentences.csv') 

##################
# GET EMBEDDINGS #
##################

def save(file, name):
    with open(f"{folder_path}embeddings/{name}", 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open(f"{folder_path}embeddings/{name}", 'rb') as handle:
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
        sleep(600)
    sleep(250)
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
        sleep(600)
    sleep(250)
    return d_embeddings


xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
xglm_small = get_all_embeddings(xglm_langs, translations_df, tokenizer, model)
save(xglm_small, "xglm_small")
del model, tokenizer

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")
xglm_med = get_all_embeddings(xglm_langs, translations_df, tokenizer, model)
save(xglm_med, "xglm_med")
del model, tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
bert = get_all_embeddings(all_languages, translations_df, translations_df, tokenizer, model)
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
mgpt = get_all_embeddings(mgpt_langs, translations_df, tokenizer, model)
save(mgpt, "mgpt")
del model, tokenizer, mgpt

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")
xglm_large = get_all_embeddings(xglm_langs, translations_df, tokenizer, model, save_intermediate=True, intermediate_modelname = "xglm_large")
save(xglm_large, "xglm_large")
del model, tokenizer, xglm_large

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")
xglm_xl = get_all_embeddings(xglm_langs, translations_df, tokenizer, model, save_intermediate=True, intermediate_modelname = "xglm_xl")
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

def load_multi(model_prefix):
    with open(f"results/multilingual_multitrain_{model_prefix}_all", 'rb') as handle:
        file = pickle.load(handle)
    return file

def load_mono(model_prefix):
    with open(f"results/monolingual_{model_prefix}_all", 'rb') as handle:
        file = pickle.load(handle)
    return file

def get_best_layerwise(res_dict, colname = "r"):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    best = mean_results[idx_max]
    best_single_langs = res_dict[idx_max]
    return idx_max, best, best_single_langs

def get_last_layerwise(res_dict, colname = "r"):
    last_layernum = max(res_dict.keys())
    last_layer_encod = res_dict[last_layernum]
    return last_layer_encod

def cosine_distance_matrix(vectors1, vectors2):
    v1_norm = vectors1 / np.linalg.norm(vectors1, axis=1)[:, np.newaxis]
    v2_norm = vectors2 / np.linalg.norm(vectors2, axis=1)[:, np.newaxis]
    cos_sim_matrix = np.dot(v1_norm, v2_norm.T)
    return 1-cos_sim_matrix

def precision_at_k(matching_ranks, k):
    return np.mean(matching_ranks <= k)

def mean_reciprocal_rank(matching_ranks):
    return np.mean(1 / matching_ranks)

def recall_at_k(matching_ranks, k):
    return np.sum(matching_ranks <= k) / len(matching_ranks)

def mean_average_precision(matching_ranks):
    precision_scores = [np.mean(matching_ranks <= rank) for rank in matching_ranks]
    return np.mean(precision_scores)

def evaluate_language_pair(language1, language2, results, layer, k=10):
    l1 = np.array([vec[layer] for vec in results[language1]])
    l2 = np.array([vec[layer] for vec in results[language2]])
    cos_sim_matrix = cosine_distance_matrix(l1, l2)
    # rank similarities across rows
    ranks = np.apply_along_axis(rankdata, 1, cos_sim_matrix, method='ordinal')
    matching_ranks = np.diag(ranks)  # diagonal (matching words) ranks
    # get retrieval metrics
    avg_rank = matching_ranks.mean()
    p_at_k = precision_at_k(matching_ranks, k) # precision @ k
    mrr = mean_reciprocal_rank(matching_ranks) # mean reciprocal rank
    r_at_k = recall_at_k(matching_ranks, k) # recall @ k
    map_score = mean_average_precision(matching_ranks) # mean avg precision
    return {
        "avg_rank": avg_rank,
        "p@k": p_at_k,
        "mrr": mrr,
        "r@k": r_at_k,
        "map": map_score
    }

###############################################################################

directory_path = folder_path+"embeddings"
modelnames = os.listdir(directory_path)
print(f"Data from {len(modelnames)} models")

########################
# multilingual results #
########################

results = []
results_single_langs = []
for model in modelnames:
    res = load(model)
    layer, encod, single_langs_encod = get_best_layerwise(load_multi(model))
    last_layer_encod = get_last_layerwise(load_multi(model))    
    rs = load_multi(model)[layer]["r"]
    se_encod = np.std(rs) / np.sqrt(len(rs))
    print(f"Loaded {model}")
    langs_ = res.keys()
    out = []
    out_langs = {lang : [] for lang in langs_}
    for l1, l2 in combinations(langs_, 2):
        res_ = evaluate_language_pair(l1, l2, res, layer)
        out.append(res_)
        for k in out_langs.keys():
            if l1 == k:
                out_langs[k].append(res_["mrr"])
            elif l2 == k:
                out_langs[k].append(res_["mrr"])
    for l in langs_:
        r_lang = single_langs_encod.loc[single_langs_encod['target_lang'] == l, 'r'].iloc[0]
        r_lang_last = last_layer_encod.loc[single_langs_encod['target_lang'] == l, 'r'].iloc[0]
        results_single_langs.append([model, l, np.mean(out_langs[l]), r_lang, r_lang_last])
    # mean and SE for each metric across all pairs
    avg_rank = np.mean([x["avg_rank"] for x in out])
    avg_rank_se = np.std([x["avg_rank"] for x in out]) / np.sqrt(len(langs_))
    p_at_k = np.mean([x["p@k"] for x in out])
    p_at_k_se = np.std([x["p@k"] for x in out]) / np.sqrt(len(langs_))
    mrr = np.mean([x["mrr"] for x in out])
    mrr_se = np.std([x["mrr"] for x in out]) / np.sqrt(len(langs_))
    r_at_k = np.mean([x["r@k"] for x in out])
    r_at_k_se = np.std([x["r@k"] for x in out]) / np.sqrt(len(langs_))
    map_score = np.mean([x["map"] for x in out])
    map_score_se = np.std([x["map"] for x in out]) / np.sqrt(len(langs_))
    results.append({
        "model": model,
        "avg_rank": avg_rank,
        "avg_rank_se": avg_rank_se,
        "p@k": p_at_k,
        "p@k_se": p_at_k_se,
        "mrr": mrr,
        "mrr_se": mrr_se,
        "r@k": r_at_k,
        "r@k_se": r_at_k_se,
        "map": map_score,
        "map_se": map_score_se,
        "encod": encod,
        "encod_se" : se_encod,
        "best_layer": layer
    })
    
    print(f"Done {model} (avg_rank: {round(avg_rank, 2)} ± {round(avg_rank_se, 2)}, "
          f"p@k: {round(p_at_k, 2)} ± {round(p_at_k_se, 2)}, "
          f"mrr: {round(mrr, 2)} ± {round(mrr_se, 2)}, "
          f"r@k: {round(r_at_k, 2)} ± {round(r_at_k_se, 2)}, "
          f"map: {round(map_score, 2)} ± {round(map_score_se, 2)}, "
          f"encod: {round(encod, 2)})")


results_df = pd.DataFrame(results)
# results_df.to_csv(folder_path+'multilingual_results.csv', index=False)
results_df = pd.read_csv(folder_path+'multilingual_results.csv')
print(results_df.corr())
print(pearsonr(results_df["mrr"], results_df["encod"]))

results_single_langs = pd.DataFrame(results_single_langs, columns = ["model", "language", "mrr", "r", "r_last"])
# results_single_langs.to_csv(folder_path+'singlelangs_multilingual_results.csv', index=False)
results_single_langs = pd.read_csv(folder_path+'singlelangs_multilingual_results.csv')
# results_single_langs.corr()
# pearsonr(results_single_langs["mrr"], results_single_langs["r"])

#######################
# monolingual results # (check) (need to do twice bc different layers)
#######################

# multilingual results 
results_mono = []
results_single_langs_mono = []
for model in modelnames:
    res = load(model)
    layer, encod, single_langs_encod = get_best_layerwise(load_mono(model), colname = "m")
    last_layer_encod = get_last_layerwise(load_mono(model))
    rs = load_mono(model)[layer]["m"]
    se_encod = np.std(rs) / np.sqrt(len(rs))
    print(f"Loaded {model}")
    langs_ = res.keys()
    out = []
    out_langs = {lang : [] for lang in langs_}
    for l1, l2 in combinations(langs_, 2):
        res_ = evaluate_language_pair(l1, l2, res, layer)
        out.append(res_)
        for k in out_langs.keys():
            if l1 == k:
                out_langs[k].append(res_["mrr"])
            elif l2 == k:
                out_langs[k].append(res_["mrr"])
    for l in langs_:
        r_lang = single_langs_encod.loc[single_langs_encod['lang'] == l, 'm'].iloc[0]
        r_lang_last = last_layer_encod.loc[single_langs_encod['lang'] == l, 'm'].iloc[0]
        results_single_langs_mono.append([model, l, np.mean(out_langs[l]), r_lang, r_lang_last])
    # mean and SE for each metric across all pairs
    avg_rank = np.mean([x["avg_rank"] for x in out])
    avg_rank_se = np.std([x["avg_rank"] for x in out]) / np.sqrt(len(langs_))
    p_at_k = np.mean([x["p@k"] for x in out])
    p_at_k_se = np.std([x["p@k"] for x in out]) / np.sqrt(len(langs_))
    mrr = np.mean([x["mrr"] for x in out])
    mrr_se = np.std([x["mrr"] for x in out]) / np.sqrt(len(langs_))
    r_at_k = np.mean([x["r@k"] for x in out])
    r_at_k_se = np.std([x["r@k"] for x in out]) / np.sqrt(len(langs_))
    map_score = np.mean([x["map"] for x in out])
    map_score_se = np.std([x["map"] for x in out]) / np.sqrt(len(langs_))
    results_mono.append({
        "model": model,
        "avg_rank": avg_rank,
        "avg_rank_se": avg_rank_se,
        "p@k": p_at_k,
        "p@k_se": p_at_k_se,
        "mrr": mrr,
        "mrr_se": mrr_se,
        "r@k": r_at_k,
        "r@k_se": r_at_k_se,
        "map": map_score,
        "map_se": map_score_se,
        "encod": encod,
        "encod_se" : se_encod,
        "best_layer": layer
    })
    
    print(f"Done {model} (avg_rank: {round(avg_rank, 2)} ± {round(avg_rank_se, 2)}, "
          f"p@k: {round(p_at_k, 2)} ± {round(p_at_k_se, 2)}, "
          f"mrr: {round(mrr, 2)} ± {round(mrr_se, 2)}, "
          f"r@k: {round(r_at_k, 2)} ± {round(r_at_k_se, 2)}, "
          f"map: {round(map_score, 2)} ± {round(map_score_se, 2)}, "
          f"encod: {round(encod, 2)})")


results_mono_df = pd.DataFrame(results_mono)
# results_mono_df.to_csv(folder_path+'monolingual_results.csv', index=False)
results_mono_df = pd.read_csv(folder_path+'monolingual_results.csv')
print(results_mono_df.corr())
print(pearsonr(results_mono_df["mrr"], results_mono_df["encod"]))

results_single_langs_mono = pd.DataFrame(results_single_langs_mono, columns = ["model", "language", "mrr", "r", "r_last"])
# results_single_langs_mono.to_csv(folder_path+'singlelangs_monolingual_results.csv', index=False)
results_single_langs_mono = pd.read_csv(folder_path+'singlelangs_monolingual_results.csv')
# results_single_langs_mono.corr()
# pearsonr(results_single_langs_mono["mrr"], results_single_langs_mono["r"])

############
# PLOTTING #
############

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]
xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

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

results_df["Family"] = results_df["model"].map(class_dict)
results_df["color"] = results_df["Family"].map(palette_d)

r, p = pearsonr(results_df['encod'], results_df['mrr']) 

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(results_df['mrr'], results_df['encod'], color=results_df['color'], alpha=1, s = 200)

coefficients = np.polyfit(results_df['mrr'], results_df['encod'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(results_df['mrr'])-.03, max(results_df['mrr'])+.03, 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, ls='--', c='gray')

for i in range(len(results_df)):
    plt.errorbar(results_df['mrr'][i], results_df['encod'][i],
                 xerr=results_df['mrr_se'][i], yerr=results_df['encod_se'][i],
                 fmt='o', color=results_df['color'][i], zorder = 5)
annotations = {
    'NLLB$_{d-small}$': (.082,-0.028),
    'NLLB$_{d-large}$': (.075,-0.015),
    'NLLB$_{large}$': (-.07,-.014),
    'XLM-Align': (-.055,.03),
    'InfoXLM$_{small}$': (-.04,-.022),
    'InfoXLM$_{large}$': (.08,-.014),
    'mMiniLM': (-.065,-.004),
    'XLM-R$_{base}$': (-0.085,0.013),
    'XLM-R$_{large}$': (-.057,.042),
    'DistilmBERT': (.08,-.018),
    'mBERT': (-.052,.007),
    'mDeBERTa': (-0.05,.008),
    'mT5$_{small}$': (.07,-.026),
    'mT5$_{base}$': (.045,-.036),
    'mT5$_{large}$': (-.065,0.011),
    'mGPT': (-.07,-0.019),
    'XGLM$_{small}$': (-.055,.039),
    'XGLM$_{med}$': (.05,-0.029),
    'XGLM$_{large}$': (.08,-0.033),
    'XGLM$_{xl}$': (.05,.008)
}

to_print = ['NLLB$_{d-small}$', 'XLM-R$_{base}$', 'XLM-R$_{large}$', 'DistilmBERT', 'mBERT', 'mT5$_{small}$', 'mT5$_{base}$', 'mT5$_{large}$', 'mGPT', 'XGLM$_{small}$', 'XGLM$_{med}$', 'XGLM$_{xl}$']

#annotations.keys()

models_with_lines = names_formatted

for i in range(len(results_df)):
    model_ = results_df['model'][i]
    model = names_nice_dict[model_]
    if model in to_print:
        offset_x, offset_y = annotations.get(model, (0.02, 0.02))
        offset_x = 1.9*offset_x
        plt.text(results_df['mrr'][i] + offset_x, results_df['encod'][i] + offset_y,
                  model, fontsize=12, ha='center', va='bottom', alpha = 0.3, bbox=dict(facecolor='white', alpha=1, zorder = 4, edgecolor='#D3D3D3'))
        if model in models_with_lines:
            plt.plot([results_df['mrr'][i], results_df['mrr'][i] + offset_x],
                      [results_df['encod'][i], results_df['encod'][i] + offset_y],
                      color='black', alpha = 0.3, lw=1, zorder = 1)
plt.text(0.98, 0.98, f"r = {round(r, 2)}, p = {round(p, 4)}", 
         fontsize=15, ha='right', va='top', alpha=1, 
         bbox=dict(facecolor='white', alpha=0.7), 
         transform=plt.gca().transAxes)
plt.xlabel('Mean Reciprocal Rank', fontsize = 17)
plt.ylabel('R across-languages', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0, 1)
plt.ylim(0.18, 0.426)
plt.yticks([0.2, 0.3, 0.4])
plt.grid(True)
plt.show()

########
# MONO #
########

results_mono_df["Family"] = results_mono_df["model"].map(class_dict)
results_mono_df["color"] = results_mono_df["Family"].map(palette_d)

r, p = pearsonr(results_mono_df['encod'], results_mono_df['mrr']) 

plt.figure(figsize=(7*.9, 9.5*.9), dpi=400)
plt.scatter(results_mono_df['mrr'], results_mono_df['encod'], color=results_mono_df['color'], alpha=1, s = 200)

coefficients = np.polyfit(results_mono_df['mrr'], results_mono_df['encod'], 1)
polynomial = np.poly1d(coefficients)
x_values = np.linspace(min(results_mono_df['mrr'])-.03, max(results_mono_df['mrr'])+.03, 100)
y_values = polynomial(x_values)

plt.plot(x_values, y_values, ls='--', c='gray')

for i in range(len(results_mono_df)):
    plt.errorbar(results_mono_df['mrr'][i], results_mono_df['encod'][i],
                 xerr=results_mono_df['mrr_se'][i], yerr=results_mono_df['encod_se'][i],
                 fmt='o', color=results_mono_df['color'][i], zorder = 5)
annotations = {
    'NLLB$_{d-small}$': (.072,-0.028),
    'NLLB$_{d-large}$': (.075,-0.015),
    'NLLB$_{large}$': (-.07,-.014),
    'XLM-Align': (-.055,.03),
    'InfoXLM$_{small}$': (-.04,-.022),
    'InfoXLM$_{large}$': (.08,-.014),
    'mMiniLM': (.055,-.025),
    'XLM-R$_{base}$': (-0.17,0.013),
    'XLM-R$_{large}$': (-.057,.042),
    'DistilmBERT': (-.07,-.052),
    'mBERT': (-.052,.007),
    'mDeBERTa': (-0.05,.008),
    'mT5$_{small}$': (-.06,-.036),
    'mT5$_{base}$': (.044,.026),
    'mT5$_{large}$': (-.05,-0.041),
    'mGPT': (-.07,0.019),
    'XGLM$_{small}$': (-.055,.039),
    'XGLM$_{med}$': (.05,0.019),
    'XGLM$_{large}$': (-.11,-0.033),
    'XGLM$_{xl}$': (-.05,.021)
}

to_print = ['NLLB$_{d-small}$', 'NLLB$_{large}$', 'mMiniLM', 'DistilmBERT', 'mBERT', 'mT5$_{small}$', 'mT5$_{base}$', 'mT5$_{large}$', 'mGPT', 'XGLM$_{small}$', 'XGLM$_{med}$', 'XGLM$_{large}$', 'XGLM$_{xl}$']

#annotations.keys()

models_with_lines = names_formatted

for i in range(len(results_mono_df)):
    model_ = results_mono_df['model'][i]
    model = names_nice_dict[model_]
    if model in to_print:
        offset_x, offset_y = annotations.get(model, (0.02, 0.02))
        offset_x = 1.9*offset_x
        plt.text(results_mono_df['mrr'][i] + offset_x, results_mono_df['encod'][i] + offset_y,
                  model, fontsize=12, ha='center', va='bottom', alpha = 0.3, bbox=dict(facecolor='white', alpha=1, zorder = 4, edgecolor='#D3D3D3'))
        if model in models_with_lines:
            plt.plot([results_mono_df['mrr'][i], results_mono_df['mrr'][i] + offset_x],
                      [results_mono_df['encod'][i], results_mono_df['encod'][i] + offset_y],
                      color='black', alpha = 0.3, lw=1, zorder = 1)
plt.text(0.025, 0.98, f"r = {round(r, 2)}, p = {round(p, 4)}", 
         fontsize=15, ha='left', va='top', alpha=1, 
         bbox=dict(facecolor='white', alpha=0.7), 
         transform=plt.gca().transAxes)
plt.xlabel('Mean Reciprocal Rank', fontsize = 17)
plt.ylabel('R across-languages', fontsize = 17)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0, 1)
plt.ylim(0.15, 0.6)
plt.yticks([0.2, 0.3, 0.4, 0.5])
plt.grid(True)
plt.show()

# check if corrs are sig different

def r_to_z(r1, r2, n = 130):    
    # fisher r-to-z transformation
    z_1 = np.arctanh(r1)
    z_2 = np.arctanh(r2)
    z_diff = z_1 - z_2
    # standard error of difference
    se_diff = np.sqrt(2*(1/(n-3)))
    z_stat = z_diff / se_diff
    p = 2 * (1 - norm.cdf(np.abs(z_stat))) # two tailed
    return z_stat, p

r_to_z(pearsonr(results_df["mrr"], results_df["encod"])[0], pearsonr(results_mono_df["mrr"], results_mono_df["encod"])[0], n = 20)