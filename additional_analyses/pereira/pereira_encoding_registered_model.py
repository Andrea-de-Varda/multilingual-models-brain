import pandas as pd
import numpy as np
from os import chdir
import re
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel, MT5EncoderModel, T5Tokenizer, DistilBertModel, DistilBertTokenizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/pereira")

df = pd.read_parquet("Pereira.parquet.gzip", columns=['UID', 'Sentence', 'ROI', 'EffectSize'])

###################
# PREPROCESS DATA #
###################

# filter language ROIs
lang_rois = ["Lang_LH_IFGorb", "Lang_LH_IFG", "Lang_LH_MFG", "Lang_LH_AntTemp", "Lang_LH_PostTemp"]
pereira = df[df.ROI.isin(lang_rois)]
pereira['ROI'] = pereira['ROI'].cat.remove_unused_categories()

# get average sentence-level response
grouped  = pereira.groupby(["UID", "ROI", "Sentence"]).agg({"EffectSize" : "mean"}) # first avg voxels in [part, roi, sent]
grouped1 = grouped.groupby(["UID", "Sentence"]).agg({"EffectSize" : "mean"}) # then avg rois in [part, sent]
grouped2 = grouped1.groupby(["Sentence"]).agg({"EffectSize" : "mean"}) # then avg rois in sent

grouped2 = grouped2.reset_index()
grouped2["Sentence"] = grouped2["Sentence"].astype(str)
grouped2["Sentence"] = grouped2["Sentence"].str[1:-1]

# grouped2.to_csv("pereira_averaged.csv")
grouped2 = pd.read_csv("pereira_averaged.csv")

################
# Reeliability #
################

def splithalf(data, total=1000):
    r = []
    num_cols = data.shape[1]
    all_columns = np.arange(num_cols)
    for n in tqdm(range(total), total=total):
        np.random.seed(n)  # seed for reproducibility
        # split columns into two halves
        cols_random = np.random.choice(all_columns, num_cols // 2, replace=False)
        cols_not_random = np.setdiff1d(all_columns, cols_random, assume_unique=True)
        # mean for each half
        part1 = np.mean(data[:, cols_random], axis=1)
        part2 = np.mean(data[:, cols_not_random], axis=1)
        r_temp, _ = pearsonr(part1, part2)
        r.append(r_temp)
    # Compute stats for uncorrected R
    mean_r = np.mean(r)
    sd_r = np.std(r)
    print(f"\nR (uncorrected) = {mean_r}, SD = {sd_r}")
    return mean_r, sd_r

response_reliab = grouped1.groupby(["Sentence"]).agg({"EffectSize" : list})["EffectSize"].tolist()

filtered_list = [[item for item in sublist if not (isinstance(item, float) and np.isnan(item))] for sublist in response_reliab]

a = np.array([x for x in filtered_list if len(x) == 6])
b = np.array([x for x in filtered_list if len(x) == 9])

r_a, sd_a = splithalf(a)
r_b, sd_b = splithalf(b)

weighted = ((r_a * a.shape[0]) + (r_b * b.shape[0])) / (a.shape[0] + b.shape[0]) # 0.3483680578920199


############
# GET DATA #
############

y = grouped2["EffectSize"].to_numpy()
sentences = grouped2["Sentence"].tolist()

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


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "xlmr_large")

###############################################################################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = None)
    embeddings.append(emb)
save(embeddings, "xglm_xl")

###############################################################################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = None)
    embeddings.append(emb)
save(embeddings, "xglm_large")

###############################################################################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = None)
    embeddings.append(emb)
save(embeddings, "xglm_med")

###############################################################################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = None)
    embeddings.append(emb)
save(embeddings, "xglm_small")

###############################################################################

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "##", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "bert_base")

###############################################################################

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "##", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "distilmbert")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "xlmr_base")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "xlmr_large")

###############################################################################

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 0, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "mt5_small")

###############################################################################

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 0, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "mt5_base")

###############################################################################

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 0, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "mt5_large")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "mdeberta")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
model = AutoModel.from_pretrained("microsoft/xlm-align-base")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "xlm_align")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "infoxlm_base")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
model = AutoModel.from_pretrained("microsoft/infoxlm-large")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "infoxlm_large")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
save(embeddings, "multiminilm")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", add_prefix_space=True)
model = AutoModel.from_pretrained("ai-forever/mGPT")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "Ġ", tokenizer, model, cased=True)
    embeddings.append(emb)
save(embeddings, "mgpt")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1, is_seq2seq = True)
    embeddings.append(emb)
save(embeddings, "nllb200_distilled_600M")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1, is_seq2seq = True)
    embeddings.append(emb)
save(embeddings, "nllb200_distilled_1B")

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1, is_seq2seq = True)
    embeddings.append(emb)
save(embeddings, "nllb200_1B")

###############################################################################
###############################################################################
###############################################################################
###############################################################################

dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 14, 
                  "nllb200_1B" : 15, 
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 10, 
                  "xlmr_large" : 15, 
                  "distilmbert" : 4, 
                  "bert_base" : 5, 
                  "mdeberta" : 9, 
                  "mt5_small" : 5, 
                  "mt5_base" : 11,
                  "mt5_large" : 14, 
                  "mgpt" : 14, 
                  "xglm_small" : 15, 
                  "xglm_med" : 10, 
                  "xglm_large" : 11, 
                  "xglm_xl" : 45}

for modelname in dict_bestlayer.keys():
    print(f"Processing with {modelname.upper()}...")
    layernum = dict_bestlayer[modelname] # best layer (transfer, study I)
    embeddings = load(modelname)
    
    # refit model on entire data (exp + random), store, use w/ new data
    X = np.vstack([vec[layernum].mean(axis = 0) for vec in embeddings])
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X)
    y_train = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X_train, y_train)

    with open(f"../../confirmatory/registered_models/pereira/{modelname}", 'wb') as handle:
        pickle.dump(reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f"../../confirmatory/registered_models/pereira/normaliz_params/{modelname}", 'wb') as handle:
        pickle.dump([X_scaler, y_scaler], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # randomized model
    np.random.seed(0)  # seed for reproducibility
    np.random.shuffle(y_train)  # shuffling y
    reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg_random.fit(X_train, y_train)

    with open(f"../../confirmatory/registered_models/pereira/{modelname}_random", 'wb') as handle:
        pickle.dump(reg_random, handle, protocol=pickle.HIGHEST_PROTOCOL)