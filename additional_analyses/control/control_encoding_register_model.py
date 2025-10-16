import pandas as pd
import numpy as np
from os import chdir
import re
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel, MT5EncoderModel, T5Tokenizer, DistilBertModel, DistilBertTokenizer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import warnings
from scipy.stats import pearsonr, norm

warnings.filterwarnings("ignore", message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/control")

############
# GET DATA #
############

rois = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp']
control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")

avg_1 = control[control["roi"].isin(rois)].groupby(["sentence", "target_UID"]).agg({"response_target" : "mean", "cond" : "first", "sentence" : "first"}).reset_index(drop=True) # first average across fROIs
df = avg_1.groupby("sentence").agg({"response_target" : "mean", "cond" : "first"}) # then average across participants

sentences = df.index.tolist()
y = df["response_target"].to_numpy()
is_baseline = (df["cond"] == "B").to_numpy()

# asked in review: by-fROI analyses

mask = control["roi"].isin(rois)
per_roi = (control[mask]
           .groupby(["sentence", "roi"], as_index=False)
           .agg(response_target=("response_target", "mean"),
                cond=("cond", "first")))
M = (per_roi
     .pivot(index="sentence", columns="roi", values="response_target")
     .reindex(sentences))
y_roi = {r: M[r].to_numpy() for r in rois}

# check corrs
corrs = {r: pearsonr(y_roi[r], y)[0] for r in rois}
print(corrs) # ok

###############
# reliability #
###############

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
    ci = norm.interval(0.95, loc=mean_r, scale=sd_r / np.sqrt(len(r)))  # 95% CI
    print(f"\nR (uncorrected) = {mean_r}, SD = {sd_r}, CI = {ci}")

avg_1["response_target"] = avg_1["response_target"].fillna(avg_1["response_target"].mean())
df_reliab = avg_1.groupby("sentence").agg({"response_target" : list, "cond" : "first"})
items_reliab = df_reliab[df_reliab["cond"] == "B"]["response_target"].tolist()

items_reliab = np.array(items_reliab)
splithalf(items_reliab)

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
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

embeddings = []
for sent in tqdm(sentences):
    emb = get_embeddings_tokens(sent.split(), "▁", tokenizer, model, cased=True, emb_start = 1, emb_end = -1)
    embeddings.append(emb)
# save(embeddings, "xlmr_large")
xlmr_large_embeddings = load("xlmr_large")

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

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# dict_bestlayer = {"nllb200_distilled_600M" : 9, 
#                   "nllb200_distilled_1B" : 14, 
#                   "nllb200_1B" : 15, 
#                   "xlm_align" : 7, 
#                   "infoxlm_base" : 7, 
#                   "infoxlm_large" : 14, 
#                   "multiminilm" : 9, 
#                   "xlmr_base" : 10, 
#                   "xlmr_large" : 15, 
#                   "distilmbert" : 4, 
#                   "bert_base" : 5, 
#                   "mdeberta" : 9, 
#                   "mt5_small" : 5, 
#                   "mt5_base" : 11,
#                   "mt5_large" : 14, 
#                   "mgpt" : 14, 
#                   "xglm_small" : 15, 
#                   "xglm_med" : 10, 
#                   "xglm_large" : 11, 
#                   "xglm_xl" : 45}

# NEW BEST LAYER after reviewers asked for changes!
dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 15,
                  "nllb200_1B" : 17,
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 9,
                  "xlmr_large" : 14,
                  "distilmbert" : 4, 
                  "bert_base" : 6,
                  "mdeberta" : 9, 
                  "mt5_small" : 2, 
                  "mt5_base" : 9,
                  "mt5_large" : 17, 
                  "mgpt" : 14, 
                  "xglm_small" : 10, 
                  "xglm_med" : 16, 
                  "xglm_large" : 40, 
                  "xglm_xl" : 48}

for modelname in dict_bestlayer.keys():
    print(f"Processing with {modelname.upper()}...")
    layernum = dict_bestlayer[modelname] # best layer (transfer, study I)
    embeddings = load(modelname)
    
    # refit model on entire data (exp + random), store, use w/ new data
    X = np.vstack([vec[layernum].mean(axis = 0) for vec in embeddings])
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X)[is_baseline]
    y_train = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()[is_baseline]
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X_train, y_train)

    with open(f"../../confirmatory/registered_models/control/{modelname}", 'wb') as handle:
        pickle.dump(reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f"../../confirmatory/registered_models/control/normaliz_params/{modelname}", 'wb') as handle:
        pickle.dump([X_scaler, y_scaler], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # randomized model -- three seeds (revision edit)
    for seed in [0, 1, 2, 3]:
        np.random.seed(seed)  # seed for reproducibility
        np.random.shuffle(y_train)  # shuffling y
        reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg_random.fit(X_train, y_train)
        with open(f"../../confirmatory/registered_models/control/{modelname}_random_{seed}", 'wb') as handle:
            pickle.dump(reg_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

###########
# by-fROI #
###########

roi_short = {
    'lang_LH_IFGorb': 'IFGorb',
    'lang_LH_IFG': 'IFG',
    'lang_LH_MFG': 'MFG',
    'lang_LH_AntTemp': 'AntTemp',
    'lang_LH_PostTemp': 'PostTemp',
}

for modelname in dict_bestlayer.keys():
    print(f"Processing with {modelname.upper()}...")
    layernum = dict_bestlayer[modelname]
    embeddings = load(modelname)
    X = np.vstack([vec[layernum].mean(axis=0) for vec in embeddings])
    for roi_long in rois:
        roi_label = roi_short[roi_long]
        y_vec = y_roi[roi_long]
        valid = ~np.isnan(y_vec)
        train_mask = is_baseline & valid

        X_scaler_r = StandardScaler()
        y_scaler_r = StandardScaler()

        X_train_r = X_scaler_r.fit_transform(X)[train_mask]
        y_train_r = y_scaler_r.fit_transform(y_vec.reshape(-1, 1)).flatten()[train_mask]

        reg_r = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg_r.fit(X_train_r, y_train_r)

        out_base = f"../../confirmatory/registered_models/control/{modelname}_{roi_label}"
        with open(out_base, 'wb') as handle:
            pickle.dump(reg_r, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"../../confirmatory/registered_models/control/normaliz_params/{modelname}_{roi_label}", 'wb') as handle:
            pickle.dump([X_scaler_r, y_scaler_r], handle, protocol=pickle.HIGHEST_PROTOCOL)

        for seed in [0, 1, 2, 3]:
            np.random.seed(seed)
            y_train_shuf_r = y_train_r.copy()
            np.random.shuffle(y_train_shuf_r)
            reg_random_r = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
            reg_random_r.fit(X_train_r, y_train_shuf_r)
            with open(f"{out_base}_random_{seed}", 'wb') as handle:
                pickle.dump(reg_random_r, handle, protocol=pickle.HIGHEST_PROTOCOL)
