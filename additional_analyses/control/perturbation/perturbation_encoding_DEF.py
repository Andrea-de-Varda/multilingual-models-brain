import pandas as pd
import numpy as np
from os import chdir
import os, re, torch, pickle, warnings
from tqdm import tqdm
from transformers import (
    XGLMTokenizer, XGLMForCausalLM,
    BertTokenizer, BertForMaskedLM,
    AutoTokenizer, AutoModelForMaskedLM, AutoModel,
    MT5EncoderModel, T5Tokenizer,
    DistilBertModel, DistilBertTokenizer
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/control")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tok_maker(a, sep, toker, cased=True):
    plainseq = " ".join(a)
    b = [re.sub(sep, "", item) for item in toker.tokenize(plainseq)]
    c = []
    if cased:
        for element in a:
            tmp = []
            while "".join(tmp) != element:
                tmp.append(b.pop(0))
            c.append(tmp)
    else:
        for element in a:
            tmp = []
            while "".join(tmp) != element.lower():
                tmp.append(b.pop(0))
            c.append(tmp)
    return c

@torch.no_grad()
def get_word_embeddings(sentence, tokenizer, model):
    input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
        hs = outputs.encoder_hidden_states  # encoder-decoder: use encoder
    else:
        hs = outputs.hidden_states
    layer_embeddings = [h[0].detach().cpu().numpy() for h in hs]
    return layer_embeddings

def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start=0, emb_end=None):
    layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs]
    toks = tok_maker(tokens, sep, tokenizer, cased)
    len_emb = layer_embs[0].shape[0]
    len_toks = len([t for tok in toks for t in tok])
    if len_emb != len_toks:
        raise ValueError("The number of embeddings does not correspond to the number of tokens. Check the special characters added by the tokenizer.")
    out_dict = {}
    for idx, embs in enumerate(layer_embs):
        out, theindex = [], 0
        for word in toks:
            if len(word) == 1:
                out.append(embs[theindex]); theindex += 1
            else:
                seg = embs[theindex:theindex+len(word)]; theindex += len(word)
                out.append(np.mean(seg, axis=0))
        out_dict[idx] = np.vstack(out)
    return out_dict

# DATA
rois = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp']
control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")

avg_1 = (control[control["roi"].isin(rois)]
         .groupby(["sentence", "target_UID"])
         .agg({"response_target":"mean","cond":"first","sentence":"first"})
         .reset_index(drop=True))
df = avg_1.groupby("sentence").agg({"response_target":"mean","cond":"first"})
df = df[df["cond"] == "B"]
y = df["response_target"].to_numpy()
sentences = df.index.tolist()

# best layers to use per model
dict_bestlayer = {
    "nllb200_distilled_600M": 9, "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48
}

# (key, tok_loader, mdl_loader, pretrained_id, sep, emb_start, emb_end)
MODEL_SPECS = [
    ("xlmr_large", AutoTokenizer, AutoModelForMaskedLM, "xlm-roberta-large", "▁", 1, -1),
    ("xglm_xl",   XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-4.5B", "▁", 1, None),
    ("xglm_large",XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-2.9B", "▁", 1, None),
    ("xglm_med",  XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-1.7B", "▁", 1, None),
    ("xglm_small",XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-564M", "▁", 1, None),
    ("bert_base", BertTokenizer,  BertForMaskedLM,      "bert-base-multilingual-cased", "##", 1, -1),
    ("distilmbert", DistilBertTokenizer, DistilBertModel,"distilbert-base-multilingual-cased", "##", 1, -1),
    ("xlmr_base", AutoTokenizer,  AutoModelForMaskedLM, "xlm-roberta-base", "▁", 1, -1),
    ("xlmr_large",AutoTokenizer,  AutoModelForMaskedLM, "xlm-roberta-large", "▁", 1, -1),
    ("mt5_small", T5Tokenizer,    MT5EncoderModel,      "google/mt5-small", "▁", 0, -1),
    ("mt5_base",  T5Tokenizer,    MT5EncoderModel,      "google/mt5-base",  "▁", 0, -1),
    ("mt5_large", T5Tokenizer,    MT5EncoderModel,      "google/mt5-large", "▁", 0, -1),
    ("mdeberta",  AutoTokenizer,  AutoModel,            "microsoft/mdeberta-v3-base", "▁", 1, -1),
    ("xlm_align", AutoTokenizer,  AutoModel,            "microsoft/xlm-align-base", "▁", 1, -1),
    ("infoxlm_base", AutoTokenizer, AutoModel,          "microsoft/infoxlm-base", "▁", 1, -1),
    ("infoxlm_large",AutoTokenizer, AutoModel,          "microsoft/infoxlm-large", "▁", 1, -1),
    ("multiminilm", AutoTokenizer, AutoModel,           "microsoft/Multilingual-MiniLM-L12-H384", "▁", 1, -1),
    ("mgpt",      AutoTokenizer,  AutoModel,            "ai-forever/mGPT", "Ġ", 0, None),
    ("nllb200_distilled_600M", AutoTokenizer, AutoModel,"facebook/nllb-200-distilled-600M", "▁", 1, -1),
    ("nllb200_distilled_1B",   AutoTokenizer, AutoModel,"facebook/nllb-200-distilled-1.3B", "▁", 1, -1),
    ("nllb200_1B",             AutoTokenizer, AutoModel,"facebook/nllb-200-1.3B",           "▁", 1, -1),
]

REG_OUT = "perturbation/registered_models"
NORM_OUT = "perturbation/registered_models/normaliz_params"
os.makedirs(REG_OUT, exist_ok=True)
os.makedirs(NORM_OUT, exist_ok=True)

with open("perturbation/perturbation_dict", 'rb') as handle:
    perturbed = pickle.load(handle)  # dict: {perturb_type: [sentences]}

for model_key, Tok, Mdl, pid, sep, emb_start, emb_end in MODEL_SPECS:
    print(f"\n=== {model_key} ({pid}) ===")
    if model_key == "mgpt":
        tok = Tok.from_pretrained(pid, add_prefix_space=True)
    else:
        tok = Tok.from_pretrained(pid)
    mdl = Mdl.from_pretrained(pid).to(device).eval()

    best_layer = dict_bestlayer[model_key]

    for perturb_type, sents in perturbed.items():
        print(f">> Training encoder for perturbation: {perturb_type}")
        X_rows = []
        for sent in tqdm(sents, leave=False):
            emb_dict = get_embeddings_tokens(sent.split(), sep, tok, mdl, cased=True, emb_start=emb_start, emb_end=emb_end)
            X_rows.append(emb_dict[best_layer].mean(axis=0))
        X = np.vstack(X_rows)

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X)
        y_train = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        reg = RidgeCV(alphas=(1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1e3,1e4))
        reg.fit(X_train, y_train)

        out_dir = os.path.join(REG_OUT, model_key)
        norm_dir = os.path.join(NORM_OUT, model_key)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)

        with open(os.path.join(out_dir, perturb_type), 'wb') as h:
            pickle.dump(reg, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(norm_dir, perturb_type), 'wb') as h:
            pickle.dump([X_scaler, y_scaler], h, protocol=pickle.HIGHEST_PROTOCOL)
