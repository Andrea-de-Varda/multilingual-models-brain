import pandas as pd
import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def tok_maker(a, sep, toker, cased = True): # there may be <unk> but inevitable for some tokenizers in some languages
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out

@torch.no_grad()
def get_word_embeddings(sentence, tokenizer, model):
    if getattr(model.config, "is_encoder_decoder", False):
        # Run only the encoder to avoid decoder_* args
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        outputs = model.get_encoder()(
            **inputs, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states
    else:
        input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
    return [h[0].detach().cpu().numpy() for h in hidden_states]

def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start=0, emb_end=None):
    layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs]
    toks = tok_maker(tokens, sep, tokenizer, cased)
    len_emb = layer_embs[0].shape[0]
    len_toks = sum(len(t) for t in toks)
    if len_emb != len_toks:
        raise ValueError("Token/embedding length mismatch. Check special tokens.")
    out_dict, idx0 = {}, 0
    for li, embs in enumerate(layer_embs):
        rows, idx = [], 0
        for word in toks:
            if len(word) == 1:
                rows.append(embs[idx]); idx += 1
            else:
                seg = embs[idx:idx+len(word)]; idx += len(word)
                rows.append(np.mean(seg, axis=0))
        out_dict[li] = np.vstack(rows)
    return out_dict

# ===== DATA =====
df = pd.read_csv("pereira_averaged.csv")
y = df["EffectSize"].to_numpy()
sentences = df["Sentence"].tolist()

dict_bestlayer = {
    "nllb200_distilled_600M": 9, "nllb200_distilled_1B": 15, "nllb200_1B": 17,
    "xlm_align": 7, "infoxlm_base": 7, "infoxlm_large": 14, "multiminilm": 9,
    "xlmr_base": 9, "xlmr_large": 14, "distilmbert": 4, "bert_base": 6,
    "mdeberta": 9, "mt5_small": 2, "mt5_base": 9, "mt5_large": 17, "mgpt": 14,
    "xglm_small": 10, "xglm_med": 16, "xglm_large": 40, "xglm_xl": 48
}

MODEL_SPECS = [
    ("xlmr_large", AutoTokenizer, AutoModelForMaskedLM, "xlm-roberta-large", "▁", 1, -1),
    ("xglm_xl",   XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-4.5B", "▁", 1, None),
    ("xglm_large",XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-2.9B", "▁", 1, None),
    ("xglm_med",  XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-1.7B", "▁", 1, None),
    ("xglm_small",XGLMTokenizer, XGLMForCausalLM,       "facebook/xglm-564M", "▁", 1, None),
    ("bert_base", BertTokenizer,  BertForMaskedLM,      "bert-base-multilingual-cased", "##", 1, -1),
    ("distilmbert", DistilBertTokenizer, DistilBertModel,"distilbert-base-multilingual-cased", "##", 1, -1),
    ("xlmr_base", AutoTokenizer,  AutoModelForMaskedLM, "xlm-roberta-base", "▁", 1, -1),
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

REG_OUT  = "perturbation/registered_models"
NORM_OUT = "perturbation/registered_models/normaliz_params"
os.makedirs(REG_OUT, exist_ok=True)
os.makedirs(NORM_OUT, exist_ok=True)

with open("perturbation/perturbation_dict", 'rb') as handle:
    perturbed = pickle.load(handle)  # {perturb_type: [sentences]}

# ===== skip logic: only train missing (model_key, perturb_type) pairs =====
for model_key, Tok, Mdl, pid, sep, emb_start, emb_end in MODEL_SPECS:
    out_dir  = os.path.join(REG_OUT,  model_key)
    norm_dir = os.path.join(NORM_OUT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    missing = [p for p in perturbed.keys()
               if not (os.path.isfile(os.path.join(out_dir,  p)) and
                       os.path.isfile(os.path.join(norm_dir, p)))]

    if len(missing) == 0:
        print(f"=== {model_key}: all perturbations already trained. Skipping model.")
        continue

    print(f"\n=== {model_key} ({pid}) === | pending: {len(missing)}")
    tok = Tok.from_pretrained(pid, add_prefix_space=True) if model_key == "mgpt" else Tok.from_pretrained(pid)
    mdl = Mdl.from_pretrained(pid).to(device).eval()
    best_layer = dict_bestlayer[model_key]

    for perturb_type in missing:
        print(f">> Training encoder for perturbation: {perturb_type}")
        sents = perturbed[perturb_type]

        X_rows = []
        for sent in tqdm(sents, leave=False):
            emb_dict = get_embeddings_tokens(
                sent.split(), sep, tok, mdl, cased=True, emb_start=emb_start, emb_end=emb_end
            )
            X_rows.append(emb_dict[best_layer].mean(axis=0))
        X = np.vstack(X_rows)

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X)
        y_train = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        reg = RidgeCV(alphas=(1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1e3,1e4))
        reg.fit(X_train, y_train)

        with open(os.path.join(out_dir, perturb_type), 'wb') as h:
            pickle.dump(reg, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(norm_dir, perturb_type), 'wb') as h:
            pickle.dump([X_scaler, y_scaler], h, protocol=pickle.HIGHEST_PROTOCOL)

    del mdl  # free VRAM between models
    torch.cuda.empty_cache()
