#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obtaining word representations from multilingual models. Clearing up context to avoid context bleed-over.
Note: I am running this in a different environment to avoid potential conflicts. 

"""
import numpy as np
import numpy.ma as ma
import pandas as pd
import re
import torch
from transformers import XGLMTokenizer, XGLMForCausalLM, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel, MT5EncoderModel, T5Tokenizer, XLMModel, XLMTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer
from os import chdir
import os
import pickle
import math

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

def embeddings(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, emb_start = 0, emb_end = None):
    if os.path.isfile(f"embeddings/split_context/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        df = df[df["end"] <= 260]
        
        chunks = {}
        for chunk_num, n in enumerate(range(0, 260, 26)):
            start, end = n, n+26
            subset = df[(df["end"] > start) & (df["end"] < end)]
        
            text = subset["text"].str.cat(sep=' ')
            if replace:
                for repl in special_replace:
                    text = text.replace(repl[0], repl[1])
                    
            embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end)
            chunks[chunk_num] = embs
        
        save(chunks, "split_context/"+saveto+"_"+lang)
        print(f"done {lang}")
        return chunks

################
# XGLM - small #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

# ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]
# no mr, af, lt, nl, no, fa, ro

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = None)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)

##############
# XGLM - med #
##############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = None)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)

################
# XGLM - large #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = None)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)

#############
# XGLM - xl #
#############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = None)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)

#####################################################################################################

##############
# mBERT base #
##############
# all languages covered

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

ca = embeddings("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
ja = embeddings("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
en = embeddings("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
es = embeddings("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
mr = embeddings("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
af = embeddings("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
vi = embeddings("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
ta = embeddings("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
lt = embeddings("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
tr = embeddings("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
nl = embeddings("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
no = embeddings("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
fa = embeddings("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
fr = embeddings("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
it = embeddings("ita", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

ca = embeddings("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
ja = embeddings("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
en = embeddings("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
es = embeddings("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
mr = embeddings("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
af = embeddings("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
vi = embeddings("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
ta = embeddings("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
lt = embeddings("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
tr = embeddings("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
nl = embeddings("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
no = embeddings("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
fa = embeddings("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
fr = embeddings("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
it = embeddings("ita", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)


#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
ita = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

#####################################################################################################

##############
# mT5, small # (encoder only)
##############
# all languages covered

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", special_replace = [('…', '...')], replace=True, emb_start = 0, emb_end = -1)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", special_replace = [('—', '-')], replace=True, emb_start = 0, emb_end = -1)

#############
# mT5, base #
#############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", special_replace = [('…', '...')], replace=True, emb_start = 0, emb_end = -1)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", special_replace = [('—', '-')], replace=True, emb_start = 0, emb_end = -1)

##############
# mT5, large #
##############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", special_replace = [('…', '...')], replace=True, emb_start = 0, emb_end = -1)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", special_replace = [('—', '-')], replace=True, emb_start = 0, emb_end = -1)
