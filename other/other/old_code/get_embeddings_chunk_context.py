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
                    
            embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model)
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

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", special_replace = [('…', '...')], replace=True)

##############
# XGLM - med #
##############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", special_replace = [('…', '...')], replace=True)

################
# XGLM - large #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", special_replace = [('…', '...')], replace=True)

#############
# XGLM - xl #
#############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", special_replace = [('…', '...')], replace=True)

#####################################################################################################

##############
# mBERT base #
##############
# all languages covered

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

ca = embeddings("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
ja = embeddings("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
en = embeddings("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
es = embeddings("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
mr = embeddings("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
af = embeddings("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
vi = embeddings("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
ta = embeddings("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
lt = embeddings("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True)
tr = embeddings("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
nl = embeddings("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
no = embeddings("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
fa = embeddings("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
fr = embeddings("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True)

#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

ca = embeddings("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
ja = embeddings("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
en = embeddings("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
es = embeddings("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
mr = embeddings("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
af = embeddings("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
vi = embeddings("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
ta = embeddings("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
lt = embeddings("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True)
tr = embeddings("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
nl = embeddings("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
no = embeddings("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
fa = embeddings("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
fr = embeddings("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True)

#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")

#####################################################################################################

##############
# mT5, small # (encoder only)
##############
# all languages covered

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small")

#############
# mT5, base #
#############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base")

##############
# mT5, large #
##############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", special_replace = [('…', '...')], replace=True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large")

#####################################################################################################

# #######
# # XLM #
# #######
# # all languages covered

# tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
# model = XLMModel.from_pretrained('xlm-mlm-100-1280')

# ca = embeddings_chunked("ca", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# ja = embeddings("ja", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# en = embeddings("en", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# es = embeddings("es", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# mr = embeddings("mr", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# af = embeddings("af", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# vi = embeddings("vi", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# ta = embeddings("ta", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# lt = embeddings("lt", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# tr = embeddings("tr", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# nl = embeddings("nl", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# no = embeddings("no", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# fa = embeddings("fa", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")
# fr = embeddings("fr", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm", special_replace = [('…', '...')], replace=True)
# ro = embeddings("ro", sep = "</w>", tokenizer = tokenizer, model = model, saveto = "xlm")

#####################################################################################################

########
# mGPT #
########

# tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/mGPT")
# model = AutoModel.from_pretrained("sberbank-ai/mGPT")

# ja = embeddings("ja", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# en = embeddings("en", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# es = embeddings("es", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# mr = embeddings("mr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# af = embeddings("af", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# vi = embeddings("vi", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# lt = embeddings("lt", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# tr = embeddings("tr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# fa = embeddings("fa", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")
# fr = embeddings("fr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT", special_replace = [('…', '...')], replace=True)
# ro = embeddings("ro", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mGPT")

# bloom
# missing = ja af lt tr nl no fa ro
