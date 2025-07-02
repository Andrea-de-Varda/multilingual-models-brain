#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obtaining word representations from multilingual models.
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

def check_tok(lang, tokenizer):
    df = pd.read_csv("transcribed/"+lang+".csv")
    df = df[df["end"] <= 260]
    text = df["text"].str.cat(sep=' ')
    
    a = [tokenizer.tokenize(w) for w in list(text)]
    for the_a, the_b in zip(a, list(text)):
        if the_a:
            if len(the_a) == 1:
                if the_a[0][1] != the_b:
                    print(the_a, the_b)
            else:
                if the_a[1] != the_b:
                    print(the_a, the_b)

def check_special_tokens(tokenizer):
    # encode and then decode to see the effect of special tokens
    sample_text = "This is a sample text."
    encoded_input = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_input = tokenizer.decode(encoded_input)
    print(decoded_input)
    tokenized_input = tokenizer.tokenize(sample_text)
    delta = len(encoded_input) - len(tokenized_input)
    print(f"Delta = {delta}")
    
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
    if os.path.isfile(f"embeddings/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        if replace:
            for repl in special_replace:
                text = text.replace(repl[0], repl[1])
        embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model)
        save(embs, saveto+"_"+lang)
        print(f"done {lang}")
        return embs
    
# if context size < len(passage), need to split the text in chunks
# if I just split and concatenate, the information about previous chunks is lost, and the embeddings would
# not represent correctly the context. Iteratively sliding the window is too computationally intensive
# >>>> dividing text into overlapping chunks

def embeddings_chunked(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, n_splits = 5, overlap = 50):
    if os.path.isfile(f"embeddings/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        if replace:
            for repl in special_replace:
                text = text.replace(repl[0], repl[1])
        
        # splitting in overlapping chunks
        words = text.split()
        chunk_size = math.ceil(len(words) / n_splits)

        n_splits = n_splits + math.ceil(overlap*n_splits / chunk_size) # add splits needed (lost because of overlap size)

        chunks = []
        for split in range(n_splits):
            if split == 0:
                start = split * chunk_size
                end = start + chunk_size
                chunks.append(" ".join(words[start : end]))
            elif split == n_splits -1:
                start = split * chunk_size - overlap * split
                end = start + chunk_size
                chunks.append(" ".join(words[start : ]))
            else:
                start = split * chunk_size - overlap * split
                end = start + chunk_size
                chunks.append(" ".join(words[start : end]))
            
        all_embs = []
        for chunk in chunks:
            embs = get_embeddings_tokens(tokens = chunk.split(), sep = sep, tokenizer = tokenizer, model = model)
            all_embs.append(embs)
        
        final_embeddings = {k: [] for k in all_embs[0].keys()}
        
        for k in all_embs[0].keys():
            for i in range(len(all_embs) - 1):
                if i == 0:
                    non_overlap = all_embs[i][k]
                    final_embeddings[k].extend(non_overlap)
                else:
                    non_overlap = all_embs[i][k][overlap:]
                    final_embeddings[k].extend(non_overlap)
            
            # Add the non-overlapping part of the last chunk
            final_embeddings[k].extend(all_embs[-1][k][overlap:])
        
        for k in final_embeddings.keys():
            final_embeddings[k] = np.array(final_embeddings[k])
            
        save(final_embeddings, saveto+"_"+lang)
        print(f"done {lang}")
        return final_embeddings

# 2048-d, 24 layers (+ emb layer)

################
# XGLM - small #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
check_special_tokens(tokenizer) # </s> This is a sample text. - 1
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

check_tok("fr", tokenizer)

##############
# XGLM - med #
##############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
check_special_tokens(tokenizer) # </s> This is a sample text. -1
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
check_special_tokens(tokenizer) # </s> This is a sample text. -1
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
check_special_tokens(tokenizer) # </s> This is a sample text. -1
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
check_special_tokens(tokenizer) # [CLS] This is a sample text. [SEP] -2
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

ca = embeddings_chunked("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
ja = embeddings_chunked("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
en = embeddings_chunked("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
es = embeddings_chunked("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
mr = embeddings_chunked("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
af = embeddings_chunked("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
vi = embeddings_chunked("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", n_splits = 6, overlap = 20) # changed as it did not fit in context (more tokens-per-word in tamil)
lt = embeddings_chunked("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True)
tr = embeddings_chunked("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
nl = embeddings_chunked("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
no = embeddings_chunked("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
fa = embeddings_chunked("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base")
fr = embeddings_chunked("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('…', '...')], replace=True)
ro = embeddings_chunked("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True)

# check_tok("ro", tokenizer)

#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
check_special_tokens(tokenizer) # [CLS] This is a sample text. [SEP] -2
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

ca = embeddings_chunked("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
ja = embeddings_chunked("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
en = embeddings_chunked("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
es = embeddings_chunked("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
mr = embeddings_chunked("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
af = embeddings_chunked("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
vi = embeddings_chunked("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", n_splits = 6, overlap = 20) # changed as it did not fit in context (more tokens-per-word in tamil)
lt = embeddings_chunked("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True)
tr = embeddings_chunked("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
nl = embeddings_chunked("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
no = embeddings_chunked("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
fa = embeddings_chunked("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert")
fr = embeddings_chunked("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('…', '...')], replace=True)
ro = embeddings_chunked("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True)

#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
check_special_tokens(tokenizer) # <s> This is a sample text.</s> -2
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", special_replace = [('…', '...')], replace=True)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base")

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
check_special_tokens(tokenizer) # <s> This is a sample text.</s> -2
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", special_replace = [('…', '...')], replace=True)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large")

#####################################################################################################

##############
# mT5, small # (encoder only)
##############
# all languages covered

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
check_special_tokens(tokenizer) # This is a sample text.</s> -1
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
check_special_tokens(tokenizer) # This is a sample text.</s> -1
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
check_special_tokens(tokenizer) # This is a sample text.</s> -1 
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