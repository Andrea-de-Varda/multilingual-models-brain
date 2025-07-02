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

    
# if context size < len(passage), need to split the text in chunks
# if I just split and concatenate, the information about previous chunks is lost, and the embeddings would
# not represent correctly the context. Iteratively sliding the window is too computationally intensive
# >>>> dividing text into overlapping chunks

def count_chunks(lang, sep, tokenizer, saveto, special_replace=None, replace=False, n_splits = 5, overlap = 50):
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
    print(lang, n_splits)


##############
# mBERT base #
##############
# all languages covered

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

ca = count_chunks("ca", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
ja = count_chunks("ja", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
en = count_chunks("en", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
es = count_chunks("es", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
mr = count_chunks("mr", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
af = count_chunks("af", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
vi = count_chunks("vi", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
ta = count_chunks("ta", sep = "##", tokenizer = tokenizer, saveto = "bert_base", n_splits = 6, overlap = 20) # changed as it did not fit in context (more tokens-per-word in tamil)
lt = count_chunks("lt", sep = "##", tokenizer = tokenizer, saveto = "bert_base", special_replace = [('–', '-')], replace=True)
tr = count_chunks("tr", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
nl = count_chunks("nl", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
no = count_chunks("no", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
fa = count_chunks("fa", sep = "##", tokenizer = tokenizer, saveto = "bert_base")
fr = count_chunks("fr", sep = "##", tokenizer = tokenizer, saveto = "bert_base", special_replace = [('…', '...')], replace=True)
ro = count_chunks("ro", sep = "##", tokenizer = tokenizer, saveto = "bert_base", special_replace = [('–', '-')], replace=True)

# check_tok("ro", tokenizer)

#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

ca = count_chunks("ca", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
ja = count_chunks("ja", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
en = count_chunks("en", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
es = count_chunks("es", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
mr = count_chunks("mr", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
af = count_chunks("af", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
vi = count_chunks("vi", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
ta = count_chunks("ta", sep = "##", tokenizer = tokenizer, saveto = "distilmbert", n_splits = 6, overlap = 20) # changed as it did not fit in context (more tokens-per-word in tamil)
lt = count_chunks("lt", sep = "##", tokenizer = tokenizer, saveto = "distilmbert", special_replace = [('–', '-')], replace=True)
tr = count_chunks("tr", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
nl = count_chunks("nl", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
no = count_chunks("no", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
fa = count_chunks("fa", sep = "##", tokenizer = tokenizer, saveto = "distilmbert")
fr = count_chunks("fr", sep = "##", tokenizer = tokenizer, saveto = "distilmbert", special_replace = [('…', '...')], replace=True)
ro = count_chunks("ro", sep = "##", tokenizer = tokenizer, saveto = "distilmbert", special_replace = [('–', '-')], replace=True)

#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

ca = count_chunks("ca", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
ja = count_chunks("ja", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
en = count_chunks("en", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
es = count_chunks("es", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
mr = count_chunks("mr", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
af = count_chunks("af", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
vi = count_chunks("vi", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
ta = count_chunks("ta", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
lt = count_chunks("lt", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
tr = count_chunks("tr", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
nl = count_chunks("nl", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
no = count_chunks("no", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
fa = count_chunks("fa", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")
fr = count_chunks("fr", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base", special_replace = [('…', '...')], replace=True)
ro = count_chunks("ro", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_base")

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

ca = count_chunks("ca", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
ja = count_chunks("ja", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
en = count_chunks("en", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
es = count_chunks("es", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
mr = count_chunks("mr", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
af = count_chunks("af", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
vi = count_chunks("vi", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
ta = count_chunks("ta", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
lt = count_chunks("lt", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
tr = count_chunks("tr", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
nl = count_chunks("nl", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
no = count_chunks("no", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
fa = count_chunks("fa", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")
fr = count_chunks("fr", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large", special_replace = [('…', '...')], replace=True)
ro = count_chunks("ro", sep = "▁", tokenizer = tokenizer, saveto = "xlmr_large")

#####################################################################################################
