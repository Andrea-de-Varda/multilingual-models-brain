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
from tqdm import tqdm

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

def tok_maker_alternat(a, sep, toker, cased = True): # there may be <unk> but inevitable for some tokenizers
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out

def get_word_embeddings(sentence, tokenizer, model, split_words = False):
    if split_words: # for some models, we need to force word splitting
        input_ids = tokenizer.encode(sentence.split(), is_split_into_words=True, return_tensors="pt")
    else:
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
    #input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    # sentence_embedding = outputs[0]
    # return sentence_embedding.cpu().numpy()[0][1:]
    hidden_states = outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy() for hidden_state in hidden_states]
    return layer_embeddings

def get_encoder_embeddings(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(**inputs, output_hidden_states=True) # output of the encoder only
    hidden_states = encoder_outputs.hidden_states
    layer_embeddings = [hidden_state[0].cpu().numpy() for hidden_state in hidden_states]
    return layer_embeddings

def get_embeddings_tokens(tokens, sep, tokenizer, model, cased=True, emb_start = 0, emb_end = None, is_seq2seq = False, alternative_tok = False, split_words = False):
    # layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model)
    if is_seq2seq:
        layer_embs = get_encoder_embeddings(" ".join(tokens), tokenizer, model)
    else:
        layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model, split_words = split_words)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs] # discard embeddings for special chars
    # toks = tok_maker(tokens, sep, tokenizer, cased)
    if alternative_tok:
        toks = tok_maker_alternat(tokens, sep, tokenizer, cased)
    else:
        toks = tok_maker(tokens, sep, tokenizer, cased)
    len_emb = layer_embs[0].shape[0]
    len_toks = len([t for tok in toks for t in tok])
    if len_emb != len_toks:
        raise ValueError(f"The number of embeddings ({len_emb}) does not correspond to the number of tokens ({len_toks}). Check the special characters added by the tokenizer.")
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

def embeddings(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, emb_start = 0, emb_end = None, is_seq2seq = False, alternative_tok = False, split_words = False):
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
                    
            embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, is_seq2seq = is_seq2seq, alternative_tok = alternative_tok, split_words = split_words)
            chunks[chunk_num] = embs
        
        save(chunks, "split_context/"+saveto+"_"+lang)
        print(f"done {lang}")
        return chunks
    
def embeddings_chunked(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, emb_start = 0, emb_end = None, is_seq2seq = False, alternative_tok = False, split_words = False):
    if os.path.isfile(f"embeddings/split_context/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        df = df[df["end"] <= 260]
        
        chunks_ = {}
        for chunk_num, n in tqdm(enumerate(range(0, 260, 26)), total = 10):
            start, end = n, n+26
            subset = df[(df["end"] > start) & (df["end"] < end)]
        
            text = subset["text"].str.cat(sep=' ')
            if replace:
                for repl in special_replace:
                    text = text.replace(repl[0], repl[1])
                    
            words = text.split()
            chunks = []
            # first, fill context size [[w1], [w1, w2], [w1, w2, w3], ... [w1 : w100]]
            ctx_size = len(words)
            if ctx_size > 100:
                print("Trimming ctx size to 100")
                ctx_size = 100
            for i in range(1, ctx_size):
                chunk = words[0:i]
                chunks.append(chunk)
            # then, iteratively slide the ctx window [[w2 : w101], [w3 : w102], ...]
            for i2 in range(0, len(words)-ctx_size+1):
                chunk2 = words[i2:i2+ctx_size]
                chunks.append(chunk2)
                
            all_embs = []
            for chunk in chunks: # get embeddings for all chunks
                embs = get_embeddings_tokens(tokens = chunk, sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, is_seq2seq = is_seq2seq, alternative_tok = alternative_tok, split_words = split_words)
                all_embs.append(embs)
            
            final_embeddings = {k: [] for k in all_embs[0].keys()}
            
            # take embeddings of the last word in each chunk
            for embs in all_embs:
                for layer in embs.keys():
                    relevant_emb = embs[layer][-1]
                    final_embeddings[layer].append(relevant_emb)
            
            # from list to array
            for k in final_embeddings.keys():
                final_embeddings[k] = np.array(final_embeddings[k])
            
            print(f"done {lang}")
            chunks_[chunk_num] = final_embeddings
        
        save(chunks_, "split_context/"+saveto+"_"+lang)
        print(f"done {lang}")
        return chunks_

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

es = embeddings_chunked("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)

#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

es = embeddings_chunked("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)


#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)

#####################################################################################################

##############
# mT5, small # (encoder only)
##############
# all languages covered

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", special_replace = [('…', '...')], replace=True, emb_start = 0, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)

#############
# mT5, base #
#############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", special_replace = [('…', '...')], replace=True, emb_start = 0, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)

##############
# mT5, large #
##############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", special_replace = [('…', '...')], replace=True, emb_start = 0, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)

##############################
# microsoft/mdeberta-v3-base #
##############################

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)

del model, tokenizer

#############
# XLM-Align #
#############

tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
model = AutoModel.from_pretrained("microsoft/xlm-align-base")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)

del model, tokenizer

#################
# Info-XLM base #
#################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)

del model, tokenizer

##################
# Info-XLM large #
##################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
model = AutoModel.from_pretrained("microsoft/infoxlm-large")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)

del model, tokenizer

###############
# MultiMiniLM #
###############

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)

del model, tokenizer

###########################
# NLLB-200-distilled-600M #
###########################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)

del model, tokenizer

###########################
# NLLB-200-distilled-1.3B #
###########################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)

del model, tokenizer

#################
# NLLB-200-1.3b #
#################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")

es = embeddings_chunked("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
mr = embeddings_chunked("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
af = embeddings_chunked("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
vi = embeddings_chunked("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
ta = embeddings_chunked("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
lt = embeddings_chunked("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
tr = embeddings_chunked("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
nl = embeddings_chunked("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
no = embeddings_chunked("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fa = embeddings_chunked("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fr = embeddings_chunked("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
ro = embeddings_chunked("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)

del model, tokenizer


########
# mGPT #
########

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", add_prefix_space=True)
model = AutoModel.from_pretrained("ai-forever/mGPT")

es = embeddings_chunked("es", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
mr = embeddings_chunked("mr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
af = embeddings_chunked("af", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
vi = embeddings_chunked("vi", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
ta = embeddings_chunked("ta", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
lt = embeddings_chunked("lt", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
tr = embeddings_chunked("tr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
# nl = embeddings_chunked("nl", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
#no = embeddings_chunked("no", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
fa = embeddings_chunked("fa", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
fr = embeddings_chunked("fr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, special_replace = [('…', '...')], replace=True, split_words = True)
ro = embeddings_chunked("ro", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
