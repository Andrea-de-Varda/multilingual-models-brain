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
    # encode and then decode to see the effect of special tokens (to remove for mapping models)
    sample_text = "This is a sample text."
    encoded_input = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_input = tokenizer.decode(encoded_input)
    print(decoded_input)
    tokenized_input = tokenizer.tokenize(sample_text)
    delta = len(encoded_input) - len(tokenized_input)
    print(f"Delta = {delta}")
    
def tok_maker_alternat(a, sep, toker, cased = True): # there may be <unk> but inevitable for some tokenizers
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out
            
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

def get_word_embeddings(sentence, tokenizer, model, split_words = False):
    if split_words: # for some models, we need to force word splitting
        input_ids = tokenizer.encode(sentence.split(), is_split_into_words=True, return_tensors="pt")
    else:
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
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
    if is_seq2seq:
        layer_embs = get_encoder_embeddings(" ".join(tokens), tokenizer, model)
    else:
        layer_embs = get_word_embeddings(" ".join(tokens), tokenizer, model, split_words = split_words)
    layer_embs = [emb[emb_start:emb_end] for emb in layer_embs] # discard embeddings for special chars
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
    if os.path.isfile(f"embeddings/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        if replace:
            for repl in special_replace:
                text = text.replace(repl[0], repl[1])
        embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, is_seq2seq = is_seq2seq, alternative_tok = alternative_tok, split_words = split_words)
        save(embs, saveto+"_"+lang)
        print(f"done {lang}")
        return embs
    
# if context size < len(passage), need to split the text in chunks
# if I just split and concatenate, the information about previous chunks is lost, and the embeddings would
# not represent correctly the context. Iteratively sliding the window is too computationally intensive
# >>>> dividing text into overlapping chunks

def embeddings_chunked(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, n_splits = 5, overlap = 50, emb_start = 0, emb_end = None, alternative_tok = False, split_words = False):
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
            embs = get_embeddings_tokens(tokens = chunk.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, alternative_tok = alternative_tok, split_words = split_words)
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

################
# XGLM - small #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
check_special_tokens(tokenizer) # </s> This is a sample text. - 1
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

check_tok("fr", tokenizer)

del model, tokenizer

##############
# XGLM - med #
##############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
check_special_tokens(tokenizer) # </s> This is a sample text. -1
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

del model, tokenizer

################
# XGLM - large #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
check_special_tokens(tokenizer) # </s> This is a sample text. -1
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

del model, tokenizer

#############
# XGLM - xl #
#############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
check_special_tokens(tokenizer) # </s> This is a sample text. -1
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

del model, tokenizer

#####################################################################################################

##############
# mBERT base #
##############
# all languages covered

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
check_special_tokens(tokenizer) # [CLS] This is a sample text. [SEP] -2
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

ca = embeddings_chunked("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
ja = embeddings_chunked("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
en = embeddings_chunked("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
es = embeddings_chunked("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", n_splits = 6, overlap = 20, emb_start = 1, emb_end = -1) # changed as it did not fit in context (more tokens-per-word in tamil)
lt = embeddings_chunked("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
it = embeddings_chunked("ita", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

del model, tokenizer


#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
check_special_tokens(tokenizer) # [CLS] This is a sample text. [SEP] -2
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

ca = embeddings_chunked("ca", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
ja = embeddings_chunked("ja", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
en = embeddings_chunked("en", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
es = embeddings_chunked("es", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
mr = embeddings_chunked("mr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
af = embeddings_chunked("af", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
vi = embeddings_chunked("vi", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", n_splits = 6, overlap = 20, emb_start = 1, emb_end = -1) # changed as it did not fit in context (more tokens-per-word in tamil)
lt = embeddings_chunked("lt", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
tr = embeddings_chunked("tr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
nl = embeddings_chunked("nl", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
no = embeddings_chunked("no", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
fa = embeddings_chunked("fa", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)
fr = embeddings_chunked("fr", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings_chunked("ro", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('–', '-')], replace=True, emb_start = 1, emb_end = -1)
it = embeddings_chunked("ita", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

del model, tokenizer

#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
check_special_tokens(tokenizer) # <s> This is a sample text.</s> -2
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)
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
it = embeddings_chunked("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

del model, tokenizer

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
check_special_tokens(tokenizer) # <s> This is a sample text.</s> -2
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)
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
it = embeddings_chunked("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

del model, tokenizer

#####################################################################################################

##############
# mT5, small # (encoder only)
##############
# all languages covered

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
check_special_tokens(tokenizer) # This is a sample text.</s> -1
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

del model, tokenizer

#############
# mT5, base #
#############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
check_special_tokens(tokenizer) # This is a sample text.</s> -1
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

del model, tokenizer

##############
# mT5, large #
##############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
check_special_tokens(tokenizer) # This is a sample text.</s> -1
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

del model, tokenizer

###############################################################################

##############
# NEW MODELS #
##############

##############################
# microsoft/mdeberta-v3-base #
##############################

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")

# ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
# ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
# en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
# it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

#############
# XLM-Align #
#############

tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("microsoft/xlm-align-base")

# ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
# ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
# en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)
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
# it = embeddings_chunked("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

#################
# Info-XLM base #
#################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("microsoft/infoxlm-base")

# ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
# ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
# en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)
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
# it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

##################
# Info-XLM large #
##################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("microsoft/infoxlm-large")

# ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
# ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
# en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)
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
# it = embeddings_chunked("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

###############
# MultiMiniLM #
###############

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

# ca = embeddings_chunked("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
# ja = embeddings_chunked("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
# en = embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)
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
# it = embeddings_chunked("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1)

###########################
# NLLB-200-distilled-600M #
###########################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")

# ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
# ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
# en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
# it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True)

###########################
# NLLB-200-distilled-1.3B #
###########################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")

# ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
# ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
# en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
# it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True)

#################
# NLLB-200-1.3b #
#################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")

# ca = embeddings("ca", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
# ja = embeddings("ja", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
# en = embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
es = embeddings("es", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
mr = embeddings("mr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
af = embeddings("af", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
vi = embeddings("vi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
ta = embeddings("ta", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
lt = embeddings("lt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
tr = embeddings("tr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
nl = embeddings("nl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
no = embeddings("no", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fa = embeddings("fa", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
fr = embeddings("fr", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", special_replace = [('…', '...')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
ro = embeddings("ro", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
# it = embeddings("ita", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", special_replace = [('—', '-')], replace=True, emb_start = 1, emb_end = -1, is_seq2seq = True)


########
# mGPT #
########

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", add_prefix_space=True)
check_special_tokens(tokenizer)
model = AutoModel.from_pretrained("ai-forever/mGPT")

# ca = embeddings("ca", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
# ja = embeddings("ja", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
# en = embeddings("en", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
es = embeddings("es", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
mr = embeddings("mr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
af = embeddings("af", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
vi = embeddings("vi", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
ta = embeddings_chunked("ta", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
lt = embeddings("lt", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
tr = embeddings("tr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
# nl = embeddings("nl", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
#no = embeddings("no", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
fa = embeddings("fa", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
fr = embeddings("fr", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, special_replace = [('…', '...')], replace=True, split_words = True)
ro = embeddings("ro", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
# it = embeddings("ita", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, special_replace = [('—', '-')], replace=True, split_words = True)



import urllib3, socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])