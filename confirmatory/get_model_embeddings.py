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
from tqdm import tqdm
from time import sleep

chdir("/home/dev/Documents/PhD/Alice/confirmatory")

def save(file, name):
    with open("embeddings/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def check_tok(lang, passage, tokenizer):
    df = pd.read_csv(f"transcribed/{passage}/"+lang+".csv")
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
    for pass_n in ["Passage_1", "Passage_2", "Passage_3"]:
        if os.path.isfile(f"embeddings/{pass_n}/{saveto}_{lang}"):
            print(f"Embeddings for {lang} are already available")
        else:
            df = pd.read_csv(f"transcribed/{pass_n}/"+lang+".csv")
            df = df[df["end"] <= 260]
            text = df["text"].str.cat(sep=' ')
            if replace:
                for repl in special_replace:
                    text = text.replace(repl[0], repl[1])
            embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, is_seq2seq = is_seq2seq, alternative_tok = alternative_tok, split_words = split_words)
            #best_layer = dict_bestlayer[saveto]
            #embs = {best_layer : embs[best_layer]}
            save(embs, f"{pass_n}/{saveto}_{lang}")
            print(f"done {lang} - {pass_n}")
            
def embeddings_chunked(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, ctx_size = 100, emb_start = 0, emb_end = None, alternative_tok = False, split_words = False, is_seq2seq = False, overwrite = False):
    for pass_n in ["Passage_1", "Passage_2", "Passage_3"]:
        if os.path.isfile(f"embeddings/{pass_n}/{saveto}_{lang}") and overwrite == False:
            print(f"Embeddings for {lang} are already available")
        else:
            df = pd.read_csv(f"transcribed/{pass_n}/"+lang+".csv")
            df = df[df["end"] <= 260]
            text = df["text"].str.cat(sep=' ')
            if replace:
                for repl in special_replace:
                    text = text.replace(repl[0], repl[1])
            
            # splitting in overlapping chunks
            words = text.split()
            
            chunks = []
            # first, fill context size
            for i in range(1, ctx_size):
                chunk = words[0:i]
                chunks.append(chunk)
            # then, iteratively slide the ctx window
            for i2 in range(0, len(words)-ctx_size+1):
                chunk2 = words[i2:i2+ctx_size]
                chunks.append(chunk2)
                
            all_embs = []
            for chunk in tqdm(chunks):
                embs = get_embeddings_tokens(tokens = chunk, sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, is_seq2seq = is_seq2seq, alternative_tok = alternative_tok, split_words = split_words)
                all_embs.append(embs)
            
            final_embeddings = {k: [] for k in all_embs[0].keys()}
            
            # take last word
            for embs in all_embs:
                for layer in embs.keys():
                    relevant_emb = embs[layer][-1]
                    final_embeddings[layer].append(relevant_emb)
            
            # to array
            for k in final_embeddings.keys():
                final_embeddings[k] = np.array(final_embeddings[k])
                
            print(f"{saveto} - done {lang}")
            sleep(30)
            save(final_embeddings, f"{pass_n}/{saveto}_{lang}")
    sleep(120)
            

################
# XGLM - small #
################

# dict_bestlayer = {"nllb200_distilled_600M" : 1, 
#                   "nllb200_distilled_1B" : 24, 
#                   "nllb200_1B" : 23, 
#                   "xlm_align" : 6, 
#                   "infoxlm_base" : 6, 
#                   "infoxlm_large" : 15, 
#                   "multiminilm" : 7, 
#                   "xlmr_base" : 6, 
#                   "xlmr_large" : 15, 
#                   "distilmbert" : 3, 
#                   "bert_base" : 6, 
#                   "mdeberta" : 7, 
#                   "mt5_small" : 8, 
#                   "mt5_base" : 12,
#                   "mt5_large" : 20, 
#                   "mgpt" : 14, 
#                   "xglm_small" : 15, 
#                   "xglm_med" : 10, 
#                   "xglm_large" : 11, 
#                   "xglm_xl" : 45}

# ["ar", "en", "de", "hi", "it", "ko", "zh", "pl", "pt", "ru"]

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

embeddings("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)
embeddings("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)
embeddings("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)

##############
# XGLM - med #
##############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

embeddings("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)
embeddings("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)
embeddings("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)

del model, tokenizer

################
# XGLM - large #
################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")

embeddings("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)
embeddings("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)
embeddings("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)

del model, tokenizer

#############
# XGLM - xl #
#############

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")

embeddings("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)
embeddings("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)
embeddings("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None, special_replace = [('…', '...')], replace=True)

del model, tokenizer

#####################################################################################################

##############
# mBERT base #
##############
# all languages covered

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

embeddings_chunked("ar", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)


# ta = embeddings_chunked("ta", sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1) # changed as it did not fit in context (more tokens-per-word in tamil)

del model, tokenizer

#####################################################################################################

###############
# DistilmBERT #
###############
# all languages covered

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

embeddings_chunked("ar", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)

del model, tokenizer

#####################################################################################################

##############
# XLM-R base #
##############
# all languages covered

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)

del model, tokenizer

###############
# XLM-R large #
###############

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)

del model, tokenizer

#####################################################################################################

##############
# mT5, small # (encoder only)
##############
# all languages covered

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1, special_replace = [('…', '...')], replace=True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1, special_replace = [('…', '...')], replace=True)

del model, tokenizer

#############
# mT5, base #
#############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1, special_replace = [('…', '...')], replace=True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1, special_replace = [('…', '...')], replace=True)

del model, tokenizer

##############
# mT5, large #
##############

tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1, special_replace = [('…', '...')], replace=True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1, special_replace = [('…', '...')], replace=True)

##############################
# microsoft/mdeberta-v3-base #
##############################

tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True)

#############
# XLM-Align #
#############

tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
model = AutoModel.from_pretrained("microsoft/xlm-align-base")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)

#################
# Info-XLM base #
#################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)


##################
# Info-XLM large #
##################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
model = AutoModel.from_pretrained("microsoft/infoxlm-large")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)

###############
# MultiMiniLM #
###############

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, alternative_tok = True)

###########################
# NLLB-200-distilled-600M #
###########################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, is_seq2seq = True, alternative_tok = True)

###########################
# NLLB-200-distilled-1.3B #
###########################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, is_seq2seq = True, alternative_tok = True)

#################
# NLLB-200-1.3b #
#################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")

embeddings_chunked("ar", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("en", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("de", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("hi", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("it", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("ko", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("zh", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("pl", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("pt", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True, alternative_tok = True)
embeddings_chunked("ru", sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, special_replace = [('…', '...')], replace=True, is_seq2seq = True, alternative_tok = True)


########
# mGPT #
########

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", add_prefix_space=True)
model = AutoModel.from_pretrained("ai-forever/mGPT")

embeddings("ar", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("de", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("hi", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("it", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("ko", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("zh", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", special_replace = [('…', '...')], replace=True, alternative_tok = True, split_words = True)
embeddings("pl", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("pt", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", alternative_tok = True, split_words = True)
embeddings("ru", sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt", special_replace = [('…', '...')], replace=True, alternative_tok = True, split_words = True)
