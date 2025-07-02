#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/NaturalStories")

def save(file, name):
    with open("embeddings/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def check_tok(lang, tokenizer):
    df = pd.read_csv("transcribed/"+lang+".csv")
    # df = df[df["end"] <= 260]
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
    out = []
    for w in a:
        tok = toker.tokenize(w)
        tok[0] = re.sub(sep, "", tok[0])
        out.append(tok)
    return out

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
        raise ValueError(f"The number of embeddings does not correspond to the number of tokens ({len_emb} embeddings for {len_toks} tokens). Check the special characters added by the tokenizer.")
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
        # df = df[df["end"] <= 260]
        text = df["text"].str.cat(sep=' ')
        if replace:
            for repl in special_replace:
                text = text.replace(repl[0], repl[1])
        embs = get_embeddings_tokens(tokens = text.split(), sep = sep, tokenizer = tokenizer, model = model, emb_start = emb_start, emb_end = emb_end, is_seq2seq = False, alternative_tok = False, split_words = False)
        save(embs, saveto+"_"+lang)
        print(f"done {lang}")
        return embs

def embeddings_chunked(lang, sep, tokenizer, model, saveto, special_replace=None, replace=False, ctx_size = 100, emb_start = 0, emb_end = None, is_seq2seq = False, alternative_tok = False, split_words = False):
    if os.path.isfile(f"embeddings/{saveto}_{lang}"):
        print(f"Embeddings for {lang} are already available")
    else:
        df = pd.read_csv("transcribed/"+lang+".csv")
        # df = df[df["end"] <= 260]
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
        
        print(f"done {lang} ({saveto})")
            
        save(final_embeddings, saveto+"_"+lang)
        print(f"done {lang}")
        return final_embeddings

###############################################################################
###############################################################################
###############################################################################

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

for story_n in range(1,11):
    story = embeddings(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_small", emb_start = 1, emb_end = None)

###############################################################################
    
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-1.7B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

for story_n in range(1,11):
    story = embeddings(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_med", emb_start = 1, emb_end = None)

###############################################################################
    
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-2.9B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-2.9B")

for story_n in range(1,11):
    story = embeddings(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_large", emb_start = 1, emb_end = None)

###############################################################################
    
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-4.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-4.5B")

for story_n in range(1,11):
    story = embeddings(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xglm_xl", emb_start = 1, emb_end = None)

###############################################################################

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "##", tokenizer = tokenizer, model = model, saveto = "bert_base", emb_start = 1, emb_end = -1)

###############################################################################
    
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "##", tokenizer = tokenizer, model = model, saveto = "distilmbert", emb_start = 1, emb_end = -1)

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_base", emb_start = 1, emb_end = -1)

###############################################################################


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlmr_large", emb_start = 1, emb_end = -1)

###############################################################################


tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = MT5EncoderModel.from_pretrained("google/mt5-small")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_small", emb_start = 0, emb_end = -1)

###############################################################################


tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5EncoderModel.from_pretrained("google/mt5-base")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_base", emb_start = 0, emb_end = -1)

###############################################################################


tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5EncoderModel.from_pretrained("google/mt5-large")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "mt5_large", emb_start = 0, emb_end = -1)

###############################################################################


tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "mdeberta", emb_start = 1, emb_end = -1)

###############################################################################


tokenizer = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
model = AutoModel.from_pretrained("microsoft/xlm-align-base")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "xlm_align", emb_start = 1, emb_end = -1)

###############################################################################


tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
model = AutoModel.from_pretrained("microsoft/infoxlm-base")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_base", emb_start = 1, emb_end = -1)

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
model = AutoModel.from_pretrained("microsoft/infoxlm-large")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "infoxlm_large", emb_start = 1, emb_end = -1)

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "multiminilm", emb_start = 1, emb_end = -1)

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT", add_prefix_space=True)
model = AutoModel.from_pretrained("ai-forever/mGPT")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "Ġ", tokenizer = tokenizer, model = model, saveto = "mgpt")

###############################################################################
    
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-600M")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_600M", emb_start = 1, emb_end = -1, is_seq2seq = True)

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-distilled-1.3B")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_distilled_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)

###############################################################################

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
model = AutoModel.from_pretrained("facebook/nllb-200-1.3B")

for story_n in range(1,11):
    story = embeddings_chunked(str(story_n), sep = "▁", tokenizer = tokenizer, model = model, saveto = "nllb200_1B", emb_start = 1, emb_end = -1, is_seq2seq = True)
    
    


