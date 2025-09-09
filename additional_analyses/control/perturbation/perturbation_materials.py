"""
Credits: good portions of this script are adapted from https://github.com/carina-kauf/perturbed-neural-nlp/blob/master/ressources/stimuli_creation/2_create_information-loss-manipulation.ipynb
"""
import re
import os
from together import Together
import numpy as np
import random
import pickle
import csv
import subprocess
import xarray as xr
from os import chdir
import pandas as pd
import string
import nltk
from tqdm import tqdm
import difflib
nltk.download('tagsets_json')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.help.upenn_tagset()

#------------------------------------------------------------------------------
# using Greta's data only

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/control")

#########################################
# loading sentences and brain responses #
#########################################

rois = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp']
control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")

avg_1 = control[control["roi"].isin(rois)].groupby(["sentence", "target_UID"]).agg({"response_target" : "mean", "cond" : "first", "sentence" : "first"}).reset_index(drop=True) # first average across fROIs
df = avg_1.groupby("sentence").agg({"response_target" : "mean", "cond" : "first"}) # then average across participants
df = df[df["cond"] == "B"] # baseline sentences only
sentences = df.index.tolist()
y = df["response_target"].to_numpy()

# PoS tagging
def pos_tag_sentences(sentences):
    tokens = [
        [w for w in nltk.word_tokenize(s) if w not in string.punctuation]
        for s in sentences
    ]
    return [nltk.pos_tag(toks) for toks in tokens]

def delete_random_elems(input_list, seed):
    """
    helper function to randomly delete 50% of POS tags from list w/o shuffling the order
    """
    np.random.seed(seed)
    random.seed(seed)

    n = int(len(input_list)/2)
    idx_to_delete = set(random.sample(range(len(input_list)), n)) #select indices to delete
    return [x for i,x in enumerate(input_list) if not i in idx_to_delete]

def get_perturbed_datasets(sentences,perturb_type,delete50percent=False):
    """
    Input:
    * original sentence list (already lower-cased and stripped from punctuation for permute_sentences.py script)
    * perturb_type = what should stay in the stimuli file?
        nouns: only nouns
        nounsverbs: only nouns and verbs
        etc
    * whether to randomly delete 50% of lexical items or not (only used to create noun_50percent condition)
    Output:
    * list of perturbed sentences
    """
    tagged = pos_tag_sentences(sentences)
    
    n = ['NN.*', 'PRP.*'] #similar to O'Connor & Andreas (2021)
    v = ['VB.*']
    a = ['JJ.*']
    adv = ['RB.*']
    
    if perturb_type == 'nouns':
        pos_list = n
    elif perturb_type == 'verbs':
        pos_list = v
    elif perturb_type == 'nounsverbs':
        pos_list = n + v
    elif perturb_type == 'nounsverbsadj':
        pos_list = n + v + a
    elif perturb_type == 'contentwords':
        pos_list = n + v + a + adv
    elif perturb_type == 'functionwords':
        pos_list = n + v + a + adv #exclude in next step
    else:
        print("Unknown condition")
        
    perturbed_sents = []
    for ind,sent in enumerate(tagged):
        if perturb_type != "functionwords": #if some kind of content words
            
            if delete50percent == False:
                pert = ' '.join([tag_tuple[0].rstrip(".") for tag_tuple in sent if re.match("|".join(pos_list), tag_tuple[1])]) + "."
                perturbed_sents.append(pert)
            else:
                full_list = [tag_tuple[0].rstrip(".") for tag_tuple in sent if re.match("|".join(pos_list), tag_tuple[1])]
                seed = ind #to have a different seed for each pick of indices, but reproducible. Else, there's a pattern in the picks
                half_list = delete_random_elems(full_list, seed)
                perturbed_sents.append(' '.join(half_list) + ".")
                
        else:
            pert = ' '.join([tag_tuple[0].rstrip(".") for tag_tuple in sent if not re.match("|".join(pos_list), tag_tuple[1])]) + "."
            perturbed_sents.append(pert)
            
    return perturbed_sents

tagged = pos_tag_sentences(sentences)

## function test
noun_sentences = get_perturbed_datasets(sentences,perturb_type='nouns')
print(noun_sentences[:20])
print(noun_sentences[-20:])

noun50_sentences = get_perturbed_datasets(sentences,perturb_type='nouns',delete50percent=True)
fn_sentences = get_perturbed_datasets(sentences,perturb_type='functionwords')
print(fn_sentences[:10])

# loop over perturbation types to create O'Connor & Andreas (2021) datasets
perturbed_dict = {}
perturbed_dict["intact"] = sentences
perturb_types = ["contentwords", "nouns", "verbs", "nounsverbs", "nounsverbsadj", "functionwords"]
for perturb in perturb_types:
    perturbed = get_perturbed_datasets(sentences,perturb_type=perturb)
    perturbed_dict[perturb] = perturbed
    print(f"Created dataset for perturbation type: {perturb}")

######################
# adding paraphrases #
######################

key = "MASKED"
client = Together(api_key=key)

paraphrases = []
for sent in tqdm(sentences):
    prompt = f"Paraphrase this sentence: <<{sent}>>. Just produce the paraphrase."
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, # greedy
        logprobs=1, max_tokens = 1000
    )
    content = response.choices[0].message.content
    paraphrases.append(content)

# manually checked, these need to be manually edited:
# targeted fixes by index (safer than startswith)
fixes = {
    767: "The one who was seen at the murder scene?", # last option
    184: "Solely due to constitutional grounds.", # formatting, text.
    321: "She has my total admiration.", # arrow, formatting
    690: "That was quite an ordeal.", # last option
    935: "What's she needed for?",} # last option

for i, s in enumerate(paraphrases):
    if i in fixes:
        paraphrases[i] = fixes[i]

perturbed_dict["paraphrase"] = paraphrases

##############
# WORD SWAPS #
##############
# then checks for Hamming distance over word positions

def local_word_swaps(sent_list, n_swaps, seed=42, min_distance=None, max_attempts=10000):
    swapped_sents = []
    rng = random.Random(seed)
    fail_count = 0
    for s_idx, sent in tqdm(enumerate(sent_list), total = len(sent_list)):
        words = sent.strip().split()
        if len(words) < 2:
            swapped_sents.append(sent)
            continue
        original_words = words[:]
        rng.seed(seed + s_idx)
        best_words = original_words[:]
        best_distance = -1
        for attempt in range(max_attempts):
            words_copy = original_words[:]
            for _ in range(min(n_swaps, len(words_copy) - 1)):
                i = rng.randint(0, len(words_copy) - 1)
                if i == 0:
                    j = 1
                elif i == len(words_copy) - 1:
                    j = len(words_copy) - 2
                else:
                    j = i + 1 if rng.random() < 0.5 else i - 1
                words_copy[i], words_copy[j] = words_copy[j], words_copy[i]
            if min_distance is not None:
                # word-position (Hamming) distance
                distance = sum(1 for a, b in zip(original_words, words_copy) if a != b)
            else:
                distance = 999  # skip check if not needed
            if distance > best_distance:
                best_distance = distance
                best_words = words_copy[:]
            if min_distance is None or distance >= min_distance:
                break
            rng.seed(seed + s_idx + attempt + 1)
        if min_distance is not None and best_distance < min_distance:
            fail_count += 1
        swapped_sents.append(" ".join(best_words))
    if min_distance is not None:
        print(f"Unable to achieve {min_distance} swaps for {fail_count} sentences")
    return swapped_sents

perturbed_dict["1LocalWordSwap"]  = local_word_swaps(sentences, 1, min_distance=1)
perturbed_dict["2LocalWordSwap"]  = local_word_swaps(sentences, 2, min_distance=2)
perturbed_dict["3LocalWordSwaps"] = local_word_swaps(sentences, 3, min_distance=3)
perturbed_dict["4LocalWordSwaps"] = local_word_swaps(sentences, 4, min_distance=4)
perturbed_dict["5LocalWordSwaps"] = local_word_swaps(sentences, 5, min_distance=5)

def reversed_order(sent_list):
    return [" ".join(sent.strip().split()[::-1]) for sent in sent_list]

perturbed_dict["Reversed"] = reversed_order(sentences)

with open("perturbation/perturbation_dict", 'wb') as handle:
    pickle.dump(perturbed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)