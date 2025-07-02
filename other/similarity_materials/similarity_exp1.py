import numpy as np
import numpy.ma as ma
import pandas as pd
from os import chdir
import os
import pickle
from itertools import combinations
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice")
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id):
    ids = words_id.astype(int)
    time = np.arange(0, 260, 2) # fMRI data is composed by 130 images, each acquired every 2 sec (total 260 sec)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0) # average embeddings for the 2 sec time window
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv") # load transcription obtained with whisper
    df = df[df["end"] <= 260]   # remove words after 260 sec (not scanned)
    time = np.arange(0, 260, 2) # acquired every 2 sec (see embed words)
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    for i in range(len(time_words)): # find the words corresponding to each TR (pronounced between the start and the end of a TR)
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

###############################################################################


all_langs = ['Afrikaans', 'Dutch', 'Farsi', 'French', 'Lithuanian', 'Marathi', 'Norwegian', 'Romanian', 'Spanish', 'Tamil', 'Turkish', 'Vietnamese']
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}

xglm_langs  = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

def get_sim(langs, model_prefix):
    n_layers = len(load(f"{model_prefix}_fr").keys())
    layerwise = []
    for n in range(n_layers):
        data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
        lang_dist = []
        for l1, l2 in combinations(data, 2):
            cosine_dist_matrix = cosine_distances(l1, l2)
            cos = cosine_dist_matrix.mean()
            lang_dist.append(cos)
        layerwise.append([n, np.mean(lang_dist), np.std(lang_dist)])
    # mean_dist = np.mean(lang_dist)
    layerwise = pd.DataFrame(layerwise, columns = ["layer", "mean", "std"])
    print(f"Done {model_prefix}")
    return layerwise

xglm_small_multi  = get_sim(xglm_langs, "xglm_small")
xglm_med_multi    = get_sim(xglm_langs, "xglm_med")
xglm_large_multi  = get_sim(xglm_langs, "xglm_large")
xglm_xl_multi     = get_sim(xglm_langs, "xglm_xl")
mbert_multi       = get_sim(all_codes, "bert_base")
distilmbert_multi = get_sim(all_codes, "distilmbert")
xlmr_base_multi   = get_sim(all_codes, "xlmr_base")
xlmr_large_multi  = get_sim(all_codes, "xlmr_large")
mt5_small_multi   = get_sim(all_codes, "mt5_small")
mt5_base_multi    = get_sim(all_codes, "mt5_base")
mt5_large_multi   = get_sim(all_codes, "mt5_large")
mdeberta_multi      = get_sim(all_codes, "mdeberta")
xlm_align_multi     = get_sim(all_codes, "xlm_align")
infoxlm_base_multi  = get_sim(all_codes, "infoxlm_base")
infoxlm_large_multi = get_sim(all_codes, "infoxlm_large")
multiminilm_multi   = get_sim(all_codes, "multiminilm")
nllb_d_600m_multi   = get_sim(all_codes, "nllb200_distilled_600M")
nllb_d_1b_multi     = get_sim(all_codes, "nllb200_distilled_1B")
nllb_1b_multi       = get_sim(all_codes, "nllb200_1B")
mgpt_multi          = get_sim(mgpt_langs, "mgpt")


all_dist = [xglm_small_multi, xglm_med_multi, xglm_large_multi, xglm_xl_multi, mbert_multi, distilmbert_multi, xlmr_base_multi, xlmr_large_multi, mt5_small_multi, mt5_base_multi, mt5_large_multi, mdeberta_multi, xlm_align_multi, infoxlm_base_multi, infoxlm_large_multi, multiminilm_multi, nllb_d_600m_multi, nllb_d_1b_multi, nllb_1b_multi, mgpt_multi]

# avg dist -- first avg across language pairs, then layers, then models
print(np.mean([dist["mean"].mean() for dist in all_dist])) # 0.29256127991398134

########
# Exp2 #
########

def load_control(name):
    with open("additional_analyses/control/embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    embeddings = {}
    for sent in file:
        for layer, vec in sent.items():
            if layer not in embeddings.keys():
                embeddings[layer] = []
            embeddings[layer].append(vec)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings

def load_pereira(name):
    with open("additional_analyses/pereira/embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    embeddings = {}
    for sent in file:
        for layer, vec in sent.items():
            if layer not in embeddings.keys():
                embeddings[layer] = []
            embeddings[layer].append(vec)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings


# natstor 

stories = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]

story_n_dict = {"1" : "boar",
                "2" : "aqua",
                "3" : "matchstickseller",
                "4" : "kingofbirds", 
                "5" : "elvis", 
                "6" : "mrsticky",
                "7" : "highschool",
                "10" : "tree",
                "9" : "tulips"}

with open("additional_analyses/NaturalStories/response/d_shift_3", 'rb') as handle:
    d_natstor = pickle.load(handle)
    
def embed_words_natstor(embeddings, words_id, total_duration):
    ids = words_id.astype(int)
    time = np.arange(0, total_duration, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words
    
def preproc_align_natstor(lang, embeddings):
    df = pd.read_csv("additional_analyses/NaturalStories/transcribed/"+lang+".csv")
    # df = df[df["end"] <= 260]
    total_duration =  len(d_natstor[story_n_dict[lang]]) * 2 # ceil(df["end"].max())
    time = np.arange(0, total_duration, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words_natstor(embeddings, words_id, total_duration)
    return embedded_words

def load_natstor(name):
    embeddings = {}
    for story in stories:
        with open(f"additional_analyses/NaturalStories/embeddings/{name}_{story}", 'rb') as handle:
            file = pickle.load(handle)
        n_layers = max(file.keys())
        for layer in range(n_layers+1):
            fmri_data = preproc_align_natstor(story, file[layer])
            if layer not in embeddings.keys():
                embeddings[layer] = []
            embeddings[layer].append(fmri_data)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings

# Study 2 embeddings

def preproc_align_study2(lang, passage, embeddings):
    df = pd.read_csv(f"confirmatory/transcribed/{passage}/{lang}.csv")
    df = df[df["end"] <= 260]
    df = df[df["text"] != " "] # !IMPORTANT! change from Exp1 analysis -- there was a whitespace in mandarin messing up the alignment between embeddings and timeseries, so removing it (and using iloc in time_words.iloc[i] bc now the indexes are not aligned - alternatively could do reset_index())
    time = np.arange(0, 260, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words.iloc[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id)
    return embedded_words

def load_study1(name):
    embeddings = {}
    if "xglm" in name:
        langs = ["es", "vi", "ta", "tr", "fr"]
    elif name == "mgpt":
        langs = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]
    else:
        langs = ["es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]
    for language in langs:
        with open(f"embeddings/{name}_{language}", 'rb') as handle:
            file = pickle.load(handle)
            n_layers = max(file.keys())
            for layer in range(n_layers+1):
                preprocessed = preproc_align(language, file[layer])
                if layer not in embeddings.keys():
                    embeddings[layer] = []
                embeddings[layer].append(preprocessed)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings

def load_study2(name):
    embeddings = {}
    for passage in ["Passage_1", "Passage_2", "Passage_3"]:
        for language in ['ar', 'de', 'hi', 'it', 'ko', 'pt', 'ru', 'zh', 'pl']:
            with open(f"confirmatory/embeddings/{passage}/{name}_{language}", 'rb') as handle:
                file = pickle.load(handle)
                n_layers = max(file.keys())
                for layer in range(n_layers+1):
                    preprocessed = preproc_align_study2(language, passage, file[layer])
                    if layer not in embeddings.keys():
                        embeddings[layer] = []
                    embeddings[layer].append(preprocessed)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings

# compare

def compare_sim(modelname):
    print(f"Processing with {modelname}")
    study1 = load_study1(modelname)
    study2  = load_study2(modelname)
    natstor = load_natstor(modelname)
    pereira = load_pereira(modelname)
    control = load_control(modelname)
    layerwise = []
    nlayers = len(natstor.keys())
    for layer in tqdm(range(nlayers), total = nlayers, desc = "Processing layers"):
        dist_natstor = cosine_distances(natstor[layer], study2[layer]).mean()
        dist_control = cosine_distances(control[layer], study2[layer]).mean()
        dist_pereira = cosine_distances(pereira[layer], study2[layer]).mean()
        dist_study1 = cosine_distances(study1[layer], study2[layer]).mean()
        layerwise.append([layer, dist_natstor, dist_control, dist_pereira, dist_study1])
    layerwise = pd.DataFrame(layerwise, columns = ["layer", "natstor", "control", "pereira", "study1"])
    return layerwise

model_names = ["xglm_small", "xglm_med", "xglm_large", "xglm_xl", "mgpt", "nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large"]

study2_all_sim = {}
for modelname in model_names:
    if modelname not in study2_all_sim.keys():
        dist_ = compare_sim(modelname)
        study2_all_sim[modelname] = dist_
        print("\n", dist_)

print(np.mean([v["study1"].mean() for v in study2_all_sim.values()])) # 0.2983657384431873
print(np.mean([v["natstor"].mean() for v in study2_all_sim.values()])) # 0.31056206040450846
print(np.mean([v["control"].mean() for v in study2_all_sim.values()])) # 0.37275597978706837
print(np.mean([v["pereira"].mean() for v in study2_all_sim.values()])) # 0.3550368941021587

# with open('other/similarity_materials/study2_sim_data.pkl', 'wb') as handle:
#     pickle.dump(study2_all_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
