import numpy as np
import pandas as pd
from os import chdir
import pickle
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances
import re
from tqdm import tqdm
import warnings
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice")
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

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
        data = [load(f"{model_prefix}_{lang}")[n] for lang in langs]
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
print(1-np.mean([dist["mean"].mean() for dist in all_dist])) # 0.5607556700706482

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
    
def load_natstor(name):
    embeddings = {}
    for story in stories:
        with open(f"additional_analyses/NaturalStories/embeddings/{name}_{story}", 'rb') as handle:
            file = pickle.load(handle)
        n_layers = max(file.keys())
        for layer in range(n_layers+1):
            embeddings_data = file[layer]
            if layer not in embeddings.keys():
                embeddings[layer] = []
            embeddings[layer].append(embeddings_data)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings

# Study 2 embeddings

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
                embeddings_data = file[layer]
                if layer not in embeddings.keys():
                    embeddings[layer] = []
                embeddings[layer].append(embeddings_data)
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
                    embeddings_data = file[layer]
                    if layer not in embeddings.keys():
                        embeddings[layer] = []
                    embeddings[layer].append(embeddings_data)
    embeddings = {k : np.vstack(v) for k, v in embeddings.items()}
    return embeddings

# compare
def compare_sim(modelname):
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

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "xglm_small", "xglm_med", "xglm_large", "xglm_xl", "mgpt"]

study2_all_sim = {}
for modelname in model_names:
    print(f"Processing {modelname}")
    try:
        dist_ = pd.read_csv(f"other/similarity_materials/sim_df/distances_{modelname}.csv")
        study2_all_sim[modelname] = dist_
    except FileNotFoundError:
        dist_ = compare_sim(modelname)
        study2_all_sim[modelname] = dist_
        dist_.to_csv(f"other/similarity_materials/sim_df/distances_{modelname}.csv", index = False)
        print("\n", dist_)

print(1-np.mean([v["study1"].mean() for v in study2_all_sim.values()]))  # 
print(1-np.mean([v["natstor"].mean() for v in study2_all_sim.values()])) # 
print(1-np.mean([v["control"].mean() for v in study2_all_sim.values()])) # 
print(1-np.mean([v["pereira"].mean() for v in study2_all_sim.values()])) # 

# with open('other/similarity_materials/study2_sim_data_all.pkl', 'wb') as handle:
#     pickle.dump(study2_all_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)

# statistical significance (repeated measures anova)

study1 = [v["study1"].mean() for v in study2_all_sim.values()]
natstor = [v["natstor"].mean() for v in study2_all_sim.values()]
control = [v["control"].mean() for v in study2_all_sim.values()]
pereira = [v["pereira"].mean() for v in study2_all_sim.values()]

data = pd.DataFrame({
    "subject": list(range(len(study1))) * 4,
    "value": study1 + natstor + control + pereira,
    "condition": ["study1"] * len(study1) + ["natstor"] * len(natstor) + 
                 ["control"] * len(control) + ["pereira"] * len(pereira)
})

rm_anova = AnovaRM(data, depvar="value", subject="subject", within=["condition"])
rm_results = rm_anova.fit()

print(rm_results.summary()) # F = 22.1986, p = 0.0000

# post-hoc t-tests
conditions = ['study1', 'natstor', 'control', 'pereira']
data_dict = {"study1": study1, "natstor": natstor, "control": control, "pereira": pereira}
results = []
for cond1, cond2 in combinations(conditions, 2):
    t_stat, p_value = ttest_rel(data_dict[cond1], data_dict[cond2])
    results.append((cond1, cond2, t_stat, p_value))
bonferroni_corrected = [(r[0], r[1], r[2], r[3], r[3] * len(results)) for r in results]
print("Pairwise comparisons with Bonferroni correction:")
for r in bonferroni_corrected:
    print(f"{r[0]} vs {r[1]}: t-stat={r[2]:.3f}, p-value={r[3]:.3f}, corrected p-value={r[4]:.3f}")
    
# Pairwise comparisons with Bonferroni correction:
# study1 vs natstor: t-stat=-3.311, p-value=0.004, corrected p-value=0.022
# study1 vs control: t-stat=-4.491, p-value=0.000, corrected p-value=0.002
# study1 vs pereira: t-stat=-5.042, p-value=0.000, corrected p-value=0.000
# natstor vs control: t-stat=-4.608, p-value=0.000, corrected p-value=0.001
# natstor vs pereira: t-stat=-5.193, p-value=0.000, corrected p-value=0.000
# control vs pereira: t-stat=-2.409, p-value=0.026, corrected p-value=0.158

#######################
# Simple word overlap #
#######################

def normalized_overlap(set1, set2): # normalized overlap (Jaccard similarity) between two sets
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

# study 1
df = pd.read_csv("transcribed/en.csv")
df = df[df["end"] <= 260]
df = df[df["text"] != " "]
text = df["text"].str.cat(sep=' ')
text = re.sub(r'[^a-zA-Z\s]', '', text)
study_1_words = set(text.lower().split())

# study 2
study_2_words = {}
for passage in ["Passage_1", "Passage_2", "Passage_3"]:
    df = pd.read_csv(f"confirmatory/transcribed/{passage}/en.csv")
    df = df[df["end"] <= 260]
    df = df[df["text"] != " "]
    text = df["text"].str.cat(sep=' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    study_2_words[passage] = set(text.lower().split())

# natstor
natstor_words = []
for story in stories:
    df = pd.read_csv("additional_analyses/NaturalStories/transcribed/"+story+".csv")
    text = df["text"].str.cat(sep=' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    natstor_words.extend(text.lower().split())
natstor_words = set(natstor_words)
    
# pereira
pereira_sentences = pd.read_csv("additional_analyses/pereira/pereira_averaged.csv")["Sentence"].tolist()
pereira_words = set([w for sent in pereira_sentences for w in re.sub(r'[^a-zA-Z\s]', '', sent).lower().split()])

# control
control = pd.read_csv("additional_analyses/control/data/brain-lang-data_participant_20230728.csv")
avg_1 = control.groupby(["sentence", "target_UID"]).agg({"response_target" : "mean", "cond" : "first", "sentence" : "first"}).reset_index(drop=True) # first average across fROIs
control_avg = avg_1.groupby("sentence").agg({"response_target" : "mean", "cond" : "first"}) 
control_sentences = control_avg.index.tolist()
control_words = set([w for sent in control_sentences for w in re.sub(r'[^a-zA-Z\s]', '', sent).lower().split()])

word_overlap = []
for passage in ["Passage_1", "Passage_2", "Passage_3"]:
    nat = normalized_overlap(study_2_words[passage], natstor_words)
    per = normalized_overlap(study_2_words[passage], pereira_words)
    con = normalized_overlap(study_2_words[passage], control_words)
    s1  = normalized_overlap(study_2_words[passage], study_1_words)
    word_overlap.append([passage, nat, per, con, s1])
    print(f"{passage:<10} - {s1:>7.2f} {nat:>7.2f} {con:>7.2f} {per:>7.2f}")
word_overlap = pd.DataFrame(word_overlap, columns = ["passage", "nat", "per", "con", "s1"])
word_overlap.mean()

# For figure: word overlap -- embedding distance/similarity