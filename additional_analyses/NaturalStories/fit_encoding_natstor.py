import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
from math import sqrt, ceil
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/NaturalStories")

def save(file, name):
    with open(name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("embeddings/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def embed_words(embeddings, words_id, total_duration):
    ids = words_id.astype(int)
    time = np.arange(0, total_duration, 2)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    emb_words = imputate_na(emb_words)
    return emb_words

def preproc_align(lang, embeddings):
    df = pd.read_csv("transcribed/"+lang+".csv")
    # df = df[df["end"] <= 260]
    total_duration =  len(d[story_n_dict[lang]]) * 2 # ceil(df["end"].max())
    time = np.arange(0, total_duration, 2) # sampled each 2 sec
    time_words = df["end"]
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to; then I'll need to aggregate representations
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
    embedded_words = embed_words(embeddings, words_id, total_duration)
    return embedded_words


def cross_story_encoding(langs, model_prefix, n_layers, d, prefix = "", overwrite = True):
    layerwise_dict = {}
    if os.path.isfile(f"results/cross_story_{prefix}{model_prefix}") and overwrite == False:
        print(f"Encoding for {model_prefix} already done")
    else:
        for n in range(n_layers+1):
            print(f"Processing layer {n}")
            fmri_data = [preproc_align(lang, load(f"{model_prefix}_{lang}")[n]) for lang in langs]
            ############################
            out_predictions = []
            for i in tqdm(range(len(langs))):
                X_data = fmri_data[:i] + fmri_data[i+1:] # exclude lang_i
                X_train = np.concatenate(X_data)
                y_names = langs[:i] + langs[i+1:]
                y_train = np.concatenate([d[story_n_dict[name]] for name in y_names])
                X_test = fmri_data[i]#.reshape(1, -1)
                y_test = d[story_n_dict[langs[i]]]
                #print(f"{langs[i]} --- {y_names}")
                # scaling
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
                # fitting
                reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                r, p = pearsonr(y_test, y_pred)
                print(langs[i], r)
                out_predictions.append([langs[i], r])
            ############################
            out_predictions = pd.DataFrame(out_predictions, columns = ["lang", "r"])
            layerwise_dict[n] = out_predictions
            mean_r = np.mean(out_predictions["r"])
            print(f"Mean r = {mean_r} ({model_prefix} - {n})")
        save(layerwise_dict, f"results/cross_story_{prefix}{model_prefix}")
    return layerwise_dict
    
###############################################################################

# load fMRI data
with open("response/d_shift_0", 'rb') as handle:
    d = pickle.load(handle)
    
with open("response/d_shift_1", 'rb') as handle:
    d1 = pickle.load(handle)
    
with open("response/d_shift_2", 'rb') as handle:
    d2 = pickle.load(handle)
    
with open("response/d_shift_3", 'rb') as handle:
    d3 = pickle.load(handle)
    
with open("response/d_shift_4", 'rb') as handle:
    d4 = pickle.load(handle)
    
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

###################
# evaluate shifts #
###################

xglm_small0 = cross_story_encoding(stories, "xglm_small", 24, d, prefix = "0shift_")
xglm_small1 = cross_story_encoding(stories, "xglm_small", 24, d1, prefix = "1shift_")
xglm_small2 = cross_story_encoding(stories, "xglm_small", 24, d2, prefix = "2shift_")
xglm_small3 = cross_story_encoding(stories, "xglm_small", 24, d3, prefix = "3shift_")
xglm_small4 = cross_story_encoding(stories, "xglm_small", 24, d4, prefix = "4shift_")

############
# PLOTTING #
############

def load_res(model_prefix):
    with open(f"results/cross_story_{model_prefix}", 'rb') as handle:
        file = pickle.load(handle)
    return file

colname = "r"
out_dfs = []
all_deltas = ["0shift_xglm_small", "1shift_xglm_small", "2shift_xglm_small", "3shift_xglm_small", "4shift_xglm_small"]
for modelname in all_deltas:
    results_dict = load_res(modelname)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), modelname, modelname] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)
model_colors = {model: color for model, color in zip(["0shift_xglm_small", "1shift_xglm_small", "2shift_xglm_small", "3shift_xglm_small", "4shift_xglm_small"], sns.color_palette("tab10", n_colors=5))}
model_markers = {model: marker for model, marker in zip(all_deltas, ['>', 'v', '^', 's', 'o'])}
out_dfs['color'] = out_dfs['Class'].map(model_colors)
out_dfs['marker'] = out_dfs['Model'].map(model_markers)


sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi=300)
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("Blues", n_colors=n_classes)
class_to_index = {cls: idx for idx, cls in enumerate(sorted(out_dfs['Class'].unique()))}
for cls in out_dfs['Class'].unique():
    subset = out_dfs[out_dfs['Class'] == cls]
    ax = sns.lineplot(
        data=subset,
        x='l', 
        y='m', 
        label=str(cls), 
        markers=True,
        dashes=False,
        linewidth=4,
        color=palette[class_to_index[cls]] 
    )
    last_x = subset['l'].iloc[-1]
    last_y = subset['m'].iloc[-1]
    nice_name = f"Shift = {str(cls)[0]}"
    ax.text(last_x+.03, last_y, nice_name, color="black", fontsize=18, weight='bold') # color = palette[class_to_index[cls]]
ax.set_xlabel('Layer position', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
#ax.set_title('NatStor encoding by shift', fontsize=30, weight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim([out_dfs['l'].min()-.02, out_dfs['l'].max()+.02])
ax.set_ylim(-.05, .5)
plt.legend().remove()
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################

####################################
# Save models to use with new data #
####################################

# NEW BEST LAYER after reviewers asked for changes!
dict_bestlayer = {"nllb200_distilled_600M" : 9, 
                  "nllb200_distilled_1B" : 15,
                  "nllb200_1B" : 17,
                  "xlm_align" : 7, 
                  "infoxlm_base" : 7, 
                  "infoxlm_large" : 14, 
                  "multiminilm" : 9, 
                  "xlmr_base" : 9,
                  "xlmr_large" : 14,
                  "distilmbert" : 4, 
                  "bert_base" : 6,
                  "mdeberta" : 9, 
                  "mt5_small" : 2, 
                  "mt5_base" : 9,
                  "mt5_large" : 17, 
                  "mgpt" : 14, 
                  "xglm_small" : 10, 
                  "xglm_med" : 16, 
                  "xglm_large" : 40, 
                  "xglm_xl" : 48}

for modelname in dict_bestlayer.keys():
    print(f"Processing with {modelname.upper()}...")
    layernum = dict_bestlayer[modelname] # best layer (transfer, study I)
    fmri_data = [preproc_align(story, load(f"{modelname}_{story}")[layernum]) for story in stories]
    
    X_train = np.concatenate(fmri_data)
    y_train = np.concatenate([d3[story_n_dict[story]] for story in stories])
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X_train, y_train)

    with open(f"../../confirmatory/registered_models/natstor/{modelname}", 'wb') as handle:
        pickle.dump(reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f"../../confirmatory/registered_models/natstor/normaliz_params/{modelname}", 'wb') as handle:
        pickle.dump([X_scaler, y_scaler], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # randomized model
    for random_idx, shift_val in enumerate([26, 52, 78, 104]):
        y_train_random = np.roll(y_train, shift_val)
        reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg_random.fit(X_train, y_train_random)
    
        with open(f"../../confirmatory/registered_models/natstor/{modelname}_random_{random_idx}", 'wb') as handle:
            pickle.dump(reg_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

#######################
# fROI-level encoding #
#######################

roi_short = {
    "Lang_LH_IFGorb": "IFGorb",
    "Lang_LH_IFG": "IFG",
    "Lang_LH_MFG": "MFG",
    "Lang_LH_AntTemp": "AntTemp",
    "Lang_LH_PostTemp": "PostTemp",
}

# load per-ROI shifted responses
d3_roi = {}
for roi_long, short in roi_short.items():
    with open(f"response/d_shift_3_{short}", "rb") as handle:
        d3_roi[roi_long] = pickle.load(handle)

for modelname in dict_bestlayer.keys():
    print(f"Processing (per-ROI) with {modelname.upper()}...")
    layernum = dict_bestlayer[modelname]
    fmri_data = [preproc_align(story, load(f"{modelname}_{story}")[layernum]) for story in stories]
    X_full = np.concatenate(fmri_data)
    for roi_long in roi_short.keys():
        roi_label = roi_short[roi_long]
        y_full = np.concatenate([d3_roi[roi_long][story_n_dict[story]] for story in stories])
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_full)
        y_train = y_scaler.fit_transform(y_full.reshape(-1, 1)).flatten()

        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, y_train)
        out_base = f"../../confirmatory/registered_models/natstor/{modelname}_{roi_label}"
        with open(out_base, "wb") as handle:
            pickle.dump(reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"../../confirmatory/registered_models/natstor/normaliz_params/{modelname}_{roi_label}", "wb") as handle:
            pickle.dump([X_scaler, y_scaler], handle, protocol=pickle.HIGHEST_PROTOCOL)
        for random_idx, shift_val in enumerate([26, 52, 78, 104]):
            y_train_random = np.roll(y_train, shift_val)
            reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
            reg_random.fit(X_train, y_train_random)
            with open(f"{out_base}_random_{random_idx}", "wb") as handle:
                pickle.dump(reg_random, handle, protocol=pickle.HIGHEST_PROTOCOL)
