import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr, norm
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import copy
import itertools

chdir("/home/dev/Documents/PhD/Alice")

def load(model_prefix, monol = True, sequential = False, reset_context = False, random = False):
    if sequential:
        if reset_context:
            if monol:
                with open(f"results/split_context/monolingual_{model_prefix}", 'rb') as handle:
                    file = pickle.load(handle)
            else:
                raise ValueError('No multilingual sequential split')
        else:
            if monol:
                with open(f"results/monolingual_sequential_{model_prefix}", 'rb') as handle:
                    file = pickle.load(handle)
            else:
                raise ValueError('No multilingual sequential split')
    elif random:
        if monol:
            with open(f"results/random/monolingual_sequential_{model_prefix}", 'rb') as handle:
                file = pickle.load(handle)
        else:
            with open(f"results/random/multilingual_{model_prefix}", 'rb') as handle:
                file = pickle.load(handle)
    else:
        if monol:
            with open(f"results/monolingual_{model_prefix}", 'rb') as handle:
                file = pickle.load(handle)
        else:
            with open(f"results/multilingual_{model_prefix}", 'rb') as handle:
                file = pickle.load(handle)
    return file

def find_median_index(lst):
    sorted_lst = sorted(lst)
    mid_point = len(lst) // 2
    if len(lst) % 2 == 1:  # If odd, take the middle value
        median = sorted_lst[mid_point]
    else:  # If even, take the lower of the two middle values
        median = sorted_lst[mid_point - 1]
    return lst.index(median)

def get_best_layerwise(res_dict, colname = "m", give_mean = True, give_all = False):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    print(f"Best layer is {idx_max}")
    if give_all:
        best = res_dict[idx_max][colname].tolist()
    else:
        if give_mean:
            best = mean_results[idx_max]
        else:
            best = sd_results[idx_max] # so the SD is the cross-lingual variation for the best layer
    return best

def get_median_layerwise(res_dict, colname = "m", give_mean = True, give_all = False):
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    sd_results   = [value[colname].std() for key, value in res_dict.items()]
    idx_median = find_median_index(mean_results)
    if give_all:
        median = res_dict[idx_median][colname].tolist()
    else:
        if give_mean:
            median = mean_results[idx_median]
        else:
            median = sd_results[idx_median] # so the SD is the cross-lingual variation for the best layer
    return median

def plot_aggregate(df, title, ylim = None, ylimstart = None):
    plt.figure(figsize=(14, 10), dpi = 300)  
    sns.set_context("talk")
    palette = sns.color_palette("tab10")  # Colorblind-friendly palette
    ax = sns.barplot(x='Model', y='Score', hue='Family', data=df,
                     dodge=False, palette=palette, edgecolor='.2')
    for i in range(len(df['Score'])):
        plt.errorbar(i, df['Score'][i], yerr=df['sd'][i], fmt='none', capsize=5, ecolor='black', capthick=2)
    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=23)
    plt.yticks(fontsize=23)
    leg = plt.legend(title='Model Family', title_fontsize='20', fontsize='18', loc='upper left', bbox_to_anchor=(1, 1))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# Specifying model names

model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["XLM-R$_{base}$", "XLM-R$_{large}$", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "DistilmBERT", "mBERT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]
model_family = ["XLM-R", "XLM-R", "mT5", "mT5", "mT5", "BERT", "BERT", "XGLM", "XGLM", "XGLM", "XGLM"]

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# Specifying language names

langs = ['ja', 'fr', 'ta', 'ca', 'es', 'tr', 'vi', 'en', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice = ['Japanese', 'French', 'Tamil', 'Catalan', 'Spanish', 'Turkish', 'Vietnamese', 'English', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

lang_dict = {k : v for k, v in zip(langs, langs_nice)}

###########################
# barplot with best layer #
###########################

# monolingual, random split ###################################################
best_monol = [get_best_layerwise(load(model)) for model in model_names]
best_monol_sd = [get_best_layerwise(load(model), give_mean = False) for model in model_names]

best_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol,
    'Family': model_family,
    'sd' : best_monol_sd
})

plot_aggregate(best_layer, "Monolingual encoding",  ylim = .85, ylimstart = -.2)

# monolingual, sequential #####################################################
best_monol_sequential = [get_best_layerwise(load(model, sequential = True)) for model in model_names]
best_monol_sequential_sd = [get_best_layerwise(load(model, sequential = True), give_mean = False) for model in model_names]

best_layer_sequential = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol_sequential,
    'Family': model_family,
    'sd' : best_monol_sequential_sd
})

plot_aggregate(best_layer_sequential, "Sequential CV",  ylim = .85, ylimstart = -.2)

# monolingual, sequential, reset context ######################################
best_monol_sequential_reset = [get_best_layerwise(load(model, sequential = True, reset_context = True)) for model in model_names]
best_monol_sequential_reset_sd = [get_best_layerwise(load(model, sequential = True, reset_context = True), give_mean = False) for model in model_names]

best_layer_sequential_reset = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol_sequential_reset,
    'Family': model_family,
    'sd' : best_monol_sequential_reset_sd
})

plot_aggregate(best_layer_sequential_reset, "Sequential CV, clean context",  ylim = .85, ylimstart = -.2)

# random ######################################################################
best_monol_sequential_random = [get_best_layerwise(load(model, random = True)) for model in model_names]
best_monol_sequential_random_sd = [get_best_layerwise(load(model, random = True), give_mean = False) for model in model_names]

best_layer_sequential_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol_sequential_random,
    'Family': model_family,
    'sd' : best_monol_sequential_random_sd
})

plot_aggregate(best_layer_sequential_random, "Monolingual encoding, random",  ylim = .85, ylimstart = -.2)

# transfer ####################################################################
best_multi = [get_best_layerwise(load(model, monol = False), colname="r") for model in model_names]
best_multi_sd = [get_best_layerwise(load(model, monol = False), colname="r", give_mean = False) for model in model_names]

best_layer_multi = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_multi,
    'Family': model_family,
    'sd' : best_multi_sd
})

plot_aggregate(best_layer_multi, "Cross-lingual transfer", ylim = .85, ylimstart = -.2)

#################################################################################
# check with median (to see that bigger-is-better does not depend on n° layers) #
#################################################################################

median_monol = [get_median_layerwise(load(model)) for model in model_names]
median_monol_sd = [get_median_layerwise(load(model), give_mean = False) for model in model_names]

median_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol,
    'Family': model_family,
    'sd' : median_monol_sd
})

plot_aggregate(median_layer, "Monolingual encoding (median)", ylim = .85)

# monolingual, sequential #####################################################
median_monol = [get_median_layerwise(load(model, sequential = True)) for model in model_names]
median_monol_sd = [get_median_layerwise(load(model, sequential = True), give_mean = False) for model in model_names]

median_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol,
    'Family': model_family,
    'sd' : median_monol_sd
})

plot_aggregate(median_layer, "Sequential CV (median)", ylim = .85, ylimstart = -.2)

# monolingual, sequential, reset context ######################################
median_monol_reset = [get_median_layerwise(load(model, sequential = True, reset_context = True)) for model in model_names]
median_monol_reset_sd = [get_median_layerwise(load(model, sequential = True, reset_context = True), give_mean = False) for model in model_names]

median_layer_reset = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol_reset,
    'Family': model_family,
    'sd' : median_monol_reset_sd
})

plot_aggregate(median_layer_reset, "Sequential CV, clean context (median)", ylim = .85, ylimstart = -.2)

# random ######################################################################
median_monol_sequential_random = [get_median_layerwise(load(model, random = True)) for model in model_names]
median_monol_sequential_random_sd = [get_median_layerwise(load(model, random = True), give_mean = False) for model in model_names]

median_layer_sequential_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol_sequential_random,
    'Family': model_family,
    'sd' : median_monol_sequential_random_sd
})

plot_aggregate(median_layer_sequential_random, "Monolingual encoding",  ylim = .85, ylimstart = -.2)

# transfer ####################################################################
median_multi = [get_median_layerwise(load(model, monol = False), colname="r") for model in model_names]
median_multi_sd = [get_median_layerwise(load(model, monol = False), colname="r", give_mean = False) for model in model_names]

median_layer_multi = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_multi,
    'Family': model_family,
    'sd' : median_multi_sd
})

plot_aggregate(median_layer_multi, "Cross-lingual transfer (median)", ylim = .85)


#########################################################
# barplot with layerwise (normalized by layer position) #
#########################################################

# within langs

colname = "m"
sequential = False # change accordingly
out_dfs = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname, sequential = sequential)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi = 300)
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)
ax = sns.lineplot(
    data=out_dfs,
    x='l', 
    y='m', 
    hue='Class',  # Color by class
    style='Model',  # Different markers for each name
    markers=True, 
    dashes=False, 
    palette=palette
)
ax.set_xlabel('Layer position', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
ax.set_title('Monolingual encoding by layer', fontsize=30, weight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(title_fontsize='22', fontsize='18', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
ax.set_xlim([out_dfs['l'].min()-.02, out_dfs['l'].max()+.02])
ax.set_ylim(.01, .72)
plt.show()

##############################

# across langs
colname = "r"

out_dfs = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname, monol=False)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(14*.7, 12*.7), dpi = 300)
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)
ax = sns.lineplot(
    data=out_dfs,
    x='l', 
    y='m', 
    hue='Class',  # Color by class
    style='Model',  # Different markers for each name
    markers=True, 
    dashes=False, 
    palette=palette
)
ax.set_xlabel('Layer position', fontsize=27, labelpad=15)
ax.set_ylabel('R', fontsize=27, labelpad=15)
ax.set_title('Cross-lingual transfer by layer', fontsize=30, weight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(title_fontsize='22', fontsize='18', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
ax.set_xlim([out_dfs['l'].min()-.02, out_dfs['l'].max()+.02])
#ax.set_ylim(.01, .72)
plt.show()

#########
# split #
#########

colname = "r"
out_dfs = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname, monol=False)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)
model_colors = {model: color for model, color in zip(['BERT', 'XGLM', 'XLM-R', 'mT5'], sns.color_palette("tab10", n_colors=4))}
model_markers = {model: marker for model, marker in zip(names_nice_dict.values(), ['>', 'v', '^', 's', 'o', '<', 'p', '*', 'h', 'H', 'D'])}
out_dfs['color'] = out_dfs['Class'].map(model_colors)
out_dfs['marker'] = out_dfs['Model'].map(model_markers)

sns.set_style('whitegrid')
sns.set_context('talk')
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)

n_cols = 2
n_rows = (n_classes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*1.2, 8*1.2), dpi=300)
axes = axes.flatten() 
class_order = ['XLM-R', 'BERT', 'mT5', 'XGLM']
for i, model_class in enumerate(class_order):
    ax = axes[i]
    group_data = out_dfs[out_dfs['Class'] == model_class]
    for model in group_data['Model'].unique():
        model_data = group_data[group_data['Model'] == model]
        sns.lineplot(
            data=model_data,
            x='l', 
            y='m',
            color=model_data['color'].iloc[0],
            marker=model_data['marker'].iloc[0],
            label=model,
            ax=ax
        )
    #ax.legend(title='Model', fontsize=13, title_fontsize=20, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title(model_class, fontsize=25)
    ax.set_xlabel('')
    ax.set_xticks([.2, .4, .6, .8])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel('')
    ax.set_xlim([group_data['l'].min()-.02, group_data['l'].max()+.02])
    ax.set_ylim(.1, .5)
    ax.get_legend().remove()
fig.text(0.56, 0.04, 'Layer position', ha='center', va='center', fontsize=23)
fig.text(0.04, 0.5, 'R', ha='center', va='center', rotation='vertical', fontsize=23)
legend_handles = [Line2D([0], [0], color=model_colors[class_dict_nice[model]], marker=model_markers[model], label=model, linestyle='-', markersize=10) for model in out_dfs['Model'].unique()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=18, title_fontsize=24, title='Model', bbox_to_anchor=(1.25, .95))
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()

################################
# Encoding results by language #
################################

colname = "m"
sequential = False
r_lang = {lang : [] for lang in langs}
sd_lang = {lang : [] for lang in langs}
for model in model_names:
    res_dict = load(model, sequential = sequential)
    # first selecting best layer
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    df = res_dict[idx_max]
    for lang in langs:
        try:
            therow = df[df.lang == lang]
            r = therow.values[0][1]
            sd = therow.values[0][2]
            r_lang[lang].append(r)
            sd_lang[lang].append(sd)
        except IndexError: # xglm models miss some languages
            r_lang[lang].append(0)
            sd_lang[lang].append(0)

data = pd.DataFrame(r_lang)
data1 = pd.DataFrame(sd_lang) # IMPORTANT: here the SD is the variation across languages

fig, axes = plt.subplots(11, 1, figsize=(10*.9, 15*.9), dpi=300)
yticks = [0, .5, 1]
for i, ax in enumerate(axes):
    ax.bar(data.columns, data.iloc[i], color='indianred', yerr=data1.iloc[i])
    ax.set_ylim(-.1, 1)
    ax.set_yticks(yticks)  # Set the y-ticks to [0, 0.3, 0.6, 1]
    # grid
    ax.xaxis.grid(False)
    ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
    if i < len(axes) - 1:
        ax.set_xticklabels([])  # Hide x-tick labels
    else:
        ax.set_xticklabels([lang_dict[l] for l in data.columns], rotation=45, ha="right")
plt.suptitle('Monolingual encoding', y=.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()

# transfer results by language 

colname = "r"
r_lang = {lang : [] for lang in langs}
for model in model_names:
    res_dict = load(model, monol=False)
    # first selecting best layer
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    df = res_dict[idx_max]
    for lang in langs:
        try:
            therow = df[df.lang == lang]
            r = therow.values[0][1]
            r_lang[lang].append(r)
        except IndexError: # xglm models miss some languages
            r_lang[lang].append(0)

data = pd.DataFrame(r_lang)

fig, axes = plt.subplots(11, 1, figsize=(10*.9, 15*.9))
yticks = [0, .5, 1]
for i, ax in enumerate(axes):
    ax.bar(data.columns, data.iloc[i], color='indianred')
    ax.set_ylim(-.45, 1)
    ax.set_yticks(yticks)  # Set the y-ticks to [0, 0.3, 0.6, 1]
    ax.axhline(y=0, color='black',lw=2)
    # grid
    ax.xaxis.grid(False)
    ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
    if i < len(axes) - 1:
        ax.set_xticklabels([])  # Hide x-tick labels
    else:
        ax.set_xticklabels([lang_dict[l] for l in data.columns], rotation=45, ha="right")
plt.suptitle('', y=.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()