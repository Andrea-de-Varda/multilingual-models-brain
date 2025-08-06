import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from os import chdir
import os
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.stats import pearsonr, norm
from math import sqrt
from time import sleep
import seaborn as sns
from adjustText import adjust_text
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns
import copy
import itertools
import sys
sys.modules['numpy._core.numeric'] = np.core.numeric

chdir("/home/dev/Documents/PhD/Alice")

all_langs = ['Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian']
all_codes = ["es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro"]

lang_code_dict = {k : v for k, v in zip(all_codes, all_langs)}
lang_code_d_reversed = {v : k for k, v in lang_code_dict.items()}

xglm_langs = ["es", "vi", "ta", "tr", "fr"]
mgpt_langs   = ["af", "fa", "fr", "lt", "mr", "ro", "es", "ta", "tr", "vi"]

###############################################################################
# plot ########################################################################
###############################################################################

def patched_load(path):
    import sys
    import numpy
    sys.modules['numpy._core.numeric'] = numpy.core.numeric
    with open(path, 'rb') as f:
        return pickle.load(f)

def load(model_prefix, froi="all", monol=True, split_context=False, random=False, md=False, rh=False, native = False):
    if not monol:
        raise ValueError('No multilingual sequential split')

    if split_context:
        filename = f"results/split_context/monolingual_{model_prefix}"
    elif md:
        filename = f"results/monolingual_md_{model_prefix}_{froi}"
    elif rh:
        filename = f"results/monolingual_rh_{model_prefix}_{froi}"
    elif random:
        filename = f"results/monolingual_{model_prefix}_{froi}_circshift"
    elif native:
        filename = f"results/monolingual_native_{model_prefix}_{froi}"
    else:
        filename = f"results/monolingual_{model_prefix}_{froi}"
    print("Loading:", filename)
    return patched_load(filename)

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

def plot_aggregate(df, title, ylim=None, ylimstart=None, sig=[]):
    plt.figure(figsize=(18*.7, 11.5*.7), dpi = 300)
    sns.set_context("talk")
    palette = sns.color_palette("tab10")  # Colorblind-friendly palette
    ax = sns.barplot(x='Model', y='Score', hue='Family', data=df,
                     dodge=False, palette=palette, edgecolor='.2')
    for i in range(len(df['Score'])):
        plt.errorbar(i, df['Score'][i], yerr=df['sd'][i]/sqrt(df["n"][i]), fmt='none', capsize=5, ecolor='black', capthick=2)

    # significance asterisks
    for i, value in enumerate(df['Score']):
        if i < len(sig):
            y = value + df['sd'][i]/sqrt(df["n"][i]) + 0.02
            plt.text(i, y, sig[i], ha='center', va='bottom', color='black', fontsize=20, weight='bold')

    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=23)
    # leg = plt.legend(title='Model Family', title_fontsize='20', fontsize='18', loc='upper left', bbox_to_anchor=(1, 1))
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(4.0)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    ax.get_legend().remove()
    plt.show()

# Specifying model names

model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# language names
langs = ['fr', 'ta', 'es', 'tr', 'vi', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice = ['French', 'Tamil', 'Spanish', 'Turkish', 'Vietnamese', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

lang_dict = {k : v for k, v in zip(langs, langs_nice)}

###############################################################################
########################
# SIGNIFICANCE TESTING #
########################

def r_to_z(r1, r2, n = 130):    
    # fisher r-to-z transformation
    z_1 = np.arctanh(r1)
    z_2 = np.arctanh(r2)
    z_diff = z_1 - z_2
    # standard error of difference
    se_diff = np.sqrt(2*(1/(n-3)))
    z_stat = z_diff / se_diff
    p = 2 * (1 - norm.cdf(np.abs(z_stat))) # two tailed
    return z_stat, p

# stouffer method
def combine_z_statistics(z_stats, return_z = False):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    if return_z:
        return z_combined
    else:
        return combined_pvalue

# asterisks for plotting
def add_significance_asterisks(data):
    significance_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]
    for item in data:
        model, p_value = item
        asterisk = '' # default non-sig
        for threshold, symbol in significance_levels:
            if p_value < threshold:
                asterisk = symbol
                break
        item.append(asterisk)
    return data

p_values_best = []
for model in model_names:
    test = get_best_layerwise(load(model), give_all = True)
    z_temp = []
    for shift in [26, 52, 78, 104]:
        rand = get_best_layerwise(load(model, random=True)[shift], give_all = True)
        zs = [] # fisher's p values
        for t, r in zip(test, rand):
            z, p = r_to_z(t, r)
            zs.append(z)
        z_ = combine_z_statistics(zs, return_z = True)
        z_temp.append(z_)
    p_tot = combine_z_statistics(z_temp)
    p_values_best.append([model, p_tot])
p_values_best = add_significance_asterisks(p_values_best)
p_values_best = pd.DataFrame(p_values_best, columns = ["model", "p", "asterisk"])

###########################
# barplot with best layer #
###########################

# monolingual, best layer #####################################################
best_monol = [get_best_layerwise(load(model)) for model in model_names]
best_monol_sd = [get_best_layerwise(load(model), give_mean = False) for model in model_names]

best_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol,
    'Family': model_family,
    'sd' : best_monol_sd,
    'n' : n_langs
})

plot_aggregate(best_layer, "",  ylim = .85, ylimstart = -.1)#, sig = p_values_best["asterisk"])

###############################################################################
###############################################################################

# monolingual, best layer #####################################################
all_best_layers = {}
for froi in ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp', 'all']:
    best_monol = [get_best_layerwise(load(model, froi = froi)) for model in model_names]
    best_monol_sd = [get_best_layerwise(load(model, froi = froi), give_mean = False) for model in model_names]
    
    best_layer = pd.DataFrame({
        'Model': names_formatted,
        'Score': best_monol,
        'Family': model_family,
        'sd' : best_monol_sd,
        'n' : n_langs
    })
    
    all_best_layers[froi] = best_layer
    
    # plot_aggregate(best_layer, "",  ylim = .85, ylimstart = -.1)#, sig = p_values_best["asterisk"])

#########################
# DEFINITIVE FINAL PLOT #
#########################

frois = ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp']
main_froi = "all"
df_main = all_best_layers[main_froi]

froi_markers = {
    'Lang_LH_AntTemp': 's',
    'Lang_LH_IFG': 'o',
    'Lang_LH_IFGorb': '^',
    'Lang_LH_MFG': 'v',
    'Lang_LH_PostTemp': 'X'
}

title = ""
ylimstart = 0
ylim = 0.75

plt.figure(figsize=(24 * .7, 11.5 * .7), dpi=300)
sns.set_context("talk")
palette_d = {
    'BERT': "steelblue",
    'DeBERTa': "teal",
    'InfoXLM': "firebrick",
    'NLLB': "tomato",
    'XGLM': "forestgreen",
    'XLM-Align': "firebrick",
    'XLM-R': "lightsteelblue",
    'mGPT': "yellowgreen",
    'mT5': "darkorange"
}

df_main["color"] = df_main["Family"].map(palette_d)
bar_positions = [1,2,3,
                 4.5, 5.5, 6.5,
                 9, 10, 11, 12, 13, 14, 15.5, 16.5, 17.5, 19, 20, 21, 22, 23]

ax = plt.gca()
for i, pos in enumerate(bar_positions):
    row = df_main.iloc[i]
    err = row['sd'] / sqrt(row['n'])
    ax.errorbar(pos, row['Score'], yerr=err, fmt='o', color=row['color'], 
                markersize=16, alpha=0.9, lw=3, capsize=5)
for froi in frois:
    df = all_best_layers[froi]
    for i, pos in enumerate(bar_positions):
        row = df.iloc[i]
        ax.plot(pos, row['Score'], marker=froi_markers[froi], 
                color=palette_d[row['Family']], markersize=9, alpha=0.4, lw=0)

def add_bracket(ax, pos1, pos2, text, y_offset=0.05, weight="normal"):
    mid = (pos1 + pos2) / 2
    y = max(df_main['Score']) + y_offset
    ax.plot([pos1, pos1, pos2, pos2], [y, y + 0.02, y + 0.02, y], color='black', lw=2)
    ax.text(mid, y + 0.03, text, ha='center', va='bottom', fontsize=19, weight=weight)

add_bracket(ax, bar_positions[0], bar_positions[2], 'translation', y_offset=-.07)
add_bracket(ax, bar_positions[3], bar_positions[5], 'contrastive', y_offset=.02)
add_bracket(ax, bar_positions[0], bar_positions[5], 'explicit', y_offset=.12, weight="bold")
add_bracket(ax, bar_positions[6], bar_positions[11], 'masked LM', y_offset=-.03)
add_bracket(ax, bar_positions[12], bar_positions[14], 'span corr.', y_offset=-.09)
add_bracket(ax, bar_positions[15], bar_positions[19], 'causal LM', y_offset=.12)
add_bracket(ax, bar_positions[6], bar_positions[19], 'implicit', y_offset=.21, weight="bold")
ax.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.3)
plt.title(title, fontsize=30, weight='bold', pad=20)
plt.xlabel('Model', fontsize=27, labelpad=20)
plt.ylabel('R', fontsize=27, labelpad=20)
plt.ylim(ylimstart, ylim)
plt.xticks(bar_positions, labels=df_main["Model"], rotation=45, ha='right', fontsize=22)
plt.yticks([.1, .2, .3, .4, .5, .6, .7], fontsize=23)
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# legend
import matplotlib.lines as mlines
custom_order = ['Lang_LH_IFGorb', 'Lang_LH_IFG', 'Lang_LH_MFG', 'Lang_LH_AntTemp', 'Lang_LH_PostTemp']
handles = [
    mlines.Line2D([], [], marker=froi_markers[froi], color='gray', linestyle='None',
                  markersize=10, label=froi[8:])
    for froi in custom_order
]
fig, ax = plt.subplots(figsize=(2, 1), dpi=300)
ax.axis('off')
legend = ax.legend(handles=handles, loc='center', frameon=False, ncol=5, fontsize=13, handletextpad=0.5)
plt.tight_layout()
plt.show()

#####################
##########################################################
###############################################################################

# monolingual, median layer ###################################################
median_monol = [get_median_layerwise(load(model)) for model in model_names]
median_monol_sd = [get_median_layerwise(load(model), give_mean = False) for model in model_names]

median_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol,
    'Family': model_family,
    'sd' : median_monol_sd,
    'n' : n_langs
})
plot_aggregate(median_layer, "", ylim = .85, ylimstart = -.1)#, sig = p_values_median["asterisk"])

# random, best layer ##########################################################
best_monol_random = [get_best_layerwise(load(model, random=True)) for model in model_names]
best_monol_sd_random = [get_best_layerwise(load(model, random=True), give_mean = False) for model in model_names]

best_layer_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol_random,
    'Family': model_family,
    'sd' : best_monol_sd_random,
    'n' : n_langs
})

plot_aggregate(best_layer_random, "",  ylim = .85, ylimstart = -.1)

# random, median layer ########################################################
median_monol_random = [get_median_layerwise(load(model, random=True)) for model in model_names]
median_monol_sd_random = [get_median_layerwise(load(model, random=True), give_mean = False) for model in model_names]

median_layer_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol_random,
    'Family': model_family,
    'sd' : median_monol_sd_random,
    'n' : n_langs
})

plot_aggregate(median_layer_random, "",  ylim = .85, ylimstart = -.1)

###############################################################################

colname = "m"
sequential = True # change accordingly
out_dfs = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs.append(res_layer)
out_dfs = pd.concat(out_dfs)
#model_colors = {model: color for model, color in zip(['BERT', 'XGLM', 'XLM-R', 'mT5'], sns.color_palette("tab10", n_colors=4))}
model_markers = {model: marker for model, marker in zip(names_nice_dict.values(), ['>', '1', '^', 's', 'o', '<', 'p', '*', 'h', 'H', 'D', 'X', ',', "8", "2", "v", "4", "3", "P", "."])}
out_dfs['color'] = out_dfs['Class'].map(palette_d)
out_dfs['marker'] = out_dfs['Model'].map(model_markers)

sns.set_style('whitegrid')
sns.set_context('talk')
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)

n_cols = 3
n_rows = (n_classes + n_cols - 1) // n_cols

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*1.2, 8*1.2), dpi=300)
axes = axes.flatten() 

class_order = ['NLLB', 'XLM-Align', 'InfoXLM', 'XLM-R', 'BERT', 'DeBERTa', 'mT5', 'mGPT', 'XGLM']
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
    ax.set_ylim(-.05, .40)
    ax.get_legend().remove()
fig.text(0.56, 0.04, 'Layer position', ha='center', va='center', fontsize=23)
fig.text(0.04, 0.5, 'R', ha='center', va='center', rotation='vertical', fontsize=23)

legend_handles = [Line2D([0], [0], color=palette_d[class_dict_nice[model]], marker=model_markers[model], label=model, linestyle='-', markersize=10) for model in out_dfs['Model'].unique()]

fig.legend(handles=legend_handles, loc='upper right', fontsize=18, title_fontsize=24, title='Model', bbox_to_anchor=(1.25, .95))
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()

################################
# Encoding results by language #
################################

colname = "m"
sequential = True
r_lang = {lang : [] for lang in langs}
for model in model_names:
    res_dict = load(model)
    # first selecting best layer
    mean_results = [value[colname].mean() for key, value in res_dict.items()]
    idx_max = np.argmax(mean_results)
    df = res_dict[idx_max]
    for lang in langs:
        try:
            therow = df[df.lang == lang]
            r = therow.values[0][1]
            #sd = therow.values[0][2]
            r_lang[lang].append(r)
            #sd_lang[lang].append(sd)
        except IndexError: # xglm models miss some languages
            r_lang[lang].append(0)
            #sd_lang[lang].append(0)

data = pd.DataFrame(r_lang)

# fig, axes = plt.subplots(20, 1, figsize=(10*.9, 24*.9), dpi=300)
# yticks = [0, .5, 1]
# for i, ax in enumerate(axes):
#     ax.bar(data.columns, data.iloc[i], color='indianred')#, yerr=data1.iloc[i])
#     for j, value in enumerate(data.iloc[i]):
#         if value == 0:
#             ax.text(j, 0, 'NA', ha='center', va='bottom', fontsize=15, color='black')
#     ax.set_ylim(-.45, 1)
#     ax.set_yticks(yticks)  # Set the y-ticks to [0, 0.3, 0.6, 1]
#     ax.axhline(y=0, color='black',lw=2)
#     # grid
#     ax.xaxis.grid(False)
#     ax.set_ylabel(names_formatted[i], rotation=0, ha='right', va='center')
#     if i < len(axes) - 1:
#         ax.set_xticklabels([])  # Hide x-tick labels
#     else:
#         ax.set_xticklabels([lang_dict[l] for l in data.columns], rotation=45, ha="right")
# plt.suptitle('', y=.97, fontsize=26, weight="bold")
# plt.tight_layout()
# plt.show()

# mean and SE
data.loc['Mean'] = data.replace(0, np.nan).mean()
y_err = data.replace(0, np.nan).std() / np.sqrt(20)
data = pd.concat([data.loc[['Mean']], data.drop('Mean')]) # move mean row to the top

fig_height = (24 + 1.2) * 0.9
fig_width = 10 * 0.9
fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
gs = gridspec.GridSpec(len(data), 1, height_ratios=[1.5] + [1] * (len(data) - 1), hspace=0.5) # first row taller
yticks = [0, 0.5, 1]
axes = [fig.add_subplot(gs[i]) for i in range(len(data))]
for i, ax in enumerate(axes):
    bar_color = 'steelblue' if i == 0 else 'indianred'  # mean row is blue, others are red
    ax.bar(data.columns, data.iloc[i], color=bar_color)
    for j, value in enumerate(data.iloc[i]):
        if value == 0:
            ax.text(j, 0, 'NA', ha='center', va='bottom', fontsize=15, color='black')
    ax.set_ylim(-0.45, 1)
    ax.set_yticks(yticks)
    ax.axhline(y=0, color='black', lw=2)
    ax.xaxis.grid(False)
    if i == 0:
        ax.errorbar(data.columns, data.iloc[i], yerr=y_err, fmt='none', capsize=5, capthick=1, color='black') # add error bars to mean row
        ax.set_ylabel('Average', rotation=0, ha='right', va='center')
        ax.set_xticklabels([])
    else:
        ax.set_ylabel(names_formatted[i - 1], rotation=0, ha='right', va='center')
        ax.set_xticklabels([])
axes[-1].set_xticklabels([lang_dict[l] for l in data.columns], rotation=45, ha="right")
plt.suptitle('', y=0.97, fontsize=26, weight="bold")
plt.tight_layout()
plt.show()


###############################################################################
###############################################################################
###############################################################################

# SPLIT CONTEXT

# monolingual, best layer #####################################################
best_monol_split = [get_best_layerwise(load(model, split_context = True)) for model in model_names]
best_monol_sd_split = [get_best_layerwise(load(model, split_context = True), give_mean = False) for model in model_names]

best_layer_split = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_monol_split,
    'Family': model_family,
    'sd' : best_monol_sd_split,
    'n' : n_langs
})

plot_aggregate(best_layer_split, "",  ylim = .85, ylimstart = -.1)

# split context, plot for paper
df = best_layer_split
title = ""
ylimstart = 0
ylim = 0.75

plt.figure(figsize=(24*.7, 11.5*.7), dpi = 300)
sns.set_context("talk")
palette = sns.color_palette("tab20", n_colors = 9)
df["color"] = df["Family"].map(palette_d)
ax = plt.gca()
bar_positions = [1,2,3,
                 4.5, 5.5, 6.5,
                 9, 10, 11, 12, 13, 14, 15.5, 16.5, 17.5, 19, 20, 21, 22, 23]
bars = ax.bar(bar_positions, df['Score'], yerr=[df['sd'][i] / sqrt(df['n'][i]) for i in range(len(df))],
              capsize=5, color=df["color"], edgecolor='.2', alpha = 0.8, lw = 3)
ax.hlines(best_layer['Score'], xmin=[x - 0.3 for x in bar_positions], xmax=[x + 0.3 for x in bar_positions],
          colors='black', linestyles=(0, (1, 1)), linewidth=2) 
plt.title(title, fontsize=30, weight='bold', pad=20)
plt.xlabel('Model', fontsize=27, labelpad=20)
plt.ylabel('R', fontsize=27, labelpad=20)
plt.ylim(ylimstart, ylim)
plt.xticks(bar_positions, labels = df["Model"], rotation=45, ha='right', fontsize=22)
plt.yticks([.1, .2, .3, .4, .5, .6, .7], fontsize=23)
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# monolingual, median layer ###################################################
median_monol = [get_median_layerwise(load(model, split_context = True)) for model in model_names]
median_monol_sd = [get_median_layerwise(load(model), give_mean = False) for model in model_names]

median_layer = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_monol,
    'Family': model_family,
    'sd' : median_monol_sd,
    'n' : n_langs
})
plot_aggregate(median_layer, "", ylim = .85, ylimstart = -.1)

###############################################################################

############################
# MULTIPLE DEMANDS NETWORK #
############################

def plot_md(MD, L, title, ylim=None, ylimstart=None, sig=[], avg_sig=None, name0="MD", name1="Language", color0="tomato", color1="navy", colname="Network"):
    MD[colname] = name0
    L[colname] = name1
    df_combined = pd.concat([MD, L], axis=0).reset_index(drop=True)
    
    md_avg = MD["Score"].mean()
    md_se = MD["Score"].std() / np.sqrt(len(MD))
    l_avg = L["Score"].mean()
    l_se = L["Score"].std() / np.sqrt(len(L))

    plt.figure(figsize=(19*0.7, 10*0.7), dpi=300)
    sns.set_context("talk")
    palette = {name0: color0, name1: color1}
    ax = sns.barplot(x='Model', y='Score', hue=colname, data=df_combined,
                     dodge=True, palette=palette, edgecolor='.2')

    # find actual colors/width used by sns for each group
    bar0 = bar1 = None
    for patch, grp in zip(ax.patches, df_combined[colname]):
        if grp == name0 and bar0 is None:
            bar0 = patch
        elif grp == name1 and bar1 is None:
            bar1 = patch
        if bar0 and bar1:
            break

    if bar0 and bar1:
        md_bar_color = bar0.get_facecolor()
        md_bar_edge = bar0.get_edgecolor()
        bar_width = bar0.get_width()

        l_bar_color = bar1.get_facecolor()
        l_bar_edge = bar1.get_edgecolor()
    else:
        md_bar_color = color0
        md_bar_edge = '.2'
        l_bar_color = color1
        l_bar_edge = '.2'
        bar_width = 0.35

    unique_models = df_combined['Model'].unique()
    xvals = np.arange(len(unique_models))
    dodge_width = bar_width  # Adjusted here to match the actual bar width

    for i, model in enumerate(unique_models):
        for j, net in enumerate([name0, name1]):
            row = df_combined[(df_combined['Model'] == model) & (df_combined[colname] == net)]
            if not row.empty:
                xpos = xvals[i] + (j - 0.5) * dodge_width
                yerr = row['sd'].values[0] / np.sqrt(row['n'].values[0])
                plt.errorbar(xpos, row['Score'].values[0], yerr=yerr, fmt='none',
                             capsize=4, ecolor='black', capthick=2)

    for i, value in enumerate(L['Score']):
        if i < len(sig):
            y = value + L['sd'][i] / np.sqrt(L["n"][i]) + 0.02
            plt.text(i, y, sig[i], ha='center', va='bottom',
                     color='black', fontsize=16, weight='bold')

    xpos_md = len(unique_models) + 1
    xpos_l = xpos_md + dodge_width

    plt.bar(xpos_md, md_avg, color=md_bar_color, edgecolor=md_bar_edge, width=bar_width)
    plt.bar(xpos_l, l_avg, color=l_bar_color, edgecolor=l_bar_edge, width=bar_width)

    plt.errorbar(xpos_md, md_avg, yerr=md_se, fmt='none',
                 capsize=5, ecolor='black', capthick=2)
    plt.errorbar(xpos_l, l_avg, yerr=l_se, fmt='none',
                 capsize=5, ecolor='black', capthick=2)
    if avg_sig is not None:
        y_position = max(md_avg + md_se, l_avg + l_se) + 0.02
        x_position = (xpos_md + xpos_l) / 2
        plt.text(x_position, y_position, avg_sig, ha='center', va='bottom',
                 color='black', fontsize=16, weight='bold')
    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    if ylimstart is not None and ylim is not None:
        plt.ylim(ylimstart, ylim)

    old_ticks = ax.get_xticks()
    old_labels = [item.get_text() for item in ax.get_xticklabels()]
    avg_tick = (xpos_md + xpos_l) / 2
    plt.xticks(np.append(old_ticks, avg_tick),
               old_labels + ["Average"],
               rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=23)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

p_values_best_md = []
z_best_md = []
for model in model_names:
    test = get_best_layerwise(load(model), give_all = True)
    rand = get_best_layerwise(load(model, md=True), give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, rand):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    z = np.sum(zs) / np.sqrt(len(zs))
    z_best_md.append(z)
    p_values_best_md.append([model, p])
p_values_best_md = add_significance_asterisks(p_values_best_md)
p_values_best_md = pd.DataFrame(p_values_best_md, columns = ["model", "p", "asterisk"])
overall_p = combine_z_statistics(z_best_md)

###############################################################################

best_md = [get_best_layerwise(load(model, md=True)) for model in model_names]
best_md_sd = [get_best_layerwise(load(model, md=True), give_mean = False) for model in model_names]

best_layer_md = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_md,
    'Family': model_family,
    'sd' : best_md_sd,
    'n' : n_langs
})

plot_md(best_layer_md, best_layer, "", ylim = .75, ylimstart = 0, sig = p_values_best_md["asterisk"], avg_sig = "***")

###############################################################################

########################
# L - RIGHT HEMISPHERE #
########################

# significance testing (rh vs lh)

p_values_best_rh = []
z_best_rh = []
for model in model_names:
    test = get_best_layerwise(load(model), give_all = True)
    rand = get_best_layerwise(load(model, rh=True), give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, rand):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    z = np.sum(zs) / np.sqrt(len(zs))
    z_best_rh.append(z)
    p_values_best_rh.append([model, p])
p_values_best_rh = add_significance_asterisks(p_values_best_rh)
p_values_best_rh = pd.DataFrame(p_values_best_rh, columns = ["model", "p", "asterisk"])
overall_p_rh = combine_z_statistics(z_best_rh)

###############################################################################

best_rh = [get_best_layerwise(load(model, rh=True)) for model in model_names]
best_rh_sd = [get_best_layerwise(load(model, rh=True), give_mean = False) for model in model_names]

best_layer_rh = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_rh,
    'Family': model_family,
    'sd' : best_rh_sd,
    'n' : n_langs
})

plot_md(best_layer_rh, best_layer, "", ylim = .75, ylimstart = 0, sig = p_values_best_rh["asterisk"], name0 = "Right", name1 = "Left", color0 = "cornflowerblue", colname = "Hemisphere", avg_sig = "***")


###############################################################################

# spatial DISTRIBUTION

spatial_results = []
for froi in ['Lang_RH_AntTemp', 'Lang_RH_IFG', 'Lang_RH_IFGorb', 'Lang_RH_MFG', 'Lang_RH_PostTemp', 'all']:
    best_monol = [get_best_layerwise(load(model, froi = froi, rh = True)) for model in model_names]
    mean_r = np.mean(best_monol)
    se = np.std(best_monol) / np.sqrt(len(best_monol))
    spatial_results.append({"froi" : froi, "network" : "RH", "r" : mean_r, "se" : se, "all_points" : best_monol})
    
# spatial_results = []
# for froi in ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp', 'all']:
#     best_monol = [get_best_layerwise(load(model, froi = froi, native = True)) for model in model_names]
#     mean_r = np.mean(best_monol)
#     se = np.std(best_monol) / np.sqrt(len(best_monol))
#     froi = re.sub("Lang_LH_", "native_", froi)
#     spatial_results.append({"froi" : froi, "network" : "RH", "r" : mean_r, "se" : se, "all_points" : best_monol})
    
for froi in ['Lang_LH_AntTemp', 'Lang_LH_IFG', 'Lang_LH_IFGorb', 'Lang_LH_MFG', 'Lang_LH_PostTemp', 'all']:
    best_monol = [get_best_layerwise(load(model, froi = froi)) for model in model_names]
    mean_r = np.mean(best_monol)
    se = np.std(best_monol) / np.sqrt(len(best_monol))
    spatial_results.append({"froi" : froi, "network" : "LH", "r" : mean_r, "se" : se, "all_points" : best_monol})
    
for froi in ['MD_LH_Precentral_A_PrecG', 'MD_LH_Precentral_B_IFGop', 'MD_LH_antParietal', 'MD_LH_insula', 'MD_LH_medialFrontal', 'MD_LH_midFrontal', 'MD_LH_midFrontalOrb', 'MD_LH_midParietal', 'MD_LH_postParietal', 'MD_LH_supFrontal', 'MD_RH_Precentral_A_PrecG', 'MD_RH_Precentral_B_IFGop', 'MD_RH_antParietal', 'MD_RH_insula', 'MD_RH_medialFrontal', 'MD_RH_midFrontal', 'MD_RH_midFrontalOrb', 'MD_RH_midParietal', 'MD_RH_postParietal', 'MD_RH_supFrontal', 'all']:
    best_monol = [get_best_layerwise(load(model, froi = froi, md=True)) for model in model_names]
    mean_r = np.mean(best_monol)
    se = np.std(best_monol) / np.sqrt(len(best_monol))
    spatial_results.append({"froi" : froi, "network" : "MD", "r" : mean_r, "se" : se, "all_points" : best_monol})
    
spatial_results = pd.DataFrame(spatial_results)

df = spatial_results.copy()
df['short_label'] = df['froi'].str.replace(r'^(Lang|MD)_[LR]H_', '', regex=True)
df['short_label'] = df['short_label'].str.replace(r'_.*', '', regex=True)
df['short_label'] = df['short_label'].replace('all', 'All')
df['group_order'] = df['network'].map({'LH': 0, 'RH': 1, 'MD': 2})
df['is_all'] = (df['short_label'] == 'All').astype(int)

group_order = ['LH', 'RH', 'MD']
# df_sorted = pd.concat([
#     pd.concat([
#         g[g['is_all'] == 1],
#         g[g['is_all'] == 0].sort_values('r', ascending=False)
#     ])
#     for net in group_order
#     for _, g in df.groupby('network') if _ == net], ignore_index=True)

lang_order = ['IFGorb', 'IFG', 'MFG', 'AntTemp', 'PostTemp']
df['lang_cat'] = pd.Categorical(df['short_label'],
                                categories=['All'] + lang_order,
                                ordered=True)
df_sorted = pd.concat([
    pd.concat([
        g[g['is_all'] == 1],                                           # “All” first
        g[g['is_all'] == 0].sort_values(
            'lang_cat' if net in ('LH', 'RH') else 'r',
            ascending=True if net in ('LH', 'RH') else False)
    ])
    for net in group_order
    for _, g in df.groupby('network') if _ == net
], ignore_index=True)

spacing = 1.2
positions = []
tick_labels = []
last_group = None
offset = 0
group_indices = {}
for i, row in df_sorted.iterrows():
    group = row['network']
    if group != last_group:
        if last_group is not None:
            offset += 1
        group_indices[group] = []
        last_group = group
    positions.append(offset)
    group_indices[group].append(offset)
    tick_labels.append(row['short_label'])
    offset += spacing
df_sorted['pos'] = positions

color_map = {'LH': 'navy', 'RH': 'cornflowerblue', 'MD': 'tomato'}
df_sorted['color'] = df_sorted['network'].map(color_map)

plt.figure(figsize=(16*.7, 6*.7), dpi=300)
ax = plt.gca()
sns.set_context("talk")
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
for i, row in df_sorted.iterrows():
    x = row['pos']
    y_vals = row['all_points']
    ax.plot([x] * len(y_vals), y_vals, marker='o', color=row['color'], markersize=5, alpha=0.4, lw=0)
ax.errorbar(df_sorted['pos'], df_sorted['r'], yerr=df_sorted['se'],
            fmt='o', markersize=11, color='black', alpha=1.0, lw=0, zorder=3)

for i, row in df_sorted.iterrows():
    ax.plot(row['pos'], row['r'], 'o', color=row['color'], markersize=11, alpha=0.85, zorder=4)

# Brackets
def add_bracket(ax, start_pos, end_pos, label, y_offset=0.01, weight='normal'):
    mid = (start_pos + end_pos) / 2
    y = df_sorted['r'].max() + y_offset
    ax.plot([start_pos, start_pos, end_pos, end_pos],
            [y, y + 0.01, y + 0.01, y], color='black', lw=1.5)
    ax.text(mid, y + 0.015, label, ha='center', va='bottom', fontsize=14, weight=weight)

add_bracket(ax, group_indices['LH'][0], group_indices['RH'][-1], 'Language', y_offset=0.3, weight='bold')
add_bracket(ax, group_indices['LH'][0], group_indices['LH'][-1], 'Left hem.', y_offset=0.2)
add_bracket(ax, group_indices['RH'][0], group_indices['RH'][-1], 'Right hem.', y_offset=0.2)
add_bracket(ax, group_indices['MD'][0], group_indices['MD'][-1], 'MD network', y_offset=0.2, weight='bold')

ax.set_xticks(df_sorted['pos'])
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)
ax.set_ylabel("R", fontsize=16)
# ax.set_ylim(df_sorted['r'].min() - 0.05, df_sorted['r'].max() + 0.1)
plt.yticks(fontsize=12)
plt.ylim(-.05, .55)
sns.despine()
plt.tight_layout()
plt.show()

# SAVE ORDER FOR MULTI (WILL RECYCLE PLOTTING CODE) 
df_sorted['key'] = df_sorted['network'] + '|' + df_sorted['froi']   # unique, no ‘all’ clash
pos_map   = dict(zip(df_sorted['key'], df_sorted['pos']))
label_map = dict(zip(df_sorted['key'], df_sorted['short_label']))
