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

def load(model_prefix, monol = True, sequential = False, reset_context = False, random = False, md = False, rh = False):
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
            if md:
                with open(f"results/multilingual_md_{model_prefix}", 'rb') as handle:
                    file = pickle.load(handle)
            elif rh:
                with open(f"results/multilingual_rh_{model_prefix}", 'rb') as handle:
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
model_names = ["xlmr_base", "xlmr_large", "mt5_small", "mt5_base", "mt5_large", "distilmbert", "bert_base", 
               "mdeberta", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "m2m100_small", "m2m100_large", "mgpt",
               "xglm_small", "xglm_med", "xglm_large", "xglm_xl"]
names_formatted = ["XLM-R$_{base}$", "XLM-R$_{large}$", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "DistilmBERT", "mBERT", 
                   "mDeBERTa", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "M2M100$_{small}$", "M2M100$_{large}$", "mGPT",
                   "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]


model_family = ["XLM-R", "XLM-R", "mT5", "mT5", "mT5", "BERT", "BERT", "DeBERTa", "XLM-Align", "InfoXLM", "InfoXLM", "MiniLM", "NLLB", "NLLB", "NLLB", "M2M100", "M2M100", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

# Specifying language names

# langs = ['ita', 'ja', 'fr', 'ta', 'ca', 'es', 'tr', 'vi', 'en', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
# langs_nice = ['Italian', 'Japanese', 'French', 'Tamil', 'Catalan', 'Spanish', 'Turkish', 'Vietnamese', 'English', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

langs = ['fr', 'ta', 'es', 'tr', 'vi', 'mr', 'af', 'nl', 'no', 'fa', 'ro', 'lt']
langs_nice = ['French', 'Tamil', 'Spanish', 'Turkish', 'Vietnamese', 'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 'Romanian', 'Lithuanian']

lang_dict = {k : v for k, v in zip(langs, langs_nice)}

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
def combine_z_statistics(z_stats):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
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
    test = get_best_layerwise(load(model, monol = False), colname = "r", give_all = True)
    rand = get_best_layerwise(load(model, monol = False, random=True), colname = "r", give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, rand):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    p_values_best.append([model, p])
p_values_best = add_significance_asterisks(p_values_best)
p_values_best = pd.DataFrame(p_values_best, columns = ["model", "p", "asterisk"])
    
p_values_median = []
for model in model_names:
    test = get_median_layerwise(load(model, monol = False), colname = "r", give_all = True)
    rand = get_median_layerwise(load(model, monol = False, random=True), colname = "r", give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, rand):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    p_values_median.append([model, p])
p_values_median = add_significance_asterisks(p_values_median)
p_values_median = pd.DataFrame(p_values_median, columns = ["model", "p", "asterisk"])

####################################
# barplot with best & median layer #
####################################

# transfer ####################################################################
best_multi = [get_best_layerwise(load(model, monol = False), colname="r") for model in model_names]
best_multi_sd = [get_best_layerwise(load(model, monol = False), colname="r", give_mean = False) for model in model_names]

best_layer_multi = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_multi,
    'Family': model_family,
    'sd' : best_multi_sd,
    'n' : n_langs
})

plot_aggregate(best_layer_multi, "", ylim = .85, ylimstart = -.1)#, sig = p_values_best["asterisk"])

# transfer ####################################################################
median_multi = [get_median_layerwise(load(model, monol = False), colname="r") for model in model_names]
median_multi_sd = [get_median_layerwise(load(model, monol = False), colname="r", give_mean = False) for model in model_names]

median_layer_multi = pd.DataFrame({
    'Model': names_formatted,
    'Score': median_multi,
    'Family': model_family,
    'sd' : median_multi_sd,
    'n' : n_langs
})

plot_aggregate(median_layer_multi, "", ylim = .85, ylimstart = -.1)#, sig = p_values_median["asterisk"])


# random ######################################################################
best_random = [get_best_layerwise(load(model, monol = False, random = True), colname="r") for model in model_names]
best_random_sd = [get_best_layerwise(load(model, monol = False, random = True), colname="r", give_mean = False) for model in model_names]

best_layer_random = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_random,
    'Family': model_family,
    'sd' : best_random_sd,
    'n' : n_langs
})

plot_aggregate(best_layer_random, "", ylim = .85, ylimstart = -.1)



#########################################################
# barplot with layerwise (normalized by layer position) #
#########################################################

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
model_markers = {model: marker for model, marker in zip(names_nice_dict.values(), ['>', 'v', '^', 's', 'o', '<', 'p', '*', 'h', 'H', 'D', '1', '2', '3', '4', '8'])}
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
    ax.set_ylim(0, .35)
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

######################################################
# relatioship between signal in fMRI (pairwise corr) #
######################################################

lang_code_d_reversed = {v : k for k, v in lang_dict.items()}
corrs = pd.read_csv("results/correlations/correlation_participants_new.csv")
corrs = corrs[["lang", "r", "p"]]
corrs = corrs[(corrs["r"] > 0) & (corrs["p"] < .05)]
corrs["code"] = corrs.lang.map(lang_code_d_reversed)

n_rows = 3
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), dpi=300)
axes = axes.flatten()

avg_corr = []
for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    r_dict_temp = {k: v[idx] for k, v in r_lang.items()}
    corrs_ = corrs.copy()
    corrs_[model_name] = corrs_["code"].map(r_dict_temp)
    corrs_ = corrs_[corrs_[model_name] != 0].dropna()
    r_ = pearsonr(corrs_["r"], corrs_[model_name])[0]
    r = round(r_, 2)
    avg_corr.append(r_)
    sns.regplot(x=corrs_["r"], y=corrs_[model_name], ci=95, scatter_kws={'s': 20}, ax=ax)
    ax.annotate(f"r = {r}", (0.37, 0.75), fontsize=17, alpha=1)
    ax.set_ylim(-0.25, 0.9)
    ax.set_xlim(0.17, 0.53)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(names_nice_dict[model_name])

for ax in axes[len(model_names):]:
    ax.axis('off')

#fig.suptitle('Encoding performance and signal reliability', fontsize=20)
fig.text(0.5, 0, 'Correlation in fMRI response', ha='center', va='center', fontsize=18)
fig.text(0, 0.5, 'Encoding performance', ha='center', va='center', rotation='vertical', fontsize=18)

plt.tight_layout()
plt.show()

print(np.mean(avg_corr))


len(set(r_lang.keys()).intersection(set(corrs["code"])))
len(set(corrs["code"]))

len(r_dict_temp.keys())

###############################################################################

######################
# MD NETWORK RESULTS #
######################

def plot_md(MD, L, title, ylim=None, ylimstart=None, sig = [], name0 = "MD", name1 = "L", color0 = "tomato", color1 = "navy", colname = "Network"):
    MD[colname] = name0
    L[colname] = name1
    df_combined = pd.concat([MD, L], axis=0).reset_index(drop=True)

    plt.figure(figsize=(14*.7, 11.5*.7), dpi=300)
    sns.set_context("talk")
    palette = {name0 : color0, name1 : color1}
    
    ax = sns.barplot(x='Model', y='Score', hue=colname, data=df_combined, dodge=True, palette=palette, edgecolor='.2')

    num_models = len(df_combined['Model'].unique())
    model_positions = np.arange(num_models)
    dodge_width = 0.35

    for i, model in enumerate(df_combined['Model'].unique()):
        for j, Network_ in enumerate([name0, name1]):
            row = df_combined[(df_combined['Model'] == model) & (df_combined[colname] == Network_)]
            xpos = model_positions[i] + (j - 0.5) * dodge_width
            yerr = row['sd'].values[0] / np.sqrt(row['n'].values[0])
            plt.errorbar(xpos, row['Score'].values[0], yerr=yerr, fmt='none', capsize=5, ecolor='black', capthick=2)
    
    for i, value in enumerate(L['Score']):
        if i < len(sig):
            y = value + L['sd'][i]/sqrt(L["n"][i]) + 0.02
            plt.text(i, y, sig[i], ha='center', va='bottom', color='black', fontsize=20, weight='bold')
            
    plt.title(title, fontsize=30, weight='bold', pad=20)
    plt.xlabel('Model', fontsize=27, labelpad=20)
    plt.ylabel('R', fontsize=27, labelpad=20)
    if ylimstart is not None and ylim is not None:
        plt.ylim(ylimstart, ylim)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=23)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    #ax.get_legend().remove()  # Optionally remove the legend
    plt.show()

# significance testing (md vs lang)

p_values_best_md = []
for model in model_names:
    test = get_best_layerwise(load(model, monol = False, md=True), colname="r", give_all = True)
    base = get_best_layerwise(load(model, monol = False), colname="r", give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, base):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    p_values_best_md.append([model, p])
p_values_best_md = add_significance_asterisks(p_values_best_md)
p_values_best_md = pd.DataFrame(p_values_best_md, columns = ["model", "p", "asterisk"])

###############################################################################

best_multi_md = [get_best_layerwise(load(model, monol = False, md=True), colname="r") for model in model_names]
best_multi_md_sd = [get_best_layerwise(load(model, monol = False, md=True), colname="r", give_mean = False) for model in model_names]

best_layer_multi_md = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_multi_md,
    'Family': model_family,
    'sd' : best_multi_md_sd,
    'n' : n_langs
})


plot_aggregate(best_layer_multi_md, "", ylim = .85, ylimstart = -.1)

plot_md(best_layer_multi_md, best_layer_multi, "", ylim = .85, ylimstart = -.1, sig = p_values_best_md["asterisk"])

############################
# RIGHT HEMISPHERE RESULTS #
############################

p_values_best_rh = []
for model in model_names:
    test = get_best_layerwise(load(model, monol = False, rh=True), colname="r", give_all = True)
    base = get_best_layerwise(load(model, monol = False), colname="r", give_all = True)
    zs = [] # fisher's p values
    for t, r in zip(test, base):
        z, p = r_to_z(t, r)
        zs.append(z)
    p = combine_z_statistics(zs)
    p_values_best_rh.append([model, p])
p_values_best_rh = add_significance_asterisks(p_values_best_rh)
p_values_best_rh = pd.DataFrame(p_values_best_rh, columns = ["model", "p", "asterisk"])

###############################################################################

best_multi_rh = [get_best_layerwise(load(model, monol = False, rh=True), colname="r") for model in model_names]
best_multi_rh_sd = [get_best_layerwise(load(model, monol = False, rh=True), colname="r", give_mean = False) for model in model_names]

best_layer_multi_rh = pd.DataFrame({
    'Model': names_formatted,
    'Score': best_multi_rh,
    'Family': model_family,
    'sd' : best_multi_rh_sd,
    'n' : n_langs
})

plot_aggregate(best_layer_multi_rh, "", ylim = .85, ylimstart = -.1)

plot_md(best_layer_multi_rh, best_layer_multi, "", ylim = .85, ylimstart = -.1, sig = p_values_best_rh["asterisk"], name0 = "Right", name1 = "Left", color0 = "cornflowerblue", colname = "Hemisphere")


