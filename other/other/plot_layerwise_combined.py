import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from os import chdir
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

chdir("/home/dev/Documents/PhD/Alice")

def patched_load(path):
    import sys
    import numpy
    sys.modules['numpy._core.numeric'] = numpy.core.numeric
    with open(path, 'rb') as f:
        return pickle.load(f)

def load(model_prefix, froi="all", monol=False, split_context=False, random=False, md=False, rh=False, native = False, multitrain = True):
    if monol:
        if md:
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
    else:
        mtpfx = "multitrain_" if multitrain else ""
        if md:
            filename = f"results/multilingual_{mtpfx}md_{model_prefix}_{froi}"
        elif rh:
            filename = f"results/multilingual_{mtpfx}rh_{model_prefix}_{froi}"
        elif random:
            filename = f"results/multilingual_{mtpfx}{model_prefix}_{froi}_circshift"
        elif native:
            filename = f"results/multilingual_{mtpfx}native_{model_prefix}_{froi}"
        else:
            filename = f"results/multilingual_{mtpfx}{model_prefix}_{froi}"
        print("Loading:", filename)
    return patched_load(filename)

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

def combine_z_statistics(z_stats):
    # combine Zs taking accounting for their signs
    z_combined = np.sum(z_stats) / np.sqrt(len(z_stats))
    combined_pvalue = 2 * norm.cdf(-abs(z_combined))
    return combined_pvalue

# Specifying model names
model_names = ["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

names_formatted = ["NLLB$_{d-small}$", "NLLB$_{d-large}$", "NLLB$_{large}$", "XLM-Align", "InfoXLM$_{small}$", "InfoXLM$_{large}$", "mMiniLM", "XLM-R$_{base}$", "XLM-R$_{large}$", "DistilmBERT", "mBERT", "mDeBERTa", "mT5$_{small}$", "mT5$_{base}$", "mT5$_{large}$", "mGPT", "XGLM$_{small}$", "XGLM$_{med}$", "XGLM$_{large}$", "XGLM$_{xl}$"]

model_family = ["NLLB", "NLLB", "NLLB", "XLM-Align", "InfoXLM", "InfoXLM", "XLM-R", "XLM-R", "XLM-R", "BERT", "BERT", "DeBERTa", "mT5", "mT5", "mT5", "mGPT", "XGLM", "XGLM", "XGLM", "XGLM"]
n_langs = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 5, 5, 5, 5] # n langs by model

names_nice_dict = {name : nice for name, nice in zip(model_names, names_formatted)}
class_dict = {name : theclass for name, theclass in zip(model_names, model_family)}
class_dict_nice = {name : theclass for name, theclass in zip(names_formatted, model_family)}

palette_d = {'BERT' : "steelblue",
             'DeBERTa' : "teal",
             'InfoXLM' : "firebrick",
             'NLLB' : "tomato",
             'XGLM' : "forestgreen",
             'XLM-Align' : "firebrick",
             'XLM-R' : "lightsteelblue",
             'mGPT' : "yellowgreen",
             'mT5' : "darkorange"}

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
# model_colors = {model: color for model, color in zip(['BERT', 'XGLM', 'XLM-R', 'mT5'], sns.color_palette("tab10", n_colors=4))}
model_markers = {model: marker for model, marker in zip(names_nice_dict.values(), ['>', '1', '^', 's', 'o', '<', 'p', '*', 'h', 'H', 'D', 'X', ',', "8", "2", "v", "4", "3", "P", "."])}
out_dfs['color'] = out_dfs['Class'].map(palette_d)
out_dfs['marker'] = out_dfs['Model'].map(model_markers)


colname = "m"
out_dfs_mono = []
for modelname in model_names:
    name_nice = names_nice_dict[modelname]
    model_class = class_dict[modelname]
    results_dict = load(modelname, monol=True)
    layers = max(results_dict.keys())
    res_layer = pd.DataFrame([[key / layers, value[colname].mean(), name_nice, model_class] for key, value in results_dict.items()], columns = ["l", "m", "Model", "Class"])
    out_dfs_mono.append(res_layer)
out_dfs_mono = pd.concat(out_dfs_mono)
# model_colors = {model: color for model, color in zip(['BERT', 'XGLM', 'XLM-R', 'mT5'], sns.color_palette("tab10", n_colors=4))}
model_markers = {model: marker for model, marker in zip(names_nice_dict.values(), ['>', '1', '^', 's', 'o', '<', 'p', '*', 'h', 'H', 'D', 'X', ',', "8", "2", "v", "4", "3", "P", "."])}
out_dfs_mono['color'] = out_dfs_mono['Class'].map(palette_d)
out_dfs_mono['marker'] = out_dfs_mono['Model'].map(model_markers)

sns.set_style('whitegrid')
sns.set_context('talk')

# Data preparation assumptions
n_classes = out_dfs['Class'].nunique()
palette = sns.color_palette("tab10", n_colors=n_classes)
class_order = ['NLLB', 'XLM-Align', 'InfoXLM', 'XLM-R', 'BERT', 'DeBERTa', 'mT5', 'mGPT', 'XGLM']

fig = plt.figure(figsize=(13*.8, 7*.8), dpi=400)  # Main figure size
gs = gridspec.GridSpec(3, 3, figure=fig)
for i, model_class in enumerate(class_order):
    # Find row and column index
    row = i // 3
    col = i % 3
    
    # Create a nested GridSpec for two subplots
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row, col], wspace=0.1)

    # Plot for out_dfs
    ax1 = fig.add_subplot(inner_gs[0, 1])
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
            ax=ax1, markersize=7
        )
    ax1.set_xticks([.2, .6])
    #ax1.tick_params(axis='x', labelsize=18)
    if row == 2:
        ax1.set_xticks([.25, .5, .75])
        ax1.tick_params(axis='x', labelsize=0)
    else:
        ax1.set_xticks([.25, .5, .75])
        ax1.tick_params(axis='x', labelsize=0)
    #ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xlim([group_data['l'].min()-.02, group_data['l'].max()+.02])
    ax1.set_ylim(-.05, .38)
    ax1.set_ylabel('')  # Remove y-axis label
    ax1.set_xlabel('')  # Remove x-axis label
    ax1.set_yticks([0, .1, .2, .3])  # Set y-ticks
    ax1.get_legend().remove()
    ax1.spines['top'].set_color('black')
    ax1.spines['top'].set_linewidth(1.5)
    ax1.spines['bottom'].set_color('black')
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_color('black')
    ax1.spines['right'].set_linewidth(1.5)

    # Plot for out_dfs_mono
    ax2 = fig.add_subplot(inner_gs[0, 0])
    group_data_mono = out_dfs_mono[out_dfs_mono['Class'] == model_class]
    for model in group_data_mono['Model'].unique():
        model_data_mono = group_data_mono[group_data_mono['Model'] == model]
        sns.lineplot(
            data=model_data_mono,
            x='l', 
            y='m',
            color=model_data_mono['color'].iloc[0],
            marker=model_data_mono['marker'].iloc[0],
            label=model,
            ax=ax2, markersize=7
        )
    #ax2.set_xticks([.2, .4, .6, .8])
    if row == 2:
        ax2.set_xticks([.25, .5, .75])
        ax2.tick_params(axis='x', labelsize=0)
    else:
        ax2.set_xticks([.25, .5, .75])
        ax2.tick_params(axis='x', labelsize=0)
    if col == 0:
        ax2.tick_params(axis='y', labelsize=16)
    else:
        ax2.set_yticklabels([])
    #ax2.tick_params(axis='x', labelsize=18)
    #ax2.tick_params(axis='y', labelsize=0)
    ax2.set_xlim([group_data_mono['l'].min()-.02, group_data_mono['l'].max()+.02])
    ax2.set_ylim(-.05, .38)
    ax2.set_ylabel('')  # Remove y-axis label
    ax2.set_xlabel('')  # Remove x-axis label
    ax2.set_yticks([0, .1, .2, .3])  # Set y-ticks
    ax2.get_legend().remove()
    ax2.spines['top'].set_color('black')
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(1.5)

    # Set a common title for the sub-subplots
    fig.text(x=(col / 3) + 0.2 - col*0.02, y=(1 - row / 3) -row*-0.01 - 0.02, s=model_class, ha='center', fontsize=22)

# Adjust the layout of the figure
fig.text(0.53, 0.02, 'Layer position', ha='center', va='center', fontsize=23)
fig.text(0.00, 0.5, 'R', ha='center', va='center', rotation='vertical', fontsize=23)
legend_handles = [Line2D([0], [0], color=palette_d[class_dict_nice[model]], marker=model_markers[model], label=model, linestyle='-', markersize=8) for model in out_dfs['Model'].unique()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=13.2, title_fontsize=18, title='Model', bbox_to_anchor=(1.2, 1.05))
fig.tight_layout()
plt.savefig("plots/layerwise.svg", format="svg", bbox_inches="tight")
plt.show()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

legend_handles = [
    Line2D(
        [0], [0],
        color=palette_d[class_dict_nice[model]],
        marker=model_markers[model],
        label=model,
        linestyle='-',
        markersize=8
    )
    for model in out_dfs['Model'].unique()
]

fig, ax = plt.subplots(figsize=(7, 3), dpi=400)
ax.axis('off')  # remove axes

legend = fig.legend(
    handles=legend_handles,
    loc='center',
    fontsize=13.2,
    title_fontsize=18,
    title='Model',
    ncol=3
)
plt.savefig("plots/layerwise_legend.svg", format="svg", bbox_inches="tight")
plt.show()


################################################
# Diff in r layerwise (compare mono vs. multi) #
################################################

df_both = pd.merge(out_dfs[["l", "m", "Model", "Class", "color"]], out_dfs_mono[["l", "m", "Model", "Class", "color"]], on = ["l", "Model", "Class", "color"], suffixes = ["_multi", "_mono"])
df_both["delta_m"] = df_both["m_mono"] - df_both["m_multi"]

p_diff = []
for index, row in df_both.iterrows():
    z, p = r_to_z(row["m_mono"], row["m_multi"])
    p_diff.append(p)
df_both["p_diff"] = p_diff

bin_edges = np.linspace(0, 1, 8) 
#bins = pd.interval_range(start=0, end=1, freq=1 / 9, closed = "left")
df_both['l_bin'] = pd.cut(df_both['l'], bins=bin_edges, include_lowest=True, right=True)
df_both["l_bin_str"] = df_both["l_bin"].astype(str)
res_agg = []
all_df_agg = []
for name, df in df_both.groupby("l_bin"):
    df_agg = df.groupby("Model").agg({"m_multi": "mean", "m_mono" : "mean", "delta_m" : "mean", "l_bin_str" : "max", "Model" : "max"})
    all_df_agg.append(df_agg)
    div = np.sqrt(len(df_agg))
    res_agg.append([df_agg.m_multi.mean(), df_agg.m_mono.mean(), df_agg.m_multi.std()/div, df_agg.m_mono.std()/div])
all_df_agg = pd.concat(all_df_agg)
res_agg = pd.DataFrame(res_agg, columns = ["r_multi", "r_mono", "sd_multi", "sd_mono"])


plt.figure(figsize = (7*.45, 5*.45), dpi=300)
x = np.array(range(len(res_agg)))/(len(res_agg)-1)
plt.plot(x, res_agg["r_multi"], color = "darkslateblue")
plt.plot(x, res_agg["r_mono"], color = "tomato")
plt.fill_between(
    x, 
    res_agg["r_multi"] - res_agg["sd_multi"], 
    res_agg["r_multi"] + res_agg["sd_multi"], 
    color='blue', 
    alpha=0.1, 
)
plt.fill_between(
    x, 
    res_agg["r_mono"] - res_agg["sd_mono"], 
    res_agg["r_mono"] + res_agg["sd_mono"], 
    color='orange', 
    alpha=0.1, 
)
plt.text(0.48, 0.146, "ACROSS", fontsize=14, weight = "bold")
plt.text(0.05, 0.187, "WITHIN", fontsize=14, weight = "bold")
ax = plt.gca()
plt.title("Averaged")
#plt.xlabel("Layer position")
plt.ylabel("Mean R")
plt.xticks([0, 0.25, 0.5, 0.75, 1])
ax.spines['top'].set_color('black')
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_color('black')
ax.spines['right'].set_linewidth(1.5)
plt.xlim(-.05, 1.05)
#plt.tight_layout()
plt.savefig("plots/layerwise_avg.svg", format="svg", bbox_inches="tight")
plt.show()

##############################
# Other plots (probably cut) #
##############################

df = out_dfs_mono
l_max = []
for model in df["Model"].unique():
    temp = df[df["Model"] == model]
    max_m_row = temp[temp['m'] == temp['m'].max()]
    l_value = max_m_row['l'].iloc[0]
    l_max.append(l_value)
np.mean(l_max)
#plt.scatter(list(range(len(l_max))), l_max)

plt.hist(l_max, bins=20, color='blue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Histogram of l_max Values')
plt.xlabel('l_max')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df["l"], df["m"])

bins = pd.interval_range(start=0, end=1, freq=0.07)
df['l_bin'] = pd.cut(df['l'], bins=bins)
binned_data = df.groupby('l_bin')['m'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(binned_data['l_bin'].astype(str), binned_data['m'], color='blue', edgecolor='black')
plt.xticks(rotation=90)  # Rotate labels to avoid overlap
plt.xlabel('l Bins')
plt.ylabel('Average m')
plt.title('Average m for Each 0.05 Bin of l')
plt.show()

df_mono = out_dfs_mono
df = out_dfs

plt.figure(figsize=(10, 6), dpi = 300)
bins = pd.interval_range(start=0, end=1, freq=0.03)
df_mono['l_bin'] = pd.cut(df_mono['l'], bins=bins)
binned_data = df_mono.groupby('l_bin')['m'].mean().reset_index()
x = [xx-.22 for xx in list(range(len(binned_data['l_bin'].astype(str))))]
plt.bar(x, binned_data['m'], color='blue', edgecolor='black', width = .34)

bins = pd.interval_range(start=0, end=1, freq=0.03)
df['l_bin'] = pd.cut(df['l'], bins=bins)
binned_data = df.groupby('l_bin')['m'].mean().reset_index()
x1 = [xx+.22 for xx in list(range(len(binned_data['l_bin'].astype(str))))]
plt.bar(x1, binned_data['m'], color='darkslateblue', edgecolor='black', width = .34)
plt.xticks([], rotation=90)  # Rotate labels to avoid overlap
plt.xlabel('Layer position')
plt.ylabel('R')
#plt.title('Average m for Each 0.05 Bin of l')
plt.show()



#####################
# Testing stats sig #
#####################

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# linear mixed-effects model with layer as a categorical predictor and random intercept for the model

df_both["l_bin_str"] = df_both["l_bin"].astype(str)
unique_bins = df_both["l_bin_str"].unique()
bin_map = {bin_val: f"bin_{i+1}" for i, bin_val in enumerate(unique_bins)}
df_both["l_bin_simple"] = df_both["l_bin_str"].map(bin_map)

model = smf.mixedlm("delta_m ~ C(l_bin_simple)", data=df_both, groups=df_both["Model"], re_formula="~1")
res = model.fit()
print(res.summary())

param_names = res.params.index

# Identify categorical parameters corresponding to non-baseline categories
cat_params = [i for i, name in enumerate(param_names) if "C(l_bin_simple)" in name]

pvals = []
comparisons = []

# single joint hypothesis that all these category coefficients = 0
hyp_matrix = np.zeros((1, len(param_names)))
for p_idx in cat_params:
    hyp_matrix[0, p_idx] = 1

# Perform the Wald test providing both the r_matrix and cov_p arguments
wres = res.wald_test(r_matrix=hyp_matrix, cov_p=res.cov_params())
print(wres)
pvals.append(wres.pvalue)
comparisons.append("baseline_vs_all")

# For a non-baseline category (e.g., bin_2): test that bin_2 differs from all others
# This involves setting up a matrix that compares bin_2's coefficient to each other bin's coefficient.
# For simplicity, if you just want to test each non-baseline bin against the baseline:
#   C(l_bin_simple)[T.bin_2] = 0
# This yields one test per non-baseline level.
for name in param_names:
    if "C(l_bin_simple)[T." in name:
        # Single test against baseline
        wres = res.wald_test(name + " = 0")
        pvals.append(wres.pvalue)   
        comparisons.append(name)

# Multiple comparison correction
reject, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

for comp, p, p_corr, r in zip(comparisons, pvals, pvals_corrected, reject):
    print(f"Comparison: {comp}")
    print(f"Raw p-value: {p:.4f}, Corrected p-value: {p_corr:.4f}, Significant: {r}")