import numpy as np
import pandas as pd
from os import chdir
import pickle
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances
import re
from tqdm import tqdm
import warnings
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

chdir("/home/dev/Documents/PhD/Alice")

##############
# Synt depth #
##############

# study 1
study_1_base = [re.sub("\n", "", line) for line in open("other/dep_length/alice_punct.txt").readlines()]
study_1 = []
for sent in study_1_base:
    sent = re.sub("\.'", "'\.", sent).strip()
    study_1.extend([s for s in sent.split(".") if s])

# natstor
stories = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]
natstor = []
for story in stories:
    df = pd.read_csv("additional_analyses/NaturalStories/transcribed/"+story+".csv")
    text = df["text"].str.cat(sep=' ')
    #text = re.sub(r'[^a-zA-Z\s]', '', text)
    natstor.extend([l.strip() for l in text.split(".")])
   
# pereira
pereira = pd.read_csv("additional_analyses/pereira/pereira_averaged.csv")["Sentence"].tolist()

# control
control = pd.read_csv("additional_analyses/control/data/brain-lang-data_participant_20230728.csv")
avg_1 = control.groupby(["sentence", "target_UID"]).agg({"response_target" : "mean", "cond" : "first", "sentence" : "first"}).reset_index(drop=True) # first average across fROIs
control_avg = avg_1.groupby("sentence").agg({"response_target" : "mean", "cond" : "first"}) 
control = control_avg.index.tolist()

##############
# dep length #
##############

nlp = spacy.load("en_core_web_sm")

def calculate_dependency_lengths(sentences):
    lengths = []
    for sentence in tqdm(sentences):
        doc = nlp(sentence)
        for token in doc:
            # distance between the head and the dependent token
            length = abs(token.i - token.head.i)
            lengths.append(length)
    return lengths

study_1_lengths = calculate_dependency_lengths(study_1)
pereira_lengths = calculate_dependency_lengths(pereira)
control_lengths = calculate_dependency_lengths(control)
natstor_lengths = calculate_dependency_lengths(natstor)

########
# plot #
########

data = [study_1_lengths, natstor_lengths, pereira_lengths, control_lengths]
labels = ["Study I", "NatStories", "Pereira2018", "Tuckute2024"]

plt.figure(figsize=(7*.8, 3*.8), dpi = 300)
sns.violinplot(data=data, density_norm="count", inner = "box", linewidth=1, fill = True, cut = 0, bw_adjust=2,linecolor="k",inner_kws=dict(box_width=5, whis_width=1.3))
means = [np.mean(lengths) for lengths in data]
#plt.scatter(range(len(means)), means, marker = "_", s = 12, color="black", zorder=3)
plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=12)
plt.ylabel("Dependency length", fontsize=12)
#plt.title("Distribution of dependency lengths", fontsize=14)
#plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

###################
# dependency TYPE #
###################

def dependency_types(sentences):
    dep_types = []
    for sentence in tqdm(sentences):
        doc = nlp(sentence)
        dep_types.extend([token.dep_ for token in doc])
    return dep_types

def bootstrap_error(data, n_resamples=1000):
    resamples = [np.random.choice(data, size=len(data), replace=True) for _ in range(n_resamples)]
    unique_counts = [len(set(resample)) for resample in resamples]
    return np.std(unique_counts)

datasets = {
    "Study I": study_1,
    "NatStories": natstor,
    "Pereira2018": pereira,
    "Tuckute2024": control
}

dependency_counts = {}
total_dep_counts = {}  # store total counts
for name, sentences in datasets.items():
    dep_types = dependency_types(sentences)
    counts = Counter(dep_types)
    dependency_counts[name] = counts  # unique counts
    total_dep_counts[name] = sum(counts.values())  # total counts

unique_dep_counts = {name: len(counts) for name, counts in dependency_counts.items()}
errors = {name: bootstrap_error(list(counts.elements())) for name, counts in dependency_counts.items()}
normalized_counts = {name: unique_dep_counts[name] / total_dep_counts[name] for name in datasets}

# unique dependency types (no normalization)
colors = sns.color_palette("muted", len(unique_dep_counts))
plt.figure(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
x_positions = range(len(unique_dep_counts))
bar_width = 0.6
for i, (label, value) in enumerate(unique_dep_counts.items()):
    plt.bar(x=i, height=value, yerr=errors[label], color=colors[i], edgecolor="black", linewidth=1.3, width=bar_width, capsize=5)
plt.xticks(ticks=x_positions, labels=unique_dep_counts.keys(), fontsize=12)
plt.ylabel("N° dependency types", fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()

# normalized dependency types
plt.figure(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
for i, (label, value) in enumerate(normalized_counts.items()):
    plt.bar(x=i, height=value, color=colors[i], edgecolor="black", linewidth=1.3, width=bar_width, capsize=5)
plt.xticks(ticks=x_positions, labels=normalized_counts.keys(), fontsize=12)
plt.ylabel("Proportion of unique dependency types", fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()


###############
# PoS N-grams #
###############

def pos_ngrams(sentences, n=2):
    ngram_types = []
    for sentence in tqdm(sentences):
        doc = nlp(sentence)
        pos_tags = [token.tag_ for token in doc]
        ngrams = zip(*[pos_tags[i:] for i in range(n)])
        ngram_types.extend(ngrams)
    return ngram_types

ngram_sizes = [2, 3, 4]
pos_ngram_counts = {name: {} for name in datasets}
total_ngram_counts = {name: {} for name in datasets}
errors = {name: {} for name in datasets}  # bootstrap errors

for name, sentences in datasets.items():
    for n in ngram_sizes:
        ngram_types = pos_ngrams(sentences, n)
        counts = Counter(ngram_types)
        pos_ngram_counts[name][n] = len(counts)  # unique ngrams
        total_ngram_counts[name][n] = sum(counts.values())  # total ngrams
        errors[name][n] = bootstrap_error([' '.join(ngram) for ngram in list(counts.elements())])

# proportions vs. total ngrams
normalized_counts = {name: {n: pos_ngram_counts[name][n] / total_ngram_counts[name][n] for n in ngram_sizes} for name in datasets}

# Unique counts (not normalized)
plt.figure(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
for i, (label, ngram_data) in enumerate(pos_ngram_counts.items()):
    for j, n in enumerate(ngram_sizes):
        x_offset = i + (j - 1) * bar_width
        plt.bar(
            x=x_offset,
            height=ngram_data[n],
            yerr=errors[label][n],
            color=colors[i],
            edgecolor="black",
            linewidth=1.3,
            width=bar_width,
            capsize=5,
            label=f"{label} (n={n})" if j == 0 else None,  # remove duplicate labels
        )
plt.xticks(ticks=range(len(pos_ngram_counts)), labels=pos_ngram_counts.keys(), fontsize=12)
plt.ylabel("Unique PoS n-grams", fontsize=12)
sns.despine()
#plt.legend(fontsize=10, loc="upper right", title="Datasets and n-grams")
plt.tight_layout()
plt.show()

# normalized counts
plt.figure(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
for i, (label, ngram_data) in enumerate(normalized_counts.items()):
    for j, n in enumerate(ngram_sizes):
        x_offset = i + (j - 1) * bar_width
        plt.bar(
            x=x_offset,
            height=ngram_data[n],
            color=colors[i],
            edgecolor="black",
            linewidth=1.3,
            width=bar_width,
            capsize=5,
            label=f"{label} (n={n})" if j == 0 else None,  # remove duplicate labels
        )
plt.xticks(ticks=range(len(normalized_counts)), labels=normalized_counts.keys(), fontsize=12)
plt.ylabel("Unique PoS n-grams (proportion)", fontsize=12)
sns.despine()
#plt.legend(fontsize=10, loc="upper right", title="Datasets and n-grams")
plt.tight_layout()
plt.show()