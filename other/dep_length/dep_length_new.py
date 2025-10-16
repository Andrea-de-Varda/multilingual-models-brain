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
control = control_avg[control_avg["cond"] == "B"].index.tolist()

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

colors = sns.color_palette("muted", 4)
labels = list(unique_dep_counts.keys())
values = list(unique_dep_counts.values())
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
for i, (lab, val) in enumerate(zip(labels, values)):
    ax.errorbar(
        x[i], val, yerr=errors[lab],
        fmt='o',
        mfc=colors[i], mec='black', mew=1.2, markersize=8,
        ecolor='black', elinewidth=1.3, capsize=5,
        zorder=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("N° dependency types", fontsize=12)
ax.set_xlim(-0.5, len(labels) - 0.5)
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

def process_sentences(sentences, ngram_sizes):
    ngram_counters = {n: Counter() for n in ngram_sizes}
    dep_counter = Counter()

    for sentence in tqdm(sentences):
        doc = nlp(sentence)
        pos_tags = [token.tag_ for token in doc]
        dep_tags = [token.dep_ for token in doc]

        # dependency counts
        dep_counter.update(dep_tags)

        # n-grams for each size
        for n in ngram_sizes:
            ngrams = zip(*[pos_tags[i:] for i in range(n)])
            ngram_counters[n].update(ngrams)

    return ngram_counters, dep_counter

def compute_entropy_with_smoothing(counter, unified_vocab):
    # Create distribution over unified vocabulary
    distribution = np.array([counter[item] + 1 for item in unified_vocab])  # Add 1 for Laplace smoothing
    probabilities = distribution / distribution.sum()
    return -np.sum(probabilities * np.log(probabilities))

def calculate_entropies(datasets, ngram_sizes):
    """Calculate entropies for n-grams and dependencies."""
    # Unified vocabularies for n-grams and dependencies
    unified_ngrams = {n: set() for n in ngram_sizes}
    unified_deps = set()

    # Collect all n-grams and dependencies across datasets
    for sentences in datasets.values():
        ngram_counters, dep_counter = process_sentences(sentences, ngram_sizes)
        for n in ngram_sizes:
            unified_ngrams[n].update(ngram_counters[n])
        unified_deps.update(dep_counter)

    # Compute entropies
    ngram_entropies = {name: {n: None for n in ngram_sizes} for name in datasets}
    dep_entropies = {}

    for name, sentences in datasets.items():
        # Process sentences once
        ngram_counters, dep_counter = process_sentences(sentences, ngram_sizes)

        # Compute entropy for each n-gram size
        for n in ngram_sizes:
            ngram_entropies[name][n] = compute_entropy_with_smoothing(ngram_counters[n], unified_ngrams[n])

        # Compute entropy for dependencies
        dep_entropies[name] = compute_entropy_with_smoothing(dep_counter, unified_deps)

    return ngram_entropies, dep_entropies

ngram_sizes = [2, 3, 4]
ngram_entropies, dep_entropies = calculate_entropies(datasets, ngram_sizes)

colors = sns.color_palette("muted", len(datasets))
bar_width = 0.2
plt.figure(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
for i, (label, ngram_data) in enumerate(ngram_entropies.items()):
    for j, n in enumerate(ngram_sizes):
        x_offset = i + (j - 1) * bar_width
        plt.bar(
            x=x_offset,
            height=ngram_data[n],
            color=colors[i],
            edgecolor="black",
            linewidth=1.3,
            width=bar_width,
            label=f"{label} (n={n})" if j == 0 else None
        )
plt.xticks(ticks=range(len(ngram_entropies)), labels=ngram_entropies.keys(), fontsize=12)
plt.ylabel("Entropy of PoS n-grams", fontsize=12)
plt.ylim(4, 9.5)
sns.despine()
plt.tight_layout()
plt.show()


colors = sns.color_palette("muted", len(datasets))
bar_width = 0.6
fig, axes = plt.subplots(1, len(ngram_sizes), fisgsize=(14, 3), dpi=300, sharey=True)
for j, n in enumerate(ngram_sizes):
    ax = axes[j]
    for i, (label, ngram_data) in enumerate(ngram_entropies.items()):
        ax.bar(
            x=i,
            height=ngram_data[n],
            color=colors[i],
            edgecolor="black",
            linewidth=1.3,
            width=bar_width,
            label=label if j == 0 else None
        )
    ax.set_title(f"N = {n}", fontsize=12, weight = "bold")
    ax.set_xticks(range(len(ngram_entropies)))
    ax.set_xticklabels(ngram_entropies.keys(), fontsize=10)
    if j == 0:
        ax.set_ylabel("Entropy of PoS N-grams", fontsize=12)
    ax.set_ylim(4, 9.5)
    sns.despine(ax=ax)
fig.legend(
    loc="upper center",
    ncol=len(datasets),
    fontsize=10,
    bbox_to_anchor=(0.5, 1.15)
)
plt.tight_layout()
plt.show()


# Plot dependency types entropy
plt.figure(figsize=(7 * 0.8, 3 * 0.8), dpi=300)
for i, (label, entropy) in enumerate(dep_entropies.items()):
    plt.bar(
        x=i,
        height=entropy,
        color=colors[i],
        edgecolor="black",
        linewidth=1.3,
        width=0.6,
    )
plt.xticks(ticks=range(len(dep_entropies)), labels=dep_entropies.keys(), fontsize=12)
plt.ylabel("Entropy of dep. types", fontsize=12)
sns.despine()
plt.tight_layout()
plt.ylim(2.5, 3.1)
plt.show()