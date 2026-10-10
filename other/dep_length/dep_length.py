"""
SI S13 (Syntactic complexity) and Supplementary Figure 9: dependency length (A) and number of unique dependency
types with bootstrapped SD (B) in the four Study II training datasets, parsed with spaCy (en_core_web_sm).
"""
import numpy as np
import pandas as pd
from os import chdir
import os
from collections import Counter
import re
from tqdm import tqdm
import warnings
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

if "__file__" in globals():  # run as a script: resolve paths relative to the repository
    chdir(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
# when run interactively, set the working directory to <repo>/. before running

##############
# Synt depth #
##############

# study 1
study_1_base = [re.sub("\n", "", line) for line in open("other/dep_length/alice_punct.txt").readlines()]
study_1 = []
for sent in study_1_base:
    sent = re.sub(r"\.'", r"'\.", sent).strip()
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
plt.savefig("plots/dependency_length.svg", format="svg", bbox_inches="tight")  # Supplementary Figure 9A
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

def bootstrap_error(data, n_resamples=1000, seed=0):
    rng = np.random.default_rng(seed)  # seeded for reproducibility (the published error bars came from an unseeded run)
    resamples = [rng.choice(data, size=len(data), replace=True) for _ in range(n_resamples)]
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
plt.savefig("plots/dependency_types.svg", format="svg", bbox_inches="tight")  # Supplementary Figure 9B
plt.show()
