"""
Descriptive statistics of the datasets reported in the Figure 1C table: participants, stimuli, fMRI volumes,
tokens and word types (unique tokens). Reliability and similarity values of Figure 1C are computed elsewhere
(see the README, "Figure 1C").

Tokens and types are counted on the material actually used for the encoding models:
  - Study I / Study II: Whisper transcriptions up to 260 s (the scanned part of each passage);
  - NatStories: the 9 analysed stories (story 8 has no fMRI data and is excluded everywhere else);
  - Tuckute2024: the 1,000 baseline sentences (cond == "B") used to train the encoding models;
  - Pereira2018: the 627 sentences.
"""
import os
import pickle
from os import chdir

import numpy as np
import pandas as pd

if "__file__" in globals():  # run as a script: resolve paths relative to the repository root
    chdir(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
# when run interactively, set the working directory to the repository root before running


def report(name, **values):
    print(f"\n{name}")
    for k, v in values.items():
        print(f"  {k:<28s} {v}")


def tokens_types(words):
    return len(words), len(set(words))


# Study I: 12 languages x 2 participants, 1 passage (130 volumes after removing the silent periods)
with open("data/dict_fROI", "rb") as f:
    d1 = pickle.load(f)
codes1 = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]
tt = []
for lang in codes1:
    df = pd.read_csv(f"transcribed/{lang}.csv")
    tt.append(tokens_types(df[df["end"] <= 260]["text"].tolist()))
names1 = ["Afrikaans", "Dutch", "Farsi", "French", "Lithuanian", "Marathi", "Norwegian", "Romanian", "Spanish",
          "Tamil", "Turkish", "Vietnamese"]  # data/dict_fROI holds all 41 languages; only these 12 were retained
report("Study I", participants=sum(len(d1[l]) for l in names1), languages=len(names1),
       volumes_per_language=len(next(iter(d1["French"].values()))["all"]),
       mean_tokens_per_language=round(np.mean([t for t, _ in tt]), 2),
       mean_types_per_language=round(np.mean([t for _, t in tt]), 2))

# Study II: 9 languages x 3 participants, 3 passages (130 volumes each)
codes2 = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]
tt = []
for lang in codes2:
    words = []
    for passage in ["Passage_1", "Passage_2", "Passage_3"]:
        df = pd.read_csv(f"confirmatory/transcribed/{passage}/{lang}.csv")
        words.extend(df[df["end"] <= 260]["text"].tolist())
    tt.append(tokens_types(words))
d2 = pd.read_csv("confirmatory/data/confirmatory_time_series.csv", usecols=["UID"])
report("Study II (all three passages, including the five excluded language-passage combinations)",
       participants=d2["UID"].nunique(), volumes_per_language=3 * 130,
       mean_tokens_per_language=round(np.mean([t for t, _ in tt]), 2),
       mean_types_per_language=round(np.mean([t for _, t in tt]), 2))

# NatStories: 9 stories (the time series per story are those used for the encoding models)
stories = ["1", "2", "3", "4", "5", "6", "7", "9", "10"]
words = []
for story in stories:
    words.extend(pd.read_csv(f"additional_analyses/NaturalStories/transcribed/{story}.csv")["text"].tolist())
with open("additional_analyses/NaturalStories/response/d_shift_3", "rb") as f:
    d_ns = pickle.load(f)
ns_path = "additional_analyses/NaturalStories/NaturalStories_TimeSeries_20220615.csv"
# runs of the 9 analysed stories (same grouping as natstor_fmri_timeseries.py)
story_runs = ["StoriesAud_4_kingofbirds_N", "Stories_ToM_4_kingofbirds_N", "Stories_4_kingofbirds_N",
              "StoriesAud_5_elvis_T", "Stories_ToM_5_elvis_T", "Stories_5_elvis_T", "ISCmega_elvis",
              "Stories_ToM_2_aqua_T", "Stories_2_aqua_T", "StoriesAud_2_aqua_T", "Stories_9_tulips_T",
              "Stories_ToM_1_boar_T", "Stories_1_boar_T", "StoriesAud_1_boar_T", "ISCmega_tree",
              "Stories_ToM_7_highschool_N", "Stories_7_highschool_N", "Stories_3_matchstickseller_N",
              "Stories_6_mrsticky_N"]
if os.path.isfile(ns_path):
    ns = pd.read_csv(ns_path, usecols=["UID", "Story"])
    participants = ns[ns["Story"].isin(story_runs)]["UID"].nunique()
else:
    participants = "needs NaturalStories_TimeSeries_20220615.csv (python download_data.py natstories_timeseries.zip)"
n_tok, n_typ = tokens_types(words)
report("NatStories", participants=participants, stories=len(d_ns),
       volumes=sum(len(v) for v in d_ns.values()), tokens=n_tok, types=n_typ)

# Pereira2018: 627 sentences; 9 participants saw the 384 sentences of Exp. 2 and 6 the 243 of Exp. 3
pereira = pd.read_csv("additional_analyses/pereira/pereira_averaged.csv")
sents = pereira["Sentence"].unique()
n_tok, n_typ = tokens_types([w for s in sents for w in s.split()])
report("Pereira2018", unique_participants=10, sentences=len(sents),
       mean_participants_per_sentence=round(np.mean([9] * 384 + [6] * 243), 2), tokens=n_tok, types=n_typ)

# Tuckute2024: 1,000 baseline sentences, each read by the same 10 participants
control = pd.read_csv("additional_analyses/control/data/brain-lang-data_participant_20230728.csv",
                      usecols=["sentence", "cond", "target_UID"])
base = control[control["cond"] == "B"]
sents = base["sentence"].unique()
n_tok, n_typ = tokens_types([w for s in sents for w in s.split()])
report("Tuckute2024 (baseline sentences)", participants=base["target_UID"].nunique(), sentences=len(sents),
       participants_per_sentence=base.groupby("sentence")["target_UID"].nunique().mean(), tokens=n_tok, types=n_typ)
