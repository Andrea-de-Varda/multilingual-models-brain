import numpy as np
import pandas as pd
from os import chdir
# 10 unique participants (9 in Exp. 2 and 6 in Exp. 3) reading 627 sentences (384 in Exp. 2, 243 in Exp. 3)

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/control")

np.mean([9]*384 + [6]*243) # pereira

# control
control = pd.read_csv("data/brain-lang-data_participant_20230728.csv")
df = control[control["roi"] == "lang_LH_netw"]#.groupby("sentence")#.agg({"response_target" : "mean", "cond" : "max"})
control["cond"].unique()
len(control[control["cond"]=="S"]["target_UID"].unique())
len(df)
1000 * 10 + 500 * 5 + 500 * 5
len(df[df["cond"]=="D"])
len(df) / 2000

sent_control = control["sentence"].unique()
tok_control = [w for sent in sent_control for w in sent.split()]
len(tok_control)
len(set(tok_control))


# Pereira 
chdir("/home/dev/Documents/PhD/Alice/additional_analyses/pereira")
grouped2 = pd.read_csv("pereira_averaged.csv")
grouped2.columns

len(grouped2["Sentence"].unique())

sent_pereira = grouped2["Sentence"].unique()
tok_pereira = [w for sent in sent_pereira for w in sent.split()]
len(tok_pereira)
len(set(tok_pereira))


# Study I
chdir("/home/dev/Documents/PhD/Alice")
all_codes = ["af", "nl", "fa", "fr", "lt", "mr", "no", "ro", "es", "ta", "tr", "vi"]
tokens, types = [], []
for lang in all_codes:
    df = pd.read_csv("transcribed/"+lang+".csv") # load transcription obtained with whisper
    df = df[df["end"] <= 260]
    tokens.append(len(df["text"]))
    types.append(len(df["text"].unique()))    
    
np.mean(tokens)
np.mean(types)

# Study II
chdir("/home/dev/Documents/PhD/Alice/confirmatory")

passages = ["Passage_1", "Passage_2", "Passage_3"]
languages = ["Arabic", "German", "Hindi", "Italian", "Korean", "Portuguese", "Russian", "Mandarin", "Polish"]
lang_codes = ["ar", "de", "hi", "it", "ko", "pt", "ru", "zh", "pl"]


all_tokens, all_types = [], []
for lang in lang_codes:
    tokens = []
    for passage in passages:
        df = pd.read_csv(f"transcribed/{passage}/"+lang+".csv")
        df = df[df["end"] <= 260]
        tokens.extend(df["text"].tolist())
    all_tokens.append(len(tokens))
    all_types.append(len(set(tokens)))

np.mean(all_tokens)
np.mean(all_types)


# Natural stories 

chdir("/home/dev/Documents/PhD/Alice/additional_analyses/NaturalStories")

ns = pd.read_csv("NaturalStories_TimeSeries_20220615.csv")

for s in ns["Story"].unique():
    temp = ns[ns.Story == s]
    n = len(set(temp.UID))
    print(f"{s:<30} {n}")

d_story_groups = {"kingofbirds" : ["StoriesAud_4_kingofbirds_N", "Stories_ToM_4_kingofbirds_N", "Stories_4_kingofbirds_N"],
                  "elvis" : ["StoriesAud_5_elvis_T", "Stories_ToM_5_elvis_T", "Stories_5_elvis_T", "ISCmega_elvis"], # ISCmega_elvis removed (different n° of TRs)
                  "aqua" : ["Stories_ToM_2_aqua_T", "Stories_2_aqua_T", "StoriesAud_2_aqua_T"],
                  "tulips" : ["Stories_9_tulips_T"],
                  "boar" : ["Stories_ToM_1_boar_T", "Stories_1_boar_T", "StoriesAud_1_boar_T"],
                  "tree" : ["ISCmega_tree"], 
                  "highschool" : ["Stories_ToM_7_highschool_N", "Stories_7_highschool_N"],
                  "matchstickseller" : ["Stories_3_matchstickseller_N"],
                  "mrsticky" : ["Stories_6_mrsticky_N"]}

ns_tr_columns = ["T_" + str(n) for n in range(1, 227)] # T_1, T_2, ... T_226

all_uid = []
for story_group, story_list in d_story_groups.items():
    print("\n", story_group)
    temp = ns[ns.Story.isin(story_list)]
    all_uid.extend(temp["UID"].unique().tolist())
    for single_story in story_list:
        temp_single = temp[temp.Story == single_story].filter(ns_tr_columns).copy().dropna(axis=1, how="all")
        print(f"{single_story:<30} {temp_single.shape}")
len(set(all_uid))


tokens = []
for n in range(1,11):
    lang = str(n)
    df = pd.read_csv("transcribed/"+lang+".csv")
    tokens.extend(df["text"].tolist())
len(tokens)
len(set(tokens))
