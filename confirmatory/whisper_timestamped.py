"""
Whisper-timestamped can be installed as:
!pip3 install git+https://github.com/linto-ai/whisper-timestamped

The relevant documentation can be found at:
https://github.com/linto-ai/whisper-timestamped

Consider than, for some languages, one could use different whisper models (e.g., vasista22/whisper-tamil-large-v2)
model = whisper.load_model("vasista22/whisper-tamil-large-v2", device="cpu")

Supported languages at https://github.com/openai/whisper/blob/main/README.md 
"""

import sys
from os import chdir
import os
import whisper_timestamped as whisper
import pickle
import pandas as pd
from glob import glob
from time import sleep

chdir("/home/dev/Documents/PhD/Alice/confirmatory")

def save(file, name):
    with open("transcribed/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("transcribed/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def report(result):
    rep = []
    for segment in result["segments"]:
        for w in segment["words"]:
            rep.append([w["text"], w["start"], w["end"]])
    df = pd.DataFrame(rep, columns=["text", "start", "end"])
    return df

model = whisper.load_model("large", device="cpu") # whisper large

languages = ["Arabic", "English", "German", "Hindi", "Italian", "Korean", "Mandarin", "Polish", "Portuguese", "Russian"]

lang_codes = ["ar", "en", "de", "hi", "it", "ko", "zh", "pl", "pt", "ru"]

for Lang, lang in zip(languages, lang_codes):
    for pass_n in ["Passage_1", "Passage_2", "Passage_3"]:
        if os.path.isfile(f"transcribed/{pass_n}/{lang}"):
            print(lang, "-", pass_n, "already done")
        else:
            print("\n\nProcessing", lang)
            # load audio file
            file = f"sound_data/{pass_n}/{Lang}.wav"
            audio = whisper.load_audio(file)
            result = whisper.transcribe(model, audio, language=lang)
            save(result, f"{pass_n}/{lang}")
            df = report(result)
            df.to_csv(f"transcribed/{pass_n}/"+lang+".csv", index=False)
            sleep(300)

# IMPORTANT NOTE for REPRODUCIBILITY!
# I had to remove some extra spaces from the french transcription, they were creating problems with the tokenization pipeline

fr = pd.read_csv("transcribed/fr.csv")
fr["text"] = fr["text"].str.replace(" ", "")
fr.to_csv("transcribed/fr.csv", index=False)

       



    