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

chdir("/home/dev/Documents/PhD/Alice")

def save(file, name):
    with open("transcribed/"+name, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(name):
    with open("transcribed/"+name, 'rb') as handle:
        file = pickle.load(handle)
    return file

def report(result, lang):
    rep = []
    for segment in result["segments"]:
        for w in segment["words"]:
            rep.append([w["text"], w["start"], w["end"]])
    df = pd.DataFrame(rep, columns=["text", "start", "end"])
    df.to_csv("transcribed/"+lang+".csv", index=False)
    return df

model = whisper.load_model("large", device="cpu") # whisper large

languages = ['Catalan', 'Japanese', 'English', 'Spanish', 'Marathi', 'Afrikaans', 'Vietnamese', 'Tamil', 'Lithuanian', 'Turkish', 'Dutch', 'Norwegian', 'Farsi', 'French', 'Romanian', 'Italian'] # no irish in Whisper

long_passages = [f'sound_data/{l}/{l}_Intact_Long/{l}.wav' for l in languages]

lang_codes = ["ca", "ja", "en", "es", "mr", "af", "vi", "ta", "lt", "tr", "nl", "no", "fa", "fr", "ro", "ita"]

for file, lang in zip(long_passages, lang_codes):
    if os.path.isfile(f"transcribed/{lang}"):
        print(lang, "already done")
    else:
        print("\n\nProcessing", lang)
        # load audio file
        audio = whisper.load_audio(file)
        result = whisper.transcribe(model, audio, language=lang)
        save(result, lang)
        df = report(result, lang)   
        sleep(60)

# IMPORTANT NOTE for REPRODUCIBILITY!
# I had to remove some extra spaces from the french transcription, they were creating problems with the tokenization pipeline

fr = pd.read_csv("transcribed/fr.csv")
fr["text"] = fr["text"].str.replace(" ", "")
fr.to_csv("transcribed/fr.csv", index=False)

# mr # Got inconsistent length for segment 26 (20 != 18). Some words have been ignored.

# fa # Got inconsistent length for segment 7 (17 != 15). Some words have been ignored.
     # Got inconsistent length for segment 43 (51 != 49). Some words have been ignored.
       



    
