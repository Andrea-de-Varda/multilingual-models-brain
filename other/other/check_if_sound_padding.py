import wave
import os

base_dir = '/home/dev/Documents/PhD/Alice/sound_data'

langs = ['French', 'Tamil', 'Spanish', 'Turkish', 'Vietnamese', 
         'Marathi', 'Afrikaans', 'Dutch', 'Norwegian', 'Farsi', 
         'Romanian', 'Lithuanian']

for lang in langs:
    target_path = os.path.join(base_dir, lang, f"{lang}_Intact_Long", f"{lang}.wav")
    if os.path.exists(target_path):
        with wave.open(target_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            minutes, seconds = divmod(duration, 60)
            print(f"{target_path}: {int(minutes)} minutes, {int(seconds)} seconds")
    else:
        print(f"File not found: {target_path}")

# confirmatory
folder_path = '/home/dev/Documents/PhD/Alice/confirmatory/sound_data/Passage_3'

for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        file_path = os.path.join(folder_path, file)
        with wave.open(file_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            minutes, seconds = divmod(duration, 60)
            print(f"{file}: {int(minutes)} minutes, {int(seconds)} seconds")
            
# English (not used) is the only one with < 4.20 sec