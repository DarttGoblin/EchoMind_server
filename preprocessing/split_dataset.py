import os
import shutil
from tqdm import tqdm

input_folder = "../.other_files/original_data/audios_silence_trimmed"
output_base = "../.other_files/original_data/audios_splited"

languages = ["egyptian", "frensh", "english"]

for lang in languages:
    os.makedirs(os.path.join(output_base, lang), exist_ok=True)

files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith(".wav")
]

for f in tqdm(files):
    for lang in languages:
        if f"_{lang}." in f.lower():
            shutil.copy(
                os.path.join(input_folder, f),
                os.path.join(output_base, lang, f)
            )
            break
