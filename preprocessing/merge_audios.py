from pydub import AudioSegment
import os
from tqdm import tqdm
import random

input_folder = "../.other_files/original_data/audios_splited/egyptian"
output_folder = "../.other_files/original_data/audios_splited_merged/egyptian"

languages = ["egyptian", "frensh", "english"]

os.makedirs(output_folder, exist_ok=True)

for lang in languages:
    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".wav", ".mp3")) and f"_{lang}." in f.lower()
    ])

    total_groups = len(files) // 3

    for i in tqdm(range(total_groups), desc=f"Merging {lang} audios (3 by 3)"):
        idx = i * 3

        audio1 = AudioSegment.from_file(os.path.join(input_folder, files[idx]))
        audio2 = AudioSegment.from_file(os.path.join(input_folder, files[idx + 1]))
        audio3 = AudioSegment.from_file(os.path.join(input_folder, files[idx + 2]))

        silence_1 = AudioSegment.silent(duration=random.randint(300, 500))
        silence_2 = AudioSegment.silent(duration=random.randint(300, 500))

        merged = audio1 + silence_1 + audio2 + silence_2 + audio3

        merged.export(
            os.path.join(output_folder, f"{i + 1}_{lang}.wav"),
            format="wav"
        )
