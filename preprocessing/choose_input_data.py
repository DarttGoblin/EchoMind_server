from pydub import AudioSegment
import os
import random
from tqdm import tqdm
import shutil

input_folder = "../.other_files/original_data/audios_splited_merged/english"
output_folder = "../.other_files/original_data/audios_splited_merged_choosed/english"

os.makedirs(output_folder, exist_ok=True)

long_audios = []

for f in tqdm(os.listdir(input_folder), desc="Scanning audio lengths"):
    if not f.lower().endswith((".wav", ".mp3")):
        continue

    audio = AudioSegment.from_file(os.path.join(input_folder, f))
    if len(audio) > 3000:
        long_audios.append(f)

selected = random.sample(long_audios, 2600)

for f in tqdm(selected, desc="Copying selected audios"):
    shutil.copy(
        os.path.join(input_folder, f),
        os.path.join(output_folder, f)
    )
