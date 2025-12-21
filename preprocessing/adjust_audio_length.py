from pydub import AudioSegment
import os
from tqdm import tqdm

input_folder = "../.other_files/original_data/audios_splited_merged_choosed/frensh"
output_folder = "../.other_files/original_data/audios_splited_merged_choosed_final/frensh"

os.makedirs(output_folder, exist_ok=True)

for f in tqdm(os.listdir(input_folder), desc="Trimming audios to 5 seconds"):
    if not f.lower().endswith((".wav", ".mp3")):
        continue

    audio = AudioSegment.from_file(os.path.join(input_folder, f))

    trimmed = audio[:3000]

    trimmed.export(
        os.path.join(output_folder, f),
        format="wav"
    )
