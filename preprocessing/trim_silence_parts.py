from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tqdm import tqdm
import os

input_folder = "../.other_files/original_data/audios_not_preprocessed"
output_folder = "../.other_files/original_data/audios_silence_trimmed"
os.makedirs(output_folder, exist_ok=True)

files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".wav", ".mp3"))
]

for filename in tqdm(files, desc="Trimming leading and trailing silence"):
    file_path = os.path.join(input_folder, filename)
    audio = AudioSegment.from_file(file_path)

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=200,
        silence_thresh=-40
    )

    if nonsilent_ranges:
        start_trim = nonsilent_ranges[0][0]
        end_trim = nonsilent_ranges[-1][1]
        trimmed_audio = audio[start_trim:end_trim]
    else:
        trimmed_audio = audio

    trimmed_audio.export(
        os.path.join(output_folder, filename),
        format="wav"
    )

print("Leading and trailing silence trimmed from all audios.")