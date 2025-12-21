from pydub import AudioSegment
from tqdm import tqdm
import os

source_folder = "../data/audios_3s/input_audios"
output_folder = "../data/audios_3s/output_audios"
effects_folder = "../data/soundeffects"

os.makedirs(output_folder, exist_ok=True)

effects = ["dog", "cat", "bird", "rain", "thunder"]
effect_audios = {}

# Load all effect audios once
for effect in effects:
    effect_path = os.path.join(effects_folder, f"{effect}.wav")
    effect_audios[effect] = AudioSegment.from_wav(effect_path)

# Process all .wav files in the source folder
audio_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".wav")]

for audio_file in tqdm(audio_files, desc="Processing Audios"):
    audio_path = os.path.join(source_folder, audio_file)
    original_audio = AudioSegment.from_wav(audio_path)
    audio_len = len(original_audio)

    for effect_name, effect_audio in effect_audios.items():
        effect_len = len(effect_audio)

        if effect_len < audio_len:
            repeats = (audio_len // effect_len) + 1
            effect_segment = effect_audio * repeats
            effect_segment = effect_segment[:audio_len]
        else:
            effect_segment = effect_audio[:audio_len]

        merged = original_audio.overlay(effect_segment)
        output_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}_{effect_name}.wav")
        merged.export(output_path, format="wav")