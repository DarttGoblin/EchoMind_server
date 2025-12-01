from pydub import AudioSegment
from tqdm import tqdm
import os

source_folder = "../input.data"
effects_folder = "../sound.effects"
output_folder = "../output.data"

os.makedirs(output_folder, exist_ok=True)

effects = [
    "rain",
    "thunderstorm",
    "drizzle",
    "hail",
    "snowfall",
    "wind",
    "leaves",
    "forest",
    "river",
    "waterfall"
]

effect_audios = {}
for effect in effects:
    effect_path = os.path.join(effects_folder, f"{effect}.wav")
    effect_audios[effect] = AudioSegment.from_wav(effect_path)

for i in tqdm(range(1, 1001), desc="Processing Audios"):
    audio_path = os.path.join(source_folder, f"{i}.wav")
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
        output_path = os.path.join(output_folder, f"{i}_{effect_name}.wav")
        merged.export(output_path, format="wav")
