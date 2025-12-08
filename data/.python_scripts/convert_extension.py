from pydub import AudioSegment
import os

mp3_folder = "sound.effects"
wav_folder = "sound.effects"
os.makedirs(wav_folder, exist_ok=True)

for filename in os.listdir(mp3_folder):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(mp3_folder, filename)
        wav_path = os.path.join(wav_folder, filename.replace(".mp3", ".wav"))
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio.export(wav_path, format="wav")