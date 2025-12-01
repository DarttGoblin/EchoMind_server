import os
import soundfile as sf

folder_path = ".other_files/original.data"
target_length_sec = 10
count = 0

for filename in os.listdir(folder_path):
    if filename.endswith((".wav")):
        file_path = os.path.join(folder_path, filename)
        data, samplerate = sf.read(file_path)
        duration = len(data) / samplerate
        if abs(duration - target_length_sec) < 0.01:
            count += 1

print(f"Number of audio files with {target_length_sec} seconds: {count}")