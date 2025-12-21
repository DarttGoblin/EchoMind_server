import os
import soundfile as sf
import matplotlib.pyplot as plt

folder_path = "../../data/audios/input_audios"
target_length_sec = 5

count = 0
lengths = []
names = []

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        data, samplerate = sf.read(file_path)

        duration = len(data) / samplerate
        lengths.append(duration)
        names.append(filename)

        if abs(duration - target_length_sec) < 0.01:
            count += 1

print(f"Number of audio files with {target_length_sec} seconds: {count}")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(lengths)
plt.axhline(target_length_sec, linestyle='--')
plt.xlabel("Audio Index")
plt.ylabel("Duration (seconds)")
plt.title("Audio Lengths")
plt.show()
