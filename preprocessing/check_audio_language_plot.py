import os
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_folder = "../data/audios_3s/input_audios"

counts = {
    "english": 0,
    "egyptian": 0,
    "frensh": 0
}

for f in tqdm(os.listdir(dataset_folder), desc="Counting audios"):
    name = f.lower()
    if name.endswith((".wav", ".mp3")):
        if "_english" in name:
            counts["english"] += 1
        elif "_egyptian" in name:
            counts["egyptian"] += 1
        elif "_frensh" in name:
            counts["frensh"] += 1

languages = list(counts.keys())
values = list(counts.values())

plt.figure()
plt.bar(languages, values)
plt.xlabel("Language")
plt.ylabel("Number of audios")
plt.title("Language distribution in 2000-audio dataset")
plt.show()
