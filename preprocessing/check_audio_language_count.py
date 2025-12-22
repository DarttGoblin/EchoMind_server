import os
from tqdm import tqdm

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

print("English:", counts["english"])
print("Egyptian:", counts["egyptian"])
print("Frensh:", counts["frensh"])
