import csv
import os
import pandas as pd
from tqdm import tqdm

input_folder = "../data/audios_3s/input_audios"
output_folder = "../data/audios_3s/output_audios"
prompt_file = "../data/prompt.csv"

effects = ["rain", "bird", "dog", "cat", "thunder"]

df = pd.read_csv(prompt_file)

prompts = {
    effect: df[df["soundeffect"] == effect]["prompt"].dropna().iloc[0]
    for effect in effects
}

def norm(p):
    return p.replace("\\", "/")

rows = []

input_files = sorted(
    [f for f in os.listdir(input_folder) if f.endswith(".wav")]
)[:3000]

for inp in tqdm(input_files):
    inp_path = os.path.join(input_folder, inp)
    base_name = os.path.splitext(inp)[0]

    for effect in effects:
        out_file = f"{base_name}_{effect}.wav"
        out_path = os.path.join(output_folder, out_file)

        if not os.path.exists(out_path):
            continue

        rows.append([
            prompts[effect],
            norm(inp_path),
            norm(out_path)
        ])

with open("15000_datapoints.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "input_audio_path", "output_audio_path"])
    writer.writerows(rows)
