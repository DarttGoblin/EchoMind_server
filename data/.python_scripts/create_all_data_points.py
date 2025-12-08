import csv
import os
import pandas as pd

input_folder = "../audios/input.data/"
output_folder = "../audios/output.data/"
prompt_file = "../prompts.xlsx"

sound_effects = ["drizzle", "forest", "hail", "leaves", "rain", "river", "snowfall", "thunderstorm", "waterfall", "wind"]

# Load prompts from Excel
prompts_df = pd.read_excel(prompt_file, sheet_name=None)
prompts = {effect: prompts_df[effect].dropna().tolist() for effect in sound_effects}

rows = []

input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".wav")])[:500]
output_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".wav")])[:5000]

# Generate 50k datapoints
for i, inp in enumerate(input_files):
    for effect in sound_effects:
        for prompt in prompts[effect]:
            for out in output_files[i*10:(i+1)*10*len(prompts[effect])//len(sound_effects)]:
                rows.append([prompt, os.path.join(input_folder, inp), os.path.join(output_folder, out)])

with open("../50000_datapoint.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "input_audio", "output_audio"])
    writer.writerows(rows)
