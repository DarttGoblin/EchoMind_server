import csv
import os
import numpy as np

input_folder = "../embeddings/input_data_embeddings/"
output_folder = "../embeddings/output_data_embeddings/"
prompt_folder = "../embeddings/prompts_embeddings/"

sound_effects = ["drizzle", "forest", "hail", "leaves", "rain", "river", "snowfall", "thunderstorm", "waterfall", "wind"]

prompts = {}
for effect in sound_effects:
    file_path = f"{prompt_folder}embeddings_{effect}.npy"
    prompts[effect] = np.load(file_path, allow_pickle=True).tolist()

rows = []
for i in range(1, 1001):
    input_path = f"{input_folder}{i}.npy"
    if not os.path.exists(input_path):
        continue

    for effect in sound_effects:
        prompt_path = f"{prompt_folder}embeddings_{effect}.npy"

        for p in prompts[effect]:
            output_path = f"{output_folder}{i}_{effect}.npy"
            if not os.path.exists(output_path):
                continue

            rows.append([prompt_path, input_path, output_path])

with open("../50000_datapoint.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_embedding", "input_embedding", "output_embedding"])
    writer.writerows(rows)
