import os
import shutil
from tqdm import tqdm

input_dir = "../data/audios/output_audios"

files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

files.sort(key=lambda x: int(x.split("_")[0]))

ranges = [
    (1, 200),
    (201, 400),
    (401, 600),
    (601, 800),
    (801, 1000)
]

for i in range(5):
    os.makedirs(os.path.join(input_dir, f"part_{i+1}"), exist_ok=True)

for file in tqdm(files):
    idx = int(file.split("_")[0])

    for i, (start, end) in enumerate(ranges):
        if start <= idx <= end:
            shutil.copy(
                os.path.join(input_dir, file),
                os.path.join(input_dir, f"part_{i+1}", file)
            )
            break
