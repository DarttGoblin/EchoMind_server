import os
import soundfile as sf

folder_path = "../.other_files/original_data/audios_splited_merged_choosed_final/english"
target_length = 3
tolerance = 0.01

shorter = []
exact = []
longer = []

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(".wav"):
        continue

    path = os.path.join(folder_path, filename)
    data, sr = sf.read(path)
    duration = len(data) / sr

    if duration < target_length - tolerance:
        shorter.append((filename, duration))
    elif duration > target_length + tolerance:
        longer.append((filename, duration))
    else:
        exact.append((filename, duration))

print(f"Total files: {len(shorter) + len(exact) + len(longer)}")
print(f"Exact {target_length}s: {len(exact)}")
print(f"Shorter than {target_length}s: {len(shorter)}")
print(f"Longer than {target_length}s: {len(longer)}")

if shorter:
    print("\nShorter files:")
    for f, d in shorter[:10]:
        print(f"{f} -> {d:.2f}s")

if longer:
    print("\nLonger files:")
    for f, d in longer[:10]:
        print(f"{f} -> {d:.2f}s")
