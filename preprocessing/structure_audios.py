import os
import shutil

source_root = "audio"  
destination = "merged_audios"

os.makedirs(destination, exist_ok=True)

counter = 1

for root, dirs, files in os.walk(source_root):
    for f in files:
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            src = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            dst = os.path.join(destination, f"audio_{counter}{ext}")
            shutil.copy2(src, dst)
            counter += 1
