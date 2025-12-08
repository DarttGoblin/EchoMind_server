import os

folder_path = "input.data"

for i, filename in enumerate(sorted(os.listdir(folder_path)), start=1):
    file_ext = '.wav'
    new_name = f"{i}{file_ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
