"""
Download DAC Model Weights using the official DAC library
UPDATED: Downloads the 16kbps version (128 latent channels)
"""

import dac
import urllib.request
import os
from pathlib import Path

print("Downloading DAC model weights for 44kHz 16kbps...")
print("This may take a few minutes depending on your connection.\n")

# Direct URL for 16kbps model
url = "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps_1.0.0.pth"
output_path = Path.home() / ".cache/descript/dac/weights_44khz_16kbps_1.0.0.pth"

# Create directory if needed
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}\n")
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='')
    
    urllib.request.urlretrieve(url, output_path, show_progress)
    print("\n\n✅ Download successful!")
    
    # Verify it's the right model
    print("Verifying model...")
    import torch
    model = dac.DAC.load(str(output_path))
    print(f"Model latent channels: {model.latent_dim}")
    
    if model.latent_dim == 128:
        print("✅ Correct! This is the 16kbps model with 128 latent channels.")
        print("   This matches your trained checkpoint!")
    else:
        print(f"⚠️  Warning: Expected 128 channels, but got {model.latent_dim}")
    
    print(f"\nModel saved to: {output_path}")
    print(f"\nYou can now run:")
    print(f"  python app.py models/1k_5k_5p_5s.pt")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nTry downloading manually from:")
    print("  https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps_1.0.0.pth")
    print(f"\nSave it to: {output_path}")