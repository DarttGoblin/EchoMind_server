"""
Download DAC Model Weights using the official DAC library
"""

import dac

print("Downloading DAC model weights for 44kHz...")
print("This may take a few minutes depending on your connection.\n")

try:
    # Download the 44kHz 8kbps model (default)
    model_path = dac.utils.download(model_type="44khz")
    
    print(f"✅ Download successful!")
    print(f"Model saved to: {model_path}")
    print(f"\nYou can now run:")
    print(f"  python app.py models/1k_5k_5p_5s.pt")
    
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("\nAlternative: Run this command directly:")
    print("  python -m dac download --model_type 44khz")