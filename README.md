This project aims to develop a system that simplify audio editing tasks by providing the service for the user
to enter text prompt of the desired edit and the input audio, and the server returns the audio edited through this Pipeline:

1. Data Loading

Reads a CSV with pairs of audio files (before/after effect) and text descriptions
Takes a subset (5,000 samples) to make training faster
Splits data: 70% training, 15% validation, 15% testing

2. Model Architecture
Two main parts work together:

Text Encoder (BERT): Understands the text prompt
U-Net: Processes the audio while using text information to guide the effect

The U-Net:

Encoder: Compresses audio down (like zooming out)
Bottleneck: Processes at lowest resolution
Decoder: Expands back up (like zooming in)
Cross-Attention: Lets audio "look at" text to know what effect to apply

3. Training

Shows model: input audio + text → expects output audio
Compares prediction to real output
Adjusts model to make better predictions
Saves best version when validation loss is lowest

4. Results

Plots training curves to see learning progress
Tests final model on unseen test data
Saves everything (model, plots, metrics)