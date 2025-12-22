EchoMind: Speech Audio Editing with Automatic Sound Effect Insertion

🚨 HOW TO RUN INFERENCE 🚨

1. Clone EchoMind interface repo through this link: https://darttgoblin.github.io/EchoMind/EchoMind.html
2. Clone EchoMind backend server repo
3. Download the models from google drive though this link: https://drive.google.com/drive/folders/18ffQVEJmYp6j8aMZgL3yBF4YfF8Yu78h?usp=sharing
4. Install the dependencies using the following command: pip install flask flask-cors torch torchaudio soundfile transformers einops werkzeug tqdm descript-audio-codec
5. Run app.py ../models/[chosen model].pt
6. Test the chosen model on you audio

EchoMind is a research-oriented project that explores automatic sound effect insertion into speech audio using deep learning. The goal is to simplify audio editing by enabling a model to understand when and how to apply sound effects based on a textual prompt, while preserving the natural structure and intelligibility of the original speech.

This project focuses exclusively on sound-effect augmentation for speech audio and is designed as a prompt-conditioned audio-to-audio learning task.

Project Motivation

Adding sound effects to speech audio is traditionally a manual process that requires both technical expertise and careful timing. Existing datasets and tools often suffer from low audio quality, poor task alignment, or lack of paired input–output examples suitable for learning precise audio transformations.

EchoMind addresses these limitations by:

Curating high-quality multilingual speech data

Designing controlled sound-effect insertion pipelines

Structuring the task as prompt-conditioned audio editing rather than raw generation

Dataset Overview

The project introduces a curated dataset of speech audios paired with sound-effect-augmented outputs, guided by textual prompts.

Speech Data

Three high-quality speech datasets are used:

Egyptian Arabic: clean recordings with diverse vocal characteristics

French: audiobook-style speech with clear articulation

English: multi-speaker virtual assistant commands

Low-quality large-scale datasets (e.g., ARCA23K, FSD50K) were intentionally excluded due to noise, inconsistent loudness, and weak alignment with audio-editing tasks.

Sound Effects

Sound effects were initially sourced from Pixabay but were discarded due to poor perceptual quality, inconsistent duration, and weak identifiability.
The final sound effects were extracted from high-quality YouTube videos and carefully selected to be:

clearly distinguishable

noise-free

consistent in perceptual strength

Final sound effects:

rain

dog barking

cat meowing

bird singing

thunder

Prompt Design

Two prompt strategies were explored:

Multi-prompt setting: five linguistic variants per sound effect (e.g., add, include, overlay)

Single-prompt setting: a simplified prompt of the form
add [soundeffect]

This allows analysis of prompt diversity versus dataset scale.

Preprocessing Pipeline

All audio data undergoes a standardized preprocessing pipeline:

Leading and trailing silence trimming (internal pauses preserved)

Audio concatenation with short silence gaps to avoid excessive short clips

Length normalization to fixed durations (5s and 3s depending on subset)

Sound-effect merging to generate paired input–output audios

This design preserves natural speech dynamics while ensuring training consistency.

Dataset Subsets

Three structured subsets are used:

Subset A
1,000 input audios (5s)
5 sound effects × 5 prompts
→ 25,000 datapoints

Subset B
1,000 input audios (5s)
5 sound effects × 1 prompt
→ 5,000 datapoints

Subset C
2,000 input audios (3s, English-only)
5 sound effects × 1 prompt
→ 10,000 datapoints

Each datapoint consists of:

input audio path

output audio path

textual prompt

All data is stored in CSV format for efficient loading and alignment.

Task Definition

The learning task is formulated as:

Input:

speech audio

textual prompt describing a sound-effect operation

Output:

speech audio with the correct sound effect applied at appropriate temporal locations

The model is expected to learn context-aware sound effect insertion rather than naive overlay.

Availability

Dataset (Kaggle):
Speech Audios With Sound Effects for Audio Editing
https://www.kaggle.com/datasets/yassinebazgour/speech-audios-with-sound-effects-for-audio-editing

Code:
Preprocessing, dataset construction, and experiments are provided in this repository.

Intended Use

This project is intended for:

research in audio editing and audio-to-audio learning

prompt-conditioned audio transformation

multimodal learning involving text and audio

It is not intended for commercial audio production without further refinement.