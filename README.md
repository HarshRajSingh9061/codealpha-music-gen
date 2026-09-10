# MusicAI — Music Generation with AI
### CodeAlpha Internship | Task 3

An AI-powered music generation system using **LSTM neural networks** and **music21** MIDI processing — built for the CodeAlpha internship.

---

## Project Structure

```
codealpha-music-gen/
├── app.py              <- Flask web dashboard
├── midi_generator.py   <- Generate classical training MIDI files
├── preprocess.py       <- MIDI parsing + sequence creation (music21)
├── model.py            <- Stacked LSTM model (TensorFlow/Keras)
├── train.py            <- Training script with live progress
├── generate.py         <- Generate new music from trained model
├── midi_data/          <- Training MIDI files (auto-generated)
├── output/             <- Generated MIDI files saved here
├── checkpoints/        <- Model weights + preprocessed cache
├── static/
│   ├── style.css       <- Glassmorphism dark UI
│   └── app.js          <- Frontend polling + live charts
├── templates/
│   └── index.html      <- Web dashboard
└── requirements.txt
```

---

## Pipeline

```
Step 1: Generate MIDI Data
  midi_generator.py -> 8 classical pieces (melodies, chords, arpeggios, Bach-style)

Step 2: Preprocess
  music21 parses MIDI -> extracts note tokens -> builds vocabulary
  Sliding window sequences of length 50 -> (X, y) arrays

Step 3: Train LSTM
  Architecture: Input(50,1) -> LSTM(512) -> LSTM(512) -> LSTM(256) -> Dense(256) -> Softmax(n_vocab)
  Training: Adam optimizer, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

Step 4: Generate
  Seed sequence -> temperature sampling -> 100+ new notes -> music21 MIDI export
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch web dashboard
python app.py
# Open http://127.0.0.1:5001

# --- OR run each step from CLI ---

# Generate MIDI training data
python midi_generator.py

# Preprocess
python preprocess.py

# Train (50 epochs default)
python train.py --epochs 50 --batch_size 64 --lr 0.001

# Generate music
python generate.py --notes 100 --temperature 1.0 --bpm 90
```

---

## CodeAlpha Requirements Fulfilled

- Collect MIDI music data -> auto-generated with music21
- Preprocess using music21 (tokenize notes/chords, sequences)
- Build LSTM deep learning model (stacked 3-layer LSTM)
- Train on dataset
- Convert generated sequences to MIDI and save as .mid files

*Made with TensorFlow, music21, Flask*
