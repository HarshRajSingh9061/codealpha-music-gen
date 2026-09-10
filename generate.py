# =====================================================
#  MusicAI — Music Generation Script
#  CodeAlpha Task 3 | generate.py
#  Generates new MIDI using trained LSTM model
# =====================================================
import sys, os, random, pickle
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from music21 import stream, note, chord, tempo, meter, instrument

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_vocab():
    """Load the trained Keras model and vocab from checkpoints."""
    from tensorflow.keras.models import load_model

    model_path = os.path.join(CHECKPOINT_DIR, "music_model.keras")
    best_path  = os.path.join(CHECKPOINT_DIR, "best_weights.keras")

    if os.path.exists(model_path):
        model = load_model(model_path)
    elif os.path.exists(best_path):
        from preprocess import load_cache
        from model import build_lstm_model
        cache   = load_cache()
        n_vocab = len(cache["vocab"])
        seq_len = cache["seq_len"]
        model   = build_lstm_model(seq_len, n_vocab)
        model.load_weights(best_path)
    else:
        raise FileNotFoundError("No trained model found. Please train first.")

    cache = load_cache()
    return model, cache


def generate_notes(model, cache, n_generate=100, temperature=1.0, seed_idx=None):
    """
    Generate a sequence of note tokens using the trained LSTM.

    temperature: > 1 = more random/creative, < 1 = more deterministic
    """
    int_to_note = cache["int_to_note"]
    note_to_int = cache["note_to_int"]
    all_notes   = cache["all_notes"]
    seq_len     = cache["seq_len"]
    n_vocab     = len(cache["vocab"])

    # Pick a random seed from the training data
    if seed_idx is None:
        seed_idx = random.randint(0, len(all_notes) - seq_len - 1)

    pattern = [note_to_int[n] for n in all_notes[seed_idx : seed_idx + seq_len]]

    generated = []
    for _ in range(n_generate):
        x = np.reshape(pattern, (1, seq_len, 1)).astype("float32") / n_vocab

        # Temperature sampling
        logits = model.predict(x, verbose=0)[0]
        logits = np.log(logits + 1e-9) / temperature
        exp_l  = np.exp(logits - np.max(logits))
        probs  = exp_l / exp_l.sum()
        idx    = np.random.choice(n_vocab, p=probs)

        generated.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]  # sliding window

    return generated


def tokens_to_midi(tokens: list, bpm: int = 90, output_name: str = "generated") -> str:
    """
    Convert a list of note/chord/rest tokens into a MIDI file.
    Returns the path to the saved .mid file.
    """
    output_stream = stream.Stream()
    output_stream.append(tempo.MetronomeMark(number=bpm))
    output_stream.append(meter.TimeSignature("4/4"))
    output_stream.append(instrument.Piano())

    DEFAULT_DUR = 0.5  # quarter note default

    for token in tokens:
        if token == "R":
            output_stream.append(note.Rest(quarterLength=DEFAULT_DUR))
        elif "." in token:
            # Chord
            pitches = token.split(".")
            try:
                c = chord.Chord(pitches, quarterLength=DEFAULT_DUR)
                c.volume.velocity = 80
                output_stream.append(c)
            except Exception:
                pass  # skip malformed chord
        else:
            try:
                n = note.Note(token, quarterLength=DEFAULT_DUR)
                n.volume.velocity = 80
                output_stream.append(n)
            except Exception:
                pass  # skip malformed note

    out_path = os.path.join(OUTPUT_DIR, f"{output_name}.mid")
    output_stream.write("midi", fp=out_path)
    print(f"[Generate] MIDI saved -> {out_path}")
    return out_path


def generate(n_notes=100, temperature=1.0, bpm=90, output_name="generated", seed_idx=None):
    """Full generation pipeline."""
    print(f"[Generate] Loading model...")
    model, cache = load_model_and_vocab()

    print(f"[Generate] Generating {n_notes} notes (temp={temperature}, bpm={bpm})...")
    tokens = generate_notes(model, cache, n_generate=n_notes, temperature=temperature, seed_idx=seed_idx)

    print(f"[Generate] Token sample: {tokens[:10]}")
    midi_path = tokens_to_midi(tokens, bpm=bpm, output_name=output_name)
    return midi_path, tokens


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes",       type=int,   default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bpm",         type=int,   default=90)
    parser.add_argument("--output",      type=str,   default="generated")
    args = parser.parse_args()

    midi_path, tokens = generate(
        n_notes=args.notes,
        temperature=args.temperature,
        bpm=args.bpm,
        output_name=args.output,
    )
    print(f"[Done] Generated MIDI: {midi_path}")
