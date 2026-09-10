# =====================================================
#  MusicAI — MIDI Preprocessing
#  CodeAlpha Task 3 | preprocess.py
#  Extracts note sequences from MIDI using music21
# =====================================================
import sys, os, glob, pickle
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from music21 import converter, instrument, note, chord, stream

MIDI_DIR   = os.path.join(os.path.dirname(__file__), "midi_data")
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CACHE_DIR, exist_ok=True)

SEQUENCE_LEN = 50   # number of notes per input sequence


def extract_notes(midi_path: str) -> list:
    """
    Parse a MIDI file and extract a flat list of note/chord tokens.
    Notes:  "C4", "F#5", etc.
    Chords: "C4.E4.G4"  (pitches joined by '.')
    Rests:  "R"
    """
    try:
        midi = converter.parse(midi_path)
    except Exception as e:
        print(f"  [WARN] Could not parse {midi_path}: {e}")
        return []

    elements_to_parse = None
    try:
        parts = instrument.partitionByInstrument(midi)
        if parts:
            elements_to_parse = parts.parts[0].recurse()
        else:
            elements_to_parse = midi.flat.notes
    except Exception:
        elements_to_parse = midi.flat.notes

    tokens = []
    for el in elements_to_parse:
        if isinstance(el, note.Note):
            tokens.append(str(el.pitch))
        elif isinstance(el, chord.Chord):
            tokens.append(".".join(str(p) for p in el.pitches))
        elif isinstance(el, note.Rest):
            tokens.append("R")

    return tokens


def build_vocabulary(all_notes: list):
    """Map each unique token to an integer and back."""
    vocab   = sorted(set(all_notes))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for i, n in enumerate(vocab)}
    return vocab, note_to_int, int_to_note


def create_sequences(all_notes, note_to_int, seq_len=SEQUENCE_LEN):
    """
    Build (X, y) pairs:
      X[i] = seq_len notes starting at i  (as integers)
      y[i] = the note right after          (one-hot)
    """
    n_vocab = len(note_to_int)
    X, y    = [], []

    for i in range(len(all_notes) - seq_len):
        seq_in  = all_notes[i : i + seq_len]
        seq_out = all_notes[i + seq_len]
        X.append([note_to_int[n] for n in seq_in])
        y.append(note_to_int[seq_out])

    n_patterns = len(X)
    # Reshape X → (n_patterns, seq_len, 1) and normalise
    X = np.reshape(X, (n_patterns, seq_len, 1)).astype("float32") / n_vocab
    # One-hot encode y
    y_onehot = np.zeros((n_patterns, n_vocab), dtype="float32")
    for i, val in enumerate(y):
        y_onehot[i, val] = 1.0

    return X, y_onehot


def preprocess_all(midi_dir=MIDI_DIR, seq_len=SEQUENCE_LEN, verbose=True):
    """
    Full pipeline:
      1. Find all MIDI files
      2. Extract notes from each
      3. Build vocabulary
      4. Create sequences
      5. Save to cache
    Returns: (X, y, vocab, note_to_int, int_to_note)
    """
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + \
                 glob.glob(os.path.join(midi_dir, "*.midi"))

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    if verbose:
        print(f"[Preprocess] Found {len(midi_files)} MIDI files")

    all_notes = []
    for path in midi_files:
        if verbose:
            print(f"  Parsing: {os.path.basename(path)}")
        tokens = extract_notes(path)
        if verbose:
            print(f"    -> {len(tokens)} tokens extracted")
        all_notes.extend(tokens)

    if verbose:
        print(f"[Preprocess] Total tokens: {len(all_notes)}")

    vocab, note_to_int, int_to_note = build_vocabulary(all_notes)

    if verbose:
        print(f"[Preprocess] Vocabulary size: {len(vocab)} unique tokens")

    X, y = create_sequences(all_notes, note_to_int, seq_len=seq_len)

    if verbose:
        print(f"[Preprocess] Training sequences: {X.shape[0]}")

    # Save cache
    cache = {
        "X":           X,
        "y":           y,
        "vocab":       vocab,
        "note_to_int": note_to_int,
        "int_to_note": int_to_note,
        "all_notes":   all_notes,
        "seq_len":     seq_len,
    }
    cache_path = os.path.join(CACHE_DIR, "preprocessed.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    if verbose:
        print(f"[Preprocess] Cache saved -> {cache_path}")

    return X, y, vocab, note_to_int, int_to_note, all_notes


def load_cache():
    """Load previously preprocessed data."""
    cache_path = os.path.join(CACHE_DIR, "preprocessed.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("No cache found. Run preprocess first.")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return cache


if __name__ == "__main__":
    preprocess_all(verbose=True)
