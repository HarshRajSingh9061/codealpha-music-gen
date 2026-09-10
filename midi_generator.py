# =====================================================
#  MusicAI — MIDI Data Generator
#  CodeAlpha Task 3 | midi_generator.py
#  Creates classical-style training MIDI files using music21
# =====================================================
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from music21 import stream, note, chord, tempo, meter, key, instrument
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "midi_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Scales & Patterns ─────────────────────────────────
C_MAJOR    = ["C4","D4","E4","F4","G4","A4","B4","C5"]
A_MINOR    = ["A3","B3","C4","D4","E4","F4","G4","A4"]
G_MAJOR    = ["G3","A3","B3","C4","D4","E4","F#4","G4"]
D_MINOR    = ["D4","E4","F4","G4","A4","Bb4","C5","D5"]
F_MAJOR    = ["F4","G4","A4","Bb4","C5","D5","E5","F5"]
E_MINOR    = ["E4","F#4","G4","A4","B4","C5","D5","E5"]

CLASSICAL_PROGRESSIONS = [
    # (root, chord_type)  — I-V-vi-IV
    [("C4","major"), ("G3","major"), ("A3","minor"), ("F3","major")],
    [("G3","major"), ("D3","major"), ("E3","minor"), ("C3","major")],
    [("F3","major"), ("C3","major"), ("D3","minor"), ("Bb3","major")],
    [("A3","minor"), ("E3","major"), ("F3","major"), ("G3","major")],
    [("D3","minor"), ("A3","major"), ("Bb3","major"), ("C3","major")],
]

DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0]

def make_chord(root_str, chord_type):
    """Build a 3-note triad."""
    from music21 import pitch
    r = pitch.Pitch(root_str)
    if chord_type == "major":
        intervals = [0, 4, 7]
    elif chord_type == "minor":
        intervals = [0, 3, 7]
    else:
        intervals = [0, 4, 7]
    pitches = [pitch.Pitch(midi=r.midi + i) for i in intervals]
    return chord.Chord(pitches)

def gen_melody(scale, n_notes=32, seed=None):
    """Generate a melody from a scale with varied rhythm."""
    if seed: random.seed(seed)
    s = stream.Part()
    s.insert(0, instrument.Piano())
    for _ in range(n_notes):
        pitch_str = random.choice(scale)
        dur = random.choice(DURATIONS)
        # Occasional rest
        if random.random() < 0.1:
            r = note.Rest(quarterLength=dur)
            s.append(r)
        else:
            n = note.Note(pitch_str, quarterLength=dur)
            n.volume.velocity = random.randint(64, 100)
            s.append(n)
    return s

def gen_chord_progression(progression, repeats=4, seed=None):
    """Generate a chord-based accompaniment."""
    if seed: random.seed(seed)
    s = stream.Part()
    s.insert(0, instrument.Piano())
    for _ in range(repeats):
        for root, ctype in progression:
            c = make_chord(root, ctype)
            c.quarterLength = 2.0
            s.append(c)
    return s

def gen_arpeggio(scale, n_notes=48, seed=None):
    """Generate an arpeggio pattern."""
    if seed: random.seed(seed)
    s = stream.Part()
    s.insert(0, instrument.Piano())
    for _ in range(n_notes):
        # Pick 3 adjacent notes from the scale as arpeggio
        idx = random.randint(0, len(scale)-3)
        for p in scale[idx:idx+3]:
            n = note.Note(p, quarterLength=0.5)
            n.volume.velocity = random.randint(60, 90)
            s.append(n)
    return s

def gen_bach_style(scale, n_notes=40, seed=None):
    """Simple counterpoint-esque melodic pattern."""
    if seed: random.seed(seed)
    s = stream.Part()
    s.insert(0, instrument.Piano())
    direction = 1
    idx = 0
    for _ in range(n_notes):
        idx = max(0, min(len(scale)-1, idx + direction * random.randint(1, 2)))
        if idx == 0 or idx == len(scale)-1:
            direction *= -1
        dur = random.choice([0.25, 0.5, 0.5, 1.0])
        n = note.Note(scale[idx], quarterLength=dur)
        n.volume.velocity = random.randint(70, 95)
        s.append(n)
    return s

def create_full_piece(name, melody_scale, progression, bpm=90, seed=None):
    """Combine melody + chord progression into a full piece."""
    sc = stream.Score()
    sc.insert(0, tempo.MetronomeMark(number=bpm))
    sc.insert(0, meter.TimeSignature("4/4"))
    sc.insert(0, key.KeySignature(0))

    melody = gen_melody(melody_scale, n_notes=48, seed=seed)
    sc.append(melody)

    chords = gen_chord_progression(progression, repeats=6, seed=seed)
    sc.append(chords)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.mid")
    sc.write("midi", fp=out_path)
    return out_path

def generate_all():
    print("[MusicAI] Generating training MIDI files...")
    files = []

    # 1. C Major melody
    files.append(create_full_piece("piece_01_c_major", C_MAJOR, CLASSICAL_PROGRESSIONS[0], bpm=80, seed=42))

    # 2. A Minor piece
    files.append(create_full_piece("piece_02_a_minor", A_MINOR, CLASSICAL_PROGRESSIONS[3], bpm=70, seed=7))

    # 3. G Major piece
    files.append(create_full_piece("piece_03_g_major", G_MAJOR, CLASSICAL_PROGRESSIONS[1], bpm=95, seed=13))

    # 4. D Minor piece
    files.append(create_full_piece("piece_04_d_minor", D_MINOR, CLASSICAL_PROGRESSIONS[4], bpm=75, seed=21))

    # 5. F Major piece
    files.append(create_full_piece("piece_05_f_major", F_MAJOR, CLASSICAL_PROGRESSIONS[2], bpm=88, seed=99))

    # 6. E Minor piece
    files.append(create_full_piece("piece_06_e_minor", E_MINOR, CLASSICAL_PROGRESSIONS[3], bpm=72, seed=55))

    # 7. Arpeggio-heavy piece
    sc = stream.Score()
    sc.insert(0, tempo.MetronomeMark(number=100))
    sc.insert(0, meter.TimeSignature("3/4"))
    p = gen_arpeggio(C_MAJOR, n_notes=60, seed=33)
    sc.append(p)
    path = os.path.join(OUTPUT_DIR, "piece_07_arpeggio.mid")
    sc.write("midi", fp=path)
    files.append(path)

    # 8. Bach-style counterpoint
    sc = stream.Score()
    sc.insert(0, tempo.MetronomeMark(number=85))
    sc.insert(0, meter.TimeSignature("4/4"))
    p = gen_bach_style(A_MINOR, n_notes=64, seed=77)
    sc.append(p)
    path = os.path.join(OUTPUT_DIR, "piece_08_bach_style.mid")
    sc.write("midi", fp=path)
    files.append(path)

    print(f"[MusicAI] Generated {len(files)} MIDI training files in: {OUTPUT_DIR}")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    return files

if __name__ == "__main__":
    generate_all()
