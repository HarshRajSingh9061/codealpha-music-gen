# =====================================================
#  MusicAI — Flask Web Application
#  CodeAlpha Task 3 | app.py
# =====================================================
import sys, os, json, threading, glob, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "output")
MIDI_DIR       = os.path.join(os.path.dirname(__file__), "midi_data")
STATUS_FILE    = os.path.join(CHECKPOINT_DIR, "training_status.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global training thread tracker
_training_thread = None
_generation_lock  = threading.Lock()


# ── Helpers ──────────────────────────────────────────
def read_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"state": "idle"}

def write_status(d):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def model_exists():
    return os.path.exists(os.path.join(CHECKPOINT_DIR, "music_model.keras")) or \
           os.path.exists(os.path.join(CHECKPOINT_DIR, "best_weights.keras"))

def cache_exists():
    return os.path.exists(os.path.join(CHECKPOINT_DIR, "preprocessed.pkl"))

def midi_data_exists():
    return bool(glob.glob(os.path.join(MIDI_DIR, "*.mid")))


# ── Routes ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    st = read_status()
    st["model_ready"]     = model_exists()
    st["cache_ready"]     = cache_exists()
    st["midi_data_ready"] = midi_data_exists()
    st["output_files"]    = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith((".mid", ".midi"))
    ], reverse=True)
    return jsonify(st)


@app.route("/generate-midi", methods=["POST"])
def generate_midi_data():
    """Step 1 — Generate training MIDI files."""
    def _run():
        write_status({"state": "generating_midi"})
        try:
            from midi_generator import generate_all
            files = generate_all()
            write_status({"state": "idle", "midi_files": len(files)})
        except Exception as e:
            write_status({"state": "error", "message": str(e)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Generating MIDI training data..."})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """Step 2 — Preprocess MIDI files into sequences."""
    def _run():
        write_status({"state": "preprocessing"})
        try:
            from preprocess import preprocess_all
            X, y, vocab, note_to_int, int_to_note, all_notes = preprocess_all(verbose=True)
            write_status({
                "state":       "idle",
                "sequences":   int(X.shape[0]),
                "vocab_size":  len(vocab),
                "total_notes": len(all_notes),
            })
        except Exception as e:
            write_status({"state": "error", "message": str(e)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Preprocessing MIDI data..."})


@app.route("/train", methods=["POST"])
def train():
    """Step 3 — Train the LSTM model."""
    global _training_thread
    if _training_thread and _training_thread.is_alive():
        return jsonify({"ok": False, "message": "Training already in progress."})

    data       = request.get_json() or {}
    epochs     = int(data.get("epochs", 50))
    batch_size = int(data.get("batch_size", 64))
    lr         = float(data.get("lr", 0.001))

    def _run():
        try:
            from train import train as run_train
            run_train(epochs=epochs, batch_size=batch_size, learning_rate=lr, use_cache=True)
        except Exception as e:
            write_status({"state": "error", "message": str(e)})

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()
    return jsonify({"ok": True, "message": f"Training started ({epochs} epochs)..."})


@app.route("/generate", methods=["POST"])
def generate():
    """Step 4 — Generate new music."""
    if not model_exists():
        return jsonify({"ok": False, "message": "No trained model found. Please train first."})

    data        = request.get_json() or {}
    n_notes     = int(data.get("n_notes", 100))
    temperature = float(data.get("temperature", 1.0))
    bpm         = int(data.get("bpm", 90))
    ts          = int(time.time())
    output_name = f"music_{ts}"

    def _run():
        write_status({"state": "generating"})
        try:
            from generate import generate as do_generate
            midi_path, tokens = do_generate(
                n_notes=n_notes,
                temperature=temperature,
                bpm=bpm,
                output_name=output_name,
            )
            write_status({
                "state":       "idle",
                "last_generated": os.path.basename(midi_path),
            })
        except Exception as e:
            write_status({"state": "error", "message": str(e)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"ok": True, "output_name": output_name, "message": "Generating music..."})


@app.route("/download/<filename>")
def download(filename):
    """Serve a generated MIDI file."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/play/<filename>")
def play(filename):
    """Stream a generated MIDI file for playback."""
    return send_from_directory(OUTPUT_DIR, filename, mimetype="audio/midi")


@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": model_exists(),
        "cache_ready": cache_exists(),
        "midi_ready":  midi_data_exists(),
    })


if __name__ == "__main__":
    print("[MusicAI] CodeAlpha Task 3: Music Generation with AI")
    print("[INFO]  Running at: http://127.0.0.1:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
