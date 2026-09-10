# =====================================================
#  MusicAI — Model Training Script
#  CodeAlpha Task 3 | train.py
# =====================================================
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
)

from preprocess import preprocess_all, load_cache
from model import build_lstm_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STATUS_FILE = os.path.join(CHECKPOINT_DIR, "training_status.json")


def save_status(status: dict):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


class StatusCallback(Callback):
    """Write training progress to a JSON file for the web UI to poll."""
    def __init__(self, n_vocab, total_epochs):
        super().__init__()
        self.n_vocab       = n_vocab
        self.total_epochs  = total_epochs
        self.history       = []
        self.start_time    = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.start_time
        entry = {
            "epoch":    epoch + 1,
            "loss":     round(float(logs.get("loss", 0)), 4),
            "accuracy": round(float(logs.get("accuracy", 0)) * 100, 2),
            "val_loss": round(float(logs.get("val_loss", 0)), 4),
            "val_acc":  round(float(logs.get("val_accuracy", 0)) * 100, 2),
        }
        self.history.append(entry)

        save_status({
            "state":         "training",
            "epoch":         epoch + 1,
            "total_epochs":  self.total_epochs,
            "progress_pct":  round((epoch + 1) / self.total_epochs * 100, 1),
            "n_vocab":       self.n_vocab,
            "latest":        entry,
            "history":       self.history,
            "elapsed_sec":   round(elapsed, 1),
        })
        print(f"  Epoch {epoch+1}/{self.total_epochs} — loss: {entry['loss']} — acc: {entry['accuracy']}%")

    def on_train_end(self, logs=None):
        save_status({
            "state":   "done",
            "history": self.history,
            "n_vocab": self.n_vocab,
        })
        print("[Train] Training complete!")


def train(epochs=50, batch_size=64, learning_rate=0.001, use_cache=True):
    print("[Train] Starting MusicAI LSTM training...")

    save_status({"state": "preprocessing"})

    # 1. Load or preprocess data
    if use_cache:
        try:
            cache      = load_cache()
            X          = cache["X"]
            y          = cache["y"]
            vocab      = cache["vocab"]
            note_to_int = cache["note_to_int"]
            int_to_note = cache["int_to_note"]
            seq_len    = cache["seq_len"]
            print(f"[Train] Loaded cache: {X.shape[0]} sequences, vocab={len(vocab)}")
        except FileNotFoundError:
            X, y, vocab, note_to_int, int_to_note, _ = preprocess_all()
            seq_len = X.shape[1]
    else:
        X, y, vocab, note_to_int, int_to_note, _ = preprocess_all()
        seq_len = X.shape[1]

    n_vocab = len(vocab)
    print(f"[Train] X: {X.shape}, y: {y.shape}, vocab: {n_vocab}")

    save_status({"state": "building_model", "n_vocab": n_vocab})

    # 2. Build model
    model = build_lstm_model(seq_len=seq_len, n_vocab=n_vocab, learning_rate=learning_rate)
    model.summary()

    # 3. Callbacks
    weight_path = os.path.join(CHECKPOINT_DIR, "best_weights.keras")
    callbacks = [
        ModelCheckpoint(weight_path, monitor="loss", save_best_only=True, verbose=0),
        EarlyStopping(monitor="loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        StatusCallback(n_vocab=n_vocab, total_epochs=epochs),
    ]

    save_status({"state": "training", "epoch": 0, "total_epochs": epochs, "n_vocab": n_vocab, "history": []})

    # 4. Train
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.1,
        verbose=0,
    )

    # 5. Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "music_model.keras")
    model.save(final_path)
    print(f"[Train] Model saved -> {final_path}")

    return model, vocab, note_to_int, int_to_note


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--no_cache",   action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_cache=not args.no_cache,
    )
