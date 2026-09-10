# =====================================================
#  MusicAI — LSTM Model Definition
#  CodeAlpha Task 3 | model.py
#  Deep Learning: Stacked LSTM + Dropout + Dense
# =====================================================
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam


def build_lstm_model(seq_len: int, n_vocab: int, learning_rate: float = 0.001):
    """
    Stacked LSTM model for music generation.

    Architecture:
      Input  : (batch, seq_len, 1)
      LSTM   : 512 units, return_sequences=True
      Dropout: 0.3
      BN     :
      LSTM   : 512 units, return_sequences=True
      Dropout: 0.3
      BN     :
      LSTM   : 256 units
      Dropout: 0.3
      Dense  : 256  -> ReLU
      Dense  : n_vocab -> Softmax
    """
    model = Sequential([
        # Layer 1
        LSTM(512, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),

        # Layer 2
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),

        # Layer 3
        LSTM(256),
        Dropout(0.3),

        # Dense layers
        Dense(256),
        Activation("relu"),
        Dropout(0.3),

        # Output
        Dense(n_vocab),
        Activation("softmax"),
    ], name="MusicAI_LSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_summary_str(model) -> str:
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


if __name__ == "__main__":
    m = build_lstm_model(seq_len=50, n_vocab=100)
    m.summary()
