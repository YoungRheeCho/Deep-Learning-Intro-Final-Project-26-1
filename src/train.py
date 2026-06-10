import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from src.models.random_forest import RFModel
from src.models.mlp import MLPModel
from src.models.lstm import LSTMModel
from src.models.autoencoder import AutoencoderModel

RESULTS_DIR = 'results'


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 60)
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]
    print(f"\nInput size: {input_size}")

    # ── Random Forest ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Random Forest")
    print("=" * 60)
    rf = RFModel()
    rf.fit(X_train, y_train)
    rf.save(os.path.join(RESULTS_DIR, 'rf_model.pkl'))
    val_metrics, _ = rf.evaluate(X_val, y_val, scenario='Val')
    print(f"RF Val Accuracy: {val_metrics['Accuracy']:.4f}")

    # ── MLP ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: MLP")
    print("=" * 60)
    mlp = MLPModel(input_size)
    mlp.fit(X_train, y_train, X_val, y_val)
    mlp.save(os.path.join(RESULTS_DIR, 'mlp_model.pt'))

    # ── LSTM ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: LSTM")
    print("=" * 60)
    lstm = LSTMModel(input_size)
    lstm.fit(X_train, y_train, X_val, y_val)
    lstm.save(os.path.join(RESULTS_DIR, 'lstm_model.pt'))

    # ── Autoencoder ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Autoencoder")
    print("=" * 60)
    ae = AutoencoderModel(input_size)
    ae.fit(X_train, y_train, X_val, y_val)
    ae.save(os.path.join(RESULTS_DIR, 'ae_model.pt'))
    np.save(os.path.join(RESULTS_DIR, 'ae_threshold.npy'), ae.threshold)

    print("\n" + "=" * 60)
    print("All models trained and saved.")
    print("=" * 60)

    return rf, mlp, lstm, ae, X_train, y_train, X_val, y_val, X_test_known, y_test_known, X_test_unknown, y_test_unknown, input_size


if __name__ == '__main__':
    main()
