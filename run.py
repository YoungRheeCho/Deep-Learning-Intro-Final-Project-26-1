"""
Main entry point: trains all models, evaluates, and generates visualizations.
Run from project root: python run.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess
from src.models.random_forest import RFModel
from src.models.mlp import MLPModel
from src.models.lstm import LSTMModel
from src.models.autoencoder import AutoencoderModel
from src.evaluate import evaluate_all, print_summary
from src.visualize import plot_roc_curves, plot_bar_comparison, plot_ae_error_distribution

RESULTS_DIR = 'results'


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Preprocessing ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]

    # ── Random Forest ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Random Forest")
    print("=" * 60)
    rf = RFModel()
    rf.fit(X_train, y_train)
    rf.save(os.path.join(RESULTS_DIR, 'rf_model.pkl'))

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

    # ── Evaluation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation")
    print("=" * 60)

    # Mix BENIGN test samples into Unknown evaluation for realistic assessment
    X_benign_test = X_test_known[y_test_known == 0]
    y_benign_test = y_test_known[y_test_known == 0]
    X_test_unknown_mixed = np.concatenate([X_benign_test, X_test_unknown])
    y_test_unknown_mixed = np.concatenate([y_benign_test, y_test_unknown])
    print(f"Unknown mixed test: {(y_test_unknown_mixed==0).sum()} BENIGN + {(y_test_unknown_mixed==1).sum()} Unknown Attack")

    df, probas = evaluate_all(rf, mlp, lstm, ae,
                               X_test_known, y_test_known,
                               X_test_unknown_mixed, y_test_unknown_mixed)
    print_summary(df)

    # ── Visualization ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Visualization")
    print("=" * 60)
    plot_roc_curves(probas, y_test_known, y_test_unknown_mixed)
    plot_bar_comparison(df)
    plot_ae_error_distribution(ae, X_train, y_train, X_test_known, y_test_known, X_test_unknown)

    print("\n" + "=" * 60)
    print("All done. Results saved to results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
