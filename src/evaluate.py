import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from src.models.random_forest import RFModel
from src.models.mlp import MLPModel
from src.models.lstm import LSTMModel
from src.models.autoencoder import AutoencoderModel

RESULTS_DIR = 'results'


def evaluate_all(rf, mlp, lstm, ae,
                 X_test_known, y_test_known,
                 X_test_unknown, y_test_unknown):
    rows = []
    probas = {}

    for model_name, model in [('Random Forest', rf), ('MLP', mlp), ('LSTM', lstm), ('Autoencoder', ae)]:
        for scenario, X, y in [('Known', X_test_known, y_test_known),
                                ('Unknown', X_test_unknown, y_test_unknown)]:
            metrics, proba = model.evaluate(X, y, scenario=f'{scenario}')
            row = {'Model': model_name, 'Scenario': scenario}
            row.update(metrics)
            rows.append(row)
            probas[(model_name, scenario)] = proba

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'), index=False)
    print(f"\nSaved: {RESULTS_DIR}/summary_table.csv")
    return df, probas


def print_summary(df):
    print("\n=== Final Results Summary ===\n")
    for scenario in ['Known', 'Unknown']:
        print(f"[{scenario} Attack Evaluation]")
        sub = df[df['Scenario'] == scenario][['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']]
        header = f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
        print(header)
        print("-" * len(header))
        for _, row in sub.iterrows():
            print(f"{row['Model']:<20} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f} "
                  f"{row['Recall']:>8.4f} {row['F1']:>8.4f} {row['ROC_AUC']:>9.4f}")
        print()


if __name__ == '__main__':
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]

    rf = RFModel()
    rf.load(os.path.join(RESULTS_DIR, 'rf_model.pkl'))

    mlp = MLPModel(input_size)
    mlp.load(os.path.join(RESULTS_DIR, 'mlp_model.pt'), input_size)

    lstm = LSTMModel(input_size)
    lstm.load(os.path.join(RESULTS_DIR, 'lstm_model.pt'))

    ae = AutoencoderModel(input_size)
    ae.load(os.path.join(RESULTS_DIR, 'ae_model.pt'))
    ae.threshold = float(np.load(os.path.join(RESULTS_DIR, 'ae_threshold.npy')))

    X_benign_test = X_test_known[y_test_known == 0]
    y_benign_test = y_test_known[y_test_known == 0]
    X_test_unknown_mixed = np.concatenate([X_benign_test, X_test_unknown])
    y_test_unknown_mixed = np.concatenate([y_benign_test, y_test_unknown])
    print(f"Unknown mixed test: {(y_test_unknown_mixed==0).sum()} BENIGN + {(y_test_unknown_mixed==1).sum()} Unknown Attack")

    df, probas = evaluate_all(rf, mlp, lstm, ae,
                               X_test_known, y_test_known,
                               X_test_unknown_mixed, y_test_unknown_mixed)
    print_summary(df)
