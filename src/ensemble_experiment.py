"""
RF + AE Ensemble Experiment

세 가지 앙상블 전략 비교:
  OR  : 둘 중 하나라도 공격 → 공격
  AND : 둘 다 공격 → 공격
  AVG : RF 확률 + AE 정규화 점수 평균, 0.5 이상이면 공격

평가 범위:
  1. Unknown 전체 (BENIGN 혼합)
  2. flow 이상형만
  3. flow 정상형만
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.special import expit  # sigmoid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from src.models.random_forest import RFModel
from src.models.mlp import MLPModel
from src.models.lstm import LSTMModel
from src.models.autoencoder import AutoencoderModel

RESULTS_DIR = 'results'
Z_THRESHOLD  = 3.0


def compute_metrics(y_true, y_pred, y_score):
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC':   roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,
    }


def print_table(rows, title):
    print(f"\n{'='*65}")
    print(f"[{title}]")
    print(f"{'Model/Strategy':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 63)
    for r in rows:
        print(f"{r['name']:<25} {r['Precision']:>10.4f} {r['Recall']:>8.4f} "
              f"{r['F1']:>8.4f} {r['ROC_AUC']:>9.4f}")


def evaluate_on(X_attack, y_attack, X_benign_test, y_benign_test,
                rf, ae, title):
    X = np.concatenate([X_benign_test, X_attack])
    y = np.concatenate([y_benign_test, y_attack])

    # ── 개별 모델 점수 ────────────────────────────────────────
    rf_proba  = rf.predict_proba(X)           # 0~1
    ae_errors = ae._reconstruction_errors(X)  # reconstruction error

    rf_pred = (rf_proba >= 0.5).astype(int)
    ae_pred = (ae_errors > ae.threshold).astype(int)

    # AE 점수 정규화: sigmoid((error - threshold) / threshold)
    # → error==threshold 일 때 0.5, error==0 일 때 ~0, error>>threshold 일 때 ~1
    ae_score = expit((ae_errors - ae.threshold) / ae.threshold)

    # ── 앙상블 전략 ────────────────────────────────────────────
    or_pred   = ((rf_pred == 1) | (ae_pred == 1)).astype(int)
    and_pred  = ((rf_pred == 1) & (ae_pred == 1)).astype(int)
    avg_score = (rf_proba + ae_score) / 2
    avg_pred  = (avg_score >= 0.5).astype(int)

    rows = []
    entries = [
        ('RF (단독)',          rf_pred,  rf_proba),
        ('AE (단독)',          ae_pred,  ae_score),
        ('앙상블 OR',          or_pred,  np.maximum(rf_proba, ae_score)),
        ('앙상블 AND',         and_pred, np.minimum(rf_proba, ae_score)),
        ('앙상블 AVG',         avg_pred, avg_score),
    ]
    for name, pred, score in entries:
        m = compute_metrics(y, pred, score)
        rows.append({'name': name, **m})

    print_table(rows, title)
    return rows


def main():
    print("Loading data...")
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]

    # ── 모델 로드 ─────────────────────────────────────────────
    rf = RFModel()
    rf.load(os.path.join(RESULTS_DIR, 'rf_model.pkl'))

    ae = AutoencoderModel(input_size)
    ae.load(os.path.join(RESULTS_DIR, 'ae_model.pt'))
    ae.threshold = float(np.load(os.path.join(RESULTS_DIR, 'ae_threshold.npy')))

    # ── flow 이상/정상 분류 (z-score) ─────────────────────────
    X_benign_train = X_train[y_train == 0]
    benign_mean = X_benign_train.mean(axis=0)
    benign_std  = X_benign_train.std(axis=0)
    benign_std[benign_std == 0] = 1.0

    max_z = np.abs((X_test_unknown - benign_mean) / benign_std).max(axis=1)
    flow_ab = max_z > Z_THRESHOLD
    flow_no = ~flow_ab

    X_benign_test = X_test_known[y_test_known == 0]
    y_benign_test = y_test_known[y_test_known == 0]

    all_rows = []

    # ── 평가 1: Unknown 전체 ──────────────────────────────────
    rows = evaluate_on(X_test_unknown, y_test_unknown,
                       X_benign_test, y_benign_test,
                       rf, ae, "Unknown 전체 (BENIGN 혼합)")
    for r in rows: r['subset'] = '전체'
    all_rows.extend(rows)

    # ── 평가 2: flow 이상형 ───────────────────────────────────
    if flow_ab.sum() > 0:
        rows = evaluate_on(X_test_unknown[flow_ab], y_test_unknown[flow_ab],
                           X_benign_test, y_benign_test,
                           rf, ae, f"flow 이상형 (z>{Z_THRESHOLD}, n={flow_ab.sum()})")
        for r in rows: r['subset'] = 'flow 이상형'
        all_rows.extend(rows)

    # ── 평가 3: flow 정상형 ───────────────────────────────────
    if flow_no.sum() > 0:
        rows = evaluate_on(X_test_unknown[flow_no], y_test_unknown[flow_no],
                           X_benign_test, y_benign_test,
                           rf, ae, f"flow 정상형 (z≤{Z_THRESHOLD}, n={flow_no.sum()})")
        for r in rows: r['subset'] = 'flow 정상형'
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_results.csv'), index=False)
    print(f"\nSaved: {RESULTS_DIR}/ensemble_results.csv")


if __name__ == '__main__':
    main()
