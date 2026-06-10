"""
Z-score 기반 라우팅 실험

flow 이상형(z > 3.0) → AE로 판단
flow 정상형(z ≤ 3.0) → RF로 판단

비교 대상: RF 단독, AE 단독, OR 앙상블, 라우팅
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from src.models.random_forest import RFModel
from src.models.autoencoder import AutoencoderModel

RESULTS_DIR = 'results'
Z_THRESHOLD = 3.0


def compute_metrics(y_true, y_pred, y_score=None):
    m = {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        m['ROC_AUC'] = roc_auc_score(y_true, y_score)
    else:
        m['ROC_AUC'] = 0.0
    return m


def print_results(rows):
    print(f"\n{'Strategy':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print('-' * 63)
    for r in rows:
        print(f"{r['Strategy']:<25} {r['Precision']:>10.4f} {r['Recall']:>8.4f} "
              f"{r['F1']:>8.4f} {r['ROC_AUC']:>9.4f}")


def main():
    print("Loading data...")
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]

    # 모델 로드
    rf = RFModel()
    rf.load(os.path.join(RESULTS_DIR, 'rf_model.pkl'))

    ae = AutoencoderModel(input_size)
    ae.load(os.path.join(RESULTS_DIR, 'ae_model.pt'))
    ae.threshold = float(np.load(os.path.join(RESULTS_DIR, 'ae_threshold.npy')))

    # BENIGN 혼합 테스트셋 구성
    X_benign = X_test_known[y_test_known == 0]
    y_benign = y_test_known[y_test_known == 0]
    X_test = np.concatenate([X_benign, X_test_unknown])
    y_test = np.concatenate([y_benign, y_test_unknown])
    print(f"Test set: {(y_test==0).sum()} BENIGN + {(y_test==1).sum()} Unknown Attack")

    # z-score 계산
    X_benign_train = X_train[y_train == 0]
    benign_mean = X_benign_train.mean(axis=0)
    benign_std  = X_benign_train.std(axis=0)
    benign_std[benign_std == 0] = 1.0

    max_z = np.abs((X_test - benign_mean) / benign_std).max(axis=1)
    flow_ab = max_z > Z_THRESHOLD   # 이상형 mask
    flow_no = ~flow_ab              # 정상형 mask

    print(f"\nTest set z-score 분류:")
    print(f"  flow 이상형 (z>{Z_THRESHOLD}): {flow_ab.sum():,}개 "
          f"(attack={y_test[flow_ab].sum():,})")
    print(f"  flow 정상형 (z≤{Z_THRESHOLD}): {flow_no.sum():,}개 "
          f"(attack={y_test[flow_no].sum():,})")

    # 개별 모델 예측
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)

    ae_errors = ae._reconstruction_errors(X_test)
    ae_pred   = (ae_errors > ae.threshold).astype(int)

    # ── 라우팅 전략 ──────────────────────────────────────────────────
    # flow 이상형 → AE 판정, flow 정상형 → RF 판정
    routing_pred = np.where(flow_ab, ae_pred, rf_pred)

    # 점수: 이상형은 AE 정규화 점수, 정상형은 RF 확률
    ae_score_norm = np.clip(ae_errors / ae.threshold, 0, 5) / 5  # 0~1 정규화
    routing_score = np.where(flow_ab, ae_score_norm, rf_proba)

    # ── OR 앙상블 (비교용) ────────────────────────────────────────────
    or_pred = ((rf_pred == 1) | (ae_pred == 1)).astype(int)
    or_score = np.maximum(rf_proba, ae_score_norm)

    # ── 결과 집계 ─────────────────────────────────────────────────────
    strategies = [
        ('RF 단독',      rf_pred,      rf_proba),
        ('AE 단독',      ae_pred,      ae_score_norm),
        ('앙상블 OR',    or_pred,      or_score),
        ('Z-score 라우팅', routing_pred, routing_score),
    ]

    rows = []
    print("\n=== Unknown Attack 탐지 성능 비교 ===")
    for name, pred, score in strategies:
        m = compute_metrics(y_test, pred, score)
        rows.append({'Strategy': name, **m})

    print_results(rows)

    # FP 분석
    print("\n=== False Positive 분석 (BENIGN 오판 건수) ===")
    benign_mask = y_test == 0
    print(f"{'Strategy':<25} {'FP 건수':>10} {'FP율':>8}")
    print('-' * 45)
    for r in rows:
        name = r['Strategy']
        pred = next(p for n, p, _ in strategies if n == name)
        fp = int(((pred == 1) & benign_mask).sum())
        fp_rate = fp / benign_mask.sum()
        print(f"{name:<25} {fp:>10,} {fp_rate:>8.2%}")

    # CSV 저장
    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'routing_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return rows


if __name__ == '__main__':
    main()
