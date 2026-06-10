"""
Flow Abnormality Experiment (Option B: z-score based)

Unknown Attack 샘플을 BENIGN 학습 분포와의 통계적 거리(z-score)로
flow 이상형 / flow 정상형으로 분류 후 각 모델 성능 비교.

기준: max |z-score| > 3 이면 flow 이상형
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess, load_data, KNOWN_ATTACKS, UNKNOWN_ATTACKS
from src.models.random_forest import RFModel
from src.models.mlp import MLPModel
from src.models.lstm import LSTMModel
from src.models.autoencoder import AutoencoderModel

RESULTS_DIR = 'results'
Z_THRESHOLD = 3.0


def get_unknown_labels(data_dir='archive'):
    df = load_data(data_dir)
    df.columns = df.columns.str.strip()
    label_col = [c for c in df.columns if c.strip().lower() == 'label'][0]
    df.rename(columns={label_col: 'Label'}, inplace=True)
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)
    df['Label'] = df['Label'].str.strip()
    unknown_present = [a for a in UNKNOWN_ATTACKS if a in df['Label'].values]
    return df[df['Label'].isin(unknown_present)]['Label'].values


def safe(s):
    return s.encode('ascii', errors='replace').decode()


def evaluate_subset(models, X_sub, y_sub, X_benign_test, y_benign_test, label):
    X_mixed = np.concatenate([X_benign_test, X_sub])
    y_mixed = np.concatenate([y_benign_test, y_sub])

    print(f"\n{'='*62}")
    print(f"[{label}]  attack={len(X_sub)}개  BENIGN={len(X_benign_test)}개")
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 58)

    rows = []
    for name, model in models:
        m, _ = model.evaluate(X_mixed, y_mixed)
        print(f"{name:<20} {m['Precision']:>10.4f} {m['Recall']:>8.4f} "
              f"{m['F1']:>8.4f} {m['ROC_AUC']:>9.4f}")
        rows.append({'Model': name, 'Subset': label, **m})
    return rows


def main():
    print("Loading data...")
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]

    # BENIGN 학습 데이터 기준 통계
    X_benign_train = X_train[y_train == 0]
    benign_mean = X_benign_train.mean(axis=0)
    benign_std  = X_benign_train.std(axis=0)
    benign_std[benign_std == 0] = 1.0  # 0으로 나누기 방지

    # Unknown Attack 샘플별 max z-score 계산
    z = np.abs((X_test_unknown - benign_mean) / benign_std)
    max_z = z.max(axis=1)

    flow_ab_mask = max_z > Z_THRESHOLD
    flow_no_mask = ~flow_ab_mask

    n_total = len(X_test_unknown)
    n_ab = flow_ab_mask.sum()
    n_no = flow_no_mask.sum()

    print(f"\nUnknown Attack 총 {n_total}개")
    print(f"  flow 이상형 (max z > {Z_THRESHOLD}): {n_ab}개 ({n_ab/n_total*100:.1f}%)")
    print(f"  flow 정상형 (max z ≤ {Z_THRESHOLD}): {n_no}개 ({n_no/n_total*100:.1f}%)")

    # 공격 유형별 분포
    print("\n공격 유형별 분포:")
    unknown_labels = get_unknown_labels('archive')
    label_df = pd.DataFrame({'label': unknown_labels, 'flow_ab': flow_ab_mask})
    for atk, grp in label_df.groupby('label'):
        ab = grp['flow_ab'].sum()
        tot = len(grp)
        print(f"  {safe(atk):<40} 이상형: {ab:>4}개 / 전체: {tot}개 ({ab/tot*100:.1f}%)")

    # 모델 로드
    rf = RFModel()
    rf.load(os.path.join(RESULTS_DIR, 'rf_model.pkl'))

    mlp = MLPModel(input_size)
    mlp.load(os.path.join(RESULTS_DIR, 'mlp_model.pt'), input_size)

    lstm = LSTMModel(input_size)
    lstm.load(os.path.join(RESULTS_DIR, 'lstm_model.pt'))

    ae = AutoencoderModel(input_size)
    ae.load(os.path.join(RESULTS_DIR, 'ae_model.pt'))
    ae.threshold = float(np.load(os.path.join(RESULTS_DIR, 'ae_threshold.npy')))

    models = [('Random Forest', rf), ('MLP', mlp), ('LSTM', lstm), ('Autoencoder', ae)]

    X_benign_test = X_test_known[y_test_known == 0]
    y_benign_test = y_test_known[y_test_known == 0]

    all_rows = []

    if n_ab > 0:
        rows = evaluate_subset(models,
                               X_test_unknown[flow_ab_mask], y_test_unknown[flow_ab_mask],
                               X_benign_test, y_benign_test,
                               f"flow 이상형 (z>{Z_THRESHOLD})")
        all_rows.extend(rows)

    if n_no > 0:
        rows = evaluate_subset(models,
                               X_test_unknown[flow_no_mask], y_test_unknown[flow_no_mask],
                               X_benign_test, y_benign_test,
                               f"flow 정상형 (z≤{Z_THRESHOLD})")
        all_rows.extend(rows)

    # 이전 Unknown 전체 결과와 비교
    print(f"\n{'='*62}")
    print("[참고: 기존 Unknown 전체 결과]")
    prev = pd.read_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'))
    sub = prev[prev['Scenario'] == 'Unknown'][['Model', 'Precision', 'Recall', 'F1', 'ROC_AUC']]
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 58)
    for _, row in sub.iterrows():
        print(f"{row['Model']:<20} {row['Precision']:>10.4f} {row['Recall']:>8.4f} "
              f"{row['F1']:>8.4f} {row['ROC_AUC']:>9.4f}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(os.path.join(RESULTS_DIR, 'flow_zscore_results.csv'), index=False)
    print(f"\nSaved: {RESULTS_DIR}/flow_zscore_results.csv")


if __name__ == '__main__':
    main()
