"""
XAI Analysis: SHAP-based feature importance for RF model.

Generates:
  - results/xai/shap_summary_all.png      : 전체 공격 유형 SHAP summary
  - results/xai/shap_per_attack.png       : 공격 유형별 top-10 feature 비교
  - results/xai/shap_known_vs_unknown.png : Known vs Unknown SHAP mean 비교
  - results/xai/shap_values.csv           : 공격 유형별 mean |SHAP| 테이블
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, KNOWN_ATTACKS, UNKNOWN_ATTACKS

RESULTS_DIR = os.path.join('results', 'xai')
RF_PATH     = os.path.join('results', 'rf_model.pkl')
N_BACKGROUND = 500   # SHAP background sample size
N_EXPLAIN    = 200   # samples per attack type to explain
RANDOM_STATE = 42


def safe(s):
    return s.encode('ascii', errors='replace').decode()


def _extract_class1(sv):
    """SHAP 0.46 returns list[(n,f),(n,f)] or ndarray (n,f,2) or (n,f)."""
    if isinstance(sv, list):
        sv = np.array(sv)[1]       # list → (n, f)
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]           # (n, f, 2) → (n, f)
    return sv.astype(np.float32)


def load_full_labeled(data_dir='archive'):
    df = load_data(data_dir)
    df.columns = df.columns.str.strip()
    label_col = [c for c in df.columns if c.strip().lower() == 'label']
    df.rename(columns={label_col[0]: 'Label'}, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['Label'] = df['Label'].str.strip()
    return df


def sample_by_label(X, labels, label, n, rng):
    idx = np.where(labels == label)[0]
    chosen = rng.choice(idx, min(n, len(idx)), replace=False)
    return X[chosen], chosen


def run(data_dir='archive'):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_full_labeled(data_dir)
    feature_cols = [c for c in df.columns if c != 'Label']
    X_raw  = df[feature_cols].values.astype(np.float32)
    labels = df['Label'].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    benign_idx = np.where(labels == 'BENIGN')[0]
    scaler.fit(X_raw[benign_idx])
    X = scaler.transform(X_raw).astype(np.float32)

    # ── 2. Load RF ─────────────────────────────────────────────────────────────
    print("Loading RF model...")
    with open(RF_PATH, 'rb') as f:
        rf_model = pickle.load(f)

    # ── 3. Background (BENIGN sample) ─────────────────────────────────────────
    print(f"Sampling {N_BACKGROUND} BENIGN for SHAP background...")
    X_bg, _ = sample_by_label(X, labels, 'BENIGN', N_BACKGROUND, rng)

    explainer = shap.TreeExplainer(rf_model, X_bg, feature_perturbation='interventional')
    print("SHAP explainer ready.")

    # ── 4. Collect SHAP per attack type ───────────────────────────────────────
    all_labels   = KNOWN_ATTACKS + [a for a in UNKNOWN_ATTACKS if a in np.unique(labels)]
    label_shap   = {}   # label -> mean |SHAP| per feature
    label_raw    = {}   # label -> SHAP matrix (n_samples x n_features)

    for atk in all_labels:
        if atk not in np.unique(labels):
            continue
        X_atk, _ = sample_by_label(X, labels, atk, N_EXPLAIN, rng)
        print(f"  Computing SHAP for {safe(atk)} (n={len(X_atk)})...")
        sv = explainer.shap_values(X_atk)
        sv = _extract_class1(sv)
        label_shap[atk] = np.abs(sv).mean(axis=0)
        label_raw[atk]  = sv

    # Also compute for BENIGN
    X_ben, _ = sample_by_label(X, labels, 'BENIGN', N_EXPLAIN, rng)
    sv_ben = _extract_class1(explainer.shap_values(X_ben))
    label_shap['BENIGN'] = np.abs(sv_ben).mean(axis=0)
    label_raw['BENIGN']  = sv_ben

    feature_names = feature_cols

    # ── 5. Save SHAP table ────────────────────────────────────────────────────
    shap_df = pd.DataFrame(
        {safe(k): v for k, v in label_shap.items()},
        index=feature_names
    )
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_values.csv'))
    print(f"Saved: {RESULTS_DIR}/shap_values.csv")

    # ── 6. Plot 1: Per-attack top-10 feature comparison ───────────────────────
    top_n = 10
    attack_groups = {
        'Known Attacks':   KNOWN_ATTACKS,
        'Unknown Attacks': [a for a in UNKNOWN_ATTACKS if a in label_shap],
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, (group_name, group_labels) in zip(axes, attack_groups.items()):
        group_labels = [a for a in group_labels if a in label_shap]
        if not group_labels:
            ax.set_visible(False)
            continue

        # Average SHAP across attack types in this group
        avg_shap = np.mean([label_shap[a] for a in group_labels], axis=0)
        top_idx  = np.argsort(avg_shap)[::-1][:top_n]

        data = {safe(a): label_shap[a][top_idx] for a in group_labels}
        plot_df = pd.DataFrame(data, index=[feature_names[i] for i in top_idx])

        plot_df.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title(f'{group_name}: Top-{top_n} Important Features (mean |SHAP|)',
                     fontsize=11)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean |SHAP value|')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('Feature Importance by Attack Group (SHAP)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'shap_per_attack.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # ── 7. Plot 2: Known vs Unknown mean |SHAP| comparison ────────────────────
    known_labels   = [a for a in KNOWN_ATTACKS if a in label_shap]
    unknown_labels = [a for a in UNKNOWN_ATTACKS if a in label_shap]

    known_avg   = np.mean([label_shap[a] for a in known_labels],   axis=0)
    unknown_avg = np.mean([label_shap[a] for a in unknown_labels], axis=0)

    # Top features by known avg
    top_idx = np.argsort(known_avg)[::-1][:15]
    feat_names_top = [feature_names[i] for i in top_idx]

    x = np.arange(len(top_idx))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, known_avg[top_idx],   width, label='Known Attacks',   color='steelblue')
    ax.bar(x + width/2, unknown_avg[top_idx], width, label='Unknown Attacks', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names_top, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title('Known vs Unknown Attacks: Feature Importance Comparison (Top-15 by Known)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'shap_known_vs_unknown.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # ── 8. Plot 3: SHAP summary (all attacks combined) ────────────────────────
    all_attack_labels = [a for a in all_labels if a in label_raw]
    X_all_shap   = np.vstack([label_raw[a]  for a in all_attack_labels])
    X_all_data   = np.vstack([
        X[rng.choice(np.where(labels == a)[0],
                     min(N_EXPLAIN, (labels == a).sum()), replace=False)]
        for a in all_attack_labels
    ])

    # Top 15 features by global mean |SHAP|
    global_importance = np.abs(X_all_shap).mean(axis=0)
    top15 = np.argsort(global_importance)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp_df = pd.DataFrame({
        'Feature':    [feature_names[i] for i in top15],
        'Mean |SHAP|': global_importance[top15]
    })
    sns.barplot(data=feat_imp_df, x='Mean |SHAP|', y='Feature',
                palette='viridis', ax=ax)
    ax.set_title('Global SHAP Feature Importance (All Attack Types, Top-15)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'shap_summary_all.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # ── 9. Print top-5 per attack ─────────────────────────────────────────────
    print("\n=== Top-5 Features per Attack Type ===")
    for atk in all_attack_labels:
        top5_idx = np.argsort(label_shap[atk])[::-1][:5]
        top5 = [(feature_names[i], label_shap[atk][i]) for i in top5_idx]
        print(f"\n{safe(atk)}:")
        for fname, val in top5:
            print(f"  {fname:<40} {val:.5f}")

    print(f"\nAll XAI results saved to: {RESULTS_DIR}/")
    return shap_df


if __name__ == '__main__':
    run(data_dir='archive')
