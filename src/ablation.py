"""
Ablation study and deep-dive analysis:
  1. AE threshold sensitivity (90/95/99th percentile)
  2. Reconstruction error distribution (detailed)
  3. RF feature importance
  4. t-SNE on AE latent space
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from src.models.autoencoder import AutoencoderModel
import torch

RESULTS_DIR = 'results'


# ── 1. Threshold Sensitivity ──────────────────────────────────────────────────

def threshold_sensitivity(ae, X_val, y_val, X_test_known, y_test_known,
                           X_test_unknown, y_test_unknown):
    print("\n[1] Threshold Sensitivity Analysis")
    print("-" * 60)

    X_val_benign = X_val[y_val == 0]
    val_errors = ae._reconstruction_errors(X_val_benign)

    percentiles = [80, 85, 90, 95, 99]
    thresholds = {p: float(np.percentile(val_errors, p)) for p in percentiles}

    rows = []
    for p, thr in thresholds.items():
        ae.threshold = thr
        for scenario, X, y in [('Known', X_test_known, y_test_known),
                                ('Unknown', X_test_unknown, y_test_unknown)]:
            metrics, _ = ae.evaluate(X, y)
            rows.append({
                'Percentile': f'{p}th',
                'Threshold': round(thr, 6),
                'Scenario': scenario,
                'Recall': round(metrics['Recall'], 4),
                'Precision': round(metrics['Precision'], 4),
                'F1': round(metrics['F1'], 4),
                'Accuracy': round(metrics['Accuracy'], 4),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scenario in zip(axes, ['Known', 'Unknown']):
        sub = df[df['Scenario'] == scenario]
        x = [r['Percentile'] for _, r in sub.iterrows()]
        ax.plot(x, sub['Recall'].values, marker='o', label='Recall', color='tomato')
        ax.plot(x, sub['Precision'].values, marker='s', label='Precision', color='steelblue')
        ax.plot(x, sub['F1'].values, marker='^', label='F1', color='green')
        ax.set_title(f'AE Threshold Sensitivity — {scenario} Attack')
        ax.set_xlabel('Threshold Percentile')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'ae_threshold_sensitivity.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Reset to 95th
    ae.threshold = thresholds[95]
    return df


# ── 2. Reconstruction Error Distribution (detailed) ───────────────────────────

def error_distribution_detail(ae, X_train, y_train,
                               X_test_known, y_test_known,
                               X_test_unknown, y_test_unknown):
    print("\n[2] Reconstruction Error Distribution")
    print("-" * 60)

    N = 3000
    benign_err  = ae._reconstruction_errors(X_train[y_train == 0][:N])
    known_err   = ae._reconstruction_errors(X_test_known[y_test_known == 1][:N])
    unknown_err = ae._reconstruction_errors(X_test_unknown[:N])

    stats = {
        'BENIGN':         benign_err,
        'Known Attack':   known_err,
        'Unknown Attack': unknown_err,
    }
    print(f"{'Group':<20} {'Mean':>10} {'Median':>10} {'95th%':>10} {'Max':>10}")
    print("-" * 55)
    for name, err in stats.items():
        print(f"{name:<20} {err.mean():>10.5f} {np.median(err):>10.5f} "
              f"{np.percentile(err,95):>10.5f} {err.max():>10.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log scale
    colors = {'BENIGN': 'green', 'Known Attack': 'orange', 'Unknown Attack': 'red'}
    for ax, use_log in zip(axes, [False, True]):
        for name, err in stats.items():
            ax.hist(err, bins=80, alpha=0.55, label=name,
                    color=colors[name], density=True)
        ax.axvline(ae.threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({ae.threshold:.4f})')
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Density' + (' (log)' if use_log else ''))
        ax.set_title('AE Error Distribution' + (' — Log Scale' if use_log else ''))
        if use_log:
            ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'ae_error_distribution_detail.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return stats


# ── 3. RF Feature Importance ──────────────────────────────────────────────────

def rf_feature_importance(rf_path, feature_names, top_n=20):
    print("\n[3] RF Feature Importance")
    print("-" * 60)

    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [(feature_names[i], importances[i]) for i in indices]

    print(f"Top {top_n} features:")
    for rank, (name, imp) in enumerate(top_features, 1):
        print(f"  {rank:2d}. {name:<40} {imp:.5f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f for f, _ in top_features]
    vals = [v for _, v in top_features]
    colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    bars = ax.barh(names[::-1], vals[::-1], color=colors_bar[::-1], edgecolor='gray', linewidth=0.4)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Random Forest — Top {top_n} Feature Importances')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'rf_feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return top_features


# ── 4. t-SNE on AE Latent Space ───────────────────────────────────────────────

def tsne_latent(ae, X_train, y_train, X_test_unknown, y_test_unknown, n_samples=1500):
    print("\n[4] t-SNE on AE Latent Space")
    print("-" * 60)
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not available for t-SNE. Skipping.")
        return

    # Sample: BENIGN, Known Attack, Unknown Attack
    idx_benign  = np.where(y_train == 0)[0]
    idx_known   = np.where(y_train == 1)[0]

    rng = np.random.default_rng(42)
    s_b = rng.choice(idx_benign, min(n_samples, len(idx_benign)), replace=False)
    s_k = rng.choice(idx_known,  min(n_samples, len(idx_known)),  replace=False)
    s_u = rng.choice(len(X_test_unknown), min(n_samples, len(X_test_unknown)), replace=False)

    X_tsne = np.concatenate([X_train[s_b], X_train[s_k], X_test_unknown[s_u]])
    labels_tsne = (
        ['BENIGN'] * len(s_b) +
        ['Known Attack'] * len(s_k) +
        ['Unknown Attack'] * len(s_u)
    )

    # Get latent representation
    ae.model.eval()
    device = ae.device
    with torch.no_grad():
        xb = torch.tensor(X_tsne, dtype=torch.float32).to(device)
        # Process in chunks to avoid OOM
        chunks = []
        for i in range(0, len(xb), 512):
            chunks.append(ae.model.encoder(xb[i:i+512]).cpu().numpy())
        Z = np.concatenate(chunks)

    print(f"Running t-SNE on {len(Z)} samples, latent dim={Z.shape[1]}...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)
    Z2d = tsne.fit_transform(Z)

    palette = {'BENIGN': 'green', 'Known Attack': 'orange', 'Unknown Attack': 'red'}
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, color in palette.items():
        mask = [l == label for l in labels_tsne]
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1], c=color, label=label,
                   alpha=0.4, s=8, linewidths=0)
    ax.set_title('t-SNE of AE Latent Space')
    ax.legend(markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'ae_tsne_latent.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    (X_train, y_train, X_val, y_val,
     X_test_known, y_test_known,
     X_test_unknown, y_test_unknown,
     scaler) = preprocess('archive')

    input_size = X_train.shape[1]

    # Load AE
    ae = AutoencoderModel(input_size)
    ae.load(os.path.join(RESULTS_DIR, 'ae_model.pt'))
    ae.threshold = float(np.load(os.path.join(RESULTS_DIR, 'ae_threshold.npy')))

    # Feature names (from a sample CSV)
    import glob
    sample_df = pd.read_csv(glob.glob('archive/*.csv')[0], low_memory=False, nrows=1)
    sample_df.columns = sample_df.columns.str.strip()
    feature_names = [c for c in sample_df.columns if c != 'Label']

    # Run analyses
    df_sensitivity = threshold_sensitivity(
        ae, X_val, y_val,
        X_test_known, y_test_known,
        X_test_unknown, y_test_unknown
    )

    error_distribution_detail(
        ae, X_train, y_train,
        X_test_known, y_test_known,
        X_test_unknown, y_test_unknown
    )

    rf_feature_importance(
        os.path.join(RESULTS_DIR, 'rf_model.pkl'),
        feature_names,
        top_n=20
    )

    tsne_latent(ae, X_train, y_train, X_test_unknown, y_test_unknown)

    print("\n" + "=" * 60)
    print("Ablation analysis complete. Results saved to results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
