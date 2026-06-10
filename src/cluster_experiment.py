"""
Experiment v2: Cluster-Aware Unknown Attack Detection

Pipeline:
  1. Load full data (BENIGN + Known + Unknown attacks)
  2. StandardScaler normalization
  3. MiniBatchKMeans clustering (K=8)
  4. Analyze cluster composition
  5. Per-cluster AE training on cluster-BENIGN only
  6. Evaluate Unknown attack detection per cluster
  7. Compare Cluster-AE vs Global-AE on same samples
  8. Visualize results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, KNOWN_ATTACKS, UNKNOWN_ATTACKS

RESULTS_DIR = os.path.join('results', 'cluster_experiment')


def safe(s):
    return s.encode('ascii', errors='replace').decode()
N_CLUSTERS = 8
RANDOM_STATE = 42
AE_EPOCHS = 30
AE_BATCH = 512
THRESHOLD_PERCENTILE = 95
MIN_BENIGN = 200


# ── Autoencoder ────────────────────────────────────────────────────────────────

class AENet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ClusterAE:
    def __init__(self, input_size, epochs=AE_EPOCHS, batch_size=AE_BATCH):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = AENet(input_size).to(self.device)
        self.threshold = None

    def fit(self, X_train_benign, X_val_benign):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train_benign, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=True
        )
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), xb)
                if torch.isnan(loss):
                    continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d} | loss: {total_loss/max(len(loader),1):.6f}")

        errors = self._errors(X_val_benign)
        self.threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))
        print(f"    Threshold ({THRESHOLD_PERCENTILE}th pct): {self.threshold:.6f}")

    def _errors(self, X):
        self.model.eval()
        out = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False
        )
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                err = ((xb - self.model(xb)) ** 2).mean(dim=1).cpu().numpy()
                out.append(err)
        return np.concatenate(out)

    def predict(self, X):
        return (self._errors(X) > self.threshold).astype(int)

    def evaluate(self, X, y):
        errors = self._errors(X)
        preds = (errors > self.threshold).astype(int)
        return {
            'Recall':    recall_score(y, preds, zero_division=0),
            'Precision': precision_score(y, preds, zero_division=0),
            'F1':        f1_score(y, preds, zero_division=0),
            'Accuracy':  accuracy_score(y, preds),
        }, errors


# ── Data loading ───────────────────────────────────────────────────────────────

def load_full_labeled(data_dir='archive'):
    print("Loading CSV files...")
    df = load_data(data_dir)
    df.columns = df.columns.str.strip()

    label_col = [c for c in df.columns if c.strip().lower() == 'label']
    if not label_col:
        raise ValueError("Label column not found.")
    df.rename(columns={label_col[0]: 'Label'}, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df)} rows with NaN. Remaining: {len(df)}")
    df['Label'] = df['Label'].str.strip()
    return df


# ── Cluster composition analysis ───────────────────────────────────────────────

def analyze_composition(labels, cluster_ids, n_clusters, all_attack_labels):
    rows = []
    for c in range(n_clusters):
        mask = cluster_ids == c
        label_series = pd.Series(labels[mask]).value_counts()
        total = int(mask.sum())
        row = {'cluster': c, 'total': total,
               'BENIGN': int(label_series.get('BENIGN', 0))}
        for atk in all_attack_labels:
            row[atk] = int(label_series.get(atk, 0))
        rows.append(row)
    return pd.DataFrame(rows)


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_composition_heatmap(comp_df, attack_cols, save_path):
    heat = comp_df.set_index('cluster')[attack_cols + ['BENIGN']].copy()
    # Normalize per cluster (percentage)
    heat_pct = heat.div(comp_df.set_index('cluster')['total'], axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(12, len(heat_pct.columns)), 5))
    sns.heatmap(heat_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': '% of cluster'})
    ax.set_title('Cluster Composition (% of each label per cluster)', fontsize=13)
    ax.set_xlabel('Label')
    ax.set_ylabel('Cluster ID')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_recall_comparison(cluster_results, global_results, save_path):
    if not cluster_results:
        return
    df = pd.DataFrame(cluster_results)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
    bars1 = ax.bar(x - width/2, df['cluster_recall'], width,
                   label='Cluster-AE', color='steelblue')
    bars2 = ax.bar(x + width/2, df['global_recall'], width,
                   label='Global-AE (baseline)', color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {int(r['cluster'])}\n(n={int(r['n_unknown'])})"
                        for _, r in df.iterrows()])
    ax.set_ylabel('Unknown Attack Recall')
    ax.set_title('Cluster-AE vs Global-AE: Unknown Attack Recall per Cluster')
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne(X_sample, labels_sample, cluster_sample, save_path):
    print("Running t-SNE on sample (n=3000)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=500)
    emb = tsne.fit_transform(X_sample)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: color by cluster
    scatter = axes[0].scatter(emb[:, 0], emb[:, 1], c=cluster_sample,
                               cmap='tab10', s=5, alpha=0.6)
    axes[0].set_title('t-SNE colored by Cluster')
    plt.colorbar(scatter, ax=axes[0])

    # Right: color by label type
    color_map = {'BENIGN': 0}
    for i, atk in enumerate(KNOWN_ATTACKS + UNKNOWN_ATTACKS, start=1):
        color_map[atk] = i
    label_colors = np.array([color_map.get(l, len(color_map)) for l in labels_sample])
    scatter2 = axes[1].scatter(emb[:, 0], emb[:, 1], c=label_colors,
                                cmap='tab20', s=5, alpha=0.6)
    axes[1].set_title('t-SNE colored by Label Type')

    plt.suptitle('t-SNE Visualization of Traffic Clusters', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Main experiment ────────────────────────────────────────────────────────────

def run(data_dir='archive', n_clusters=N_CLUSTERS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    df = load_full_labeled(data_dir)
    feature_cols = [c for c in df.columns if c != 'Label']
    X_raw = df[feature_cols].values.astype(np.float32)
    labels = df['Label'].values
    input_size = X_raw.shape[1]

    all_attack_labels = [a for a in (KNOWN_ATTACKS + UNKNOWN_ATTACKS)
                         if a in np.unique(labels)]
    unknown_present = [a for a in UNKNOWN_ATTACKS if a in np.unique(labels)]

    print(f"\nTotal samples: {len(X_raw)}")
    print(f"Unknown attack types present: {[safe(a) for a in unknown_present]}")

    # ── 2. Scale ───────────────────────────────────────────────────────────────
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    benign_idx = np.where(labels == 'BENIGN')[0]
    scaler.fit(X_raw[benign_idx])
    X = scaler.transform(X_raw).astype(np.float32)

    # ── 3. Cluster ─────────────────────────────────────────────────────────────
    print(f"\nRunning MiniBatchKMeans (K={n_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                             batch_size=10000, n_init=5)
    cluster_ids = kmeans.fit_predict(X)
    print("Clustering done.")

    # ── 4. Composition ─────────────────────────────────────────────────────────
    print("\n── Cluster Composition ──")
    comp_df = analyze_composition(labels, cluster_ids, n_clusters, all_attack_labels)
    comp_df.to_csv(os.path.join(RESULTS_DIR, 'cluster_composition.csv'), index=False)

    for _, row in comp_df.iterrows():
        unknown_counts = {safe(a): int(row[a]) for a in unknown_present if row[a] > 0}
        print(f"  Cluster {int(row['cluster'])} (n={int(row['total'])}) | "
              f"BENIGN={int(row['BENIGN'])} | Unknown={unknown_counts}")

    plot_composition_heatmap(
        comp_df, all_attack_labels,
        os.path.join(RESULTS_DIR, 'cluster_composition_heatmap.png')
    )

    # ── 5. Load Global AE for baseline ────────────────────────────────────────
    global_ae_path = os.path.join('results', 'ae_model.pt')
    global_threshold_path = os.path.join('results', 'ae_threshold.npy')
    global_ae = None
    if os.path.exists(global_ae_path) and os.path.exists(global_threshold_path):
        global_ae = ClusterAE(input_size)
        global_ae.model.load_state_dict(
            torch.load(global_ae_path, map_location=global_ae.device)
        )
        global_ae.threshold = float(np.load(global_threshold_path))
        print(f"\nLoaded Global AE (threshold={global_ae.threshold:.6f})")
    else:
        print("\n[WARN] Global AE not found — skipping baseline comparison.")

    # ── 6. Per-cluster AE ─────────────────────────────────────────────────────
    cluster_results = []
    all_cluster_metrics = []

    for c in range(n_clusters):
        c_mask = cluster_ids == c
        X_c = X[c_mask]
        labels_c = labels[c_mask]

        benign_mask = labels_c == 'BENIGN'
        unknown_mask = np.isin(labels_c, unknown_present)

        n_benign = int(benign_mask.sum())
        n_unknown = int(unknown_mask.sum())

        if n_benign < MIN_BENIGN or n_unknown == 0:
            print(f"\nCluster {c}: Skip (BENIGN={n_benign}, Unknown={n_unknown})")

            continue

        print(f"\n{'='*55}")
        print(f"Cluster {c} | BENIGN={n_benign} | Unknown={n_unknown}")

        X_benign = X_c[benign_mask]
        X_unknown = X_c[unknown_mask]
        labels_unknown = labels_c[unknown_mask]
        y_unknown = np.ones(n_unknown, dtype=int)

        # Train/val split on BENIGN
        n_val = max(100, int(0.15 * n_benign))
        perm = np.random.permutation(n_benign)
        X_b_train = X_benign[perm[n_val:]]
        X_b_val   = X_benign[perm[:n_val]]

        print(f"  Training Cluster-AE on {len(X_b_train)} BENIGN samples...")
        ae = ClusterAE(input_size)
        ae.fit(X_b_train, X_b_val)

        # Evaluate Cluster-AE
        metrics_cluster, errors_cluster = ae.evaluate(X_unknown, y_unknown)
        print(f"  [Cluster-AE] Recall={metrics_cluster['Recall']:.4f} "
              f"Precision={metrics_cluster['Precision']:.4f} "
              f"F1={metrics_cluster['F1']:.4f}")

        # Evaluate Global-AE on same samples
        metrics_global = {'Recall': None, 'Precision': None, 'F1': None}
        if global_ae is not None:
            metrics_global, _ = global_ae.evaluate(X_unknown, y_unknown)
            print(f"  [Global-AE]  Recall={metrics_global['Recall']:.4f} "
                  f"Precision={metrics_global['Precision']:.4f} "
                  f"F1={metrics_global['F1']:.4f}")

        # Per-attack-type breakdown
        per_attack = {}
        for atk in unknown_present:
            atk_mask = labels_unknown == atk
            if atk_mask.sum() > 0:
                m, _ = ae.evaluate(X_unknown[atk_mask],
                                   np.ones(atk_mask.sum(), dtype=int))
                per_attack[atk] = m['Recall']
                print(f"    {safe(atk)[:30]:<30} Recall={m['Recall']:.4f} "
                      f"(n={atk_mask.sum()})")

        row = {
            'cluster': c,
            'n_benign_train': len(X_b_train),
            'n_unknown': n_unknown,
            'cluster_recall':    metrics_cluster['Recall'],
            'cluster_precision': metrics_cluster['Precision'],
            'cluster_f1':        metrics_cluster['F1'],
            'global_recall':     metrics_global['Recall'],
            'global_precision':  metrics_global['Precision'],
            'global_f1':         metrics_global['F1'],
        }
        for atk, rec in per_attack.items():
            short = atk.replace(' ', '_')[:20]
            row[f'recall_{short}'] = rec
        cluster_results.append(row)
        all_cluster_metrics.append(row)

    # ── 7. Save results ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_cluster_metrics)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'cluster_ae_results.csv'), index=False)

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("=== Cluster-AE vs Global-AE Summary ===")
    if not results_df.empty:
        cols = ['cluster', 'n_benign_train', 'n_unknown',
                'cluster_recall', 'global_recall',
                'cluster_f1', 'global_f1']
        print(results_df[cols].to_string(index=False))

        avg_cluster = results_df['cluster_recall'].mean()
        avg_global  = results_df['global_recall'].mean() if results_df['global_recall'].notna().any() else None
        print(f"\nAvg Cluster-AE Recall across clusters: {avg_cluster:.4f}")
        if avg_global is not None:
            print(f"Avg Global-AE  Recall across clusters: {avg_global:.4f}")
            delta = avg_cluster - avg_global
            print(f"Delta (Cluster - Global): {delta:+.4f}")
    else:
        print("No valid clusters found for evaluation.")

    # ── 9. Visualizations ──────────────────────────────────────────────────────
    plot_recall_comparison(
        cluster_results, [],
        os.path.join(RESULTS_DIR, 'cluster_recall_comparison.png')
    )

    # t-SNE on a stratified sample
    sample_size = 3000
    indices = []
    unique_labels = np.unique(labels)
    per_label = max(1, sample_size // len(unique_labels))
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        chosen = np.random.choice(idx, min(per_label, len(idx)), replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    np.random.shuffle(indices)
    indices = indices[:sample_size]

    plot_tsne(X[indices], labels[indices], cluster_ids[indices],
              os.path.join(RESULTS_DIR, 'tsne_clusters.png'))

    print(f"\nAll results saved to: {RESULTS_DIR}/")
    return results_df, comp_df


if __name__ == '__main__':
    run(data_dir='archive', n_clusters=N_CLUSTERS)
