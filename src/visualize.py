import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

RESULTS_DIR = 'results'


def plot_roc_curves(probas, y_known, y_unknown):
    model_names = ['Random Forest', 'MLP', 'LSTM', 'Autoencoder']
    colors = ['blue', 'green', 'orange', 'red']

    for scenario, y_true in [('known', y_known), ('unknown', y_unknown)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, color in zip(model_names, colors):
            proba = probas.get((name, scenario.capitalize()))
            if proba is None:
                continue
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {scenario.capitalize()} Attack')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'roc_curves_{scenario}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def plot_bar_comparison(df):
    pivot = df.pivot(index='Model', columns='Scenario', values='F1')
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind='bar', ax=ax, colormap='RdYlGn', edgecolor='black', linewidth=0.5)
    ax.set_title('F1-Score Comparison: Known vs Unknown Attack')
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Scenario')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'bar_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ae_error_distribution(ae_model,
                                X_train, y_train,
                                X_test_known, y_test_known,
                                X_test_unknown):
    benign_errors = ae_model._reconstruction_errors(X_train[y_train == 0][:5000])
    known_errors = ae_model._reconstruction_errors(X_test_known[y_test_known == 1][:5000])
    unknown_errors = ae_model._reconstruction_errors(X_test_unknown[:5000])

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 80
    ax.hist(benign_errors, bins=bins, alpha=0.6, label='BENIGN', color='green', density=True)
    ax.hist(known_errors, bins=bins, alpha=0.6, label='Known Attack', color='orange', density=True)
    ax.hist(unknown_errors, bins=bins, alpha=0.6, label='Unknown Attack', color='red', density=True)
    ax.axvline(ae_model.threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold ({ae_model.threshold:.4f})')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Autoencoder Reconstruction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'ae_error_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
