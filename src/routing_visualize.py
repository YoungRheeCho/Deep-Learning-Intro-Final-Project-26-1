"""라우팅 실험 결과 시각화"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = 'results'


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'routing_results.csv'))

    strategy_labels = ['RF Only', 'AE Only', 'Ensemble OR', 'Z-score Routing']
    strategies = strategy_labels
    precision  = df['Precision'].tolist()
    recall     = df['Recall'].tolist()
    f1         = df['F1'].tolist()

    fp_counts = [204, 17160, 17355, 16355]  # RF, AE, OR, Routing

    x = np.arange(len(strategies))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Z-score Routing vs Other Strategies', fontsize=13, fontweight='bold')

    # 왼쪽: Precision / Recall / F1
    bars_p = ax1.bar(x - width, precision, width, label='Precision', color='#4472C4', alpha=0.85)
    bars_r = ax1.bar(x,         recall,    width, label='Recall',    color='#ED7D31', alpha=0.85)
    bars_f = ax1.bar(x + width, f1,        width, label='F1',        color='#70AD47', alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                         f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Score')
    ax1.set_title('Precision / Recall / F1')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 오른쪽: FP 건수 (log scale)
    colors_fp = ['#4472C4', '#ED7D31', '#A9A9A9', '#FF0000']
    bars2 = ax2.bar(x, fp_counts, color=colors_fp, alpha=0.85)
    for bar, val in zip(bars2, fp_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                 f'{val:,}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=9)
    ax2.set_yscale('log')
    ax2.set_ylabel('False Positives (log scale)')
    ax2.set_title('False Positive Count (BENIGN 오판)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'routing_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
