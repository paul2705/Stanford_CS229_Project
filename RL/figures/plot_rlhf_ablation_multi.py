import os
import matplotlib.pyplot as plt
import numpy as np


MODELS = ["blenderbot", "mvp", "bart-base", "bart-paraphrase", "bart-large"]
LABELS = [m.replace('-', '\n') for m in MODELS]

BASELINE = {
    "blenderbot": {"J": 0.4333, "STA": 0.9488, "FL": 0.6361},
    "mvp": {"J": 0.5995, "STA": 0.9250, "FL": 0.7451},
    "bart-base": {"J": 0.5877, "STA": 0.9088, "FL": 0.7329},
    "bart-paraphrase": {"J": 0.6072, "STA": 0.9209, "FL": 0.7552},
    "bart-large": {"J": 0.6126, "STA": 0.9260, "FL": 0.7628},
}

J_BLEU = {
    "blenderbot": {"J": 0.4907, "STA": 0.9554, "FL": 0.7238},
    "mvp": {"J": 0.6262, "STA": 0.9463, "FL": 0.7694},
    "bart-base": {"J": 0.5979, "STA": 0.9270, "FL": 0.7324},
    "bart-paraphrase": {"J": 0.6155, "STA": 0.9351, "FL": 0.7552},
    "bart-large": {"J": 0.6511, "STA": 0.9539, "FL": 0.8089},
}

J_ONLY = {
    "blenderbot": {"J": 0.4859, "STA": 0.9589, "FL": 0.7131},
    "mvp": {"J": 0.6235, "STA": 0.9483, "FL": 0.7699},
    "bart-base": {"J": 0.6109, "STA": 0.9270, "FL": 0.7471},
    "bart-paraphrase": {"J": 0.6180, "STA": 0.9442, "FL": 0.7527},
    "bart-large": {"J": 0.6332, "STA": 0.9503, "FL": 0.8044},
}

METRICS = ["J", "STA", "FL"]
Y_RANGES = {
    "J": (0.42, 0.70),
    "STA": (0.90, 0.97),
    "FL": (0.63, 0.82),
}

COLORS = ["#5FB3FF", "#34A0A4", "#E76F51"]

# Adjust these to change the figure footprint and typography.
FIGSIZE = (17, 5)
TITLE_SIZE = 14
AX_TITLE_SIZE = 11.5
TICK_FONT_SIZE = 12.5
LABEL_FONT_SIZE = 14.5
LEGEND_FONT_SIZE = 12.5
ANNOTATION_FONT_SIZE = 7.5
BAR_WIDTH = 0.29
ANNOTATION_OFFSET = 1.5


def plot_multi_metric_figure(output_path: str) -> None:
    """Render the ParaDetox multi-metric comparison chart and save it."""

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True)

    x = np.arange(len(MODELS))

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        baseline_vals = [BASELINE[m][metric] for m in MODELS]
        j_only_vals = [J_ONLY[m][metric] for m in MODELS]
        j_bleu_vals = [J_BLEU[m][metric] for m in MODELS]

        ax.bar(
            x - BAR_WIDTH,
            baseline_vals,
            BAR_WIDTH,
            label='Baseline (SFT)' if idx == 0 else '',
            color=COLORS[0],
            edgecolor='white',
        )
        ax.bar(
            x,
            j_only_vals,
            BAR_WIDTH,
            label='RLHF (J-only)' if idx == 0 else '',
            color=COLORS[1],
            edgecolor='white',
        )
        ax.bar(
            x + BAR_WIDTH,
            j_bleu_vals,
            BAR_WIDTH,
            label='RLHF (J+BLEU)' if idx == 0 else '',
            color=COLORS[2],
            edgecolor='white',
        )

        ax.set_title(metric, fontsize=AX_TITLE_SIZE, weight='bold')
        ax.set_ylim(*Y_RANGES[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=TICK_FONT_SIZE)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bars in ax.containers:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, ANNOTATION_OFFSET),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE,
                    color='#1C1C1C',
                )

    axes[0].set_ylabel('Score', fontsize=LABEL_FONT_SIZE)
    axes[0].legend(loc='upper left', frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=320)
    print(f"Saved figure to {output_path}")


if __name__ == '__main__':
    OUTPUT_DIR = os.path.join('RL', 'figures')
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'rlhf_ablation_multi.png')
    plot_multi_metric_figure(OUTPUT_PATH)
