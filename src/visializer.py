import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scorer import BEST_THRESHOLD as threshold

logger = logging.getLogger(__name__)

def setup_style():
    japan_pastel = [
        "#FAD6A5", "#F5B7B1", "#D7BDE2", "#A9CCE3", "#A3E4D7",
        "#F9E79F", "#F5CBA7", "#D2B4DE", "#AED6F1", "#A2D9CE"
    ]
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette=japan_pastel,
        rc={
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
        }
    )

    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["legend.frameon"] = False


def plot_predicted_distribution(predictions, output_path):
    logger.info('Plotting predicted distribution...')

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(predictions, label='Predicted Probabilities', color="#F9A69E", fill=True, ax=ax)
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')

    y_min, y_max = ax.get_ylim()
    ax.fill_betweenx([y_min, y_max], threshold, 1, color="#7AF983", alpha=0.2, label='Positive Class Region')
    ax.fill_betweenx([y_min, y_max], 0, threshold, color="#6FBAED", alpha=0.2, label='Negative Class Region')

    plt.suptitle('Distribution of Predicted Values', fontsize=16, fontweight='bold', y=1.02)
    ax.set_xlabel('Predicted Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title='Legend', title_fontsize='12', fontsize='11', loc='upper right')
    plt.savefig(output_path)
    logger.info('Predicted distribution plot saved to: %s', output_path)


def plot_graphs(predictions, output_path):
    logger.info('Setting up style')
    setup_style()

    logger.info('Plotting predicted distribution')
    plot_predicted_distribution(predictions, output_path)