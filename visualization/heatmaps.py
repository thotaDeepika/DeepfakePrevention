"""
visualization/heatmaps.py
Similarity heatmap visualization for SafeUpload.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict


def plot_similarity_heatmaps(
    results: Dict,
    output_dir: str = "outputs",
    figsize_per_model: tuple = (10, 4),
):
    """
    Plot pairwise similarity heatmaps for original vs protected images.

    Args:
        results:    output from IdentityEvaluator.evaluate()
        output_dir: directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(results)

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(figsize_per_model[0], figsize_per_model[1] * n_models)
    )
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (model_name, res) in enumerate(results.items()):
        orig_mat = res["original_sim_matrix"]
        prot_mat = res["protected_sim_matrix"]
        orig_mean = res["original_mean_similarity"]
        prot_mean = res["protected_mean_similarity"]
        reduction = orig_mean - prot_mean
        reduction_pct = (reduction / orig_mean * 100) if orig_mean > 0 else 0

        # Clamp for visualization
        vmin, vmax = 0.0, 1.0

        # Original heatmap
        ax_orig = axes[row_idx, 0]
        sns.heatmap(
            orig_mat, ax=ax_orig,
            vmin=vmin, vmax=vmax,
            cmap="RdYlGn",
            annot=True if orig_mat.shape[0] <= 8 else False,
            fmt=".2f",
            square=True,
            cbar=True,
        )
        ax_orig.set_title(
            f"{model_name} — Original\nMean Sim = {orig_mean:.4f}",
            fontweight="bold", fontsize=11
        )
        ax_orig.set_xlabel("Image Index")
        ax_orig.set_ylabel("Image Index")

        # Protected heatmap
        ax_prot = axes[row_idx, 1]
        sns.heatmap(
            prot_mat, ax=ax_prot,
            vmin=vmin, vmax=vmax,
            cmap="RdYlGn",
            annot=True if prot_mat.shape[0] <= 8 else False,
            fmt=".2f",
            square=True,
            cbar=True,
        )
        ax_prot.set_title(
            f"{model_name} — Protected\nMean Sim = {prot_mean:.4f}  (↓{reduction_pct:.1f}%)",
            fontweight="bold", fontsize=11,
            color="darkred" if reduction_pct > 10 else "black",
        )
        ax_prot.set_xlabel("Image Index")
        ax_prot.set_ylabel("Image Index")

    plt.suptitle(
        "Identity Consistency Heatmaps: Original vs Protected",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "similarity_heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Heatmaps saved: {save_path}")


def plot_similarity_bar_chart(results: Dict, output_dir: str = "outputs"):
    """
    Bar chart comparing original vs protected mean similarities per model.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    orig_sims = [results[m]["original_mean_similarity"] for m in model_names]
    prot_sims = [results[m]["protected_mean_similarity"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), 5))
    bars1 = ax.bar(x - width/2, orig_sims, width, label="Original", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, prot_sims, width, label="Protected", color="#F44336", alpha=0.85)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Embedding Model")
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title("Identity Consistency: Original vs Protected Images", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")

    # Add annotation box
    reductions = [(orig_sims[i] - prot_sims[i]) / orig_sims[i] * 100
                  for i in range(len(model_names)) if orig_sims[i] > 0]
    if reductions:
        avg_reduction = np.mean(reductions)
        ax.text(0.02, 0.97,
                f"Avg reduction: {avg_reduction:.1f}%",
                transform=ax.transAxes,
                fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "similarity_bar_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Bar chart saved: {save_path}")
