"""
visualization/heatmap_viz.py
Similarity heatmap visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SimilarityHeatmapVisualizer:

    def __init__(self, output_dir: str = "outputs/viz"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_heatmap(
        self,
        sim_matrix: np.ndarray,
        labels: list,
        title: str = "Similarity Matrix",
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "RdYlGn",
        save_path: str = None,
    ) -> str:
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
        sns.heatmap(
            sim_matrix,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Cosine Similarity"},
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_comparison_heatmaps(
        self,
        orig_matrix: np.ndarray,
        prot_matrix: np.ndarray,
        labels: list,
        model_name: str = "Model",
        save_path: str = None,
    ) -> str:
        """Side-by-side original vs protected heatmaps."""
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 2 * len(labels)), max(5, len(labels))))

        vmin = min(orig_matrix.min(), prot_matrix.min()) - 0.05
        vmax = 1.0

        for ax, mat, title in zip(
            axes,
            [orig_matrix, prot_matrix],
            [f"Original — {model_name}", f"Protected — {model_name}"],
        ):
            sns.heatmap(
                mat, ax=ax, vmin=vmin, vmax=vmax,
                cmap="RdYlGn", annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Cosine Sim"},
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle(
            f"Identity Similarity: Original vs Protected\n"
            f"Avg: {np.mean(orig_matrix[~np.eye(len(labels), dtype=bool)]):.3f} → "
            f"{np.mean(prot_matrix[~np.eye(len(labels), dtype=bool)]):.3f}",
            fontsize=13, y=1.02
        )
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, f"heatmap_comparison_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_all_models(
        self,
        original_matrices: dict,
        protected_matrices: dict,
        image_labels: list,
    ) -> dict:
        """Generate comparison heatmaps for all models."""
        paths = {}
        for name in original_matrices:
            path = self.plot_comparison_heatmaps(
                original_matrices[name],
                protected_matrices[name],
                labels=image_labels,
                model_name=name,
            )
            paths[name] = path
        return paths

    def plot_similarity_bar(
        self,
        summary_df,
        save_path: str = None,
    ) -> str:
        """Bar chart comparing per-model similarity reduction."""
        import pandas as pd
        fig, ax = plt.subplots(figsize=(max(8, len(summary_df) * 2), 5))

        x = np.arange(len(summary_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, summary_df["orig_mean_sim"],
                       width, label="Original", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width/2, summary_df["prot_mean_sim"],
                       width, label="Protected", color="#F44336", alpha=0.85)

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Mean Pairwise Cosine Similarity")
        ax.set_title("Identity Consistency Reduction by Model", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["model"], fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Threshold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "similarity_bar_chart.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
