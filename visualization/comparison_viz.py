"""
visualization/comparison_viz.py
Before/After comparison visualizations and perturbation maps.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os


class ComparisonVisualizer:

    def __init__(self, output_dir: str = "outputs/viz"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_before_after(
        self,
        original_images: list,
        protected_images: list,
        metrics_list: list = None,
        save_path: str = None,
        max_cols: int = 3,
    ) -> str:
        """
        Side-by-side before/after gallery with perturbation map.
        3 rows per image: original | protected | perturbation×10
        """
        n = len(original_images)
        cols = min(n, max_cols)
        rows = 3  # original / protected / amplified perturbation

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        if cols == 1:
            axes = axes[:, np.newaxis]

        row_labels = ["Original", "Protected", "Perturbation ×10"]
        colors = ["#2196F3", "#F44336", "#9C27B0"]

        for col_idx in range(cols):
            orig = original_images[col_idx]
            prot = protected_images[col_idx]

            orig_np = np.array(orig.resize((224, 224))) / 255.0
            prot_np = np.array(prot.resize((224, 224))) / 255.0
            delta_np = np.clip((prot_np - orig_np) * 10 + 0.5, 0, 1)

            for row_idx, (img_np, label, color) in enumerate(
                zip([orig_np, prot_np, delta_np], row_labels, colors)
            ):
                ax = axes[row_idx][col_idx]
                ax.imshow(img_np)
                ax.axis("off")

                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=11, color=color, fontweight="bold",
                                  rotation=90, labelpad=4)

                if row_idx == 0:
                    title = f"Image {col_idx+1}"
                    if metrics_list and col_idx < len(metrics_list):
                        m = metrics_list[col_idx]
                        title += f"\nSSIM={m.get('ssim','?')} PSNR={m.get('psnr','?')}"
                    ax.set_title(title, fontsize=9)

        plt.suptitle("SafeUpload: Original vs Protected Images",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "before_after_gallery.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_perturbation_stats(
        self,
        metrics_list: list,
        save_path: str = None,
    ) -> str:
        """Bar chart of SSIM, PSNR per image."""
        n = len(metrics_list)
        labels = [f"Img {i+1}" for i in range(n)]
        ssims = [m.get("ssim", 0) for m in metrics_list]
        psnrs = [m.get("psnr", 0) for m in metrics_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.bar(labels, ssims, color="#2196F3", alpha=0.85)
        ax1.axhline(y=0.85, color="green", linestyle="--", label="Target (0.85)")
        ax1.set_title("SSIM (higher = better quality)", fontweight="bold")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(labels, psnrs, color="#4CAF50", alpha=0.85)
        ax2.axhline(y=30, color="orange", linestyle="--", label="Target (30 dB)")
        ax2.set_title("PSNR (higher = better quality)", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Visual Quality Preservation Metrics", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "quality_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_summary_dashboard(
        self,
        overall_metrics: dict,
        summary_df,
        avg_ssim: float,
        avg_psnr: float,
        save_path: str = None,
    ) -> str:
        """Single-page summary dashboard."""
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor("#1a1a2e")

        # Title
        fig.text(0.5, 0.96, "SafeUpload — Identity Cloaking Summary",
                 ha="center", va="top", fontsize=18, fontweight="bold",
                 color="white")
        fig.text(0.5, 0.91,
                 "\"Transferable adversarial identity cloaking reduces AI face representation consistency\"",
                 ha="center", va="top", fontsize=11, color="#aaaaaa", style="italic")

        # Metric boxes
        metrics = [
            ("Original\nIdentity Sim", f"{overall_metrics.get('overall_orig_sim', 0):.3f}",
             "#2196F3", "Higher = consistent"),
            ("Protected\nIdentity Sim", f"{overall_metrics.get('overall_prot_sim', 0):.3f}",
             "#F44336", "Lower = cloaked"),
            ("Reduction", f"{overall_metrics.get('overall_reduction_pct', 0):.1f}%",
             "#4CAF50", "Identity disruption"),
            ("Avg SSIM", f"{avg_ssim:.3f}", "#FF9800", "Visual quality"),
            ("Avg PSNR", f"{avg_psnr:.1f} dB", "#9C27B0", "Signal quality"),
        ]

        for i, (label, value, color, note) in enumerate(metrics):
            x = 0.05 + i * 0.19
            ax = fig.add_axes([x, 0.60, 0.17, 0.25])
            ax.set_facecolor(color + "33")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(0.5, 0.72, value, ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)
            ax.text(0.5, 0.35, label, ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
            ax.text(0.5, 0.10, note, ha="center", va="center",
                    fontsize=8, color="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_visible(False)
            rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                                  edgecolor=color, linewidth=2)
            ax.add_patch(rect)

        # Per-model bar chart
        ax2 = fig.add_axes([0.05, 0.08, 0.88, 0.42])
        ax2.set_facecolor("#16213e")
        models = summary_df["model"].tolist()
        orig_vals = summary_df["orig_mean_sim"].tolist()
        prot_vals = summary_df["prot_mean_sim"].tolist()
        x = np.arange(len(models))
        w = 0.35
        ax2.bar(x - w/2, orig_vals, w, label="Original", color="#2196F3", alpha=0.9)
        ax2.bar(x + w/2, prot_vals, w, label="Protected", color="#F44336", alpha=0.9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=12, color="white")
        ax2.set_ylabel("Mean Pairwise Cosine Similarity", color="white")
        ax2.tick_params(colors="white")
        ax2.legend(fontsize=11, facecolor="#1a1a2e", labelcolor="white")
        ax2.set_ylim(0, 1.0)
        ax2.set_title("Per-Model Identity Consistency Reduction",
                      color="white", fontsize=12, fontweight="bold")
        ax2.spines["bottom"].set_color("#444")
        ax2.spines["left"].set_color("#444")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, alpha=0.2, axis="y")

        path = save_path or os.path.join(self.output_dir, "summary_dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        return path
