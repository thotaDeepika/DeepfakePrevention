"""
visualization/embedding_viz.py
PCA, t-SNE, and UMAP embedding visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


class EmbeddingVisualizer:

    ORIG_COLOR = "#2196F3"   # Blue
    PROT_COLOR = "#F44336"   # Red
    CENTER_COLOR = "#4CAF50" # Green

    def __init__(self, output_dir: str = "outputs/viz"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_pca(
        self,
        orig_embs: np.ndarray,
        prot_embs: np.ndarray,
        model_name: str = "Model",
        save_path: str = None,
    ) -> str:
        """Plot PCA 2D projection of original vs protected embeddings."""
        all_embs = np.vstack([orig_embs, prot_embs])
        pca = PCA(n_components=2, random_state=42)
        all_proj = pca.fit_transform(all_embs)

        n_orig = orig_embs.shape[0]
        orig_proj = all_proj[:n_orig]
        prot_proj = all_proj[n_orig:]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(orig_proj[:, 0], orig_proj[:, 1],
                   c=self.ORIG_COLOR, s=120, marker="o",
                   label="Original", zorder=3, edgecolors="white", linewidths=1.5)
        ax.scatter(prot_proj[:, 0], prot_proj[:, 1],
                   c=self.PROT_COLOR, s=120, marker="^",
                   label="Protected", zorder=3, edgecolors="white", linewidths=1.5)

        # Draw convex hulls if enough points
        if n_orig >= 3:
            self._draw_hull(ax, orig_proj, self.ORIG_COLOR, alpha=0.12)
            self._draw_hull(ax, prot_proj, self.PROT_COLOR, alpha=0.12)

        # Connect pairs
        for o, p in zip(orig_proj, prot_proj):
            ax.annotate("", xy=p, xytext=o,
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=1))

        ax.set_title(f"PCA Embedding Space — {model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, f"pca_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_tsne(
        self,
        orig_embs: np.ndarray,
        prot_embs: np.ndarray,
        model_name: str = "Model",
        save_path: str = None,
        perplexity: int = None,
    ) -> str:
        """Plot t-SNE 2D projection."""
        n_orig = orig_embs.shape[0]
        all_embs = np.vstack([orig_embs, prot_embs])

        # t-SNE needs > perplexity samples
        perp = min(perplexity or 5, max(2, len(all_embs) - 1))

        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    n_iter=1000, learning_rate="auto", init="pca")
        all_proj = tsne.fit_transform(all_embs)

        orig_proj = all_proj[:n_orig]
        prot_proj = all_proj[n_orig:]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(orig_proj[:, 0], orig_proj[:, 1],
                   c=self.ORIG_COLOR, s=120, marker="o",
                   label="Original", zorder=3, edgecolors="white", linewidths=1.5)
        ax.scatter(prot_proj[:, 0], prot_proj[:, 1],
                   c=self.PROT_COLOR, s=120, marker="^",
                   label="Protected", zorder=3, edgecolors="white", linewidths=1.5)

        ax.set_title(f"t-SNE Embedding Space — {model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, f"tsne_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_umap(
        self,
        orig_embs: np.ndarray,
        prot_embs: np.ndarray,
        model_name: str = "Model",
        save_path: str = None,
    ) -> str:
        """Plot UMAP 2D projection (falls back to PCA if umap unavailable)."""
        try:
            import umap
            n_orig = orig_embs.shape[0]
            all_embs = np.vstack([orig_embs, prot_embs])
            n_neighbors = min(5, len(all_embs) - 1)
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=n_neighbors, min_dist=0.1)
            all_proj = reducer.fit_transform(all_embs)
            orig_proj = all_proj[:n_orig]
            prot_proj = all_proj[n_orig:]
            title = f"UMAP Embedding Space — {model_name}"
            xlabel, ylabel = "UMAP 1", "UMAP 2"
        except ImportError:
            print("[Viz] UMAP not available, falling back to PCA.")
            return self.plot_pca(orig_embs, prot_embs, model_name, save_path)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(orig_proj[:, 0], orig_proj[:, 1],
                   c=self.ORIG_COLOR, s=120, marker="o",
                   label="Original", zorder=3, edgecolors="white", linewidths=1.5)
        ax.scatter(prot_proj[:, 0], prot_proj[:, 1],
                   c=self.PROT_COLOR, s=120, marker="^",
                   label="Protected", zorder=3, edgecolors="white", linewidths=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, f"umap_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def plot_all(
        self,
        orig_embs_dict: dict,
        prot_embs_dict: dict,
    ) -> dict:
        """Generate PCA, t-SNE, UMAP for all models."""
        paths = {}
        for name in orig_embs_dict:
            orig = orig_embs_dict[name]
            prot = prot_embs_dict[name]
            if isinstance(orig, __import__("torch").Tensor):
                orig = orig.cpu().numpy()
                prot = prot.cpu().numpy()
            paths[name] = {
                "pca": self.plot_pca(orig, prot, name),
                "tsne": self.plot_tsne(orig, prot, name),
                "umap": self.plot_umap(orig, prot, name),
            }
        return paths

    @staticmethod
    def _draw_hull(ax, points, color, alpha=0.15):
        """Draw convex hull around point cloud."""
        try:
            from scipy.spatial import ConvexHull
            if len(points) < 3:
                return
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=alpha)
        except Exception:
            pass
