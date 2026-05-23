"""
visualization/embedding_plots.py
PCA, t-SNE, UMAP embedding scatter visualizations for SafeUpload.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image
from typing import Dict, List


def plot_embedding_scatter(
    original_faces: List[np.ndarray],
    protected_faces: List[np.ndarray],
    models_dict: Dict,
    output_dir: str = "outputs",
    device: str = "cuda",
    methods: List[str] = ("pca", "tsne", "umap"),
):
    """
    Visualize original vs protected embeddings using PCA, t-SNE, and UMAP.

    Args:
        original_faces:  list of [H,W,3] numpy arrays
        protected_faces: list of [H,W,3] numpy arrays
        models_dict:     {name: model_instance}
        output_dir:      save directory
        device:          torch device
        methods:         dimensionality reduction methods to use
    """
    os.makedirs(output_dir, exist_ok=True)

    orig_pils = [Image.fromarray(f.astype(np.uint8)) for f in original_faces]
    prot_pils = [Image.fromarray(f.astype(np.uint8)) for f in protected_faces]

    for model_name, model in models_dict.items():
        print(f"  Generating embedding scatter for {model_name}...")

        with torch.no_grad():
            orig_embs = np.vstack([
                model.extract_embedding(f).cpu().numpy() for f in orig_pils
            ])
            prot_embs = np.vstack([
                model.extract_embedding(f).cpu().numpy() for f in prot_pils
            ])

        all_embs = np.vstack([orig_embs, prot_embs])
        N = len(orig_embs)
        labels = ["Original"] * N + ["Protected"] * N
        colors = ["#2196F3"] * N + ["#F44336"] * N
        markers = ["o"] * N + ["^"] * N

        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            try:
                reduced = _reduce_embeddings(all_embs, method=method, n_components=2)
            except Exception as e:
                print(f"    {method} failed: {e}")
                ax.text(0.5, 0.5, f"{method}\nfailed", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Plot original
            ax.scatter(
                reduced[:N, 0], reduced[:N, 1],
                c="#2196F3", marker="o", s=120, alpha=0.85,
                edgecolors="white", linewidths=1, label="Original", zorder=3
            )
            # Plot protected
            ax.scatter(
                reduced[N:, 0], reduced[N:, 1],
                c="#F44336", marker="^", s=120, alpha=0.85,
                edgecolors="white", linewidths=1, label="Protected", zorder=3
            )

            # Connect original-protected pairs
            for i in range(N):
                ax.plot(
                    [reduced[i, 0], reduced[N + i, 0]],
                    [reduced[i, 1], reduced[N + i, 1]],
                    color="gray", alpha=0.4, linewidth=1, linestyle="--"
                )

            ax.set_title(f"{method.upper()}", fontweight="bold")
            ax.legend(loc="best")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Embedding Scatter — {model_name}\n"
            "Protected embeddings should scatter from original cluster",
            fontweight="bold", fontsize=12
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"embedding_scatter_{model_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {save_path}")


def _reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
) -> np.ndarray:
    """Apply dimensionality reduction to embeddings."""
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(5, max(2, embeddings.shape[0] // 2 - 1))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=500,
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        import umap
        n_neighbors = min(5, max(2, embeddings.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}")
