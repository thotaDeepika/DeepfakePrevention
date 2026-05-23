"""
visualization/dashboard.py
Summary dashboard for SafeUpload results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from typing import Dict, List, Optional


def plot_summary_dashboard(
    original_faces: List[np.ndarray],
    protected_faces: List[np.ndarray],
    results: Dict,
    perturbations: List[np.ndarray],
    output_dir: str = "outputs",
):
    """
    Plot comprehensive summary dashboard:
    - Before/after gallery
    - Perturbation visualization (amplified)
    - Similarity reduction bar chart per model
    - Quality metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    N = len(original_faces)
    n_models = len(results)

    # ------------------------------------------------
    # Figure layout
    # ------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(
        3, max(N, 3),
        figure=fig,
        hspace=0.45, wspace=0.3
    )

    # Row 0: Original faces
    for i, face in enumerate(original_faces[:max(N, 3)]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(face)
        ax.set_title(f"Original {i+1}", fontsize=9, fontweight="bold")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Original", fontsize=10, rotation=0, labelpad=40, va="center")

    # Row 1: Protected faces
    for i, face in enumerate(protected_faces[:max(N, 3)]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(face)
        ax.set_title(f"Protected {i+1}", fontsize=9, fontweight="bold", color="darkred")
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Protected", fontsize=10, rotation=0, labelpad=40, va="center")

    # Row 2: Similarity bar chart (span all columns)
    ax_bar = fig.add_subplot(gs[2, :])
    _plot_similarity_bars(results, ax_bar)

    # Title and annotations
    model_names = list(results.keys())
    orig_sims = [results[m]["original_mean_similarity"] for m in model_names]
    prot_sims = [results[m]["protected_mean_similarity"] for m in model_names]
    avg_orig = np.mean(orig_sims) if orig_sims else 0
    avg_prot = np.mean(prot_sims) if prot_sims else 0

    # Quality
    ssim_val = results[model_names[0]].get("ssim", 0) if model_names else 0
    psnr_val = results[model_names[0]].get("psnr", 0) if model_names else 0

    subtitle = (
        f"Avg Identity Similarity: {avg_orig:.4f} → {avg_prot:.4f} "
        f"(↓{(avg_orig - avg_prot) / max(avg_orig, 1e-8) * 100:.1f}%)    "
        f"SSIM: {ssim_val:.4f}  PSNR: {psnr_val:.2f} dB"
    )

    fig.suptitle(
        f"SafeUpload — Identity Cloaking Results\n{subtitle}",
        fontsize=13, fontweight="bold", y=1.02
    )

    # Claim box
    claim = ("\"Humans still recognize the person,\n"
             "but AI identity representations become\n"
             "less stable and less reliable.\"")
    fig.text(
        0.98, 0.02, claim,
        ha="right", va="bottom",
        fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    save_path = os.path.join(output_dir, "summary_dashboard.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Dashboard saved: {save_path}")

    # Also save before/after gallery separately
    _save_before_after(original_faces, protected_faces, perturbations, output_dir)


def _plot_similarity_bars(results: Dict, ax: plt.Axes):
    """Bar chart of original vs protected similarity."""
    model_names = list(results.keys())
    orig_sims = [results[m]["original_mean_similarity"] for m in model_names]
    prot_sims = [results[m]["protected_mean_similarity"] for m in model_names]

    x = np.arange(len(model_names))
    w = 0.35

    bars1 = ax.bar(x - w/2, orig_sims, w, label="Original", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w/2, prot_sims, w, label="Protected", color="#F44336", alpha=0.85)

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title("Identity Consistency Reduction Across Models", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)


def _save_before_after(
    original_faces, protected_faces, perturbations, output_dir
):
    """Save before/after comparison grid."""
    N = len(original_faces)
    fig, axes = plt.subplots(3, N, figsize=(4 * N, 12))
    if N == 1:
        axes = axes.reshape(-1, 1)

    row_labels = ["Original", "Protected", "Perturbation (×10)"]
    for i in range(N):
        orig = original_faces[i]
        prot = protected_faces[i]

        if perturbations and i < len(perturbations):
            delta_vis = np.clip(perturbations[i] * 10 * 255 + 128, 0, 255).astype(np.uint8)
        else:
            delta_vis = np.abs(prot.astype(int) - orig.astype(int))
            delta_vis = np.clip(delta_vis * 10 + 128, 0, 255).astype(np.uint8)

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Image {i+1}", fontsize=9)

        axes[1, i].imshow(prot)
        axes[1, i].axis("off")

        axes[2, i].imshow(delta_vis)
        axes[2, i].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold",
                                rotation=0, labelpad=50, va="center")

    plt.suptitle("Before vs After: Identity Cloaking", fontweight="bold", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "before_after_gallery.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gallery saved: {save_path}")
