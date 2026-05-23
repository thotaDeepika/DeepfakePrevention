# SafeUpload: Transferable Identity Cloaking for Public Images

> **Research Prototype** | Defensive Privacy Tool | Ethical Use Only

SafeUpload applies subtle transferable adversarial perturbations to face images before public upload, reducing AI identity consistency while preserving human-recognizable appearance.

---

## Research Claim

> "Transferable adversarial identity cloaking can reduce the consistency and reliability of AI-based face representation systems across multiple images and models while preserving human-recognizable appearance."

---

## Quick Start (Google Colab)

1. Open `SafeUpload_Colab.ipynb` in Colab
2. Set Runtime → **GPU** (T4 or better)
3. Run **Cell 1** to install dependencies
4. Run **Cell 2** to upload and extract this project ZIP
5. Run **Cell 3** to verify GPU
6. Run **Cell 4** to upload your face images ZIP (5–6 images of same person)
7. Run remaining cells sequentially

---

## Project Structure

```
safeupload/
├── SafeUpload_Colab.ipynb       ← Main entry point
├── requirements.txt
├── pipeline.py                  ← Full pipeline orchestrator
│
├── attacks/
│   ├── safeupload_attack.py     ← Main attack (used by notebook)
│   └── mca_attack.py            ← MCA-based identity cloaking core
│
├── models/
│   ├── facenet.py               ← FaceNet (512-d)
│   ├── arcface.py               ← ArcFace (512-d, insightface)
│   ├── clip_b16.py              ← CLIP ViT-B/16 (512-d)
│   └── clip_l14.py              ← CLIP ViT-L/14 (768-d)
│
├── losses/
│   └── identity_diversion_loss.py   ← Directional identity diversion loss
│
├── utils/
│   ├── preprocessing.py         ← Face detection & alignment
│   ├── identity_manifold.py     ← Identity manifold construction
│   ├── pseudo_target.py         ← Synthetic pseudo-target generation
│   ├── image_transforms.py      ← MCA crops & ATA augmentations
│   ├── face_utils.py            ← Face detection helpers
│   └── io_utils.py              ← ZIP loading, CSV export
│
├── evaluation/
│   ├── identity_evaluator.py    ← Pairwise similarity evaluation
│   └── vision_eval.py           ← Gemini ambiguity evaluation
│
├── visualization/
│   ├── heatmaps.py              ← Similarity heatmaps
│   ├── embedding_plots.py       ← PCA/t-SNE/UMAP scatter
│   ├── dashboard.py             ← Summary dashboard
│   ├── heatmap_viz.py           ← (pipeline viz)
│   ├── embedding_viz.py         ← (pipeline viz)
│   └── comparison_viz.py        ← (pipeline viz)
│
└── outputs/                     ← All results saved here
```

---

## Algorithm: MCA-Inspired Identity Cloaking

Adapted from the M-Attack-V2 paper ("Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting"):

### Key Components

| Component | Purpose |
|-----------|---------|
| **Multi-Crop Alignment (MCA)** | Average gradients over K=8 random crops per step — reduces variance, improves transferability |
| **Patch Momentum (Adam)** | Stable directional optimization with gradient replay |
| **Ensemble Attack** | Joint optimization across FaceNet + ArcFace + CLIP-B/16 + CLIP-L/14 |
| **Directional Identity Diversion** | Push embeddings away from identity center toward synthetic pseudo-target |
| **Visual Quality Preservation** | Gaussian smoothing + edge-aware masking + TV regularization |

### Loss Function

```
L = sim(adv_emb, identity_center)         # reduce identity similarity
  - λ₁ · sim(adv_emb, pseudo_target)      # divert toward synthetic target
  + λ₂ · TV(δ)                            # smoothness
  + λ₃ · MSE(x_adv, x_orig)              # perceptual quality
```

### Pseudo-Target Generation

No real people are targeted. Synthetic targets are generated via:
- **Orthogonal PCA**: find direction orthogonal to identity subspace
- **Gram-Schmidt orthogonalization** against identity center
- Ensures face-space semantics are preserved while identity is diverted

---

## Attack Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `eps` | 8/255 | L-∞ perturbation budget |
| `steps` | 50 | Optimization iterations |
| `K` | 8 | MCA crops per step |
| `eot` | 3 | Expectation over transformations |
| `alpha` | 1.5/255 | Adam step size |
| `lambda1` | 0.7 | Pseudo-target weight |
| `lambda2` | 0.1 | Smoothness weight |
| `lambda3` | 0.1 | Perceptual quality weight |

---

## Expected Results

| Metric | Original | Protected |
|--------|----------|-----------|
| Pairwise cosine similarity | 0.85–0.95 | 0.45–0.65 |
| SSIM | — | > 0.85 |
| PSNR | — | > 30 dB |

---

## Ethical Guidelines

✅ **Allowed**:
- Protecting your own photos before public upload
- Privacy research and evaluation
- Defensive identity cloaking

❌ **Not Allowed**:
- Targeting real people without consent
- Face swapping or impersonation
- Creating malicious identity manipulation tools

---

## Citation

This work adapts techniques from:

> Zhao et al., "Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting", MBZUAI VILA Lab, 2026.
