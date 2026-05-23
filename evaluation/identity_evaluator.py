"""
evaluation/identity_evaluator.py
Multi-model identity consistency evaluator for SafeUpload.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List


class IdentityEvaluator:
    """Evaluates identity consistency reduction between original and protected images."""

    def __init__(self, models_dict: dict, device: str = "cuda"):
        self.models_dict = models_dict
        self.device = device

    def evaluate(self, original_faces, protected_faces) -> Dict:
        results = {}
        orig_pils = [Image.fromarray(f.astype(np.uint8)) for f in original_faces]
        prot_pils = [Image.fromarray(f.astype(np.uint8)) for f in protected_faces]

        ssim_vals, psnr_vals = self._compute_image_quality(original_faces, protected_faces)
        mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
        mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0

        for name, model in self.models_dict.items():
            print(f"  Evaluating {name}...")
            with torch.no_grad():
                orig_embs = self._extract_all(model, orig_pils)
                prot_embs = self._extract_all(model, prot_pils)

            orig_matrix = self._pairwise_sim(orig_embs)
            prot_matrix = self._pairwise_sim(prot_embs)
            orig_upper = self._upper_triangle(orig_matrix)
            prot_upper = self._upper_triangle(prot_matrix)

            results[name] = {
                "original_sim_matrix": orig_matrix,
                "protected_sim_matrix": prot_matrix,
                "original_mean_similarity": float(orig_upper.mean()) if len(orig_upper) > 0 else 0.0,
                "protected_mean_similarity": float(prot_upper.mean()) if len(prot_upper) > 0 else 0.0,
                "original_std": float(orig_upper.std()) if len(orig_upper) > 0 else 0.0,
                "protected_std": float(prot_upper.std()) if len(prot_upper) > 0 else 0.0,
                "ssim": mean_ssim,
                "psnr": mean_psnr,
                "original_embeddings": orig_embs.cpu().numpy(),
                "protected_embeddings": prot_embs.cpu().numpy(),
            }
        return results

    @torch.no_grad()
    def _extract_all(self, model, pil_faces):
        return torch.cat([model.extract_embedding(f) for f in pil_faces], dim=0)

    @staticmethod
    def _pairwise_sim(embs):
        N = embs.shape[0]
        sim = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                sim[i, j] = F.cosine_similarity(embs[i:i+1], embs[j:j+1], dim=1).item()
        return sim

    @staticmethod
    def _upper_triangle(matrix):
        N = matrix.shape[0]
        if N < 2:
            return np.array([])
        return matrix[np.triu_indices(N, k=1)]

    @staticmethod
    def _compute_image_quality(originals, protected):
        ssim_vals, psnr_vals = [], []
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
            for o, p in zip(originals, protected):
                o_f = o.astype(np.float32) / 255.0
                p_f = p.astype(np.float32) / 255.0
                ssim_vals.append(ssim_fn(o_f, p_f, data_range=1.0, channel_axis=2))
                psnr_vals.append(psnr_fn(o_f, p_f, data_range=1.0))
        except Exception as e:
            print(f"  Quality metrics error: {e}")
        return ssim_vals, psnr_vals
