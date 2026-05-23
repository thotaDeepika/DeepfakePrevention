"""
utils/pseudo_target.py
Synthetic pseudo-target embedding generation compatible with IdentityManifold.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class PseudoTargetGenerator:
    """
    Generates pseudo-target embeddings per model for directional identity diversion.
    No real person is targeted.
    """

    STRATEGIES = ("orthogonal", "orthogonal_pca", "random_manifold", "averaged_neutral")

    def __init__(self, manifold=None, device: str = "cuda", seed: int = 42):
        self.manifold = manifold
        self.device = device
        self.rng = np.random.default_rng(seed)

    def generate(self, strategy: str = "orthogonal_pca") -> Dict[str, torch.Tensor]:
        """Generate pseudo-targets for each model. Returns {model_name: [1, D]}."""
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}"
        assert self.manifold is not None, "IdentityManifold required"

        result = {}
        for name in self.manifold.models_dict:
            center = self.manifold.centers[name]
            embeddings = self.manifold.embeddings[name]
            D = center.shape[1]

            if strategy == "orthogonal":
                pt = self._orthogonal(center, D)
            elif strategy == "orthogonal_pca":
                pt = self._pca_orthogonal(center, embeddings, D)
            elif strategy == "random_manifold":
                pt = self._random_manifold(center, D)
            else:
                pt = self._averaged_neutral(D)

            result[name] = pt.to(self.device)
            sim = F.cosine_similarity(center, result[name], dim=1).item()
            print(f"  [{name}] pseudo-target cosine sim to center: {sim:.4f}")

        return result

    def _orthogonal(self, center: torch.Tensor, D: int) -> torch.Tensor:
        c = center.squeeze(0).cpu().numpy()
        v = self.rng.standard_normal(D).astype(np.float32)
        v -= (np.dot(v, c) / (np.dot(c, c) + 1e-8)) * c
        v /= np.linalg.norm(v) + 1e-8
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    def _pca_orthogonal(self, center, embeddings, D):
        from sklearn.decomposition import PCA
        emb_np = embeddings.cpu().numpy()
        n_comp = min(emb_np.shape[0] - 1, min(D, 8))
        if n_comp < 1:
            return self._orthogonal(center, D)
        try:
            pca = PCA(n_components=n_comp)
            pca.fit(emb_np)
            v = self.rng.standard_normal(D).astype(np.float32)
            for comp in pca.components_:
                v -= np.dot(v, comp) * comp
            c = center.squeeze(0).cpu().numpy()
            v -= (np.dot(v, c) / (np.dot(c, c) + 1e-8)) * c
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                return self._orthogonal(center, D)
            v /= norm
            return torch.tensor(v, dtype=torch.float32).unsqueeze(0)
        except Exception:
            return self._orthogonal(center, D)

    def _random_manifold(self, center, D):
        c = center.squeeze(0).cpu().numpy()
        v = self.rng.standard_normal(D).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        if np.dot(v, c) > 0:
            v = -v
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    def _averaged_neutral(self, D):
        vectors = [self.rng.standard_normal(D).astype(np.float32) for _ in range(16)]
        avg = np.mean(vectors, axis=0)
        avg /= np.linalg.norm(avg) + 1e-8
        return torch.tensor(avg, dtype=torch.float32).unsqueeze(0)
