"""
utils/identity_manifold.py
Identity manifold construction from multi-image, multi-model embeddings.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from PIL import Image


class IdentityManifold:
    """
    Constructs the identity manifold from a set of aligned face images.

    For each model, computes:
        - embeddings: [N, D] for all N images
        - center: mean embedding (identity center)
        - pairwise similarity matrix: [N, N]
    """

    def __init__(self, models_dict: dict, device: str = "cuda"):
        """
        Args:
            models_dict: dict of {name: model_instance}
            device:      torch device
        """
        self.models_dict = models_dict
        self.device = device
        self.embeddings: Dict[str, torch.Tensor] = {}  # model_name -> [N, D]
        self.centers: Dict[str, torch.Tensor] = {}     # model_name -> [1, D]
        self.n_images = 0

    def build(self, faces: List[np.ndarray]):
        """
        Build identity manifold from aligned face images.

        Args:
            faces: list of numpy arrays [H, W, 3] uint8
        """
        self.n_images = len(faces)
        face_pils = [Image.fromarray(f.astype(np.uint8)) for f in faces]

        for name, model in self.models_dict.items():
            print(f"  Extracting embeddings: {name}...")
            embs = self._extract_all(model, face_pils)
            self.embeddings[name] = embs
            center = embs.mean(dim=0, keepdim=True)
            self.centers[name] = F.normalize(center, dim=1)

        print(f"[Manifold] Built from {self.n_images} images, {len(self.models_dict)} models.")

    @torch.no_grad()
    def _extract_all(self, model, face_pils: List[Image.Image]) -> torch.Tensor:
        """Extract and stack embeddings for all faces."""
        embs = []
        for pil in face_pils:
            e = model.extract_embedding(pil)
            embs.append(e)
        return torch.cat(embs, dim=0)  # [N, D]

    def pairwise_similarity(self) -> Dict[str, np.ndarray]:
        """
        Compute pairwise cosine similarity matrices for each model.
        Returns dict of {model_name: [N, N] numpy array}
        """
        result = {}
        for name, embs in self.embeddings.items():
            N = embs.shape[0]
            sim_matrix = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    s = F.cosine_similarity(
                        embs[i:i+1], embs[j:j+1], dim=1
                    ).item()
                    sim_matrix[i, j] = s
            result[name] = sim_matrix
        return result

    def pairwise_similarity_protected(
        self,
        protected_faces: List[np.ndarray],
        models_dict: dict = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute pairwise similarity for protected faces.
        """
        models = models_dict or self.models_dict
        face_pils = [Image.fromarray(f.astype(np.uint8)) for f in protected_faces]
        result = {}

        for name, model in models.items():
            with torch.no_grad():
                embs = self._extract_all(model, face_pils)
            N = embs.shape[0]
            sim_matrix = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    s = F.cosine_similarity(
                        embs[i:i+1], embs[j:j+1], dim=1
                    ).item()
                    sim_matrix[i, j] = s
            result[name] = sim_matrix

        return result
