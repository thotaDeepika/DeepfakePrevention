"""
losses/identity_diversion_loss.py

Targeted Identity Diversion Loss for SafeUpload.

L = -sim(adv_emb, identity_center)          # reduce identity similarity
  + lambda1 * sim(adv_emb, pseudo_target)   # divert toward pseudo target
  + lambda2 * smoothness_loss               # perceptual smoothness
  + lambda3 * perceptual_quality_loss       # preserve visual quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityDiversionLoss(nn.Module):
    """
    Computes directional identity cloaking loss.

    Args:
        lambda1: weight for pseudo-target attraction
        lambda2: weight for smoothness regularisation
        lambda3: weight for perceptual quality (SSIM proxy)
    """

    def __init__(
        self,
        lambda1: float = 0.5,
        lambda2: float = 0.1,
        lambda3: float = 0.05,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(
        self,
        adv_embedding: torch.Tensor,
        identity_center: torch.Tensor,
        pseudo_target: torch.Tensor,
        adv_image: torch.Tensor,
        original_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            adv_embedding:   [1, D] adversarial image embedding
            identity_center: [1, D] mean identity embedding
            pseudo_target:   [1, D] synthetic target embedding
            adv_image:       [1, 3, H, W] adversarial image in [0,1]
            original_image:  [1, 3, H, W] clean image in [0,1]

        Returns:
            scalar loss tensor
        """
        # 1. Identity dissimilarity: push away from identity center
        sim_identity = F.cosine_similarity(adv_embedding, identity_center, dim=1)
        loss_identity = -sim_identity.mean()  # minimise = maximise dissimilarity

        # 2. Pseudo-target attraction: pull toward synthetic direction
        sim_pseudo = F.cosine_similarity(adv_embedding, pseudo_target, dim=1)
        loss_pseudo = -self.lambda1 * sim_pseudo.mean()  # maximise similarity to pseudo

        # 3. Smoothness loss: total variation
        loss_smooth = self.lambda2 * self._total_variation(adv_image - original_image)

        # 4. Perceptual quality: L2 in pixel space (soft constraint)
        loss_perceptual = self.lambda3 * F.mse_loss(adv_image, original_image)

        total = loss_identity + loss_pseudo + loss_smooth + loss_perceptual
        return total

    def forward_ensemble(
        self,
        adv_embeddings: list,
        identity_center: torch.Tensor,
        pseudo_target: torch.Tensor,
        adv_image: torch.Tensor,
        original_image: torch.Tensor,
        weights: list = None,
    ) -> torch.Tensor:
        """
        Ensemble version: average loss across multiple model embeddings.

        Args:
            adv_embeddings: list of [1, D_i] tensors from different models
            weights:        per-model weights (uniform if None)
        """
        if weights is None:
            weights = [1.0 / len(adv_embeddings)] * len(adv_embeddings)

        total_identity_loss = 0.0
        total_pseudo_loss = 0.0

        for emb, w in zip(adv_embeddings, weights):
            # Project pseudo_target to same dim if needed
            pt = self._match_dim(pseudo_target, emb)
            ic = self._match_dim(identity_center, emb)

            sim_id = F.cosine_similarity(emb, ic, dim=1).mean()
            sim_ps = F.cosine_similarity(emb, pt, dim=1).mean()

            total_identity_loss += w * (-sim_id)
            total_pseudo_loss += w * (-self.lambda1 * sim_ps)

        loss_smooth = self.lambda2 * self._total_variation(adv_image - original_image)
        loss_perceptual = self.lambda3 * F.mse_loss(adv_image, original_image)

        return total_identity_loss + total_pseudo_loss + loss_smooth + loss_perceptual

    @staticmethod
    def _total_variation(delta: torch.Tensor) -> torch.Tensor:
        """Anisotropic total variation of perturbation delta."""
        tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
        tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    @staticmethod
    def _match_dim(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        If target and reference have different embedding dims,
        apply a simple linear projection (random, fixed) to match dims.
        Used to handle FaceNet (512) vs CLIP-L/14 (768) differences.
        """
        t_dim = target.shape[1]
        r_dim = reference.shape[1]
        if t_dim == r_dim:
            return target
        # Simple truncation or padding
        if t_dim > r_dim:
            return F.normalize(target[:, :r_dim], dim=1)
        else:
            padded = F.pad(target, (0, r_dim - t_dim))
            return F.normalize(padded, dim=1)
