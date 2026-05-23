"""
attacks/mca_attack.py

SafeUpload core attack: MCA-Inspired Identity Cloaking.

Pipeline per image:
  1. Extract multi-model embeddings
  2. Build identity manifold (center + pseudo-target)
  3. Optimise with MCA + ensemble loss (Adam/MI-FGSM)
  4. Apply visual quality preservation
  5. Return protected image + metrics

Key adaptations from M-Attack-V2 paper:
  - Multi-Crop Alignment (MCA): average gradients over K crops
  - Auxiliary Target Alignment (ATA): mild transforms on pseudo-target
  - Patch Momentum: Adam with gradient replay
  - Ensemble: FaceNet + CLIP-B16 + CLIP-L14 (ArcFace optional)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

from models import FaceNetModel, ArcFaceModel, CLIPb16Model, CLIPl14Model
from losses import IdentityDiversionLoss
from utils.pseudo_target import PseudoTargetGenerator
from utils.image_transforms import MultiCropTransform, AugmentTransform
from utils.face_utils import pil_to_tensor, tensor_to_pil


class MCAIdentityCloakAttack:
    """
    Multi-Crop Alignment Identity Cloaking Attack.

    Args:
        eps:          L-inf perturbation budget (default 8/255)
        steps:        optimisation steps (default 50)
        K:            number of MCA crops per step (default 8)
        eot:          expectation over transformations repeats (default 3)
        lr:           Adam learning rate
        lambda1:      pseudo-target attraction weight
        lambda2:      smoothness weight
        lambda3:      perceptual weight
        use_arcface:  include ArcFace in ensemble (slower, requires insightface)
        device:       torch device
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 50,
        K: int = 8,
        eot: int = 3,
        lr: float = 0.005,
        lambda1: float = 0.5,
        lambda2: float = 0.1,
        lambda3: float = 0.05,
        pseudo_strategy: str = "pca_project",
        use_arcface: bool = False,
        device: str = "cuda",
    ):
        self.eps = eps
        self.steps = steps
        self.K = K
        self.eot = eot
        self.lr = lr
        self.device = device
        self.use_arcface = use_arcface

        # Load models
        print("[Attack] Loading embedding models...")
        self.facenet = FaceNetModel(device=device)
        self.clip_b16 = CLIPb16Model(device=device)
        self.clip_l14 = CLIPl14Model(device=device)
        if use_arcface:
            try:
                self.arcface = ArcFaceModel(device=device)
                print("[Attack] ArcFace loaded.")
            except Exception as e:
                print(f"[Attack] ArcFace failed: {e}. Skipping.")
                self.arcface = None
                self.use_arcface = False
        else:
            self.arcface = None

        # Ensemble model list and weights
        self.models = [self.facenet, self.clip_b16, self.clip_l14]
        self.model_names = ["FaceNet", "CLIP-B16", "CLIP-L14"]
        if self.arcface:
            self.models.append(self.arcface)
            self.model_names.append("ArcFace")
        self.weights = [1.0 / len(self.models)] * len(self.models)

        # Loss
        self.loss_fn = IdentityDiversionLoss(
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
        )

        # Transforms
        self.mca_transform = MultiCropTransform(
            K=K,
            crop_scale=(0.65, 1.0),
            jpeg_prob=0.3,
            blur_prob=0.2,
            jitter_prob=0.2,
            translate_prob=0.3,
            output_size=224,
        )
        self.ata_transform = AugmentTransform(output_size=224)

        # Pseudo-target generator
        self.pseudo_gen = PseudoTargetGenerator(strategy=pseudo_strategy)

        print(f"[Attack] Ready. Models: {self.model_names}")
        print(f"[Attack] eps={eps:.4f}, steps={steps}, K={K}, eot={eot}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_identity_manifold(self, images_tensor: torch.Tensor):
        """
        Compute identity center and per-model embeddings.

        Args:
            images_tensor: [N, 3, H, W] in [0,1]

        Returns:
            identity_center: dict model_name -> [1, D]
            all_embeddings:  dict model_name -> [N, D]
        """
        identity_center = {}
        all_embeddings = {}

        for model, name in zip(self.models, self.model_names):
            embs = []
            for i in range(images_tensor.shape[0]):
                e = model.extract_embedding(images_tensor[i:i+1])
                embs.append(e)
            embs_t = torch.cat(embs, dim=0)   # [N, D]
            center = embs_t.mean(dim=0, keepdim=True)
            center = F.normalize(center, dim=1)
            identity_center[name] = center
            all_embeddings[name] = embs_t

        return identity_center, all_embeddings

    def cloak_single(
        self,
        img: Image.Image,
        identity_center: dict,
        all_embeddings: dict,
        verbose: bool = True,
    ) -> tuple:
        """
        Apply identity cloaking to a single PIL image.

        Returns:
            protected_img (PIL.Image), metrics (dict)
        """
        # Convert to tensor
        x_orig = pil_to_tensor(img, device=self.device)  # [1, 3, H, W]

        # Generate pseudo-targets per model
        pseudo_targets = {}
        for name, model in zip(self.model_names, self.models):
            center = identity_center[name]
            embs = all_embeddings[name]
            pt = self.pseudo_gen.generate(center, embs)
            # Match dim if needed
            pseudo_targets[name] = pt

        # Initialise perturbation
        delta = torch.zeros_like(x_orig, requires_grad=False)
        delta = delta.to(self.device)

        # Adam state
        m = torch.zeros_like(delta)
        v = torch.zeros_like(delta)
        beta1, beta2, eta = 0.9, 0.99, 1e-8

        best_loss = float("inf")
        best_delta = delta.clone()

        pbar = tqdm(range(1, self.steps + 1), desc="Cloaking", leave=False) if verbose else range(1, self.steps + 1)

        for step in pbar:
            delta.requires_grad_(True)
            x_adv = (x_orig + delta).clamp(0, 1)

            # Accumulate gradients over K crops × eot × ensemble
            grad_accum = torch.zeros_like(delta)
            total_loss = 0.0

            # MCA: K random crops
            crops = self.mca_transform(x_adv)  # list of K tensors [1,3,224,224]

            for crop in crops:
                for _ in range(self.eot):
                    # Get embeddings for this crop from all models
                    adv_embs = []
                    for model, name in zip(self.models, self.model_names):
                        emb = model.extract_embedding_grad(crop)
                        adv_embs.append(emb)

                    # Compute ensemble loss
                    ic_list = [identity_center[n] for n in self.model_names]
                    pt_list = [pseudo_targets[n] for n in self.model_names]

                    # Per-model loss
                    loss = torch.tensor(0.0, device=self.device)
                    for emb, ic, pt, w in zip(adv_embs, ic_list, pt_list, self.weights):
                        pt_matched = self.loss_fn._match_dim(pt, emb)
                        ic_matched = self.loss_fn._match_dim(ic, emb)
                        sim_id = F.cosine_similarity(emb, ic_matched, dim=1).mean()
                        sim_ps = F.cosine_similarity(emb, pt_matched, dim=1).mean()
                        loss = loss + w * (-sim_id - self.loss_fn.lambda1 * sim_ps)

                    # Add regularisation on full image
                    loss = loss + self.loss_fn.lambda2 * self.loss_fn._total_variation(delta)
                    loss = loss + self.loss_fn.lambda3 * F.mse_loss(x_adv, x_orig)

                    loss.backward()
                    total_loss += loss.item()

                    if delta.grad is not None:
                        grad_accum += delta.grad.detach()
                        delta.grad.zero_()

            # Average gradients
            grad = grad_accum / (self.K * self.eot)

            # Adam update (Patch Momentum)
            with torch.no_grad():
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                update = self.lr * m_hat / (v_hat.sqrt() + eta)

                delta = delta.detach() - update  # minimise loss
                # L-inf projection
                delta = delta.clamp(-self.eps, self.eps)
                # Ensure valid image
                delta = (x_orig + delta).clamp(0, 1) - x_orig

            avg_loss = total_loss / (self.K * self.eot)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_delta = delta.clone()

            if verbose and hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "eps_used": f"{delta.abs().max().item():.4f}"})

        # Apply best delta with quality preservation
        x_protected = self._apply_quality_preservation(x_orig, best_delta)

        # Compute final metrics
        metrics = self._compute_metrics(x_orig, x_protected, best_delta)

        protected_pil = tensor_to_pil(x_protected)
        return protected_pil, metrics

    def cloak_batch(
        self,
        images: list,
        identity_center: dict,
        all_embeddings: dict,
        verbose: bool = True,
    ) -> list:
        """
        Cloak a list of PIL images.
        Returns list of (protected_pil, metrics).
        """
        results = []
        for i, img in enumerate(images):
            if verbose:
                print(f"\n[Attack] Cloaking image {i+1}/{len(images)}...")
            protected, metrics = self.cloak_single(
                img, identity_center, all_embeddings, verbose=verbose
            )
            results.append((protected, metrics))
        return results

    # ------------------------------------------------------------------
    # Quality preservation
    # ------------------------------------------------------------------

    def _apply_quality_preservation(
        self,
        x_orig: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply visual quality preservation techniques:
        1. Gaussian smoothing of perturbation
        2. Edge-aware masking (reduce perturbation at strong edges)
        3. Low-frequency constraint
        4. Final L-inf clip
        """
        with torch.no_grad():
            delta_smooth = self._gaussian_smooth(delta, kernel_size=3, sigma=0.8)
            edge_mask = self._compute_edge_mask(x_orig)
            # Reduce perturbation near edges (preserve fine detail)
            delta_masked = delta_smooth * (1.0 - 0.4 * edge_mask)
            # Re-clip
            delta_final = delta_masked.clamp(-self.eps, self.eps)
            x_protected = (x_orig + delta_final).clamp(0, 1)
        return x_protected

    @staticmethod
    def _gaussian_smooth(x: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian smoothing to perturbation."""
        import math
        # Build Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1).to(x.device)
        pad = kernel_size // 2
        x_pad = F.pad(x, [pad] * 4, mode="reflect")
        return F.conv2d(x_pad, kernel, groups=3)

    @staticmethod
    def _compute_edge_mask(x: torch.Tensor) -> torch.Tensor:
        """
        Simple Sobel-based edge mask in [0,1].
        High values = strong edges → reduce perturbation there.
        """
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3).to(x.device)
        sobel_y = sobel_x.transpose(2, 3)
        pad = F.pad(gray, [1, 1, 1, 1], mode="reflect")
        gx = F.conv2d(pad, sobel_x)
        gy = F.conv2d(pad, sobel_y)
        magnitude = (gx ** 2 + gy ** 2).sqrt()
        mask = (magnitude / (magnitude.max() + 1e-8)).clamp(0, 1)
        return mask

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_metrics(
        self,
        x_orig: torch.Tensor,
        x_protected: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        import numpy as np

        orig_np = x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
        prot_np = x_protected.squeeze(0).permute(1, 2, 0).cpu().numpy()

        ssim_val = ssim(orig_np, prot_np, data_range=1.0, channel_axis=2)
        psnr_val = psnr(orig_np, prot_np, data_range=1.0)
        linf = delta.abs().max().item()
        l2 = delta.norm(p=2).item()

        return {
            "ssim": round(ssim_val, 4),
            "psnr": round(psnr_val, 2),
            "linf": round(linf, 5),
            "l2": round(l2, 4),
        }
