"""
attacks/safeupload_attack.py
SafeUpload Attack — MCA-inspired transferable identity cloaking.

Fixed: "Trying to backward through the graph a second time"
Root causes fixed:
  1. Each crop now gets its OWN fresh forward pass (not shared with siblings)
  2. x_orig is kept detached; delta is the sole leaf requiring grad
  3. Regularisation losses are computed on detached tensors (no graph reuse)
  4. grad accumulation uses .clone() before zeroing
  5. retain_graph=False (default) — one backward per loss, graph freed immediately
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

from utils.image_transforms import MultiCropTransform


class SafeUploadAttack:
    """
    MCA-inspired transferable identity cloaking.

    Paper-aligned components:
    - Multi-Crop Alignment (MCA): K independent crops, each its own forward+backward
    - Patch Momentum: Adam-style update with bias correction
    - Ensemble: FaceNet + ArcFace + CLIP-B/16 + CLIP-L/14 joint loss
    - Directional diversion: push away from identity centre → pseudo-target
    - Quality preservation: Gaussian smooth + edge-aware masking post-process
    """

    def __init__(
        self,
        models_dict: Dict,
        manifold,
        pseudo_targets: Dict[str, torch.Tensor],
        config: Dict,
        device: str = "cuda",
    ):
        self.models_dict = models_dict
        self.manifold   = manifold
        self.pseudo_targets = pseudo_targets
        self.device     = device

        self.eps     = config.get("eps",     8 / 255)
        self.steps   = config.get("steps",   50)
        self.K       = config.get("K",       8)
        self.eot     = config.get("eot",     3)
        self.alpha   = config.get("alpha",   1.5 / 255)
        self.beta1   = config.get("beta1",   0.9)
        self.beta2   = config.get("beta2",   0.99)
        self.lambda1 = config.get("lambda1", 0.7)
        self.lambda2 = config.get("lambda2", 0.1)
        self.lambda3 = config.get("lambda3", 0.1)
        self.model_weights = config.get(
            "model_weights",
            {n: 1.0 / len(models_dict) for n in models_dict},
        )

        self.mca_transform = MultiCropTransform(
            K=self.K,
            crop_scale=(0.65, 1.0),
            jpeg_prob=0.3,
            blur_prob=0.2,
            jitter_prob=0.2,
            translate_prob=0.3,
            output_size=224,
        )

        print(
            f"[SafeUploadAttack] eps={self.eps*255:.1f}/255  "
            f"steps={self.steps}  K={self.K}  eot={self.eot}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack(self, face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cloak a single face image.

        Args:
            face: [H, W, 3] uint8 numpy array

        Returns:
            protected: [H, W, 3] uint8 numpy array
            delta_vis: [H, W, 3] float32 perturbation
        """
        # x_orig: [1, 3, H, W] float32 in [0,1] — always detached, never a leaf
        x_orig = self._to_tensor(face)  # no requires_grad

        # delta: the ONLY leaf we differentiate through
        delta = torch.zeros_like(x_orig)  # [1,3,H,W], no grad yet

        # Adam buffers (maintained outside torch autograd)
        m = torch.zeros_like(delta)
        v = torch.zeros_like(delta)
        eps_adam = 1e-8

        best_loss  = float("inf")
        best_delta = torch.zeros_like(delta)

        for step in tqdm(range(1, self.steps + 1), desc="  MCA Steps", leave=False):

            # ── Accumulate gradients across K crops × eot repeats ──────
            grad_accum     = torch.zeros_like(delta)
            total_loss_val = 0.0
            n_backward     = 0

            for _k in range(self.K):
                for _e in range(self.eot):

                    # ── Fresh leaf for each (crop, eot) iteration ───────
                    # delta_leaf is a NEW leaf; x_adv is built from it.
                    # The graph is created fresh and freed after .backward().
                    delta_leaf = delta.detach().requires_grad_(True)
                    x_adv = (x_orig + delta_leaf).clamp(0, 1)

                    # Apply ONE random crop to THIS x_adv
                    crop = self._single_crop(x_adv)  # [1,3,224,224], in graph

                    # ── Ensemble embedding loss ─────────────────────────
                    loss = self._embedding_loss(crop)

                    # ── Regularisation (detached — keeps graph small) ───
                    # TV on detached delta, MSE on detached x_adv:
                    #   these don't need gradients; they are scalar penalties
                    #   added to the scalar loss BEFORE .backward() so that
                    #   they contribute to the gradient through delta_leaf
                    #   via the chain that goes: loss → crop → x_adv → delta_leaf
                    # BUT: tv and mse need x_adv / delta in-graph for grads.
                    # We add them through delta_leaf / x_adv (already in graph).
                    tv_loss  = self._total_variation(delta_leaf)
                    mse_loss = F.mse_loss(x_adv, x_orig)  # x_orig detached ✓

                    total_loss = loss + self.lambda2 * tv_loss + self.lambda3 * mse_loss

                    # ── Single backward, graph freed immediately ─────────
                    total_loss.backward()   # retain_graph=False (default) ✓

                    total_loss_val += total_loss.item()
                    n_backward     += 1

                    # Accumulate gradient from this leaf
                    if delta_leaf.grad is not None:
                        grad_accum = grad_accum + delta_leaf.grad.detach().clone()

                    # ── Aggressive Memory Cleanup ─────────────────────────
                    # Free tensors immediately to keep VRAM and RAM spikes low
                    # This is crucial for running K=8, EOT=3 on 4GB GPUs
                    del total_loss, loss, tv_loss, mse_loss, crop, x_adv, delta_leaf
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            # ── MCA variance reduction: average over K*eot grads ───────
            n_total = max(n_backward, 1)
            grad    = grad_accum / n_total

            # ── Adam update (Patch Momentum) ─────────────────────────
            with torch.no_grad():
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                m_hat = m / (1 - self.beta1 ** step)
                v_hat = v / (1 - self.beta2 ** step)

                # Gradient DESCENT on loss (loss already contains +sim_id, -sim_pt)
                step_dir = self.alpha * m_hat / (v_hat.sqrt() + eps_adam)
                delta    = delta.detach() - step_dir

                # L-inf projection onto epsilon ball
                delta = delta.clamp(-self.eps, self.eps)

                # Keep perturbed image in valid pixel range [0, 1]
                delta = (x_orig + delta).clamp(0, 1) - x_orig

            avg_loss = total_loss_val / n_total
            if avg_loss < best_loss:
                best_loss  = avg_loss
                best_delta = delta.detach().clone()

        # ── Post-process for quality preservation ─────────────────────
        with torch.no_grad():
            x_protected = self._quality_postprocess(x_orig, best_delta)

        protected_np = self._to_numpy(x_protected)
        delta_np     = best_delta.squeeze(0).permute(1, 2, 0).cpu().numpy()

        ssim_v, psnr_v = self._compute_quality(x_orig, x_protected)
        linf = best_delta.abs().max().item() * 255
        print(f"  SSIM={ssim_v:.4f}  PSNR={psnr_v:.2f}dB  Linf={linf:.2f}/255")

        return protected_np, delta_np

    # ------------------------------------------------------------------
    # Core loss: ensemble embedding diversion
    # ------------------------------------------------------------------

    def _embedding_loss(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble identity diversion loss for a single crop.
        crop is [1,3,224,224] with grad_fn (in-graph).
        Returns a scalar loss tensor (in-graph, grad_fn present).
        """
        loss = torch.zeros(1, device=self.device, requires_grad=False)

        for name, model in self.models_dict.items():
            w  = self.model_weights.get(name, 0.25)
            ic = self.manifold.centers[name].detach()      # identity centre [1,D]
            pt = self.pseudo_targets[name].detach()        # pseudo-target  [1,D]

            adv_emb = model.extract_embedding_grad(crop)   # [1,D], in-graph

            ic_m = self._match_dim(ic, adv_emb)
            pt_m = self._match_dim(pt, adv_emb)

            sim_id = F.cosine_similarity(adv_emb, ic_m, dim=1).mean()
            sim_pt = F.cosine_similarity(adv_emb, pt_m, dim=1).mean()

            # Minimise: sim to identity, maximise: sim to pseudo-target
            model_loss = sim_id - self.lambda1 * sim_pt
            loss = loss + w * model_loss

        return loss

    # ------------------------------------------------------------------
    # Crop: ONE crop per call, differentiable
    # ------------------------------------------------------------------

    def _single_crop(self, x_adv: torch.Tensor) -> torch.Tensor:
        """
        Apply a single random crop+resize+aug to x_adv.
        Keeps the computational graph intact (no in-place ops, no PIL conversion).
        """
        import random
        _, _, H, W = x_adv.shape
        output_size = 224

        # Random crop parameters
        scale = random.uniform(0.65, 1.0)
        crop_h = max(1, int(H * scale))
        crop_w = max(1, int(W * scale))
        top  = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))

        cropped = x_adv[:, :, top:top + crop_h, left:left + crop_w]

        resized = F.interpolate(
            cropped,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )

        # Optional: colour jitter (in-graph safe, no in-place)
        if random.random() < 0.2:
            brightness = random.uniform(0.88, 1.12)
            resized = (resized * brightness).clamp(0, 1)

        # Optional: translation via affine grid (in-graph safe)
        if random.random() < 0.3:
            dx = random.uniform(-0.04, 0.04)
            dy = random.uniform(-0.04, 0.04)
            theta = torch.tensor(
                [[1.0, 0.0, dx], [0.0, 1.0, dy]],
                dtype=torch.float32, device=x_adv.device,
            ).unsqueeze(0)
            grid   = F.affine_grid(theta, resized.size(), align_corners=False)
            resized = F.grid_sample(
                resized, grid, align_corners=False, padding_mode="reflection"
            )

        return resized

    # ------------------------------------------------------------------
    # Quality post-processing (no_grad, called once after loop)
    # ------------------------------------------------------------------

    def _quality_postprocess(
        self, x_orig: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        delta_s  = self._gaussian_smooth(delta)
        edge_m   = self._edge_mask(x_orig)
        delta_m  = delta_s * (1.0 - 0.35 * edge_m)
        delta_f  = delta_m.clamp(-self.eps, self.eps)
        return (x_orig + delta_f).clamp(0, 1)

    @staticmethod
    def _gaussian_smooth(x: torch.Tensor, k: int = 3, sigma: float = 0.8) -> torch.Tensor:
        coords  = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        gauss   = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss  /= gauss.sum()
        kernel  = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).view(1, 1, k, k).repeat(3, 1, 1, 1)
        pad     = k // 2
        return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), kernel, groups=3)

    @staticmethod
    def _edge_mask(x: torch.Tensor) -> torch.Tensor:
        gray   = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        sx     = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               dtype=torch.float32, device=x.device).view(1,1,3,3)
        sy     = sx.transpose(2, 3)
        pad    = F.pad(gray, [1,1,1,1], mode="reflect")
        mag    = ((F.conv2d(pad, sx) ** 2 + F.conv2d(pad, sy) ** 2) ** 0.5)
        return (mag / (mag.max() + 1e-8)).clamp(0, 1)

    @staticmethod
    def _total_variation(delta: torch.Tensor) -> torch.Tensor:
        return (
            (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
            + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, face: np.ndarray) -> torch.Tensor:
        """[H,W,3] uint8 → [1,3,H,W] float32 [0,1], NO grad."""
        arr = face.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    @staticmethod
    def _match_dim(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        t_d, r_d = target.shape[1], ref.shape[1]
        if t_d == r_d:
            return target
        if t_d > r_d:
            return F.normalize(target[:, :r_d], dim=1)
        return F.normalize(F.pad(target, (0, r_d - t_d)), dim=1)

    @staticmethod
    def _compute_quality(x_orig: torch.Tensor, x_prot: torch.Tensor):
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
            o = x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
            p = x_prot.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (
                ssim_fn(o, p, data_range=1.0, channel_axis=2),
                psnr_fn(o, p, data_range=1.0),
            )
        except Exception:
            return 0.0, 0.0
