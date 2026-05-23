"""
utils/image_transforms.py
Multi-Crop and augmentation transforms for MCA-inspired optimisation.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import io
import numpy as np
from PIL import Image


class MultiCropTransform:
    """
    Generates K independent random crops per iteration for
    Multi-Crop Alignment (MCA).

    Each crop applies:
      - random resized crop
      - small random translation (simulate minor shifts)
      - optional JPEG compression simulation
      - optional slight Gaussian blur
      - optional small colour jitter
    """

    def __init__(
        self,
        K: int = 8,
        crop_scale: tuple = (0.65, 1.0),
        crop_ratio: tuple = (0.85, 1.15),
        jpeg_prob: float = 0.3,
        blur_prob: float = 0.2,
        jitter_prob: float = 0.2,
        translate_prob: float = 0.3,
        output_size: int = 224,
    ):
        self.K = K
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.jpeg_prob = jpeg_prob
        self.blur_prob = blur_prob
        self.jitter_prob = jitter_prob
        self.translate_prob = translate_prob
        self.output_size = output_size

    def __call__(self, x: torch.Tensor) -> list:
        """
        x: [1, 3, H, W] or [3, H, W] float [0,1]
        Returns: list of K tensors [1, 3, output_size, output_size]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        crops = []
        for _ in range(self.K):
            crops.append(self._random_crop(x))
        return crops

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Random resized crop
        scale = random.uniform(*self.crop_scale)
        ratio = random.uniform(*self.crop_ratio)
        crop_h = int(H * scale)
        crop_w = int(crop_h * ratio)
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))

        cropped = x[:, :, top:top+crop_h, left:left+crop_w]
        resized = F.interpolate(
            cropped, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False
        )

        # JPEG compression simulation
        if random.random() < self.jpeg_prob:
            resized = self._jpeg_compress(resized)

        # Slight Gaussian blur
        if random.random() < self.blur_prob:
            resized = self._gaussian_blur(resized)

        # Colour jitter
        if random.random() < self.jitter_prob:
            resized = self._colour_jitter(resized)

        # Small translation (grid sample)
        if random.random() < self.translate_prob:
            resized = self._translate(resized)

        return resized.clamp(0, 1)

    @staticmethod
    def _jpeg_compress(x: torch.Tensor, quality: int = None) -> torch.Tensor:
        """Differentiability-free JPEG simulation via PIL round-trip."""
        quality = quality or random.randint(50, 90)
        arr = (x.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        reloaded = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
        out = torch.from_numpy(reloaded).permute(2, 0, 1).unsqueeze(0).to(x.device)
        # Preserve gradient by blending: x_grad + (jpeg_out - x).detach()
        return x + (out - x).detach()

    @staticmethod
    def _gaussian_blur(x: torch.Tensor) -> torch.Tensor:
        k = random.choice([3, 5])
        sigma = random.uniform(0.3, 1.5)
        # Simple approximation: box filter via avg_pool
        pad = k // 2
        blurred = F.avg_pool2d(
            F.pad(x, [pad]*4, mode="reflect"),
            kernel_size=k, stride=1
        )
        return blurred

    @staticmethod
    def _colour_jitter(x: torch.Tensor) -> torch.Tensor:
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        x = (x * brightness).clamp(0, 1)
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = ((x - mean) * contrast + mean).clamp(0, 1)
        return x

    @staticmethod
    def _translate(x: torch.Tensor) -> torch.Tensor:
        dx = random.uniform(-0.05, 0.05)
        dy = random.uniform(-0.05, 0.05)
        theta = torch.tensor(
            [[1.0, 0.0, dx], [0.0, 1.0, dy]],
            dtype=torch.float32
        ).unsqueeze(0).to(x.device)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


class AugmentTransform:
    """
    Mild augmentations for ATA auxiliary target perturbation.
    Intentionally conservative to stay within target manifold.
    """

    def __init__(self, output_size: int = 224):
        self.output_size = output_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Mild random resized crop [0.9, 1.0]
        _, _, H, W = x.shape
        scale = random.uniform(0.90, 1.0)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))
        x = x[:, :, top:top+crop_h, left:left+crop_w]
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode="bilinear", align_corners=False)

        # Random horizontal flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[3])

        # Mild rotation ±10 degrees
        angle = random.uniform(-10, 10)
        x = self._rotate(x, angle)

        return x.clamp(0, 1)

    @staticmethod
    def _rotate(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        import math
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=torch.float32
        ).unsqueeze(0).to(x.device)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
