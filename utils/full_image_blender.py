"""
utils/full_image_blender.py

Detects face bounding box in a full image, crops an expanded region,
runs the SafeUpload attack on that crop, then blends the protected crop
back into the full-resolution image using seamless Poisson/Gaussian blending.

Rules:
  - Attack runs ONLY on the face crop (unchanged pipeline)
  - Output is FULL original image with protected face blended in
  - No hard paste / visible square edges
  - Bounding box expanded 12% on each side to include hairline + jawline context
"""

import numpy as np
from PIL import Image
import cv2
import torch
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def detect_face_bbox(
    full_img: Image.Image,
    device: str = "cuda",
    expand_ratio: float = 0.12,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect largest face bounding box and expand it by expand_ratio on each side.

    Returns:
        (x1, y1, x2, y2) in pixel coords of the EXPANDED bbox, clipped to image,
        or None if no face found.
    """
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(
            keep_all=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=device,
        )
        img_rgb = full_img.convert("RGB")
        boxes, probs = mtcnn.detect(img_rgb)

        if boxes is None or len(boxes) == 0:
            return None

        # Pick highest-confidence detection
        best_idx = int(np.argmax(probs))
        if probs[best_idx] < 0.80:
            return None

        x1, y1, x2, y2 = boxes[best_idx]
        W, H = full_img.size

        # Expand bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * expand_ratio
        pad_y = bh * expand_ratio

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(W, int(x2 + pad_x))
        y2 = min(H, int(y2 + pad_y))

        return (x1, y1, x2, y2)

    except Exception as e:
        print(f"[Blender] Face detection failed: {e}")
        return None


def crop_face_region(
    full_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    attack_size: int = 256,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop face region from full image and resize to attack_size.

    Returns:
        face_crop_np: [attack_size, attack_size, 3] uint8
        original_crop_size: (W, H) of the bbox crop before resizing
    """
    x1, y1, x2, y2 = bbox
    crop = full_img.convert("RGB").crop((x1, y1, x2, y2))
    original_size = crop.size  # (W, H)
    resized = crop.resize((attack_size, attack_size), Image.LANCZOS)
    return np.array(resized), original_size


def blend_protected_back(
    full_img: Image.Image,
    protected_crop: np.ndarray,
    bbox: Tuple[int, int, int, int],
    original_crop_size: Tuple[int, int],
    blend_mode: str = "poisson",
    feather_px: int = 18,
) -> Image.Image:
    """
    Blend the protected face crop back into the full image seamlessly.

    Args:
        full_img:           Original full PIL image
        protected_crop:     Protected face [H, W, 3] uint8 (attack-size)
        bbox:               (x1, y1, x2, y2) expanded bbox in full image
        original_crop_size: (W, H) of the bbox region before resize-to-attack
        blend_mode:         'poisson' | 'feather'  (Poisson preferred)
        feather_px:         Gaussian feather radius for 'feather' mode

    Returns:
        Full PIL image with protected face seamlessly blended in.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    # Resize protected crop back to original bbox size
    prot_pil = Image.fromarray(protected_crop.astype(np.uint8))
    prot_resized = prot_pil.resize((bw, bh), Image.LANCZOS)
    prot_np = np.array(prot_resized)

    full_np = np.array(full_img.convert("RGB"))

    if blend_mode == "poisson":
        result = _poisson_blend(full_np, prot_np, x1, y1, x2, y2)
    else:
        result = _feather_blend(full_np, prot_np, x1, y1, x2, y2, feather_px)

    return Image.fromarray(result.astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────
# Blending implementations
# ─────────────────────────────────────────────────────────────────────

def _poisson_blend(
    full_np: np.ndarray,
    patch_np: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """
    Poisson seamless clone (OpenCV) with feather fallback on failure.
    Handles edge cases: bbox touching image border, tiny regions, etc.
    """
    H_full, W_full = full_np.shape[:2]
    bh = y2 - y1
    bw = x2 - x1

    # Poisson needs centre point strictly inside image
    cx = x1 + bw // 2
    cy = y1 + bh // 2

    # Safety: centre must be away from borders
    margin = 8
    if (cx < margin or cy < margin or
            cx > W_full - margin or cy > H_full - margin or
            bw < 20 or bh < 20):
        return _feather_blend(full_np, patch_np, x1, y1, x2, y2, feather_px=18)

    try:
        # Build elliptical mask (avoids hard corners better than rect mask)
        mask = _ellipse_mask(bh, bw)

        src_bgr  = cv2.cvtColor(patch_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        dst_bgr  = cv2.cvtColor(full_np.astype(np.uint8),  cv2.COLOR_RGB2BGR)

        blended_bgr = cv2.seamlessClone(
            src_bgr, dst_bgr, mask,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    except cv2.error:
        # Graceful fallback to feather blend
        return _feather_blend(full_np, patch_np, x1, y1, x2, y2, feather_px=18)


def _feather_blend(
    full_np: np.ndarray,
    patch_np: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    feather_px: int = 18,
) -> np.ndarray:
    """
    Gaussian feathered alpha-blend. Works everywhere, always safe.
    """
    bh = y2 - y1
    bw = x2 - x1

    # Build smooth alpha mask: 1 in centre, fade to 0 at edges
    alpha = _feather_mask(bh, bw, feather_px)          # [bh, bw, 1]

    result = full_np.copy().astype(np.float32)
    roi    = result[y1:y2, x1:x2]
    patch  = patch_np.astype(np.float32)

    # Clamp patch to roi shape (in case of 1px rounding)
    ph, pw = roi.shape[:2]
    patch  = patch[:ph, :pw]
    alpha  = alpha[:ph, :pw]

    result[y1:y2, x1:x2] = alpha * patch + (1.0 - alpha) * roi
    return np.clip(result, 0, 255)


# ─────────────────────────────────────────────────────────────────────
# Mask helpers
# ─────────────────────────────────────────────────────────────────────

def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """White filled ellipse on black background, uint8 [h, w]."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # Shrink by 4px so Poisson clone can work without border artefacts
    cv2.ellipse(
        mask,
        center=(w // 2, h // 2),
        axes=(max(1, w // 2 - 4), max(1, h // 2 - 4)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    return mask


def _feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    """
    Smooth alpha mask [h, w, 1] float32.
    1.0 at centre, smoothly fades to 0 at borders.
    """
    # Start with solid 1s, erode edges, then Gaussian-blur the border
    mask = np.ones((h, w), dtype=np.float32)
    border = max(1, feather)

    # Ramp on each side
    for i in range(border):
        alpha_val = i / border
        if i < h:
            mask[i, :]    = np.minimum(mask[i, :],    alpha_val)
            mask[h-1-i, :] = np.minimum(mask[h-1-i, :], alpha_val)
        if i < w:
            mask[:, i]    = np.minimum(mask[:, i],    alpha_val)
            mask[:, w-1-i] = np.minimum(mask[:, w-1-i], alpha_val)

    # Gaussian smooth
    k = max(3, feather | 1)  # must be odd
    mask = cv2.GaussianBlur(mask, (k, k), sigmaX=feather / 3.0)
    return mask[:, :, np.newaxis]  # [h, w, 1]


# ─────────────────────────────────────────────────────────────────────
# One-shot convenience wrapper used by the web UI
# ─────────────────────────────────────────────────────────────────────

def protect_full_image(
    full_img: Image.Image,
    attacker,
    device: str = "cuda",
    attack_size: int = 256,
    expand_ratio: float = 0.12,
    blend_mode: str = "poisson",
) -> Tuple[Image.Image, Optional[np.ndarray], Optional[Tuple]]:
    """
    End-to-end: detect face → cloak crop → blend back into full image.

    Args:
        full_img:     Full-resolution PIL image
        attacker:     SafeUploadAttack instance (already initialised)
        device:       torch device
        attack_size:  Size to resize crop to before attack
        expand_ratio: Bbox expansion fraction (0.12 = 12%)
        blend_mode:   'poisson' | 'feather'

    Returns:
        protected_full: Full PIL image with seamlessly blended protected face
        protected_crop: [H,W,3] uint8 of just the protected crop (for eval)
        bbox:           (x1,y1,x2,y2) or None
    """
    W, H = full_img.size

    # 1. Detect bbox
    bbox = detect_face_bbox(full_img, device=device, expand_ratio=expand_ratio)
    if bbox is None:
        print("[Blender] No face detected — running attack on full image centre crop.")
        # Fallback: centre 60% square crop
        m = min(W, H)
        margin = int(m * 0.20)
        cx, cy = W // 2, H // 2
        half = (m - 2 * margin) // 2
        bbox = (cx - half, cy - half, cx + half, cy + half)

    # 2. Crop + resize to attack size
    face_crop, original_crop_size = crop_face_region(full_img, bbox, attack_size)

    # 3. Run attack (unchanged pipeline, operates on crop only)
    protected_crop, delta = attacker.attack(face_crop)

    # 4. Blend back
    protected_full = blend_protected_back(
        full_img, protected_crop, bbox, original_crop_size,
        blend_mode=blend_mode,
    )

    return protected_full, protected_crop, bbox
