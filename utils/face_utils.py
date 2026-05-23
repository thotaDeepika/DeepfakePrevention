"""
utils/face_utils.py
Face detection and alignment utilities.
Uses MTCNN from facenet_pytorch as primary detector.
"""

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import cv2


_MTCNN = None


def _get_mtcnn(device="cuda"):
    global _MTCNN
    if _MTCNN is None:
        _MTCNN = MTCNN(
            image_size=256,
            margin=40,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            keep_all=False,
            device=device,
        )
    return _MTCNN


def detect_and_align_face(
    img: Image.Image,
    output_size: int = 256,
    device: str = "cuda",
    fallback_center_crop: bool = True,
) -> Image.Image:
    """
    Detect and align largest face in image.
    Returns a PIL image of size (output_size, output_size).
    If no face detected and fallback_center_crop=True, returns a center crop.
    """
    mtcnn = _get_mtcnn(device)

    # Try MTCNN detection
    try:
        face_tensor, prob = mtcnn(img, return_prob=True)
        if face_tensor is not None and prob > 0.85:
            # face_tensor: [3, H, W] uint8 0-255
            arr = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            face_pil = Image.fromarray(arr).resize((output_size, output_size), Image.LANCZOS)
            return face_pil
    except Exception:
        pass

    if fallback_center_crop:
        return _center_crop(img, output_size)

    return None


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    """Square centre crop + resize."""
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    return img.resize((size, size), Image.LANCZOS)


def load_image_from_path(path: str) -> Image.Image:
    """Load and convert image to RGB PIL."""
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Convert PIL [H,W,3] to [1,3,H,W] float tensor in [0,1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [1,3,H,W] or [3,H,W] float [0,1] tensor to PIL."""
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)
