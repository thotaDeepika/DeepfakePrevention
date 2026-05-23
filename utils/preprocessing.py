"""
utils/preprocessing.py
Face detection, alignment, and preprocessing pipeline for SafeUpload.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional


class FacePreprocessor:
    """
    Handles face detection and alignment from raw images.
    Uses MTCNN as primary detector with center-crop fallback.
    """

    def __init__(self, output_size: int = 256, device: str = "cuda"):
        self.output_size = output_size
        self.device = device
        self._mtcnn = None
        self._load_mtcnn()

    def _load_mtcnn(self):
        try:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                image_size=self.output_size,
                margin=40,
                min_face_size=60,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                keep_all=False,
                device=self.device,
            )
            print("[Preprocessor] MTCNN loaded.")
        except Exception as e:
            print(f"[Preprocessor] MTCNN failed ({e}), using center-crop fallback.")
            self._mtcnn = None

    def detect_and_align(self, img: Image.Image) -> Optional[np.ndarray]:
        """
        Detect and align face in image.
        Returns numpy array [H, W, 3] uint8 or None if failed.
        """
        img_rgb = img.convert("RGB")

        if self._mtcnn is not None:
            try:
                face_tensor, prob = self._mtcnn(img_rgb, return_prob=True)
                if face_tensor is not None and prob is not None and prob > 0.85:
                    arr = face_tensor.permute(1, 2, 0).cpu().numpy()
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                    face_pil = Image.fromarray(arr).resize(
                        (self.output_size, self.output_size), Image.LANCZOS
                    )
                    return np.array(face_pil)
            except Exception as e:
                print(f"[Preprocessor] MTCNN detection failed: {e}")

        # Fallback: center crop
        return self._center_crop(img_rgb)

    def _center_crop(self, img: Image.Image) -> np.ndarray:
        """Square center crop and resize."""
        w, h = img.size
        m = min(w, h)
        left = (w - m) // 2
        top = (h - m) // 2
        cropped = img.crop((left, top, left + m, top + m))
        resized = cropped.resize((self.output_size, self.output_size), Image.LANCZOS)
        return np.array(resized)

    def process_images(
        self,
        image_paths: List[str],
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Process a list of image paths.
        Returns aligned face arrays and valid paths.
        """
        aligned_faces = []
        valid_paths = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                face = self.detect_and_align(img)
                if face is not None:
                    aligned_faces.append(face)
                    valid_paths.append(path)
                    print(f"  ✓ Aligned: {os.path.basename(path)}")
                else:
                    print(f"  ✗ No face detected: {os.path.basename(path)}")
            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(path)}: {e}")

        print(f"\n[Preprocessor] {len(aligned_faces)}/{len(image_paths)} faces aligned.")
        return aligned_faces, valid_paths

    def faces_to_tensor(
        self,
        faces: List[np.ndarray],
        device: str = None,
    ) -> torch.Tensor:
        """
        Convert list of numpy face arrays to a batched float tensor.
        Returns [N, 3, H, W] in [0, 1].
        """
        device = device or self.device
        tensors = []
        for face in faces:
            arr = face.astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            tensors.append(t)
        return torch.stack(tensors, dim=0).to(device)
