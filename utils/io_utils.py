"""
utils/io_utils.py
File I/O utilities: ZIP loading, image saving, CSV export.
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_zip_images(zip_path: str, extract_dir: str = "/tmp/safeupload_input") -> List[Tuple[str, Image.Image]]:
    """
    Extract ZIP and load all supported images.
    Returns list of (filename, PIL.Image) tuples.
    """
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    images = []
    for root, _, files in os.walk(extract_dir):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTS:
                fpath = os.path.join(root, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    images.append((fname, img))
                except Exception as e:
                    print(f"[IO] Warning: could not load {fpath}: {e}")

    print(f"[IO] Loaded {len(images)} images from ZIP.")
    return images


def load_images_from_folder(folder: str) -> List[Tuple[str, Image.Image]]:
    """Load all images from a directory."""
    images = []
    for fname in sorted(os.listdir(folder)):
        ext = Path(fname).suffix.lower()
        if ext in SUPPORTED_EXTS:
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                images.append((fname, img))
            except Exception as e:
                print(f"[IO] Warning: {e}")
    return images


def save_protected_images(
    protected_images: List[Tuple[str, Image.Image]],
    output_dir: str,
    suffix: str = "_protected",
) -> List[str]:
    """
    Save protected images to output_dir.
    Returns list of saved paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for fname, img in protected_images:
        stem = Path(fname).stem
        ext = Path(fname).suffix or ".png"
        out_name = f"{stem}{suffix}{ext}"
        out_path = os.path.join(output_dir, out_name)
        img.save(out_path, quality=95)
        saved_paths.append(out_path)
    print(f"[IO] Saved {len(saved_paths)} protected images to {output_dir}")
    return saved_paths


def save_evaluation_csv(metrics: Dict, output_path: str):
    """Save evaluation metrics dict to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    print(f"[IO] Saved evaluation CSV to {output_path}")


def save_pairwise_csv(matrix: "np.ndarray", labels: List[str], output_path: str):
    """Save pairwise similarity matrix to CSV."""
    import numpy as np
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(output_path)
    print(f"[IO] Saved pairwise matrix to {output_path}")


def create_output_zip(output_dir: str, zip_path: str):
    """Zip all outputs for download."""
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", output_dir)
    print(f"[IO] Created output ZIP: {zip_path}")
