"""
install_colab.py
Run this cell in Colab BEFORE anything else to install all dependencies.
Usage: exec(open('install_colab.py').read())
"""

import subprocess
import sys


def pip_install(pkg, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", pkg]
    if extra_args:
        cmd.extend(extra_args)
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  ⚠  Warning installing {pkg}: {ret.stderr[-300:]}")
    else:
        print(f"  ✓  {pkg}")


print("=" * 60)
print("SafeUpload Dependency Installer")
print("=" * 60)

# ── 1. Face detection & recognition ─────────────────────────────
print("\n[1/5] Face models...")
pip_install("facenet-pytorch==2.5.3")

# ── 2. Open CLIP ─────────────────────────────────────────────────
print("\n[2/5] Open CLIP...")
pip_install("open-clip-torch==2.24.0")

# ── 3. ArcFace / insightface ─────────────────────────────────────
print("\n[3/5] InsightFace (ArcFace)...")
pip_install("insightface==0.7.3")
pip_install("onnxruntime-gpu==1.16.3")

# ── 4. Computer vision utilities ─────────────────────────────────
print("\n[4/5] CV utilities...")
pip_install("opencv-python-headless==4.8.1.78")
pip_install("scikit-image==0.22.0")
pip_install("scikit-learn==1.3.2")
pip_install("umap-learn==0.5.5")

# ── 5. Visualization & data ──────────────────────────────────────
print("\n[5/5] Visualization & data...")
pip_install("seaborn==0.13.0")
pip_install("plotly==5.18.0")
pip_install("pandas==2.1.4")
pip_install("tqdm==4.66.1")
pip_install("google-generativeai==0.8.3")
pip_install("einops==0.7.0")

print("\n" + "=" * 60)
print("✅ All SafeUpload dependencies installed!")
print("=" * 60)

# Verify key imports
print("\nVerifying critical imports...")
try:
    import torch
    print(f"  torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ torch: {e}")

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("  facenet_pytorch: OK")
except ImportError as e:
    print(f"  ✗ facenet_pytorch: {e}")

try:
    import open_clip
    print(f"  open_clip: OK")
except ImportError as e:
    print(f"  ✗ open_clip: {e}")

try:
    import insightface
    print(f"  insightface: OK")
except ImportError as e:
    print(f"  ✗ insightface: {e}")

try:
    import sklearn, umap, seaborn, plotly
    print("  sklearn / umap / seaborn / plotly: OK")
except ImportError as e:
    print(f"  ✗ {e}")
