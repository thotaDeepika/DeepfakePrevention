# ============================================================
# SafeUpload: Transferable Identity Cloaking for Public Images
# Colab Notebook — GPU Runtime Required
# ============================================================
# 
# USAGE:
#   1. Runtime > Change runtime type > GPU (T4 or better)
#   2. Run all cells in order
#   3. Upload your ZIP of 5-6 face images when prompted
#   4. Results saved to /content/safeupload_outputs/
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Install Dependencies
# ─────────────────────────────────────────────────────────────
INSTALL_CODE = '''
import subprocess, sys

packages = [
    "torch torchvision --index-url https://download.pytorch.org/whl/cu118",
    "facenet-pytorch==2.5.3",
    "open-clip-torch==2.24.0",
    "insightface>=0.7.3",
    "onnxruntime-gpu",
    "scikit-image",
    "scikit-learn",
    "umap-learn",
    "seaborn",
    "Pillow",
    "tqdm",
    "pandas",
    "scipy",
    "opencv-python-headless",
    "einops",
    "timm",
    "google-generativeai",
]

for pkg in packages:
    print(f"Installing {pkg.split()[0]}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + pkg.split())

print("\\n✅ All dependencies installed!")
'''

# ─────────────────────────────────────────────────────────────
# CELL 2: Clone / Mount Project
# ─────────────────────────────────────────────────────────────
SETUP_CODE = '''
import os, sys

# If running from Google Drive (recommended):
# from google.colab import drive
# drive.mount("/content/drive")
# PROJECT_DIR = "/content/drive/MyDrive/safeupload"

# Or clone directly:
# !git clone https://github.com/YOUR_USERNAME/safeupload.git /content/safeupload

PROJECT_DIR = "/content/safeupload"

if not os.path.exists(PROJECT_DIR):
    raise FileNotFoundError(
        f"Project not found at {PROJECT_DIR}. "
        "Please upload the safeupload/ folder to Colab or Google Drive."
    )

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import torch
print(f"✅ Project loaded from {PROJECT_DIR}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
'''

# ─────────────────────────────────────────────────────────────
# CELL 3: Upload Images
# ─────────────────────────────────────────────────────────────
UPLOAD_CODE = '''
from google.colab import files
import os

print("📁 Upload a ZIP file containing 5-6 face images of the SAME person.")
print("   Supported: .jpg, .jpeg, .png, .bmp")

uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded!")

zip_filename = list(uploaded.keys())[0]
ZIP_PATH = f"/content/{zip_filename}"

print(f"\\n✅ Uploaded: {ZIP_PATH}")
print(f"   Size: {os.path.getsize(ZIP_PATH) / 1024:.1f} KB")
'''

# ─────────────────────────────────────────────────────────────
# CELL 4: Run SafeUpload Pipeline
# ─────────────────────────────────────────────────────────────
RUN_CODE = '''
import sys
sys.path.insert(0, "/content/safeupload")

from pipeline import SafeUploadPipeline

# ── CONFIGURATION ──────────────────────────────────────────
OUTPUT_DIR = "/content/safeupload_outputs"

EPS        = 8 / 255    # Perturbation budget (8/255 ≈ 0.031)
STEPS      = 50         # Optimisation steps (50-100 recommended)
K          = 8          # MCA crops per step
EOT        = 3          # EOT repeats
IMG_SIZE   = 256        # Face alignment output size
USE_ARCFACE = False     # Set True if insightface is working correctly
PSEUDO_STRATEGY = "pca_project"  # or "orthogonal", "random_manifold"

# Optional: Gemini API key for vision evaluation
GEMINI_API_KEY = ""     # Set your key here or leave blank
RUN_GEMINI = bool(GEMINI_API_KEY)
# ───────────────────────────────────────────────────────────

pipeline = SafeUploadPipeline(
    output_dir=OUTPUT_DIR,
    eps=EPS,
    steps=STEPS,
    K=K,
    eot=EOT,
    img_size=IMG_SIZE,
    use_arcface=USE_ARCFACE,
    pseudo_strategy=PSEUDO_STRATEGY,
    gemini_api_key=GEMINI_API_KEY or None,
)

results = pipeline.run(
    zip_path=ZIP_PATH,
    run_gemini=RUN_GEMINI,
    verbose=True,
)

print("\\n✅ Pipeline complete!")
'''

# ─────────────────────────────────────────────────────────────
# CELL 5: Display Results
# ─────────────────────────────────────────────────────────────
DISPLAY_CODE = '''
from IPython.display import display, Image as IPImage, HTML
import os
import pandas as pd

OUTPUT_DIR = "/content/safeupload_outputs"

def show_image(path, caption="", width=700):
    if os.path.exists(path):
        display(HTML(f"<h3>{caption}</h3>"))
        display(IPImage(path, width=width))
    else:
        print(f"[Not found] {path}")

# Summary dashboard
show_image(f"{OUTPUT_DIR}/viz/summary_dashboard.png", "📊 Summary Dashboard")

# Before/after gallery
show_image(f"{OUTPUT_DIR}/viz/before_after_gallery.png", "🖼️ Before vs After Gallery")

# Similarity bar chart
show_image(f"{OUTPUT_DIR}/viz/similarity_bar_chart.png", "📉 Identity Similarity Reduction")

# Quality metrics
show_image(f"{OUTPUT_DIR}/viz/quality_metrics.png", "✨ Visual Quality Metrics")

# Per-model summaries
print("\\n📋 Per-Model Evaluation Summary:")
csv_path = f"{OUTPUT_DIR}/metrics/per_model_summary.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    display(df.style.background_gradient(subset=["reduction_pct"], cmap="Greens"))

# Summary metrics
print("\\n📊 Overall Metrics:")
summary_csv = f"{OUTPUT_DIR}/metrics/summary.csv"
if os.path.exists(summary_csv):
    display(pd.read_csv(summary_csv))
'''

# ─────────────────────────────────────────────────────────────
# CELL 6: Show Embedding Scatter Plots
# ─────────────────────────────────────────────────────────────
EMBEDDING_DISPLAY_CODE = '''
from IPython.display import display, Image as IPImage, HTML
import os
import glob

OUTPUT_DIR = "/content/safeupload_outputs"
viz_dir = f"{OUTPUT_DIR}/viz"

for method in ["pca", "tsne", "umap"]:
    paths = sorted(glob.glob(f"{viz_dir}/{method}_*.png"))
    if paths:
        display(HTML(f"<h3>🔵 {method.upper()} Embedding Visualization</h3>"))
        for path in paths:
            model_name = os.path.basename(path).replace(f"{method}_", "").replace(".png", "")
            display(HTML(f"<b>{model_name}</b>"))
            display(IPImage(path, width=550))
'''

# ─────────────────────────────────────────────────────────────
# CELL 7: Show Heatmaps
# ─────────────────────────────────────────────────────────────
HEATMAP_DISPLAY_CODE = '''
from IPython.display import display, Image as IPImage, HTML
import os
import glob

OUTPUT_DIR = "/content/safeupload_outputs"
viz_dir = f"{OUTPUT_DIR}/viz"

display(HTML("<h3>🌡️ Identity Similarity Heatmaps (Original vs Protected)</h3>"))
paths = sorted(glob.glob(f"{viz_dir}/heatmap_comparison_*.png"))
for path in paths:
    model = os.path.basename(path).replace("heatmap_comparison_", "").replace(".png", "")
    display(HTML(f"<b>{model}</b>"))
    display(IPImage(path, width=700))
'''

# ─────────────────────────────────────────────────────────────
# CELL 8: Download All Results
# ─────────────────────────────────────────────────────────────
DOWNLOAD_CODE = '''
from google.colab import files
import os

OUTPUT_DIR = "/content/safeupload_outputs"
zip_path = f"{OUTPUT_DIR}/safeupload_results.zip"

if os.path.exists(zip_path):
    print(f"📦 Downloading {zip_path}...")
    files.download(zip_path)
else:
    # Re-create zip
    import shutil
    zip_base = "/content/safeupload_results"
    shutil.make_archive(zip_base, "zip", OUTPUT_DIR)
    files.download(f"{zip_base}.zip")
    
print("✅ Download started!")
'''

# ─────────────────────────────────────────────────────────────
# CELL 9: Gemini Evaluation Results (if API key provided)
# ─────────────────────────────────────────────────────────────
GEMINI_DISPLAY_CODE = '''
import json, os
from IPython.display import display, HTML, Image as IPImage

OUTPUT_DIR = "/content/safeupload_outputs"
results_path = f"{OUTPUT_DIR}/gemini_eval/gemini_results.json"
orig_grid = f"{OUTPUT_DIR}/gemini_eval/original_grid.png"
prot_grid = f"{OUTPUT_DIR}/gemini_eval/protected_grid.png"

if not os.path.exists(results_path):
    print("⚠️ Gemini evaluation not run. Set GEMINI_API_KEY and RUN_GEMINI=True in Cell 4.")
else:
    display(HTML("<h3>🤖 Gemini Vision Ambiguity Evaluation</h3>"))
    
    with open(results_path) as f:
        results = json.load(f)
    
    display(HTML("<b>Image Grids:</b>"))
    if os.path.exists(orig_grid):
        display(HTML("<i>Original Images:</i>"))
        display(IPImage(orig_grid, width=500))
    if os.path.exists(prot_grid):
        display(HTML("<i>Protected Images:</i>"))
        display(IPImage(prot_grid, width=500))
    
    display(HTML("<b>Gemini Responses:</b>"))
    for key, data in results.items():
        prompt = data["prompt"]
        orig_r = data["original"]
        prot_r = data["protected"]
        
        html = f"""
        <div style="border:1px solid #ddd; padding:10px; margin:10px 0; border-radius:8px">
            <b>Prompt:</b> {prompt}<br><br>
            <table style="width:100%">
            <tr>
                <td style="width:50%; background:#e3f2fd; padding:8px; border-radius:4px">
                    <b>🔵 Original</b><br>
                    Confidence: {orig_r.get("confidence", "N/A")}%<br>
                    Same Person: {orig_r.get("same_person", "N/A")}<br>
                    <i>{orig_r.get("response", "")[:200]}...</i>
                </td>
                <td style="width:50%; background:#fce4ec; padding:8px; border-radius:4px">
                    <b>🔴 Protected</b><br>
                    Confidence: {prot_r.get("confidence", "N/A")}%<br>
                    Same Person: {prot_r.get("same_person", "N/A")}<br>
                    <i>{prot_r.get("response", "")[:200]}...</i>
                </td>
            </tr>
            </table>
        </div>
        """
        display(HTML(html))
'''

# ─────────────────────────────────────────────────────────────
# Actual executable notebook content
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SafeUpload Colab Notebook Script")
    print("="*50)
    print("This file contains the code for each Colab cell.")
    print("Copy each CELL section into a separate Colab cell.")
    print("\nCell order:")
    print("  1. Install Dependencies")
    print("  2. Setup Project")
    print("  3. Upload Images (ZIP)")
    print("  4. Run Pipeline")
    print("  5. Display Results")
    print("  6. Embedding Scatter Plots")
    print("  7. Similarity Heatmaps")
    print("  8. Download Results")
    print("  9. Gemini Evaluation (optional)")
