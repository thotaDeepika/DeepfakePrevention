"""
app/app.py  —  SafeUpload Web UI Backend (v2)

Routes:
  GET  /                → serve index.html
  POST /protect         → upload image → detect face → cloak crop →
                          blend back into full image → evaluate → return JSON
  GET  /download/<id>   → download protected full image
  POST /gemini_eval     → optional Gemini 2.5 Flash evaluation
"""

import os, sys, uuid, time, traceback, io, base64, json
from pathlib import Path

# Force UTF-8 output on Windows to avoid charmap encode errors
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── dirs ──────────────────────────────────────────────────────────────
UPLOAD_DIR   = PROJECT_ROOT / "app" / "uploads"
OUTPUT_DIR   = PROJECT_ROOT / "app" / "outputs"
TEMPLATE_DIR = PROJECT_ROOT / "app" / "templates"
for d in [UPLOAD_DIR, OUTPUT_DIR, TEMPLATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[App] Device: {DEVICE}")

# ── attack config — Original Uncompromised Settings (Cloud GPU Ready) ─
ATTACK_CONFIG = {
    "eps": 8 / 255, "steps": 50, "K": 8, "eot": 3,
    "alpha": 1.5 / 255, "beta1": 0.9, "beta2": 0.99,
    "lambda1": 0.7, "lambda2": 0.1, "lambda3": 0.1,
    "model_weights": {"facenet": 0.3, "arcface": 0.3,
                      "clip_b16": 0.2, "clip_l14": 0.2},
}

# ── lazy model cache ───────────────────────────────────────────────────
_models_dict = None

def get_models():
    global _models_dict
    if _models_dict is None:
        from models.facenet  import FaceNetModel
        from models.arcface  import ArcFaceModel
        from models.clip_b16 import CLIPb16Model
        from models.clip_l14 import CLIPl14Model
        print("[App] Loading embedding models…")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _models_dict = {
            "facenet":  FaceNetModel(device=DEVICE),
            "arcface":  ArcFaceModel(device=DEVICE),
            "clip_b16": CLIPb16Model(device=DEVICE),
            "clip_l14": CLIPl14Model(device=DEVICE),
        }
        print("[App] Models ready.")
    return _models_dict


def _pil_to_b64(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _build_manifold_from_augmentations(face_np: np.ndarray, models_dict: dict):
    """
    Build identity manifold from N augmented views of a single face crop.
    Uses 5 views: original + hflip + brightness variants + slight crop.
    This gives meaningful pairwise similarities for single-image evaluation.
    """
    from utils.identity_manifold import IdentityManifold
    import torchvision.transforms.functional as TF

    base = Image.fromarray(face_np.astype(np.uint8))
    H, W = face_np.shape[:2]

    views = [
        base,
        TF.hflip(base),
        TF.adjust_brightness(base, 1.08),
        TF.adjust_brightness(base, 0.93),
        TF.adjust_contrast(base, 1.06),
    ]
    manifold = IdentityManifold(models_dict, device=DEVICE)
    manifold.build([np.array(v) for v in views])
    return manifold


# ── routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(TEMPLATE_DIR), "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE})


@app.route("/protect", methods=["POST"])
def protect():
    t0 = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Max 1024px for highest uncompromised quality (Requires ~16GB+ VRAM on Cloud GPU)
        full_img = Image.open(file.stream).convert("RGB")
        W0, H0 = full_img.size
        MAX_DIM = 1024
        if max(W0, H0) > MAX_DIM:
            scale = MAX_DIM / max(W0, H0)
            full_img = full_img.resize(
                (int(W0 * scale), int(H0 * scale)), Image.LANCZOS)

        # ── 2. Load models (flush GPU cache first) ───────────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        models_dict = get_models()

        # ── 3. Face detection + expand bbox ─────────────────────────
        from utils.full_image_blender import (
            detect_face_bbox, crop_face_region, blend_protected_back,
        )
        bbox = detect_face_bbox(full_img, device=DEVICE, expand_ratio=0.12)
        if bbox is None:
            W, H = full_img.size
            m = min(W, H)
            half = int(m * 0.30)
            cx, cy = W // 2, H // 2
            bbox = (cx - half, cy - half, cx + half, cy + half)

        face_crop_np, orig_crop_size = crop_face_region(
            full_img, bbox, attack_size=256)

        # ── 4. Identity manifold from augmented views ────────────────
        manifold = _build_manifold_from_augmentations(face_crop_np, models_dict)

        from utils.pseudo_target import PseudoTargetGenerator
        ptg = PseudoTargetGenerator(manifold, device=DEVICE)
        pseudo_targets = ptg.generate(strategy="orthogonal_pca")

        # ── 5. Run identity cloaking attack (crop only) ──────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from attacks.safeupload_attack import SafeUploadAttack
        attacker = SafeUploadAttack(
            models_dict, manifold, pseudo_targets, ATTACK_CONFIG, DEVICE)
        protected_crop_np, delta = attacker.attack(face_crop_np)

        # ── 6. Blend protected crop back into full image ─────────────
        protected_full_img = blend_protected_back(
            full_img, protected_crop_np, bbox, orig_crop_size,
            blend_mode="poisson",
        )

        # ── 7. Evaluate identity reduction ───────────────────────────
        #   Pass both the 5 original augmented crops AND 5 protected
        #   augmented crops so pairwise similarity is well-defined.
        from evaluation.identity_evaluator import IdentityEvaluator
        import torchvision.transforms.functional as TF

        prot_pil = Image.fromarray(protected_crop_np.astype(np.uint8))
        prot_views = [
            np.array(prot_pil),
            np.array(TF.hflip(prot_pil)),
            np.array(TF.adjust_brightness(prot_pil, 1.08)),
            np.array(TF.adjust_brightness(prot_pil, 0.93)),
            np.array(TF.adjust_contrast(prot_pil, 1.06)),
        ]

        orig_views = [np.array(v) for v in [
            Image.fromarray(face_crop_np.astype(np.uint8)),
            TF.hflip(Image.fromarray(face_crop_np.astype(np.uint8))),
            TF.adjust_brightness(Image.fromarray(face_crop_np.astype(np.uint8)), 1.08),
            TF.adjust_brightness(Image.fromarray(face_crop_np.astype(np.uint8)), 0.93),
            TF.adjust_contrast(Image.fromarray(face_crop_np.astype(np.uint8)), 1.06),
        ]]

        evaluator = IdentityEvaluator(models_dict, device=DEVICE)
        eval_results = evaluator.evaluate(
            original_faces=orig_views,
            protected_faces=prot_views,
        )

        metrics = {}
        for m_name, res in eval_results.items():
            o = float(res["original_mean_similarity"])
            p = float(res["protected_mean_similarity"])
            metrics[m_name] = {
                "original_sim":  round(o, 4),
                "protected_sim": round(p, 4),
                "reduction_pct": round((o - p) / max(o, 1e-8) * 100, 1),
                "ssim":          round(float(res.get("ssim", 0)), 4),
                "psnr":          round(float(res.get("psnr", 0)), 2),
            }

        # ── 8. Save outputs ──────────────────────────────────────────
        job_id = str(uuid.uuid4())[:8]
        orig_path = OUTPUT_DIR / f"{job_id}_original.jpg"
        prot_path = OUTPUT_DIR / f"{job_id}_protected.jpg"
        full_img.save(str(orig_path), quality=92)
        protected_full_img.save(str(prot_path), quality=92)

        elapsed = round(time.time() - t0, 1)
        print(f"[App] Job {job_id} done in {elapsed}s — "
              f"metrics: {[(k, v['reduction_pct']) for k,v in metrics.items()]}")

        return jsonify({
            "job_id":        job_id,
            "original_b64":  _pil_to_b64(full_img),
            "protected_b64": _pil_to_b64(protected_full_img),
            "metrics":       metrics,
            "bbox":          list(bbox),
            "elapsed_sec":   elapsed,
            "device":        DEVICE,
            "image_size":    list(full_img.size),
        })

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        traceback.print_exc()
        return jsonify({
            "error": "GPU out of memory. Try a smaller image (under 640×640). "
                     f"Details: {str(e)}"
        }), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download/<job_id>")
def download(job_id):
    if not job_id.replace("-", "").replace("_", "").isalnum():
        return jsonify({"error": "Invalid job_id"}), 400
    prot_path = OUTPUT_DIR / f"{job_id}_protected.jpg"
    if not prot_path.exists():
        return jsonify({"error": "Job not found"}), 404
    return send_file(
        str(prot_path), mimetype="image/jpeg",
        as_attachment=True,
        download_name=f"safeupload_protected_{job_id}.jpg",
    )


@app.route("/gemini_eval", methods=["POST"])
def gemini_eval():
    data    = request.get_json(force=True)
    job_id  = data.get("job_id", "")
    api_key = data.get("api_key", "")
    if not api_key:
        return jsonify({"error": "No API key"}), 400
    orig_path = OUTPUT_DIR / f"{job_id}_original.jpg"
    prot_path = OUTPUT_DIR / f"{job_id}_protected.jpg"
    if not orig_path.exists():
        return jsonify({"error": "Job not found"}), 404
    try:
        from evaluation.vision_eval import GeminiEvaluator
        ev = GeminiEvaluator(api_key=api_key)
        res = ev.run_full_evaluation(
            original_images=[Image.open(str(orig_path)).convert("RGB")],
            protected_images=[Image.open(str(prot_path)).convert("RGB")],
            output_dir=str(OUTPUT_DIR / f"{job_id}_gemini"),
        )
        return jsonify({"gemini_results": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[App] http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
