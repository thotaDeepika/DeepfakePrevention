import os
import sys
from pathlib import Path
import traceback
import time
import uuid
from PIL import Image
import numpy as np
import torch

from app.celery_worker import celery_app

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = PROJECT_ROOT / "app" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_CONFIG = {
    "eps": 8 / 255, "steps": 50, "K": 8, "eot": 3,
    "alpha": 1.5 / 255, "beta1": 0.9, "beta2": 0.99,
    "lambda1": 0.7, "lambda2": 0.1, "lambda3": 0.1,
    "model_weights": {"facenet": 0.3, "arcface": 0.3,
                      "clip_b16": 0.2, "clip_l14": 0.2},
}

_models_dict = None
def get_models():
    global _models_dict
    if _models_dict is None:
        from models.facenet import FaceNetModel
        from models.arcface import ArcFaceModel
        from models.clip_b16 import CLIPb16Model
        from models.clip_l14 import CLIPl14Model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _models_dict = {
            "facenet": FaceNetModel(device=DEVICE),
            "arcface": ArcFaceModel(device=DEVICE),
            "clip_b16": CLIPb16Model(device=DEVICE),
            "clip_l14": CLIPl14Model(device=DEVICE),
        }
    return _models_dict

@celery_app.task(bind=True)
def run_protection_task(self, job_id, image_path):
    t0 = time.time()
    try:
        self.update_state(state='PROGRESS', meta={'message': 'Loading image...'})
        full_img = Image.open(image_path).convert("RGB")
        W0, H0 = full_img.size
        MAX_DIM = 1024
        if max(W0, H0) > MAX_DIM:
            scale = MAX_DIM / max(W0, H0)
            full_img = full_img.resize((int(W0 * scale), int(H0 * scale)), Image.LANCZOS)

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        self.update_state(state='PROGRESS', meta={'message': 'Loading models...'})
        models_dict = get_models()

        self.update_state(state='PROGRESS', meta={'message': 'Detecting face...'})
        from utils.full_image_blender import detect_face_bbox, crop_face_region, blend_protected_back
        bbox = detect_face_bbox(full_img, device=DEVICE, expand_ratio=0.12)
        if bbox is None:
            W, H = full_img.size
            m = min(W, H)
            half = int(m * 0.30)
            cx, cy = W // 2, H // 2
            bbox = (cx - half, cy - half, cx + half, cy + half)

        face_crop_np, orig_crop_size = crop_face_region(full_img, bbox, attack_size=256)

        self.update_state(state='PROGRESS', meta={'message': 'Running MCA attack (this takes time)...'})
        
        # Build manifold
        import torchvision.transforms.functional as TF
        from utils.identity_manifold import IdentityManifold
        base = Image.fromarray(face_crop_np.astype(np.uint8))
        views = [base, TF.hflip(base), TF.adjust_brightness(base, 1.08), TF.adjust_brightness(base, 0.93), TF.adjust_contrast(base, 1.06)]
        manifold = IdentityManifold(models_dict, device=DEVICE)
        manifold.build([np.array(v) for v in views])

        from utils.pseudo_target import PseudoTargetGenerator
        ptg = PseudoTargetGenerator(manifold, device=DEVICE)
        pseudo_targets = ptg.generate(strategy="orthogonal_pca")

        from attacks.safeupload_attack import SafeUploadAttack
        attacker = SafeUploadAttack(models_dict, manifold, pseudo_targets, ATTACK_CONFIG, DEVICE)
        protected_crop_np, delta = attacker.attack(face_crop_np)

        self.update_state(state='PROGRESS', meta={'message': 'Blending protected image...'})
        protected_full_img = blend_protected_back(full_img, protected_crop_np, bbox, orig_crop_size, blend_mode="poisson")

        self.update_state(state='PROGRESS', meta={'message': 'Evaluating metrics...'})
        from evaluation.identity_evaluator import IdentityEvaluator
        prot_pil = Image.fromarray(protected_crop_np.astype(np.uint8))
        prot_views = [np.array(prot_pil), np.array(TF.hflip(prot_pil)), np.array(TF.adjust_brightness(prot_pil, 1.08)), np.array(TF.adjust_brightness(prot_pil, 0.93)), np.array(TF.adjust_contrast(prot_pil, 1.06))]
        orig_views = [np.array(v) for v in views]
        evaluator = IdentityEvaluator(models_dict, device=DEVICE)
        eval_results = evaluator.evaluate(original_faces=orig_views, protected_faces=prot_views)

        metrics = {}
        for m_name, res in eval_results.items():
            o = float(res["original_mean_similarity"])
            p = float(res["protected_mean_similarity"])
            metrics[m_name] = {
                "original_sim": round(o, 4),
                "protected_sim": round(p, 4),
                "reduction_pct": round((o - p) / max(o, 1e-8) * 100, 1),
                "ssim": round(float(res.get("ssim", 0)), 4),
                "psnr": round(float(res.get("psnr", 0)), 2),
            }

        orig_path = OUTPUT_DIR / f"{job_id}_original.jpg"
        prot_path = OUTPUT_DIR / f"{job_id}_protected.jpg"
        full_img.save(str(orig_path), quality=92)
        protected_full_img.save(str(prot_path), quality=92)
        
        import io, base64
        def pil_to_b64(img):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode()

        elapsed = round(time.time() - t0, 1)
        return {
            "job_id": job_id,
            "original_b64": pil_to_b64(full_img),
            "protected_b64": pil_to_b64(protected_full_img),
            "metrics": metrics,
            "bbox": list(bbox),
            "elapsed_sec": elapsed,
            "device": DEVICE
        }

    except Exception as e:
        traceback.print_exc()
        raise self.retry(exc=e, countdown=5, max_retries=1)
