"""
pipeline.py
SafeUpload full pipeline orchestrator.

Usage:
    from pipeline import SafeUploadPipeline
    pipe = SafeUploadPipeline()
    results = pipe.run(zip_path="faces.zip")
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from attacks.mca_attack import MCAIdentityCloakAttack
from evaluation.identity_evaluator import IdentityConsistencyEvaluator
from evaluation.vision_eval import GeminiAmbiguityEvaluator
from visualization.embedding_viz import EmbeddingVisualizer
from visualization.heatmap_viz import SimilarityHeatmapVisualizer
from visualization.comparison_viz import ComparisonVisualizer
from utils.face_utils import detect_and_align_face, pil_to_tensor, tensor_to_pil
from utils.io_utils import (
    load_zip_images, load_images_from_folder,
    save_protected_images, save_evaluation_csv,
    save_pairwise_csv, create_output_zip
)


class SafeUploadPipeline:
    """
    End-to-end SafeUpload pipeline.

    Args:
        output_dir:     where to save all outputs
        eps:            L-inf perturbation budget (8/255 recommended)
        steps:          optimisation steps (50 recommended)
        K:              MCA crops per step
        eot:            expectation over transformations
        img_size:       face alignment output size
        use_arcface:    include ArcFace in ensemble
        gemini_api_key: for vision evaluation (optional)
        device:         'cuda' or 'cpu'
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        eps: float = 8 / 255,
        steps: int = 50,
        K: int = 8,
        eot: int = 3,
        img_size: int = 256,
        use_arcface: bool = False,
        pseudo_strategy: str = "pca_project",
        gemini_api_key: str = None,
        device: str = None,
    ):
        self.output_dir = output_dir
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Pipeline] Device: {self.device}")
        print(f"[Pipeline] Output dir: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/protected", exist_ok=True)
        os.makedirs(f"{output_dir}/viz", exist_ok=True)
        os.makedirs(f"{output_dir}/metrics", exist_ok=True)
        os.makedirs(f"{output_dir}/gemini_eval", exist_ok=True)

        # Core attack
        self.attack = MCAIdentityCloakAttack(
            eps=eps,
            steps=steps,
            K=K,
            eot=eot,
            use_arcface=use_arcface,
            pseudo_strategy=pseudo_strategy,
            device=self.device,
        )

        # Evaluation
        model_dict = {
            "FaceNet": self.attack.facenet,
            "CLIP-B16": self.attack.clip_b16,
            "CLIP-L14": self.attack.clip_l14,
        }
        if use_arcface and self.attack.arcface:
            model_dict["ArcFace"] = self.attack.arcface

        self.evaluator = IdentityConsistencyEvaluator(
            models=model_dict,
            device=self.device,
        )

        self.gemini_eval = GeminiAmbiguityEvaluator(api_key=gemini_api_key)

        # Visualization
        viz_dir = f"{output_dir}/viz"
        self.emb_viz = EmbeddingVisualizer(output_dir=viz_dir)
        self.heatmap_viz = SimilarityHeatmapVisualizer(output_dir=viz_dir)
        self.comp_viz = ComparisonVisualizer(output_dir=viz_dir)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(
        self,
        zip_path: str = None,
        image_folder: str = None,
        images: List[Image.Image] = None,
        run_gemini: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Full pipeline run.

        Args:
            zip_path:      path to ZIP of face images
            image_folder:  alternative: folder of images
            images:        alternative: list of PIL images
            run_gemini:    whether to run Gemini evaluation
            verbose:       print progress

        Returns:
            results dict with all metrics and paths
        """
        # 1. Load images
        raw_images = self._load_images(zip_path, image_folder, images)
        if len(raw_images) < 2:
            raise ValueError(f"Need at least 2 images, got {len(raw_images)}")
        print(f"[Pipeline] Loaded {len(raw_images)} images.")

        # 2. Face detection + alignment
        print("[Pipeline] Step 2: Face detection & alignment...")
        aligned_images = self._align_faces(raw_images, verbose=verbose)
        if len(aligned_images) < 2:
            raise ValueError("Face detection failed for most images.")
        print(f"[Pipeline] {len(aligned_images)} faces aligned.")

        image_labels = [f"Img{i+1}" for i in range(len(aligned_images))]

        # 3. Convert to tensors
        orig_tensors = torch.cat(
            [pil_to_tensor(img, self.device) for img in aligned_images], dim=0
        )  # [N, 3, H, W]

        # 4. Build identity manifold
        print("[Pipeline] Step 3-4: Building identity manifold...")
        identity_center, all_embeddings = self.attack.build_identity_manifold(orig_tensors)

        # 5. MCA optimization + protection
        print("[Pipeline] Step 5-8: Applying MCA identity cloaking...")
        cloak_results = self.attack.cloak_batch(
            aligned_images, identity_center, all_embeddings, verbose=verbose
        )
        protected_images = [r[0] for r in cloak_results]
        per_image_metrics = [r[1] for r in cloak_results]

        # 6. Save protected images
        named_protected = list(zip(
            [f"img_{i+1}.png" for i in range(len(protected_images))],
            protected_images
        ))
        save_protected_images(named_protected, f"{self.output_dir}/protected")

        # 7. Multi-image identity consistency evaluation
        print("[Pipeline] Step 9-10: Identity consistency evaluation...")
        prot_tensors = torch.cat(
            [pil_to_tensor(img, self.device) for img in protected_images], dim=0
        )
        eval_results = self.evaluator.evaluate(orig_tensors, prot_tensors)

        # 8. Visualization
        print("[Pipeline] Step 11: Generating visualizations...")
        self._generate_visualizations(
            eval_results, aligned_images, protected_images,
            per_image_metrics, image_labels, all_embeddings
        )

        # 9. Save metrics CSV
        print("[Pipeline] Saving metrics...")
        overall = eval_results["overall"]
        avg_ssim = np.mean([m["ssim"] for m in per_image_metrics])
        avg_psnr = np.mean([m["psnr"] for m in per_image_metrics])
        summary_metrics = {
            **overall,
            "avg_ssim": round(float(avg_ssim), 4),
            "avg_psnr": round(float(avg_psnr), 2),
            "n_images": len(aligned_images),
            "eps": self.attack.eps,
            "steps": self.attack.steps,
            "K": self.attack.K,
        }
        save_evaluation_csv(summary_metrics, f"{self.output_dir}/metrics/summary.csv")
        eval_results["summary"].to_csv(
            f"{self.output_dir}/metrics/per_model_summary.csv", index=False
        )

        # 10. Gemini evaluation (optional)
        gemini_results = None
        if run_gemini:
            print("[Pipeline] Step 12: Gemini ambiguity evaluation...")
            try:
                gemini_results = self.gemini_eval.run_full_evaluation(
                    aligned_images, protected_images,
                    output_dir=f"{self.output_dir}/gemini_eval"
                )
            except Exception as e:
                print(f"[Pipeline] Gemini eval error: {e}")

        # 11. Create output ZIP
        zip_out = f"{self.output_dir}/safeupload_results.zip"
        create_output_zip(self.output_dir, zip_out)

        print(f"\n{'='*60}")
        print("SafeUpload Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Original identity similarity:  {overall['overall_orig_sim']:.4f}")
        print(f"Protected identity similarity: {overall['overall_prot_sim']:.4f}")
        print(f"Reduction:                     {overall['overall_reduction_pct']:.1f}%")
        print(f"Avg SSIM:  {avg_ssim:.4f}  (target: >0.85)")
        print(f"Avg PSNR:  {avg_psnr:.2f} dB  (target: >30 dB)")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_dir}/")

        return {
            "aligned_images": aligned_images,
            "protected_images": protected_images,
            "per_image_metrics": per_image_metrics,
            "eval_results": eval_results,
            "summary_metrics": summary_metrics,
            "gemini_results": gemini_results,
            "output_dir": self.output_dir,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _load_images(self, zip_path, image_folder, images):
        if images is not None:
            return images
        if zip_path:
            named = load_zip_images(zip_path)
            return [img for _, img in named]
        if image_folder:
            named = load_images_from_folder(image_folder)
            return [img for _, img in named]
        raise ValueError("Provide zip_path, image_folder, or images list.")

    def _align_faces(self, images: list, verbose: bool = True) -> list:
        aligned = []
        for i, img in enumerate(images):
            face = detect_and_align_face(
                img, output_size=self.img_size,
                device=self.device, fallback_center_crop=True
            )
            if face is not None:
                aligned.append(face)
                if verbose:
                    print(f"  [{i+1}/{len(images)}] Face aligned: {face.size}")
            else:
                print(f"  [{i+1}/{len(images)}] WARNING: No face detected, skipping.")
        return aligned

    def _generate_visualizations(
        self,
        eval_results: dict,
        orig_images: list,
        prot_images: list,
        per_image_metrics: list,
        image_labels: list,
        all_embeddings: dict,
    ):
        # Compute protected embeddings for viz
        prot_tensors = torch.cat(
            [pil_to_tensor(img, self.device) for img in prot_images], dim=0
        )
        prot_embs_dict = {}
        for name, model in self.evaluator.models.items():
            embs = []
            for i in range(prot_tensors.shape[0]):
                e = model.extract_embedding(prot_tensors[i:i+1])
                embs.append(e)
            prot_embs_dict[name] = torch.cat(embs, dim=0)

        # Embedding scatter plots
        for name in all_embeddings:
            orig_np = all_embeddings[name].cpu().numpy()
            prot_np = prot_embs_dict[name].cpu().numpy()
            self.emb_viz.plot_pca(orig_np, prot_np, model_name=name)
            if len(orig_np) >= 4:
                self.emb_viz.plot_tsne(orig_np, prot_np, model_name=name)
            self.emb_viz.plot_umap(orig_np, prot_np, model_name=name)

        # Heatmaps
        self.heatmap_viz.plot_all_models(
            eval_results["original_matrices"],
            eval_results["protected_matrices"],
            image_labels,
        )
        self.heatmap_viz.plot_similarity_bar(eval_results["summary"])

        # Before/after gallery
        self.comp_viz.plot_before_after(
            orig_images[:6], prot_images[:6], per_image_metrics[:6]
        )
        self.comp_viz.plot_perturbation_stats(per_image_metrics)

        # Summary dashboard
        avg_ssim = float(np.mean([m["ssim"] for m in per_image_metrics]))
        avg_psnr = float(np.mean([m["psnr"] for m in per_image_metrics]))
        self.comp_viz.plot_summary_dashboard(
            eval_results["overall"],
            eval_results["summary"],
            avg_ssim,
            avg_psnr,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SafeUpload Identity Cloaking Pipeline")
    parser.add_argument("--zip", type=str, help="Path to ZIP of face images")
    parser.add_argument("--folder", type=str, help="Path to folder of face images")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--eps", type=float, default=8/255, help="Perturbation budget")
    parser.add_argument("--steps", type=int, default=50, help="Optimisation steps")
    parser.add_argument("--K", type=int, default=8, help="MCA crops per step")
    parser.add_argument("--eot", type=int, default=3, help="EOT repeats")
    parser.add_argument("--gemini_key", type=str, default=None)
    parser.add_argument("--run_gemini", action="store_true")
    parser.add_argument("--use_arcface", action="store_true")
    args = parser.parse_args()

    pipe = SafeUploadPipeline(
        output_dir=args.output,
        eps=args.eps,
        steps=args.steps,
        K=args.K,
        eot=args.eot,
        use_arcface=args.use_arcface,
        gemini_api_key=args.gemini_key,
    )
    pipe.run(
        zip_path=args.zip,
        image_folder=args.folder,
        run_gemini=args.run_gemini,
    )
