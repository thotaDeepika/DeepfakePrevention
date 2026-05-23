"""
evaluation/vision_eval.py

Gemini multimodal ambiguity evaluation for SafeUpload.
Prepares image grids and evaluation prompts.
Queries Gemini API with original vs protected image grids.
"""

import os
import io
import base64
import json
from typing import List, Tuple
from PIL import Image
import numpy as np


# Evaluation prompts for identity consistency
EVAL_PROMPTS = [
    "Do all of these images depict the same person? Please answer Yes or No, then explain your confidence level.",
    "How confident are you (0-100%) that all images show the same individual? Describe any similarities or differences you notice.",
    "Rate the identity consistency across these images from 1-10, where 10 means you are certain it's the same person.",
]


class GeminiAmbiguityEvaluator:
    """
    Evaluates identity ambiguity using Google Gemini Vision.
    Sends image grids and measures confidence/consistency.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel("gemini-2.5-flash")
            except Exception as e:
                print(f"[GeminiEval] Could not initialise Gemini: {e}")
                self._client = "unavailable"
        return self._client

    def create_image_grid(
        self,
        images: List[Image.Image],
        grid_cols: int = 3,
        img_size: int = 224,
        border: int = 4,
    ) -> Image.Image:
        """Create a grid image from a list of PIL images."""
        n = len(images)
        cols = min(grid_cols, n)
        rows = (n + cols - 1) // cols

        grid_w = cols * (img_size + border) + border
        grid_h = rows * (img_size + border) + border
        grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))

        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            img_resized = img.resize((img_size, img_size), Image.LANCZOS)
            x = border + c * (img_size + border)
            y = border + r * (img_size + border)
            grid.paste(img_resized, (x, y))

        return grid

    def _pil_to_base64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def evaluate_grid(
        self,
        grid_image: Image.Image,
        prompt: str = None,
    ) -> dict:
        """
        Query Gemini with an image grid.
        Returns dict with response text and parsed metrics.
        """
        client = self._get_client()
        if client == "unavailable" or not self.api_key:
            return {
                "response": "Gemini API not available (no API key).",
                "confidence": None,
                "same_person": None,
            }

        prompt = prompt or EVAL_PROMPTS[1]

        try:
            import google.generativeai as genai
            buf = io.BytesIO()
            grid_image.save(buf, format="PNG")
            buf.seek(0)
            image_part = {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.read()).decode(),
            }
            response = client.generate_content([prompt, image_part])
            text = response.text if hasattr(response, "text") else str(response)
            return self._parse_response(text)
        except Exception as e:
            return {
                "response": f"Error: {e}",
                "confidence": None,
                "same_person": None,
            }

    def run_full_evaluation(
        self,
        original_images: List[Image.Image],
        protected_images: List[Image.Image],
        output_dir: str = "outputs/gemini_eval",
    ) -> dict:
        """
        Run full Gemini evaluation on original and protected grids.
        Saves grid images and returns comparison results.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Create grids
        orig_grid = self.create_image_grid(original_images)
        prot_grid = self.create_image_grid(protected_images)

        # Save grids
        orig_grid.save(os.path.join(output_dir, "original_grid.png"))
        prot_grid.save(os.path.join(output_dir, "protected_grid.png"))

        results = {}

        for prompt_idx, prompt in enumerate(EVAL_PROMPTS):
            print(f"\n[GeminiEval] Prompt {prompt_idx+1}: {prompt[:60]}...")

            orig_result = self.evaluate_grid(orig_grid, prompt)
            prot_result = self.evaluate_grid(prot_grid, prompt)

            print(f"  Original: {orig_result['response'][:120]}")
            print(f"  Protected: {prot_result['response'][:120]}")

            results[f"prompt_{prompt_idx+1}"] = {
                "prompt": prompt,
                "original": orig_result,
                "protected": prot_result,
            }

        # Save results JSON
        results_path = os.path.join(output_dir, "gemini_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[GeminiEval] Results saved to {results_path}")
        return results

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse Gemini response to extract key signals."""
        text_lower = text.lower()

        # Try to extract confidence
        confidence = None
        import re
        pct_match = re.search(r"(\d{1,3})\s*%", text)
        if pct_match:
            confidence = int(pct_match.group(1))

        score_match = re.search(r"\b([1-9]|10)\s*/\s*10\b", text)
        if score_match and confidence is None:
            confidence = int(score_match.group(1)) * 10

        # Try to determine yes/no
        same_person = None
        if "yes" in text_lower[:50]:
            same_person = True
        elif "no" in text_lower[:50]:
            same_person = False

        return {
            "response": text,
            "confidence": confidence,
            "same_person": same_person,
        }


class GeminiEvaluator(GeminiAmbiguityEvaluator):
    """Notebook-friendly alias for GeminiAmbiguityEvaluator."""

    def evaluate(self, original_faces, protected_faces, output_dir="outputs"):
        import numpy as np
        orig_pils = [Image.fromarray(f.astype(np.uint8)) for f in original_faces]
        prot_pils = [Image.fromarray(f.astype(np.uint8)) for f in protected_faces]
        results = self.run_full_evaluation(
            original_images=orig_pils,
            protected_images=prot_pils,
            output_dir=os.path.join(output_dir, "gemini_eval"),
        )
        return results
