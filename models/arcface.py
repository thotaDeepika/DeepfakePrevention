"""
models/arcface.py
ArcFace wrapper using insightface buffalo_l model.
Falls back to a lightweight ONNX-based extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class ArcFaceModel(nn.Module):
    """
    ArcFace (buffalo_l) embedding wrapper via insightface.
    Returns 512-d normalised embeddings.
    """

    EMBEDDING_DIM = 512
    INPUT_SIZE = (112, 112)

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self._load_model()

        self.preprocess = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # Load recognition model only (no detection needed here)
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                allowed_modules=["recognition"],
            )
            self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(112, 112))
            self._use_insightface = True
        except Exception as e:
            print(f"[ArcFace] insightface load failed ({e}). Falling back to FaceNet weights.")
            # Graceful fallback: use FaceNet so pipeline doesn't break
            from facenet_pytorch import InceptionResnetV1
            self._fallback = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            for p in self._fallback.parameters():
                p.requires_grad_(False)
            self._use_insightface = False

    def _pil_to_np(self, img):
        return np.array(img.convert("RGB"))

    def _pil_to_tensor(self, img):
        return self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)

    def _tensor_preprocess(self, x):
        x = F.interpolate(x, size=self.INPUT_SIZE, mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    @torch.no_grad()
    def _insightface_embed(self, img_np):
        """Run insightface recognition on a numpy BGR image."""
        import cv2
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Resize to 112x112 for recognition model
        bgr = cv2.resize(bgr, self.INPUT_SIZE)
        # insightface recognition model expects 112x112 RGB blob
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Get embedding from recognition model directly
        model = self._app.models.get("recognition")
        if model is None:
            # Try alternative key
            for k, v in self._app.models.items():
                model = v
                break
        emb = model.get_feat(rgb[np.newaxis])  # [1, 512]
        return torch.tensor(emb, dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def extract_embedding(self, img):
        if self._use_insightface:
            if isinstance(img, Image.Image):
                arr = self._pil_to_np(img)
            elif isinstance(img, torch.Tensor):
                arr = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                raise TypeError(f"Unsupported: {type(img)}")
            emb = self._insightface_embed(arr)
        else:
            if isinstance(img, Image.Image):
                t = self._pil_to_tensor(img)
            else:
                t = self._tensor_preprocess(img if img.dim() == 4 else img.unsqueeze(0))
            emb = self._fallback(t)
        return F.normalize(emb, dim=1)

    def extract_embedding_grad(self, x):
        """
        Differentiable pass. Uses fallback model (FaceNet).
        ArcFace via insightface is not differentiable through Python.
        """
        if not self._use_insightface:
            t = self._tensor_preprocess(x)
            return F.normalize(self._fallback(t), dim=1)
        # For gradient optimisation, use a differentiable proxy (FaceNet fallback)
        # We keep ArcFace for no-grad evaluation, FaceNet for gradient pass
        from facenet_pytorch import InceptionResnetV1
        if not hasattr(self, "_grad_proxy"):
            self._grad_proxy = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        t = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        t = (t - 0.5) / 0.5
        return F.normalize(self._grad_proxy(t), dim=1)

    @torch.no_grad()
    def batch_embedding(self, imgs):
        if isinstance(imgs, (list, tuple)):
            embs = [self.extract_embedding(im) for im in imgs]
            return torch.cat(embs, dim=0)
        else:
            # Tensor batch
            results = []
            for i in range(imgs.shape[0]):
                results.append(self.extract_embedding(imgs[i:i+1]))
            return torch.cat(results, dim=0)

    @staticmethod
    def similarity(a, b):
        return F.cosine_similarity(a, b, dim=1)
