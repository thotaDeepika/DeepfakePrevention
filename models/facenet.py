"""
models/facenet.py
FaceNet embedding model wrapper for SafeUpload.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np


class FaceNetModel(nn.Module):
    INPUT_SIZE = (160, 160)
    EMBEDDING_DIM = 512

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.preprocess = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _pil_to_tensor(self, img):
        return self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)

    def _tensor_preprocess(self, x):
        x = F.interpolate(x, size=self.INPUT_SIZE, mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    @torch.no_grad()
    def extract_embedding(self, img):
        if isinstance(img, Image.Image):
            t = self._pil_to_tensor(img)
        elif isinstance(img, torch.Tensor):
            t = self._tensor_preprocess(img if img.dim() == 4 else img.unsqueeze(0))
        else:
            raise TypeError(f"Unsupported type: {type(img)}")
        emb = self.model(t)
        return F.normalize(emb, dim=1)

    def extract_embedding_grad(self, x):
        """Differentiable. x: [B,3,H,W] in [0,1]"""
        t = self._tensor_preprocess(x)
        emb = self.model(t)
        return F.normalize(emb, dim=1)

    @torch.no_grad()
    def batch_embedding(self, imgs):
        if isinstance(imgs, (list, tuple)):
            tensors = torch.cat([self._pil_to_tensor(im) for im in imgs], dim=0)
        else:
            tensors = self._tensor_preprocess(imgs)
        embs = self.model(tensors)
        return F.normalize(embs, dim=1)

    @staticmethod
    def similarity(a, b):
        return F.cosine_similarity(a, b, dim=1)
