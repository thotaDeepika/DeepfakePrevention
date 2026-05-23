"""
models/clip_l14.py
OpenAI CLIP ViT-L/14 wrapper for SafeUpload.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision import transforms
from PIL import Image


class CLIPl14Model(nn.Module):
    EMBEDDING_DIM = 768
    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._clip_preprocess = preprocess
        self._normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self._input_size = (224, 224)

    def _pil_to_tensor(self, img):
        return self._clip_preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)

    def _tensor_preprocess(self, x):
        x = F.interpolate(x, size=self._input_size, mode="bilinear", align_corners=False)
        return self._normalize(x)

    @torch.no_grad()
    def extract_embedding(self, img):
        if isinstance(img, Image.Image):
            t = self._pil_to_tensor(img)
        elif isinstance(img, torch.Tensor):
            t = self._tensor_preprocess(img if img.dim() == 4 else img.unsqueeze(0))
        else:
            raise TypeError(f"Unsupported: {type(img)}")
        emb = self.model.encode_image(t)
        return F.normalize(emb.float(), dim=1)

    def extract_embedding_grad(self, x):
        t = self._tensor_preprocess(x)
        emb = self.model.encode_image(t)
        return F.normalize(emb.float(), dim=1)

    @torch.no_grad()
    def batch_embedding(self, imgs):
        if isinstance(imgs, (list, tuple)):
            tensors = torch.cat([self._pil_to_tensor(im) for im in imgs], dim=0)
        else:
            tensors = self._tensor_preprocess(imgs)
        embs = self.model.encode_image(tensors)
        return F.normalize(embs.float(), dim=1)

    @staticmethod
    def similarity(a, b):
        return F.cosine_similarity(a, b, dim=1)
