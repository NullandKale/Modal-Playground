# depth_estimator.py
from typing import Optional
import numpy as np
from PIL import Image

def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

class DepthEstimator:
    def __init__(self, model: str, device: str):
        from transformers import pipeline
        if device == "auto": device = "cuda" if _cuda_ok() else "cpu"
        self.pipe = pipeline(task="depth-estimation", model=model, device=(0 if device=="cuda" else (-1 if device=="cpu" else (0 if _cuda_ok() else -1))))

    def infer01(self, pil_img: Image.Image) -> np.ndarray:
        out = self.pipe(pil_img)
        depth_img = out.get("depth") or out.get("predicted_depth")
        if isinstance(depth_img, Image.Image):
            img = depth_img
        else:
            arr = np.array(depth_img, dtype=np.float32); arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            img = Image.fromarray((arr * 255.0).astype(np.uint8))
        if img.size != pil_img.size:
            img = img.resize(pil_img.size, Image.BICUBIC)
        return np.asarray(img).astype(np.float32) / 255.0
