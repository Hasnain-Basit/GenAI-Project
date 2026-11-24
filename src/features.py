"""Feature extraction (CPU-friendly). Default: HOG. Optional: frozen ResNet-18 (needs torch)."""
from __future__ import annotations

import numpy as np
from skimage.feature import hog
from skimage.transform import resize

from .common import absp, load_gray


def hog_feat(img: np.ndarray) -> np.ndarray:
    im = resize(img, (128, 128), anti_aliasing=True)
    return hog(im, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


_resnet = None


def resnet_feat(img: np.ndarray) -> np.ndarray:
    global _resnet
    import torch
    import torchvision

    if _resnet is None:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
        _resnet = m.eval()
    im = resize(img, (224, 224), anti_aliasing=True).astype(np.float32)
    x = np.stack([im] * 3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = torch.from_numpy((x - mean) / std).unsqueeze(0)
    with torch.no_grad():
        return _resnet(x).squeeze(0).numpy()


def featurize(paths, kind: str = "hog") -> np.ndarray:
    fn = hog_feat if kind == "hog" else resnet_feat
    return np.stack([fn(load_gray(absp(p))) for p in paths])
