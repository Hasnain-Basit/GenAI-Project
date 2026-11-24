"""Shared constants and helpers for the AI4009 synthetic-defect project."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
SYN_RAW = DATA / "synthetic" / "raw"
SYN_FILT = DATA / "synthetic" / "filtered"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
MARKERS = ROOT / ".stages"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 200  # NEU native resolution

for _d in (PROC, SYN_RAW, SYN_FILT, RESULTS, FIGS, MARKERS):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_gray(path, size: int | None = None) -> np.ndarray:
    """Load an image as float32 grayscale in [0,1]."""
    im = Image.open(path).convert("L")
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def save_gray(arr: np.ndarray, path) -> None:
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


def rel(path) -> str:
    """Path relative to project root (portable CSVs)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def absp(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def mark_done(stage: str, info: dict | None = None) -> None:
    (MARKERS / f"{stage}.done").write_text(json.dumps(info or {}, indent=2))


def is_done(stage: str) -> bool:
    return (MARKERS / f"{stage}.done").exists()


def clear_done(stage: str) -> None:
    f = MARKERS / f"{stage}.done"
    if f.exists():
        f.unlink()
