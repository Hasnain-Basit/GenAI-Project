"""Stage 1: locate NEU images, standardise them, make stratified 70/15/15 split
and the limited (30 per class) real training subset.

Usage:
    python -m src.prepare_dataset --raw data/raw [--seed 42] [--limited 30]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .common import CLASSES, IMG_SIZE, PROC, RAW, absp, mark_done, rel

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def class_from_name(stem: str) -> str | None:
    s = stem.lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"_?\d+$", "", s)  # strip trailing index
    keys = {
        "crazing": "crazing",
        "inclusion": "inclusion",
        "patches": "patches",
        "pitted_surface": "pitted_surface",
        "pitted": "pitted_surface",
        "rolled_in_scale": "rolled-in_scale",
        "rolled_in": "rolled-in_scale",
        "scratches": "scratches",
    }
    for k, v in keys.items():
        if s.startswith(k):
            return v
    return None


def try_kaggle_download(raw) -> bool:
    """Optional: works only if the kaggle CLI is installed and ~/.kaggle/kaggle.json exists."""
    if shutil.which("kaggle") is None:
        return False
    raw.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", "kaustubhdikshit/neu-surface-defect-database",
           "-p", str(raw), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warn] kaggle download failed: {e}")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limited", type=int, default=30, help="real images per class in limited setting")
    ap.add_argument("--try-kaggle", action="store_true")
    a = ap.parse_args(argv)

    raw = absp(a.raw)
    if a.try_kaggle and not any(raw.rglob("*")):
        try_kaggle_download(raw)

    files = [p for p in raw.rglob("*") if p.suffix.lower() in EXTS] if raw.exists() else []
    if not files:
        print(f"[error] No images found under {raw}.\n"
              "Download the NEU surface defect database (see README.md, section 'Dataset'),\n"
              "unzip it anywhere inside data/raw/, then re-run.")
        return 1

    seen, rows = set(), []
    out_dir = PROC / "images"
    for p in sorted(files):
        if p.stem.lower() in seen:
            continue
        label = class_from_name(p.stem)
        if label is None:
            # fall back to parent folder name
            label = class_from_name(p.parent.name)
        if label is None:
            continue
        seen.add(p.stem.lower())
        dst = out_dir / label / f"{p.stem.lower()}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                im = Image.open(p).convert("L")
                if im.size != (IMG_SIZE, IMG_SIZE):
                    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                im.save(dst)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] skipping unreadable {p.name}: {e}")
                continue
        rows.append({"path": rel(dst), "label": label})

    df = pd.DataFrame(rows)
    counts = df.label.value_counts().reindex(CLASSES).fillna(0).astype(int)
    print("Images per class:\n", counts.to_string())
    if (counts == 0).any():
        print("[error] Some classes have no images. Check file names (e.g. crazing_1.jpg).")
        return 1
    if len(df) != 1800:
        print(f"[warn] Expected 1800 images, found {len(df)}. Continuing with what was found.")

    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=a.seed)
    tr.assign(split="train").to_csv(PROC / "train.csv", index=False)
    va.assign(split="val").to_csv(PROC / "val.csv", index=False)
    te.assign(split="test").to_csv(PROC / "test.csv", index=False)

    lim = pd.concat([tr[tr.label == c].sample(n=min(a.limited, int((tr.label == c).sum())), random_state=a.seed)
                     for c in CLASSES]).reset_index(drop=True)
    lim.to_csv(PROC / "train_limited.csv", index=False)

    summary = pd.DataFrame({
        "train": tr.label.value_counts().reindex(CLASSES),
        "val": va.label.value_counts().reindex(CLASSES),
        "test": te.label.value_counts().reindex(CLASSES),
        "train_limited": lim.label.value_counts().reindex(CLASSES),
    })
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(PROC / "split_summary.csv")
    print("\nSplit summary:\n", summary.to_string())
    mark_done("prepare", {"n": int(len(df))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
