"""Stage 2: synthetic defect image generation.

Backends
--------
sd_img2img (default, REAL latent-diffusion generation)
    Pretrained Stable Diffusion (a Latent Diffusion Model, Rombach et al. 2022) run
    in image-to-image mode on CPU with a class-specific text prompt. The init images
    come ONLY from the limited real training subset (train_limited.csv), so no
    validation/test image is ever used. NO LoRA fine-tuning is performed.

augment (NOT diffusion)
    Classical augmentation (rotation/flip/crop/brightness/noise). Provided only as a
    clearly-labelled fallback / control. Outputs are tagged method='classical_augmentation'
    in the manifest and must never be described as diffusion-generated.

Usage:
    python -m src.generate_synthetic --backend sd_img2img --per-class 20
    python -m src.generate_synthetic --backend augment   --per-class 20   # control only
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from .common import CLASSES, PROC, SYN_RAW, absp, load_gray, mark_done, rel, save_gray

PROMPTS = {
    "crazing": "macro photo of hot-rolled steel strip surface with crazing, a network of fine "
               "interconnected hairline cracks, grayscale industrial inspection image",
    "inclusion": "macro photo of hot-rolled steel strip surface with inclusion defect, dark "
                 "elongated embedded non-metallic streaks and spots, grayscale industrial inspection image",
    "patches": "macro photo of hot-rolled steel strip surface with patches defect, irregular "
               "dark blotchy patches of uneven texture, grayscale industrial inspection image",
    "pitted_surface": "macro photo of hot-rolled steel strip surface with pitted surface defect, "
                      "many small pits and cavities scattered over the surface, grayscale industrial inspection image",
    "rolled-in_scale": "macro photo of hot-rolled steel strip surface with rolled-in scale defect, "
                       "dark scale oxide pressed into the surface forming irregular streaky marks, grayscale industrial inspection image",
    "scratches": "macro photo of hot-rolled steel strip surface with scratches, long thin straight "
                 "linear scratch lines across the metal, grayscale industrial inspection image",
}
NEG_PROMPT = "color, text, watermark, logo, cartoon, painting, blurry, people, 3d render"
MANIFEST = SYN_RAW / "manifest.csv"
FIELDS = ["path", "label", "method", "source_path", "prompt", "seed", "strength", "guidance", "steps", "model"]


def existing_rows() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST)
    return pd.DataFrame(columns=FIELDS)


def append_row(row: dict) -> None:
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def classical_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    im = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.rotate(float(rng.choice([0, 90, 180, 270])) + float(rng.uniform(-8, 8)),
                   resample=Image.BILINEAR, fillcolor=int(np.mean(img) * 255))
    w, h = im.size
    s = rng.uniform(0.8, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.85, 1.15)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.85, 1.15)))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = arr + rng.normal(0, 0.01, arr.shape).astype(np.float32)
    return np.clip(arr, 0, 1)


def run_augment(df, per_class, seed, out_dir):
    rng = np.random.default_rng(seed)
    done = existing_rows()
    for label in CLASSES:
        src = df[df.label == label].reset_index(drop=True)
        have = int((done.label == label).sum()) if len(done) else 0
        for k in range(have, per_class):
            r = src.iloc[k % len(src)]
            img = classical_augment(load_gray(absp(r.path)), rng)
            out = out_dir / label / f"aug_{label}_{k:03d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            save_gray(img, out)
            append_row(dict(path=rel(out), label=label, method="classical_augmentation",
                            source_path=r.path, prompt="", seed=seed + k))
        print(f"  {label}: {per_class} augmented images ready")


def run_sd(df, per_class, seed, out_dir, a):
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as e:
        print(f"[error] {e}\nInstall CPU torch + diffusers (see README) or use --backend augment (control only, NOT diffusion).")
        return 1
    torch.set_num_threads(a.threads or max(1, (torch.get_num_threads())))
    print(f"Loading {a.model} on CPU (first run downloads ~4 GB)...")
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            a.model, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    except Exception as e:  # noqa: BLE001
        print(f"[error] Could not load model '{a.model}': {e}\n"
              "Fix: check internet / the model id (try --model with another SD 1.x repo id you have access to),\n"
              "or run --backend augment as a clearly-labelled NON-diffusion control.")
        return 1
    pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)

    done = existing_rows()
    S = a.size
    for label in CLASSES:
        src = df[df.label == label].reset_index(drop=True)
        have = int((done.label == label).sum()) if len(done) else 0
        for k in range(have, per_class):
            r = src.iloc[k % len(src)]
            init = Image.open(absp(r.path)).convert("L").resize((S, S), Image.BICUBIC).convert("RGB")
            gen = torch.Generator("cpu").manual_seed(seed * 1000 + k + CLASSES.index(label) * 100000)
            t0 = time.time()
            try:
                res = pipe(prompt=PROMPTS[label], negative_prompt=NEG_PROMPT, image=init,
                           strength=a.strength, guidance_scale=a.guidance,
                           num_inference_steps=a.steps, generator=gen).images[0]
            except Exception as e:  # noqa: BLE001
                print(f"[warn] generation failed for {label} #{k}: {e}")
                continue
            res = res.convert("L").resize((200, 200), Image.LANCZOS)
            out = out_dir / label / f"sd_{label}_{k:03d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.save(out)
            append_row(dict(path=rel(out), label=label, method="sd_img2img_latent_diffusion",
                            source_path=r.path, prompt=PROMPTS[label], seed=seed * 1000 + k,
                            strength=a.strength, guidance=a.guidance, steps=a.steps, model=a.model))
            print(f"  {label} {k + 1}/{per_class}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sd_img2img", "augment"], default="sd_img2img")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--size", type=int, default=256, help="generation resolution (multiple of 8)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.55, help="0..1, higher = further from the real init image")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    lim = PROC / "train_limited.csv"
    if not lim.exists():
        print("[error] Run prepare_dataset first.")
        return 1
    df = pd.read_csv(lim)
    SYN_RAW.mkdir(parents=True, exist_ok=True)

    # Never mix backends in one run folder: that would blur what the images are.
    prev = existing_rows()
    want = "classical_augmentation" if a.backend == "augment" else "sd_img2img_latent_diffusion"
    if len(prev) and (prev.method != want).any():
        print(f"[error] {MANIFEST} already holds images from another backend. "
              "Delete data/synthetic/ (or run with --force in main.py) before switching backends.")
        return 1

    print(f"Backend: {a.backend} | per-class target: {a.per_class} | init images: limited real train only")
    rc = run_sd(df, a.per_class, a.seed, SYN_RAW, a) if a.backend == "sd_img2img" \
        else run_augment(df, a.per_class, a.seed, SYN_RAW) or 0
    if rc == 0:
        mark_done("generate", {"backend": a.backend, "per_class": a.per_class})
    return rc


if __name__ == "__main__":
    sys.exit(main())
