"""Run the whole pipeline with resume support.

    python main.py                       # all stages, SD img2img generation (needs torch+diffusers)
    python main.py --backend augment     # control-only run WITHOUT diffusion (labelled as such)
    python main.py --from-stage train    # re-run from a stage onward
    python main.py --force               # redo everything

Stages (in order): prepare, generate, filter, train, evaluate, visualize, report
A stage with a '.done' marker in ./.stages is skipped unless --force / --from-stage is used.
"""
from __future__ import annotations

import argparse
import shutil
import sys

from src import (evaluate, filter_images, generate_synthetic, make_report, prepare_dataset,
                 train_classifier, visualize)
from src.common import MARKERS, ROOT, clear_done, is_done

STAGES = ["prepare", "generate", "filter", "train", "evaluate", "visualize", "report"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sd_img2img", "augment"], default="sd_img2img")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--limited", type=int, default=30)
    ap.add_argument("--features", choices=["hog", "resnet18"], default="hog")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--from-stage", choices=STAGES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.force:
        for s in STAGES:
            clear_done(s)
        shutil.rmtree(ROOT / "data" / "synthetic", ignore_errors=True)
    if a.from_stage:
        for s in STAGES[STAGES.index(a.from_stage):]:
            clear_done(s)

    jobs = {
        "prepare": lambda: prepare_dataset.main(["--limited", str(a.limited)]),
        "generate": lambda: generate_synthetic.main(
            ["--backend", a.backend, "--per-class", str(a.per_class), "--model", a.model,
             "--size", str(a.size), "--steps", str(a.steps), "--strength", str(a.strength)]),
        "filter": lambda: filter_images.main([]),
        "train": lambda: train_classifier.main(["--features", a.features]),
        "evaluate": lambda: evaluate.main([]),
        "visualize": lambda: visualize.main([]),
        "report": lambda: make_report.main([]),
    }
    for s in STAGES:
        if is_done(s):
            print(f"[skip] {s} (already done)")
            continue
        print(f"\n===== {s} =====")
        rc = jobs[s]()
        if rc:
            print(f"\n[stopped] stage '{s}' failed (code {rc}). Fix the message above and re-run: python main.py")
            return rc
    print("\nAll stages complete. See results/, figures/, report/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
