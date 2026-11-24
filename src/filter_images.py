"""Stage 3: quality inspection and filtering of synthetic images.

A synthetic image is KEPT only if all checks pass:
  1. not blank / not saturated        (pixel std and mean within sane range)
  2. not a near-copy of its source     (SSIM(source, synthetic) <= --ssim-max)
  3. still resembles the source domain (SSIM(source, synthetic) >= --ssim-min)
  4. class-consistent                  (a HOG+logistic-regression 'judge' trained ONLY on the
                                        limited REAL training subset gives the intended class
                                        probability >= --min-prob)

Note: check 4 prefers samples that a 30-image-per-class model finds easy, which can bias the kept
set toward typical-looking defects. This is reported in the report as a limitation.
SynSur itself filters with DreamSim and CLIPScore; this is a lightweight CPU substitute, not a replication.

Usage: python -m src.filter_images [--min-prob 0.4]
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim

from .common import CLASSES, CLASS_TO_IDX, PROC, SYN_FILT, SYN_RAW, RESULTS, absp, load_gray, mark_done
from .features import featurize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prob", type=float, default=0.40)
    ap.add_argument("--ssim-min", type=float, default=0.02)
    ap.add_argument("--ssim-max", type=float, default=0.90)
    ap.add_argument("--min-std", type=float, default=0.02)
    a = ap.parse_args(argv)

    man_p = SYN_RAW / "manifest.csv"
    if not man_p.exists():
        print("[error] No synthetic manifest. Run generate_synthetic first.")
        return 1
    man = pd.read_csv(man_p)
    lim = pd.read_csv(PROC / "train_limited.csv")

    Xr = featurize(lim.path)
    yr = lim.label.map(CLASS_TO_IDX).values
    sc = StandardScaler().fit(Xr)
    judge = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xr), yr)

    Xs = featurize(man.path)
    probs = judge.predict_proba(sc.transform(Xs))

    rows = []
    for i, r in man.iterrows():
        img = load_gray(absp(r.path), 200)
        src = load_gray(absp(r.source_path), 200)
        s = float(ssim(src, img, data_range=1.0))
        std, mean = float(img.std()), float(img.mean())
        p_cls = float(probs[i, CLASS_TO_IDX[r.label]])
        reasons = []
        if std < a.min_std:
            reasons.append("blank_or_flat")
        if not (0.05 < mean < 0.95):
            reasons.append("saturated")
        if s > a.ssim_max:
            reasons.append("near_copy_of_source")
        if s < a.ssim_min:
            reasons.append("unrelated_to_source")
        if p_cls < a.min_prob:
            reasons.append("class_inconsistent")
        rows.append({**r.to_dict(), "ssim_to_source": round(s, 4), "pixel_std": round(std, 4),
                     "pixel_mean": round(mean, 4), "judge_prob_true_class": round(p_cls, 4),
                     "keep": len(reasons) == 0, "reject_reasons": ";".join(reasons)})
    rep = pd.DataFrame(rows)
    rep.to_csv(RESULTS / "filter_report.csv", index=False)

    if SYN_FILT.exists():
        shutil.rmtree(SYN_FILT)
    kept = rep[rep.keep].copy()
    new_paths = []
    for _, r in kept.iterrows():
        dst = SYN_FILT / r.label / absp(r.path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(absp(r.path), dst)
        new_paths.append(str(dst.relative_to(SYN_FILT.parent.parent.parent)))
    kept["path"] = new_paths
    kept[["path", "label", "method", "source_path"]].to_csv(PROC / "synthetic_filtered.csv", index=False)

    summ = (rep.groupby("label").agg(generated=("keep", "size"), kept=("keep", "sum"))
              .reindex(CLASSES).fillna(0).astype(int))
    summ["kept_rate"] = (summ.kept / summ.generated.replace(0, np.nan)).round(3)
    summ.to_csv(RESULTS / "filter_summary.csv")
    print(summ.to_string())
    print(f"\nKept {int(rep.keep.sum())}/{len(rep)}. Reject reasons:\n",
          rep[~rep.keep].reject_reasons.str.split(";").explode().value_counts().to_string())
    mark_done("filter", {"kept": int(rep.keep.sum()), "total": int(len(rep))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
