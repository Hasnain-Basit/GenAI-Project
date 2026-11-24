"""Stage 4: train classifiers for every experimental configuration and save test predictions.

Configurations (all evaluated on the SAME held-out real test set):
  A_full_real          all real training images (70% split)
  B_limited_real       30 real images per class
  C_limited_plus_syn   B + filtered diffusion-generated images
  D_synthetic_only     filtered diffusion-generated images only (no real training images)
  E_limited_plus_aug   B + equal number of CLASSICAL augmentations of B (control; NOT diffusion)

Classifier: standardised features -> SVM (RBF). C and gamma picked on the real validation set.
Features: HOG (default) or frozen ResNet-18 (--features resnet18, needs torch).

Usage: python -m src.train_classifier [--features hog]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from .common import CLASSES, CLASS_TO_IDX, PROC, RESULTS, absp, load_gray, mark_done
from .features import hog_feat, resnet_feat
from .generate_synthetic import classical_augment

CACHE = PROC / "feature_cache"


def feats(paths, kind):
    """Feature matrix with per-image disk cache (resume-friendly)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = hog_feat if kind == "hog" else resnet_feat
    out = []
    for p in paths:
        f = CACHE / (kind + "_" + str(p).replace("/", "__").replace("\\", "__") + ".npy")
        if f.exists():
            out.append(np.load(f))
        else:
            v = fn(load_gray(absp(p)))
            np.save(f, v)
            out.append(v)
    return np.stack(out)


def fit_select(Xtr, ytr, Xva, yva, seed):
    best, best_f1 = None, -1.0
    for C in (1, 10, 100):
        for gamma in ("scale", 0.001):
            m = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf", random_state=seed))
            m.fit(Xtr, ytr)
            f1 = f1_score(yva, m.predict(Xva), average="macro")
            if f1 > best_f1:
                best, best_f1, best_cfg = m, f1, (C, gamma)
    return best, best_cfg, best_f1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["hog", "resnet18"], default="hog")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    kind = "hog" if a.features == "hog" else "resnet18"

    for need in ("train.csv", "val.csv", "test.csv", "train_limited.csv"):
        if not (PROC / need).exists():
            print(f"[error] {need} missing. Run prepare_dataset first.")
            return 1
    tr, va, te, lim = (pd.read_csv(PROC / n) for n in ("train.csv", "val.csv", "test.csv", "train_limited.csv"))
    syn_p = PROC / "synthetic_filtered.csv"
    syn = pd.read_csv(syn_p) if syn_p.exists() else pd.DataFrame(columns=["path", "label"])

    y = lambda d: d.label.map(CLASS_TO_IDX).values  # noqa: E731
    Xva, Xte = feats(va.path, kind), feats(te.path, kind)
    exps = {}
    exps["A_full_real"] = (feats(tr.path, kind), y(tr), len(tr))
    exps["B_limited_real"] = (feats(lim.path, kind), y(lim), len(lim))

    if len(syn):
        Xs = feats(syn.path, kind)
        Xl = feats(lim.path, kind)
        exps["C_limited_plus_syn"] = (np.vstack([Xl, Xs]), np.concatenate([y(lim), y(syn)]), len(lim) + len(syn))
        exps["D_synthetic_only"] = (Xs, y(syn), len(syn))
        # Control: same number of extra images per class, produced by classical augmentation (not diffusion)
        rng = np.random.default_rng(a.seed)
        aug_X, aug_y = [], []
        for c in CLASSES:
            n_c = int((syn.label == c).sum())
            src = lim[lim.label == c].reset_index(drop=True)
            for k in range(n_c):
                img = classical_augment(load_gray(absp(src.iloc[k % len(src)].path)), rng)
                aug_X.append(hog_feat(img) if kind == "hog" else resnet_feat(img))
                aug_y.append(CLASS_TO_IDX[c])
        if aug_X:
            exps["E_limited_plus_aug"] = (np.vstack([Xl, np.stack(aug_X)]),
                                           np.concatenate([y(lim), np.array(aug_y)]), len(lim) + len(aug_X))
    else:
        print("[info] No filtered synthetic images -> skipping C, D, E.")

    meta = []
    for name, (Xtr, ytr, n) in exps.items():
        if len(set(ytr)) < len(CLASSES) and name != "D_synthetic_only":
            print(f"[warn] {name}: not all classes present, skipping")
            continue
        model, cfg, vf1 = fit_select(Xtr, ytr, Xva, y(va), a.seed)
        pred = model.predict(Xte)
        pd.DataFrame({"path": te.path, "true": te.label,
                      "pred": [CLASSES[i] for i in pred]}).to_csv(RESULTS / f"preds_{name}.csv", index=False)
        meta.append({"experiment": name, "n_train": n, "features": kind,
                     "svm_C": cfg[0], "svm_gamma": cfg[1], "val_macro_f1": round(vf1, 4)})
        print(f"{name:22s} n_train={n:5d}  val macroF1={vf1:.3f}  (C={cfg[0]}, gamma={cfg[1]})")

    pd.DataFrame(meta).to_csv(RESULTS / "train_meta.csv", index=False)
    mark_done("train", {"experiments": [m["experiment"] for m in meta]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
