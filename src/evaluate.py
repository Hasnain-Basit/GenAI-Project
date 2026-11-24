"""Stage 5: compute metrics from saved test predictions.

Outputs (results/):
  metrics_summary.csv       accuracy, macro precision/recall/F1 (+ bootstrap 95% CI for accuracy and macro-F1)
  per_class_f1.csv          per-class F1 per experiment
  confusion_<exp>.csv       confusion matrix (rows=true, cols=predicted)
  paired_comparisons.csv    paired bootstrap of metric differences between experiments

Usage: python -m src.evaluate
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .common import CLASSES, RESULTS, mark_done

ORDER = ["A_full_real", "B_limited_real", "C_limited_plus_syn", "D_synthetic_only", "E_limited_plus_aug"]
PAIRS = [("C_limited_plus_syn", "B_limited_real"), ("E_limited_plus_aug", "B_limited_real"),
         ("C_limited_plus_syn", "E_limited_plus_aug"), ("A_full_real", "B_limited_real")]
NB = 2000


def boot_idx(n, rng):
    return rng.integers(0, n, size=(NB, n))


def main(argv=None) -> int:
    preds = {n: pd.read_csv(RESULTS / f"preds_{n}.csv") for n in ORDER if (RESULTS / f"preds_{n}.csv").exists()}
    if not preds:
        print("[error] No prediction files. Run train_classifier first.")
        return 1
    rng = np.random.default_rng(0)
    some = next(iter(preds.values()))
    idx = boot_idx(len(some), rng)  # same resamples for every experiment => paired comparisons

    summ, pcf = [], []
    cache = {}
    for n, d in preds.items():
        yt, yp = d.true.values, d.pred.values
        assert (yt == some.true.values).all(), "Test sets differ between experiments!"
        acc = accuracy_score(yt, yp)
        p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=CLASSES, average="macro", zero_division=0)
        accs = np.array([accuracy_score(yt[i], yp[i]) for i in idx])
        f1s = np.array([f1_score(yt[i], yp[i], labels=CLASSES, average="macro", zero_division=0) for i in idx])
        cache[n] = (accs, f1s)
        summ.append({"experiment": n, "n_test": len(d), "accuracy": acc, "acc_ci_low": np.percentile(accs, 2.5),
                     "acc_ci_high": np.percentile(accs, 97.5), "macro_precision": p, "macro_recall": r,
                     "macro_f1": f, "f1_ci_low": np.percentile(f1s, 2.5), "f1_ci_high": np.percentile(f1s, 97.5)})
        pc = f1_score(yt, yp, labels=CLASSES, average=None, zero_division=0)
        pcf.append({"experiment": n, **{c: v for c, v in zip(CLASSES, pc)}})
        pd.DataFrame(confusion_matrix(yt, yp, labels=CLASSES), index=CLASSES, columns=CLASSES) \
            .to_csv(RESULTS / f"confusion_{n}.csv")

    pd.DataFrame(summ).round(4).to_csv(RESULTS / "metrics_summary.csv", index=False)
    pd.DataFrame(pcf).round(4).to_csv(RESULTS / "per_class_f1.csv", index=False)

    comp = []
    for a, b in PAIRS:
        if a in cache and b in cache:
            for mi, mname in enumerate(("accuracy", "macro_f1")):
                diff = cache[a][mi] - cache[b][mi]
                comp.append({"A": a, "B": b, "metric": mname,
                             "mean_diff_A_minus_B": diff.mean(), "ci_low": np.percentile(diff, 2.5),
                             "ci_high": np.percentile(diff, 97.5),
                             "ci_excludes_zero": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0)})
    pd.DataFrame(comp).round(4).to_csv(RESULTS / "paired_comparisons.csv", index=False)

    print(pd.DataFrame(summ).round(3)[["experiment", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].to_string(index=False))
    mark_done("evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
