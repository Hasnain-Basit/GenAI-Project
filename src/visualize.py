"""Stage 6: figures. Everything is drawn from the saved CSVs / images (no numbers typed by hand).

Outputs (figures/): accuracy_comparison.png, f1_comparison.png, per_class_f1.png,
confusion_matrices.png, gallery_real_vs_synthetic.png, filter_summary.png

Usage: python -m src.visualize
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .common import CLASSES, FIGS, PROC, RESULTS, SYN_RAW, absp, mark_done

LABEL = {"A_full_real": "A: Full real", "B_limited_real": "B: Limited real (30/cls)",
         "C_limited_plus_syn": "C: B + diffusion synth", "D_synthetic_only": "D: Synthetic only",
         "E_limited_plus_aug": "E: B + classical aug (control)"}
COL = {"A_full_real": "#4c78a8", "B_limited_real": "#9ecae9", "C_limited_plus_syn": "#f58518",
       "D_synthetic_only": "#54a24b", "E_limited_plus_aug": "#b279a2"}


def bar(ms, metric, lo, hi, title, fn):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(ms))
    vals = ms[metric].values
    err = np.vstack([vals - ms[lo].values, ms[hi].values - vals])
    ax.bar(x, vals, yerr=err, capsize=4, color=[COL[e] for e in ms.experiment])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[e] for e in ms.experiment], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title + "\n(held-out real test set; error bars = bootstrap 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / fn, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    p = RESULTS / "metrics_summary.csv"
    if not p.exists():
        print("[error] Run evaluate first.")
        return 1
    ms = pd.read_csv(p)
    bar(ms, "accuracy", "acc_ci_low", "acc_ci_high", "Accuracy comparison", "accuracy_comparison.png")
    bar(ms, "macro_f1", "f1_ci_low", "f1_ci_high", "Macro-F1 comparison", "f1_comparison.png")

    pc = pd.read_csv(RESULTS / "per_class_f1.csv").set_index("experiment")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(pc)
    for i, (e, row) in enumerate(pc.iterrows()):
        ax.bar(np.arange(len(CLASSES)) + i * w, row[CLASSES].values, w, label=LABEL[e], color=COL[e])
    ax.set_xticks(np.arange(len(CLASSES)) + 0.4 - w / 2)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 on the held-out real test set")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "per_class_f1.png", dpi=200)
    plt.close(fig)

    exps = list(ms.experiment)
    cols = min(3, len(exps))
    rows = int(np.ceil(len(exps) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, e in zip(axs.ravel(), exps):
        cm = pd.read_csv(RESULTS / f"confusion_{e}.csv", index_col=0)
        ax.axis("on")
        ax.imshow(cm.values, cmap="Blues")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = cm.values[i, j]
                ax.text(j, i, v, ha="center", va="center", fontsize=8,
                        color="white" if v > cm.values.max() / 2 else "black")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(LABEL[e], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    # Gallery: per class, 4 real source images (top) vs 4 synthetic images (bottom)
    man_p = SYN_RAW / "manifest.csv"
    if man_p.exists():
        man = pd.read_csv(man_p)
        rep_p = RESULTS / "filter_report.csv"
        keep = pd.read_csv(rep_p).set_index("path").keep.to_dict() if rep_p.exists() else {}
        lim = pd.read_csv(PROC / "train_limited.csv")
        n = 4
        fig, axs = plt.subplots(len(CLASSES) * 2, n, figsize=(2 * n, 2 * len(CLASSES) * 2))
        for ci, c in enumerate(CLASSES):
            real = lim[lim.label == c].head(n)
            syn = man[man.label == c].head(n)
            for j in range(n):
                for k, (df, tag) in enumerate(((real, "real"), (syn, "synthetic"))):
                    ax = axs[ci * 2 + k, j]
                    ax.axis("off")
                    if j < len(df):
                        ax.imshow(Image.open(absp(df.iloc[j].path)).convert("L"), cmap="gray", vmin=0, vmax=255)
                        t = f"{c} [{tag}]"
                        if tag == "synthetic" and keep:
                            t += "" if keep.get(df.iloc[j].path, True) else " (rejected)"
                        ax.set_title(t, fontsize=6)
        method = man.method.iloc[0] if len(man) else "?"
        fig.suptitle(f"Real (odd rows) vs synthetic (even rows) | synthetic method: {method}", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / "gallery_real_vs_synthetic.png", dpi=150)
        plt.close(fig)

    fs = RESULTS / "filter_summary.csv"
    if fs.exists():
        d = pd.read_csv(fs, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.generated, 0.4, label="generated")
        ax.bar(x + 0.2, d.kept, 0.4, label="kept after filtering")
        ax.set_xticks(x)
        ax.set_xticklabels(d.index, rotation=15)
        ax.legend()
        ax.set_title("Synthetic images per class: generated vs kept")
        fig.tight_layout()
        fig.savefig(FIGS / "filter_summary.png", dpi=200)
        plt.close(fig)

    print("Figures written to", FIGS)
    mark_done("visualize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
