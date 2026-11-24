"""Stage 7: build report/report.md and report/report.html from report_template.md using the REAL
numbers in results/*.csv. Nothing is hard-coded: if a result does not exist the section says so.

Open report/report.html in a browser, Ctrl+A, Ctrl+C, paste into Word, export as PDF.
Edit TEAM / COURSE strings in report_template.md (or the .md output) before submitting.
"""
from __future__ import annotations

import base64
import re
import sys

import pandas as pd

from .common import CLASSES, FIGS, PROC, RESULTS, ROOT, SYN_RAW, mark_done

NAMES = {"A_full_real": "A. Full real", "B_limited_real": "B. Limited real (30/class)",
         "C_limited_plus_syn": "C. Limited real + diffusion synthetic", "D_synthetic_only": "D. Synthetic only",
         "E_limited_plus_aug": "E. Limited real + classical augmentation (control)"}


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def findings(ms: pd.DataFrame, pc: pd.DataFrame) -> str:
    g = lambda e, c: float(ms.set_index("experiment").loc[e, c])  # noqa: E731
    have = set(ms.experiment)
    L = []
    if {"A_full_real", "B_limited_real"} <= have:
        L.append(f"- Reducing real training data from the full training split to 30 images per class changed test accuracy from "
                 f"{g('A_full_real', 'accuracy'):.3f} to {g('B_limited_real', 'accuracy'):.3f} "
                 f"(macro-F1 {g('A_full_real', 'macro_f1'):.3f} to {g('B_limited_real', 'macro_f1'):.3f}).")
    pc_ = pd.read_csv(RESULTS / "paired_comparisons.csv") if (RESULTS / "paired_comparisons.csv").exists() else None

    def pair(a, b, label):
        if pc_ is None or not {a, b} <= have:
            return
        s = pc_[(pc_.A == a) & (pc_.B == b)].set_index("metric")
        if s.empty:
            return
        for m in ("accuracy", "macro_f1"):
            r = s.loc[m]
            d, lo, hi = r.mean_diff_A_minus_B, r.ci_low, r.ci_high
            verdict = ("a difference whose 95% bootstrap interval excludes zero" if r.ci_excludes_zero
                       else "a difference that is NOT distinguishable from zero given the test-set size")
            L.append(f"- {label}, {m.replace('_', ' ')}: mean difference {d:+.3f} (95% bootstrap CI {lo:+.3f} to {hi:+.3f}), {verdict}.")

    pair("C_limited_plus_syn", "B_limited_real", "Adding filtered synthetic images (see generation method above) to the limited real set (C minus B)")
    pair("E_limited_plus_aug", "B_limited_real", "Adding the same number of classically augmented images (E minus B)")
    pair("C_limited_plus_syn", "E_limited_plus_aug", "Filtered synthetic versus classically augmented additions (C minus E)")
    if "D_synthetic_only" in have:
        L.append(f"- Training on synthetic images only (D) gave accuracy {g('D_synthetic_only', 'accuracy'):.3f} "
                 f"against a chance level of {1 / len(CLASSES):.3f} for six balanced classes.")
    if not L:
        L.append("- (Not enough experiments were available to compute comparisons.)")
    worst = pc.set_index("experiment")
    if "B_limited_real" in worst.index:
        r = worst.loc["B_limited_real", CLASSES].astype(float)
        L.append(f"- In the limited-real setting the weakest class was {r.idxmin()} (F1 {r.min():.3f}) and the strongest was {r.idxmax()} (F1 {r.max():.3f}).")
    return "\n".join(L)


def embed(md: str, outdir) -> str:
    def rep(m):
        alt, path = m.group(1), m.group(2)
        f = (ROOT / path).resolve()
        if not f.exists():
            return f"*[missing figure: {path}]*"
        b64 = base64.b64encode(f.read_bytes()).decode()
        return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%">'
    return re.sub(r"!\[(.*?)\]\((figures/.*?)\)", rep, md)


def main(argv=None) -> int:
    tpl = (ROOT / "report_template.md").read_text(encoding="utf-8")
    if not (RESULTS / "metrics_summary.csv").exists():
        print("[error] No results yet. Run evaluate first.")
        return 1
    ms = pd.read_csv(RESULTS / "metrics_summary.csv")
    pc = pd.read_csv(RESULTS / "per_class_f1.csv")
    meta = pd.read_csv(RESULTS / "train_meta.csv") if (RESULTS / "train_meta.csv").exists() else pd.DataFrame()
    man = pd.read_csv(SYN_RAW / "manifest.csv") if (SYN_RAW / "manifest.csv").exists() else pd.DataFrame()
    rep = pd.read_csv(RESULTS / "filter_report.csv") if (RESULTS / "filter_report.csv").exists() else pd.DataFrame()

    t = ms.copy()
    t.insert(1, "n_train", t.experiment.map(meta.set_index("experiment").n_train) if len(meta) else "")
    t["experiment"] = t.experiment.map(NAMES)
    t["Accuracy [95% CI]"] = t.apply(lambda r: f"{r.accuracy:.3f} [{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}]", axis=1)
    t["Macro-F1 [95% CI]"] = t.apply(lambda r: f"{r.macro_f1:.3f} [{r.f1_ci_low:.3f}, {r.f1_ci_high:.3f}]", axis=1)
    tbl = t[["experiment", "n_train", "Accuracy [95% CI]", "macro_precision", "macro_recall", "Macro-F1 [95% CI]"]]
    tbl.columns = ["Configuration", "Train images", "Accuracy [95% CI]", "Macro precision", "Macro recall", "Macro-F1 [95% CI]"]

    pcf = pc.copy()
    pcf["experiment"] = pcf.experiment.map(NAMES)

    paired = pd.read_csv(RESULTS / "paired_comparisons.csv") if (RESULTS / "paired_comparisons.csv").exists() else pd.DataFrame()
    split = pd.read_csv(PROC / "split_summary.csv") if (PROC / "split_summary.csv").exists() else pd.DataFrame()
    fsum = pd.read_csv(RESULTS / "filter_summary.csv") if (RESULTS / "filter_summary.csv").exists() else pd.DataFrame()

    method = man.method.iloc[0] if len(man) else "none"
    if method == "sd_img2img_latent_diffusion":
        p = man.iloc[0]
        backend = (f"Synthetic images were produced by a pretrained latent diffusion model (`{p.model}`) run on CPU in "
                   f"image-to-image mode (strength {p.strength}, guidance {p.guidance}, {int(p.steps)} steps), initialised from the "
                   "30-per-class limited real training images only and conditioned on one class-specific text prompt per class. "
                   "**No LoRA or other fine-tuning was performed**, and no inpainting masks or vision-language-model prompts were used.")
    elif method == "classical_augmentation":
        backend = ("**WARNING: this run used the classical-augmentation backend, NOT diffusion.** The 'synthetic' images are "
                   "geometric/photometric augmentations of real images. This report must not claim diffusion-based generation for these results.")
    else:
        backend = "No synthetic images were generated in this run."

    subs = {
        "{{BACKEND}}": backend,
        "{{N_GEN}}": str(len(man)),
        "{{N_KEPT}}": str(int(rep.keep.sum())) if len(rep) else "0",
        "{{TABLE_SPLIT}}": md_table(split) if len(split) else "(split summary missing)",
        "{{TABLE_FILTER}}": md_table(fsum) if len(fsum) else "(filter summary missing)",
        "{{TABLE_METRICS}}": md_table(tbl),
        "{{TABLE_PERCLASS}}": md_table(pcf),
        "{{TABLE_PAIRED}}": md_table(paired) if len(paired) else "(no paired comparisons)",
        "{{FINDINGS}}": findings(ms, pc),
        "{{N_TEST}}": str(int(ms.n_test.iloc[0])),
        "{{N_TEST_STEP}}": f"{100 / int(ms.n_test.iloc[0]):.2f} percentage points",
        "{{FEATURES}}": str(meta.features.iloc[0]).upper() if len(meta) else "HOG",
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)

    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(tpl, encoding="utf-8")
    try:
        import markdown
        body = markdown.markdown(embed(tpl, out), extensions=["tables"])
        css = ("body{font-family:'Times New Roman',serif;font-size:12pt;line-height:1.4;max-width:800px;margin:auto}"
               "table{border-collapse:collapse;font-size:10pt;margin:8px 0}td,th{border:1px solid #888;padding:3px 6px}"
               "h1{font-size:20pt}h2{font-size:15pt}h3{font-size:13pt}")
        (out / "report.html").write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>",
                                         encoding="utf-8")
        print("Wrote report/report.md and report/report.html")
    except ImportError:
        print("Wrote report/report.md (pip install markdown to also get report.html)")
    mark_done("report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
