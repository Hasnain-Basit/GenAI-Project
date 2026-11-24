# Presentation: 10 slides, ~10 minutes
Fill every [BRACKET] from `results/metrics_summary.csv` and `report/report.md` after your run. Do not quote numbers from anywhere else.

## Slide 1 – Title
**High-Fidelity Synthetic Defect Generation for Industrial Visual Inspection using Latent Diffusion Models**
AI4009 Generative AI | [Team names, roll numbers] | [Date]
Figure: one real NEU image next to one synthetic image (from `figures/gallery_real_vs_synthetic.png`).
*Script (30 s):* We ask a simple question: can synthetic images from a latent diffusion model help a classifier when real defect images are scarce?

## Slide 2 – Problem and motivation
- Defects are rare; labelling needs experts
- Small datasets → weak classifiers
- Generative AI could enlarge scarce datasets, but must be tested on REAL images
Figure: `figures/split` table or class counts from `data/processed/split_summary.csv`.
*Script (45 s):* Collecting balanced defect data is slow and costly. If generated images really help, inspection systems can be built earlier. We test that claim rather than assume it.

## Slide 3 – Research papers
- SynSur (Kühn et al., 2026, arXiv:2604.26633): VLM prompts, LoRA-adapted Stable Diffusion, mask-guided inpainting, DreamSim/CLIPScore filtering, evaluated on BSData and MSD scratches with YOLOX, YOLOv26, LW-DETR
- Finding: synthetic data best strengthens scarce real data, not replaces it
- Latent Diffusion (Rombach et al., CVPR 2022): diffusion in autoencoder latent space + cross-attention conditioning
*Script (60 s):* SynSur is our reference pipeline. We do not replicate it; we test a reduced CPU version of the central idea.

## Slide 4 – Methodology
Pipeline diagram (Section 9 of the report): prepare → generate → filter → train → evaluate.
- Configs A full real, B 30/class, C B+synthetic, D synthetic only, E B+classical augmentation (control)
- Same real test set for all
*Script (60 s):* Config E matters: if plain augmentation gives the same gain, diffusion adds nothing beyond cheap tricks.

## Slide 5 – Dataset and generative model
- NEU surface defects: 6 classes, 1,800 gray 200×200 images, 70/15/15 stratified split
- Pretrained Stable Diffusion (LDM), CPU, image-to-image from the 30-per-class real subset, one text prompt per class
- No LoRA fine-tuning (CPU), no masks, no VLM prompts
*Script (60 s):* Be explicit about what we did NOT do compared with SynSur.

## Slide 6 – Synthetic examples
Figure: `figures/gallery_real_vs_synthetic.png`; `figures/filter_summary.png`.
- [N_GEN] generated, [N_KEPT] kept after filtering; main reject reason: [from `results/filter_report.csv`]
*Script (75 s):* Point out one convincing and one poor class, honestly.

## Slide 7 – Classification experiments
- Features: [HOG / ResNet-18] → RBF-SVM; hyper-parameters chosen on real validation set
- Metrics: accuracy, macro P/R/F1, per-class F1, confusion matrices; bootstrap 95% CI (2,000 resamples)
*Script (45 s):* Test set has only [N_TEST] images, so we always report intervals.

## Slide 8 – Results
Figures: `figures/accuracy_comparison.png`, `figures/f1_comparison.png`, `figures/confusion_matrices.png`.
- Table from `results/metrics_summary.csv`; paired differences from `results/paired_comparisons.csv`
- State: "C vs B: [difference, CI]" and "C vs E: [difference, CI]". Say "not distinguishable" if the CI includes 0.
*Script (120 s):* Read the numbers as they are. A null result is still a valid finding.

## Slide 9 – Limitations and ethics
- Small test set, one split, one limited subset; no fine-tuning; low resolution; filter favours easy samples
- Synthetic images must be labelled; never in test sets; risk of missed defects; model licence and bias
*Script (60 s).*

## Slide 10 – Conclusion
- Built a resumable CPU pipeline; measured effect: [one sentence from report findings]
- Next: LoRA on GPU, inpainting for localised defects, DreamSim/CLIPScore, multiple seeds
*Script (30 s):* Thank you; questions.

## Live demonstration plan (3 min, uses saved outputs only)
1. Show `data/synthetic/raw/manifest.csv` (prompt, seed, source image per generated file).
2. Open `figures/gallery_real_vs_synthetic.png` and `results/filter_report.csv` (kept vs rejected, reasons).
3. Open `results/metrics_summary.csv` and `figures/accuracy_comparison.png`.
4. Run `python main.py` in the terminal: every stage prints `[skip] ... (already done)`, showing resumability.
5. If asked to generate live: `python -m src.generate_synthetic --per-class 1` (takes minutes on CPU; have it pre-run).
