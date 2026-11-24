# AI4009 – Synthetic Defect Generation with Latent Diffusion (NEU, CPU-only)

Question tested: does adding filtered, diffusion-generated images to a 30-per-class real training set change
classification performance on a held-out **real** test set? Five configurations (A–E) are compared with bootstrap CIs.

## Folder structure
```
ai4009_project/
├── main.py                  # runs all stages, resumable
├── requirements.txt
├── README.md
├── report_template.md       # filled from real results by src/make_report.py
├── slides.md                # slide text, speaker script, demo plan
├── src/
│   ├── common.py  features.py
│   ├── prepare_dataset.py   # split 70/15/15 + 30-per-class subset
│   ├── generate_synthetic.py# Stable Diffusion img2img (CPU) | classical-augment control
│   ├── filter_images.py     # quality + class-consistency filtering
│   ├── train_classifier.py  # configs A–E, HOG/ResNet-18 + SVM
│   ├── evaluate.py          # metrics, confusion matrices, bootstrap CIs
│   ├── visualize.py         # graphs + real-vs-synthetic gallery
│   └── make_report.py       # report/report.md + report/report.html
├── data/raw/                # PUT NEU IMAGES HERE
├── results/  figures/  report/
```

## 1. Install (Windows/macOS/Linux, Python 3.10+)
```
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install numpy pandas scikit-learn scikit-image Pillow matplotlib seaborn tqdm markdown
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors
```

## 2. Dataset (NEU surface defect database)
Get the 1,800 images (six classes, 300 each; files like `crazing_1.jpg`, `scratches_17.jpg`) from one of:
- Kaggle "NEU surface defect database" (needs a free Kaggle login; the project can also try
  `python -m src.prepare_dataset --try-kaggle` if the `kaggle` CLI and `~/.kaggle/kaggle.json` are set up – the dataset slug
  `kaustubhdikshit/neu-surface-defect-database` is from memory, not verified);
- the NEU-DET page on IEEE DataPort, or the original authors' page (Kechen Song, Northeastern University, China).

Unzip anywhere under `data/raw/`. Sub-folders do not matter; the class is read from the file name.

## 3. Run
```
python main.py
```
First run downloads Stable Diffusion (~4 GB). CPU time for 6 classes × 20 images at 256 px is roughly tens of minutes to
a couple of hours depending on the laptop. Quick test first: `python main.py --per-class 2`, then `python main.py --force`.

Useful options: `--per-class 10`, `--steps 20`, `--strength 0.5`, `--features resnet18`, `--from-stage train`, `--force`.
Re-running `python main.py` skips finished stages; generation also resumes image-by-image.

## If the diffusion model will not load
Model ids on Hugging Face change; the default is `stable-diffusion-v1-5/stable-diffusion-v1-5` (not verified from this
environment). Try another Stable Diffusion 1.x repo id via `--model <id>`. As a last resort
`python main.py --backend augment` runs the pipeline with **classical augmentation, which is NOT diffusion**; the code tags
these images accordingly and the generated report prints a warning. Do not present those results as diffusion results.

## Outputs
`results/metrics_summary.csv`, `per_class_f1.csv`, `confusion_*.csv`, `paired_comparisons.csv`, `filter_report.csv`;
`figures/*.png`; `report/report.html` (open in browser → select all → paste into Word → export PDF).

## Honest scope
No LoRA fine-tuning, no inpainting masks, no VLM prompts (all used by SynSur). Synthetic images start from the 30-per-class
real images only; validation/test images are never used for generation or filtering. Check all citations before submitting.
