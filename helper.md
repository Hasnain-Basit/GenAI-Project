# AI4009 – Synthetic Data Generation for Steel Surface Defect Classification

## 1. Project Overview

This project investigates the use of diffusion-generated synthetic images to improve steel surface defect classification when real training data is limited.

The project uses the NEU Surface Defect Database (NEU-DET), containing six categories of steel surface defects:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

The implementation includes dataset preparation, synthetic image generation using Stable Diffusion v1.5, synthetic image filtering, classifier training, evaluation, and experimental comparison.

## 2. Requirements

- Windows 10/11
- Python 3.10 or compatible version supported by the dependencies
- Git
- Internet connection for downloading dependencies and the Stable Diffusion model
- Recommended: 16 GB RAM or more
- CPU supported; NVIDIA GPU is optional

**Note:** CPU-based Stable Diffusion generation can be very slow. A GPU with sufficient VRAM is recommended for faster generation.

## 3. Clone the Repository

Open Command Prompt and navigate to your desired directory.

```cmd
git clone YOUR_GITHUB_REPOSITORY_URL
cd ai4009_project
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual GitHub repository URL.

## 4. Create a Python Virtual Environment

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate
```

Upgrade pip:

```cmd
python -m pip install --upgrade pip
```

## 5. Install Dependencies

For CPU-based execution, install PyTorch:

```cmd
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```cmd
python -m pip install -r requirements.txt
```

Verify the installation:

```cmd
python -c "import torch, diffusers, transformers, sklearn, cv2; print('Dependencies installed successfully'); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 6. Dataset Setup

Download the NEU Surface Defect Database (NEU-DET) separately.

Extract the dataset into the following directory:

```text
data/raw/
```

The dataset should contain the six defect categories and their corresponding images.

The complete dataset contains 1,800 images, with 300 images per category.

**Important:** The raw dataset is not included in the GitHub repository. Users must download and place it locally before running the project.

## 7. Prepare the Dataset

From the project root, execute:

```cmd
python -m src.prepare_dataset
```

This stage prepares the dataset and creates:

- Training data
- Validation data
- Testing data
- Limited-real training subset (30 images per class)

The expected split is:

| Dataset          | Images |
| ---------------- | -----: |
| Training         |  1,260 |
| Validation       |    270 |
| Testing          |    270 |
| Limited Training |    180 |

## 8. Generate Synthetic Images

Execute:

```cmd
python -m src.generate_synthetic
```

The project uses Stable Diffusion v1.5 through the image-to-image pipeline.

On the first execution, the model is downloaded automatically from Hugging Face. The download is approximately 4.27 GB.

The configured synthetic generation target is 20 images per class, giving 120 synthetic images in total.

**Note:** The first run requires an internet connection. Subsequent runs can reuse the locally cached model.

CPU generation may take several hours.

## 9. Synthetic Image Filtering

After successful generation, execute the project's synthetic filtering module:

```cmd
python -m src.filter_synthetic
```

Verify that the filtering script exists and that its module name matches this command before execution.

## 10. Run Experiments

The project evaluates the following experimental configurations:

- **Experiment A:** Full real training dataset.
- **Experiment B:** Limited real training dataset.
- **Experiment C:** Limited real data combined with filtered synthetic data.
- **Experiment D:** Synthetic-only training.
- **Experiment E:** Limited real data with classical augmentation.

Run the experiment or training modules according to the commands implemented in the project source code.

## 11. Evaluation and Results

The project evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

Results and visualizations are generated from the actual experiment outputs.

## 12. Reproducibility Notes

- Use the same dataset split when comparing experiments.
- Keep the test set separate from training and synthetic image generation.
- Do not include validation or test images in the synthetic generation inputs.
- Preserve experiment configurations and random seeds where supported.
- Record the hardware and execution environment used for benchmarking.

## 13. Storage Notes

The Stable Diffusion model is cached locally by Hugging Face, normally under:

```text
C:\Users\YOUR_USERNAME\.cache\huggingface\hub\
```

The model is downloaded once and reused in subsequent executions.

Generated images, datasets, and experiment outputs are stored locally and are not automatically uploaded to GitHub.

## 14. Troubleshooting

### Missing Python package

```cmd
python -m pip install -r requirements.txt
```

### PyTorch not installed

```cmd
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Check Python version

```cmd
python --version
```

### Check available GPU support

```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

### Hugging Face download issues

Check your internet connection and retry the generation command. The model cache allows previously downloaded files to be reused.

---

## 15. Project Execution Order

1. Clone the repository.
2. Create and activate the virtual environment.
3. Install PyTorch and project dependencies.
4. Download and extract the NEU-DET dataset.
5. Run dataset preparation.
6. Generate synthetic images.
7. Filter generated images.
8. Run all five experiments.
9. Evaluate and compare results.
10. Generate the final report and visualizations.
