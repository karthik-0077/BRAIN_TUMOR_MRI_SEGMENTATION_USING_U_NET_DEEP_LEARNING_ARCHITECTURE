# Brain Tumor MRI Segmentation — Attention U-Net

Pixel-level segmentation of brain tumors (low-grade glioma) from MRI slices, using an **Attention U-Net** as the primary model and a **lightweight DeepLabV3-style ASPP network** as a comparison baseline. Both are trained and evaluated on the same data and metrics so their results are directly comparable.

Given an MRI slice, the model outputs a binary mask of the tumor region — not just "tumor present," but exactly *where*.

## Table of Contents
- [Dataset](#dataset)
- [Approach](#approach)
- [Architectures](#architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Known Issues](#known-issues)
- [License](#license)

## Dataset

[LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle, `kaggle_3m`), originally from The Cancer Genome Atlas (TCGA) low-grade glioma collection. Each MRI slice (`.tif`) is paired with a binary tumor mask (`<image>_mask.tif`). The dataset is **not included** in this repo — download it from Kaggle and update the `dataset_path` variable in the notebook.

## Approach

1. **Pairing & splitting** — images are matched to masks, labeled tumor-positive/negative, and split 64/16/20 (train/val/test) with stratification so all splits reflect the same tumor-presence ratio.
2. **Class imbalance handling** — brain MRI tumor segmentation is a heavily imbalanced problem (most slices are tumor-free, and tumor pixels are ~1–2% of a positive slice). This is addressed at two levels:
   - **Slice-level**: every tumor-positive training slice is kept; tumor-negative slices are downsampled to a 2:1 negative:positive ratio.
   - **Loss-level**: a Focal Tversky loss weights false negatives more heavily than false positives, so missing real tumor pixels is penalized harder than over-predicting.
3. **Augmentation** — random horizontal/vertical flips and 90° rotations (applied identically to image and mask to keep them aligned), plus brightness/contrast jitter (image only).
4. **Loss function** — `combo_loss = 0.5 * BCE + 0.5 * Focal Tversky` (α=0.3, β=0.7, γ=0.75), combining pixel-wise classification signal with an imbalance-aware region overlap metric.
5. **Training** — `AdamW` with cosine-decay learning rate (1e-3 → 1e-5) for up to 100 epochs, early stopping on validation Dice, followed by a fine-tuning phase at a lower fixed learning rate (1e-4).
6. **Evaluation** — Dice and IoU are reported both over all test images and over tumor-positive test images only (the latter is the clinically meaningful number, since including empty-mask slices inflates scores).

## Architectures

### Attention U-Net (primary model)
A U-Net encoder/decoder (32→64→128→256→512 filters, with batch norm, ReLU, and spatial dropout) with **attention gates** on every skip connection. Each gate learns to weight the encoder's features by relevance to the decoder's current prediction, suppressing irrelevant background before it's merged into the decoder path — rather than passing the raw skip connection through unfiltered as in a standard U-Net.

### DeepLabV3-like (comparison baseline)
A lighter encoder/decoder network using an **ASPP (Atrous Spatial Pyramid Pooling)** block — parallel dilated convolutions at rates 1/2/4/6 — to capture multi-scale context before upsampling back to full resolution.

## Results

Final metrics are computed on the held-out test set, both overall and restricted to tumor-positive slices. Fill in after running the full notebook (values are saved to `final_practical_model_comparison_table.csv`):

| Model            | Status                  | Test Accuracy | Overall Dice | Overall IoU | Tumor-positive Dice | Tumor-positive IoU | Precision | Recall | F1 |
|------------------|--------------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Attention U-Net  | Implemented and trained | | | | | | | | |
| DeepLabV3-like   | Implemented and trained | | | | | | | | |

Sample prediction grids (original MRI / ground-truth mask / predicted mask / overlay) are saved for the best-scoring tumor-positive test slices under `final_attention_unet_test_predictions/` and `deeplabv3_like_test_predictions/`.

## Project Structure

```
.
├── Brain_Tumor_MRI_Segmentation_Using_U_Net_Deep_Learning_Architecture.ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── results/                     # created at runtime
    ├── best_attention_unet.keras
    ├── best_deeplabv3_like.keras
    ├── attention_unet_history.csv
    ├── deeplabv3_like_training_history.csv
    ├── final_practical_model_comparison_table.csv
    ├── final_attention_unet_test_predictions/
    └── deeplabv3_like_test_predictions/
```

## Setup

**Windows (PowerShell or Command Prompt):**

```bat
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> If PowerShell blocks the activation script with an execution-policy error, run this once in PowerShell (as your normal user, not admin), then retry `venv\Scripts\activate`:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Download the [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) and place it locally, then update these two variables near the top of the notebook to match your paths (the notebook was originally written for Google Colab + Drive, so they currently point at `/content/drive/...`). On Windows, use either double backslashes or forward slashes in the path:

```python
dataset_path = "C:/Users/<you>/Datasets/kaggle_3m"
save_dir = "C:/Users/<you>/Datasets/results"
```

## Usage

Run the notebook top to bottom in Jupyter, JupyterLab, or Colab:

```bat
jupyter notebook Brain_Tumor_MRI_Segmentation_Using_U_Net_Deep_Learning_Architecture.ipynb
```

It will:
1. Load and pair images/masks, split into train/val/test.
2. Build the `tf.data` pipeline with augmentation.
3. Train the Attention U-Net, then fine-tune it.
4. Evaluate on the test set (Dice, IoU, precision, recall, F1, confusion matrix) and save sample predictions.
5. Build, train, and evaluate the DeepLabV3-like comparison model.
6. Save a final side-by-side comparison table.

A GPU is strongly recommended. On Windows, native GPU support for TensorFlow 2.10+ is Linux/WSL2-only — if you have an NVIDIA GPU and want to use it on Windows, either install TensorFlow inside **WSL2** (Windows Subsystem for Linux), or run the notebook on **Google Colab** instead (free GPU, no local setup). Without a GPU, training two 256×256 segmentation models on CPU will be very slow.

## Known Issues

Fixed as of the current notebook version — the training-curve cell now builds `full_history` from `history.history` + `finetune_history.history` before plotting, and the duplicate/broken cells that referenced undefined variables have been removed.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The LGG MRI Segmentation dataset has its own license/terms on Kaggle; review those separately before redistributing any data.
