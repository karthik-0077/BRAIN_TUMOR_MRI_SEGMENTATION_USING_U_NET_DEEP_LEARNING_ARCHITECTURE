# Brain Tumor MRI Segmentation Using Deep Learning

This project uses deep learning to automatically segment brain tumors from MRI images. It implements and evaluates two semantic-segmentation models:

- **Attention U-Net**
- **Lightweight DeepLabV3-like model**

The Attention U-Net achieved the strongest performance, reaching an **overall Dice score of 0.8562** and an **overall IoU of 0.7486** on the test set.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Features](#main-features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Loss Function and Metrics](#loss-function-and-metrics)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Installation](#installation)
- [How to Run the Project](#how-to-run-the-project)
- [Generated Outputs](#generated-outputs)
- [Recommended Repository Structure](#recommended-repository-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Medical Disclaimer](#medical-disclaimer)

---

## Project Overview

Brain tumor segmentation is the process of identifying the exact tumor region in a brain MRI scan. Manual segmentation can be time-consuming and may vary between specialists.

This project builds a binary segmentation system that receives an MRI image and produces a mask containing:

- `0` for background pixels
- `1` for tumor pixels

The complete workflow includes:

1. Pairing MRI images with their corresponding tumor masks.
2. Resizing and normalizing images.
3. Creating stratified training, validation, and test sets.
4. Applying image augmentation.
5. Training an Attention U-Net.
6. Fine-tuning the best Attention U-Net checkpoint.
7. Training a lightweight DeepLabV3-like comparison model.
8. Evaluating both models using Dice, IoU, precision, recall, F1-score, and pixel accuracy.
9. Saving models, metrics, training curves, confusion matrices, and sample predictions.

---

## Main Features

- Automatic image-and-mask pairing using `_mask.tif` filenames.
- MRI images resized to `256 × 256`.
- Binary mask preservation using nearest-neighbor interpolation.
- Stratified train, validation, and test splitting.
- Training-set balancing based on tumor-positive and empty-mask slices.
- On-the-fly data augmentation with TensorFlow.
- Attention gates in U-Net skip connections.
- Batch normalization, spatial dropout, and L2 regularization.
- Combined Binary Cross-Entropy and Focal Tversky loss.
- AdamW optimization with cosine learning-rate decay.
- Fine-tuning from the best saved Attention U-Net checkpoint.
- Comparison with a lightweight DeepLabV3-like architecture.
- Automatic checkpointing and early stopping.
- Pixel-level confusion matrix and classification report.
- Visualization of ground truth masks, predictions, and MRI overlays.

---

## Dataset

The notebook expects a brain MRI segmentation dataset stored inside a folder named `kaggle_3m`.

It found:

- **3,929 paired MRI images and masks**
- **2,514 training samples**
- **629 validation samples**
- **786 test samples**
- **275 tumor-positive samples in the test set**

Each MRI image must have a matching mask using the following naming format:

```text
image_name.tif
image_name_mask.tif
```

An example dataset structure is:

```text
kaggle_3m/
├── patient_folder_1/
│   ├── image_1.tif
│   ├── image_1_mask.tif
│   ├── image_2.tif
│   └── image_2_mask.tif
├── patient_folder_2/
│   ├── image_1.tif
│   └── image_1_mask.tif
└── ...
```

> The dataset itself is not included in this repository. Download it separately and update `dataset_path` in the notebook.

---

## Data Preprocessing

### MRI Images

Each MRI image is:

1. Read with OpenCV.
2. Converted from BGR to RGB.
3. Resized to `256 × 256`.
4. Normalized from the range `0–255` to `0–1`.
5. Converted to `float32`.

### Segmentation Masks

Each mask is:

1. Read in grayscale.
2. Resized using nearest-neighbor interpolation.
3. Converted into a binary mask.
4. Expanded to the shape `(256, 256, 1)`.

Nearest-neighbor interpolation is important for masks because it preserves the original class boundaries and avoids creating incorrect intermediate pixel values.

### Dataset Split

The dataset is divided using stratified splitting so that tumor-positive and tumor-negative MRI slices are represented across all subsets.

```text
Training:   2,514 images
Validation:   629 images
Testing:      786 images
```

The training-balancing logic keeps all tumor-positive images and limits tumor-negative images to a maximum ratio of approximately `2:1`.

### Data Augmentation

The training pipeline applies:

- Random horizontal flipping
- Random vertical flipping
- Random rotations in multiples of 90 degrees
- Random brightness adjustment
- Random contrast adjustment

Geometric transformations are applied to both the MRI image and its mask. Brightness and contrast changes are applied only to the MRI image.

---

## Model Architectures

## 1. Attention U-Net

The main model is an Attention U-Net with an encoder-decoder architecture.

### Encoder

The encoder uses convolution blocks with:

- Two `3 × 3` convolution layers
- Batch normalization
- ReLU activation
- Max pooling
- L2 regularization
- Spatial dropout in deeper layers

The encoder filter sizes are:

```text
32 → 64 → 128 → 256
```

The bridge uses:

```text
512 filters
```

### Decoder

The decoder uses:

- Transposed convolution for upsampling
- Attention gates on skip connections
- Feature concatenation
- Convolution blocks

The decoder filter sizes are:

```text
256 → 128 → 64 → 32
```

The final layer uses a `1 × 1` convolution with sigmoid activation to create a binary segmentation probability map.

### Attention Gates

Attention gates help the model focus on useful tumor-related features from encoder skip connections while reducing irrelevant background information.

---

## 2. Lightweight DeepLabV3-like Model

A lightweight DeepLabV3-like model is included for comparison.

It contains:

- A convolutional encoder
- Three max-pooling stages
- An Atrous Spatial Pyramid Pooling-style block
- Dilated convolutions with rates `1`, `2`, `4`, and `6`
- Bilinear upsampling
- A sigmoid segmentation output

The model has approximately:

```text
718,001 total parameters
```

---

## Loss Function and Metrics

### Combined Loss

The project uses a custom loss function:

```text
Combined Loss = 0.5 × Binary Cross-Entropy
              + 0.5 × Focal Tversky Loss
```

Focal Tversky loss helps handle severe class imbalance because tumor pixels occupy only a small portion of most MRI images.

The loss uses:

```text
alpha = 0.3
beta  = 0.7
gamma = 0.75
```

A larger `beta` gives a higher penalty to false negatives, encouraging the model to avoid missing tumor pixels.

### Evaluation Metrics

The following metrics are calculated:

- Pixel accuracy
- Dice coefficient
- Intersection over Union
- Precision
- Recall
- F1-score
- Confusion matrix

Dice and IoU are more informative than accuracy for this project because the background class contains far more pixels than the tumor class.

---

## Experimental Setup

### General Configuration

```text
Image size:       256 × 256
Batch size:       16
Prediction limit: 0.5
Random seed:      42
```

### Attention U-Net Training

```text
Optimizer:             AdamW
Initial learning rate: 1e-3
Minimum learning rate: 1e-5
Weight decay:          1e-5
Maximum epochs:        100
Early-stopping patience: 15
```

A cosine-decay learning-rate schedule is used during the initial training stage.

### Attention U-Net Fine-Tuning

```text
Optimizer:             AdamW
Learning rate:         1e-4
Weight decay:          1e-5
Maximum epochs:        40
Early-stopping patience: 25
```

### DeepLabV3-like Training

```text
Optimizer:             Adam
Learning rate:         1e-5
Maximum epochs:        60
Early-stopping patience: 12
```

`ReduceLROnPlateau` is used to reduce the learning rate when validation loss stops improving.

---

## Results

### Final Model Comparison

| Model | Test Accuracy | Overall Dice | Overall IoU | Tumor-positive Dice | Tumor-positive IoU | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Attention U-Net** | **0.9971** | **0.8562** | **0.7486** | **0.8655** | **0.7629** | **0.8410** | 0.8721 | **0.8562** |
| DeepLabV3-like | 0.9953 | 0.7993 | 0.6656 | 0.8180 | 0.6921 | 0.7003 | **0.9307** | 0.7993 |

### Best Model

The **Attention U-Net** produced the best overall segmentation results.

It achieved:

```text
Test Accuracy:       0.9971
Overall Dice:        0.8562
Overall IoU:         0.7486
Tumor-positive Dice: 0.8655
Tumor-positive IoU:  0.7629
Precision:           0.8410
Recall:              0.8721
F1-score:            0.8562
```

### Attention U-Net Confusion Matrix

```text
                    Predicted
                 Background    Tumor
Actual Background  50,912,057   84,818
Actual Tumor            65,819  448,602
```

### Train, Validation, and Test Performance

| Dataset | Accuracy | Dice | IoU | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0.9971 | 0.8633 | 0.7595 | 0.8417 | 0.8861 | 0.8633 |
| Validation | 0.9969 | 0.8538 | 0.7448 | 0.8182 | 0.8926 | 0.8538 |
| Test | 0.9971 | 0.8562 | 0.7486 | 0.8410 | 0.8721 | 0.8562 |

The similar train, validation, and test scores suggest that the Attention U-Net generalizes reasonably well on this dataset.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY-NAME` with your GitHub details.

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Required Libraries

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn jupyter
```

### Main Dependencies

```text
Python
TensorFlow / Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook or Google Colab
```

A GPU-enabled environment is recommended for model training.

---

## Limitations

- The dataset contains a strong class imbalance between background and tumor pixels.
- Pixel accuracy can appear very high because most pixels belong to the background.
- The current model performs binary segmentation only.
- The project does not classify tumor type or grade.
- The results come from one dataset split using a fixed random seed.
- The notebook loads the full dataset into memory.
- Patient-level data leakage should be checked if slices from the same patient can appear in different subsets.
- The system has not been clinically validated.
- MRI scans from other hospitals, scanners, or acquisition protocols may produce different results.

---

## Future Improvements

Possible improvements include:

- Patient-level train, validation, and test splitting.
- K-fold cross-validation.
- Mixed-precision training.
- Patch-based training for larger MRI images.
- Pretrained encoders such as EfficientNet or ResNet.
- U-Net++, ResUNet, TransUNet, or transformer-based segmentation models.
- Threshold optimization using the validation set.
- Connected-component post-processing.
- Data augmentation with elastic deformation.
- Model explainability and uncertainty estimation.
- Multi-class tumor-region segmentation.
- Deployment through Streamlit, Flask, FastAPI, or TensorFlow Serving.

---

## Medical Disclaimer

This project is intended for **educational and research purposes only**. It is not a certified medical device and must not be used as a replacement for diagnosis, treatment decisions, or evaluation by qualified healthcare professionals.
