<div align="center">

# 🔬 Retinal Vessel Segmentation · DRIVE Dataset

### MONAI Hackathon — End-to-End Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MONAI](https://img.shields.io/badge/MONAI-Latest-00BFFF?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgo=&logoColor=white)](https://monai.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.50.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Detecting retinal blood vessels using UNet, Attention UNet and DeepLabV3+ — with Grad-CAM explainability, Monte Carlo Dropout uncertainty and a live Gradio demo.**

</div>

---

## 📌 What This Project Does

Retinal vessel segmentation is a critical step in diagnosing **diabetic retinopathy**, **glaucoma** and **hypertensive retinopathy**. This project builds a complete clinical-grade segmentation pipeline on the [DRIVE dataset](https://kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) — from raw fundus images to explainable AI predictions — using **MONAI** as the medical imaging backbone.

---

## 🏆 Results at a Glance

| Model | Dice ↑ | Precision ↑ | Recall ↑ | Specificity ↑ | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---|---|---|---|---|
| **Baseline UNet** ⭐ | **0.6828** | 0.5763 | 0.8461 | 0.9415 | **0.9477** | **0.8024** |
| Baseline UNet + PP | 0.6741 | 0.5651 | 0.8437 | 0.9390 | 0.9477 | 0.8024 |
| Attention UNet | 0.4535 | 0.3374 | 0.7074 | 0.8676 | 0.8574 | 0.3102 |
| Attention UNet + PP | 0.4480 | 0.3353 | 0.6899 | 0.8697 | 0.8574 | 0.3102 |
| DeepLabV3+ | ~0.35 | — | — | — | — | — |
| DeepLabV3+ + TTA | ~0.33 | — | — | — | — | — |

> ⭐ **Best model: Baseline UNet** — highest Dice (0.6828) and ROC-AUC (0.9477) on DRIVE test set  
> PP = Morphological Post-Processing | TTA = Test-Time Augmentation (8 views)

---

## 🗂️ Project Structure

```
retinal-vessel-segmentation-monai/
│
├── DRIVE_ML.ipynb              # Main notebook — all 6 tasks end-to-end
├── README.md                   # This file
├── INTERVIEW_PREP.md           # Technical Q&A
```

---

## ⚙️ Pipeline Overview

```
Raw DRIVE Images (565×584)
        │
        ▼
┌─────────────────────────┐
│   MONAI Transforms      │  Resize → Normalize → RandFlip → RandRotate90
│   Augmentation Pipeline │  RandZoom → GaussianNoise → ToTensor
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Model Training        │  Baseline UNet (660K params)
│   Early Stop + LR Sched │  Attention UNet | DeepLabV3+ (26.7M params)
│   Dice + BCE Loss       │  Adam · CosineAnnealingLR · 30 epochs
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Post-Processing       │  Morphological Closing (r=1)
│                         │  Connected Component Filter (min=50px)
│                         │  Morphological Opening (r=1)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Evaluation            │  Dice · Precision · Recall · F1
│   8 Metrics             │  Specificity · HD95 · ROC-AUC · PR-AUC
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Innovations           │  Grad-CAM Explainability
│                         │  Monte Carlo Dropout Uncertainty (20 passes)
│                         │  Test-Time Augmentation (8 views)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Gradio Web App        │  Upload → Predict → Overlay
│   Live Demo             │  Model toggle · Post-processing toggle
└─────────────────────────┘
```

---

## 🧠 Models

### 1. Baseline UNet
- Built with `monai.networks.nets.UNet`
- `channels=(16, 32, 64, 128, 256)` · `strides=(2,2,2,2)` · `num_res_units=2`
- Instance Normalisation · **660,281 parameters**
- Loss: `DiceLoss(sigmoid=True, squared_pred=True)`

### 2. Attention UNet
- Built with `monai.networks.nets.AttentionUnet`
- Attention gates on every skip connection — suppresses background noise
- Same spatial config as Baseline · suited for class-imbalanced vessel data

### 3. DeepLabV3+ *(Innovation)*
- `segmentation_models_pytorch` · ResNet50 ImageNet backbone
- **Atrous Spatial Pyramid Pooling (ASPP)** — dilation rates 6, 12, 18
- **26.7M parameters** · captures multi-scale vessel features simultaneously

---

## 💡 Innovations

### 🔍 Grad-CAM Explainability
Visualises *which pixels* the model attends to during prediction. Target layer: `encoder.layer4[-1].conv3` of ResNet50. Confirms the model focuses on actual vessel regions — critical for clinical trust.

### 🎲 Monte Carlo Dropout Uncertainty
Runs **20 stochastic forward passes** with dropout active at inference. Pixel-wise standard deviation = uncertainty map. Flags vessel boundaries and thin capillaries as high-uncertainty regions for human review.

### 🔄 Test-Time Augmentation (TTA)
Runs inference on **8 augmented views** (flips, rotations, transpose) per image and averages probability maps. Reduces orientation bias and boundary uncertainty for free — no retraining needed.

### 🎛️ Interactive Gradio App
- Model toggle: Baseline UNet ↔ Attention UNet
- Post-processing ON/OFF with adjustable closing radius
- Overlay alpha slider for visual clarity
- Shows: original · ground truth · raw prediction · post-processed · probability heatmap · overlay

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Medical AI Framework | MONAI |
| Deep Learning | PyTorch (ROCm/AMD GPU) |
| Advanced Models | segmentation-models-pytorch |
| Post-Processing | scikit-image (morphology) |
| Explainability | Grad-CAM (manual implementation) |
| Web App | Gradio 5.50.0 |
| Data Handling | NumPy, Pandas, Pillow |
| Visualisation | Matplotlib, Seaborn |

---

## 🚀 How to Run

### Option 1 — Google Colab *(Recommended)*
```
1. Open DRIVE_ML.ipynb in Google Colab
2. Set runtime → GPU (T4 or A100)
3. Run all cells sequentially
4. Gradio app launches at the end with a public link
```

### Option 2 — Local (AMD GPU / ROCm)
```bash
pip install monai torch torchvision segmentation-models-pytorch gradio scikit-image matplotlib
jupyter notebook DRIVE_ML.ipynb
```

---

## 📊 Key Observations

- **Baseline UNet outperforms Attention UNet** on this dataset — limited training data (20 images) favours simpler architectures with fewer parameters to optimise
- **Post-processing marginally reduces Dice** for Baseline UNet because morphological operations occasionally erode thin true-positive vessel segments
- **Attention UNet shows high Recall (0.7074) but low Precision (0.3374)** — it detects most vessels but over-predicts in peripheral low-contrast regions
- **DeepLabV3+ underperforms** due to insufficient data to fine-tune 26.7M parameters — would benefit significantly from a larger training set
- **Grad-CAM confirms** the model attends to vessel regions rather than retinal landmarks or image artefacts

---

<div align="center">
<i>Built with MONAI · PyTorch · ROCm · Gradio</i>
</div>
