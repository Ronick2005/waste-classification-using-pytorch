# Waste Classification using PyTorch

A deep learning project for binary classification of waste into **Organic** and **Recyclable** categories using Convolutional Neural Networks (CNN) implemented in PyTorch.

## Project Overview

This project implements an automated waste classification system that can distinguish between organic and recyclable waste materials. The solution uses a custom CNN architecture trained on a comprehensive dataset of waste images to support smart waste management and environmental sustainability.

### Key Features
- **High Accuracy**: Achieved 89.38% test accuracy
- **Binary Classification**: Organic vs Recyclable waste categorization
- **Custom CNN Architecture**: 6.9M parameters optimized for waste classification
- **Production Ready**: Includes model checkpointing and inference pipeline
- **Comprehensive Evaluation**: Detailed performance metrics and analysis

## Results Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **89.38%** |
| **Training Dataset** | 22,564 images |
| **Test Dataset** | 2,513 images |
| **Model Parameters** | 6,878,338 |
| **Training Time** | 10 epochs |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Organic** | 87.20% | 94.86% | 90.87% | 1,401 |
| **Recyclable** | 92.72% | 82.46% | 87.29% | 1,112 |
| **Weighted Avg** | 89.65% | 89.38% | 89.29% | 2,513 |

## Architecture

Our custom CNN architecture consists of:
Input (224×224×3) ↓ Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25) ↓ Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25) ↓ Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.3) ↓ Block 4: Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.3) ↓ AdaptiveAvgPool2D(7×7) ↓ Flatten → Dense(512) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(2) ↓ Output (Organic=0, Recyclable=1)

## Dataset Structure
DATASET/ ├── TRAIN/ (22,564 images) │ ├── O/ # Organic waste (food scraps, biodegradable materials) │ └── R/ # Recyclable waste (plastic, paper, metal, glass) └── TEST/ (2,513 images) ├── O/ # Organic waste test samples └── R/ # Recyclable waste test samples


## Quick Start

### Prerequisites

```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn
pip install pillow numpy pandas
```

### The Model
```bash
git clone https://github.com/Ronick2005/waste-classification-using-pytorch.git
cd waste-classification-using-pytorch

jupyter notebook waste-classification-pytorch.ipynb
```
