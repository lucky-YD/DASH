# DASH
ICASSP 2026 (Submission)

## This project implements a dual-branch deep learning model (DASH), specifically designed for heart rate prediction tasks. The model extracts features from inputs through two parallel Swin Transformer branches and then fuses these features to perform final heart rate prediction.

## # DASH: Dual-Branch Multi-Layer Self-Attention Network for Heart Rate Prediction

## Core Features
- **Feature 1: 2D Time-Frequency Input**: Converts 1D physiological signals into 2D time-frequency images, preserving both temporal and frequency information for richer feature extraction
- **Feature 2: Time-Domain Windowing**: Processes long time-series signals through windowing, enhancing the model's ability to capture dynamic changes and improving temporal resolution of predictions
- **Feature 3: Dual-Branch Multi-Layer Self-Attention**: Uses parallel branches to process different features, combined with multi-layer self-attention to model long-range dependencies and improve prediction accuracy

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Size | 256*256 | Height and width of input time-frequency images |
| patch_size | 32 | Image patch size for dividing input into non-overlapping patches |
| window_size | 8 | Window size for self-attention calculations |
| embedding dimension | 4 | Feature embedding dimension |
| λ | 0.1 | Regularization parameter used during model training |

## Installation Dependencies
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint
```
