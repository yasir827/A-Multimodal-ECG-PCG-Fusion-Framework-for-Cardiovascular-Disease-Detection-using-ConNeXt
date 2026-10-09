# Multimodal ConvNeXt-Tiny Framework for Cardiovascular Disease Detection

An end-to-end deep learning pipeline for automated cardiovascular disease detection using integrated Electrocardiogram (ECG) and Phonocardiogram (PCG) signals, powered by a fine-tuned ConvNeXt-Tiny architecture.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Key Performance Metrics](#key-performance-metrics)
3. [Repository Structure & Pipeline Architecture](#repository-structure--pipeline-architecture)
4. [Dataset & Preprocessing](#dataset--preprocessing)
5. [Model Training & Hyperparameters](#model-training--hyperparameters)
6. [Getting Started & Installation](#getting-started--installation)
7. [Dataset Source](#dataset-source)

---

## 🔍 Overview
Cardiovascular diseases (CVDs) remain a leading cause of global mortality. This repository provides a complete, production-ready framework that leverages multimodal physiological signals (ECG electrical activity and PCG acoustic recordings). By converting raw biosignals into log-power mel-spectrograms via Short-Time Fourier Transform (STFT) and processing them through hierarchical depthwise convolutions, the model achieves state-of-the-art diagnostic accuracy.

---

## 📊 Key Performance Metrics
Evaluated on the PhysioNet Training-A dataset (83 independent test samples), the proposed framework demonstrates robust generalization and clinical viability:

* **Test Accuracy**: **97.59%**
* **Balanced Accuracy**: **95.83%**
* **ROC-AUC**: **0.9266**
* **Statistical Significance**: $p = 3.61 \times 10^{-22}$ (Exact One-Sided Binomial Test against 50% baseline)
* **95% Wilson Confidence Interval**: **91.63% – 99.34%**
* **Class-Specific Performance**:
  * **Normal Precision**: 1.0000 | **Recall**: 0.9167 | **F1-Score**: 0.9565
  * **Abnormal Precision**: 0.9672 | **Recall**: 1.0000 | **F1-Score**: 0.9833

### Confusion Matrix
$$\begin{bmatrix} 22 & 2 \\ 0 & 59 \end{bmatrix}$$
*(True Negatives: 22, False Positives: 2, False Negatives: 0, True Positives: 59)*

---

## 🛠️ Repository Structure & Pipeline Architecture
The codebase is structured into sequential Colab execution blocks:
* **Data Ingestion & Splitting**: Secure Google Drive mounting, 7z archive extraction, and rigorous leakage-safe 80/20 train-test partitioning.
* **Data Augmentation & Balancing**: Equalization of classes to 2,000 samples per class using stochastic augmentation strategies (Gaussian noise, amplitude gain, time shifting, and time stretching).
* **Signal Preprocessing & Feature Extraction**: Mean removal, amplitude scaling, and STFT-based log-power mel-spectrogram conversion ($128$ mel bins, $1024$ FFT window, $256$ hop length).
* **ConvNeXt-Tiny Adaptation**: Modification of the first convolutional layer to handle single-channel input features and updating the classification head for binary diagnostics.
* **Model Training & Evaluation**: Optimization via AdamW, Cosine Annealing learning rate scheduling, and Cross-Entropy loss with label smoothing ($\alpha = 0.1$).

---

## ⚙️ Dataset & Preprocessing
* **Sampling Rate ($SR$)**: 2000 Hz
* **Duration**: 20 seconds per segment (padded or truncated to 40,000 samples)
* **Input Tensor Shape**: $[1, 128, 157]$ (Channels $\times$ Mel Bands $\times$ Time Frames)

---

## 🚀 Getting Started & Installation

### Prerequisites
Ensure you have Python 3.8+ and PyTorch installed. Run the following command to install required dependencies:
```bash
pip install torch torchvision torchaudio librosa soundfile scikit-learn pandas matplotlib statsmodels scipy
