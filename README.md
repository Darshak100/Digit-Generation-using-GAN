# Handwritten Digit Generation Using Deep Convolutional GANs (DCGAN)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation and analysis for the project **"Handwritten Digit Generation Using Generative Adversarial Networks"** . The project implements a Deep Convolutional Generative Adversarial Network (DCGAN) from scratch to generate realistic, high-quality handwritten digits (0-9) using the MNIST dataset.

The trained model achieves an excellent **Fréchet Inception Distance (FID) of 5.94** and produces synthetic digits that are highly effective for downstream tasks, as validated by a CNN classifier.

![Sample Generations](fig2_sample_progression.png)
*Progression of generated images from random noise (Epoch 1) to realistic digits (Epoch 100).*

## 🚀 Key Results & Quantitative Metrics

Our DCGAN model demonstrates strong performance across multiple evaluation metrics, bridging the gap between research and practical utility.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Fréchet Inception Distance (FID)** | **5.94** | Excellent. Distribution is very close to real MNIST data (target < 20). |
| **Inception Score (IS)** | 2.443 ± 0.036 | Moderate, indicating good quality and diversity. |
| **Downstream Classifier Acc.** | **87.83%** | Validates practical utility. A CNN trained *only* on our synthetic digits achieves high accuracy on the real MNIST test set. |
| **Real-Data Classifier Baseline** | 99.19% | Standard baseline for comparison. |
| **Training Epochs** | 100 | Sufficient for stable convergence on MNIST. |

## 📁 Project Structure


## 🧠 Methodology

We implement a standard DCGAN architecture following the guidelines from Radford et al. (2015). The key design choices are summarized below.

### Generator Architecture
Transforms a 100-dimensional latent noise vector `z` into a `28x28x1` grayscale image using transposed convolutions.

| Layer | Output Shape | Activation | Key Parameters |
| :--- | :--- | :--- | :--- |
| Dense | 7x7x256 | LeakyReLU | Input: 100-dim noise |
| Conv2DTranspose | 14x14x128 | LeakyReLU | 5x5, stride 2 |
| Conv2DTranspose | 28x28x64 | LeakyReLU | 5x5, stride 2 |
| Conv2DTranspose | 28x28x1 | **Tanh** | 5x5, stride 1 |

![Generator Architecture](1_okBKjQcLOpeswTyZZfvZWA.png)

### Discriminator Architecture
Classifies images as real or fake using strided convolutions for downsampling.

| Layer | Output Shape | Activation | Key Parameters |
| :--- | :--- | :--- | :--- |
| Conv2D | 14x14x64 | LeakyReLU | 5x5, stride 2, Dropout(0.3) |
| Conv2D | 7x7x128 | LeakyReLU | 5x5, stride 2, Dropout(0.3) |
| Flatten | 6272 | - | - |
| Dense | 1 | **Sigmoid** | Output: Real/Fake probability |

![Discriminator Architecture](1_UipjlvULzSCCr1szzZpKYQ.jpg)

### Training Configuration
- **Optimizer**: Adam for both networks.
- **Learning Rate**: `2e-4`, `beta_1 = 0.5`
- **Loss Function**: Binary Cross-Entropy (BCE).
- **Stabilization Techniques**:
  - Label Smoothing (Real target = 0.9)
  - Batch Normalization
  - Dropout (0.3) in the Discriminator

## 📊 Results & Visualizations

### 1. Training Dynamics
The loss curves show stable adversarial training without mode collapse or vanishing gradients. The generator and discriminator losses balance each other throughout the 100 epochs.

![Loss Curves](fig1_loss_curves.png)

### 2. Latent Space Interpolation
Linear interpolation between latent vectors produces smooth, semantically meaningful transitions, proving the model has learned a continuous and structured representation of digits.

![Latent Interpolation](fig7_latent_interpolation.png)

### 3. Feature Space (t-SNE)
The t-SNE plot shows substantial overlap between real and generated data points, confirming that synthetic digits occupy the same regions of the feature space as real MNIST digits.

![t-SNE Visualization](fig8_tsne.png)

### 4. Downstream Task Validation
A simple CNN classifier trained exclusively on 60,000 generated images achieved **87.83% accuracy** on the real MNIST test set. This is strong evidence of the high structural fidelity of our generated digits.

![Classifier Comparison](fig9_classifier_comparison.png)

### 5. Per-Class Performance
The confusion matrix reveals that the model generates all 10 digits successfully, with some expected challenges on structurally complex digits like '0', '6', and '8'.

| Digit | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| 0 | 0.95 | 0.79 | 0.86 |
| 1 | 0.97 | 0.98 | 0.97 |
| 2 | 0.80 | 0.81 | 0.80 |
| 3 | 0.93 | 0.90 | 0.91 |
| 4 | 0.76 | 0.92 | 0.84 |
| 5 | 0.78 | 0.91 | 0.84 |
| 6 | 0.89 | 0.90 | 0.89 |
| 7 | 0.97 | 0.84 | 0.90 |
| 8 | 0.93 | 0.87 | 0.90 |
| 9 | 0.85 | 0.87 | 0.86 |

![Confusion Matrix](fig10_confusion_matrix.png)

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib

### Training the Model
The core training loop alternates between updating the Discriminator and the Generator. A simplified version is shown below.

```python
# Pseudo-code for the training step
for epoch in range(EPOCHS):
    for real_images in dataset:
        # 1. Train Discriminator on real and fake images
        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        with tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)
            d_loss = discriminator_loss(real_output, fake_output)
        # 2. Train Generator
        with tf.GradientTape() as gen_tape:
            fake_images = generator(noise, training=True)
            fake_output = discriminator(fake_images, training=True)
            g_loss = generator_loss(fake_output)
        # Apply gradients...
# Generate 16 random digits
noise = tf.random.normal([16, LATENT_DIM])
generated_images = generator(noise, training=False)
