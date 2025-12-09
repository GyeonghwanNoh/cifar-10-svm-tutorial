# CIFAR-10 SVM Tutorial

Learn Support Vector Machine (SVM) from scratch using NumPy and CIFAR-10 dataset.

## Overview

This tutorial implements a linear SVM classifier for image classification on the CIFAR-10 dataset. All core components (loss function, gradient computation, weight updates) are built from scratch using NumPy—no deep learning frameworks for training!

## CIFAR-10 Dataset

CIFAR-10 contains 60,000 32×32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

<img width="1402" height="559" alt="image" src="https://github.com/user-attachments/assets/44001692-72c2-4d1a-9c29-458bce4d13b7" />


*Sample images from CIFAR-10 dataset showing different classes*

## Training Results

After training a linear SVM on 5,000 training samples for 1,500 iterations:

- **Training Accuracy: 41.24%**
- **Test Accuracy: 34.80%**
<img width="1037" height="826" alt="image" src="https://github.com/user-attachments/assets/68d645dd-72b2-4638-949e-451bb8f99cae" />
*Loss curve showing convergence during training*

### What do these results mean?

- **34.80% test accuracy** may seem low, but it's **3.5× better than random guessing (10%)**
- Linear SVM has limitations with complex image data—pixels don't capture visual patterns well
- For better results, you'd need:
  - Non-linear kernels (RBF, polynomial)
  - Feature engineering (HOG, SIFT)
  - Deep learning (CNNs) - can achieve 90%+ accuracy

## Tutorial Steps

1. **step1_simple_example.py** - Basic SVM with fake data (3 images, 4 features)
2. **step1_repeat_training.py** - Training loop over 10 epochs
3. **step2_load_cifar10.py** - Download and preprocess CIFAR-10
4. **step2_visualize.py** - Visualize CIFAR-10 images
5. **step3_svm_loss.py** - Vectorized SVM loss function 
6. **step4_train_cifar10.py** - Full training on real data

## Key Features

- ✅ Pure NumPy implementation (no PyTorch/TensorFlow for training)
- ✅ Vectorized operations for efficiency
- ✅ Stochastic Gradient Descent with mini-batches
- ✅ L2 regularization to prevent overfitting

## Requirements

```bash
pip install numpy matplotlib torch torchvision
```

## Quick Start

```bash
# Train SVM on CIFAR-10
python step4_train_cifar10.py
```

This will:
1. Download CIFAR-10 dataset (170 MB)
2. Train linear SVM for 1,500 iterations
3. Display training and test accuracy
4. Save loss curve as `loss_curve.png`

## Learning Goals

- Understand SVM loss function (hinge loss)
- Compute gradients for gradient descent
- Implement training loop from scratch
- Apply machine learning to real image data
