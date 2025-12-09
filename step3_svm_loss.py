"""
Step 3: SVM Loss Function
Fully-vectorized implementation (CS231n style)
"""

import numpy as np


def svm_loss(X, y, W, reg=0.0):
    """
    Fully-vectorized SVM loss function with L2 regularization
    
    Parameters:
    - X: data (N, D), y: labels (N,), W: weights (D, C), reg: regularization
    
    Returns:
    - loss: scalar, dW: gradient (D, C)
    """
    num_train = X.shape[0]
    
    # Compute margins (hinge loss)
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train), y].reshape(-1, 1)  # Extract correct class scores
    margins = np.maximum(0, scores - correct_scores + 1.0)
    margins[np.arange(num_train), y] = 0  # Exclude correct class from loss
    
    # Loss computation: data loss + regularization loss
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)  # L2 regularization
    
    # Gradient computation
    binary = (margins > 0).astype(float)
    binary[np.arange(num_train), y] = -np.sum(binary, axis=1)
    dW = X.T.dot(binary) / num_train + 2 * reg * W
    
    return loss, dW


if __name__ == '__main__':
    print("=" * 60)
    print("Step 3: Testing SVM Loss Function")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test data (CS231n style: N samples, D features, C classes)
    X = np.random.randn(5, 10)  # 5 samples, 10 features
    y = np.array([0, 1, 2, 0, 1])  # 5 labels
    W = np.random.randn(10, 3) * 0.01  # 10 features, 3 classes
    
    print(f"\nTest data: X{X.shape}, y{y.shape}, W{W.shape}")
    
    print("\n" + "-" * 60)
    print("Without regularization")
    print("-" * 60)
    loss, dW = svm_loss(X, y, W, reg=0.0)
    print(f"Loss: {loss:.4f}")
    print(f"Gradient shape: {dW.shape}")
    
    print("\n" + "-" * 60)
    print("With regularization (reg=0.5)")
    print("-" * 60)
    loss_reg, dW_reg = svm_loss(X, y, W, reg=0.5)
    print(f"Loss: {loss_reg:.4f}")
    print(f"Regularization adds: {loss_reg - loss:.4f}")
    
    print("\n" + "=" * 60)
    print("SVM loss function validated successfully!")
    print("=" * 60)
