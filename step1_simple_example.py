"""
Step 1: Simple SVM Example
Understanding how SVM works with small fake data!
"""

import numpy as np

print("=" * 60)
print("Step 1: SVM Basics - Understanding Loss and Gradient")
print("=" * 60)

print("\n[Step 1] Data Preparation")
print("-" * 60)

# Assume we have 3 images (only 4 features instead of 3072 pixels for simplicity)
X = np.array([
    [1.0, 2.0, 3.0, 1.0],  # Image 0 (last 1.0 is bias)
    [4.0, 5.0, 6.0, 1.0],  # Image 1
    [7.0, 8.0, 9.0, 1.0],  # Image 2
])

# Ground truth labels (3 classes: 0, 1, 2)
y = np.array([0, 1, 2])

# Weight matrix (randomly initialized with small values)
# Shape: (num_features=4) x (num_classes=3)
np.random.seed(42)
W = np.random.randn(4, 3) * 0.01  # Small random values from normal distribution

print(f"Image data X shape: {X.shape} (3 images, 4 features each)")
print(f"Ground truth labels y: {y}")
print(f"Weight W shape: {W.shape} (4 features × 3 classes)")
print(f"\nWeight W:\n{W}")

print("\n[Step 2] Score Calculation (X @ W)")
print("-" * 60)

scores = X @ W  # Matrix multiplication
print(f"Score matrix:\n{scores}")
print("\nEach row = [Class0 score, Class1 score, Class2 score] for each image")

print("\n[Step 3] SVM Loss Calculation")
print("-" * 60)

delta = 1.0  # Margin (safety distance)
num_train = X.shape[0] #3
num_classes = W.shape[1]

loss = 0.0
dW = np.zeros_like(W)  # Array to store gradient

print(f"Margin (delta): {delta}\n")

for i in range(num_train):
    print(f"Image {i} (correct: class {y[i]})")
    print(f"  Scores: {scores[i]}")
    
    correct_class_score = scores[i, y[i]]
    print(f"  Correct class {y[i]} score: {correct_class_score:.4f}")
    
    for j in range(num_classes):
        if j == y[i]:  # Skip correct class
            continue
        
        # Hinge Loss formula: max(0, wrong_score - correct_score + margin)
        margin = scores[i, j] - correct_class_score + delta
        
        if margin > 0:
            loss += margin
            print(f"  Class {j}: {scores[i, j]:.4f} - {correct_class_score:.4f} + {delta} = {margin:.4f} ❌ Loss incurred!")
            
            # Compute gradient
            dW[:, j] += X[i]       # Wrong class: need to decrease score
            dW[:, y[i]] -= X[i]    # Correct class: need to increase score
        else:
            print(f"  Class {j}: {scores[i, j]:.4f} - {correct_class_score:.4f} + {delta} = {margin:.4f} ✅ OK")
    
    print()

# Calculate average loss
loss /= num_train
dW /= num_train

print(f"Total Loss (average): {loss:.4f}")
print("\nHigher loss means worse model!")

# ========================================
# 4. Check Gradient
# ========================================
print("\n[Step 4] Gradient Check")
print("-" * 60)
print("Gradient dW:")
print(dW)
print("\nPositive(+): This weight increases loss → need to decrease")
print("Negative(-): This weight decreases loss → need to increase")

print("\n[Step 5] Weight Update (Gradient Descent)")
print("-" * 60)

learning_rate = 0.2  # Learning rate
print(f"Learning rate: {learning_rate}")

print("\nBefore update W (partial):")
print(W[:, 0])  # Weights for class 0

# Gradient Descent!
W_new = W - learning_rate * dW

print("\nAfter update W (partial):")
print(W_new[:, 0])  # Weights for class 0

print("\n[Step 6] Recalculate Scores After Update")
print("-" * 60)

scores_new = X @ W_new
print("New scores:")
print(scores_new)

# Calculate new loss
loss_new = 0.0
for i in range(num_train):
    correct_class_score = scores_new[i, y[i]]
    for j in range(num_classes):
        if j == y[i]:
            continue
        margin = scores_new[i, j] - correct_class_score + delta
        if margin > 0:
            loss_new += margin

loss_new /= num_train

print(f"\nPrevious Loss: {loss:.4f}")
print(f"New Loss: {loss_new:.4f}")
print(f"Improvement: {loss - loss_new:.4f} {'✅ Better!' if loss_new < loss else '❌ Worse!'}")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("1. Score = X @ W (image × weights)")
print("2. Loss = penalty if correct score is not sufficiently higher than others")
print("3. Gradient = direction to reduce loss")
print("4. W = W - learning_rate × gradient (weight update)")
print("5. Repeating this process gradually reduces loss!")
print("=" * 60)
