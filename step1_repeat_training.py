"""
Step 1-2: Repeated Training Version
Train multiple times and watch the loss decrease!
"""

import numpy as np

print("=" * 60)
print("SVM Repeated Training - Watch Loss Decrease")
print("=" * 60)

print("\n[Step 1] Data Preparation")
print("-" * 60)

X = np.array([
    [1.0, 2.0, 3.0, 1.0],
    [4.0, 5.0, 6.0, 1.0],
    [7.0, 8.0, 9.0, 1.0],
])

y = np.array([0, 1, 2])

np.random.seed(42)
W = np.random.randn(4, 3) * 0.01

print(f"Images: {X.shape}, Labels: {y.shape}, Weights: {W.shape}")
print(f"Initial weights W:\n{W}\n")

num_epochs = 10
learning_rate = 0.012
delta = 1.0

print(f"Training configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Margin: {delta}\n")

print("=" * 60)
print("Start Training!")
print("=" * 60)

loss_history = []

for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
    print("-" * 60)

    scores = X @ W

    num_train = X.shape[0]
    num_classes = W.shape[1]

    loss = 0.0
    dW = np.zeros_like(W)

    for i in range(num_train):
        correct_class_score = scores[i, y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue

            margin = scores[i, j] - correct_class_score + delta

            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train

    loss_history.append(loss)

    W = W - learning_rate * dW

    print(f"Loss: {loss:.4f}")
    if epoch > 0:
        improvement = loss_history[epoch - 1] - loss
        print(f"Improvement: {improvement:.4f}")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)

print(f"\nFinal Loss: {loss_history[-1]:.4f}")
print(f"Initial Loss: {loss_history[0]:.4f}")
print(f"Total Improvement: {loss_history[0] - loss_history[-1]:.4f}")

print("\n" + "=" * 60)
print("Loss History (Bar Chart)")
print("=" * 60)

max_loss = max(loss_history)
for i, loss in enumerate(loss_history):
    bar_length = int((loss / max_loss) * 40)
    bar = '' * bar_length
    print(f"Epoch {i+1:2d} | {bar} {loss:.4f}")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("1. Loss decreases with each iteration")
print("2. Learning rate determines step size")
print("3. Too large learning rate  oscillation")
print("4. Too small learning rate  slow learning")
print("5. Gradient Descent works!")
print("=" * 60)
