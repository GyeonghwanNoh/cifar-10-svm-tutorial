import numpy as np
import matplotlib.pyplot as plt
from step2_load_cifar10 import download_cifar10, preprocess_data
from step3_svm_loss import svm_loss


class LinearSVM:
    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate=1e-7, reg=2.5e4, num_iters=1500, 
              batch_size=200, verbose=True):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        
        loss_history = []
        
        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            loss, grad = svm_loss(X_batch, y_batch, self.W, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad
            
            if verbose and it % 100 == 0:
                print(f'Iteration {it}/{num_iters}: loss {loss:.4f}')
        
        return loss_history
    
    def predict(self, X):
        return np.argmax(X.dot(self.W), axis=1)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == '__main__':
    # Load and preprocess data
    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = download_cifar10()
    x_train, y_train, x_test, y_test, _ = preprocess_data(
        x_train, y_train, x_test, y_test, num_train=5000, num_test=500
    )
    
    # Train SVM
    print("Training SVM...\n")
    svm = LinearSVM()
    loss_history = svm.train(x_train, y_train, learning_rate=1e-7, reg=2.5e4, 
                             num_iters=1500, batch_size=200, verbose=True)
    
    # Evaluate
    print(f"\nTrain accuracy: {svm.accuracy(x_train, y_train):.2%}")
    print(f"Test accuracy: {svm.accuracy(x_test, y_test):.2%}")
    
    # Save loss curve
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved!")
