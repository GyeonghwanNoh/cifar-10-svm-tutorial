"""
Step 2: Load CIFAR-10 Dataset
Download and preprocess real image data!
"""

import numpy as np
import pickle
import os


def download_cifar10():
    """
    Download CIFAR-10 dataset
    
    Automatically downloads using PyTorch!
    """
    print("Downloading CIFAR-10 dataset...")
    
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        # Define transformation to convert data to numpy array
        transform = transforms.ToTensor()
        
        # Download training data
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True, 
            transform=transform
        )
        
        # Download test data
        testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True, 
            transform=transform
        )
        
        print("✅ Download complete!")
        
        # Convert PyTorch data to numpy arrays
        x_train = []
        y_train = []
        for img, label in trainset:
            # Convert (C, H, W) → (H, W, C) and restore to 0-255 range
            img_np = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            x_train.append(img_np)
            y_train.append(label)
        
        x_test = []
        y_test = []
        for img, label in testset:
            img_np = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            x_test.append(img_np)
            y_test.append(label)
        
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(-1, 1)
        x_test = np.array(x_test)
        y_test = np.array(y_test).reshape(-1, 1)
        
        return x_train, y_train, x_test, y_test
        
    except ImportError:
        print("❌ PyTorch is not installed!")
        print("Install: pip install torch torchvision")
        return None, None, None, None


def preprocess_data(x_train, y_train, x_test, y_test, 
                    num_train=5000, num_test=500):
    """
    Preprocess data
    
    Parameters:
    - x_train, y_train: Training data
    - x_test, y_test: Test data
    - num_train: Number of training samples to use (Save memory)
    - num_test: Number of test samples to use
    """
    print("\nPreprocessing data...")
    
    # 1. Use subset only (using all data is too slow)
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]
    
    # 2. Flatten images to 1D
    #    (N, 32, 32, 3) → (N, 3072)
    x_train = x_train.reshape(num_train, -1)
    x_test = x_test.reshape(num_test, -1)
    
    # 3. Convert to float
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    
    # 4. Remove mean (important!)
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    
    # 5. Add bias (append 1 at the end)
    x_train = np.hstack([x_train, np.ones((num_train, 1))])
    x_test = np.hstack([x_test, np.ones((num_test, 1))])
    
    # 6. Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print("✅ Preprocessing complete!")
    print(f"  Training data: {x_train.shape}")
    print(f"  Training labels: {y_train.shape}")
    print(f"  Test data: {x_test.shape}")
    print(f"  Test labels: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test, mean_image


def get_class_names():
    """CIFAR-10 class names"""
    return [
        'airplane',    # 0
        'automobile',  # 1
        'bird',        # 2
        'cat',         # 3
        'deer',        # 4
        'dog',         # 5
        'frog',        # 6
        'horse',       # 7
        'ship',        # 8
        'truck'        # 9
    ]


if __name__ == '__main__':
    print("=" * 60)
    print("CIFAR-10 Data Loading Test")
    print("=" * 60)
    
    # 1. Download
    x_train, y_train, x_test, y_test = download_cifar10()
    
    if x_train is None:
        print("\n❌ Cannot load data.")
        print("Please install PyTorch: pip install torch torchvision")
        exit()
    
    # 2. Original data info
    print("\nOriginal data:")
    print(f"  Training images: {x_train.shape}")  # (50000, 32, 32, 3)
    print(f"  Training labels: {y_train.shape}")  # (50000, 1)
    print(f"  Test images: {x_test.shape}")  # (10000, 32, 32, 3)
    print(f"  Test labels: {y_test.shape}")  # (10000, 1)
    
    # 3. Preprocess
    x_train_prep, y_train_prep, x_test_prep, y_test_prep, mean_img = \
        preprocess_data(x_train, y_train, x_test, y_test, 
                       num_train=1000, num_test=200)
    
    # 4. Check class distribution
    print("\nClass distribution (Training data):")
    class_names = get_class_names()
    for i in range(10):
        count = np.sum(y_train_prep == i)
        print(f"  {i} ({class_names[i]:12s}): {count:4d} samples")
    
    # 5. Check sample data
    print("\nSample data:")
    print(f"  First image feature count: {x_train_prep.shape[1]}")
    print(f"  First image label: {y_train_prep[0]} ({class_names[y_train_prep[0]]})")
    print(f"  Pixel value range: [{x_train_prep.min():.2f}, {x_train_prep.max():.2f}]")
    
    print("\n" + "=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)
