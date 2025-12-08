"""
Step 2-2: CIFAR-10 Image Visualization
Visualize the downloaded images!
"""

import numpy as np
import matplotlib.pyplot as plt
from step2_load_cifar10 import download_cifar10, get_class_names


def show_images(images, labels, class_names, num_images=10):
    """
    Display images in a grid
    
    Parameters:
    - images: Image array (N, 32, 32, 3)
    - labels: Label array (N,)
    - class_names: List of class names
    - num_images: Number of images to display
    """
    # Calculate grid size
    rows = 2
    cols = num_images // rows
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.title(f"{class_names[labels[i]]}\n(Class {labels[i]})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_class_samples(images, labels, class_names):
    """
    Display one sample from each class
    """
    plt.figure(figsize=(15, 6))
    
    for class_id in range(10):
        # Find first image of this class
        idx = np.where(labels == class_id)[0][0]
        
        plt.subplot(2, 5, class_id + 1)
        plt.imshow(images[idx])
        plt.title(f"{class_names[class_id]}\n(Class {class_id})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_random_samples(images, labels, class_names, num_samples=16):
    """
    Display random images
    """
    # Select random indices
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    rows = 4
    cols = num_samples // rows
    
    plt.figure(figsize=(12, 10))
    
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[idx])
        plt.title(f"{class_names[labels[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("CIFAR-10 Image Visualization")
    print("=" * 60)
    
    # 1. Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test = download_cifar10()
    
    if x_train is None:
        print("❌ Cannot load data.")
        exit()
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    class_names = get_class_names()
    
    print(f"✅ Loading complete!")
    print(f"  Training images: {x_train.shape}")
    print(f"  Test images: {x_test.shape}")
    
    # 2. Show first 10 images
    print("\n" + "=" * 60)
    print("1. First 10 Images")
    print("=" * 60)
    show_images(x_train[:10], y_train[:10], class_names, num_images=10)
    
    # 3. Show one sample per class
    print("\n" + "=" * 60)
    print("2. Sample Images per Class")
    print("=" * 60)
    show_class_samples(x_train, y_train, class_names)
    
    # 4. Show 16 random images
    print("\n" + "=" * 60)
    print("3. 16 Random Samples")
    print("=" * 60)
    show_random_samples(x_train, y_train, class_names, num_samples=16)
    
    print("\n" + "=" * 60)
    print("✅ Visualization complete!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("  - Close the graph window to see the next image")
    print("  - Image size: 32×32 pixels (small!)")
    print("  - But SVM can still learn from these small images!")
