"""Load the FashionMNIST dataset with TorchVision.

Creates two datasets:
  * a training/validation dataset (the FashionMNIST train split), and
  * a test dataset (the FashionMNIST test split).

The training/validation dataset is then split randomly (seed 42) into
55,000 training samples and 5,000 validation samples.
"""

import torch
from torch.utils.data import random_split
import torchvision
from torchvision import transforms

# Convert the PIL images to tensors.
transform = transforms.ToTensor()

# Training/validation dataset (60,000 images).
train_val_dataset = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# Test dataset (10,000 images).
test_dataset = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# Split the 60,000 train/val images into 55,000 train and 5,000 validation,
# using a fixed seed of 42 for reproducibility.
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    train_val_dataset,
    [55_000, 5_000],
    generator=generator,
)

if __name__ == "__main__":
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
