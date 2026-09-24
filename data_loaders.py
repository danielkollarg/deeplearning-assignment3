"""Create DataLoaders for the FashionMNIST datasets.

Builds three loaders from the datasets defined in ``load_data.py``:
  * ``train_loader``      -- shuffled, batch size 32
  * ``val_loader``        -- not shuffled, batch size 32
  * ``test_loader``       -- not shuffled, batch size 32

A fixed seed of 42 is used so the shuffling of the training loader is
reproducible from run to run.
"""

import torch
from torch.utils.data import DataLoader

from load_data import train_dataset, val_dataset, test_dataset

BATCH_SIZE = 32
SEED = 42

# Seed the generator that drives the training loader's shuffling so the
# batch order is reproducible.
generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

if __name__ == "__main__":
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")

    # Sanity check: inspect the shape of one training batch.
    images, labels = next(iter(train_loader))
    print(f"Image batch shape:  {tuple(images.shape)}")
    print(f"Label batch shape:  {tuple(labels.shape)}")
