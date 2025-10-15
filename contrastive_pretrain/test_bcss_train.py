"""
Test script to verify BCSS dataset setup for training
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import BCSSDataset
from modules import transform
import torch

print("=" * 60)
print("Testing BCSS Dataset for Contrastive Learning")
print("=" * 60)

# Create BCSS dataset
print("\n1. Creating BCSS dataset...")
dataset = BCSSDataset(
    csv_path='../data/BCSS_128/BCSS_128.csv',
    split='train',
    transform=None,
    return_mask=False,
    label_mode='dominant'
)
print(f"   Dataset size: {len(dataset)}")

# Test getting a sample
print("\n2. Loading a sample...")
image, label = dataset[0]
print(f"   Image: {type(image)}, size: {image.size}")
print(f"   Label: {label}")

# Create contrastive transform
print("\n3. Creating contrastive transform...")
train_transform = transform.Transforms(size=128, blur=True)
print("   Transform created")

# Test transform
print("\n4. Testing transform...")
x_i, x_j = train_transform(image)
print(f"   Augmented view 1: {x_i.shape}, range: [{x_i.min():.3f}, {x_i.max():.3f}]")
print(f"   Augmented view 2: {x_j.shape}, range: [{x_j.min():.3f}, {x_j.max():.3f}]")

# Create wrapped dataset
print("\n5. Creating contrastive dataset wrapper...")
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, bcss_dataset, transform):
        self.bcss_dataset = bcss_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.bcss_dataset)
    
    def __getitem__(self, idx):
        image, label = self.bcss_dataset[idx]
        x_i, x_j = self.transform(image)
        return (x_i, x_j), label

contrastive_dataset = ContrastiveDataset(dataset, train_transform)
print(f"   Contrastive dataset size: {len(contrastive_dataset)}")

# Test loading from contrastive dataset
print("\n6. Loading from contrastive dataset...")
(x_i, x_j), label = contrastive_dataset[0]
print(f"   Views: {x_i.shape}, {x_j.shape}")
print(f"   Label: {label}")

# Create dataloader
print("\n7. Creating dataloader...")
dataloader = torch.utils.data.DataLoader(
    contrastive_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=0
)
print(f"   Dataloader batches: {len(dataloader)}")

# Load a batch
print("\n8. Loading a batch...")
for (batch_i, batch_j), batch_labels in dataloader:
    print(f"   Batch view 1: {batch_i.shape}")
    print(f"   Batch view 2: {batch_j.shape}")
    print(f"   Batch labels: {batch_labels.shape}, samples: {batch_labels[:5]}")
    break

print("\n" + "=" * 60)
print("✅ All tests passed! Ready for training.")
print("=" * 60)

