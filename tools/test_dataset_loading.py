"""
Test script to verify that the BCSS 128x128 dataset can be loaded by the training code.
"""

import sys
import os
import importlib.util

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Load WSIDataset directly
current_dir = os.path.dirname(os.path.abspath(__file__))
wsi_dataset_path = os.path.join(os.path.dirname(current_dir), 
                                'contrastive_pretrain/datasets/wsi_dataset.py')

spec = importlib.util.spec_from_file_location("wsi_dataset", wsi_dataset_path)
wsi_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wsi_dataset_module)
WSIDataset = wsi_dataset_module.WSIDataset


def test_dataset_loading():
    """Test that the dataset can be loaded correctly."""
    
    print("=" * 60)
    print("Testing BCSS 128x128 Dataset Loading")
    print("=" * 60)
    
    # Define the path to the CSV
    csv_path = './data/BCSS_128/BCSS_128.csv'
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        return False
    
    print(f"\n✓ Found dataset CSV: {csv_path}")
    
    # Define transforms (similar to training)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n✓ Created data transforms")
    
    # Create dataset
    try:
        dataset = WSIDataset(df_list=csv_path, train=True, transform=transform)
        print(f"\n✓ Created WSIDataset")
        print(f"  - Total samples: {len(dataset)}")
    except Exception as e:
        print(f"\nERROR creating dataset: {e}")
        return False
    
    # Create dataloader
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        print(f"\n✓ Created DataLoader")
        print(f"  - Batch size: 16")
        print(f"  - Total batches: {len(dataloader)}")
    except Exception as e:
        print(f"\nERROR creating dataloader: {e}")
        return False
    
    # Load a few batches
    try:
        print("\n✓ Loading sample batches...")
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break
            print(f"  - Batch {i+1}: shape {batch.shape}, dtype {batch.dtype}, "
                  f"min {batch.min():.3f}, max {batch.max():.3f}")
    except Exception as e:
        print(f"\nERROR loading batches: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Dataset is ready for training.")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)

