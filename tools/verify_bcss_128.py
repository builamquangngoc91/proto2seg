"""
Script to verify the BCSS 128x128 dataset.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np


def verify_dataset(csv_path):
    """
    Verify the BCSS 128x128 dataset.
    
    Args:
        csv_path: Path to the CSV file
    """
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset Statistics:")
    print(f"Total patches: {len(df)}")
    print(f"Train patches: {len(df[df['split']=='train'])}")
    print(f"Test patches: {len(df[df['split']=='test'])}")
    
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    
    # Sample and verify a few patches
    print(f"\nVerifying sample patches...")
    sample_indices = np.random.choice(len(df), min(10, len(df)), replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        patch_path = row['patch_path']
        mask_path = row['mask_path']
        
        # Check if files exist
        if not os.path.exists(patch_path):
            print(f"ERROR: Patch not found: {patch_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"ERROR: Mask not found: {mask_path}")
            continue
        
        # Load and verify dimensions
        try:
            patch = Image.open(patch_path)
            mask = Image.open(mask_path)
            
            if patch.size != (128, 128):
                print(f"ERROR: Incorrect patch size {patch.size} at {patch_path}")
            if mask.size != (128, 128):
                print(f"ERROR: Incorrect mask size {mask.size} at {mask_path}")
            
            # Check mask values
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            if not all(v in range(22) for v in unique_values):
                print(f"WARNING: Unexpected mask values {unique_values} in {mask_path}")
            
        except Exception as e:
            print(f"ERROR loading {patch_path}: {e}")
    
    print(f"\nVerification complete!")
    
    # Print first few rows
    print(f"\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Count patches per original image
    print(f"\nPatches per original image (first 10):")
    patches_per_image = df.groupby('original_image').size().sort_values(ascending=False)
    print(patches_per_image.head(10))
    
    return df


if __name__ == '__main__':
    csv_path = './data/BCSS_128/BCSS_128.csv'
    verify_dataset(csv_path)

