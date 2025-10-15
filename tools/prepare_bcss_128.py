"""
Script to prepare BCSS dataset with 128x128 patches.
Based on the Proto2Seg paper implementation.
"""

import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse


def extract_patches(image, mask, patch_size=128, stride=128):
    """
    Extract non-overlapping patches from image and mask.
    
    Args:
        image: PIL Image
        mask: PIL Image
        patch_size: Size of the patch (default: 128)
        stride: Stride for patch extraction (default: 128, non-overlapping)
    
    Returns:
        List of tuples (image_patch, mask_patch, x, y)
    """
    patches = []
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    h, w = img_array.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = img_array[y:y+patch_size, x:x+patch_size]
            mask_patch = mask_array[y:y+patch_size, x:x+patch_size]
            
            # Filter out patches that are mostly background (class 0)
            # Keep patches with at least 10% non-background pixels
            non_bg_ratio = (mask_patch > 0).sum() / (patch_size * patch_size)
            if non_bg_ratio >= 0.1:
                patches.append((img_patch, mask_patch, x, y))
    
    return patches


def prepare_bcss_dataset(
    source_rgb_dir,
    source_mask_dir,
    output_dir,
    patch_size=128,
    stride=128,
    train_ratio=0.8,
    seed=42
):
    """
    Prepare BCSS dataset by extracting patches.
    
    Args:
        source_rgb_dir: Directory containing RGB images
        source_mask_dir: Directory containing mask images
        output_dir: Output directory for patches
        patch_size: Size of patches (default: 128)
        stride: Stride for extraction (default: 128)
        train_ratio: Ratio of training samples (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
    """
    np.random.seed(seed)
    
    # Create output directories
    patch_dir = os.path.join(output_dir, 'patches')
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Get all RGB image files
    rgb_files = sorted([f for f in os.listdir(source_rgb_dir) if f.endswith('.png')])
    
    print(f"Found {len(rgb_files)} RGB images")
    
    # Randomly split files into train/test
    np.random.shuffle(rgb_files)
    split_idx = int(len(rgb_files) * train_ratio)
    train_files = rgb_files[:split_idx]
    test_files = rgb_files[split_idx:]
    
    print(f"Train images: {len(train_files)}, Test images: {len(test_files)}")
    
    # Store patch information
    patch_data = []
    
    # Process training files
    print("\nProcessing training images...")
    for rgb_file in tqdm(train_files):
        # Load RGB and mask
        rgb_path = os.path.join(source_rgb_dir, rgb_file)
        mask_path = os.path.join(source_mask_dir, rgb_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {rgb_file}, skipping...")
            continue
        
        rgb_img = Image.open(rgb_path)
        mask_img = Image.open(mask_path)
        
        # Extract patches
        patches = extract_patches(rgb_img, mask_img, patch_size, stride)
        
        # Save patches
        base_name = os.path.splitext(rgb_file)[0]
        for idx, (img_patch, mask_patch, x, y) in enumerate(patches):
            patch_name = f"{base_name}_patch_{idx:04d}_x{x}_y{y}.png"
            
            # Save image patch
            img_patch_path = os.path.join(patch_dir, patch_name)
            Image.fromarray(img_patch).save(img_patch_path)
            
            # Save mask patch
            mask_patch_path = os.path.join(mask_dir, patch_name)
            Image.fromarray(mask_patch).save(mask_patch_path)
            
            # Store information
            patch_data.append({
                'patch_path': img_patch_path,
                'mask_path': mask_patch_path,
                'split': 'train',
                'original_image': rgb_file,
                'x': x,
                'y': y
            })
    
    # Process test files
    print("\nProcessing test images...")
    for rgb_file in tqdm(test_files):
        # Load RGB and mask
        rgb_path = os.path.join(source_rgb_dir, rgb_file)
        mask_path = os.path.join(source_mask_dir, rgb_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {rgb_file}, skipping...")
            continue
        
        rgb_img = Image.open(rgb_path)
        mask_img = Image.open(mask_path)
        
        # Extract patches
        patches = extract_patches(rgb_img, mask_img, patch_size, stride)
        
        # Save patches
        base_name = os.path.splitext(rgb_file)[0]
        for idx, (img_patch, mask_patch, x, y) in enumerate(patches):
            patch_name = f"{base_name}_patch_{idx:04d}_x{x}_y{y}.png"
            
            # Save image patch
            img_patch_path = os.path.join(patch_dir, patch_name)
            Image.fromarray(img_patch).save(img_patch_path)
            
            # Save mask patch
            mask_patch_path = os.path.join(mask_dir, patch_name)
            Image.fromarray(mask_patch).save(mask_patch_path)
            
            # Store information
            patch_data.append({
                'patch_path': img_patch_path,
                'mask_path': mask_patch_path,
                'split': 'test',
                'original_image': rgb_file,
                'x': x,
                'y': y
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(patch_data)
    csv_path = os.path.join(output_dir, 'BCSS_128.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nDataset preparation complete!")
    print(f"Total patches: {len(df)}")
    print(f"Train patches: {len(df[df['split']=='train'])}")
    print(f"Test patches: {len(df[df['split']=='test'])}")
    print(f"CSV saved to: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare BCSS dataset with 128x128 patches')
    parser.add_argument('--source-rgb-dir', type=str, 
                        default='./datasets/bcss/rgbs_colorNormalized',
                        help='Directory containing source RGB images')
    parser.add_argument('--source-mask-dir', type=str,
                        default='./datasets/bcss/masks',
                        help='Directory containing source mask images')
    parser.add_argument('--output-dir', type=str,
                        default='./data/BCSS_128',
                        help='Output directory for patches')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='Patch size (default: 128)')
    parser.add_argument('--stride', type=int, default=128,
                        help='Stride for patch extraction (default: 128, non-overlapping)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of training samples (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_bcss_dataset(
        source_rgb_dir=args.source_rgb_dir,
        source_mask_dir=args.source_mask_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

