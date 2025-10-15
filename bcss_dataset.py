"""
BCSS Dataset Class for Proto2Seg
Breast Cancer Semantic Segmentation Dataset with 128x128 patches

This module provides a PyTorch Dataset class for loading BCSS patches
for contrastive learning, clustering, and segmentation tasks.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple, Callable


class BCSSDataset(Dataset):
    """
    BCSS Dataset for histopathology image segmentation.
    
    Loads 128x128 patches from the BCSS dataset with corresponding masks.
    Supports train/test splits and various data augmentation strategies.
    
    Args:
        csv_path: Path to the CSV file containing patch information
        split: Data split to use ('train', 'test', or 'all')
        transform: Optional transform to apply to images
        mask_transform: Optional transform to apply to masks
        return_mask: Whether to return masks (default: False)
        label_mode: How to generate labels ('class', 'dominant', 'distribution')
            - 'class': Returns single dominant class (for classification)
            - 'dominant': Returns dominant class only if >80% of pixels
            - 'distribution': Returns class distribution (for prototype learning)
        
    Label Mapping:
        0: Background/Outside ROI (zero weight in training)
        1: Tumor
        2: Stroma
        3: Lymphocytic infiltrate
        4: Necrosis or debris
    """
    
    # Class names for reference
    CLASS_NAMES = [
        'background',
        'tumor',
        'stroma',
        'lymphocytic_infiltrate',
        'necrosis_debris'
    ]
    
    # Label mapping from original BCSS to 5 classes
    LABEL_MAPPING = {
        0: 0,   # outside_roi -> background
        1: 1,   # tumor -> tumor
        2: 2,   # stroma -> stroma
        3: 3,   # lymphocytic_infiltrate -> lymphocytic
        4: 4,   # necrosis_or_debris -> necrosis
        5: 0,   # glandular_secretions -> background
        6: 0,   # blood -> background
        7: 0,   # exclude -> background
        8: 0,   # metaplasia_NOS -> background
        9: 0,   # fat -> background
        10: 3,  # plasma_cells -> lymphocytic
        11: 3,  # other_immune_infiltrate -> lymphocytic
        12: 0,  # mucoid_material -> background
        13: 0,  # normal_acinus_or_duct -> background
        14: 0,  # lymphatics -> background
        15: 0,  # undetermined -> background
        16: 0,  # nerve -> background
        17: 0,  # skin_adnexa -> background
        18: 0,  # blood_vessel -> background
        19: 1,  # angioinvasion -> tumor
        20: 1,  # dcis -> tumor
        21: 0,  # other -> background
    }
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_mask: bool = False,
        label_mode: str = 'dominant'
    ):
        super().__init__()
        
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Filter by split
        if split != 'all':
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}'")
        
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_mask = return_mask
        self.label_mode = label_mode
        
        print(f"BCSSDataset initialized: {len(self.df)} samples ({split} split)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _map_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Map original BCSS labels to 5-class labels.
        
        Args:
            mask: Original mask with values 0-21
            
        Returns:
            Mapped mask with values 0-4
        """
        mapped_mask = np.zeros_like(mask)
        for orig_label, new_label in self.LABEL_MAPPING.items():
            mapped_mask[mask == orig_label] = new_label
        return mapped_mask
    
    def _get_label_from_mask(self, mask: np.ndarray) -> int:
        """
        Generate label from mask based on label_mode.
        
        Args:
            mask: Mask array with mapped labels (0-4)
            
        Returns:
            Label integer
        """
        # Compute class distribution
        target = np.array([(mask == i).sum() for i in range(5)])
        target = target / (target.sum() + 1e-8)
        
        if self.label_mode == 'class':
            # Always return dominant class
            return int(np.argmax(target))
        
        elif self.label_mode == 'dominant':
            # Return dominant class only if >80%, otherwise return 5 (uncertain)
            if target.max() > 0.8:
                return int(np.argmax(target))
            else:
                return 5  # Uncertain/mixed class
        
        elif self.label_mode == 'distribution':
            # Return the distribution as tensor (for contrastive learning)
            return torch.from_numpy(target).float()
        
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item by index.
        
        Returns:
            If return_mask is False:
                (image, label) or just image (if label_mode is None)
            If return_mask is True:
                (image, mask, label)
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['patch_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        
        # Load and process mask
        mask_path = row['mask_path']
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # Map mask to 5 classes
        mask_array = self._map_mask(mask_array)
        
        # Generate label
        label = self._get_label_from_mask(mask_array)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mask_transform is not None:
            mask_pil = Image.fromarray(mask_array.astype(np.uint8))
            mask_array = self.mask_transform(mask_pil)
        else:
            mask_array = torch.from_numpy(mask_array).long()
        
        # Return based on settings
        if self.return_mask:
            return image, mask_array, label
        else:
            return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Compute the distribution of dominant classes in the dataset.
        
        Returns:
            Dictionary with class counts
        """
        class_counts = {i: 0 for i in range(6)}  # 0-4 plus 5 for uncertain
        
        for idx in range(len(self)):
            row = self.df.iloc[idx]
            mask_path = row['mask_path']
            mask = np.array(Image.open(mask_path))
            mask = self._map_mask(mask)
            label = self._get_label_from_mask(mask)
            
            if isinstance(label, torch.Tensor):
                label = int(torch.argmax(label))
            
            class_counts[label] += 1
        
        return class_counts
    
    def get_sample_info(self, idx: int) -> dict:
        """
        Get detailed information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        row = self.df.iloc[idx]
        
        mask = np.array(Image.open(row['mask_path']))
        mask = self._map_mask(mask)
        
        unique, counts = np.unique(mask, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        
        return {
            'index': idx,
            'patch_path': row['patch_path'],
            'mask_path': row['mask_path'],
            'split': row['split'],
            'original_image': row['original_image'],
            'position': (int(row['x']), int(row['y'])),
            'class_distribution': class_distribution,
            'dominant_class': int(np.argmax([(mask == i).sum() for i in range(5)])),
        }


def get_bcss_transforms(
    image_size: int = 128,
    augment: bool = True,
    normalize: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get standard transforms for BCSS dataset.
    
    Args:
        image_size: Size of the images (default: 128)
        augment: Whether to apply data augmentation
        normalize: Whether to normalize images
        
    Returns:
        (image_transform, mask_transform)
    """
    image_transforms = []
    
    if augment:
        image_transforms.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    else:
        image_transforms.append(transforms.Resize((image_size, image_size)))
    
    image_transforms.append(transforms.ToTensor())
    
    if normalize:
        # ImageNet normalization (standard for pretrained models)
        image_transforms.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    image_transform = transforms.Compose(image_transforms)
    
    # Mask transform (no augmentation, just resize if needed)
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return image_transform, mask_transform


def create_bcss_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 128,
    return_mask: bool = False,
    label_mode: str = 'dominant',
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders for BCSS dataset.
    
    Args:
        csv_path: Path to the CSV file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        image_size: Size of images
        return_mask: Whether to return masks
        label_mode: Label generation mode
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        (train_loader, test_loader)
    """
    # Get transforms
    train_transform, _ = get_bcss_transforms(image_size, augment=True, normalize=True)
    test_transform, _ = get_bcss_transforms(image_size, augment=False, normalize=True)
    
    # Create datasets
    train_dataset = BCSSDataset(
        csv_path=csv_path,
        split='train',
        transform=train_transform,
        return_mask=return_mask,
        label_mode=label_mode
    )
    
    test_dataset = BCSSDataset(
        csv_path=csv_path,
        split='test',
        transform=test_transform,
        return_mask=return_mask,
        label_mode=label_mode
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        **kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        **kwargs
    )
    
    return train_loader, test_loader


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("BCSS Dataset Module Test")
    print("=" * 60)
    
    # Test dataset creation
    csv_path = './data/BCSS_128/BCSS_128.csv'
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        print("Please run tools/prepare_bcss_128.py first")
        exit(1)
    
    print("\n1. Creating dataset with dominant label mode...")
    dataset = BCSSDataset(
        csv_path=csv_path,
        split='train',
        return_mask=True,
        label_mode='dominant'
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Test loading a sample
    print("\n2. Loading sample...")
    image, mask, label = dataset[0]
    print(f"   Image type: {type(image)}, shape: {image.size if hasattr(image, 'size') else 'PIL Image'}")
    print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"   Label: {label} ({dataset.CLASS_NAMES[label] if label < 5 else 'uncertain'})")
    
    # Get sample info
    print("\n3. Getting sample info...")
    info = dataset.get_sample_info(0)
    print(f"   Original image: {info['original_image']}")
    print(f"   Position: {info['position']}")
    print(f"   Dominant class: {dataset.CLASS_NAMES[info['dominant_class']]}")
    print(f"   Class distribution: {info['class_distribution']}")
    
    # Create dataloaders
    print("\n4. Creating dataloaders...")
    train_transform, _ = get_bcss_transforms(image_size=128, augment=True)
    test_transform, _ = get_bcss_transforms(image_size=128, augment=False)
    
    train_dataset = BCSSDataset(
        csv_path=csv_path,
        split='train',
        transform=train_transform,
        return_mask=False,
        label_mode='dominant'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    print(f"   Train loader batches: {len(train_loader)}")
    
    # Load a batch
    print("\n5. Loading a batch...")
    for batch_images, batch_labels in train_loader:
        print(f"   Batch images shape: {batch_images.shape}")
        print(f"   Batch labels shape: {batch_labels.shape}")
        print(f"   Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
        print(f"   Labels: {batch_labels.tolist()[:10]}...")
        break
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

