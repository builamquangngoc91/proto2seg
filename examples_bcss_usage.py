"""
Example Usage of BCSS Dataset

This script demonstrates various ways to use the BCSSDataset class
for different tasks in the Proto2Seg pipeline.
"""

import torch
import numpy as np

# Import from datasets package
from datasets import BCSSDataset, get_bcss_transforms, create_bcss_dataloaders


def example_1_basic_loading():
    """Example 1: Basic dataset loading"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Dataset Loading")
    print("=" * 60)
    
    # Create dataset
    dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='train',
        return_mask=True,
        label_mode='dominant'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    image, mask, label = dataset[100]
    print(f"Image: {type(image)}, {image.size if hasattr(image, 'size') else image.shape}")
    print(f"Mask: {mask.shape}, unique values: {torch.unique(mask)}")
    print(f"Label: {label} ({dataset.CLASS_NAMES[label] if label < 5 else 'uncertain'})")


def example_2_with_transforms():
    """Example 2: Using custom transforms"""
    print("\n" + "=" * 60)
    print("Example 2: Using Transforms")
    print("=" * 60)
    
    # Get transforms
    train_transform, _ = get_bcss_transforms(
        image_size=128,
        augment=True,
        normalize=True
    )
    
    # Create dataset with transforms
    dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='train',
        transform=train_transform,
        return_mask=True,
        label_mode='dominant'
    )
    
    # Load a sample
    image, mask, label = dataset[0]
    print(f"Transformed image: {image.shape}, dtype: {image.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")


def example_3_dataloaders():
    """Example 3: Creating dataloaders"""
    print("\n" + "=" * 60)
    print("Example 3: Creating DataLoaders")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, test_loader = create_bcss_dataloaders(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        batch_size=32,
        num_workers=0,
        image_size=128,
        return_mask=False,
        label_mode='dominant'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels[:10]}")
        break


def example_4_contrastive_learning():
    """Example 4: Setup for contrastive learning"""
    print("\n" + "=" * 60)
    print("Example 4: Contrastive Learning Setup")
    print("=" * 60)
    
    # Use distribution mode for contrastive learning
    dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='train',
        return_mask=False,
        label_mode='distribution'  # Returns class distribution
    )
    
    # Load a sample
    image, distribution = dataset[0]
    print(f"Image: {type(image)}")
    print(f"Class distribution: {distribution}")
    print(f"Sum: {distribution.sum()}")


def example_5_class_statistics():
    """Example 5: Get dataset statistics"""
    print("\n" + "=" * 60)
    print("Example 5: Dataset Statistics")
    print("=" * 60)
    
    dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='train',
        return_mask=False,
        label_mode='dominant'
    )
    
    # Get sample info
    info = dataset.get_sample_info(0)
    print("\nSample Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Note: Computing full class distribution takes time
    # Uncomment to run:
    # print("\nComputing class distribution (this may take a minute)...")
    # class_dist = dataset.get_class_distribution()
    # print("Class distribution:")
    # for class_id, count in class_dist.items():
    #     if class_id < 5:
    #         print(f"  {dataset.CLASS_NAMES[class_id]}: {count}")
    #     else:
    #         print(f"  uncertain/mixed: {count}")


def example_6_segmentation_training():
    """Example 6: Setup for segmentation training"""
    print("\n" + "=" * 60)
    print("Example 6: Segmentation Training Setup")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    # Get transforms without random crop (for segmentation)
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset that returns masks
    dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='train',
        transform=transform,
        return_mask=True,
        label_mode='class'  # Use 'class' for segmentation
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    # Load a batch
    for images, masks, labels in loader:
        print(f"Images: {images.shape}")
        print(f"Masks: {masks.shape}, unique values: {torch.unique(masks)}")
        print(f"Labels: {labels.shape}")
        break


def example_7_comparison_splits():
    """Example 7: Compare train and test splits"""
    print("\n" + "=" * 60)
    print("Example 7: Train vs Test Splits")
    print("=" * 60)
    
    train_dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='train',
        return_mask=False,
        label_mode='dominant'
    )
    
    test_dataset = BCSSDataset(
        csv_path='./data/BCSS_128/BCSS_128.csv',
        split='test',
        return_mask=False,
        label_mode='dominant'
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Total: {len(train_dataset) + len(test_dataset)}")
    print(f"Train ratio: {len(train_dataset) / (len(train_dataset) + len(test_dataset)):.2%}")


# Run all examples
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("BCSS Dataset Usage Examples")
    print("=" * 60)
    
    try:
        example_1_basic_loading()
        example_2_with_transforms()
        example_3_dataloaders()
        example_4_contrastive_learning()
        example_5_class_statistics()
        example_6_segmentation_training()
        example_7_comparison_splits()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

