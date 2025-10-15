# BCSS Dataset Module Documentation

## Overview

The `datasets` package provides a comprehensive PyTorch Dataset class for working with the BCSS (Breast Cancer Semantic Segmentation) 128×128 patch dataset. This module is designed to support all stages of the Proto2Seg pipeline: contrastive learning, clustering, coarse segmentation, and refinement.

## Files

- **`datasets/bcss_dataset.py`**: Main dataset module with `BCSSDataset` class and utilities
- **`datasets/__init__.py`**: Package initialization with clean imports
- **`examples_bcss_usage.py`**: Comprehensive usage examples for different scenarios

## Quick Start

### Basic Usage

```python
from datasets import BCSSDataset

# Create dataset
dataset = BCSSDataset(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    split='train',
    return_mask=True,
    label_mode='dominant'
)

# Load a sample
image, mask, label = dataset[0]
```

### Using Dataloaders

```python
from datasets import create_bcss_dataloaders

# Create train and test loaders
train_loader, test_loader = create_bcss_dataloaders(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    batch_size=32,
    num_workers=4,
    return_mask=False,
    label_mode='dominant'
)

# Use in training loop
for images, labels in train_loader:
    # Your training code here
    pass
```

## BCSSDataset Class

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | str | required | Path to CSV file with patch information |
| `split` | str | `'train'` | Data split: 'train', 'test', or 'all' |
| `transform` | callable | `None` | Transform to apply to images |
| `mask_transform` | callable | `None` | Transform to apply to masks |
| `return_mask` | bool | `False` | Whether to return masks |
| `label_mode` | str | `'dominant'` | Label generation mode (see below) |

### Label Modes

The `label_mode` parameter controls how labels are generated from masks:

#### 1. `'class'` - Classification Mode
Returns the dominant class (most frequent class in the patch).

```python
dataset = BCSSDataset(..., label_mode='class')
image, label = dataset[0]
# label: int (0-4)
```

**Use case**: Simple classification tasks

#### 2. `'dominant'` - Confident Classification Mode
Returns the dominant class only if it occupies >80% of the patch, otherwise returns 5 (uncertain/mixed).

```python
dataset = BCSSDataset(..., label_mode='dominant')
image, label = dataset[0]
# label: int (0-5, where 5 = uncertain)
```

**Use case**: Proto2Seg contrastive learning (filters mixed patches)

#### 3. `'distribution'` - Probability Distribution Mode
Returns the probability distribution over all 5 classes.

```python
dataset = BCSSDataset(..., label_mode='distribution')
image, distribution = dataset[0]
# distribution: tensor([0.0, 0.85, 0.10, 0.05, 0.0])
```

**Use case**: Soft-label training, prototype learning

### Label Mapping

The dataset maps original BCSS labels (22 classes) to 5 tissue classes:

| Class ID | Class Name | Original BCSS Labels |
|----------|------------|---------------------|
| 0 | Background/Outside ROI | 0, 5-9, 12-18, 21 |
| 1 | Tumor | 1, 19, 20 |
| 2 | Stroma | 2 |
| 3 | Lymphocytic Infiltrate | 3, 10, 11 |
| 4 | Necrosis/Debris | 4 |

### Methods

#### `__len__()`
Returns the number of samples in the dataset.

#### `__getitem__(idx)`
Returns sample at index `idx`. Output depends on `return_mask` and `label_mode`:

- If `return_mask=False`: `(image, label)`
- If `return_mask=True`: `(image, mask, label)`

#### `get_class_distribution()`
Computes the distribution of dominant classes across the entire dataset.

```python
class_counts = dataset.get_class_distribution()
# Returns: {0: 1000, 1: 5000, 2: 3000, 3: 2000, 4: 500, 5: 1500}
```

#### `get_sample_info(idx)`
Returns detailed information about a specific sample.

```python
info = dataset.get_sample_info(0)
# Returns dict with: patch_path, mask_path, split, original_image, 
#                    position, class_distribution, dominant_class
```

## Utility Functions

### `get_bcss_transforms()`

Creates standard transforms for BCSS images.

```python
from datasets import get_bcss_transforms

image_transform, mask_transform = get_bcss_transforms(
    image_size=128,
    augment=True,
    normalize=True
)
```

**Parameters:**
- `image_size`: Target image size (default: 128)
- `augment`: Apply data augmentation (default: True)
- `normalize`: Apply ImageNet normalization (default: True)

**Augmentations (when enabled):**
- Random resized crop (scale 0.8-1.0)
- Random horizontal flip
- Random vertical flip
- Color jitter (brightness, contrast, saturation, hue)

### `create_bcss_dataloaders()`

Convenience function to create train and test dataloaders.

```python
from datasets import create_bcss_dataloaders

train_loader, test_loader = create_bcss_dataloaders(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    batch_size=32,
    num_workers=4,
    image_size=128,
    return_mask=False,
    label_mode='dominant'
)
```

**Parameters:**
- `csv_path`: Path to CSV file
- `batch_size`: Batch size (default: 32)
- `num_workers`: Number of workers (default: 4)
- `image_size`: Image size (default: 128)
- `return_mask`: Return masks (default: False)
- `label_mode`: Label mode (default: 'dominant')
- `**kwargs`: Additional DataLoader arguments

## Usage Examples

### Example 1: Contrastive Learning

```python
from datasets import BCSSDataset, get_bcss_transforms
from torch.utils.data import DataLoader

# Get transforms with augmentation
transform, _ = get_bcss_transforms(augment=True, normalize=True)

# Create dataset with distribution labels
dataset = BCSSDataset(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    split='train',
    transform=transform,
    return_mask=False,
    label_mode='distribution'  # Returns class probabilities
)

loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4)

# Training loop
for images, distributions in loader:
    # images: [B, 3, 128, 128]
    # distributions: [B, 5]
    features = model(images)
    loss = contrastive_loss(features, distributions)
```

### Example 2: Segmentation Training

```python
from datasets import BCSSDataset
import torchvision.transforms as transforms

# Use transforms without random crop for segmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create dataset with masks
dataset = BCSSDataset(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    split='train',
    transform=transform,
    return_mask=True,  # Return masks for segmentation
    label_mode='class'
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
for images, masks, labels in loader:
    # images: [B, 3, 128, 128]
    # masks: [B, 128, 128] with values 0-4
    predictions = segmentation_model(images)
    loss = segmentation_loss(predictions, masks)
```

### Example 3: Prototype Learning

```python
from datasets import create_bcss_dataloaders

# Create dataloaders
train_loader, test_loader = create_bcss_dataloaders(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    batch_size=128,
    num_workers=4,
    label_mode='dominant'  # Filter mixed patches
)

# Extract features for clustering
features_list = []
labels_list = []

with torch.no_grad():
    for images, labels in train_loader:
        features = encoder(images)
        features_list.append(features)
        labels_list.append(labels)

# Perform clustering on features to find prototypes
```

### Example 4: Dataset Analysis

```python
from datasets import BCSSDataset

dataset = BCSSDataset(
    csv_path='./data/BCSS_128/BCSS_128.csv',
    split='train',
    label_mode='dominant'
)

# Get sample information
info = dataset.get_sample_info(100)
print(f"Original image: {info['original_image']}")
print(f"Position: {info['position']}")
print(f"Dominant class: {dataset.CLASS_NAMES[info['dominant_class']]}")
print(f"Class distribution: {info['class_distribution']}")

# Compute overall class distribution
class_dist = dataset.get_class_distribution()
for class_id, count in class_dist.items():
    if class_id < 5:
        print(f"{dataset.CLASS_NAMES[class_id]}: {count}")
    else:
        print(f"uncertain/mixed: {count}")
```

## Testing

Run built-in tests:

```bash
# Test the dataset module
python datasets/bcss_dataset.py

# Or from project root
python -m datasets.bcss_dataset

# Run all usage examples
python examples_bcss_usage.py
```

## Integration with Proto2Seg Pipeline

### Stage 1: Contrastive Pretraining

Use `label_mode='distribution'` or `'dominant'`:

```python
dataset = BCSSDataset(..., label_mode='dominant', return_mask=False)
```

### Stage 2: Clustering

Extract features using the pretrained encoder:

```python
dataset = BCSSDataset(..., label_mode='class', return_mask=False)
```

### Stage 3: Coarse Segmentation

Use masks for prototype matching:

```python
dataset = BCSSDataset(..., return_mask=True, label_mode='class')
```

### Stage 4: Refinement

Train segmentation network with masks:

```python
dataset = BCSSDataset(..., return_mask=True, label_mode='class')
```

## Class Names Reference

```python
BCSSDataset.CLASS_NAMES = [
    'background',              # 0
    'tumor',                   # 1
    'stroma',                  # 2
    'lymphocytic_infiltrate',  # 3
    'necrosis_debris'          # 4
]
```

## Notes

1. **Background Class (0)**: Should be assigned zero weight during training as it represents regions outside the ROI.

2. **Uncertain/Mixed Patches (5)**: When using `label_mode='dominant'`, patches without a clear dominant class (>80%) are labeled as 5. Consider filtering these for contrastive learning.

3. **Data Augmentation**: Use strong augmentation for contrastive learning but minimal augmentation for segmentation to preserve spatial structure.

4. **Memory Efficiency**: Set `return_mask=False` if masks are not needed to reduce memory usage.

5. **Normalization**: The default normalization uses ImageNet statistics `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`, which is standard for pretrained models.

## Troubleshooting

### Issue: FileNotFoundError
**Solution**: Make sure the dataset has been prepared using `tools/prepare_bcss_128.py`

### Issue: Low performance
**Solution**: Increase `num_workers` in DataLoader for faster data loading

### Issue: Out of memory
**Solution**: Reduce `batch_size` or set `return_mask=False`

### Issue: Class imbalance
**Solution**: Use `get_class_distribution()` to understand class distribution and apply appropriate weighting

## See Also

- `data/BCSS_128/README.md` - Dataset documentation
- `BCSS_DATASET_SETUP.md` - Complete setup guide
- `QUICK_START.md` - Quick start guide
- `tools/prepare_bcss_128.py` - Dataset preparation script

---

**Created**: October 2025  
**Author**: Proto2Seg Implementation  
**License**: CC BY 4.0 (following BCSS dataset license)

