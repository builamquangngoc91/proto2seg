# BCSS Training Guide

## Quick Start

### Train Contrastive Learning Model

```bash
cd contrastive_pretrain
python train.py --config config/config_bcss.yaml
```

## What Was Modified

### 1. Created `datasets/bcss_dataset.py`
- **Location**: `datasets/bcss_dataset.py`
- **Features**:
  - Complete PyTorch Dataset class for BCSS
  - Automatic path resolution (works from any directory)
  - 3 label modes: `'class'`, `'dominant'`, `'distribution'`
  - 5-class tissue segmentation (background, tumor, stroma, lymphocytic, necrosis)
  - Built-in label mapping from 22 original classes to 5 classes

### 2. Created `datasets/__init__.py`
- **Purpose**: Makes datasets a proper Python package
- **Usage**: `from datasets import BCSSDataset`

### 3. Modified `contrastive_pretrain/train.py`
- **Changes**:
  - Added BCSSDataset import
  - Auto-detects BCSS dataset from CSV path
  - Creates `ContrastiveDataset` wrapper for two-view augmentation
  - Sets `class_num = 5` for BCSS (vs 7 for other datasets)
  - Maintains backward compatibility with existing datasets

### 4. Configuration
- **File**: `contrastive_pretrain/config/config_bcss.yaml`
- **Key Settings**:
  ```yaml
  dataset: "wsi"
  df_list: '../data/BCSS_128/BCSS_128.csv'
  image_size: 128
  batch_size: 1024
  epochs: 200
  ```

## Training Pipeline

### Stage 1: Contrastive Pretraining (Current)

```bash
cd contrastive_pretrain
python train.py --config config/config_bcss.yaml
```

**What it does**:
- Loads BCSS 128×128 patches
- Applies contrastive learning transforms (crops, flips, color jitter, blur)
- Trains ResNet18 encoder with instance-level contrastive loss
- Saves checkpoints to `./output/Simclr_res18_bcss128/`

**Expected output**:
- Model checkpoints every 50 epochs
- Final model after 200 epochs
- Training loss should decrease over epochs

### Stage 2: Prototype Clustering

```bash
cd prototype_dict_building_and_coarse_segmentation
python cluster.py --config config/bcss.yaml
```

### Stage 3: Coarse Segmentation

```bash
cd prototype_dict_building_and_coarse_segmentation
python coarse_seg_cluster_query.py --config config/bcss.yaml --n 5
```

### Stage 4: Refinement

```bash
cd refinement
python -m torch.distributed.launch --nproc_per_node 8 train_seg.py --config configs/bcss.yaml
```

## Testing Setup

### Test Dataset Loading

```bash
# Test from project root
python datasets/bcss_dataset.py

# Test from contrastive_pretrain
cd contrastive_pretrain
python test_bcss_train.py
```

### Test Training Pipeline

```bash
# Quick test with fewer epochs
cd contrastive_pretrain
python train.py --config config/config_bcss.yaml --epochs 1
```

## Dataset Details

### Location
- **CSV**: `data/BCSS_128/BCSS_128.csv`
- **Patches**: `data/BCSS_128/patches/` (122,489 patches)
- **Masks**: `data/BCSS_128/masks/` (122,489 masks)

### Statistics
- **Total patches**: 122,489
- **Train**: 94,984 (77.5%)
- **Test**: 27,505 (22.5%)
- **Patch size**: 128 × 128 pixels
- **Dataset size**: 4.4 GB

### Classes
| ID | Class Name | Description |
|----|------------|-------------|
| 0 | Background | Outside ROI (zero weight) |
| 1 | Tumor | Tumor tissue |
| 2 | Stroma | Stromal tissue |
| 3 | Lymphocytic | Immune infiltrate |
| 4 | Necrosis | Necrotic tissue/debris |

## How It Works

### 1. BCSSDataset Class
```python
from datasets import BCSSDataset

dataset = BCSSDataset(
    csv_path='../data/BCSS_128/BCSS_128.csv',
    split='train',
    return_mask=False,  # Don't need masks for contrastive learning
    label_mode='dominant'  # Filter mixed patches
)
```

### 2. Contrastive Transform
The training script creates two augmented views of each image:
- Random resized crop
- Random horizontal/vertical flip
- Color jitter
- Random grayscale
- Gaussian blur (50% probability)

### 3. Contrastive Learning
- **Loss**: Instance-level contrastive loss (SimCLR)
- **Goal**: Learn representations where augmented views of the same image are similar
- **Output**: Pretrained encoder for downstream tasks

## Troubleshooting

### Issue: Import Error
```
ModuleNotFoundError: No module named 'datasets'
```
**Solution**: Make sure you're running from the project root or contrastive_pretrain directory

### Issue: File Not Found
```
FileNotFoundError: Image not found
```
**Solution**: The BCSSDataset now handles paths automatically. Make sure the CSV path is correct.

### Issue: CUDA Out of Memory
**Solution**: Reduce batch_size in config file:
```yaml
batch_size: 512  # or lower
```

### Issue: Slow Training
**Solution**: Increase num_workers in config:
```yaml
workers: 4  # or higher
```

## Files Reference

### Created Files
- `datasets/bcss_dataset.py` - Main dataset class
- `datasets/__init__.py` - Package initialization
- `contrastive_pretrain/test_bcss_train.py` - Test script
- `examples_bcss_usage.py` - Usage examples
- `BCSS_DATASET_MODULE.md` - Detailed documentation
- `TRAINING_GUIDE.md` - This file

### Modified Files
- `contrastive_pretrain/train.py` - Added BCSS support
- `contrastive_pretrain/datasets/wsi_dataset.py` - Fixed transform storage

### Configuration Files
- `contrastive_pretrain/config/config_bcss.yaml` - Training config
- Already configured and ready to use

## Next Steps

1. ✅ Dataset prepared (122,489 patches)
2. ✅ BCSSDataset class created
3. ✅ Training script modified
4. 🔲 **Run contrastive pretraining** (you are here)
5. 🔲 Update clustering config for 5 classes
6. 🔲 Run prototype clustering
7. 🔲 Generate coarse segmentation
8. 🔲 Train refinement network

## Expected Results

Based on the Proto2Seg paper:
- **Contrastive Learning**: Strong tissue-specific features
- **Clustering**: Clear tissue prototypes (5 classes)
- **Coarse Segmentation**: Good initial segmentation
- **Refinement**: Near-supervised performance

## Tips

1. **Monitor Training**: Check loss decreases over epochs
2. **Save Checkpoints**: Checkpoints saved every 50 epochs
3. **GPU Usage**: Training requires CUDA-capable GPU
4. **Batch Size**: Adjust based on GPU memory (default: 1024)
5. **Workers**: Set to 0 if encountering data loading issues

---

**Status**: ✅ Ready for Training  
**Last Updated**: October 2025  
**Framework**: PyTorch + Proto2Seg

