# BCSS 128x128 Dataset Setup - Complete

## Overview

The BCSS (Breast Cancer Semantic Segmentation) dataset has been successfully prepared with 128x128 patches according to the Proto2Seg paper specifications.

## Dataset Statistics

- **Total patches**: 122,489
- **Training patches**: 94,984 (80%)
- **Test patches**: 27,505 (20%)
- **Patch size**: 128 × 128 pixels
- **Dataset size**: 4.4 GB
- **Original images**: 151 TCGA breast cancer WSI ROIs

## Files Created

### 1. Dataset Files
```
data/BCSS_128/
├── patches/              # 122,489 RGB patches (128x128)
├── masks/                # 122,489 mask patches (128x128)
├── BCSS_128.csv          # Metadata file with paths and splits
└── README.md             # Dataset documentation
```

### 2. Preparation Tools
```
tools/
├── prepare_bcss_128.py   # Main preparation script
└── verify_bcss_128.py    # Verification script
```

## Dataset Preparation Process

The dataset was created by:

1. **Source Data**: Starting from 151 BCSS RGB images and corresponding masks from `datasets/bcss/`
2. **Patch Extraction**: Extracting non-overlapping 128×128 patches with stride=128
3. **Quality Filtering**: Keeping only patches with ≥10% non-background pixels
4. **Train/Test Split**: Randomly splitting original images 80/20 (seed=42)
5. **CSV Generation**: Creating metadata file with patch paths and split information

## Usage with Proto2Seg

### Step 1: Verify Configuration

The configuration file `contrastive_pretrain/config/config_bcss.yaml` is already set up:

```yaml
image_size: 128
df_list: '../data/BCSS_128/BCSS_128.csv'
batch_size: 1024
epochs: 200
```

### Step 2: Run Contrastive Pretraining

```bash
cd contrastive_pretrain
python train.py --config config/config_bcss.yaml
```

### Step 3: Prototype Identification

```bash
cd prototype_dict_building_and_coarse_segmentation
python cluster.py --config config/bcss.yaml
```

### Step 4: Coarse Segmentation

```bash
cd prototype_dict_building_and_coarse_segmentation
python coarse_seg_cluster_query.py --config config/bcss.yaml --n 5
```

### Step 5: Refinement

```bash
cd refinement
python -m torch.distributed.launch --nproc_per_node 8 train_seg.py --config configs/bcss.yaml
```

## Label Information

### Class Mapping (5 classes)

The masks use the following class mapping:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Background/Outside ROI | Don't care class (zero weight in training) |
| 1 | Tumor | Tumor tissue (includes angioinvasion, DCIS) |
| 2 | Stroma | Stromal tissue |
| 3 | Lymphocytic Infiltrate | Immune cells (includes plasma cells) |
| 4 | Necrosis/Debris | Necrotic tissue or debris |

**Important**: Class 0 should be assigned zero weight during training as it represents regions outside the ROI.

## Dataset Verification

To verify the dataset integrity:

```bash
python tools/verify_bcss_128.py
```

This will:
- Check that all patch files exist
- Verify all patches are 128×128
- Display dataset statistics
- Show sample data

## Reproducing the Dataset

If you need to recreate the dataset with different parameters:

```bash
python tools/prepare_bcss_128.py \
    --source-rgb-dir ./datasets/bcss/rgbs_colorNormalized \
    --source-mask-dir ./datasets/bcss/masks \
    --output-dir ./data/BCSS_128 \
    --patch-size 128 \
    --stride 128 \
    --train-ratio 0.8 \
    --seed 42
```

### Available Parameters

- `--patch-size`: Size of patches (default: 128)
- `--stride`: Stride for extraction (default: 128, non-overlapping)
- `--train-ratio`: Train/test split ratio (default: 0.8)
- `--seed`: Random seed for reproducibility (default: 42)

## Performance Expectations

Based on the paper, training on this dataset should achieve:

- **Contrastive Pretraining**: ~200 epochs, ~4 minutes per epoch (GPU dependent)
- **Dataset Loading**: Fast with efficient DataLoader (batch_size=1024)
- **Expected Results**: High-quality tissue prototypes for 5 classes

## Next Steps

1. ✅ Dataset preparation complete
2. ✅ Configuration files ready
3. 🔲 Run contrastive pretraining
4. 🔲 Perform clustering for prototype identification
5. 🔲 Generate coarse segmentation
6. 🔲 Train refinement network
7. 🔲 Evaluate final segmentation results

## Citation

If you use this dataset, please cite:

```bibtex
@article{amgad2019structured,
  title={Structured crowdsourcing enables convolutional segmentation of histology images},
  author={Amgad, Mohamed and Elfandy, Habiba and ... and Cooper, Lee AD},
  journal={Bioinformatics},
  year={2019},
  doi={10.1093/bioinformatics/btz083}
}

@article{pan2022human,
  title={Human-machine Interactive Tissue Prototype Learning for Label-efficient Histopathology Image Segmentation},
  author={Pan, Wentao and Yan, Jiangpeng and Chen, Hanbo and Yang, Jiawei and Xu, Zhe and Li, Xiu and Yao, Jianhua},
  booktitle={Information Processing In Medical Imaging},
  year={2023}
}
```

## License

This dataset follows the CC BY 4.0 license of the original BCSS dataset.

---

**Status**: ✅ Ready for training
**Created**: October 2025
**Tool**: `tools/prepare_bcss_128.py`

