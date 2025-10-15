# Quick Start - BCSS 128x128 Dataset

## ✅ Dataset Ready

The BCSS dataset has been successfully prepared with **122,489 patches** (128×128 pixels).

## 🚀 Start Training

### Step 1: Contrastive Pretraining

```bash
cd contrastive_pretrain
python train.py --config config/config_bcss.yaml
```

**Expected**: ~200 epochs, creates tissue feature representations

### Step 2: Prototype Clustering

```bash
cd prototype_dict_building_and_coarse_segmentation
python cluster.py --config config/bcss.yaml
```

**Expected**: Identifies tissue prototypes for 5 classes

### Step 3: Coarse Segmentation

```bash
cd prototype_dict_building_and_coarse_segmentation
python coarse_seg_cluster_query.py --config config/bcss.yaml --n 5
```

**Expected**: Generates initial segmentation predictions

### Step 4: Refinement Training

```bash
cd refinement
python -m torch.distributed.launch --nproc_per_node 8 train_seg.py --config configs/bcss.yaml
```

**Expected**: Final high-quality segmentation model

### Step 5: Testing

```bash
cd refinement
python test.py --dir [path/to/log] --dataset-name bcss
```

## 📊 Dataset Info

- **Location**: `./data/BCSS_128/`
- **CSV**: `./data/BCSS_128/BCSS_128.csv`
- **Train**: 94,984 patches
- **Test**: 27,505 patches
- **Classes**: 5 (background, tumor, stroma, lymphocytic, necrosis)

## 🔍 Verification

Test the dataset loading:

```bash
python tools/test_dataset_loading.py
```

Verify dataset integrity:

```bash
python tools/verify_bcss_128.py
```

## 📖 Documentation

- **Dataset Details**: `data/BCSS_128/README.md`
- **Complete Guide**: `BCSS_DATASET_SETUP.md`
- **Paper**: `2211.14491v2.pdf`

## 🛠️ Tools

- **Prepare Dataset**: `tools/prepare_bcss_128.py`
- **Verify Dataset**: `tools/verify_bcss_128.py`
- **Test Loading**: `tools/test_dataset_loading.py`

## ⚙️ Configuration

The config is already set up in `contrastive_pretrain/config/config_bcss.yaml`:

```yaml
image_size: 128
df_list: '../data/BCSS_128/BCSS_128.csv'
batch_size: 1024
epochs: 200
```

## 📝 Notes

1. Class 0 (background) should have zero weight in loss computation
2. Uses ResNet18 for contrastive learning
3. Non-overlapping 128×128 patches with 80/20 train/test split
4. Data augmentation: random crop, horizontal/vertical flip

## 🎯 Expected Results

Following the paper methodology, you should achieve:
- Strong tissue prototypes from contrastive learning
- Accurate coarse segmentation from prototype matching
- High-quality refined segmentation (close to supervised methods)

---

**Status**: ✅ Ready for Training  
**Created**: October 2025  
**Based on**: Proto2Seg (IPMI 2023)

