"""
Datasets module for contrastive pretraining
"""

from .wsi_dataset import WSIDataset
from .bcss_dataset import BCSSDataset

__all__ = ['WSIDataset', 'BCSSDataset']

