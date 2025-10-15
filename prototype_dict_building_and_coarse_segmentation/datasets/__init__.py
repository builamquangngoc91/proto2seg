"""
Datasets module for prototype building and coarse segmentation
"""

from .bcss import BCSS
from .cam import Cam16
from .prototype_seg import PrototypeSegDataset, BCSS_Seg

__all__ = ['BCSS', 'Cam16', 'PrototypeSegDataset', 'BCSS_Seg']

