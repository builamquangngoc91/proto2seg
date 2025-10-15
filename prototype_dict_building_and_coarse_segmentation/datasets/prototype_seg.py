import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

class PrototypeSegDataset(Dataset):
    def __init__(self, df_list = '', datadir='', train=True):
        self.datadir=datadir

        all_df = pd.read_csv(df_list)
        train_df = all_df[all_df['type']=='train']
        train_df.reset_index(drop=True, inplace=True)
        test_df = all_df[all_df['type']=='valid']
        test_df.reset_index(drop=True, inplace=True)

        if train == True:
            self.list_df = train_df
        else:
            self.list_df = test_df

        self.length = self.list_df.shape[0]

    def __getitem__(self, idx):
        impth = self.datadir+'train_patchs/'+self.list_df['slide_id'][idx]+'/images/'+self.list_df['image_id'][idx]+'.png'
        mskth = self.datadir+'train_patchs/'+self.list_df['slide_id'][idx]+'/masks/'+self.list_df['image_id'][idx]+'.png'
        img = Image.open(impth)
        msk = Image.open(mskth)

        img = tf.to_tensor(img)
        msk = np.array(msk)
        msk = np.where(msk > 0, 1, msk)
        if msk.sum()/(2048*2048) > 0:
            target = 1
        else:
            target = 0 
        msk = torch.from_numpy(msk) 
        return img, msk, target, self.list_df['slide_id'][idx] + '_' + self.list_df['image_id'][idx]+'.png'

    def __len__(self):
        return self.length

class BCSS_Seg(Dataset):
    def __init__(self, df_list='', datadir="", train=True):
        self.datadir=datadir

        all_df = pd.read_csv(df_list)
        train_df = all_df[all_df['split']=='train']
        train_df.reset_index(drop=True, inplace=True)
        test_df = all_df[all_df['split']=='test']
        test_df.reset_index(drop=True, inplace=True)

        if train == True:
            self.list_df = train_df
        else:
            self.list_df = test_df

        self.length = self.list_df.shape[0]
        
        # Determine base directory for resolving relative paths
        if os.path.isabs(df_list):
            csv_dir = os.path.dirname(df_list)
        else:
            csv_dir = os.path.dirname(os.path.abspath(df_list))
        # Go up from CSV location to project root (CSV is in data/BCSS_128/)
        self.base_dir = os.path.dirname(os.path.dirname(csv_dir))

    def __getitem__(self, idx):
        # Check if using new CSV format (with full paths) or old format (just filenames)
        patch_path = self.list_df['patch_path'][idx]
        
        if 'mask_path' in self.list_df.columns:
            # New CSV format with full relative paths
            mask_path = self.list_df['mask_path'][idx]
            
            # Remove leading ./ and make absolute
            patch_path = patch_path.lstrip('./')
            mask_path = mask_path.lstrip('./')
            
            image_path = os.path.join(self.base_dir, patch_path)
            annotation_path = os.path.join(self.base_dir, mask_path)
        else:
            # Old CSV format with just filename
            image_path = os.path.join(self.datadir, "image_1024_patches_roi", patch_path)
            annotation_path = os.path.join(self.datadir, "mask_1024_patches_5label", patch_path)

        img = Image.open(image_path)
        msk = Image.open(annotation_path)

        msk = np.array(msk)
        target = np.array([(msk == i).sum() for i in range(5)])
        target = target / target.sum()

        img = tf.to_tensor(img)
        msk = torch.from_numpy(msk).to(dtype=torch.long)

        return img, msk, target, os.path.basename(image_path)

    def __len__(self):
        return self.length