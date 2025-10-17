import torch
import os
import csv
import numpy as np
from torch.utils import data
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf


class Augmentation:
    def __init__(self):
        pass

    def rotate(self, image, mask, angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-180, 180])
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        return image, mask

    def flip(self, image, mask, conf=None):
        if random.random() > 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
            if conf is not None:
                conf = tf.hflip(conf)
        if random.random() < 0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
            if conf is not None:
                conf = tf.vflip(conf)
        if conf is not None:
            return image, mask, conf
        else:
            return image, mask

    def randomCrop(self, image, mask, size=512):
        resize_size = size
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=[size, size])
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        return image, mask

    def adjustContrast(self, image, mask):
        factor = transforms.RandomRotation.get_params([0.8, 1.2])
        image = tf.adjust_contrast(image, factor)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([0.8, 1.2])
        image = tf.adjust_brightness(image, factor)
        return image, mask

    def centerCrop(self, image, mask, size=512):
        if size == None:
            size = image.size
        image = tf.center_crop(image, size)
        mask = tf.center_crop(mask, size)
        return image, mask

    def adjustSaturation(self, image, mask):
        factor = transforms.RandomRotation.get_params([0.8, 1.2])
        image = tf.adjust_saturation(image, factor)
        return image, mask

    def randomResizeCrop(self, image, mask, conf=None, scale=(0.3, 1.0), ratio=(1, 1)):
        msk = np.array(mask)
        h_image, w_image = msk.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size,
                               interpolation=transforms.InterpolationMode.NEAREST)
        if conf is not None:
            conf = tf.resized_crop(
                conf, i, j, h, w, resize_size, interpolation=transforms.InterpolationMode.NEAREST)
            return image, mask, conf
        return image, mask

    def randomGrayscale(self, image, mask, p=0.2):
        if random.random() < p:
            image = tf.rgb_to_grayscale(image, num_output_channels=3)
        return image, mask


class BCSSDataset(data.Dataset):
    """BCSS Dataset"""
    num_classes = 5

    def __init__(self,
                 split_file="",
                 dataset_path="",
                 train=True,
                 prototype_mask_folder=None,
                 return_gt=False):

        super(BCSSDataset, self).__init__()
        self.train = train
        self.split_file = split_file
        self.dataset_path = dataset_path
        self.return_gt = return_gt
        self.image_folder = "image_1024_patches_roi"
        self.gt_mask_folder = "mask_1024_patches_5label"
        self.prototype_mask_folder = prototype_mask_folder
        self.dataset_samples = []
        self.load_samples()

    def load_samples(self,):
        with open(self.split_file) as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)  # Skip header if present
            
            for row in csv_reader:
                if len(row) == 0:
                    continue
                    
                # Handle both old format (name, split) and new format (patch_path, mask_path, split, ...)
                if len(row) >= 3:
                    # New CSV format: patch_path, mask_path, split, ...
                    patch_path, mask_path, split = row[0], row[1], row[2]
                    
                    # Remove leading ./ if present
                    patch_path = patch_path.lstrip('./')
                    mask_path = mask_path.lstrip('./')
                    
                    # Build full paths
                    image_path = os.path.join(self.dataset_path, patch_path)
                    
                    if self.return_gt:
                        annotation_path = os.path.join(self.dataset_path, mask_path)
                    else:
                        # Use prototype mask if available
                        if self.prototype_mask_folder:
                            # Extract just the filename from patch_path
                            filename = os.path.basename(patch_path)
                            annotation_path = os.path.join(self.prototype_mask_folder, filename)
                        else:
                            annotation_path = os.path.join(self.dataset_path, mask_path)
                    
                    if (self.train and split == "train") or (not self.train and split == "test"):
                        self.dataset_samples.append([image_path, annotation_path])
                else:
                    # Old format: name, split
                    name, split = row[0], row[1]
                    if (self.train and split == "train") or (not self.train and split == "test"):
                        if self.return_gt:
                            self.dataset_samples.append([
                                os.path.join(self.dataset_path,
                                             self.image_folder, name),
                                os.path.join(self.dataset_path,
                                             self.gt_mask_folder, name)
                            ])
                        else:
                            self.dataset_samples.append([
                                os.path.join(self.dataset_path,
                                             self.image_folder, name),
                                os.path.join(self.dataset_path,
                                             self.prototype_mask_folder, name)
                            ])

    def __len__(self,):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        image_path, annotation_path = self.dataset_samples[index]

        img = Image.open(image_path)
        msk = Image.open(annotation_path)

        if self.train is True:
            aug = Augmentation()
            img, msk = aug.flip(img, msk)
            img, msk = aug.adjustContrast(img, msk)
            img, msk = aug.adjustBrightness(img, msk)
            img, msk = aug.adjustSaturation(img, msk)
            img, msk = aug.randomGrayscale(img, msk)
            img, msk = aug.randomResizeCrop(img, msk)

        img = tf.to_tensor(img)
        msk = np.array(msk)
        
        # Validate mask values: should be in [0, num_classes-1] or 255 (ignore index)
        # Clamp any invalid values to prevent CUDA assertion errors
        invalid_mask = (msk >= self.num_classes) & (msk != 255)
        if invalid_mask.any():
            # Set invalid values to ignore_index
            msk[invalid_mask] = 255
        
        msk = torch.from_numpy(msk).to(dtype=torch.long)

        return img, msk
