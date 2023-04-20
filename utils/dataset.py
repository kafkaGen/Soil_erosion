import os

import numpy as np
import torch
import cv2 as cv

from settings.config import Config


class SoilErosionDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transforms=None):
        self.subset = subset
        self.images_path = os.path.join(Config.data_path, 
                                        self.subset, 'images')
        self.masks_path = os.path.join(Config.data_path, 
                                       self.subset, 'masks')
        self.images = sorted(os.listdir(self.images_path))
        self.masks = sorted(os.listdir(self.masks_path))
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.images[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])
        
        img = cv.imread(img_path, )
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE).astype(np.float32)[:, :, np.newaxis]
        # somehow '2' appears in this binary masks
        mask = (mask != 0).astype(mask.dtype)
        
        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        
        return img, mask