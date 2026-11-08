import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class DatasetScan(Dataset):
    def __init__(self, path_to_file: str, path_to_dir: str, patch_size: int = 512, probability:float =0.3):
        self.df = pd.read_csv(path_to_file, header=0)
        self.path_to_dir = path_to_dir
        self.patch_size = patch_size
        self.probability = probability
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        row = self.df.iloc[item]
        img = np.array(Image.open(row['full']), dtype=np.float32) / 65535.0
        mask = (np.array(Image.open(row['mask']), dtype=np.float32) > 0).astype(np.float32)

        mask_y, mask_x = np.where(mask > 0)
        h, w = mask.shape
        if len(mask_y) > 0 and np.randon.rand() < self.probability:
            idx_mask = np.randon.randint(0, len(mask_y))
            temp_y, temp_x = mask_y[idx_mask], mask_x[idx_mask]
        else:
            temp_y, temp_x = np.randon.randint(0, h), np.randon.randint(0, w)

        y = max(0, min(temp_y - self.patch_size//2, h - self.patch_size))
        x = max(0, min(temp_x - self.patch_size // 2, w - self.patch_size))
        patch_img = img[y: y + self.patch_size, x: x + self.patch_size]
        patch_mask = mask[y: y + self.patch_size, x: x + self.patch_size]

        tensor_img = torch.from_numpy(patch_img).unsqueeze(0)
        tensor_mask = torch.from_numpy(patch_mask).unsqueeze(0)
        return tensor_img, tensor_mask