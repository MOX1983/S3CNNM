import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class DatasetScan(Dataset):
    def __init__(self, df: pd.DataFrame, patch_size: int = 512, probability: float = 0.5, repeats: int = 4, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.probability = probability
        self.repeats = repeats
        self.train = train
    def __len__(self):
        return len(self.df) * self.repeats
    def __getitem__(self, item):
        row = self.df.iloc[item % len(self.df)]
        img = np.array(Image.open(row['full']), dtype=np.float32) / 65535.0
        mask = (np.array(Image.open(row['mask']), dtype=np.float32) > 0).astype(np.float32)

        mask_y, mask_x = np.where(mask > 0)
        h, w = mask.shape
        if len(mask_y) > 0 and np.random.rand() < self.probability:
            idx_mask = np.random.randint(0, len(mask_y))
            temp_y, temp_x = mask_y[idx_mask], mask_x[idx_mask]
        else:
            temp_y, temp_x = np.random.randint(0, h), np.random.randint(0, w)

        m = self.patch_size // 8
        off_y = np.random.randint(m, self.patch_size - m)
        off_x = np.random.randint(m, self.patch_size - m)
        y = max(0, min(temp_y - off_y, h - self.patch_size))
        x = max(0, min(temp_x - off_x, w - self.patch_size))
        patch_img = img[y: y + self.patch_size, x: x + self.patch_size]
        patch_mask = mask[y: y + self.patch_size, x: x + self.patch_size]

        if self.train:
            if np.random.rand() < 0.5:
                patch_img, patch_mask = patch_img[:, ::-1], patch_mask[:, ::-1]
            if np.random.rand() < 0.5:
                patch_img, patch_mask = patch_img[::-1], patch_mask[::-1]

        tensor_img = torch.from_numpy(np.ascontiguousarray(patch_img)).unsqueeze(0)
        tensor_mask = torch.from_numpy(np.ascontiguousarray(patch_mask)).unsqueeze(0)
        return tensor_img, tensor_mask