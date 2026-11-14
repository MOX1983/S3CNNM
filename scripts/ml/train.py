import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from dataset import DatasetScan
from unet import UNetModel
from LossMet import LossUnet
from ..load_yml import config

def train_model():
    dataset = DatasetScan(config.to_save_dir + "\\" + config.name_new_csv, config.to_save_dir)