"""
Prepare data for PE uncertainty models
- pridict-v1
"""

import os 
import sys
import pdb

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

import numpy as np
import pandas as pd

from . import data_utils


class PE_Dataset(pl.LightningDataModule):

    """
    Args:
        "data": "pridict-v1",
        "vocab_char_dict": "ACGTN",
        "train_data_path": "../../data/pridict_data/pridict-90k-cleaned_train.csv",
        "test_data_path": "../../data/pridict_data/pridict-90k-cleaned_test.csv",
        "batch_size": 128,
        "log_dir": "../../logs/",
        "val_split": 0.1,
        "pegrna_length": 100
    """


    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config

        self.data = data_config["data"]
        self.vocab_char_dict = data_config["vocab_char_dict"]
        self.train_data_path = data_config["train_data_path"]
        self.test_data_path = data_config["test_data_path"]
        self.batch_size = data_config["batch_size"]
        self.log_dir = data_config["log_dir"]
        self.val_split = data_config["val_split"]
        self.pegrna_length = data_config["pegrna_length"]

        print(f"loading training data from {self.train_data_path}")
        self.train_data = pd.read_csv(self.train_data_path)
        print(f"loading testing data from {self.test_data_path}")
        self.test_data = pd.read_csv(self.test_data_path)

        self._prepare_data()
        self._setup()


    def train_dataloader(self, shuffle_bool = True):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = shuffle_bool)

    def val_dataloader(self, shuffle_bool = False):
        return DataLoader(self.val_data, batch_size = self.batch_size, shuffle = shuffle_bool)

    def test_dataloader(self, shuffle_bool = False):
        return DataLoader(self.test_data, batch_size = self.batch_size, shuffle = shuffle_bool)
    
    def load_vocab(self):
        pass
    
    def _prepare_data(self):

        train_df = data_utils.read_data(self.train_data_path)
        test_df = data_utils.read_data(self.test_data_path)

        # get data columns


        pass

    def _setup(self):
        pass



