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
        self.val_split = data_config["val_split"]
        self.pegrna_length = data_config["pegrna_length"]
        self.sequence_length = data_config["sequence_length"]

        print(f"loading training data from {self.train_data_path}")
        self.train_data = pd.read_csv(self.train_data_path)
        print(f"loading testing data from {self.test_data_path}")
        self.test_data = pd.read_csv(self.test_data_path)

        self._prepare_data()
        self._setup()


    def train_dataloader(self, shuffle_bool=True):
        # Setup MPS generator if needed
        generator = None
        if torch.backends.mps.is_available() and torch.device(torch.empty(1).device).type == 'mps':
            generator = torch.Generator(device='mps')
        elif torch.cuda.is_available():
            generator = torch.Generator(device='cuda')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_bool,
            generator=generator if shuffle_bool else None
        )

    def val_dataloader(self, shuffle_bool=False):
        # Setup MPS generator if needed
        generator = None
        if shuffle_bool:
            if torch.backends.mps.is_available() and torch.device(torch.empty(1).device).type == 'mps':
                generator = torch.Generator(device='mps')
            elif torch.cuda.is_available():
                generator = torch.Generator(device='cuda')
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_bool,
            generator=generator if shuffle_bool else None
        )

    def test_dataloader(self, shuffle_bool=False):
        # Setup MPS generator if needed
        generator = None
        if shuffle_bool:
            if torch.backends.mps.is_available() and torch.device(torch.empty(1).device).type == 'mps':
                generator = torch.Generator(device='mps')
            elif torch.cuda.is_available():
                generator = torch.Generator(device='cuda')
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_bool,
            generator=generator if shuffle_bool else None
        )
    
    def load_vocab(self):
        pass
    
    def _prepare_data(self):

        train_df = data_utils.read_data(self.train_data_path)
        test_df = data_utils.read_data(self.test_data_path)

        # get data columns
        initial_sequence = data_utils.dataset_to_data_columns_dict(self.data)['initial_sequence']
        mutated_sequence = data_utils.dataset_to_data_columns_dict(self.data)['mutated_sequence']
        total_read_count = data_utils.dataset_to_data_columns_dict(self.data)['total_read_count']
        edited_percentage = data_utils.dataset_to_data_columns_dict(self.data)['edited_percentage']
        unedited_percentage = data_utils.dataset_to_data_columns_dict(self.data)['unedited_percentage']
        indel_percentage = data_utils.dataset_to_data_columns_dict(self.data)['indel_percentage']
        protospacer_location = data_utils.dataset_to_data_columns_dict(self.data)['protospacer_location']
        pbs_location = data_utils.dataset_to_data_columns_dict(self.data)['pbs_location']
        rt_initial_location = data_utils.dataset_to_data_columns_dict(self.data)['rt_initial_location']
        rt_mutated_location = data_utils.dataset_to_data_columns_dict(self.data)['rt_mutated_location']



        train_initial_seq, train_mutated_seq, train_total_read_count, train_edited_percentage, train_unedited_percentage, train_indel_percentage, train_protospacer_location, train_pbs_location, train_rt_initial_location, train_rt_mutated_location = train_df[initial_sequence], train_df[mutated_sequence], train_df[total_read_count], train_df[edited_percentage], train_df[unedited_percentage], train_df[indel_percentage], train_df[protospacer_location], train_df[pbs_location], train_df[rt_initial_location], train_df[rt_mutated_location]
        train_initial_seq = data_utils.df_one_hot_encode_seq(train_initial_seq, vocab = None)
        train_mutated_seq = data_utils.df_one_hot_encode_seq(train_mutated_seq, vocab = None)

        test_initial_seq, test_mutated_seq, test_total_read_count, test_edited_percentage, test_unedited_percentage, test_indel_percentage, test_protospacer_location, test_pbs_location, test_rt_initial_location, test_rt_mutated_location = test_df[initial_sequence], test_df[mutated_sequence], test_df[total_read_count], test_df[edited_percentage], test_df[unedited_percentage], test_df[indel_percentage], test_df[protospacer_location], test_df[pbs_location], test_df[rt_initial_location], test_df[rt_mutated_location]
        test_initial_seq = data_utils.df_one_hot_encode_seq(test_initial_seq, vocab = None)
        test_mutated_seq = data_utils.df_one_hot_encode_seq(test_mutated_seq, vocab = None)

        ## for pridict-v1: protospacer_location, pbs_location, rt_initial_location, rt_mutated_location are all in the format [M,N] where max(N) is 99
        # this part assumes that for both mutated and initial sequences, the sequence legth is 99 -> self.sequence_length

        train_protospacer_location_mask = data_utils.get_binary_location_mask(train_protospacer_location, self.sequence_length)
        train_pbs_location_mask = data_utils.get_binary_location_mask(train_pbs_location, self.sequence_length)
        train_rt_initial_location_mask = data_utils.get_binary_location_mask(train_rt_initial_location, self.sequence_length)
        train_rt_mutated_location_mask = data_utils.get_binary_location_mask(train_rt_mutated_location, self.sequence_length)

        test_protospacer_location_mask = data_utils.get_binary_location_mask(test_protospacer_location, self.sequence_length)
        test_pbs_location_mask = data_utils.get_binary_location_mask(test_pbs_location, self.sequence_length)
        test_rt_initial_location_mask = data_utils.get_binary_location_mask(test_rt_initial_location, self.sequence_length)
        test_rt_mutated_location_mask = data_utils.get_binary_location_mask(test_rt_mutated_location, self.sequence_length)
        

        self.train_tuple = (train_initial_seq, train_mutated_seq, train_total_read_count, train_edited_percentage, train_unedited_percentage, train_indel_percentage, train_protospacer_location_mask, train_pbs_location_mask, train_rt_initial_location_mask, train_rt_mutated_location_mask)
        self.test_tuple = (test_initial_seq, test_mutated_seq, test_total_read_count, test_edited_percentage, test_unedited_percentage, test_indel_percentage, test_protospacer_location_mask, test_pbs_location_mask, test_rt_initial_location_mask, test_rt_mutated_location_mask)

        self.train_data_size = len(train_df)
        self.test_data_size = len(test_df)

        print(f"train data size: {self.train_data_size}")
        print(f"test data size: {self.test_data_size}")

    def _setup(self):
        
        train_initial_seq, train_mutated_seq, train_total_read_count, train_edited_percentage, train_unedited_percentage, train_indel_percentage, train_protospacer_location_mask, train_pbs_location_mask, train_rt_initial_location_mask, train_rt_mutated_location_mask = self.train_tuple
        test_initial_seq, test_mutated_seq, test_total_read_count, test_edited_percentage, test_unedited_percentage, test_indel_percentage, test_protospacer_location_mask, test_pbs_location_mask, test_rt_initial_location_mask, test_rt_mutated_location_mask = self.test_tuple

        train_initial_seq, val_initial_seq, train_mutated_seq, val_mutated_seq, train_total_read_count, val_total_read_count, train_edited_percentage, val_edited_percentage, train_unedited_percentage, val_unedited_percentage, train_indel_percentage, val_indel_percentage, train_protospacer_location_mask, val_protospacer_location_mask, train_pbs_location_mask, val_pbs_location_mask, train_rt_initial_location_mask, val_rt_initial_location_mask, train_rt_mutated_location_mask, val_rt_mutated_location_mask = train_test_split(train_initial_seq, train_mutated_seq, train_total_read_count, train_edited_percentage, train_unedited_percentage, train_indel_percentage, train_protospacer_location_mask, train_pbs_location_mask, train_rt_initial_location_mask, train_rt_mutated_location_mask, test_size = self.val_split, random_state = 42)
        
        train_split_list = [torch.tensor(np.stack(train_initial_seq)).to(torch.int64), torch.tensor(np.stack(train_mutated_seq)).to(torch.int64), torch.tensor(np.stack(train_total_read_count)).to(torch.int64), torch.tensor(np.stack(train_edited_percentage)).to(torch.float32), torch.tensor(np.stack(train_unedited_percentage)).to(torch.float32), torch.tensor(np.stack(train_indel_percentage)).to(torch.float32), torch.tensor(np.stack(train_protospacer_location_mask)).to(torch.int64), torch.tensor(np.stack(train_pbs_location_mask)).to(torch.int64), torch.tensor(np.stack(train_rt_initial_location_mask)).to(torch.int64), torch.tensor(np.stack(train_rt_mutated_location_mask)).to(torch.int64)]
        val_split_list = [torch.tensor(np.stack(val_initial_seq)).to(torch.int64), torch.tensor(np.stack(val_mutated_seq)).to(torch.int64), torch.tensor(np.stack(val_total_read_count)).to(torch.int64), torch.tensor(np.stack(val_edited_percentage)).to(torch.float32), torch.tensor(np.stack(val_unedited_percentage)).to(torch.float32), torch.tensor(np.stack(val_indel_percentage)).to(torch.float32), torch.tensor(np.stack(val_protospacer_location_mask)).to(torch.int64), torch.tensor(np.stack(val_pbs_location_mask)).to(torch.int64), torch.tensor(np.stack(val_rt_initial_location_mask)).to(torch.int64), torch.tensor(np.stack(val_rt_mutated_location_mask)).to(torch.int64)]
        test_split_list = [torch.tensor(np.stack(test_initial_seq)).to(torch.int64), torch.tensor(np.stack(test_mutated_seq)).to(torch.int64), torch.tensor(np.stack(test_total_read_count)).to(torch.int64), torch.tensor(np.stack(test_edited_percentage)).to(torch.float32), torch.tensor(np.stack(test_unedited_percentage)).to(torch.float32), torch.tensor(np.stack(test_indel_percentage)).to(torch.float32), torch.tensor(np.stack(test_protospacer_location_mask)).to(torch.int64), torch.tensor(np.stack(test_pbs_location_mask)).to(torch.int64), torch.tensor(np.stack(test_rt_initial_location_mask)).to(torch.int64), torch.tensor(np.stack(test_rt_mutated_location_mask)).to(torch.int64)]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.val_dataset = torch.utils.data.TensorDataset(*val_split_list)
        self.test_dataset = torch.utils.data.TensorDataset(*test_split_list)

