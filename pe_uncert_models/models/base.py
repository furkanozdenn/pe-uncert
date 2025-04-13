'''
Pytorch lightning base classes for TbtcVAE model.
'''

import torch
from torch import nn

import pytorch_lightning as pl
import argparse

import pdb


class crispAIPEBase(pl.LightningModule):
    def __init__(self, hparams):
        super(crispAIPEBase, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.lr = hparams.lr

    def configure_optimizers(self):
        opt = (torch.optim.Adam(self.parameters(), lr=self.lr))
        return opt

    def shared_step(self, batch):


        """
        batch[0] -> initial_sequence 
        batch[1] -> mutated_sequence
        batch[2] -> total_read_count
        batch[3] -> edited_percentage
        batch[4] -> unedited_percentage
        batch[5] -> indel_percentage
        batch[6] -> protospacer_location
        batch[7] -> pbs_location
        batch[8] -> rt_initial_location
        batch[9] -> rt_mutated_location
        """

        data = batch
        targets = batch[2], batch[3], batch[4], batch[5]

        preds = self(data)
        
        return preds, targets

    def forward(self, batch):
        raise NotImplementedError

    def relabel(self, loss_dict, label):
        loss_dict = {label + str(key): val for key,val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        preds, targets = self.shared_step(batch)
        # assert len(preds) == len(targets) +1, f"Preds: {len(preds)}, Targets: {len(targets)}"

        train_loss, train_loss_logs = self.loss_function(
            predictions = preds, targets = targets
        )

        train_loss_logs = self.relabel(train_loss_logs, "train_")
        

        self.log_dict(train_loss_logs, on_step = True, on_epoch = False)

        # convert dtype to double 
        train_loss = train_loss.double()
        return train_loss

    def validation_step(self, batch, batch_idx):
        preds, targets = self.shared_step(batch)
        # assert len(preds) == len(targets) + 1, f"Preds: {len(preds)}, Targets: {len(targets)}"

        val_loss, val_loss_logs = self.loss_function(
            predictions = preds, targets = targets, valid_step = True
        )

        val_loss_logs = self.relabel(val_loss_logs, "val_")

        self.log_dict(val_loss_logs, on_step = False, on_epoch = True)

        # convert dtype to double 
        val_loss = val_loss.double()
        return val_loss

    def loss_function(self, predictions, targets, valid_step = False):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_criterion(self):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

    def get_scheduler(self):
        raise NotImplementedError

    def get_dataloader(self, mode):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')
