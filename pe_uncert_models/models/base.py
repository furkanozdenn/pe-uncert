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

        # data, *targets = batch
        '''data contains multiple modalities.


        TODO: handle this in shared step instead of data module

        Dataloader:
        batch[0] -> sgrna
        batch[1] -> sgrna same for reconstruction
        batch[2] -> target
        batch[3] -> activity

        '''

        # data is now batch[0] and batch[2]
        data = batch[0], batch[2], batch[3] # sgrna, target, activity

        # targets is now batch[1] and batch[3]
        targets = batch[1], batch[3]

        preds = self(data)
        return preds, targets

    def forward(self, batch):
        raise NotImplementedError

    def relabel(self, loss_dict, label):
        loss_dict = {label + str(key): val for key,val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        preds, targets = self.shared_step(batch)
        assert len(preds) == len(targets) +1, f"Preds: {len(preds)}, Targets: {len(targets)}"

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
        assert len(preds) == len(targets) + 1, f"Preds: {len(preds)}, Targets: {len(targets)}"

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
