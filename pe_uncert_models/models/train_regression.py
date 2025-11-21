"""Training script for crispAIPERegression using PyTorch Lightning
"""

import datetime
import time
import os 
import sys
import json
import platform

import numpy as np
import pandas as pd

import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from pe_uncert_models.models.crispAIPE_regression import crispAIPERegression
from pe_uncert_models.data_utils.data import PE_Dataset


from scipy.stats import spearmanr

import torch
import pdb

if __name__ == '__main__':
    parser = ArgumentParser()

    # add only config file .json as a required param
    parser.add_argument('--config', type=str, required=True, help='path to config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # hparams from config file  
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']

    # add all arguments in config to parser 
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    cl_args = parser.parse_args()
    # parser = Trainer.add_argparse_args(parser) deprecated

    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_file_name = args.config.split('/')[-1].split('.')[0]
    save_dir =  os.path.join(config_training['log_dir'], config_file_name, date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    wandb_logger = WandbLogger(
        name = config_file_name,
        project = config_data['project_name'],
        log_model = True,
        save_dir = save_dir,
        offline = False,
    )

    wandb_logger.log_hyperparams(config)
    wandb_logger.experiment.log({'timestamp': date_suffix})

    early_stop_callback = EarlyStopping(
        monitor='val_loss', #TODO: set this somewhere
        patience=config_training['patience'],
        verbose=True,
        mode='min',
        min_delta=0.001
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='best_model-{epoch:02d}-val_loss_{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        verbose=True,
        save_last=True,  # Also save the last model for potential recovery
    )

    rich_progress = RichProgressBar(
        refresh_rate=1,
        leave=True
    )

    # construct data class 
    data = PE_Dataset(data_config = config_data)
    # construct model
    model = crispAIPERegression(hparams = {**config_model, **config_data, **config_training})


    if config_training['cpu']:
        trainer = pl.Trainer(
            max_epochs=config_training['max_epochs'],
            accelerator='cpu',
            log_every_n_steps=10,
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback, rich_progress]
        )
    elif platform.system() == 'Darwin' and 'arm' in platform.machine():
        # Check if MPS is available
        if torch.backends.mps.is_available():
            print("Using Apple Silicon GPU (MPS)")
            # Explicitly set the default device
            torch.set_default_device('mps')
            
            # Force PyTorch to use MPS
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            trainer = pl.Trainer(
                max_epochs=config_training['max_epochs'],
                accelerator='mps',  
                devices=1,          
                log_every_n_steps=10,
                logger=wandb_logger,
                callbacks=[early_stop_callback, checkpoint_callback, rich_progress],
                # These flags help ensure GPU usage
                deterministic=False,
            )
            
            # Print device info for debugging
            print(f"Default device: {torch.device('mps')}")
            print(f"PyTorch MPS device available: {torch.backends.mps.is_available()}")
            print(f"PyTorch MPS device built: {torch.backends.mps.is_built()}")
        else:
            print("MPS is not available, falling back to CPU")
            trainer = pl.Trainer(
                max_epochs=config_training['max_epochs'],
                accelerator='cpu',
                log_every_n_steps=10,
                logger=wandb_logger,
                callbacks=[early_stop_callback, checkpoint_callback, rich_progress]
            )
    else:
        print(f'Using GPUs: {config_training["gpus"]}')
        trainer = pl.Trainer(
            max_epochs=config_training['max_epochs'],
            strategy='dp',
            accelerator='gpu',
            devices=config_training['gpus'],  # this can be the gpu list
            log_every_n_steps=10,
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback, rich_progress]
        )

    trainer.fit(
        model = model,
        train_dataloaders = data.train_dataloader(),
        val_dataloaders = data.val_dataloader(),
    )

    # save model
    trainer.save_checkpoint(os.path.join(save_dir, 'model.ckpt'))
    train_time = datetime.datetime.now() - now
    hours, remainder = divmod(train_time.seconds, 3600)
    print(f'Training complete, took {hours}h:{remainder//60}m')

    """begin evaluation for trained model
    """

    print("Evaluating model")

    # Evaluate on test set
    test_results = trainer.test(model, dataloaders=data.test_dataloader())
    print(f"Test results: {test_results}")

    # Save test results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2) 