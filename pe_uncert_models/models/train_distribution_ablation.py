"""
Training script for distribution ablation study.
Trains Softmax and Logit-Normal variants (Dirichlet already trained).
"""

import os
import sys
import json
import argparse
import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pe_uncert_models.data_utils.data import PE_Dataset
from pe_uncert_models.models.distribution_ablation import crispAIPE_Softmax, crispAIPE_LogitNormal


def main():
    parser = argparse.ArgumentParser(description='Train distribution ablation study models')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--distribution', type=str, required=True, 
                       choices=['softmax', 'logit_normal'],
                       help='Distribution type to train')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and just return checkpoint path')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']

    # Resolve relative paths - fix data paths
    # Get project root (pe-uncert directory)
    current_file = os.path.abspath(__file__)
    # From pe_uncert_models/models/train_distribution_ablation.py -> pe-uncert
    project_root = os.path.abspath(os.path.join(current_file, '../../..'))
    
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Handle ../../ paths (relative to project root)
                if config_data[path_key].startswith('../../'):
                    # Remove ../../ prefix and join with project root
                    rel_path = config_data[path_key][6:]  # Remove '../../'
                    config_data[path_key] = os.path.normpath(
                        os.path.join(project_root, rel_path)
                    )
                # Handle ../ paths (relative to config file location)
                elif config_data[path_key].startswith('../'):
                    config_dir = os.path.dirname(os.path.abspath(args.config))
                    config_data[path_key] = os.path.normpath(
                        os.path.join(config_dir, config_data[path_key])
                    )
                else:
                    config_dir = os.path.dirname(os.path.abspath(args.config))
                    config_data[path_key] = os.path.normpath(
                        os.path.join(config_dir, config_data[path_key])
                    )

    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_file_name = args.config.split('/')[-1].split('.')[0]
    distribution_name = args.distribution
    
    # Use log_dir from config, but handle relative paths
    log_dir = config_training.get('log_dir', '../logs')
    if not os.path.isabs(log_dir):
        log_dir = os.path.normpath(os.path.join(os.path.dirname(args.config), log_dir))
    
    save_dir = os.path.join(log_dir, f"{config_file_name}_{distribution_name}", date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Check for existing checkpoint
    best_checkpoint = None
    parent_dir = os.path.dirname(save_dir)
    if os.path.exists(parent_dir):
        for run_dir in sorted(os.listdir(parent_dir), reverse=True):
            run_path = os.path.join(parent_dir, run_dir)
            if os.path.isdir(run_path):
                for file in os.listdir(run_path):
                    if file.startswith('best_model-'):
                        best_checkpoint = os.path.join(run_path, file)
                        print(f"Found existing checkpoint: {best_checkpoint}")
                        if args.skip_training:
                            print(f"Using existing checkpoint: {best_checkpoint}")
                            exit(0)
                        break
                if best_checkpoint:
                    break

    # Initialize data module
    data_module = PE_Dataset(data_config=config_data)

    # Initialize model based on distribution type
    hparams = {**config_model, **config_data, **config_training}
    
    if args.distribution == 'softmax':
        model = crispAIPE_Softmax(hparams=hparams)
    elif args.distribution == 'logit_normal':
        model = crispAIPE_LogitNormal(hparams=hparams)
    else:
        raise ValueError(f"Unknown distribution: {args.distribution}")

    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config_data.get('project_name', 'crispAIPE_distribution_ablation'),
        name=f"{config_file_name}_{distribution_name}",
        save_dir=save_dir
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='best_model-{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config_training.get('patience', 8),
        mode='min',
        min_delta=0.001
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config_training.get('max_epochs', 100),
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=50
    )

    # Train
    print(f"Training {args.distribution} model...")
    trainer.fit(model, data_module)
    
    print(f"Training complete! Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

