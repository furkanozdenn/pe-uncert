"""Training script for ablation study variants (transformer-only, CNN-only)
"""

import datetime
import os 
import sys
import json
import platform
import argparse

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from pe_uncert_models.data_utils.data import PE_Dataset
import torch
import torch.nn as nn
import math
import argparse as argparse_module

# Import base class
from pe_uncert_models.models.base import crispAIPEBase
from pe_uncert_models.models.block_nets import ConvNet

# Import PositionalEncoding and model variants (define them here to avoid import issues)
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerOnlyModel(crispAIPEBase):
    """Transformer-only variant: removes CNN, uses only transformer encoder."""
    
    def __init__(self, hparams):
        super(TransformerOnlyModel, self).__init__(hparams)
        
        if isinstance(hparams, dict):
            hparams = argparse_module.Namespace(**hparams)
        
        self.save_hyperparameters()
        
        # Model parameters (same as crispAIPE)
        self.input_dim = getattr(hparams, 'input_dim', 5)
        self.direction_channels = 2
        self.location_channels = 4
        self.unified_dim = self.input_dim + self.direction_channels + self.location_channels
        
        self.batch_size = getattr(hparams, 'batch_size', 128)
        self.lr = getattr(hparams, 'lr', 6e-4)
        self.model_name = "TransformerOnly"
        
        self.sequence_length = getattr(hparams, 'sequence_length', 99)
        self.target_seq_flank_len = getattr(hparams, 'target_seq_flank_len', 0)
        
        # Transformer encoder parameters
        self.embedding_dim = getattr(hparams, 'embedding_dim', 8)
        self.nhead = getattr(hparams, 'nhead', 2)
        self.num_encoder_layers = getattr(hparams, 'num_encoder_layers', 2)
        self.dim_feedforward = getattr(hparams, 'dim_feedforward', 32)
        self.dropout = getattr(hparams, 'dropout', 0.1)
        
        # Embedding layer
        self.vocab_size = self.input_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len=self.sequence_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)
        
        # Projection for unified representation
        self.final_dim = self.embedding_dim + self.unified_dim
        
        # Global pooling and MLP (no CNN)
        self.dirichlet_mlp = nn.Sequential(
            nn.Linear(self.final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus()
        )
        
        print(f"TransformerOnly model initialized with {self._num_params():,} parameters")
    
    def _num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _convert_onehot_to_indices(self, onehot_tensor):
        return torch.argmax(onehot_tensor, dim=-1)
    
    def _create_unified_representation(self, initial_seq, mutated_seq, locations):
        """Same as crispAIPE"""
        batch_size, seq_len, _ = initial_seq.shape
        
        initial_bool = initial_seq.bool()
        mutated_bool = mutated_seq.bool()
        or_result = (initial_bool | mutated_bool).float()
        
        initial_indices = torch.argmax(initial_seq, dim=-1)
        mutated_indices = torch.argmax(mutated_seq, dim=-1)
        
        direction = torch.zeros(batch_size, seq_len, 2, device=initial_seq.device)
        direction[:, :, 0:1] = (initial_indices < mutated_indices).unsqueeze(-1).float()
        direction[:, :, 1:2] = (initial_indices > mutated_indices).unsqueeze(-1).float()
        
        location_channels = []
        for loc in locations:
            if len(loc.shape) == 2:
                loc = loc.unsqueeze(-1)
            location_channels.append(loc)
        
        location_tensor = torch.cat(location_channels, dim=-1)
        unified_rep = torch.cat([or_result, direction, location_tensor], dim=-1)
        
        return unified_rep
    
    def forward(self, batch):
        initial_sequence = batch[0]
        
        # Transformer processing
        token_indices = self._convert_onehot_to_indices(initial_sequence)
        embedded_seq = self.embedding(token_indices)
        embedded_seq = self.pos_encoder(embedded_seq)
        transformer_embeddings = self.transformer_encoder(embedded_seq)
        
        # Unified representation
        mutated_sequence = batch[1]
        location_tensors = [batch[6], batch[7], batch[8], batch[9]]
        unified_rep = self._create_unified_representation(
            initial_sequence, mutated_sequence, location_tensors
        )
        
        # Concatenate transformer embeddings with unified representation
        final_representation = torch.cat([transformer_embeddings, unified_rep], dim=-1)
        
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(final_representation, dim=1)  # [batch_size, final_dim]
        
        # Generate Dirichlet parameters
        dirichlet_params = self.dirichlet_mlp(pooled_features)
        
        x_hat = final_representation
        y_hat = dirichlet_params
        
        return x_hat, y_hat
    
    def loss_function(self, predictions, targets, valid_step=False):
        """Same loss function as crispAIPE"""
        x_hat, y_hat = predictions
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        dirichlet_targets = torch.stack([
            edited_percentage, unedited_percentage, indel_percentage
        ], dim=1)
        dirichlet_targets = dirichlet_targets / torch.sum(dirichlet_targets, dim=1, keepdim=True)
        
        alpha = y_hat
        alpha_0 = torch.sum(alpha, dim=1, keepdim=True)
        
        log_beta = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(alpha_0.squeeze())
        epsilon = 1e-10
        log_likelihood_kernel = torch.sum((alpha - 1.0) * torch.log(dirichlet_targets + epsilon), dim=1)
        nll = log_beta - log_likelihood_kernel
        total_loss = torch.mean(nll)
        
        if getattr(self.hparams, 'weight_by_read_count', False):
            weights = total_read_count.float() / torch.mean(total_read_count.float())
            weighted_nll = nll * weights
            total_loss = torch.mean(weighted_nll)
        
        mloss_dict = {
            'dirichlet_nll': total_loss.item(),
            'alpha_mean': torch.mean(alpha).item(),
            'alpha_0_mean': torch.mean(alpha_0).item()
        }
        
        return total_loss, mloss_dict


class CNNOnlyModel(crispAIPEBase):
    """CNN-only variant: removes transformer, uses only CNN on unified representation."""
    
    def __init__(self, hparams):
        super(CNNOnlyModel, self).__init__(hparams)
        
        if isinstance(hparams, dict):
            hparams = argparse_module.Namespace(**hparams)
        
        self.save_hyperparameters()
        
        # Model parameters
        self.input_dim = getattr(hparams, 'input_dim', 5)
        self.direction_channels = 2
        self.location_channels = 4
        self.unified_dim = self.input_dim + self.direction_channels + self.location_channels
        
        self.batch_size = getattr(hparams, 'batch_size', 128)
        self.lr = getattr(hparams, 'lr', 6e-4)
        self.model_name = "CNNOnly"
        
        self.sequence_length = getattr(hparams, 'sequence_length', 99)
        self.target_seq_flank_len = getattr(hparams, 'target_seq_flank_len', 0)
        
        # No transformer, just use unified representation directly
        self.conv_net = ConvNet(self.unified_dim, 64)
        
        # MLP to output 3 Dirichlet parameters
        self.dirichlet_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus()
        )
        
        print(f"CNNOnly model initialized with {self._num_params():,} parameters")
    
    def _num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _create_unified_representation(self, initial_seq, mutated_seq, locations):
        """Same as crispAIPE"""
        batch_size, seq_len, _ = initial_seq.shape
        
        initial_bool = initial_seq.bool()
        mutated_bool = mutated_seq.bool()
        or_result = (initial_bool | mutated_bool).float()
        
        initial_indices = torch.argmax(initial_seq, dim=-1)
        mutated_indices = torch.argmax(mutated_seq, dim=-1)
        
        direction = torch.zeros(batch_size, seq_len, 2, device=initial_seq.device)
        direction[:, :, 0:1] = (initial_indices < mutated_indices).unsqueeze(-1).float()
        direction[:, :, 1:2] = (initial_indices > mutated_indices).unsqueeze(-1).float()
        
        location_channels = []
        for loc in locations:
            if len(loc.shape) == 2:
                loc = loc.unsqueeze(-1)
            location_channels.append(loc)
        
        location_tensor = torch.cat(location_channels, dim=-1)
        unified_rep = torch.cat([or_result, direction, location_tensor], dim=-1)
        
        return unified_rep
    
    def forward(self, batch):
        # Create unified representation
        initial_sequence = batch[0]
        mutated_sequence = batch[1]
        location_tensors = [batch[6], batch[7], batch[8], batch[9]]
        unified_rep = self._create_unified_representation(
            initial_sequence, mutated_sequence, location_tensors
        )
        
        # Transpose for ConvNet (from [batch, seq_len, channels] to [batch, channels, seq_len])
        unified_rep_t = unified_rep.transpose(1, 2)
        
        # Process with CNN
        conv_features = self.conv_net(unified_rep_t)  # [batch_size, 64, seq_len]
        
        # Global max pooling
        pooled_features = torch.max(conv_features, dim=2)[0]  # [batch_size, 64]
        
        # Generate Dirichlet parameters
        dirichlet_params = self.dirichlet_mlp(pooled_features)
        
        x_hat = unified_rep
        y_hat = dirichlet_params
        
        return x_hat, y_hat
    
    def loss_function(self, predictions, targets, valid_step=False):
        """Same loss function as crispAIPE"""
        x_hat, y_hat = predictions
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        dirichlet_targets = torch.stack([
            edited_percentage, unedited_percentage, indel_percentage
        ], dim=1)
        dirichlet_targets = dirichlet_targets / torch.sum(dirichlet_targets, dim=1, keepdim=True)
        
        alpha = y_hat
        alpha_0 = torch.sum(alpha, dim=1, keepdim=True)
        
        log_beta = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(alpha_0.squeeze())
        epsilon = 1e-10
        log_likelihood_kernel = torch.sum((alpha - 1.0) * torch.log(dirichlet_targets + epsilon), dim=1)
        nll = log_beta - log_likelihood_kernel
        total_loss = torch.mean(nll)
        
        if getattr(self.hparams, 'weight_by_read_count', False):
            weights = total_read_count.float() / torch.mean(total_read_count.float())
            weighted_nll = nll * weights
            total_loss = torch.mean(weighted_nll)
        
        mloss_dict = {
            'dirichlet_nll': total_loss.item(),
            'alpha_mean': torch.mean(alpha).item(),
            'alpha_0_mean': torch.mean(alpha_0).item()
        }
        
        return total_loss, mloss_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ablation study model variants')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['transformer_only', 'cnn_only'],
                       help='Type of model to train')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and just return checkpoint path')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # hparams from config file  
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']

    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_file_name = args.config.split('/')[-1].split('.')[0]
    save_dir = os.path.join(config_training['log_dir'], config_file_name, date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Check if model already exists
    best_checkpoint = None
    for file in os.listdir(save_dir) if os.path.exists(save_dir) else []:
        if file.startswith('best_model-'):
            best_checkpoint = os.path.join(save_dir, file)
            break
    
    # Also check in parent directory for existing runs
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
    
    if args.skip_training and best_checkpoint:
        print(f"Checkpoint found: {best_checkpoint}")
        exit(0)
    
    wandb_logger = WandbLogger(
        name = f"{config_file_name}_{args.model_type}",
        project = config_data['project_name'],
        log_model = True,
        save_dir = save_dir,
        offline = False,
    )

    wandb_logger.log_hyperparams(config)
    wandb_logger.experiment.log({'timestamp': date_suffix, 'model_type': args.model_type})

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
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
        save_last=True,
    )

    rich_progress = RichProgressBar(
        refresh_rate=1,
        leave=True
    )

    # Fix data paths - resolve relative to config file location
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Resolve relative to config file directory
                config_data[path_key] = os.path.normpath(
                    os.path.join(config_dir, config_data[path_key])
                )
    
    # construct data class 
    data = PE_Dataset(data_config=config_data)
    
    # construct model based on type
    if args.model_type == 'transformer_only':
        model = TransformerOnlyModel(hparams={**config_model, **config_data, **config_training})
    elif args.model_type == 'cnn_only':
        model = CNNOnlyModel(hparams={**config_model, **config_data, **config_training})
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if config_training['cpu']:
        trainer = pl.Trainer(
            max_epochs=config_training['max_epochs'],
            accelerator='cpu',
            log_every_n_steps=10,
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback, rich_progress]
        )
    elif platform.system() == 'Darwin' and 'arm' in platform.machine():
        if torch.backends.mps.is_available():
            print("Using Apple Silicon GPU (MPS)")
            torch.set_default_device('mps')
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            trainer = pl.Trainer(
                max_epochs=config_training['max_epochs'],
                accelerator='mps',  
                devices=1,          
                log_every_n_steps=10,
                logger=wandb_logger,
                callbacks=[early_stop_callback, checkpoint_callback, rich_progress],
                deterministic=False,
            )
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
        print(f'Using GPUs: {config_training.get("gpus", 1)}')
        trainer = pl.Trainer(
            max_epochs=config_training['max_epochs'],
            strategy='dp',
            accelerator='gpu',
            devices=config_training.get('gpus', 1),
            log_every_n_steps=10,
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback, rich_progress]
        )

    print(f"\n{'='*60}")
    print(f"Training {args.model_type} model")
    print(f"{'='*60}\n")

    trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    # save model
    trainer.save_checkpoint(os.path.join(save_dir, 'model.ckpt'))
    train_time = datetime.datetime.now() - now
    hours, remainder = divmod(train_time.seconds, 3600)
    print(f'\nTraining complete, took {hours}h:{remainder//60}m')
    
    # Print checkpoint path
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"\nBest model checkpoint: {best_checkpoint_path}")
    print(f"Save directory: {save_dir}")

