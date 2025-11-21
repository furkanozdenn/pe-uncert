"""
Model ablation study script for crispAIPE architecture.
Compares transformer-only, CNN-only, and hybrid (current) configurations.

This script creates Supplementary Table 1 showing architectural ablation results.

Example usage:
    python test/model_ablation.py --config pe_uncert_models/configs/crispAIPE_conf1.json \
        --hybrid_checkpoint <path_to_hybrid_model> \
        --output_dir test/figures/ablation_study
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import json
from scipy import stats
from tqdm import tqdm

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.models.block_nets import ConvNet
from pe_uncert_models.data_utils.data import PE_Dataset
from pe_uncert_models.models.base import crispAIPEBase
import pytorch_lightning as pl
import math


class TransformerOnlyModel(crispAIPEBase):
    """Transformer-only variant: removes CNN, uses only transformer encoder."""
    
    def __init__(self, hparams):
        super(TransformerOnlyModel, self).__init__(hparams)
        
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        
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
        # Use global average pooling over sequence dimension
        self.dirichlet_mlp = nn.Sequential(
            nn.Linear(self.final_dim, 64),  # Process each position
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
            hparams = argparse.Namespace(**hparams)
        
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
        # Process unified representation with CNN
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


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer (same as crispAIPE)"""
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


def load_model_and_data(config_path, checkpoint_path, model_class, variant_config_path=None):
    """Load model from checkpoint and prepare test dataset.
    
    Args:
        config_path: Base config path (for data paths)
        checkpoint_path: Path to model checkpoint
        model_class: Model class to load
        variant_config_path: Optional path to variant-specific config (for transformer/cnn-only)
    """
    # Use variant config if provided, otherwise use base config
    config_to_use = variant_config_path if variant_config_path else config_path
    
    with open(config_to_use, 'r') as f:
        config = json.load(f)
    
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']
    
    # Fix data paths - resolve relative to config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Resolve relative to config file directory
                config_data[path_key] = os.path.normpath(
                    os.path.join(config_dir, config_data[path_key])
                )
    
    # Load dataset
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model from checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    return model, data_module


def evaluate_model(model, data_module, batch_size=64):
    """Evaluate model and return metrics."""
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Evaluate on test set
    all_predictions = []
    all_targets = []
    test_losses = []
    
    print(f"Evaluating {model.model_name} on test set...")
    for batch in tqdm(test_loader):
        batch = [b.to(device) for b in batch]
        
        with torch.no_grad():
            x_hat, y_hat = model(batch)
            loss, _ = model.loss_function((x_hat, y_hat), batch[2:6], valid_step=True)
            test_losses.append(loss.item())
            
            # Calculate expected proportions
            alpha_sum = torch.sum(y_hat, dim=1, keepdim=True)
            expected_props = y_hat / alpha_sum
            
            # Get targets
            _, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
            ground_truth = torch.stack([
                edited_percentage, unedited_percentage, indel_percentage
            ], dim=1)
            ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
            
            all_predictions.append(expected_props.cpu())
            all_targets.append(ground_truth.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate Spearman correlations
    pred_edited = all_predictions[:, 0]
    pred_unedited = all_predictions[:, 1]
    pred_indel = all_predictions[:, 2]
    
    true_edited = all_targets[:, 0]
    true_unedited = all_targets[:, 1]
    true_indel = all_targets[:, 2]
    
    edited_corr, edited_p = stats.spearmanr(true_edited, pred_edited)
    unedited_corr, unedited_p = stats.spearmanr(true_unedited, pred_unedited)
    indel_corr, indel_p = stats.spearmanr(true_indel, pred_indel)
    
    # Overall correlation
    all_pred = all_predictions.flatten()
    all_true = all_targets.flatten()
    overall_corr, overall_p = stats.spearmanr(all_true, all_pred)
    
    # Calculate validation loss
    val_losses = []
    for batch in val_loader:
        batch = [b.to(device) for b in batch]
        with torch.no_grad():
            x_hat, y_hat = model(batch)
            loss, _ = model.loss_function((x_hat, y_hat), batch[2:6], valid_step=True)
            val_losses.append(loss.item())
    
    avg_test_loss = np.mean(test_losses)
    avg_val_loss = np.mean(val_losses)
    
    results = {
        'test_loss': avg_test_loss,
        'val_loss': avg_val_loss,
        'edited_spearman': edited_corr,
        'unedited_spearman': unedited_corr,
        'indel_spearman': indel_corr,
        'overall_spearman': overall_corr,
        'n_samples': len(all_targets)
    }
    
    return results


def create_ablation_table(results_dict, output_dir):
    """Create Supplementary Table 1 comparing all architectures."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for table
    architectures = ['Transformer-only', 'CNN-only', 'Hybrid (crispAIPE)']
    data = []
    
    for arch in architectures:
        if arch in results_dict:
            r = results_dict[arch]
            data.append({
                'Architecture': arch,
                'Val Loss': f"{r['val_loss']:.4f}",
                'Test Loss': f"{r['test_loss']:.4f}",
                'Edited ρ': f"{r['edited_spearman']:.4f}",
                'Unedited ρ': f"{r['unedited_spearman']:.4f}",
                'Indel ρ': f"{r['indel_spearman']:.4f}",
                'Overall ρ': f"{r['overall_spearman']:.4f}"
            })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'ablation_table_s1.csv')
    df.to_csv(csv_path, index=False)
    print(f"Ablation table saved to {csv_path}")
    
    # Create formatted table plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for col_idx, col in enumerate(df.columns[1:], 1):  # Skip 'Architecture' column
        if 'Loss' in col:
            # Lower is better
            best_idx = df[col].str.replace(',', '').astype(float).idxmin() + 1
        else:
            # Higher is better
            best_idx = df[col].str.replace(',', '').astype(float).idxmax() + 1
        
        table[(best_idx, col_idx)].set_facecolor('#D5E8D4')
    
    plt.title('Supplementary Table 1: Architectural Ablation Study', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'ablation_table_s1.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'ablation_table_s1.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Ablation table figure saved to {fig_path}")
    
    # Also create a bar plot comparison
    create_ablation_barplot(results_dict, output_dir)
    
    return df


def create_ablation_barplot(results_dict, output_dir):
    """Create bar plot comparing architectures."""
    architectures = ['Transformer-only', 'CNN-only', 'Hybrid (crispAIPE)']
    
    # Extract metrics
    metrics = ['overall_spearman', 'edited_spearman', 'unedited_spearman', 'indel_spearman']
    metric_labels = ['Overall ρ', 'Edited ρ', 'Unedited ρ', 'Indel ρ']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Spearman correlations
    ax1 = axes[0]
    x = np.arange(len(architectures))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_dict[arch][metric] if arch in results_dict else 0 
                  for arch in architectures]
        offset = (i - len(metrics)/2) * width + width/2
        bars = ax1.bar(x + offset, values, width, label=label, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Architecture', fontsize=12)
    ax1.set_ylabel('Spearman Correlation (ρ)', fontsize=12)
    ax1.set_title('Spearman Correlations by Architecture', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(architectures, fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Plot 2: Losses
    ax2 = axes[1]
    val_losses = [results_dict[arch]['val_loss'] if arch in results_dict else 0 
                  for arch in architectures]
    test_losses = [results_dict[arch]['test_loss'] if arch in results_dict else 0 
                   for arch in architectures]
    
    x = np.arange(len(architectures))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, val_losses, width, label='Validation Loss', alpha=0.8, color='#4472C4')
    bars2 = ax2.bar(x + width/2, test_losses, width, label='Test Loss', alpha=0.8, color='#ED7D31')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Architecture', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss by Architecture', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(architectures, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'ablation_comparison.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Ablation comparison plot saved to {plot_path}")


def save_results_json(results_dict, output_dir):
    """Save results to JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert numpy types to native Python types
    json_results = {}
    for arch, results in results_dict.items():
        json_results[arch] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in results.items()
        }
    
    json_path = os.path.join(output_dir, 'ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model ablation study comparing transformer-only, CNN-only, and hybrid architectures'
    )
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--hybrid_checkpoint', type=str, default=None,
                       help='Path to hybrid (crispAIPE) model checkpoint')
    parser.add_argument('--transformer_checkpoint', type=str, default=None,
                       help='Path to transformer-only model checkpoint')
    parser.add_argument('--cnn_checkpoint', type=str, default=None,
                       help='Path to CNN-only model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test/figures/ablation_study',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']
    
    # Fix data paths - resolve relative to config file location
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Resolve relative to config file directory
                config_data[path_key] = os.path.normpath(
                    os.path.join(config_dir, config_data[path_key])
                )
    
    # Load dataset (shared across all models)
    print("Loading dataset...")
    data_module = PE_Dataset(data_config=config_data)
    
    results_dict = {}
    
    # Evaluate Hybrid (crispAIPE) model
    if args.hybrid_checkpoint and os.path.exists(args.hybrid_checkpoint):
        print("\n" + "="*60)
        print("Evaluating Hybrid (crispAIPE) model...")
        print("="*60)
        model, _ = load_model_and_data(args.config, args.hybrid_checkpoint, crispAIPE)
        results = evaluate_model(model, data_module, args.batch_size)
        results_dict['Hybrid (crispAIPE)'] = results
        print(f"Hybrid Results: Overall ρ = {results['overall_spearman']:.4f}, "
              f"Val Loss = {results['val_loss']:.4f}")
    
    # Evaluate Transformer-only model
    if args.transformer_checkpoint and os.path.exists(args.transformer_checkpoint):
        print("\n" + "="*60)
        print("Evaluating Transformer-only model...")
        print("="*60)
        # Use transformer-only config if it exists
        transformer_config = os.path.join(os.path.dirname(args.config), 'crispAIPE_transformer_only_conf.json')
        if not os.path.exists(transformer_config):
            transformer_config = None
        model, _ = load_model_and_data(args.config, args.transformer_checkpoint, TransformerOnlyModel, transformer_config)
        results = evaluate_model(model, data_module, args.batch_size)
        results_dict['Transformer-only'] = results
        print(f"Transformer-only Results: Overall ρ = {results['overall_spearman']:.4f}, "
              f"Val Loss = {results['val_loss']:.4f}")
    
    # Evaluate CNN-only model
    if args.cnn_checkpoint and os.path.exists(args.cnn_checkpoint):
        print("\n" + "="*60)
        print("Evaluating CNN-only model...")
        print("="*60)
        # Use CNN-only config if it exists
        cnn_config = os.path.join(os.path.dirname(args.config), 'crispAIPE_cnn_only_conf.json')
        if not os.path.exists(cnn_config):
            cnn_config = None
        model, _ = load_model_and_data(args.config, args.cnn_checkpoint, CNNOnlyModel, cnn_config)
        results = evaluate_model(model, data_module, args.batch_size)
        results_dict['CNN-only'] = results
        print(f"CNN-only Results: Overall ρ = {results['overall_spearman']:.4f}, "
              f"Val Loss = {results['val_loss']:.4f}")
    
    if not results_dict:
        print("\nERROR: No model checkpoints provided or found!")
        print("Please provide at least one checkpoint using:")
        print("  --hybrid_checkpoint")
        print("  --transformer_checkpoint")
        print("  --cnn_checkpoint")
        return
    
    # Create ablation table and plots
    print("\n" + "="*60)
    print("Creating ablation study table and plots...")
    print("="*60)
    
    df = create_ablation_table(results_dict, args.output_dir)
    save_results_json(results_dict, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

