"""
Attention-based interpretability analysis for crispAIPE model.
Creates a 4-panel figure showing attention patterns by uncertainty level.

example cmd:
python test/attention_interpretability.py --config pe_uncert_models/configs/crispAIPE_train_test_split_conf.json --checkpoint pe_uncert_models/logs/crispAIPE_train_test_split_conf/2025-06-08-15-59-36/best_model-epoch=41-val_loss_val_loss=-3.0687.ckpt --output_dir ./figures/train_test_split_model
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import dirichlet

import json
from tqdm import tqdm

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate attention interpretability plots for crispAIPE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--n_examples', type=int, default=1000, help='Number of test examples to analyze')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    return parser.parse_args()


def load_model_and_data(config_path, checkpoint_path):
    """Load the model from checkpoint and prepare the test dataset."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']
    
    # Fix data paths - make them relative to the config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Convert relative paths to absolute paths
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                config_data[path_key] = os.path.normpath(
                    os.path.join(config_dir, config_data[path_key])
                )
    
    # Load dataset
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model from checkpoint
    model = crispAIPE.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    return model, data_module


def extract_attention_weights(model, batch):
    """Extract attention weights from the transformer layers."""
    device = next(model.parameters()).device
    batch = [b.to(device) for b in batch]
    
    # Extract attention manually from the model
    initial_sequence = batch[0]  # Shape: (batch_size, seq_len, input_dim)
    
    # Convert one-hot vectors to token indices
    token_indices = model._convert_onehot_to_indices(initial_sequence)  # Shape: (batch_size, seq_len)
    
    # Apply embedding
    embedded_seq = model.embedding(token_indices)  # Shape: (batch_size, seq_len, embedding_dim)
    
    # Add positional encoding
    embedded_seq = model.pos_encoder(embedded_seq)
    
    # Pass through transformer encoder and capture attention weights
    transformer_output = embedded_seq
    attention_weights = []
    
    with torch.no_grad():
        for layer in model.transformer_encoder.layers:
            # Multi-head self attention with explicit attention weight capture
            attn_output, attn_weights = layer.self_attn(
                transformer_output, transformer_output, transformer_output,
                need_weights=True, average_attn_weights=True
            )
            attention_weights.append(attn_weights.detach().cpu())
            
            # Complete the layer forward pass
            transformer_output = layer.norm1(transformer_output + attn_output)
            ff_output = layer.linear1(transformer_output)
            ff_output = F.relu(ff_output)
            ff_output = layer.dropout(ff_output)
            ff_output = layer.linear2(ff_output)
            transformer_output = layer.norm2(transformer_output + ff_output)
    
    # Get the full model predictions
    with torch.no_grad():
        _, alpha_params = model(batch)
    
    return attention_weights, alpha_params


def calculate_uncertainty_measures(alpha_params):
    """Calculate various uncertainty measures from Dirichlet parameters."""
    # Convert to numpy if tensor
    if torch.is_tensor(alpha_params):
        alpha_params = alpha_params.cpu().numpy()
    
    # Calculate concentration (sum of alphas) - higher means more certain
    alpha_sum = np.sum(alpha_params, axis=1)
    
    # Calculate uncertainty as 1/concentration (normalized)
    uncertainty = 1.0 / alpha_sum
    
    # Calculate entropy of the Dirichlet (expected entropy of samples)
    expected_probs = alpha_params / alpha_sum[:, np.newaxis]
    entropy = -np.sum(expected_probs * np.log(expected_probs + 1e-10), axis=1)
    
    # Calculate Dirichlet entropy (uncertainty in the parameters themselves)
    from scipy.special import digamma, gammaln
    alpha_sum_digamma = digamma(alpha_sum)
    dirichlet_entropy = (
        gammaln(alpha_sum) - np.sum(gammaln(alpha_params), axis=1) +
        np.sum((alpha_params - 1) * (digamma(alpha_params) - alpha_sum_digamma[:, np.newaxis]), axis=1)
    )
    
    return {
        'concentration': alpha_sum,
        'uncertainty': uncertainty,
        'entropy': entropy,
        'dirichlet_entropy': dirichlet_entropy
    }


def analyze_attention_by_uncertainty(model, data_module, n_examples=1000, batch_size=64):
    """Analyze attention patterns for high vs low uncertainty predictions."""
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_attention_weights = []
    all_uncertainty_measures = []
    all_sequences = []
    all_predictions = []
    all_ground_truth = []
    
    samples_collected = 0
    
    print("Collecting attention weights and uncertainty measures...")
    for batch in tqdm(test_loader):
        if samples_collected >= n_examples:
            break
            
        # Extract attention weights and predictions
        attention_weights, alpha_params = extract_attention_weights(model, batch)
        
        # Calculate uncertainty measures
        uncertainty_measures = calculate_uncertainty_measures(alpha_params)
        
        # Store sequences and ground truth
        sequences = batch[0].cpu().numpy()
        _, edited_pct, unedited_pct, indel_pct = batch[2:6]
        ground_truth = torch.stack([edited_pct, unedited_pct, indel_pct], dim=1).cpu().numpy()
        
        # Calculate predictions
        alpha_np = alpha_params.cpu().numpy()
        predictions = alpha_np / np.sum(alpha_np, axis=1, keepdims=True)
        
        # Store data
        if attention_weights:
            # Use the last (top) layer attention
            last_layer_attention = attention_weights[-1]  # Shape: (batch_size, seq_len, seq_len)
            
            batch_size_actual = min(len(sequences), n_examples - samples_collected)
            
            all_attention_weights.extend(last_layer_attention[:batch_size_actual])
            all_uncertainty_measures.extend([
                {k: v[:batch_size_actual] for k, v in uncertainty_measures.items()}
            ])
            all_sequences.extend(sequences[:batch_size_actual])
            all_predictions.extend(predictions[:batch_size_actual])
            all_ground_truth.extend(ground_truth[:batch_size_actual])
            
            samples_collected += batch_size_actual
    
    # Combine uncertainty measures
    combined_uncertainty = {}
    for key in all_uncertainty_measures[0].keys():
        combined_uncertainty[key] = np.concatenate([um[key] for um in all_uncertainty_measures])
    
    return {
        'attention_weights': all_attention_weights,
        'uncertainty_measures': combined_uncertainty,
        'sequences': np.array(all_sequences),
        'predictions': np.array(all_predictions),
        'ground_truth': np.array(all_ground_truth)
    }


def create_attention_heatmap(attention_matrix, title, ax, vocab_dict=None):
    """Create a heatmap of attention weights."""
    # Convert to numpy if it's a tensor
    if torch.is_tensor(attention_matrix):
        attention_matrix = attention_matrix.numpy()
        
    # Average attention across heads if multi-head
    if len(attention_matrix.shape) == 3:
        attention_matrix = np.mean(attention_matrix, axis=0)
    
    # Create heatmap
    sns.heatmap(attention_matrix, ax=ax, cmap='Blues', cbar=True, 
                xticklabels=False, yticklabels=False)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Sequence Position (Key)', fontsize=12)
    ax.set_ylabel('Sequence Position (Query)', fontsize=12)


def plot_attention_interpretability(data, output_dir):
    """Create 4-panel figure showing attention patterns by uncertainty level."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    attention_weights = data['attention_weights']
    uncertainty = data['uncertainty_measures']['uncertainty']
    sequences = data['sequences']
    
    # Sort by uncertainty
    uncertainty_order = np.argsort(uncertainty)
    
    # Select high and low uncertainty examples
    n_total = len(uncertainty)
    n_low = n_total // 10  # Bottom 10% (most certain)
    n_high = n_total // 10  # Top 10% (most uncertain)
    
    low_uncertainty_indices = uncertainty_order[:n_low]
    high_uncertainty_indices = uncertainty_order[-n_high:]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Attention Patterns by Uncertainty Level', fontsize=18, fontweight='bold')
    
    # Panel 1: Single high uncertainty example
    high_idx = high_uncertainty_indices[0]
    high_attention = attention_weights[high_idx]
    create_attention_heatmap(
        high_attention, 
        f'High Uncertainty Example\n(Uncertainty: {uncertainty[high_idx]:.4f})',
        axes[0, 0]
    )
    
    # Panel 2: Single low uncertainty example  
    low_idx = low_uncertainty_indices[0]
    low_attention = attention_weights[low_idx]
    create_attention_heatmap(
        low_attention,
        f'Low Uncertainty Example\n(Uncertainty: {uncertainty[low_idx]:.4f})',
        axes[0, 1]
    )
    
    # Panel 3: Average attention for high uncertainty cases
    high_attention_matrices = [attention_weights[i] for i in high_uncertainty_indices]
    avg_high_attention = np.mean(high_attention_matrices, axis=0)
    create_attention_heatmap(
        avg_high_attention,
        f'Average High Uncertainty\n(n={len(high_uncertainty_indices)}, μ={np.mean(uncertainty[high_uncertainty_indices]):.4f})',
        axes[1, 0]
    )
    
    # Panel 4: Average attention for low uncertainty cases
    low_attention_matrices = [attention_weights[i] for i in low_uncertainty_indices]
    avg_low_attention = np.mean(low_attention_matrices, axis=0)
    create_attention_heatmap(
        avg_low_attention,
        f'Average Low Uncertainty\n(n={len(low_uncertainty_indices)}, μ={np.mean(uncertainty[low_uncertainty_indices]):.4f})',
        axes[1, 1]
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_by_uncertainty.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attention_by_uncertainty.pdf'), bbox_inches='tight')
    plt.close()
    
    # Create additional analysis plot: Attention concentration vs Uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate attention concentration (how focused the attention is)
    attention_concentrations = []
    for attn_matrix in attention_weights:
        # Convert to numpy if it's a tensor
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        
        # Calculate entropy of attention (lower entropy = more concentrated)
        attn_matrix_safe = np.clip(attn_matrix, 1e-10, 1.0)  # Ensure valid probability range
        attn_entropy = -np.sum(attn_matrix_safe * np.log(attn_matrix_safe), axis=1)
        attention_concentrations.append(np.mean(attn_entropy))
    
    # Scatter plot: Attention concentration vs Uncertainty
    axes[0].scatter(attention_concentrations, uncertainty, alpha=0.6, s=20)
    axes[0].set_xlabel('Attention Entropy (Higher = More Diffuse)', fontsize=12)
    axes[0].set_ylabel('Prediction Uncertainty', fontsize=12)
    axes[0].set_title('Attention Concentration vs Uncertainty', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = np.corrcoef(attention_concentrations, uncertainty)[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Histogram: Distribution of uncertainties
    axes[1].hist(uncertainty, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(uncertainty[high_uncertainty_indices]), color='red', linestyle='--', 
                   label=f'High Uncertainty (μ={np.mean(uncertainty[high_uncertainty_indices]):.4f})')
    axes[1].axvline(np.mean(uncertainty[low_uncertainty_indices]), color='blue', linestyle='--',
                   label=f'Low Uncertainty (μ={np.mean(uncertainty[low_uncertainty_indices]):.4f})')
    axes[1].set_xlabel('Prediction Uncertainty', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Uncertainties', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attention_uncertainty_analysis.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Attention interpretability plots saved to {output_dir}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total examples analyzed: {len(uncertainty)}")
    print(f"High uncertainty examples: {len(high_uncertainty_indices)} (top 10%)")
    print(f"Low uncertainty examples: {len(low_uncertainty_indices)} (bottom 10%)")
    print(f"High uncertainty range: {uncertainty[high_uncertainty_indices].min():.4f} - {uncertainty[high_uncertainty_indices].max():.4f}")
    print(f"Low uncertainty range: {uncertainty[low_uncertainty_indices].min():.4f} - {uncertainty[low_uncertainty_indices].max():.4f}")
    print(f"Attention-Uncertainty correlation: {correlation:.3f}")


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print("Analyzing attention patterns...")
    data = analyze_attention_by_uncertainty(
        model, data_module, n_examples=args.n_examples, batch_size=args.batch_size
    )
    
    print("Creating interpretability plots...")
    plot_attention_interpretability(data, args.output_dir)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 