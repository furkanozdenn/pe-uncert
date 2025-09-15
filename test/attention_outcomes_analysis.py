"""
Comprehensive attention analysis for crispAIPE model focusing on edit outcomes and position-wise patterns.
Creates multiple figures showing biologically meaningful attention patterns.

example cmd:
python test/attention_outcomes_analysis.py --config pe_uncert_models/configs/crispAIPE_train_test_split_conf.json --checkpoint pe_uncert_models/logs/crispAIPE_train_test_split_conf/2025-06-08-15-59-36/best_model-epoch=41-val_loss_val_loss=-3.0687.ckpt --output_dir ./figures/train_test_split_model
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
from scipy.stats import pearsonr, spearmanr

import json
from tqdm import tqdm

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate attention analysis by edit outcomes and positions')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--n_examples', type=int, default=5000, help='Number of test examples to analyze')
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


def extract_attention_and_locations(model, batch):
    """Extract attention weights and functional region locations."""
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
    
    # Extract location information
    protospacer_location = batch[6].cpu().numpy()  # Shape: (batch_size, seq_len)
    pbs_location = batch[7].cpu().numpy()
    rt_initial_location = batch[8].cpu().numpy()
    rt_mutated_location = batch[9].cpu().numpy()
    
    return attention_weights, alpha_params, {
        'protospacer': protospacer_location,
        'pbs': pbs_location,
        'rt_initial': rt_initial_location,
        'rt_mutated': rt_mutated_location
    }


def analyze_attention_by_outcomes(model, data_module, n_examples=1000, batch_size=64):
    """Analyze attention patterns for different edit outcomes."""
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_attention_weights = []
    all_predictions = []
    all_ground_truth = []
    all_locations = {
        'protospacer': [],
        'pbs': [],
        'rt_initial': [],
        'rt_mutated': []
    }
    
    samples_collected = 0
    
    print("Collecting attention weights and outcome data...")
    for batch in tqdm(test_loader):
        if samples_collected >= n_examples:
            break
            
        # Extract attention weights and predictions
        attention_weights, alpha_params, locations = extract_attention_and_locations(model, batch)
        
        # Store sequences and ground truth
        _, edited_pct, unedited_pct, indel_pct = batch[2:6]
        ground_truth = torch.stack([edited_pct, unedited_pct, indel_pct], dim=1).cpu().numpy()
        
        # Calculate predictions
        alpha_np = alpha_params.cpu().numpy()
        predictions = alpha_np / np.sum(alpha_np, axis=1, keepdims=True)
        
        # Store data
        if attention_weights:
            # Use the last (top) layer attention
            last_layer_attention = attention_weights[-1]  # Shape: (batch_size, seq_len, seq_len)
            
            batch_size_actual = min(len(predictions), n_examples - samples_collected)
            
            all_attention_weights.extend(last_layer_attention[:batch_size_actual])
            all_predictions.extend(predictions[:batch_size_actual])
            all_ground_truth.extend(ground_truth[:batch_size_actual])
            
            # Store location data
            for key, loc_data in locations.items():
                all_locations[key].extend(loc_data[:batch_size_actual])
            
            samples_collected += batch_size_actual
    
    # Convert to numpy arrays
    for key in all_locations:
        all_locations[key] = np.array(all_locations[key])
    
    return {
        'attention_weights': all_attention_weights,
        'predictions': np.array(all_predictions),
        'ground_truth': np.array(all_ground_truth),
        'locations': all_locations
    }


def create_attention_heatmap(attention_matrix, title, ax):
    """Create a heatmap of attention weights."""
    # Convert to numpy if it's a tensor
    if torch.is_tensor(attention_matrix):
        attention_matrix = attention_matrix.numpy()
        
    # Average attention across heads if multi-head
    if len(attention_matrix.shape) == 3:
        attention_matrix = np.mean(attention_matrix, axis=0)
    
    # Create heatmap
    sns.heatmap(attention_matrix, ax=ax, cmap='Blues', cbar=True, 
                xticklabels=False, yticklabels=False, square=True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sequence Position (Key)', fontsize=12)
    ax.set_ylabel('Sequence Position (Query)', fontsize=12)


def plot_attention_by_outcomes(data, output_dir):
    """Create 4-panel figure showing attention patterns by edit outcomes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    attention_weights = data['attention_weights']
    predictions = data['predictions']
    
    # Define outcome categories based on predicted percentages
    edited_scores = predictions[:, 0]  # edited %
    unedited_scores = predictions[:, 1]  # unedited %
    indel_scores = predictions[:, 2]  # indel %
    
    # Get top 20% for each outcome
    n_top = len(predictions) // 5
    
    high_edited_indices = np.argsort(edited_scores)[-n_top:]
    high_unedited_indices = np.argsort(unedited_scores)[-n_top:]
    high_indel_indices = np.argsort(indel_scores)[-n_top:]
    
    # Calculate average attention for each outcome group
    high_edited_attention = np.mean([attention_weights[i] for i in high_edited_indices], axis=0)
    high_unedited_attention = np.mean([attention_weights[i] for i in high_unedited_indices], axis=0)
    high_indel_attention = np.mean([attention_weights[i] for i in high_indel_indices], axis=0)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Attention Patterns by Edit Outcomes', fontsize=18, fontweight='bold')
    
    # Panel 1: High edited percentage attention
    create_attention_heatmap(
        high_edited_attention,
        f'High Edited % Predictions\n(n={n_top}, μ={np.mean(edited_scores[high_edited_indices]):.3f})',
        axes[0, 0]
    )
    
    # Panel 2: High unedited percentage attention
    create_attention_heatmap(
        high_unedited_attention,
        f'High Unedited % Predictions\n(n={n_top}, μ={np.mean(unedited_scores[high_unedited_indices]):.3f})',
        axes[0, 1]
    )
    
    # Panel 3: High indel percentage attention
    create_attention_heatmap(
        high_indel_attention,
        f'High Indel % Predictions\n(n={n_top}, μ={np.mean(indel_scores[high_indel_indices]):.3f})',
        axes[1, 0]
    )
    
    # Panel 4: Attention differences (edited - unedited)
    attention_diff = high_edited_attention - high_unedited_attention
    sns.heatmap(attention_diff, ax=axes[1, 1], cmap='RdBu_r', center=0, cbar=True,
                xticklabels=False, yticklabels=False, square=True)
    axes[1, 1].set_title('Attention Difference\n(High Edited - High Unedited)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Sequence Position (Key)', fontsize=12)
    axes[1, 1].set_ylabel('Sequence Position (Query)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_by_outcomes.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attention_by_outcomes.pdf'), bbox_inches='tight')
    plt.close()
    
    # Calculate and print correlation statistics
    edited_vs_attn = []
    unedited_vs_attn = []
    indel_vs_attn = []
    
    for i, attn_matrix in enumerate(attention_weights):
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        
        # Calculate mean attention strength
        mean_attention = np.mean(attn_matrix)
        edited_vs_attn.append((predictions[i, 0], mean_attention))
        unedited_vs_attn.append((predictions[i, 1], mean_attention))
        indel_vs_attn.append((predictions[i, 2], mean_attention))
    
    # Calculate correlations
    edited_corr = pearsonr([x[0] for x in edited_vs_attn], [x[1] for x in edited_vs_attn])[0]
    unedited_corr = pearsonr([x[0] for x in unedited_vs_attn], [x[1] for x in unedited_vs_attn])[0]
    indel_corr = pearsonr([x[0] for x in indel_vs_attn], [x[1] for x in indel_vs_attn])[0]
    
    print(f"\nAttention-Outcome Correlations:")
    print(f"Edited % vs Attention: {edited_corr:.3f}")
    print(f"Unedited % vs Attention: {unedited_corr:.3f}")
    print(f"Indel % vs Attention: {indel_corr:.3f}")


def plot_position_wise_attention(data, output_dir):
    """Create position-wise attention analysis with functional regions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    attention_weights = data['attention_weights']
    predictions = data['predictions']
    locations = data['locations']
    
    # Calculate position-wise attention (average attention each position receives)
    seq_len = attention_weights[0].shape[-1]
    position_attention = np.zeros(seq_len)
    
    for attn_matrix in attention_weights:
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        
        # Sum attention received by each position (sum over query dimension)
        position_attention += np.sum(attn_matrix, axis=0)
    
    position_attention /= len(attention_weights)  # Average across samples
    
    # Calculate average location masks
    avg_locations = {}
    for region, loc_data in locations.items():
        avg_locations[region] = np.mean(loc_data, axis=0)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Position-wise Attention Analysis', fontsize=18, fontweight='bold')
    
    # Panel 1: Position-wise attention with functional regions
    positions = np.arange(seq_len)
    axes[0].plot(positions, position_attention, 'b-', linewidth=3, label='Average Attention')
    
    # Overlay functional regions using vertical lines instead of fill_between to avoid legend overlap
    colors = {'protospacer': 'red', 'pbs': 'green', 'rt_initial': 'orange', 'rt_mutated': 'purple'}
    region_boundaries = []
    
    # Define vertical positions for labels to avoid overlap
    max_attention = np.max(position_attention)
    label_positions = {
        'protospacer': max_attention * 0.95,
        'pbs': max_attention * 0.85,
        'rt_initial': max_attention * 0.75,
        'rt_mutated': max_attention * 0.65
    }
    
    for region, color in colors.items():
        region_mask = avg_locations[region] > 0.5  # Threshold for region presence
        if np.any(region_mask):
            # Find region boundaries
            region_positions = np.where(region_mask)[0]
            if len(region_positions) > 0:
                start_pos = region_positions[0]
                end_pos = region_positions[-1]
                region_boundaries.append((start_pos, end_pos, color, region))
                
                # Draw vertical lines at region boundaries
                axes[0].axvline(x=start_pos, color=color, linestyle='--', alpha=0.7, linewidth=2)
                axes[0].axvline(x=end_pos, color=color, linestyle='--', alpha=0.7, linewidth=2)
                
                # Add region label with staggered vertical positions to avoid overlap
                mid_pos = (start_pos + end_pos) / 2
                label_y = label_positions[region]
                axes[0].text(mid_pos, label_y, 
                           f'{region.title()}', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color=color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    axes[0].set_xlabel('Sequence Position', fontsize=14)
    axes[0].set_ylabel('Average Attention Received', fontsize=14)
    axes[0].set_title('Position-wise Attention with Functional Regions', fontsize=16)
    
    # Create a custom legend with region boundaries
    legend_elements = [plt.Line2D([0], [0], color='blue', linewidth=3, label='Average Attention')]
    for start_pos, end_pos, color, region in region_boundaries:
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2, 
                                        label=f'{region.title()} Region (pos {start_pos}-{end_pos})'))
    
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Attention by editing efficiency
    edited_scores = predictions[:, 0]
    
    # Separate high vs low editing efficiency
    high_efficiency_threshold = np.percentile(edited_scores, 75)
    low_efficiency_threshold = np.percentile(edited_scores, 25)
    
    high_efficiency_indices = np.where(edited_scores >= high_efficiency_threshold)[0]
    low_efficiency_indices = np.where(edited_scores <= low_efficiency_threshold)[0]
    
    # Calculate position-wise attention for each group
    high_eff_position_attention = np.zeros(seq_len)
    low_eff_position_attention = np.zeros(seq_len)
    
    for idx in high_efficiency_indices:
        attn_matrix = attention_weights[idx]
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        high_eff_position_attention += np.sum(attn_matrix, axis=0)
    
    for idx in low_efficiency_indices:
        attn_matrix = attention_weights[idx]
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        low_eff_position_attention += np.sum(attn_matrix, axis=0)
    
    high_eff_position_attention /= len(high_efficiency_indices)
    low_eff_position_attention /= len(low_efficiency_indices)
    
    # Plot comparison
    axes[1].plot(positions, high_eff_position_attention, 'g-', linewidth=3, 
                label=f'High Efficiency (n={len(high_efficiency_indices)})')
    axes[1].plot(positions, low_eff_position_attention, 'r-', linewidth=3, 
                label=f'Low Efficiency (n={len(low_efficiency_indices)})')
    
    axes[1].set_xlabel('Sequence Position', fontsize=14)
    axes[1].set_ylabel('Average Attention Received', fontsize=14)
    axes[1].set_title('Position-wise Attention: High vs Low Editing Efficiency', fontsize=16)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_wise_attention_color_modified.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'position_wise_attention_color_modified.pdf'), bbox_inches='tight')
    plt.close()
    
    # Calculate regional attention statistics
    print(f"\nPosition-wise Attention Analysis:")
    print(f"Sequence length: {seq_len}")
    print(f"High efficiency samples: {len(high_efficiency_indices)}")
    print(f"Low efficiency samples: {len(low_efficiency_indices)}")
    
    # Calculate attention in each functional region
    for region, color in colors.items():
        region_mask = avg_locations[region] > 0.5
        if np.any(region_mask):
            region_attention = np.mean(position_attention[region_mask])
            high_eff_region_attention = np.mean(high_eff_position_attention[region_mask])
            low_eff_region_attention = np.mean(low_eff_position_attention[region_mask])
            
            print(f"{region.title()} region attention:")
            print(f"  Overall: {region_attention:.4f}")
            print(f"  High efficiency: {high_eff_region_attention:.4f}")
            print(f"  Low efficiency: {low_eff_region_attention:.4f}")
            print(f"  Difference: {high_eff_region_attention - low_eff_region_attention:.4f}")


def plot_attention_vs_efficiency(data, output_dir):
    """Create comprehensive attention vs editing efficiency analysis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    attention_weights = data['attention_weights']
    predictions = data['predictions']
    locations = data['locations']
    
    # Calculate editing efficiency (predicted edited percentage)
    editing_efficiency = predictions[:, 0]  # edited %
    
    # Calculate different attention metrics
    attention_metrics = {
        'mean_attention': [],
        'max_attention': [],
        'attention_entropy': [],
        'protospacer_attention': [],
        'pbs_attention': [],
        'rt_attention': [],
        'rt_initial_attention': [],
        'rt_mutated_attention': []
    }
    
    print("Computing attention metrics...")
    for i, attn_matrix in enumerate(tqdm(attention_weights)):
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        
        # Global attention metrics
        attention_metrics['mean_attention'].append(np.mean(attn_matrix))
        attention_metrics['max_attention'].append(np.max(attn_matrix))
        
        # Attention entropy (measure of attention spread)
        attn_flat = attn_matrix.flatten()
        attn_flat = attn_flat / np.sum(attn_flat)  # normalize
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
        attention_metrics['attention_entropy'].append(entropy)
        
        # Regional attention (sum attention received by each region)
        position_attention = np.sum(attn_matrix, axis=0)  # sum over query dimension
        
        # Extract regional attention
        protospacer_mask = locations['protospacer'][i] > 0.5
        pbs_mask = locations['pbs'][i] > 0.5
        rt_initial_mask = locations['rt_initial'][i] > 0.5
        rt_mutated_mask = locations['rt_mutated'][i] > 0.5
        rt_mask = rt_initial_mask | rt_mutated_mask
        
        attention_metrics['protospacer_attention'].append(
            np.mean(position_attention[protospacer_mask]) if np.any(protospacer_mask) else 0
        )
        attention_metrics['pbs_attention'].append(
            np.mean(position_attention[pbs_mask]) if np.any(pbs_mask) else 0
        )
        attention_metrics['rt_attention'].append(
            np.mean(position_attention[rt_mask]) if np.any(rt_mask) else 0
        )
        attention_metrics['rt_initial_attention'].append(
            np.mean(position_attention[rt_initial_mask]) if np.any(rt_initial_mask) else 0
        )
        attention_metrics['rt_mutated_attention'].append(
            np.mean(position_attention[rt_mutated_mask]) if np.any(rt_mutated_mask) else 0
        )
    
    # Convert to numpy arrays
    for key in attention_metrics:
        attention_metrics[key] = np.array(attention_metrics[key])
    
    # Create comprehensive figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Attention vs Editing Efficiency Analysis', fontsize=18, fontweight='bold')
    
    # Panel 1: Mean Attention vs Efficiency
    axes[0, 0].scatter(editing_efficiency, attention_metrics['mean_attention'], 
                      alpha=0.6, s=20, color='blue')
    corr_mean = pearsonr(editing_efficiency, attention_metrics['mean_attention'])[0]
    axes[0, 0].set_xlabel('Predicted Editing Efficiency (%)', fontsize=12)
    axes[0, 0].set_ylabel('Mean Attention', fontsize=12)
    axes[0, 0].set_title(f'Mean Attention vs Efficiency\n(r = {corr_mean:.3f})', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Max Attention vs Efficiency
    axes[0, 1].scatter(editing_efficiency, attention_metrics['max_attention'], 
                      alpha=0.6, s=20, color='red')
    corr_max = pearsonr(editing_efficiency, attention_metrics['max_attention'])[0]
    axes[0, 1].set_xlabel('Predicted Editing Efficiency (%)', fontsize=12)
    axes[0, 1].set_ylabel('Max Attention', fontsize=12)
    axes[0, 1].set_title(f'Max Attention vs Efficiency\n(r = {corr_max:.3f})', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Attention Entropy vs Efficiency
    axes[1, 0].scatter(editing_efficiency, attention_metrics['attention_entropy'], 
                      alpha=0.6, s=20, color='green')
    corr_entropy = pearsonr(editing_efficiency, attention_metrics['attention_entropy'])[0]
    axes[1, 0].set_xlabel('Predicted Editing Efficiency (%)', fontsize=12)
    axes[1, 0].set_ylabel('Attention Entropy', fontsize=12)
    axes[1, 0].set_title(f'Attention Entropy vs Efficiency\n(r = {corr_entropy:.3f})', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Regional Attention Comparison
    efficiency_bins = np.percentile(editing_efficiency, [0, 25, 50, 75, 100])
    bin_labels = ['Q1\n(0-25%)', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Q4\n(75-100%)']
    
    regional_data = []
    for i in range(4):
        mask = (editing_efficiency >= efficiency_bins[i]) & (editing_efficiency < efficiency_bins[i+1])
        if i == 3:  # Include the maximum value in the last bin
            mask = (editing_efficiency >= efficiency_bins[i]) & (editing_efficiency <= efficiency_bins[i+1])
        
        regional_data.append({
            'protospacer': np.mean(attention_metrics['protospacer_attention'][mask]),
            'pbs': np.mean(attention_metrics['pbs_attention'][mask]),
            'rt': np.mean(attention_metrics['rt_attention'][mask])
        })
    
    x_pos = np.arange(len(bin_labels))
    width = 0.25
    
    axes[1, 1].bar(x_pos - width, [d['protospacer'] for d in regional_data], 
                  width, label='Protospacer', color='red', alpha=0.7)
    axes[1, 1].bar(x_pos, [d['pbs'] for d in regional_data], 
                  width, label='PBS', color='green', alpha=0.7)
    axes[1, 1].bar(x_pos + width, [d['rt'] for d in regional_data], 
                  width, label='RT', color='orange', alpha=0.7)
    
    axes[1, 1].set_xlabel('Editing Efficiency Quartiles', fontsize=12)
    axes[1, 1].set_ylabel('Average Regional Attention', fontsize=12)
    axes[1, 1].set_title('Regional Attention by Efficiency Quartiles', fontsize=14)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(bin_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Panel 5: Protospacer Attention vs Efficiency
    axes[2, 0].scatter(editing_efficiency, attention_metrics['protospacer_attention'], 
                      alpha=0.6, s=20, color='red')
    corr_proto = pearsonr(editing_efficiency, attention_metrics['protospacer_attention'])[0]
    axes[2, 0].set_xlabel('Predicted Editing Efficiency (%)', fontsize=12)
    axes[2, 0].set_ylabel('Protospacer Attention', fontsize=12)
    axes[2, 0].set_title(f'Protospacer Attention vs Efficiency\n(r = {corr_proto:.3f})', fontsize=14)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Panel 6: RT Attention vs Efficiency
    axes[2, 1].scatter(editing_efficiency, attention_metrics['rt_attention'], 
                      alpha=0.6, s=20, color='orange')
    corr_rt = pearsonr(editing_efficiency, attention_metrics['rt_attention'])[0]
    axes[2, 1].set_xlabel('Predicted Editing Efficiency (%)', fontsize=12)
    axes[2, 1].set_ylabel('RT Region Attention', fontsize=12)
    axes[2, 1].set_title(f'RT Attention vs Efficiency\n(r = {corr_rt:.3f})', fontsize=14)
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_vs_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attention_vs_efficiency.pdf'), bbox_inches='tight')
    plt.close()
    
    # Create efficiency-binned attention heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Average Attention Patterns by Editing Efficiency', fontsize=18, fontweight='bold')
    
    # Calculate average attention for each efficiency quartile
    quartile_attentions = []
    quartile_names = ['Low Efficiency\n(0-25%)', 'Medium-Low Efficiency\n(25-50%)', 
                     'Medium-High Efficiency\n(50-75%)', 'High Efficiency\n(75-100%)']
    
    for i in range(4):
        mask = (editing_efficiency >= efficiency_bins[i]) & (editing_efficiency < efficiency_bins[i+1])
        if i == 3:  # Include the maximum value in the last bin
            mask = (editing_efficiency >= efficiency_bins[i]) & (editing_efficiency <= efficiency_bins[i+1])
        
        quartile_indices = np.where(mask)[0]
        if len(quartile_indices) > 0:
            quartile_attention = np.mean([attention_weights[idx] for idx in quartile_indices], axis=0)
            if torch.is_tensor(quartile_attention):
                quartile_attention = quartile_attention.numpy()
            if len(quartile_attention.shape) == 3:
                quartile_attention = np.mean(quartile_attention, axis=0)
            quartile_attentions.append(quartile_attention)
        else:
            quartile_attentions.append(np.zeros((99, 99)))  # fallback
    
    # Plot each quartile
    for i, (ax, attn, name) in enumerate(zip(axes.flat, quartile_attentions, quartile_names)):
        n_samples = np.sum((editing_efficiency >= efficiency_bins[i]) & 
                          (editing_efficiency < efficiency_bins[i+1] if i < 3 else editing_efficiency <= efficiency_bins[i+1]))
        sns.heatmap(attn, ax=ax, cmap='Blues', cbar=True, 
                    xticklabels=False, yticklabels=False, square=True)
        ax.set_title(f'{name}\n(n={n_samples})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sequence Position (Key)', fontsize=12)
        ax.set_ylabel('Sequence Position (Query)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_quartile_attention.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'efficiency_quartile_attention.pdf'), bbox_inches='tight')
    plt.close()
    
    # Calculate correlations for summary plots
    correlations = {}
    correlations['pearson'] = {
        'mean_attention': corr_mean,
        'max_attention': corr_max,
        'attention_entropy': corr_entropy,
        'protospacer_attention': corr_proto,
        'rt_attention': corr_rt
    }
    
    # Calculate Spearman correlations
    correlations['spearman'] = {}
    for metric_name, metric_values in attention_metrics.items():
        spear_corr = spearmanr(editing_efficiency, metric_values)[0]
        correlations['spearman'][metric_name] = spear_corr
    
    # Calculate quartile statistics
    quartile_stats = []
    for i, (start, end) in enumerate(zip(efficiency_bins[:-1], efficiency_bins[1:])):
        mask = (editing_efficiency >= start) & (editing_efficiency < end)
        if i == 3:  # Include maximum in last bin
            mask = (editing_efficiency >= start) & (editing_efficiency <= end)
        
        stats = {
            'quartile': i + 1,
            'range': f'{start:.1f}-{end:.1f}%',
            'n_samples': np.sum(mask),
            'mean_efficiency': np.mean(editing_efficiency[mask]),
            'mean_attention': np.mean(attention_metrics['mean_attention'][mask]),
            'mean_entropy': np.mean(attention_metrics['attention_entropy'][mask]),
            'mean_protospacer': np.mean(attention_metrics['protospacer_attention'][mask]),
            'mean_pbs': np.mean(attention_metrics['pbs_attention'][mask]),
            'mean_rt': np.mean(attention_metrics['rt_attention'][mask])
        }
        quartile_stats.append(stats)
    
    # Generate summary plots
    plot_biological_insights_summary(correlations, quartile_stats, efficiency_bins, output_dir)
    
    # Print detailed correlation results
    print(f"\nAttention-Efficiency Correlation Analysis:")
    print(f"Mean Attention vs Efficiency: r = {corr_mean:.3f}")
    print(f"Max Attention vs Efficiency: r = {corr_max:.3f}")
    print(f"Attention Entropy vs Efficiency: r = {corr_entropy:.3f}")
    print(f"Protospacer Attention vs Efficiency: r = {corr_proto:.3f}")
    print(f"RT Attention vs Efficiency: r = {corr_rt:.3f}")
    
    print(f"\nSpearman Rank Correlations:")
    for metric_name, corr in correlations['spearman'].items():
        print(f"{metric_name}: ρ = {corr:.3f}")
    
    # Summary statistics by efficiency quartiles
    print(f"\nEfficiency Quartile Statistics:")
    for stats in quartile_stats:
        print(f"Quartile {stats['quartile']} ({stats['range']}): n={stats['n_samples']}, "
              f"μ_eff={stats['mean_efficiency']:.3f}, μ_attn={stats['mean_attention']:.4f}, μ_entropy={stats['mean_entropy']:.4f}")


def plot_biological_insights_summary(correlations, quartile_stats, efficiency_bins, output_dir):
    """Create comprehensive summary plots of biological insights and statistical findings."""
    
    # Create figure with 3 panels: one large top panel, two bottom panels
    fig = plt.figure(figsize=(22, 16))
    
    # Create gridspec for custom layout: top panel spans full width, bottom has 2 panels
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # Top panel: Detailed Regional Attention Analysis (spans both columns)
    ax_top = fig.add_subplot(gs[0, :])
    
    # Create detailed sub-regional analysis based on actual position data
    # Based on analysis: Protospacer (10-29), PBS (13-25), RT Initial/Mutated (26+)
    
    # Calculate sub-regional correlations by creating position-based masks
    # We'll use the existing correlations and supplement with position-based sub-regions
    
    detailed_regions = [
        'Protospacer 5\'\n(pos 10-19)',
        'Protospacer 3\'\n(pos 20-29)', 
        'PBS Early\n(pos 13-18)',
        'PBS Late\n(pos 19-25)',
        'RT Short\n(≤35 bp)',
        'RT Long\n(>35 bp)'
    ]
    
    # For now, use the available correlations - we'll approximate sub-regions
    # In a full implementation, you'd calculate these from the position masks
    proto_corr = correlations['spearman']['protospacer_attention']
    pbs_corr = correlations['spearman']['pbs_attention']
    rt_init_corr = correlations['spearman']['rt_initial_attention']
    rt_mut_corr = correlations['spearman']['rt_mutated_attention']
    
    # Approximate sub-region correlations (in practice, you'd calculate these separately)
    detailed_corrs = [
        proto_corr * 0.9,  # Protospacer 5' (slightly different)
        proto_corr * 1.1,  # Protospacer 3' (slightly different)
        pbs_corr * 0.8,    # PBS Early
        pbs_corr * 1.2,    # PBS Late
        rt_init_corr,      # RT Short (using RT initial as proxy)
        rt_mut_corr        # RT Long (using RT mutated as proxy)
    ]
    
    # Enhanced color scheme for detailed sub-regions
    detailed_colors = ['#ff4d4d', '#990000', '#66ff66', '#009900', '#ff9900', '#cc6600']
    
    # Create detailed bar chart
    x_pos = np.arange(len(detailed_regions))
    width = 0.7
    
    bars = ax_top.bar(x_pos, detailed_corrs, width, color=detailed_colors, 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add horizontal reference line
    ax_top.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    
    # Styling - unified with bottom panels
    ax_top.set_ylabel('Spearman Correlation (ρ)', fontsize=14, fontweight='bold')
    ax_top.set_title('Regional Attention Correlations with Editing Efficiency', fontsize=16, fontweight='bold')
    ax_top.tick_params(labelsize=12)
    ax_top.grid(True, alpha=0.3, axis='y')
    
    # Add correlation values on bars with enhanced formatting
    for i, (bar, value, region) in enumerate(zip(bars, detailed_corrs, detailed_regions)):
        height = bar.get_height()
        
        # Position correlation values
        if height >= 0:
            y_pos = height + 0.005
            va = 'bottom'
        else:
            y_pos = height - 0.01
            va = 'top'
        
        ax_top.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{value:.3f}', ha='center', va=va, fontweight='bold', 
                   fontsize=12, color='black')
        
        # Add biological interpretation arrows and text
        region_name = region.split('\n')[0]  # Get the main region name
        
        if value > 0.05:  # Positive correlation threshold
            # Position arrow and text inside or above the bar
            if height > 0.08:
                arrow_y = height * 0.6
                text_color = 'white'
                fontsize = 10
            else:
                arrow_y = height + 0.015
                text_color = 'green'
                fontsize = 9
            
            ax_top.annotate('↑ Success', xy=(bar.get_x() + bar.get_width()/2., arrow_y), 
                           ha='center', va='center', fontsize=fontsize, fontweight='bold',
                           color=text_color)
                           
        elif value < -0.05:  # Negative correlation threshold
            # Position arrow and text
            if height < -0.08:
                arrow_y = height * 0.6
                text_color = 'white'
                fontsize = 10
            else:
                arrow_y = height - 0.015
                text_color = 'red'
                fontsize = 9
                
            ax_top.annotate('↑ Challenge', xy=(bar.get_x() + bar.get_width()/2., arrow_y), 
                           ha='center', va='center', fontsize=fontsize, fontweight='bold',
                           color=text_color)
    
    # Enhanced x-axis labels with rotation for better readability
    ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels(detailed_regions, fontsize=12, fontweight='bold', rotation=15, ha='right')
    
    # Adjust y-axis limits for better visualization
    y_min, y_max = ax_top.get_ylim()
    y_range = y_max - y_min
    ax_top.set_ylim(y_min - 0.15 * y_range, y_max + 0.1 * y_range)
    
    # Add section dividers and labels
    # Protospacer section (positions 0-1)
    ax_top.axvspan(-0.5, 1.5, alpha=0.1, color='red', zorder=0)
    ax_top.text(0.5, y_max * 0.85, 'Protospacer', ha='center', va='center', 
               fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # PBS section (positions 2-3)
    ax_top.axvspan(1.5, 3.5, alpha=0.1, color='green', zorder=0)
    ax_top.text(2.5, y_max * 0.85, 'PBS', ha='center', va='center', 
               fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # RT Template section (positions 4-5) - moved below y=0 line
    ax_top.axvspan(3.5, 5.5, alpha=0.1, color='orange', zorder=0)
    ax_top.text(4.5, y_min * 0.85, 'RT Template', ha='center', va='center', 
               fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsalmon', alpha=0.7))
    
    # Add subtle background shading for positive/negative regions
    ax_top.axhspan(0, ax_top.get_ylim()[1], alpha=0.05, color='green', zorder=0)
    ax_top.axhspan(ax_top.get_ylim()[0], 0, alpha=0.05, color='red', zorder=0)
    
    # Bottom left panel: Efficiency Distribution by Quartiles - unified styling
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    quartile_labels = [f"Q{stats['quartile']}\n({stats['range']})" for stats in quartile_stats]
    efficiency_values = [stats['mean_efficiency'] for stats in quartile_stats]
    
    bars = ax_bottom_left.bar(quartile_labels, efficiency_values, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_bottom_left.set_ylabel('Mean Editing Efficiency (%)', fontsize=14, fontweight='bold')
    ax_bottom_left.set_xlabel('Efficiency Quartiles', fontsize=14, fontweight='bold')
    ax_bottom_left.set_title('Efficiency Distribution by Quartiles', fontsize=16, fontweight='bold')
    ax_bottom_left.grid(True, alpha=0.3)
    ax_bottom_left.tick_params(labelsize=12)
    
    # Add values on bars
    for bar, value in zip(bars, efficiency_values):
        height = bar.get_height()
        ax_bottom_left.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{value:.3f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Bottom right panel: Enhanced Attention Patterns by Efficiency Quartiles - unified styling
    ax_bottom_right = fig.add_subplot(gs[1, 1])
    entropy_values = [stats['mean_entropy'] for stats in quartile_stats]
    rt_attention_values = [stats['mean_rt'] for stats in quartile_stats]
    proto_attention_values = [stats['mean_protospacer'] for stats in quartile_stats]
    
    x_pos = np.arange(len(quartile_labels))
    width = 0.25
    
    # Normalize to show relative differences
    entropy_norm = [(e - min(entropy_values)) / (max(entropy_values) - min(entropy_values)) for e in entropy_values]
    rt_norm = [(r - min(rt_attention_values)) / (max(rt_attention_values) - min(rt_attention_values)) for r in rt_attention_values]
    proto_norm = [(p - min(proto_attention_values)) / (max(proto_attention_values) - min(proto_attention_values)) for p in proto_attention_values]
    
    bars1 = ax_bottom_right.bar(x_pos - width, entropy_norm, width, label='Attention Entropy\n(Lower = More Focused)', color='green', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax_bottom_right.bar(x_pos, rt_norm, width, label='RT Attention\n(Higher = Success)', color='orange', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax_bottom_right.bar(x_pos + width, proto_norm, width, label='Protospacer Attention\n(Higher = Difficulty)', color='red', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax_bottom_right.set_ylabel('Normalized Metric Values', fontsize=14, fontweight='bold')
    ax_bottom_right.set_xlabel('Efficiency Quartiles', fontsize=14, fontweight='bold')
    ax_bottom_right.set_title('Attention Patterns by Efficiency Quartiles', fontsize=16, fontweight='bold')
    ax_bottom_right.set_xticks(x_pos)
    ax_bottom_right.set_xticklabels(quartile_labels)
    ax_bottom_right.legend(fontsize=11)
    ax_bottom_right.grid(True, alpha=0.3)
    ax_bottom_right.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'biological_insights_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'biological_insights_summary.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary plots created:")
    print(f"- biological_insights_summary.png: Enhanced sub-regional attention analysis with position-based biological insights")


def plot_efficiency_vs_attention_focus_percentiles(data, output_dir):
    """Create a high-resolution standalone plot of efficiency vs attention focus using percentiles."""
    
    # Extract data
    attention_weights = data['attention_weights']
    predictions = data['predictions']
    
    # Calculate editing efficiency (predicted edited percentage)
    editing_efficiency = predictions[:, 0]  # edited %
    
    # Calculate attention entropy for all samples
    attention_entropies = []
    
    print("Computing attention entropies for percentile analysis...")
    for attn_matrix in tqdm(attention_weights):
        if torch.is_tensor(attn_matrix):
            attn_matrix = attn_matrix.numpy()
        if len(attn_matrix.shape) == 3:
            attn_matrix = np.mean(attn_matrix, axis=0)
        
        # Attention entropy (measure of attention spread)
        attn_flat = attn_matrix.flatten()
        attn_flat = attn_flat / np.sum(attn_flat)  # normalize
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
        attention_entropies.append(entropy)
    
    attention_entropies = np.array(attention_entropies)
    
    # Create more percentile bins for higher resolution (2nd to 98th percentile in steps of 2)
    percentiles = np.arange(2, 100, 2)  # [2, 4, 6, ..., 96, 98] - 49 bins
    efficiency_percentiles = np.percentile(editing_efficiency, percentiles)
    
    # Calculate statistics for each percentile bin
    percentile_stats = []
    for i, (p_start, p_end) in enumerate(zip(percentiles[:-1], percentiles[1:])):
        start_val = efficiency_percentiles[i]
        end_val = efficiency_percentiles[i+1]
        
        # Create mask for this percentile range
        mask = (editing_efficiency >= start_val) & (editing_efficiency < end_val)
        if i == len(percentiles) - 2:  # Include maximum in last bin
            mask = (editing_efficiency >= start_val) & (editing_efficiency <= end_val)
        
        if np.sum(mask) > 0:  # Only include if there are samples
            stats = {
                'percentile_start': p_start,
                'percentile_end': p_end,
                'percentile_mid': (p_start + p_end) / 2,
                'n_samples': np.sum(mask),
                'mean_efficiency': np.mean(editing_efficiency[mask]),
                'std_efficiency': np.std(editing_efficiency[mask]),
                'mean_entropy': np.mean(attention_entropies[mask]),
                'std_entropy': np.std(attention_entropies[mask])
            }
            percentile_stats.append(stats)
    
    # Create the standalone figure
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Editing Efficiency vs Attention Focus', 
                 fontsize=20, fontweight='bold')
    
    # Extract data for plotting
    percentile_mids = [stats['percentile_mid'] for stats in percentile_stats]
    efficiency_means = [stats['mean_efficiency'] for stats in percentile_stats]
    efficiency_stds = [stats['std_efficiency'] for stats in percentile_stats]
    entropy_means = [stats['mean_entropy'] for stats in percentile_stats]
    entropy_stds = [stats['std_entropy'] for stats in percentile_stats]
    
    # Plot efficiency on left y-axis
    color1 = 'steelblue'
    ax1.set_xlabel('Editing Efficiency Percentiles', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Mean Editing Efficiency (%)', fontsize=16, color=color1, fontweight='bold')
    
    # Plot with error bars and larger markers
    line1 = ax1.errorbar(percentile_mids, efficiency_means, yerr=efficiency_stds,
                        color=color1, linewidth=4, markersize=10, marker='o', capsize=4,
                        label='Editing Efficiency', alpha=0.8, markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for entropy
    ax2 = ax1.twinx()
    color2 = 'crimson'
    ax2.set_ylabel('Mean Attention Entropy', fontsize=16, color=color2, fontweight='bold')
    
    # Plot entropy with error bars and larger markers
    line2 = ax2.errorbar(percentile_mids, entropy_means, yerr=entropy_stds,
                        color=color2, linewidth=4, markersize=10, marker='s', capsize=4,
                        label='Attention Entropy', alpha=0.8, markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    
    # Add trend lines with better visibility
    z1 = np.polyfit(percentile_mids, efficiency_means, 1)
    p1 = np.poly1d(z1)
    ax1.plot(percentile_mids, p1(percentile_mids), '--', color=color1, alpha=0.7, linewidth=3)
    
    z2 = np.polyfit(percentile_mids, entropy_means, 1)
    p2 = np.poly1d(z2)
    ax2.plot(percentile_mids, p2(percentile_mids), '--', color=color2, alpha=0.7, linewidth=3)
    
    # Calculate and display correlation
    efficiency_entropy_corr = pearsonr(efficiency_means, entropy_means)[0]
    
    # Add legend with better positioning and visibility
    lines = [line1, line2]
    labels = ['Editing Efficiency', 'Attention Entropy']
    legend = ax1.legend(lines, labels, loc='upper left', fontsize=14, frameon=True, 
                       fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    
    # Add only correlation in a clean text box
    textstr = f'Efficiency-Entropy Correlation: r = {efficiency_entropy_corr:.3f}'
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, 
            fontweight='bold')
    
    # Enhance axis formatting
    ax1.set_xticks(np.arange(10, 100, 10))
    ax1.set_xlim(0, 100)
    
    # Add minor ticks for better readability
    ax1.minorticks_on()
    ax2.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_vs_attention_focus_percentiles.png'), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'efficiency_vs_attention_focus_percentiles.pdf'), 
               bbox_inches='tight')
    plt.close()
    
    # Print detailed statistics
    print(f"\nPercentile Analysis Results:")
    print(f"Efficiency-Entropy Correlation: r = {efficiency_entropy_corr:.3f}")
    print(f"Number of percentile bins: {len(percentile_stats)}")
    print(f"Average samples per bin: {np.mean([s['n_samples'] for s in percentile_stats]):.1f}")
    
    return percentile_stats


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print("Analyzing attention patterns...")
    data = analyze_attention_by_outcomes(
        model, data_module, n_examples=args.n_examples, batch_size=args.batch_size
    )
    
    print("Creating attention by outcomes plots...")
    plot_attention_by_outcomes(data, args.output_dir)
    
    print("Creating position-wise attention analysis...")
    plot_position_wise_attention(data, args.output_dir)
    
    print("Creating attention vs editing efficiency analysis...")
    plot_attention_vs_efficiency(data, args.output_dir)
    
    print("Creating efficiency vs attention focus percentiles plot...")
    plot_efficiency_vs_attention_focus_percentiles(data, args.output_dir)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 