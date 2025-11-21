"""
Regression performance comparison script for crispAIPE-regression vs competing models (DeepPrime and EasyPrime).
Compares Spearman correlations for point prediction task on DeepPrime test data.

example cmd:
python test/regression_performance_comparison.py --config pe_uncert_models/configs/crispAIPE_regression_deepprime_conf.json --checkpoint pe_uncert_models/logs/crispAIPE_regression_deepprime_conf/2025-06-25-23-13-43/best_model-epoch=08-val_loss_val_loss=0.0018.ckpt --output_dir test/figures/train_test_split_model
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
import json
from tqdm import tqdm

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE_regression import crispAIPERegression
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Compare crispAIPE-regression performance with competing models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test/figures/train_test_split_model', 
                       help='Output directory for figures')
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
    
    # Fix data paths - make them relative to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Convert relative paths to absolute paths
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Handle paths that start with ../
                if config_data[path_key].startswith('../'):
                    config_data[path_key] = os.path.normpath(
                        os.path.join(project_root, config_data[path_key][3:])
                    )
                else:
                    config_data[path_key] = os.path.normpath(
                        os.path.join(project_root, config_data[path_key])
                    )
    
    # Load dataset
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model from checkpoint
    model = crispAIPERegression.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    return model, data_module, config_data


def calculate_crispAIPE_regression_correlations(model, data_module, batch_size=64):
    """Calculate Spearman correlation for crispAIPE-regression model."""
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_targets = []
    
    print("Calculating crispAIPE-regression correlations...")
    for batch in tqdm(test_loader):
        # Move batch to the same device as model
        batch = [b.to(device) for b in batch]
        
        # Get predictions
        with torch.no_grad():
            x_hat, y_hat = model(batch)
            
            # Unscale predictions to original range
            y_hat_unscaled = model.unscale_predictions(y_hat)
            
            # Get targets (edited percentage)
            _, edited_percentage, _, _ = batch[2:6]
            
            # Store predictions and ground truth
            all_predictions.append(y_hat_unscaled.cpu())
            all_targets.append(edited_percentage.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Flatten arrays if needed
    all_predictions = all_predictions.flatten()
    all_targets = all_targets.flatten()
    
    # Calculate Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(all_targets, all_predictions)
    
    print(f"crispAIPE-regression Results:")
    print(f"  Spearman correlation: {spearman_corr:.3f}")
    print(f"  Dataset size: {len(all_targets)} samples")
    
    return {
        'spearman_correlation': spearman_corr,
        'p_value': spearman_p,
        'n_samples': len(all_targets)
    }


def get_competing_model_performance():
    """Get the performance values for competing models from the literature."""
    # Performance values for DeepPrime test data
    competing_models = {
        'DeepPrime': {
            'spearman_correlation': 0.74
        },
        'EasyPrime': {
            'spearman_correlation': 0.67
        }
    }
    
    return competing_models


def create_regression_performance_comparison_plot(crispAIPE_results, competing_models, output_dir):
    """Create a bar plot comparing regression performance across all models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for plotting
    models = ['crispAIPE-regression', 'DeepPrime', 'EasyPrime']
    metrics = ['Spearman\nCorrelation']
    
    # Extract values for each model
    data = [
        [crispAIPE_results['spearman_correlation'], 
         competing_models['DeepPrime']['spearman_correlation'],
         competing_models['EasyPrime']['spearman_correlation']]
    ]
    
    # Create the plot
    plt.style.use('default')  # Use default style to avoid gray background
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the bar positions
    x = np.arange(len(models))
    width = 0.6
    
    # Create bars for each model
    bars = ax.bar(x, data[0], width, 
                  color=['#2E86AB', '#A23B72', '#F18F01'], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel('Spearman Correlation (ρ)', fontsize=16)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits with some padding
    all_values = data[0]
    y_min = max(0, min(all_values) - 0.05)
    y_max = min(1, max(all_values) + 0.05)
    ax.set_ylim(y_min, y_max)
    
    # Add horizontal grid lines
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    
    # Set white background and remove outer frame
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'regression_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'regression_performance_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Regression performance comparison plot saved to {plot_path}")
    
    return crispAIPE_results['spearman_correlation']


def save_regression_performance_results(crispAIPE_results, competing_models, crispAIPE_spearman, output_dir):
    """Save detailed regression performance results to a JSON file."""
    results = {
        'crispAIPE_regression': {
            'spearman_correlation': float(crispAIPE_results['spearman_correlation']),
            'p_value': float(crispAIPE_results['p_value']),
            'n_samples': crispAIPE_results['n_samples']
        },
        'DeepPrime': {
            'spearman_correlation': competing_models['DeepPrime']['spearman_correlation']
        },
        'EasyPrime': {
            'spearman_correlation': competing_models['EasyPrime']['spearman_correlation']
        }
    }
    
    results_path = os.path.join(output_dir, 'regression_performance_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed regression performance results saved to {results_path}")


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module, config_data = load_model_and_data(args.config, args.checkpoint)
    
    # Calculate crispAIPE-regression correlations
    crispAIPE_results = calculate_crispAIPE_regression_correlations(model, data_module, args.batch_size)
    
    # Get competing model performance
    competing_models = get_competing_model_performance()
    
    # Create regression performance comparison plot
    print("Creating regression performance comparison plot...")
    crispAIPE_spearman = create_regression_performance_comparison_plot(
        crispAIPE_results, competing_models, args.output_dir
    )
    
    # Save detailed results
    save_regression_performance_results(crispAIPE_results, competing_models, 
                                     crispAIPE_spearman, args.output_dir)
    
    # Print summary
    print(f"\n" + "="*60)
    print("REGRESSION PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    print(f"crispAIPE-regression: {crispAIPE_spearman:.3f}")
    print(f"DeepPrime: {competing_models['DeepPrime']['spearman_correlation']:.3f}")
    print(f"EasyPrime: {competing_models['EasyPrime']['spearman_correlation']:.3f}")
    print(f"\ncrispAIPE-regression vs DeepPrime: {crispAIPE_spearman - competing_models['DeepPrime']['spearman_correlation']:+.3f}")
    print(f"crispAIPE-regression vs EasyPrime: {crispAIPE_spearman - competing_models['EasyPrime']['spearman_correlation']:+.3f}")
    print("="*60)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
