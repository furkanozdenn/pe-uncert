"""
Script to calculate Spearman correlations between predicted and actual values
for a trained crispAIPE model on test data.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import seaborn as sns

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Spearman correlations for model predictions')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--calc_all', action='store_true', help='Calculate correlations for all outcomes, not just edited %')
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
                # Make path absolute relative to config file location
                config_data[path_key] = os.path.normpath(
                    os.path.join(config_dir, config_data[path_key])
                )
            print(f"Using {path_key}: {config_data[path_key]}")
    
    # Load dataset
    print(f"Loading dataset with config: {config_data}")
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model from checkpoint
    model = crispAIPE.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    return model, data_module


def calculate_correlations(model, data_module, output_dir, calc_all=False):
    """Calculate Spearman and Pearson correlations between predictions and ground truth."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Process all test batches
    for batch in test_loader:
        # Move batch to the same device as model
        batch = [b.to(device) for b in batch]
        
        # Get predictions
        with torch.no_grad():
            _, alpha_params = model(batch)
        
        # Extract ground truth proportions
        _, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
        
        # Stack the ground truth proportions
        ground_truth = torch.stack([
            edited_percentage, 
            unedited_percentage, 
            indel_percentage
        ], dim=1)
        
        # Ensure proportions sum to 1
        ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
        
        # Calculate expected values from Dirichlet distribution
        alpha_sum = torch.sum(alpha_params, dim=1, keepdim=True)
        expected_props = alpha_params / alpha_sum
        
        # Store predictions and ground truth
        all_predictions.append(expected_props.cpu())
        all_ground_truth.append(ground_truth.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_ground_truth = torch.cat(all_ground_truth, dim=0).numpy()
    
    # Extract individual components
    pred_edited = all_predictions[:, 0]
    pred_unedited = all_predictions[:, 1]
    pred_indel = all_predictions[:, 2]
    
    true_edited = all_ground_truth[:, 0]
    true_unedited = all_ground_truth[:, 1]
    true_indel = all_ground_truth[:, 2]
    
    # Calculate Spearman correlations
    spearman_edited, p_spearman_edited = spearmanr(true_edited, pred_edited)
    pearson_edited, p_pearson_edited = pearsonr(true_edited, pred_edited)
    
    # Create a scatter plot for edited percentage
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    
    # Create a DataFrame for better visualization with Seaborn
    df = pd.DataFrame({
        'True Edited %': true_edited,
        'Predicted Edited %': pred_edited
    })
    
    # Use Seaborn for better aesthetics
    sns.scatterplot(x='True Edited %', y='Predicted Edited %', data=df, alpha=0.6)
    
    # Add regression line
    sns.regplot(x='True Edited %', y='Predicted Edited %', data=df, scatter=False, color='red')
    
    # Add perfect prediction line (y=x)
    min_val = min(df['True Edited %'].min(), df['Predicted Edited %'].min())
    max_val = max(df['True Edited %'].max(), df['Predicted Edited %'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Add correlation coefficients to the plot
    plt.annotate(f'Spearman ρ: {spearman_edited:.4f} (p: {p_spearman_edited:.4e})\n'
                f'Pearson r: {pearson_edited:.4f} (p: {p_pearson_edited:.4e})',
                xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top')
    
    plt.title('Correlation between Predicted and Actual Edited Percentage')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    edited_plot_path = os.path.join(output_dir, 'edited_correlation.png')
    plt.savefig(edited_plot_path)
    print(f"Edited percentage correlation plot saved to {edited_plot_path}")
    
    # If calc_all is True, calculate and visualize correlations for all outcome types
    if calc_all:
        # Calculate correlations for all types
        spearman_unedited, p_spearman_unedited = spearmanr(true_unedited, pred_unedited)
        spearman_indel, p_spearman_indel = spearmanr(true_indel, pred_indel)
        
        pearson_unedited, p_pearson_unedited = pearsonr(true_unedited, pred_unedited)
        pearson_indel, p_pearson_indel = pearsonr(true_indel, pred_indel)
        
        # Calculate overall correlation by combining all outcomes
        all_pred = all_predictions.flatten()
        all_true = all_ground_truth.flatten()
        spearman_overall, p_spearman_overall = spearmanr(all_true, all_pred)
        pearson_overall, p_pearson_overall = pearsonr(all_true, all_pred)
        
        # Create a DataFrame for the results
        results = pd.DataFrame({
            'Outcome Type': ['Edited', 'Unedited', 'Indel', 'Overall'],
            'Spearman ρ': [spearman_edited, spearman_unedited, spearman_indel, spearman_overall],
            'Spearman p-value': [p_spearman_edited, p_spearman_unedited, p_spearman_indel, p_spearman_overall],
            'Pearson r': [pearson_edited, pearson_unedited, pearson_indel, pearson_overall],
            'Pearson p-value': [p_pearson_edited, p_pearson_unedited, p_pearson_indel, p_pearson_overall]
        })
        
        # Save the results to a CSV file
        results_path = os.path.join(output_dir, 'correlation_results.csv')
        results.to_csv(results_path, index=False)
        print(f"Correlation results saved to {results_path}")
        
        # Create subplots for all outcome types
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Flatten axes array for easier indexing
        axes = axes.flatten()
        
        # Create scatter plots for each outcome type
        for i, (pred, true, title, corr) in enumerate([
            (pred_edited, true_edited, 'Edited %', spearman_edited),
            (pred_unedited, true_unedited, 'Unedited %', spearman_unedited),
            (pred_indel, true_indel, 'Indel %', spearman_indel),
            (all_pred, all_true, 'All Outcomes Combined', spearman_overall)
        ]):
            ax = axes[i]
            
            df_temp = pd.DataFrame({
                f'True {title}': true,
                f'Predicted {title}': pred
            })
            
            sns.scatterplot(x=f'True {title}', y=f'Predicted {title}', data=df_temp, alpha=0.6, ax=ax)
            sns.regplot(x=f'True {title}', y=f'Predicted {title}', data=df_temp, scatter=False, color='red', ax=ax)
            
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
            
            ax.annotate(f'Spearman ρ: {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                       ha='left', va='top')
            
            ax.set_title(f'Correlation for {title}')
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        all_plot_path = os.path.join(output_dir, 'all_correlations.png')
        plt.savefig(all_plot_path)
        print(f"All correlations plot saved to {all_plot_path}")
    
    # Return the correlation results for edited percentage
    return {
        'spearman_edited': spearman_edited,
        'p_spearman_edited': p_spearman_edited,
        'pearson_edited': pearson_edited,
        'p_pearson_edited': p_pearson_edited
    }


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading model from {args.checkpoint}")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print(f"Calculating correlations")
    results = calculate_correlations(model, data_module, args.output_dir, args.calc_all)
    
    # Print the results
    print("\n" + "="*50)
    print("CORRELATION RESULTS FOR EDITED PERCENTAGE:")
    print("="*50)
    print(f"Spearman ρ: {results['spearman_edited']:.4f} (p-value: {results['p_spearman_edited']:.4e})")
    print(f"Pearson r: {results['pearson_edited']:.4f} (p-value: {results['p_pearson_edited']:.4e})")
    print("="*50 + "\n")
    
    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
