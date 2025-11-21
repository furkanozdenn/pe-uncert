"""
Script to generate Spearman correlation performance figures for the regression model.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import pytorch_lightning as pl

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from pe_uncert_models.models.crispAIPE_regression import crispAIPERegression
from pe_uncert_models.data_utils.data import PE_Dataset

def load_model_and_data(checkpoint_path, config_path):
    """Load the trained model and test data."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create data module
    data = PE_Dataset(data_config=config['data_parameters'])
    
    # Load model
    model = crispAIPERegression.load_from_checkpoint(
        checkpoint_path,
        hparams={**config['model_parameters'], **config['data_parameters'], **config['training_parameters']}
    )
    
    return model, data

def get_predictions_and_targets(model, data, device='cpu'):
    """Get predictions and targets from the test set."""
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data.test_dataloader():
            # Move batch to device
            batch = [b.to(device) for b in batch]
            
            # Get predictions
            x_hat, y_hat = model(batch)
            
            # Unscale predictions to original range
            y_hat_unscaled = model.unscale_predictions(y_hat)
            
            # Get targets
            total_read_count, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
            
            # Move to CPU and convert to numpy
            all_predictions.append(y_hat_unscaled.cpu().numpy())
            all_targets.append(edited_percentage.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_predictions, all_targets

def create_correlation_plots(predictions, targets, save_dir):
    """Create various correlation plots."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot with correlation line
    ax1 = axes[0, 0]
    ax1.scatter(targets, predictions, alpha=0.6, s=20, color='steelblue')
    
    # Add correlation line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    ax1.plot(targets, p(targets), "r--", alpha=0.8, linewidth=2)
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.5, linewidth=1, label='Perfect Prediction')
    
    # Calculate correlation
    spearman_corr, spearman_p = stats.spearmanr(targets, predictions)
    pearson_corr, pearson_p = stats.pearsonr(targets, predictions)
    
    ax1.set_xlabel('Ground Truth Editing Score', fontsize=12)
    ax1.set_ylabel('Predicted Editing Score', fontsize=12)
    ax1.text(0.05, 0.95, f'Spearman ρ = {spearman_corr:.4f}\nPearson r = {pearson_corr:.4f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    residuals = predictions - targets
    ax2.scatter(predictions, residuals, alpha=0.6, s=20, color='coral')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add trend line for residuals
    z_res = np.polyfit(predictions, residuals, 1)
    p_res = np.poly1d(z_res)
    ax2.plot(predictions, p_res(predictions), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Predicted Editing Score', fontsize=12)
    ax2.set_ylabel('Residuals (Predicted - Ground Truth)', fontsize=12)
    ax2.text(0.05, 0.95, f'Mean Residual = {residuals.mean():.4f}\nStd Residual = {residuals.std():.4f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(targets, bins=50, alpha=0.7, label='Ground Truth', color='steelblue', density=True)
    ax3.hist(predictions, bins=50, alpha=0.7, label='Predictions', color='coral', density=True)
    ax3.set_xlabel('Editing Score', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error analysis
    ax4 = axes[1, 1]
    abs_errors = np.abs(residuals)
    ax4.hist(abs_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {abs_errors.mean():.3f}')
    ax4.axvline(np.median(abs_errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(abs_errors):.3f}')
    
    ax4.set_xlabel('Absolute Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.text(0.05, 0.95, f'RMSE = {np.sqrt(np.mean(residuals**2)):.3f}', 
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regression_correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'regression_correlation_analysis.pdf'), bbox_inches='tight')
    plt.show()
    
    return spearman_corr, pearson_corr, residuals

def create_detailed_correlation_plot(predictions, targets, save_dir):
    """Create a detailed correlation plot with confidence intervals."""
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(targets, predictions, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add correlation line
    from scipy import stats
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(targets, predictions)
    
    # Generate points for the regression line
    x_line = np.linspace(targets.min(), targets.max(), 100)
    y_line = slope * x_line + intercept
    
    # Plot regression line
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression Line (r = {r_value:.4f})')
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    spearman_corr, spearman_p = stats.spearmanr(targets, predictions)
    mse = np.mean((predictions - targets)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Add text box with metrics
    textstr = f'Spearman ρ = {spearman_corr:.4f}\nPearson r = {r_value:.4f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nR² = {r_value**2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.xlabel('Ground Truth Editing Score', fontsize=14)
    plt.ylabel('Predicted Editing Score', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_correlation_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'detailed_correlation_plot.pdf'), bbox_inches='tight')
    plt.show()

def create_performance_summary(predictions, targets, save_dir):
    """Create a performance summary table and save it."""
    
    # Calculate various metrics
    spearman_corr, spearman_p = stats.spearmanr(targets, predictions)
    pearson_corr, pearson_p = stats.pearsonr(targets, predictions)
    mse = np.mean((predictions - targets)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    r_squared = pearson_corr**2
    
    # Create summary dictionary
    summary = {
        'Spearman Correlation': float(spearman_corr),
        'Spearman p-value': float(spearman_p),
        'Pearson Correlation': float(pearson_corr),
        'Pearson p-value': float(pearson_p),
        'R-squared': float(r_squared),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'Mean Target': float(np.mean(targets)),
        'Std Target': float(np.std(targets)),
        'Mean Prediction': float(np.mean(predictions)),
        'Std Prediction': float(np.std(predictions)),
        'Min Target': float(np.min(targets)),
        'Max Target': float(np.max(targets)),
        'Min Prediction': float(np.min(predictions)),
        'Max Prediction': float(np.max(predictions))
    }
    
    # Save summary to JSON
    with open(os.path.join(save_dir, 'performance_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create and save summary table
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    summary_df.to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)
    
    print("Performance Summary:")
    print("=" * 50)
    for metric, value in summary.items():
        if 'p-value' in metric:
            print(f"{metric}: {value:.2e}")
        else:
            print(f"{metric}: {value:.4f}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Generate correlation plots for regression model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='./regression_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading model and data...")
    model, data = load_model_and_data(args.checkpoint, args.config)
    
    print("Generating predictions...")
    predictions, targets = get_predictions_and_targets(model, data, args.device)
    
    print(f"Generated predictions for {len(predictions)} samples")
    print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    print("Creating correlation plots...")
    spearman_corr, pearson_corr, residuals = create_correlation_plots(predictions, targets, args.save_dir)
    
    print("Creating detailed correlation plot...")
    create_detailed_correlation_plot(predictions, targets, args.save_dir)
    
    print("Creating performance summary...")
    summary = create_performance_summary(predictions, targets, args.save_dir)
    
    print(f"\nResults saved to: {args.save_dir}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")

if __name__ == '__main__':
    main() 