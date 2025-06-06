"""
Diagnostics plot script for crispAIPE model
Uses PCA-based confidence intervals on 2D Dirichlet distribution samples.

example cmd:
python test/diagnostics_plot.py --config pe_uncert_models/configs/crispAIPE_conf1.json --checkpoint pe_uncert_models/logs/crispAIPE_conf1/2025-04-20-23-13-47/best_model-epoch=28-val_loss_val_loss=-3.1244.ckpt --output_dir ./figures --n_samples 1000 --batch_size 32
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import dirichlet, chi2
from sklearn.decomposition import PCA

import json
from tqdm import tqdm

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate diagnostics plot for crispAIPE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to draw from distributions')
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


def simplex_to_2d(simplex_points):
    """
    Extract first two dimensions from 3D simplex points (edited %, unedited %).
    The third dimension (indel %) is implicitly 1 - edited % - unedited %.
    This matches the approach used in sample_distributions.py
    """
    # Simply return the first two dimensions: edited % and unedited %
    return simplex_points[:, :2]


def compute_pca_confidence_region(samples_2d, confidence_level=0.95):
    """
    Compute PCA-based confidence region around the mean of 2D samples.
    
    Args:
        samples_2d: (N, 2) array of 2D samples
        confidence_level: confidence level for the region
        
    Returns:
        mean: center of confidence region
        principal_axes: eigenvectors (columns are principal components)
        eigenvalues: eigenvalues of covariance matrix
        confidence_scale: scaling factor for confidence ellipse
    """
    # Center the data
    mean = np.mean(samples_2d, axis=0)
    centered_samples = samples_2d - mean
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(centered_samples)
    
    # Get principal components and explained variance
    principal_axes = pca.components_.T  # columns are principal components
    eigenvalues = pca.explained_variance_
    
    # Calculate confidence scale based on chi-square distribution
    # For 2D, use chi-square with 2 degrees of freedom
    confidence_scale = np.sqrt(chi2.ppf(confidence_level, df=2))
    
    return mean, principal_axes, eigenvalues, confidence_scale


def point_in_confidence_region(point, mean, principal_axes, eigenvalues, confidence_scale):
    """
    Check if a point is within the PCA-based confidence region.
    
    Args:
        point: (2,) array representing the point to check
        mean: center of confidence region
        principal_axes: eigenvectors from PCA
        eigenvalues: eigenvalues from PCA
        confidence_scale: scaling factor for confidence ellipse
        
    Returns:
        bool: True if point is within the confidence region
    """
    # Transform point to PCA space
    centered_point = point - mean
    pca_coords = principal_axes.T @ centered_point
    
    # Check if point is within confidence ellipse
    # Ellipse equation: (x1/sqrt(λ1))² + (x2/sqrt(λ2))² ≤ confidence_scale²
    ellipse_coords = pca_coords / np.sqrt(eigenvalues)
    distance_squared = np.sum(ellipse_coords**2)
    
    return distance_squared <= confidence_scale**2


def generate_predictions_and_samples(model, data_module, n_samples=5000, batch_size=64):
    """
    Generate predictions and sample distributions for the test set.
    
    Returns:
        predictions: (N, 3) array of predicted means
        samples: (N, n_samples, 3) array of samples from predicted distributions
        ground_truth: (N, 3) array of ground truth proportions
    """
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_samples = []
    all_ground_truth = []
    
    print("Generating predictions and samples...")
    for batch in tqdm(test_loader):
        batch = [b.to(device) for b in batch]
        
        with torch.no_grad():
            # Get Dirichlet parameters
            _, alpha_params = model(batch)
            
            # Get ground truth proportions
            _, edited_pct, unedited_pct, indel_pct = batch[2:6]
            ground_truth = torch.stack([edited_pct, unedited_pct, indel_pct], dim=1)
            ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
            
            # Calculate predicted means
            alpha_sum = torch.sum(alpha_params, dim=1, keepdim=True)
            predictions = alpha_params / alpha_sum
            
            # Sample from Dirichlet distributions
            alpha_np = alpha_params.cpu().numpy()
            samples = np.array([dirichlet.rvs(alpha, size=n_samples) for alpha in alpha_np])
            
            all_predictions.append(predictions.cpu().numpy())
            all_samples.append(samples)
            all_ground_truth.append(ground_truth.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    samples = np.vstack(all_samples)
    ground_truth = np.vstack(all_ground_truth)
    
    return predictions, samples, ground_truth


def compute_diagnostics(predictions, samples, ground_truth, confidence_levels):
    """
    Compute diagnostic statistics for PCA-based confidence regions.
    
    Returns:
        observed_coverage: actual coverage rates for each confidence level
        predicted_coverage: expected coverage rates (should match confidence_levels)
    """
    n_samples = len(predictions)
    observed_coverage = []
    
    print("Computing diagnostic statistics...")
    for conf_level in tqdm(confidence_levels):
        coverage_count = 0
        
        for i in range(n_samples):
            # Convert samples to 2D space
            samples_2d = simplex_to_2d(samples[i])
            ground_truth_2d = simplex_to_2d(ground_truth[i:i+1])[0]
            
            # Compute PCA confidence region
            mean, principal_axes, eigenvalues, confidence_scale = compute_pca_confidence_region(
                samples_2d, confidence_level=conf_level
            )
            
            # Check if ground truth is within confidence region
            if point_in_confidence_region(ground_truth_2d, mean, principal_axes, eigenvalues, confidence_scale):
                coverage_count += 1
        
        observed_coverage.append(coverage_count / n_samples)
    
    return np.array(observed_coverage), confidence_levels


def plot_diagnostics(confidence_levels, observed_coverage, output_dir):
    """Create and save the diagnostic plot."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
    
    # Observed coverage
    ax.plot(confidence_levels, observed_coverage, 'b-', linewidth=2, marker='o', 
            markersize=6, label='crispAIPE (PCA-based)')
    
    # Formatting
    ax.set_xlabel('Predicted Confidence Level', fontsize=14)
    ax.set_ylabel('Observed Coverage Rate', fontsize=14)
    ax.set_title('Uncertainty Calibration - crispAIPE Model', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add text box with statistics
    mae = np.mean(np.abs(observed_coverage - confidence_levels))
    textstr = f'Mean Absolute Error: {mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crispAIPE_diagnostics.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'crispAIPE_diagnostics.pdf'))
    plt.close()
    
    print(f"Diagnostic plot saved to {output_dir}")


def plot_sample_confidence_regions(predictions, samples, ground_truth, output_dir, n_examples=4):
    """Plot examples of confidence regions in 2D space."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Select random examples
    indices = np.random.choice(len(predictions), n_examples, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Convert to 2D
        samples_2d = simplex_to_2d(samples[idx])
        ground_truth_2d = simplex_to_2d(ground_truth[idx:idx+1])[0]
        prediction_2d = simplex_to_2d(predictions[idx:idx+1])[0]
        
        # Plot samples
        ax.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, s=1, color='lightblue')
        
        # Plot confidence regions for different levels
        confidence_levels = [0.5, 0.8, 0.95]
        colors = ['red', 'orange', 'green']
        
        for conf_level, color in zip(confidence_levels, colors):
            mean, principal_axes, eigenvalues, confidence_scale = compute_pca_confidence_region(
                samples_2d, confidence_level=conf_level
            )
            
            # Draw ellipse
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_points = np.column_stack([
                confidence_scale * np.sqrt(eigenvalues[0]) * np.cos(theta),
                confidence_scale * np.sqrt(eigenvalues[1]) * np.sin(theta)
            ])
            
            # Rotate and translate ellipse
            rotated_ellipse = ellipse_points @ principal_axes.T + mean
            ax.plot(rotated_ellipse[:, 0], rotated_ellipse[:, 1], 
                   color=color, linewidth=2, label=f'{int(conf_level*100)}% CI')
        
        # Plot ground truth and prediction
        ax.scatter(*ground_truth_2d, color='red', s=100, marker='*', 
                  label='Ground Truth', zorder=5)
        ax.scatter(*prediction_2d, color='blue', s=100, marker='o', 
                  label='Prediction Mean', zorder=5)
        
        ax.set_title(f'Example {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Edited %')
        ax.set_ylabel('Unedited %')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_regions_examples.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'confidence_regions_examples.pdf'))
    plt.close()
    
    print(f"Confidence region examples saved to {output_dir}")


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print("Generating predictions and samples...")
    predictions, samples, ground_truth = generate_predictions_and_samples(
        model, data_module, n_samples=args.n_samples, batch_size=args.batch_size
    )
    
    print(f"Dataset size: {len(predictions)} samples")
    
    # Define confidence levels to test
    confidence_levels = np.arange(0.1, 1.0, 0.1)
    
    print("Computing diagnostic statistics...")
    observed_coverage, predicted_coverage = compute_diagnostics(
        predictions, samples, ground_truth, confidence_levels
    )
    
    print("Creating diagnostic plots...")
    plot_diagnostics(confidence_levels, observed_coverage, args.output_dir)
    
    print("Creating confidence region examples...")
    plot_sample_confidence_regions(predictions, samples, ground_truth, args.output_dir)
    
    # Print summary statistics
    mae = np.mean(np.abs(observed_coverage - predicted_coverage))
    print(f"\nDiagnostic Results:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Max Error: {np.max(np.abs(observed_coverage - predicted_coverage)):.4f}")
    
    print(f"\nDetailed Results:")
    for conf, obs in zip(confidence_levels, observed_coverage):
        print(f"Confidence Level: {conf:.1f}, Observed Coverage: {obs:.3f}")
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
