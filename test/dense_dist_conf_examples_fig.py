"""
Script to create a 4x4 grid of distribution examples and confidence intervals.
First two rows (8 figures): predicted distributions
Last two rows (8 figures): confidence intervals
Examples are ordered by increasing edited activity.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import dirichlet, chi2, gaussian_kde
from sklearn.decomposition import PCA

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Create 4x4 grid of distribution examples and confidence intervals')
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
    """
    return simplex_points[:, :2]


def compute_pca_confidence_region(samples_2d, confidence_level=0.95):
    """
    Compute PCA-based confidence region around the mean of 2D samples.
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
    confidence_scale = np.sqrt(chi2.ppf(confidence_level, df=2))
    
    return mean, principal_axes, eigenvalues, confidence_scale


def generate_predictions_and_samples(model, data_module, n_samples=5000, batch_size=64):
    """
    Generate predictions and sample distributions for the test set.
    
    Returns:
        predictions: (N, 3) array of predicted means
        alpha_params_all: (N, 3) array of alpha parameters
        samples: (N, n_samples, 3) array of samples from predicted distributions
        ground_truth: (N, 3) array of ground truth proportions
    """
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_alpha_params = []
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
            all_alpha_params.append(alpha_params.cpu().numpy())
            all_samples.append(samples)
            all_ground_truth.append(ground_truth.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    alpha_params_all = np.vstack(all_alpha_params)
    samples = np.vstack(all_samples)
    ground_truth = np.vstack(all_ground_truth)
    
    return predictions, alpha_params_all, samples, ground_truth


def plot_distribution_example(ax, alpha, ground_truth, sample_idx):
    """Plot a single distribution example using the same style as sample_distributions.py."""
    # Plot ternary grid (same as sample_distributions.py)
    corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for k in range(11):
        p1 = corners[0] * (10-k)/10 + corners[1] * k/10
        p2 = corners[0] * (10-k)/10 + corners[2] * k/10
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.2)
        
        p1 = corners[1] * (10-k)/10 + corners[0] * k/10
        p2 = corners[1] * (10-k)/10 + corners[2] * k/10
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.2)
        
        p1 = corners[2] * (10-k)/10 + corners[0] * k/10
        p2 = corners[2] * (10-k)/10 + corners[1] * k/10
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.2)
    
    # Sample from Dirichlet distribution
    dir_samples = dirichlet.rvs(alpha, size=5000)
    
    # Calculate density
    xy = dir_samples[:, :2]  # Use first two dimensions for 2D plot
    z = gaussian_kde(xy.T)(xy.T)
    
    # Sort by density for better visualization
    idx = z.argsort()
    x, y, z = xy[idx, 0], xy[idx, 1], z[idx]
    
    # Plot scatter with color representing density
    scatter = ax.scatter(x, y, c=z, s=20, alpha=0.6, cmap='viridis')
    
    # Plot ground truth
    ax.scatter(ground_truth[0], ground_truth[1], color='red', s=100, marker='*', 
               label='Ground Truth', zorder=5)
    
    # Calculate expected value of Dirichlet
    expected = alpha / alpha.sum()
    ax.scatter(expected[0], expected[1], color='blue', s=100, marker='o', 
               label='Predicted Mean', zorder=5)
    
    # Calculate alpha0 (precision/concentration)
    alpha0 = alpha.sum()
    
    # Remove title and set axis labels
    ax.set_xlabel('Edited %', fontsize=8)
    ax.set_ylabel('Unedited %', fontsize=8)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add legend only for first subplot
    if sample_idx == 0:
        ax.legend(fontsize=7, loc='upper right')
    
    # Add annotations for ground truth and expected values (smaller for subplot)
    ax.annotate(f"GT: [{ground_truth[0]:.3f}, {ground_truth[1]:.3f}, {ground_truth[2]:.3f}]", 
               xy=(ground_truth[0], ground_truth[1]), xytext=(ground_truth[0]-0.2, ground_truth[1]-0.1),
               arrowprops=dict(arrowstyle="->"), fontsize=6)
    
    ax.annotate(f"Pred: [{expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f}]", 
               xy=(expected[0], expected[1]), xytext=(expected[0]-0.2, expected[1]+0.1),
               arrowprops=dict(arrowstyle="->"), fontsize=6)


def plot_confidence_region_example(ax, samples_2d, ground_truth_2d, prediction_2d, sample_idx):
    """Plot a single confidence region example."""
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
    
    # Show edited percentage in title
    edited_pct = ground_truth_2d[0]
    ax.set_title(f'Edited: {edited_pct:.1%}', fontsize=9)
    ax.set_xlabel('Edited %', fontsize=8)
    ax.set_ylabel('Unedited %', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add legend only for first subplot
    if sample_idx == 0:
        ax.legend(fontsize=8, loc='upper right')


def create_dense_distribution_figure(predictions, alpha_params_all, samples, ground_truth, output_dir):
    """Create the 4x4 grid of distribution examples and confidence intervals."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sort examples by increasing edited activity (ground truth edited percentage)
    sorted_indices = np.argsort(ground_truth[:, 0])
    
    # Select 16 examples more evenly distributed across the sorted range
    n_examples = 16
    n_total = len(sorted_indices)
    # Use percentiles to get better distribution
    percentiles = np.linspace(0, 95, n_examples)  # 0% to 95% to avoid extreme outliers
    selected_indices = []
    for p in percentiles:
        idx = int(p * n_total / 100)
        selected_indices.append(sorted_indices[idx])
    selected_indices = np.array(selected_indices)
    
    print(f"Selected examples with edited activity range: {ground_truth[selected_indices, 0].min():.3f} to {ground_truth[selected_indices, 0].max():.3f}")
    
    # Create the 4x4 grid
    plt.style.use('default')
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # First two rows: distribution examples
    for i in range(8):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        idx = selected_indices[i]
        alpha = alpha_params_all[idx]  # Use the actual alpha parameters from the model
        gt = ground_truth[idx]
        
        plot_distribution_example(ax, alpha, gt, i)
    
    # Last two rows: confidence region examples
    for i in range(8, 16):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        idx = selected_indices[i]
        samples_2d = simplex_to_2d(samples[idx])
        ground_truth_2d = simplex_to_2d(ground_truth[idx:idx+1])[0]
        prediction_2d = simplex_to_2d(predictions[idx:idx+1])[0]
        
        plot_confidence_region_example(ax, samples_2d, ground_truth_2d, prediction_2d, i-8)
    
    # Remove overall title and section labels for cleaner look
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.08)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'dense_dist_conf_examples.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'dense_dist_conf_examples.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Dense distribution and confidence interval examples saved to {output_dir}")


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility (using different seed for different examples)
    np.random.seed(123)
    torch.manual_seed(123)
    
    print("Loading model and data...")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print("Generating predictions and samples...")
    predictions, alpha_params_all, samples, ground_truth = generate_predictions_and_samples(
        model, data_module, n_samples=args.n_samples, batch_size=args.batch_size
    )
    
    print(f"Dataset size: {len(predictions)} samples")
    
    print("Creating dense distribution figure...")
    create_dense_distribution_figure(predictions, alpha_params_all, samples, ground_truth, args.output_dir)
    
    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
