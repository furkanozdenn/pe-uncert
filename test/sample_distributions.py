"""
Script to sample distributions from a trained crispAIPE model.
This visualizes the Dirichlet distribution predictions vs ground truth data.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import dirichlet

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Sample distributions from trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
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


def sample_from_test_data(data_module, num_samples):
    """Sample a few examples from the test dataset."""
    test_loader = data_module.test_dataloader()
    
    # Get a single batch
    batch = next(iter(test_loader))
    
    # Make sure we don't try to sample more than we have
    num_samples = min(num_samples, len(batch[0]))
    
    # Select random indices
    indices = np.random.choice(len(batch[0]), num_samples, replace=False)
    
    # Extract samples
    samples = [tuple(b[indices] for b in batch)]
    
    return samples


def visualize_distributions(model, samples, output_dir):
    """Visualize the predicted Dirichlet distributions vs ground truth."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = next(model.parameters()).device
    
    for i, batch in enumerate(samples):
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
        
        # Move tensors to CPU for plotting
        alpha_params_np = alpha_params.cpu().numpy()
        ground_truth_np = ground_truth.cpu().numpy()
        
        # For each sample in the batch
        for j in range(len(alpha_params_np)):
            alpha = alpha_params_np[j]
            gt = ground_truth_np[j]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot ternary grid
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
            from scipy.stats import gaussian_kde
            xy = dir_samples[:, :2]  # Use first two dimensions for 2D plot
            z = gaussian_kde(xy.T)(xy.T)
            
            # Sort by density for better visualization
            idx = z.argsort()
            x, y, z = xy[idx, 0], xy[idx, 1], z[idx]
            
            # Plot scatter with color representing density
            scatter = ax.scatter(x, y, c=z, s=20, alpha=0.6, cmap='viridis')
            
            # Plot ground truth
            ax.scatter(gt[0], gt[1], color='red', s=100, marker='*', label='Ground Truth')
            
            # Calculate expected value of Dirichlet
            expected = alpha / alpha.sum()
            ax.scatter(expected[0], expected[1], color='blue', s=100, marker='o', label='Predicted Mean')
            
            # Calculate alpha0 (precision/concentration)
            alpha0 = alpha.sum()
            
            # Add labels and title
            ax.set_title(f'Dirichlet Distribution (α={alpha.round(2)}, α₀={alpha0:.2f})')
            ax.set_xlabel('Edited %')
            ax.set_ylabel('Unedited %')
            ax.legend()
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Density')
            
            # Add annotations for ground truth and expected values
            ax.annotate(f"GT: [{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}]", 
                       xy=(gt[0], gt[1]), xytext=(gt[0]-0.2, gt[1]-0.1),
                       arrowprops=dict(arrowstyle="->"))
            
            ax.annotate(f"Pred: [{expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f}]", 
                       xy=(expected[0], expected[1]), xytext=(expected[0]-0.2, expected[1]+0.1),
                       arrowprops=dict(arrowstyle="->"))
            
            # Set axis limits
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Add note that the third dimension is implicitly 1-x-y
            plt.figtext(0.5, 0.01, "Note: Indel % = 1 - Edited % - Unedited %", 
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i}_{j}.png'))
            plt.close()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Loading model from {args.checkpoint}")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print(f"Sampling {args.num_samples} examples from test data")
    samples = sample_from_test_data(data_module, args.num_samples)
    
    print(f"Visualizing distributions")
    visualize_distributions(model, samples, args.output_dir)
    
    print(f"Done! Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
