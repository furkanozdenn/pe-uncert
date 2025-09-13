"""
Enhanced diagnostics plot script for crispAIPE model with baseline comparisons.
Includes Random Forest and Quantile Regression baselines adapted for pridict data.

example cmd:
python test/diagnostics_plot_with_baselines.py --config pe_uncert_models/configs/crispAIPE_train_test_split_conf.json --checkpoint pe_uncert_models/logs/crispAIPE_train_test_split_conf/2025-06-08-15-59-36/best_model-epoch=41-val_loss_val_loss=-3.0687.ckpt --output_dir ./figures/train_test_split_model --n_samples 1000 --batch_size 32
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

import json
from tqdm import tqdm

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate diagnostics plot for crispAIPE with baselines')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to draw from distributions')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--baseline_samples', type=int, default=10000, help='Number of training samples for baselines')
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
    
    return model, data_module, config_data


def simplex_to_2d(simplex_points):
    """Extract first two dimensions from 3D simplex points (edited %, unedited %)."""
    return simplex_points[:, :2]


def compute_pca_confidence_region(samples_2d, confidence_level=0.95):
    """Compute PCA-based confidence region around the mean of 2D samples."""
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


def point_in_confidence_region(point, mean, principal_axes, eigenvalues, confidence_scale):
    """Check if a point is within the PCA-based confidence region."""
    # Transform point to PCA space
    centered_point = point - mean
    pca_coords = principal_axes.T @ centered_point
    
    # Check if point is within confidence ellipse
    ellipse_coords = pca_coords / np.sqrt(eigenvalues)
    distance_squared = np.sum(ellipse_coords**2)
    
    return distance_squared <= confidence_scale**2


def prepare_baseline_data(data_module, baseline_samples):
    """Prepare training and test data for baseline models."""
    # Load training data
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Extract training data
    X_train_list = []
    y_train_list = []
    
    count = 0
    for batch in train_loader:
        sequences, _, edited_pct, unedited_pct, indel_pct = batch[:5]
        
        # Stack the ground truth proportions
        ground_truth = torch.stack([edited_pct, unedited_pct, indel_pct], dim=1)
        ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
        
        # Use sequences as features (flatten the one-hot encoded sequences)
        X_batch = sequences.view(sequences.shape[0], -1)
        
        X_train_list.append(X_batch.numpy())
        y_train_list.append(ground_truth.numpy())
        
        count += len(sequences)
        if count >= baseline_samples:
            break
    
    X_train = np.vstack(X_train_list)[:baseline_samples]
    y_train = np.vstack(y_train_list)[:baseline_samples]
    
    # Extract test data
    X_test_list = []
    y_test_list = []
    
    for batch in test_loader:
        sequences, _, edited_pct, unedited_pct, indel_pct = batch[:5]
        
        ground_truth = torch.stack([edited_pct, unedited_pct, indel_pct], dim=1)
        ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
        
        X_batch = sequences.view(sequences.shape[0], -1)
        
        X_test_list.append(X_batch.numpy())
        y_test_list.append(ground_truth.numpy())
    
    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)
    
    return X_train, y_train, X_test, y_test


def train_baseline_models(X_train, y_train):
    """Train baseline models for uncertainty estimation."""
    print("Training baseline models...")
    
    # Random Forest with multiple trees for uncertainty
    print("Training Random Forest...")
    rf_models = []
    for outcome_idx in range(3):  # edited, unedited, indel
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train[:, outcome_idx])
        rf_models.append(rf)
    
    # Quantile Regression for confidence intervals
    print("Training Quantile Regression models...")
    quantiles = np.arange(0.05, 1.0, 0.05)
    quantile_models = {}
    
    for outcome_idx in range(3):
        quantile_models[outcome_idx] = []
        for q in tqdm(quantiles, desc=f"Training quantile models for outcome {outcome_idx}"):
            q_model = QuantileRegressor(quantile=q, alpha=0.01)
            # Use a subset for faster training
            subset_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            q_model.fit(X_train[indices], y_train[indices, outcome_idx])
            quantile_models[outcome_idx].append(q_model)
    
    return rf_models, quantile_models, quantiles


def generate_predictions_and_samples(model, data_module, n_samples=5000, batch_size=64):
    """Generate predictions and sample distributions for the test set."""
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_samples = []
    all_ground_truth = []
    
    print("Generating crispAIPE predictions and samples...")
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


def generate_baseline_predictions(rf_models, quantile_models, quantiles, X_test):
    """Generate predictions from baseline models."""
    print("Generating baseline predictions...")
    
    # Random Forest predictions (ensemble uncertainty)
    rf_predictions = []
    rf_samples = []
    
    for outcome_idx in range(3):
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test) for tree in rf_models[outcome_idx].estimators_])
        rf_predictions.append(np.mean(tree_predictions, axis=0))
        rf_samples.append(tree_predictions.T)  # transpose to get (n_samples, n_trees)
    
    rf_predictions = np.array(rf_predictions).T  # shape: (n_test, 3)
    rf_samples = np.array(rf_samples).transpose(1, 0, 2)  # shape: (n_test, 3, n_trees)
    
    # Quantile Regression predictions
    qr_predictions = []
    for outcome_idx in range(3):
        outcome_quantiles = np.array([q_model.predict(X_test) for q_model in quantile_models[outcome_idx]])
        qr_predictions.append(outcome_quantiles.T)  # transpose to get (n_test, n_quantiles)
    
    qr_predictions = np.array(qr_predictions).transpose(1, 0, 2)  # shape: (n_test, 3, n_quantiles)
    
    return rf_predictions, rf_samples, qr_predictions, quantiles


def compute_baseline_coverage(baseline_samples, ground_truth, confidence_levels, baseline_name):
    """Compute coverage for baseline methods."""
    n_samples = len(ground_truth)
    observed_coverage = []
    
    print(f"Computing {baseline_name} coverage...")
    for conf_level in tqdm(confidence_levels):
        coverage_count = 0
        
        for i in range(n_samples):
            # Convert to 2D space
            ground_truth_2d = simplex_to_2d(ground_truth[i:i+1])[0]
            
            if baseline_name == "Random Forest":
                # Use tree predictions as samples
                samples_3d = baseline_samples[i]  # shape: (3, n_trees)
                # Convert to (n_trees, 3) and normalize
                samples_3d = samples_3d.T
                samples_3d = np.maximum(samples_3d, 0)  # ensure non-negative
                samples_3d = samples_3d / np.sum(samples_3d, axis=1, keepdims=True)
                samples_2d = simplex_to_2d(samples_3d)
            else:  # Quantile Regression
                # Generate samples from quantile predictions
                # This is a simplified approach - in practice, you'd use the quantiles directly
                samples_3d = baseline_samples[i]  # shape: (3, n_quantiles)
                samples_3d = samples_3d.T
                samples_3d = np.maximum(samples_3d, 0)  # ensure non-negative
                samples_3d = samples_3d / np.sum(samples_3d, axis=1, keepdims=True)
                samples_2d = simplex_to_2d(samples_3d)
            
            # Compute PCA confidence region
            try:
                mean, principal_axes, eigenvalues, confidence_scale = compute_pca_confidence_region(
                    samples_2d, confidence_level=conf_level
                )
                
                # Check if ground truth is within confidence region
                if point_in_confidence_region(ground_truth_2d, mean, principal_axes, eigenvalues, confidence_scale):
                    coverage_count += 1
            except:
                # If PCA fails (e.g., not enough variance), assume no coverage
                pass
        
        observed_coverage.append(coverage_count / n_samples)
    
    return np.array(observed_coverage)


def compute_diagnostics(predictions, samples, ground_truth, confidence_levels):
    """Compute diagnostic statistics for PCA-based confidence regions."""
    n_samples = len(predictions)
    observed_coverage = []
    
    print("Computing crispAIPE diagnostic statistics...")
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


def plot_diagnostics_with_baselines(confidence_levels, crispAIPE_coverage, rf_coverage, qr_coverage, output_dir):
    """Create and save the diagnostic plot with baselines."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    
    # Model coverages
    ax.plot(confidence_levels, crispAIPE_coverage, 'b-', linewidth=3, marker='o', 
            markersize=8, label='crispAIPE (PCA-based)')
    ax.plot(confidence_levels, rf_coverage, 'r-', linewidth=3, marker='s', 
            markersize=8, label='Random Forest')
    ax.plot(confidence_levels, qr_coverage, 'g-', linewidth=3, marker='^', 
            markersize=8, label='Quantile Regression')
    
    # Formatting
    ax.set_xlabel('Predicted Confidence Level', fontsize=16)
    ax.set_ylabel('Observed Coverage Rate', fontsize=16)
    ax.set_title('Uncertainty Calibration Comparison', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add text box with statistics
    crispAIPE_mae = np.mean(np.abs(crispAIPE_coverage - confidence_levels))
    rf_mae = np.mean(np.abs(rf_coverage - confidence_levels))
    qr_mae = np.mean(np.abs(qr_coverage - confidence_levels))
    
    textstr = f'Mean Absolute Error:\ncrispAIPE: {crispAIPE_mae:.3f}\nRandom Forest: {rf_mae:.3f}\nQuantile Reg.: {qr_mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostics_with_baselines.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'diagnostics_with_baselines.pdf'))
    plt.close()
    
    print(f"Diagnostic plot with baselines saved to {output_dir}")
    
    return crispAIPE_mae, rf_mae, qr_mae


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module, config_data = load_model_and_data(args.config, args.checkpoint)
    
    # Prepare data for baselines
    print("Preparing baseline data...")
    X_train, y_train, X_test, y_test = prepare_baseline_data(data_module, args.baseline_samples)
    
    # Train baseline models
    rf_models, quantile_models, quantiles = train_baseline_models(X_train, y_train)
    
    # Generate crispAIPE predictions
    print("Generating crispAIPE predictions...")
    predictions, samples, ground_truth = generate_predictions_and_samples(
        model, data_module, n_samples=args.n_samples, batch_size=args.batch_size
    )
    
    # Generate baseline predictions
    rf_predictions, rf_samples, qr_predictions, qr_quantiles = generate_baseline_predictions(
        rf_models, quantile_models, quantiles, X_test
    )
    
    print(f"Dataset size: {len(predictions)} samples")
    
    # Define confidence levels to test
    confidence_levels = np.arange(0.1, 1.0, 0.1)
    
    # Compute diagnostics for all methods
    crispAIPE_coverage, _ = compute_diagnostics(predictions, samples, ground_truth, confidence_levels)
    rf_coverage = compute_baseline_coverage(rf_samples, ground_truth, confidence_levels, "Random Forest")
    qr_coverage = compute_baseline_coverage(qr_predictions, ground_truth, confidence_levels, "Quantile Regression")
    
    # Create diagnostic plots
    print("Creating diagnostic plots...")
    crispAIPE_mae, rf_mae, qr_mae = plot_diagnostics_with_baselines(
        confidence_levels, crispAIPE_coverage, rf_coverage, qr_coverage, args.output_dir
    )
    
    # Print summary statistics
    print(f"\nDiagnostic Results:")
    print(f"crispAIPE MAE: {crispAIPE_mae:.4f}")
    print(f"Random Forest MAE: {rf_mae:.4f}")
    print(f"Quantile Regression MAE: {qr_mae:.4f}")
    
    print(f"\nDetailed Results:")
    print("Conf Level | crispAIPE | Random Forest | Quantile Reg.")
    print("-" * 55)
    for conf, crisp, rf, qr in zip(confidence_levels, crispAIPE_coverage, rf_coverage, qr_coverage):
        print(f"    {conf:.1f}    |   {crisp:.3f}   |     {rf:.3f}     |     {qr:.3f}")
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 