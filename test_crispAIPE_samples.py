#!/usr/bin/env python3
"""
Test crispAIPE model with random batch and single example samples from test set.

This script:
1. Loads a trained crispAIPE model from checkpoint
2. Samples a random batch from the test set
3. Samples a single random example from the test set
4. Runs inference on both and displays results
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import random

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test crispAIPE with random samples')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of random batch to sample (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='test_samples_output',
                        help='Output directory for results (default: test_samples_output)')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_data(config_path, checkpoint_path):
    """Load model from checkpoint and prepare test dataset."""
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
            print(f"Using {path_key}: {config_data[path_key]}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = crispAIPE.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    # Move model to appropriate device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        print(f"Using GPU: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        model = model.to(device)
        print(f"Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        model = model.to(device)
        print(f"Using CPU")
    
    return model, data_module, device, config_data


def get_random_batch(data_module, batch_size, device):
    """Get a random batch from the test dataset."""
    test_loader = data_module.test_dataloader(shuffle_bool=True)
    
    # Get a batch
    batch = next(iter(test_loader))
    
    # Limit batch size if needed
    actual_batch_size = min(batch_size, len(batch[0]))
    
    if actual_batch_size < len(batch[0]):
        # Randomly select indices
        indices = np.random.choice(len(batch[0]), actual_batch_size, replace=False)
        batch = [b[indices] for b in batch]
    
    # Move to device
    batch = [b.to(device) for b in batch]
    
    return batch


def get_single_example(data_module, device):
    """Get a single random example from the test dataset."""
    test_loader = data_module.test_dataloader(shuffle_bool=True)
    
    # Get a batch
    batch = next(iter(test_loader))
    
    # Select a random index
    random_idx = np.random.randint(0, len(batch[0]))
    
    # Extract single example (add batch dimension)
    single_example = [b[random_idx:random_idx+1] for b in batch]
    
    # Move to device
    single_example = [b.to(device) for b in single_example]
    
    return single_example, random_idx


def run_inference(model, batch, device):
    """Run inference on a batch and return predictions."""
    with torch.no_grad():
        x_hat, y_hat = model(batch)
        
        # Calculate loss
        loss, _ = model.loss_function(
            (x_hat, y_hat), 
            batch[2:6], 
            valid_step=True
        )
        
        # Get expected proportions from Dirichlet parameters
        alpha_sum = torch.sum(y_hat, dim=1, keepdim=True)
        expected_props = y_hat / alpha_sum
        
        # Get ground truth
        _, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
        ground_truth = torch.stack([
            edited_percentage, 
            unedited_percentage, 
            indel_percentage
        ], dim=1)
        ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
        
        return {
            'loss': loss.item(),
            'predictions': expected_props.cpu().numpy(),
            'ground_truth': ground_truth.cpu().numpy(),
            'alpha_params': y_hat.cpu().numpy(),
            'total_read_count': batch[2].cpu().numpy()
        }


def print_results(results, title, is_single=False):
    """Print inference results in a readable format."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Loss: {results['loss']:.6f}")
    print(f"\nTotal Read Count: {results['total_read_count'].flatten()}")
    
    if is_single:
        print(f"\nGround Truth Proportions:")
        print(f"  Edited:    {results['ground_truth'][0, 0]:.4f}")
        print(f"  Unedited:  {results['ground_truth'][0, 1]:.4f}")
        print(f"  Indel:     {results['ground_truth'][0, 2]:.4f}")
        
        print(f"\nPredicted Proportions (Expected from Dirichlet):")
        print(f"  Edited:    {results['predictions'][0, 0]:.4f}")
        print(f"  Unedited:  {results['predictions'][0, 1]:.4f}")
        print(f"  Indel:     {results['predictions'][0, 2]:.4f}")
        
        print(f"\nDirichlet Alpha Parameters:")
        print(f"  Alpha (Edited):   {results['alpha_params'][0, 0]:.4f}")
        print(f"  Alpha (Unedited): {results['alpha_params'][0, 1]:.4f}")
        print(f"  Alpha (Indel):    {results['alpha_params'][0, 2]:.4f}")
        
        # Calculate errors
        errors = np.abs(results['predictions'][0] - results['ground_truth'][0])
        print(f"\nAbsolute Errors:")
        print(f"  Edited:    {errors[0]:.4f}")
        print(f"  Unedited:  {errors[1]:.4f}")
        print(f"  Indel:     {errors[2]:.4f}")
        print(f"  Mean Error: {np.mean(errors):.4f}")
    else:
        batch_size = len(results['predictions'])
        print(f"\nBatch Size: {batch_size}")
        print(f"\nGround Truth vs Predictions (first 5 examples):")
        print(f"{'Idx':<5} {'GT Edited':<12} {'GT Unedited':<14} {'GT Indel':<12} "
              f"{'Pred Edited':<14} {'Pred Unedited':<16} {'Pred Indel':<14} {'Error':<10}")
        print("-" * 100)
        
        for i in range(min(5, batch_size)):
            errors = np.abs(results['predictions'][i] - results['ground_truth'][i])
            mean_error = np.mean(errors)
            print(f"{i:<5} "
                  f"{results['ground_truth'][i, 0]:<12.4f} "
                  f"{results['ground_truth'][i, 1]:<14.4f} "
                  f"{results['ground_truth'][i, 2]:<12.4f} "
                  f"{results['predictions'][i, 0]:<14.4f} "
                  f"{results['predictions'][i, 1]:<16.4f} "
                  f"{results['predictions'][i, 2]:<14.4f} "
                  f"{mean_error:<10.4f}")
        
        # Calculate mean errors across batch
        all_errors = np.abs(results['predictions'] - results['ground_truth'])
        mean_errors = np.mean(all_errors, axis=0)
        overall_mean = np.mean(all_errors)
        print(f"\nMean Errors Across Batch:")
        print(f"  Edited:    {mean_errors[0]:.4f}")
        print(f"  Unedited:  {mean_errors[1]:.4f}")
        print(f"  Indel:     {mean_errors[2]:.4f}")
        print(f"  Overall:   {overall_mean:.4f}")


def save_results(batch_results, single_results, output_dir):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save batch results
    batch_file = os.path.join(output_dir, 'batch_results.json')
    batch_data = {
        'loss': float(batch_results['loss']),
        'predictions': batch_results['predictions'].tolist(),
        'ground_truth': batch_results['ground_truth'].tolist(),
        'alpha_params': batch_results['alpha_params'].tolist(),
        'total_read_count': batch_results['total_read_count'].tolist()
    }
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    print(f"\nBatch results saved to: {batch_file}")
    
    # Save single example results
    single_file = os.path.join(output_dir, 'single_example_results.json')
    single_data = {
        'loss': float(single_results['loss']),
        'predictions': single_results['predictions'].tolist(),
        'ground_truth': single_results['ground_truth'].tolist(),
        'alpha_params': single_results['alpha_params'].tolist(),
        'total_read_count': single_results['total_read_count'].tolist()
    }
    with open(single_file, 'w') as f:
        json.dump(single_data, f, indent=2)
    print(f"Single example results saved to: {single_file}")


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Load model and data
    model, data_module, device, config_data = load_model_and_data(
        args.config, 
        args.checkpoint
    )
    
    # Get random batch
    print(f"\n{'='*80}")
    print("Sampling Random Batch from Test Set")
    print(f"{'='*80}")
    batch = get_random_batch(data_module, args.batch_size, device)
    print(f"Batch shape: {[b.shape for b in batch]}")
    
    # Run inference on batch
    batch_results = run_inference(model, batch, device)
    print_results(batch_results, "BATCH INFERENCE RESULTS", is_single=False)
    
    # Get single example
    print(f"\n{'='*80}")
    print("Sampling Single Random Example from Test Set")
    print(f"{'='*80}")
    single_example, example_idx = get_single_example(data_module, device)
    print(f"Example index: {example_idx}")
    print(f"Single example shape: {[b.shape for b in single_example]}")
    
    # Run inference on single example
    single_results = run_inference(model, single_example, device)
    print_results(single_results, "SINGLE EXAMPLE INFERENCE RESULTS", is_single=True)
    
    # Save results
    save_results(batch_results, single_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print("Inference Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

