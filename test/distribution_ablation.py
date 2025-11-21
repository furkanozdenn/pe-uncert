"""
Distribution ablation study evaluation script.
Compares Dirichlet (current), Softmax/Multinomial, and Logit-Normal distributions.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.data_utils.data import PE_Dataset
from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.models.distribution_ablation import crispAIPE_Softmax, crispAIPE_LogitNormal


def load_model_and_data(config_path, checkpoint_path, model_class, variant_config_path=None):
    """Load model from checkpoint and data module."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Use variant config if provided, otherwise use base config
    if variant_config_path and os.path.exists(variant_config_path):
        with open(variant_config_path, 'r') as f:
            config = json.load(f)
    
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']
    
    # Fix data paths
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Handle ../../ paths (relative to project root)
                if config_data[path_key].startswith('../../'):
                    # Remove ../../ prefix and join with project root
                    rel_path = config_data[path_key][6:]  # Remove '../../'
                    config_data[path_key] = os.path.normpath(
                        os.path.join(project_root, rel_path)
                    )
                # Handle ../ paths (relative to config file location)
                elif config_data[path_key].startswith('../'):
                    config_data[path_key] = os.path.normpath(
                        os.path.join(config_dir, config_data[path_key])
                    )
                else:
                    config_data[path_key] = os.path.normpath(
                        os.path.join(config_dir, config_data[path_key])
                    )
    
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model with specific config parameters
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    return model, data_module


def evaluate_model(model, data_module, batch_size=64):
    """Evaluate model and return metrics."""
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Evaluate on test set
    all_predictions = []
    all_targets = []
    test_losses = []
    
    print(f"Evaluating {model.model_name} on test set...")
    for batch in tqdm(test_loader):
        batch = [b.to(device) for b in batch]
        
        with torch.no_grad():
            x_hat, y_hat = model(batch)
            loss, _ = model.loss_function((x_hat, y_hat), batch[2:6], valid_step=True)
            test_losses.append(loss.item())
            
            # Get predictions based on distribution type
            if isinstance(model, crispAIPE):
                # Dirichlet: use expected proportions
                alpha_sum = torch.sum(y_hat, dim=1, keepdim=True)
                expected_props = y_hat / alpha_sum
            elif isinstance(model, crispAIPE_Softmax):
                # Softmax: apply softmax to logits
                expected_props = torch.softmax(y_hat, dim=1)
            elif isinstance(model, crispAIPE_LogitNormal):
                # Logit-Normal: use mean in logit space, apply softmax
                mean = y_hat[:, :3]
                expected_props = torch.softmax(mean, dim=1)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            # Get targets
            _, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
            ground_truth = torch.stack([
                edited_percentage, unedited_percentage, indel_percentage
            ], dim=1)
            ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
            
            all_predictions.append(expected_props.cpu())
            all_targets.append(ground_truth.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate Spearman correlations
    pred_edited = all_predictions[:, 0]
    pred_unedited = all_predictions[:, 1]
    pred_indel = all_predictions[:, 2]
    
    true_edited = all_targets[:, 0]
    true_unedited = all_targets[:, 1]
    true_indel = all_targets[:, 2]
    
    edited_corr, edited_p = stats.spearmanr(true_edited, pred_edited)
    unedited_corr, unedited_p = stats.spearmanr(true_unedited, pred_unedited)
    indel_corr, indel_p = stats.spearmanr(true_indel, pred_indel)
    
    # Overall correlation - ensure same shape
    all_pred = all_predictions.flatten()
    all_true = all_targets.flatten()
    
    # Ensure same length (should be batch_size * 3)
    if len(all_pred) != len(all_true):
        min_len = min(len(all_pred), len(all_true))
        all_pred = all_pred[:min_len]
        all_true = all_true[:min_len]
    
    overall_corr, overall_p = stats.spearmanr(all_true, all_pred)
    
    # Calculate validation loss
    val_losses = []
    for batch in val_loader:
        batch = [b.to(device) for b in batch]
        with torch.no_grad():
            x_hat, y_hat = model(batch)
            loss, _ = model.loss_function((x_hat, y_hat), batch[2:6], valid_step=True)
            val_losses.append(loss.item())
    
    avg_test_loss = np.mean(test_losses)
    avg_val_loss = np.mean(val_losses)
    
    results = {
        'test_loss': avg_test_loss,
        'val_loss': avg_val_loss,
        'edited_spearman': edited_corr,
        'unedited_spearman': unedited_corr,
        'indel_spearman': indel_corr,
        'overall_spearman': overall_corr,
        'n_samples': len(all_targets)
    }
    
    return results


def create_distribution_table(results_dict, output_dir):
    """Create Supplementary Table comparing distributions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for table
    distributions = ['Dirichlet', 'Softmax', 'Logit-Normal']
    data = []
    
    for dist in distributions:
        if dist in results_dict:
            r = results_dict[dist]
            data.append({
                'Distribution': dist,
                'Val Loss': f"{r['val_loss']:.4f}",
                'Test Loss': f"{r['test_loss']:.4f}",
                'Edited ρ': f"{r['edited_spearman']:.4f}",
                'Unedited ρ': f"{r['unedited_spearman']:.4f}",
                'Indel ρ': f"{r['indel_spearman']:.4f}",
                'Overall ρ': f"{r['overall_spearman']:.4f}"
            })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'distribution_ablation_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Distribution ablation table saved to {csv_path}")
    
    # Create formatted table plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for col_idx, col in enumerate(df.columns[1:], 1):  # Skip 'Distribution' column
        if 'Loss' in col:
            # Lower is better
            best_idx = df[col].str.replace(',', '').astype(float).idxmin() + 1
        else:
            # Higher is better
            best_idx = df[col].str.replace(',', '').astype(float).idxmax() + 1
        
        table[(best_idx, col_idx)].set_facecolor('#D5E8D4')
    
    plt.title('Distribution Ablation Study', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'distribution_ablation_table.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'distribution_ablation_table.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Distribution ablation table figure saved to {fig_path}")
    
    return df


def save_distribution_results(results_dict, output_dir):
    """Save results to JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert numpy types to native Python types
    json_results = {}
    for dist, results in results_dict.items():
        json_results[dist] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in results.items()
        }
    
    json_path = os.path.join(output_dir, 'distribution_ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Distribution ablation study comparing Dirichlet, Softmax, and Logit-Normal'
    )
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--dirichlet_checkpoint', type=str, default=None,
                       help='Path to Dirichlet (crispAIPE) model checkpoint')
    parser.add_argument('--softmax_checkpoint', type=str, default=None,
                       help='Path to Softmax model checkpoint')
    parser.add_argument('--logit_normal_checkpoint', type=str, default=None,
                       help='Path to Logit-Normal model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test/figures/distribution_ablation',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']
    
    # Fix data paths
    config_dir = os.path.dirname(os.path.abspath(args.config))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for path_key in ['train_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                # Handle ../../ paths (relative to project root)
                if config_data[path_key].startswith('../../'):
                    # Remove ../../ prefix and join with project root
                    rel_path = config_data[path_key][6:]  # Remove '../../'
                    config_data[path_key] = os.path.normpath(
                        os.path.join(project_root, rel_path)
                    )
                # Handle ../ paths (relative to config file location)
                elif config_data[path_key].startswith('../'):
                    config_data[path_key] = os.path.normpath(
                        os.path.join(config_dir, config_data[path_key])
                    )
                else:
                    config_data[path_key] = os.path.normpath(
                        os.path.join(config_dir, config_data[path_key])
                    )
    
    # Load dataset (shared across all models)
    print("Loading dataset...")
    data_module = PE_Dataset(data_config=config_data)
    
    results_dict = {}
    
    # Evaluate Dirichlet (crispAIPE) model
    if args.dirichlet_checkpoint and os.path.exists(args.dirichlet_checkpoint):
        print("\n" + "="*60)
        print("Evaluating Dirichlet (crispAIPE) model...")
        print("="*60)
        model, _ = load_model_and_data(args.config, args.dirichlet_checkpoint, crispAIPE)
        results = evaluate_model(model, data_module, args.batch_size)
        results_dict['Dirichlet'] = results
        print(f"Dirichlet Results: Overall ρ = {results['overall_spearman']:.4f}, "
              f"Val Loss = {results['val_loss']:.4f}")
    
    # Evaluate Softmax model
    if args.softmax_checkpoint and os.path.exists(args.softmax_checkpoint):
        print("\n" + "="*60)
        print("Evaluating Softmax model...")
        print("="*60)
        softmax_config = os.path.join(os.path.dirname(args.config), 'crispAIPE_softmax_conf.json')
        if not os.path.exists(softmax_config):
            softmax_config = None
        model, _ = load_model_and_data(args.config, args.softmax_checkpoint, crispAIPE_Softmax, softmax_config)
        results = evaluate_model(model, data_module, args.batch_size)
        results_dict['Softmax'] = results
        print(f"Softmax Results: Overall ρ = {results['overall_spearman']:.4f}, "
              f"Val Loss = {results['val_loss']:.4f}")
    
    # Evaluate Logit-Normal model
    if args.logit_normal_checkpoint and os.path.exists(args.logit_normal_checkpoint):
        print("\n" + "="*60)
        print("Evaluating Logit-Normal model...")
        print("="*60)
        logit_config = os.path.join(os.path.dirname(args.config), 'crispAIPE_logit_normal_conf.json')
        if not os.path.exists(logit_config):
            logit_config = None
        model, _ = load_model_and_data(args.config, args.logit_normal_checkpoint, crispAIPE_LogitNormal, logit_config)
        results = evaluate_model(model, data_module, args.batch_size)
        results_dict['Logit-Normal'] = results
        print(f"Logit-Normal Results: Overall ρ = {results['overall_spearman']:.4f}, "
              f"Val Loss = {results['val_loss']:.4f}")
    
    if not results_dict:
        print("\nNo models evaluated! Please provide at least one checkpoint path.")
        print("Required arguments:")
        print("  --dirichlet_checkpoint")
        print("  --softmax_checkpoint")
        print("  --logit_normal_checkpoint")
        return
    
    # Create table and save results
    print("\n" + "="*60)
    print("Creating distribution ablation study table and plots...")
    print("="*60)
    create_distribution_table(results_dict, args.output_dir)
    save_distribution_results(results_dict, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("DISTRIBUTION ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'Distribution':<20} {'Val Loss':<12} {'Test Loss':<12} {'Edited ρ':<10} {'Unedited ρ':<12} {'Indel ρ':<10} {'Overall ρ':<10}")
    print("-" * 60)
    for dist, results in results_dict.items():
        print(f"{dist:<20} {results['val_loss']:>11.4f} {results['test_loss']:>11.4f} "
              f"{results['edited_spearman']:>9.4f} {results['unedited_spearman']:>11.4f} "
              f"{results['indel_spearman']:>9.4f} {results['overall_spearman']:>9.4f}")
    print("="*60)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

