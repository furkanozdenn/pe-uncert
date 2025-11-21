"""
Automated script to run complete ablation study:
1. Train transformer-only model
2. Train CNN-only model  
3. Evaluate all three architectures (transformer-only, CNN-only, hybrid)
4. Generate Supplementary Table S3

Usage:
    python test/run_ablation_study.py --config pe_uncert_models/configs/crispAIPE_conf1.json \
        --hybrid_checkpoint <path_to_hybrid_checkpoint> \
        [--skip_training] [--output_dir test/figures/ablation_study]
"""

import os
import sys
import argparse
import subprocess
import json
import glob
from pathlib import Path

def find_latest_checkpoint(log_dir, config_name):
    """Find the latest checkpoint in a log directory."""
    pattern = os.path.join(log_dir, config_name, "*", "best_model-*.ckpt")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        # Sort by modification time, return most recent
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    return None

def train_model(config_path, model_type, skip_if_exists=True):
    """Train an ablation model variant."""
    print(f"\n{'='*70}")
    print(f"Training {model_type} model")
    print(f"{'='*70}\n")
    
    # Check if checkpoint already exists
    with open(config_path, 'r') as f:
        config = json.load(f)
    log_dir = config['training_parameters']['log_dir']
    config_name = os.path.basename(config_path).replace('.json', '')
    
    existing_checkpoint = find_latest_checkpoint(log_dir, config_name)
    
    if existing_checkpoint and skip_if_exists:
        print(f"Found existing checkpoint: {existing_checkpoint}")
        print(f"Skipping training for {model_type} model.\n")
        return existing_checkpoint
    
    # Run training
    cmd = [
        sys.executable,
        'pe_uncert_models/models/train_ablation.py',
        '--config', config_path,
        '--model_type', model_type
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {model_type} model")
    
    # Find the newly created checkpoint
    checkpoint = find_latest_checkpoint(log_dir, config_name)
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found after training {model_type} model")
    
    print(f"Training complete. Checkpoint: {checkpoint}\n")
    return checkpoint

def run_ablation_evaluation(config_path, checkpoints, output_dir):
    """Run the ablation evaluation script."""
    print(f"\n{'='*70}")
    print("Running ablation evaluation")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        'test/model_ablation.py',
        '--config', config_path,
        '--output_dir', output_dir
    ]
    
    if checkpoints.get('hybrid'):
        cmd.extend(['--hybrid_checkpoint', checkpoints['hybrid']])
    if checkpoints.get('transformer'):
        cmd.extend(['--transformer_checkpoint', checkpoints['transformer']])
    if checkpoints.get('cnn'):
        cmd.extend(['--cnn_checkpoint', checkpoints['cnn']])
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    if result.returncode != 0:
        raise RuntimeError("Ablation evaluation failed")
    
    print("\nAblation evaluation complete!\n")

def main():
    parser = argparse.ArgumentParser(
        description='Automated ablation study: train models and generate Supplementary Table S3'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base config file (for hybrid model)')
    parser.add_argument('--hybrid_checkpoint', type=str, default=None,
                       help='Path to hybrid (crispAIPE) model checkpoint. If not provided, will search for latest.')
    parser.add_argument('--transformer_config', type=str, 
                       default='pe_uncert_models/configs/crispAIPE_transformer_only_conf.json',
                       help='Path to transformer-only config file')
    parser.add_argument('--cnn_config', type=str,
                       default='pe_uncert_models/configs/crispAIPE_cnn_only_conf.json',
                       help='Path to CNN-only config file')
    parser.add_argument('--output_dir', type=str, default='test/figures/ablation_study',
                       help='Output directory for results')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and use existing checkpoints if available')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Force retraining even if checkpoints exist')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    project_root = os.path.dirname(os.path.dirname(__file__))
    args.config = os.path.abspath(os.path.join(project_root, args.config))
    args.transformer_config = os.path.abspath(os.path.join(project_root, args.transformer_config))
    args.cnn_config = os.path.abspath(os.path.join(project_root, args.cnn_config))
    args.output_dir = os.path.abspath(os.path.join(project_root, args.output_dir))
    
    if args.hybrid_checkpoint:
        args.hybrid_checkpoint = os.path.abspath(os.path.join(project_root, args.hybrid_checkpoint))
    
    print("="*70)
    print("ABLATION STUDY AUTOMATION")
    print("="*70)
    print(f"Base config: {args.config}")
    print(f"Transformer config: {args.transformer_config}")
    print(f"CNN config: {args.cnn_config}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    checkpoints = {}
    
    # Find or use provided hybrid checkpoint
    if args.hybrid_checkpoint and os.path.exists(args.hybrid_checkpoint):
        checkpoints['hybrid'] = args.hybrid_checkpoint
        print(f"Using provided hybrid checkpoint: {checkpoints['hybrid']}\n")
    else:
        # Try to find latest hybrid checkpoint
        with open(args.config, 'r') as f:
            config = json.load(f)
        log_dir = config['training_parameters']['log_dir']
        config_name = os.path.basename(args.config).replace('.json', '')
        hybrid_checkpoint = find_latest_checkpoint(log_dir, config_name)
        if hybrid_checkpoint:
            checkpoints['hybrid'] = hybrid_checkpoint
            print(f"Found hybrid checkpoint: {checkpoints['hybrid']}\n")
        else:
            print("WARNING: No hybrid checkpoint found. Ablation study will be incomplete.\n")
    
    # Train or find transformer-only model
    if not args.skip_training or args.force_retrain:
        try:
            checkpoints['transformer'] = train_model(
                args.transformer_config, 
                'transformer_only',
                skip_if_exists=not args.force_retrain
            )
        except Exception as e:
            print(f"ERROR training transformer-only model: {e}")
            print("Continuing with available models...\n")
    else:
        # Just find existing checkpoint
        with open(args.transformer_config, 'r') as f:
            config = json.load(f)
        log_dir = config['training_parameters']['log_dir']
        config_name = os.path.basename(args.transformer_config).replace('.json', '')
        transformer_checkpoint = find_latest_checkpoint(log_dir, config_name)
        if transformer_checkpoint:
            checkpoints['transformer'] = transformer_checkpoint
            print(f"Using existing transformer checkpoint: {checkpoints['transformer']}\n")
        else:
            print("WARNING: No transformer-only checkpoint found.\n")
    
    # Train or find CNN-only model
    if not args.skip_training or args.force_retrain:
        try:
            checkpoints['cnn'] = train_model(
                args.cnn_config,
                'cnn_only',
                skip_if_exists=not args.force_retrain
            )
        except Exception as e:
            print(f"ERROR training CNN-only model: {e}")
            print("Continuing with available models...\n")
    else:
        # Just find existing checkpoint
        with open(args.cnn_config, 'r') as f:
            config = json.load(f)
        log_dir = config['training_parameters']['log_dir']
        config_name = os.path.basename(args.cnn_config).replace('.json', '')
        cnn_checkpoint = find_latest_checkpoint(log_dir, config_name)
        if cnn_checkpoint:
            checkpoints['cnn'] = cnn_checkpoint
            print(f"Using existing CNN checkpoint: {checkpoints['cnn']}\n")
        else:
            print("WARNING: No CNN-only checkpoint found.\n")
    
    # Run ablation evaluation
    if not checkpoints:
        print("ERROR: No checkpoints available. Cannot run ablation evaluation.")
        return 1
    
    print(f"\n{'='*70}")
    print("SUMMARY OF CHECKPOINTS")
    print(f"{'='*70}")
    for model_type, checkpoint in checkpoints.items():
        print(f"{model_type.capitalize()}: {checkpoint}")
    print(f"{'='*70}\n")
    
    try:
        run_ablation_evaluation(args.config, checkpoints, args.output_dir)
        
        print("\n" + "="*70)
        print("ABLATION STUDY COMPLETE!")
        print("="*70)
        print(f"Results saved to: {args.output_dir}")
        print(f"  - ablation_table_s3.csv")
        print(f"  - ablation_table_s3.png/pdf")
        print(f"  - ablation_comparison.png/pdf")
        print(f"  - ablation_results.json")
        print("="*70)
        
        return 0
    except Exception as e:
        print(f"\nERROR during ablation evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())


