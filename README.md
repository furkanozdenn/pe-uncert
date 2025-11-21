# crispAIPE: Probabilistic Modelling of Prime Editing Variant Correction Efficiency

A deep learning model for predicting editing outcomes and uncertainty in CRISPR prime editing experiments. The model uses a hybrid architecture combining Transformer encoders and CNNs to predict the distribution of editing outcomes (edited, unedited, and indel percentages).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Testing and Inference](#running-crispAIPE)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- PyTorch Lightning
- CUDA (optional, for GPU acceleration)
- Apple Silicon MPS support (for M1/M2 Macs)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pe-uncert
```

2. Install the package:
```bash
pip install -e .
```

3. Install additional dependencies:
```bash
pip install pytorch-lightning wandb scipy scikit-learn pandas numpy
```

4. (Optional) Activate your virtual environment:
```bash
source /path/to/your/venv/bin/activate
you may use pe-uncert.yaml env provided.
```

## Quick Start

### Testing with Random Samples

Test the model with random batches and single examples:

```bash
python3 test_crispAIPE_samples.py \
  --config pe_uncert_models/configs/crispAIPE_train_test_split_conf.json \
  --checkpoint pe_uncert_models/logs/crispAIPE_train_test_split_conf/2025-06-08-15-59-36/best_model-epoch=41-val_loss_val_loss=-3.0687.ckpt \
  --batch_size 16
```

## Data Requirements

The model expects CSV files with the following columns:

- `initial_sequence`: The original DNA sequence (one-hot encoded)
- `mutated_sequence`: The target mutated sequence (one-hot encoded)
- `total_read_count`: Total number of sequencing reads
- `edited_percentage`: Percentage of edited reads
- `unedited_percentage`: Percentage of unedited reads
- `indel_percentage`: Percentage of indel reads
- `protospacer_location`: Binary mask indicating protospacer location
- `pbs_location`: Binary mask indicating PBS location
- `rt_initial_location`: Binary mask indicating RT template initial location
- `rt_mutated_location`: Binary mask indicating RT template mutated location

### Data Format

The model uses the PRIDICT dataset format. Example data files:
- Training: `data/pridict_data/pridict-train.csv`
- Test: `data/pridict_data/pridict-test.csv`

#### View All Options

```bash
./train_crispAIPE.sh --help
```

## Runnin crispAIPE

### Test with Random Samples

The `test_crispAIPE_samples.py` script allows you to test the model with random batches and single examples:

```bash
python3 test_crispAIPE_samples.py \
  --config <config_path> \
  --checkpoint <checkpoint_path> \
  --batch_size 16 \
  --seed 42 \
  --output_dir test_samples_output
```

#### Arguments

- `--config`: Path to the config JSON file used for training
- `--checkpoint`: Path to the model checkpoint file (.ckpt)
- `--batch_size`: Size of random batch to sample (default: 16)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Output directory for results (default: test_samples_output)

#### Output

The script generates:
1. **Console output**: Detailed inference results showing:
   - Loss values
   - Ground truth vs predicted proportions
   - Error metrics
   - Dirichlet alpha parameters

2. **JSON files**:
   - `batch_results.json`: Full batch inference results
   - `single_example_results.json`: Single example inference results

### Example Output

```
BATCH INFERENCE RESULTS
================================================================================
Loss: -1.462086
Batch Size: 16

Ground Truth vs Predictions (first 5 examples):
Idx   GT Edited    GT Unedited    GT Indel     Pred Edited    Pred Unedited    Pred Indel     Error     
----------------------------------------------------------------------------------------------------
0     0.1882       0.6921         0.1197       0.5607         0.1235           0.3158         0.3791    
...

Mean Errors Across Batch:
  Edited:    0.1476
  Unedited:  0.1549
  Indel:     0.0499
  Overall:   0.1175
```

## Configuration Options

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_dna_flank_len` | Target DNA flank length | 0 |
| `kmer_size` | K-mer size for encoding | 3 |
| `n_embd` | Embedding dimension | 64 |
| `d_model` | Model dimension | 16 |
| `n_layer` | Number of transformer layers | 4 |
| `dropout` | Dropout rate | 0.1 |
| `embedding_dim` | Embedding dimension | 64 |
| `nhead` | Number of attention heads | 4 |
| `bottleneck_dim` | Bottleneck dimension | 8 |
| `assesor_type` | Assessor type (multinomial/softmax/logit_normal) | multinomial |

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `train_data_path` | Path to training CSV | `data/pridict_data/pridict-train.csv` |
| `test_data_path` | Path to test CSV | `data/pridict_data/pridict-test.csv` |
| `batch_size` | Batch size | 128 |
| `val_split` | Validation split ratio | 0.1 |
| `sequence_length` | Sequence length | 99 |
| `pegrna_length` | PEGRNA length | 99 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | 6e-4 |
| `weight_decay` | Weight decay | 0.01 |
| `warmup_epochs` | Warmup epochs | 10 |
| `max_epochs` | Maximum epochs | 100 |
| `early_stopping` | Enable early stopping | true |
| `patience` | Early stopping patience | 8 |
| `cpu` | Use CPU only | false |
| `gpus` | GPU IDs (comma-separated) | "" (auto-detect) |


## Configuration Files

Pre-configured settings are available in `pe_uncert_models/configs/`:

- `crispAIPE_train_test_split_conf.json`: Train-test split configuration (recommended)
- `crispAIPE_conf1.json`: Original configuration
- `crispAIPE_softmax_conf.json`: Softmax variant
- `crispAIPE_logit_normal_conf.json`: Logit-normal variant
- `crispAIPE_cnn_only_conf.json`: CNN-only ablation
- `crispAIPE_transformer_only_conf.json`: Transformer-only ablation

### Using Different Configurations

Modify the config JSON file or use command-line arguments:

```bash
./train_crispAIPE.sh \
  --config pe_uncert_models/configs/crispAIPE_softmax_conf.json \
  --assesor_type softmax
```

The script automatically detects and uses:
- CUDA GPUs (if available)
- Apple Silicon MPS (M1/M2 Macs)
- CPU (fallback)

Force CPU mode:
```bash
./train_crispAIPE.sh --cpu true
```

## Citation

If you use this code in your research, please cite:

```bibtex
coming soon
```

## License

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

This work is funded by Google DeepMind.  
The author has applied a **CC BY** public copyright licence to any Author Accepted Manuscript (AAM) version arising from this submission.

## Contact

furkan.ozden@cs.ox.ac.uk - peter.minary@cs.ox.ac.uk

