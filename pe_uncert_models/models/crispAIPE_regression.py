"""
Regression version of crispAIPE model that predicts continuous values instead of Dirichlet parameters.
"""
import wandb
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import pdb
from pe_uncert_models.models.block_nets import ConvNet, LSTMNet
from pe_uncert_models.models.base import crispAIPEBase


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough pe
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]


class crispAIPERegression(crispAIPEBase):

    def __init__(self, hparams):
        super(crispAIPERegression, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        # model parameters
        # self.input_dim = hparams.input_dim # this is the size of the vocabulary 
        self.input_dim = getattr(hparams, 'input_dim', 5)
        
        # The unified representation will have:
        # - 5 channels from one-hot
        # - 2 channels for mismatch direction
        # - 4 channels for locations (protospacer, pbs, rt_initial, rt_mutated)
        # Total: 11 channels
        self.direction_channels = 2
        self.location_channels = 4
        self.unified_dim = self.input_dim + self.direction_channels + self.location_channels

        self.batch_size = getattr(hparams, 'batch_size', 128)
        self.lr = getattr(hparams, 'lr', 6e-4)
        self.model_name = "crispAIPERegression"

        self.sequence_length = getattr(hparams, 'sequence_length', 99) # assuming initial sequence and modified sequence are of the same length
        self.target_seq_flank_len = getattr(hparams, 'target_seq_flank_len', 0) # keeping no flank around DNA sequence
        
        # Transformer encoder parameters - significantly reduced for fewer parameters
        self.embedding_dim = getattr(hparams, 'embedding_dim', 8)
        self.nhead = getattr(hparams, 'nhead', 2)
        self.num_encoder_layers = getattr(hparams, 'num_encoder_layers', 2)  # Reduced from 6 to 2
        self.dim_feedforward = getattr(hparams, 'dim_feedforward', 32)  # Reduced from 256 to 32
        self.dropout = getattr(hparams, 'dropout', 0.1)
        
        # Min-max scaling parameters for edited percentage
        # These values are calculated from the deepprime dataset
        self.edited_percentage_min = 0.0
        self.edited_percentage_max = 61.56111929  # Maximum value found in the dataset
        
        # Embedding layer for transformer approach
        self.vocab_size = self.input_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len=self.sequence_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)
        
        # Projection for the final concatenated representation
        # The concatenated representation will have embedding_dim + unified_dim channels
        self.final_dim = self.embedding_dim + self.unified_dim
        
        # Keep the ConvNet for other processing
        self.conv_net = ConvNet(self.final_dim, 64)  # Process the combined representation
        
        # MLP to output a single continuous value (regression output)
        self.regression_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Remove ReLU from final layer to allow negative values
        )
        
        # Initialize the final layer with positive bias to help with non-zero predictions
        with torch.no_grad():
            self.regression_mlp[-1].bias.fill_(0.1)  # Small positive bias

        print(f"Number of parameters: {self._num_params()}")
        self._print_param_breakdown()
        print(f"Min-max scaling: min={self.edited_percentage_min}, max={self.edited_percentage_max}")

    def scale_targets(self, targets):
        """
        Scale target values from original range to [0,1] using min-max scaling.
        
        Args:
            targets: Tensor of target values in original range
            
        Returns:
            Tensor of scaled values in [0,1] range
        """
        return (targets - self.edited_percentage_min) / (self.edited_percentage_max - self.edited_percentage_min)
    
    def unscale_predictions(self, predictions):
        """
        Unscale predictions from [0,1] range back to original range.
        
        Args:
            predictions: Tensor of predictions in [0,1] range
            
        Returns:
            Tensor of unscaled predictions in original range
        """
        return predictions * (self.edited_percentage_max - self.edited_percentage_min) + self.edited_percentage_min

    def _num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _print_param_breakdown(self):
        """Print breakdown of parameters by component"""
        embedding_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters() if p.requires_grad)
        convnet_params = sum(p.numel() for p in self.conv_net.parameters() if p.requires_grad)
        mlp_params = sum(p.numel() for p in self.regression_mlp.parameters() if p.requires_grad)
        
        print(f"Parameter breakdown:")
        print(f"  Embedding: {embedding_params:,}")
        print(f"  Transformer encoder: {transformer_params:,}")
        print(f"  ConvNet: {convnet_params:,}")
        print(f"  Regression MLP: {mlp_params:,}")
    
    def _convert_onehot_to_indices(self, onehot_tensor):
        """
        Convert one-hot encoded tensor to token indices.
        
        Args:
            onehot_tensor: Tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len] with token indices
        """
        # Get the index of the maximum value along the last dimension
        # This converts a one-hot vector [0,1,0,0,0] to the index 1
        indices = torch.argmax(onehot_tensor, dim=-1)
        return indices
    
    def _create_unified_representation(self, initial_seq, mutated_seq, locations):
        """
        Create a unified representation by performing logical OR between initial and mutated sequences,
        adding two channels for direction of mismatch, and 4 channels for various locations.
        
        This is a utility method that can be used by other subnetworks.
        
        Args:
            initial_seq: Initial sequence tensor of shape [batch_size, seq_len, input_dim]
            mutated_seq: Mutated sequence tensor of shape [batch_size, seq_len, input_dim]
            locations: List of 4 location tensors, each of shape [batch_size, seq_len]
                - protospacer_location
                - pbs_location
                - rt_initial_location
                - rt_mutated_location
            
        Returns:
            Unified representation tensor of shape [batch_size, seq_len, input_dim+2+4]
        """
        batch_size, seq_len, _ = initial_seq.shape
        
        # Perform logical OR between initial and mutated sequences
        # Convert to boolean tensors for OR operation, then back to float
        initial_bool = initial_seq.bool()
        mutated_bool = mutated_seq.bool()
        or_result = (initial_bool | mutated_bool).float()
        
        # Create direction channels
        # Direction [1,0]: initial -> mutated (value increases)
        # Direction [0,1]: initial <- mutated (value decreases)
        # Direction [0,0]: no mismatch
        
        # First identify positions where there's a mismatch
        # Get the indices of the "hot" position in each one-hot vector
        initial_indices = torch.argmax(initial_seq, dim=-1)  # [batch_size, seq_len]
        mutated_indices = torch.argmax(mutated_seq, dim=-1)  # [batch_size, seq_len]
        
        # Create direction tensors
        direction = torch.zeros(batch_size, seq_len, 2, device=initial_seq.device)
        
        # Calculate where initial < mutated (direction [1,0])
        direction_up = (initial_indices < mutated_indices).unsqueeze(-1)
        direction[:, :, 0:1] = direction_up.float()
        
        # Calculate where initial > mutated (direction [0,1])
        direction_down = (initial_indices > mutated_indices).unsqueeze(-1)
        direction[:, :, 1:2] = direction_down.float()
        
        # Process location vectors - add a dimension for concatenation
        location_channels = []
        for loc in locations:
            # Ensure the location tensor has the right shape [batch_size, seq_len, 1]
            if len(loc.shape) == 2:
                loc = loc.unsqueeze(-1)
            location_channels.append(loc)
        
        # Concatenate all location channels
        location_tensor = torch.cat(location_channels, dim=-1)  # [batch_size, seq_len, 4]
        
        # Concatenate OR result with direction and location channels
        unified_rep = torch.cat([or_result, direction, location_tensor], dim=-1)  # [batch_size, seq_len, input_dim+2+4]
        
        return unified_rep

    def forward(self, batch):
        """
        Process the input sequence through the transformer encoder and combine with unified representation.
        
        Args:
            batch: List containing various inputs
                batch[0]: initial_sequence - tensor of shape [batch_size, seq_len, input_dim]
                batch[1]: mutated_sequence - tensor of shape [batch_size, seq_len, input_dim]
                batch[6]: protospacer_location - tensor of shape [batch_size, seq_len]
                batch[7]: pbs_location - tensor of shape [batch_size, seq_len]
                batch[8]: rt_initial_location - tensor of shape [batch_size, seq_len]
                batch[9]: rt_mutated_location - tensor of shape [batch_size, seq_len]
                
        Returns:
            A tuple containing processed outputs
        """
        # Extract the initial sequence for transformer processing
        initial_sequence = batch[0]  # Shape: [batch_size, seq_len, input_dim]
        
        # Convert one-hot vectors to token indices
        token_indices = self._convert_onehot_to_indices(initial_sequence)  # Shape: [batch_size, seq_len]
        
        # Apply embedding
        embedded_seq = self.embedding(token_indices)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding
        embedded_seq = self.pos_encoder(embedded_seq)
        
        # Apply transformer encoder
        # No need for attention mask as we want to attend to all positions
        transformer_embeddings = self.transformer_encoder(embedded_seq)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Create unified representation with location information
        mutated_sequence = batch[1]
        location_tensors = [batch[6], batch[7], batch[8], batch[9]]
        unified_rep = self._create_unified_representation(
            initial_sequence, 
            mutated_sequence, 
            location_tensors
        )  # Shape: [batch_size, seq_len, unified_dim]
        
        # Concatenate transformer embeddings with unified representation
        final_representation = torch.cat([transformer_embeddings, unified_rep], dim=-1)
        # Shape: [batch_size, seq_len, embedding_dim + unified_dim]
        
        # Transpose the tensor for ConvNet (from [batch, seq_len, channels] to [batch, channels, seq_len])
        final_representation_t = final_representation.transpose(1, 2)
        
        # Process the final representation with ConvNet
        conv_features = self.conv_net(final_representation_t)  # Shape: [batch_size, 64, seq_len]
        
        # Global max pooling to get a fixed-size representation
        pooled_features = torch.max(conv_features, dim=2)[0]  # Shape: [batch_size, 64]
        
        # Generate continuous regression output using MLP
        regression_output = self.regression_mlp(pooled_features)  # Shape: [batch_size, 1]
        
        # Debug: Print some statistics about the predictions
        if hasattr(self, 'debug_step') and self.debug_step < 5:
            print(f"Debug step {self.debug_step}:")
            print(f"  Pooled features mean: {torch.mean(pooled_features):.4f}")
            print(f"  Pooled features std: {torch.std(pooled_features):.4f}")
            print(f"  Regression output mean: {torch.mean(regression_output):.4f}")
            print(f"  Regression output std: {torch.std(regression_output):.4f}")
            print(f"  Regression output min: {torch.min(regression_output):.4f}")
            print(f"  Regression output max: {torch.max(regression_output):.4f}")
            self.debug_step += 1
        elif not hasattr(self, 'debug_step'):
            self.debug_step = 0
        
        # Return the final representation and regression output
        x_hat = final_representation
        y_hat = regression_output.squeeze(-1)  # Remove the last dimension to get [batch_size]

        return x_hat, y_hat
    
    def loss_function(self, predictions, targets, valid_step=False):
        """
        Implements the Mean Squared Error (MSE) loss for regression.
        
        Args:
            predictions: Tuple containing (x_hat, y_hat) where:
                - x_hat: The final representation 
                - y_hat: Continuous regression output of shape [batch_size]
            
            targets: Tuple containing:
                - batch[2]: total_read_count - total number of reads
                - batch[3]: edited_percentage - proportion of edited outcomes (used as regression target)
                - batch[4]: unedited_percentage - proportion of unedited outcomes
                - batch[5]: indel_percentage - proportion of indel outcomes
                
        Returns:
            total_loss: The MSE loss
            mloss_dict: Dictionary with loss components for logging
        """
        x_hat, y_hat = predictions
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        # Use edited_percentage as the regression target
        target_values = edited_percentage
        
        # Ensure target_values is a tensor and has the right shape
        if not isinstance(target_values, torch.Tensor):
            target_values = torch.tensor(target_values, device=y_hat.device, dtype=y_hat.dtype)
        
        # Ensure target_values has the same shape as y_hat
        if len(target_values.shape) == 0:
            target_values = target_values.unsqueeze(0)
        
        # Apply min-max scaling to ground truth values to [0,1] range
        target_values_scaled = self.scale_targets(target_values)
        
        # Compute Mean Squared Error loss on scaled values
        mse_loss = F.mse_loss(y_hat, target_values_scaled, reduction='mean')
        
        # Optional: Weight the loss by the total read count (more weight for more confident samples)
        # This assumes higher read count means more confidence in the target values
        if getattr(self.hparams, 'weight_by_read_count', False):
            # Convert to float and normalize
            weights = total_read_count.float() / torch.mean(total_read_count.float())
            weighted_mse = F.mse_loss(y_hat, target_values_scaled, reduction='none')
            weighted_mse = weighted_mse * weights
            mse_loss = torch.mean(weighted_mse)
        
        # Create dictionary with loss components for logging
        # Log both scaled and original values for monitoring
        mloss_dict = {
            'mse_loss': mse_loss.item(),
            'pred_mean': torch.mean(y_hat).item(),
            'target_scaled_mean': torch.mean(target_values_scaled).item(),
            'target_original_mean': torch.mean(target_values).item(),
            'pred_std': torch.std(y_hat).item(),
            'target_scaled_std': torch.std(target_values_scaled).item(),
            'target_original_std': torch.std(target_values).item(),
            'pred_min': torch.min(y_hat).item(),
            'pred_max': torch.max(y_hat).item(),
            'target_scaled_min': torch.min(target_values_scaled).item(),
            'target_scaled_max': torch.max(target_values_scaled).item(),
            'target_original_min': torch.min(target_values).item(),
            'target_original_max': torch.max(target_values).item()
        }
        
        return mse_loss, mloss_dict
    
    def validation_step(self, batch, batch_idx):
        preds, targets = self.shared_step(batch)

        val_loss, val_loss_logs = self.loss_function(
            predictions = preds, targets = targets, valid_step = True
        )

        val_loss_logs = self.relabel(val_loss_logs, "val_")
        
        # Log to progress bar as well
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log_dict(val_loss_logs, on_step=False, on_epoch=True)

        # Store the last batch and predictions for display in on_validation_epoch_end
        if batch_idx == 0:  # Just use the first batch
            self.last_val_batch = batch
            self.last_val_preds = preds

        # Don't convert to double if using MPS
        if torch.backends.mps.is_available() and torch.device(val_loss.device).type == 'mps':
            # Keep as float32 for MPS
            return val_loss
        else:
            # Convert to double for CPU/CUDA
            val_loss = val_loss.double()
            return val_loss

    def on_validation_epoch_end(self):
        """Print example predictions at the end of each validation epoch."""
        # Get a batch from validation dataset
        if not hasattr(self, 'last_val_batch') or not hasattr(self, 'last_val_preds'):
            print("No validation examples available to print")
            return
        
        batch = self.last_val_batch
        preds = self.last_val_preds
        
        # Extract predictions and targets
        x_hat, y_hat = preds  # y_hat is single regression value (scaled)
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
        
        # Unscale predictions to original range for comparison
        y_hat_unscaled = self.unscale_predictions(y_hat)
        
        # Get a few examples to print (up to 5)
        n_examples = min(5, len(edited_percentage))
        
        print("\n" + "="*80)
        print(f"EPOCH {self.current_epoch} VALIDATION EXAMPLES:")
        print("="*80)
        
        for i in range(n_examples):
            print(f"Example {i+1}:")
            print(f"Ground truth: Edited={edited_percentage[i]:.3f}, Unedited={unedited_percentage[i]:.3f}, Indel={indel_percentage[i]:.3f}")
            print(f"Prediction (scaled):   Edited={y_hat[i]:.3f}")
            print(f"Prediction (original): Edited={y_hat_unscaled[i]:.3f}")
            print(f"Error (original):     {abs(y_hat_unscaled[i] - edited_percentage[i]):.3f}")
            print("-"*50)
        
        print("="*80 + "\n")

    def test_step(self, batch, batch_idx):
        """Collect predictions and targets during test phase"""
        preds, targets = self.shared_step(batch)
        
        # Store the predictions and targets
        x_hat, y_hat = preds
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        # Unscale predictions to original range for evaluation
        y_hat_unscaled = self.unscale_predictions(y_hat)
        
        # Store values for later correlation calculation
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_targets = []
        
        # Move tensors to CPU for collection
        y_hat_unscaled_cpu = y_hat_unscaled.detach().cpu()
        edited_perc_cpu = edited_percentage.detach().cpu()
        
        self.test_predictions.append(y_hat_unscaled_cpu)
        self.test_targets.append(edited_perc_cpu)
        
        # Return the loss for logging (reuse validation loss calculation)
        val_loss, _ = self.loss_function(predictions=preds, targets=targets, valid_step=True)
        
        # Log to progress bar
        self.log("test_loss", val_loss, prog_bar=True, on_epoch=True)
        
        return val_loss

    def on_test_epoch_end(self):
        """Calculate Spearman correlations after testing all batches"""
        import scipy.stats as stats
        
        if not hasattr(self, 'test_predictions') or not hasattr(self, 'test_targets'):
            print("No test data available for correlation analysis")
            return
        
        # Concatenate all batches
        all_predictions = torch.cat(self.test_predictions, dim=0).numpy()
        all_targets = torch.cat(self.test_targets, dim=0).numpy()
        
        # Calculate Spearman correlation for edited percentage
        edited_corr, edited_p = stats.spearmanr(all_targets, all_predictions)
        
        # Calculate MSE
        mse = np.mean((all_targets - all_predictions) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Print results
        print("\n" + "="*80)
        print("FINAL TEST RESULTS - REGRESSION METRICS:")
        print("="*80)
        print(f"Spearman correlation: rho = {edited_corr:.4f} (p-value: {edited_p:.4e})")
        print(f"Mean Squared Error:   MSE = {mse:.6f}")
        print(f"Root Mean Squared Error: RMSE = {rmse:.6f}")
        print(f"R-squared:            R² = {r_squared:.4f}")
        print("="*80 + "\n")
        
        # Log to wandb
        self.log("test_spearman_corr", edited_corr)
        self.log("test_mse", mse)
        self.log("test_rmse", rmse)
        self.log("test_r_squared", r_squared)
        
        # Clean up
        del self.test_predictions
        del self.test_targets 