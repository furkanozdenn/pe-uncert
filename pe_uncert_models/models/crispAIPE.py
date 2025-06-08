"""
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


class crispAIPE(crispAIPEBase):

    def __init__(self, hparams):
        super(crispAIPE, self).__init__(hparams)

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
        self.model_name = "crispAIPE"

        self.sequence_length = getattr(hparams, 'sequence_length', 99) # assuming initial sequence and modified sequence are of the same length
        self.target_seq_flank_len = getattr(hparams, 'target_seq_flank_len', 0) # keeping no flank around DNA sequence
        
        # Transformer encoder parameters - significantly reduced for fewer parameters
        self.embedding_dim = getattr(hparams, 'embedding_dim', 8)
        self.nhead = getattr(hparams, 'nhead', 2)
        self.num_encoder_layers = getattr(hparams, 'num_encoder_layers', 2)  # Reduced from 6 to 2
        self.dim_feedforward = getattr(hparams, 'dim_feedforward', 32)  # Reduced from 256 to 32
        self.dropout = getattr(hparams, 'dropout', 0.1)
        
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
        
        # MLP to output 3 Dirichlet parameters
        self.dirichlet_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus()  # Ensure parameters are positive for Dirichlet
        )

        print(f"Number of parameters: {self._num_params()}")
        self._print_param_breakdown()


    def _num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _print_param_breakdown(self):
        """Print breakdown of parameters by component"""
        embedding_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters() if p.requires_grad)
        convnet_params = sum(p.numel() for p in self.conv_net.parameters() if p.requires_grad)
        mlp_params = sum(p.numel() for p in self.dirichlet_mlp.parameters() if p.requires_grad)
        
        print(f"Parameter breakdown:")
        print(f"  Embedding: {embedding_params:,}")
        print(f"  Transformer encoder: {transformer_params:,}")
        print(f"  ConvNet: {convnet_params:,}")
        print(f"  Dirichlet MLP: {mlp_params:,}")
    
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
        
        # Generate Dirichlet parameters using MLP
        dirichlet_params = self.dirichlet_mlp(pooled_features)  # Shape: [batch_size, 3]
        
        # Return the final representation and Dirichlet parameters
        x_hat = final_representation
        y_hat = dirichlet_params

        return x_hat, y_hat
    
    def loss_function(self, predictions, targets, valid_step=False):
        """
        Implements the negative log likelihood loss for a Dirichlet distribution.
        
        Args:
            predictions: Tuple containing (x_hat, y_hat) where:
                - x_hat: The final representation 
                - y_hat: Dirichlet parameters (α) of shape [batch_size, 3]
            
            targets: Tuple containing:
                - batch[2]: total_read_count - total number of reads
                - batch[3]: edited_percentage - proportion of edited outcomes
                - batch[4]: unedited_percentage - proportion of unedited outcomes
                - batch[5]: indel_percentage - proportion of indel outcomes
                
        Returns:
            total_loss: The negative log likelihood loss
            mloss_dict: Dictionary with loss components for logging
        """
        x_hat, y_hat = predictions
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        # Stack the percentage targets into a tensor of shape [batch_size, 3]
        # Each row contains [edited, unedited, indel] proportions
        dirichlet_targets = torch.stack([
            edited_percentage, 
            unedited_percentage, 
            indel_percentage
        ], dim=1)
        
        # Ensure the proportions sum to 1 (normalize)
        dirichlet_targets = dirichlet_targets / torch.sum(dirichlet_targets, dim=1, keepdim=True)
        
        # The Dirichlet parameters (concentration parameters alpha)
        alpha = y_hat  # Shape: [batch_size, 3]
        
        # Compute alpha_0 = sum(alpha)
        alpha_0 = torch.sum(alpha, dim=1, keepdim=True)
        
        # Compute negative log likelihood
        # NLL = log(B(α)) - ∑ᵢ (αᵢ-1) log(xᵢ)
        # Where B(α) is the beta function: B(α) = ∏ᵢ Γ(αᵢ) / Γ(∑ᵢ αᵢ)
        # log(B(α)) = ∑ᵢ log(Γ(αᵢ)) - log(Γ(∑ᵢ αᵢ))
        
        # Compute log(B(α)) using lgamma (log of gamma function)
        log_beta = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(alpha_0.squeeze())
        
        # Compute ∑ᵢ (αᵢ-1) log(xᵢ)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_likelihood_kernel = torch.sum((alpha - 1.0) * torch.log(dirichlet_targets + epsilon), dim=1)
        
        # Compute negative log likelihood
        nll = log_beta - log_likelihood_kernel
        
        # Compute the mean NLL across the batch
        total_loss = torch.mean(nll)
        
        # Optional: Weight the loss by the total read count (more weight for more confident samples)
        # This assumes higher read count means more confidence in the proportions
        if getattr(self.hparams, 'weight_by_read_count', False):
            # Convert to float and normalize
            weights = total_read_count.float() / torch.mean(total_read_count.float())
            weighted_nll = nll * weights
            total_loss = torch.mean(weighted_nll)
        
        # Create dictionary with loss components for logging
        mloss_dict = {
            'dirichlet_nll': total_loss.item(),
            'alpha_mean': torch.mean(alpha).item(),
            'alpha_0_mean': torch.mean(alpha_0).item()
        }
        
        return total_loss, mloss_dict