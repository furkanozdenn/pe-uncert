"""
Distribution ablation study models.
Compares Dirichlet (current), Softmax/Multinomial, and Logit-Normal distributions.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from .base import crispAIPEBase
from .crispAIPE import crispAIPE
from .block_nets import ConvNet


class crispAIPE_Softmax(crispAIPE):
    """
    Variant using Softmax/Multinomial distribution.
    Outputs logits, applies softmax for predictions, uses cross-entropy loss.
    No uncertainty quantification.
    Inherits architecture from crispAIPE, only changes the final MLP and loss.
    """
    
    def __init__(self, hparams):
        # Initialize parent but replace the dirichlet_mlp with logits_mlp
        super(crispAIPE_Softmax, self).__init__(hparams)
        self.model_name = "crispAIPE_Softmax"
        
        # Replace Dirichlet MLP with logits MLP
        self.logits_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
            # No activation - raw logits for softmax
        )
        
        # Remove dirichlet_mlp reference
        if hasattr(self, 'dirichlet_mlp'):
            del self.dirichlet_mlp
        
        print(f"{self.model_name} model initialized with {self._num_params():,} parameters")
    
    def forward(self, batch):
        """Forward pass - same as crispAIPE but outputs logits instead of Dirichlet params"""
        # Use parent's forward but replace the final MLP call
        initial_sequence = batch[0]
        token_indices = self._convert_onehot_to_indices(initial_sequence)
        embedded_seq = self.embedding(token_indices)
        embedded_seq = self.pos_encoder(embedded_seq)
        transformer_embeddings = self.transformer_encoder(embedded_seq)
        
        mutated_sequence = batch[1]
        location_tensors = [batch[6], batch[7], batch[8], batch[9]]
        unified_rep = self._create_unified_representation(
            initial_sequence, mutated_sequence, location_tensors
        )
        
        final_representation = torch.cat([transformer_embeddings, unified_rep], dim=-1)
        final_representation_t = final_representation.transpose(1, 2)
        conv_features = self.conv_net(final_representation_t)
        pooled_features = torch.max(conv_features, dim=2)[0]
        
        # Generate logits instead of Dirichlet parameters
        logits = self.logits_mlp(pooled_features)  # Shape: [batch_size, 3]
        
        x_hat = final_representation
        y_hat = logits
        
        return x_hat, y_hat
    
    def loss_function(self, predictions, targets, valid_step=False):
        """
        Cross-entropy loss for multinomial distribution.
        Applies softmax to logits and computes cross-entropy with target proportions.
        """
        x_hat, y_hat = predictions
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        # Stack targets into proportions
        target_proportions = torch.stack([
            edited_percentage, unedited_percentage, indel_percentage
        ], dim=1)
        
        # Normalize to sum to 1
        target_proportions = target_proportions / torch.sum(target_proportions, dim=1, keepdim=True)
        
        # Apply softmax to logits to get predicted proportions
        pred_proportions = torch.softmax(y_hat, dim=1)
        
        # Cross-entropy loss: -sum(target * log(pred))
        epsilon = 1e-10
        log_pred = torch.log(pred_proportions + epsilon)
        cross_entropy = -torch.sum(target_proportions * log_pred, dim=1)
        
        total_loss = torch.mean(cross_entropy)
        
        # Optional weighting by read count
        if getattr(self.hparams, 'weight_by_read_count', False):
            weights = total_read_count.float() / torch.mean(total_read_count.float())
            weighted_loss = cross_entropy * weights
            total_loss = torch.mean(weighted_loss)
        
        mloss_dict = {
            'cross_entropy': total_loss.item(),
            'logits_mean': torch.mean(y_hat).item()
        }
        
        return total_loss, mloss_dict


class crispAIPE_LogitNormal(crispAIPE):
    """
    Variant using Logit-Normal distribution.
    Outputs mean and variance in logit space, samples from normal, applies softmax.
    Provides uncertainty quantification similar to Dirichlet.
    Inherits architecture from crispAIPE, only changes the final MLP and loss.
    """
    
    def __init__(self, hparams):
        super(crispAIPE_LogitNormal, self).__init__(hparams)
        self.model_name = "crispAIPE_LogitNormal"
        
        # Replace Dirichlet MLP with logit-normal MLP
        self.logit_normal_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 3 for mean, 3 for log-variance
        )
        
        # Remove dirichlet_mlp reference
        if hasattr(self, 'dirichlet_mlp'):
            del self.dirichlet_mlp
        
        print(f"{self.model_name} model initialized with {self._num_params():,} parameters")
    
    def forward(self, batch):
        """Forward pass - outputs mean and log-variance in logit space"""
        # Use parent's forward but replace the final MLP call
        initial_sequence = batch[0]
        token_indices = self._convert_onehot_to_indices(initial_sequence)
        embedded_seq = self.embedding(token_indices)
        embedded_seq = self.pos_encoder(embedded_seq)
        transformer_embeddings = self.transformer_encoder(embedded_seq)
        
        mutated_sequence = batch[1]
        location_tensors = [batch[6], batch[7], batch[8], batch[9]]
        unified_rep = self._create_unified_representation(
            initial_sequence, mutated_sequence, location_tensors
        )
        
        final_representation = torch.cat([transformer_embeddings, unified_rep], dim=-1)
        final_representation_t = final_representation.transpose(1, 2)
        conv_features = self.conv_net(final_representation_t)
        pooled_features = torch.max(conv_features, dim=2)[0]
        
        # Generate mean and log-variance instead of Dirichlet parameters
        params = self.logit_normal_mlp(pooled_features)  # Shape: [batch_size, 6]
        mean = params[:, :3]  # First 3: mean in logit space
        log_var = params[:, 3:]  # Last 3: log-variance
        
        x_hat = final_representation
        y_hat = torch.cat([mean, log_var], dim=1)  # Return both for loss computation
        
        return x_hat, y_hat
    
    def loss_function(self, predictions, targets, valid_step=False):
        """
        Loss for Logit-Normal distribution.
        Uses MSE in logit space as a simpler approximation.
        """
        x_hat, y_hat = predictions
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        # Split mean and log-variance
        mean = y_hat[:, :3]
        log_var = y_hat[:, 3:]
        var = torch.exp(log_var)
        
        # Stack targets
        target_proportions = torch.stack([
            edited_percentage, unedited_percentage, indel_percentage
        ], dim=1)
        target_proportions = target_proportions / torch.sum(target_proportions, dim=1, keepdim=True)
        
        # Convert target proportions to logit space (clamp to avoid numerical issues)
        epsilon = 1e-6
        target_proportions_clamped = torch.clamp(target_proportions, epsilon, 1 - epsilon)
        target_logits = torch.log(target_proportions_clamped) - torch.log(1 - target_proportions_clamped)
        
        # Use MSE in logit space as loss (simpler than full NLL)
        # Also add a regularization term to prevent variance from exploding
        mse_loss = torch.mean((target_logits - mean)**2, dim=1)
        var_penalty = torch.mean(var, dim=1)  # Penalize large variance
        
        # Combined loss
        nll = mse_loss + 0.1 * var_penalty
        
        total_loss = torch.mean(nll)
        
        # Optional weighting
        if getattr(self.hparams, 'weight_by_read_count', False):
            weights = total_read_count.float() / torch.mean(total_read_count.float())
            weighted_loss = nll * weights
            total_loss = torch.mean(weighted_loss)
        
        mloss_dict = {
            'logit_normal_loss': total_loss.item(),
            'mean_mean': torch.mean(mean).item(),
            'var_mean': torch.mean(var).item()
        }
        
        return total_loss, mloss_dict

