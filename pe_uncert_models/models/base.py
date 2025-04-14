'''
Pytorch lightning base classes for TbtcVAE model.
'''

import torch
from torch import nn

import pytorch_lightning as pl
import argparse

import pdb


class crispAIPEBase(pl.LightningModule):
    def __init__(self, hparams):
        super(crispAIPEBase, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.lr = hparams.lr

    def configure_optimizers(self):
        opt = (torch.optim.Adam(self.parameters(), lr=self.lr))
        return opt

    def shared_step(self, batch):


        """
        batch[0] -> initial_sequence 
        batch[1] -> mutated_sequence
        batch[2] -> total_read_count
        batch[3] -> edited_percentage
        batch[4] -> unedited_percentage
        batch[5] -> indel_percentage
        batch[6] -> protospacer_location
        batch[7] -> pbs_location
        batch[8] -> rt_initial_location
        batch[9] -> rt_mutated_location
        """

        data = batch
        targets = batch[2], batch[3], batch[4], batch[5]

        preds = self(data)
        
        return preds, targets

    def forward(self, batch):
        raise NotImplementedError

    def relabel(self, loss_dict, label):
        loss_dict = {label + str(key): val for key,val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):
        preds, targets = self.shared_step(batch)
        # assert len(preds) == len(targets) +1, f"Preds: {len(preds)}, Targets: {len(targets)}"

        train_loss, train_loss_logs = self.loss_function(
            predictions = preds, targets = targets
        )

        train_loss_logs = self.relabel(train_loss_logs, "train_")
        
        # Log to progress bar as well as to logger
        self.log("train_loss", train_loss, prog_bar=True, on_step=True)
        self.log_dict(train_loss_logs, on_step=True, on_epoch=False)

        # Don't convert to double if using MPS
        if torch.backends.mps.is_available() and torch.device(train_loss.device).type == 'mps':
            # Keep as float32 for MPS
            return train_loss
        else:
            # Convert to double for CPU/CUDA
            train_loss = train_loss.double()
            return train_loss

    def validation_step(self, batch, batch_idx):
        preds, targets = self.shared_step(batch)
        # assert len(preds) == len(targets) + 1, f"Preds: {len(preds)}, Targets: {len(targets)}"

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

    def loss_function(self, predictions, targets, valid_step = False):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_criterion(self):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

    def get_scheduler(self):
        raise NotImplementedError

    def get_dataloader(self, mode):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    def on_validation_epoch_end(self):
        """Print example predictions at the end of each validation epoch."""
        # Get a batch from validation dataset
        if not hasattr(self, 'last_val_batch') or not hasattr(self, 'last_val_preds'):
            print("No validation examples available to print")
            return
        
        batch = self.last_val_batch
        preds = self.last_val_preds
        
        # Extract predictions and targets
        x_hat, y_hat = preds  # y_hat is Dirichlet alpha parameters
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = batch[2:6]
        
        # Calculate expected values from the Dirichlet distribution
        # E[X_i] = α_i / Σ_j α_j
        alpha_sum = torch.sum(y_hat, dim=1, keepdim=True)
        expected_proportions = y_hat / alpha_sum
        
        # Get a few examples to print (up to 5)
        n_examples = min(5, len(edited_percentage))
        
        print("\n" + "="*80)
        print(f"EPOCH {self.current_epoch} VALIDATION EXAMPLES:")
        print("="*80)
        
        for i in range(n_examples):
            print(f"Example {i+1}:")
            print(f"Ground truth: Edited={edited_percentage[i]:.3f}, Unedited={unedited_percentage[i]:.3f}, Indel={indel_percentage[i]:.3f}")
            print(f"Prediction:   Edited={expected_proportions[i,0]:.3f}, Unedited={expected_proportions[i,1]:.3f}, Indel={expected_proportions[i,2]:.3f}")
            print(f"Alphas:       α_edit={y_hat[i,0]:.3f}, α_unedited={y_hat[i,1]:.3f}, α_indel={y_hat[i,2]:.3f}, Σα={alpha_sum[i,0]:.3f}")
            
            # Calculate uncertainty (lower alpha_sum = higher uncertainty)
            print(f"Uncertainty indicator: {1/alpha_sum[i,0]:.5f} (lower is more certain)")
            print("-"*50)
        
        print("="*80 + "\n")

    def test_step(self, batch, batch_idx):
        """Collect predictions and targets during test phase"""
        preds, targets = self.shared_step(batch)
        
        # Store the predictions and targets
        x_hat, y_hat = preds
        total_read_count, edited_percentage, unedited_percentage, indel_percentage = targets
        
        # Calculate expected values from Dirichlet distribution
        alpha_sum = torch.sum(y_hat, dim=1, keepdim=True)
        expected_proportions = y_hat / alpha_sum
        
        # Store values for later correlation calculation
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_targets = []
        
        # Move tensors to CPU for collection
        expected_props_cpu = expected_proportions.detach().cpu()
        edited_perc_cpu = edited_percentage.detach().cpu() 
        unedited_perc_cpu = unedited_percentage.detach().cpu()
        indel_perc_cpu = indel_percentage.detach().cpu()
        
        self.test_predictions.append(expected_props_cpu)
        self.test_targets.append(
            torch.stack([edited_perc_cpu, unedited_perc_cpu, indel_perc_cpu], dim=1)
        )
        
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
        
        # Extract individual outcome types
        pred_edited = all_predictions[:, 0]
        pred_unedited = all_predictions[:, 1]
        pred_indel = all_predictions[:, 2]
        
        true_edited = all_targets[:, 0]
        true_unedited = all_targets[:, 1]
        true_indel = all_targets[:, 2]
        
        # Calculate Spearman correlations
        edited_corr, edited_p = stats.spearmanr(true_edited, pred_edited)
        unedited_corr, unedited_p = stats.spearmanr(true_unedited, pred_unedited)
        indel_corr, indel_p = stats.spearmanr(true_indel, pred_indel)
        
        # Calculate overall correlation by combining all outcomes
        all_pred = all_predictions.flatten()
        all_true = all_targets.flatten() 
        overall_corr, overall_p = stats.spearmanr(all_true, all_pred)
        
        # Print results
        print("\n" + "="*80)
        print("FINAL TEST RESULTS - SPEARMAN CORRELATIONS:")
        print("="*80)
        print(f"Edited outcomes:   rho = {edited_corr:.4f} (p-value: {edited_p:.4e})")
        print(f"Unedited outcomes: rho = {unedited_corr:.4f} (p-value: {unedited_p:.4e})")
        print(f"Indel outcomes:    rho = {indel_corr:.4f} (p-value: {indel_p:.4e})")
        print(f"Overall:           rho = {overall_corr:.4f} (p-value: {overall_p:.4e})")
        print("="*80 + "\n")
        
        # Log to wandb
        self.log("test_edited_corr", edited_corr)
        self.log("test_unedited_corr", unedited_corr)
        self.log("test_indel_corr", indel_corr)
        self.log("test_overall_corr", overall_corr)
        
        # Clean up
        del self.test_predictions
        del self.test_targets
