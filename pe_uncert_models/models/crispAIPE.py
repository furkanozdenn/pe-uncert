"""
Transformer-based target conditioned Variational Auto Encoder (TbtcVAE) model. 
"""
import wandb
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


from pe_uncert_models.models.block_nets import ConvNet, LSTMNet, BottleNeck, FusionActivityAssesor, SeqConcatActivityAssesor,\
    TargetConvNetEncoder
from pe_uncert_models.models.base import crispAIPEBase
from pe_uncert_models.models.transformers import PositionalEncoding, TransformerEncoder

class crispAIPE(crispAIPEBase):

    def __init__(self, hparams):
        super(crispAIPE, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        # model parameters
        self.input_dim = hparams.input_dim # this is the size of the vocabulary 
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim
        self.embedding_dim = hparams.embedding_dim
        self.decoder_kernel_size = hparams.decoder_kernel_size
        self.layers = hparams.layers
        self.dropout = hparams.dropout
        self.nhead = hparams.nhead
        self.src_mask = None
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        self.model_name = "TbtcVAE"
        self.bottleneck_dim = 8 # hparams.bottleneck_dim TODO: add to hparams
        self.self_supervised = False # TODO: implement self supervised for unlabelled samples (SSVAE)

        self.sgrna_seq_len = hparams.sgrna_seq_len
        self.target_seq_flank_len = hparams.target_seq_flank_len
        self.target_seq_len = self.sgrna_seq_len + self.target_seq_flank_len * 2

        # check hparams for input_dim and embedding_dim - should be compatible with vocab_kmer tokens
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(d_model = self.embedding_dim, max_len = self.sgrna_seq_len)
        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.embedding_dim, 1), nn.Softmax(dim = 1)
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers = self.layers,
            input_dim = self.embedding_dim,
            num_heads = self.nhead,
            dim_feedforward = self.hidden_dim,
            dropout = self.dropout
        )

        self.bottleneck_module = BottleNeck(self.embedding_dim, self.latent_dim)
        self.z_rep = None

        ## init decoder model
        self._init_decoder(hparams)

        ## init assesor model
        self._init_assesor(hparams)

        ## init target encoder model
        self._init_target_encoder(hparams)

        ## init bottleneck encoder
        self._init_bottleneck_encoder(hparams)


    def _init_bottleneck_encoder(self, hparams):
        """Initialize bottleneck encoder model
        Inputs: sgRNA encoding, target encoding and activity score. (16 + 16 + 1 -> 16 hidden -> out)
        Outputs: The two parameters of the latent space representation (mu, logvar)
        Infer the probability distribution of p(z|x) from the input data by fitting variational
        distribution q_φ(z|x,y,t) 
        """
        layers = [
            nn.Linear(self.sgrna_seq_len + self.target_seq_len + 1, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_dim),
        ]
        self.bottleneck_encoder = nn.Sequential(*layers)


    def _init_decoder(self, hparams):
        layers = [
            nn.Linear(self.latent_dim, self.sgrna_seq_len * (self.hidden_dim // 2)),
            ConvNet(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size = 3, padding = 1),
        ]
        self.decoder = nn.ModuleList(layers)


    def _init_assesor(self, hparams):
        """Initilaize Assesor model
        TODO:
        Choose from hparams.assesor_type
        Now it only implements MLP-Assesor (target+sgRNA encodings (latent_dim * 2) -> 16 hidden -> 1 activity)
        """
        self.assesor_type = None
        if self.assesor_type == True:
            # MLP-Assesor
            layers = [
                nn.Linear(self.latent_dim * 2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ]
            self.assesor_module = nn.Sequential(*layers)
        else:
            raise ValueError(f"Assesor type {self.assesor_type} not supported. Choose from 'fusion_activity_assesor', 'seq_concat_activity_assesor'")

    def _init_target_encoder(self, hparams):
        """Initialize target encoder model
        TODO: Implement other decoders
        """
        self.target_encoder = TargetConvNetEncoder(hparams)


    def transformer_encoding(self, embedded_batch):
        """Transformer encoding of embedded_batch
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            self.src_mask = self._generate_square_subsequent_mask(embedded_batch.size(1))
        pos_encoded_batch = self.pos_encoder(embedded_batch)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)

        return output_embed

    def encode(self, batch):
        embedded_batch = self.embed(batch)
        output_embed = self.transformer_encoding(embedded_batch)
        glob_attn = self.glob_attn_module(output_embed)
        z_rep = torch.bmm(glob_attn.transpose(-1,1), output_embed).squeeze()

        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)
        
        z_rep = self.bottleneck_module(z_rep)

        return z_rep

    def decode(self, z_rep):
        """Decode z_rep to sgRNA sequence
        """
        h_rep = z_rep

        for indx, layer in enumerate(self.decoder):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.sgrna_seq_len)
            
            h_rep = layer(h_rep)
        
        return h_rep





    def forward(self, batch):

        """
        modify batch to be a tuple with target_encoding as second element
        to be used by assess_activity function
        """

        batch_target_encoding = batch[1]
        # pdb.set_trace()
        # batch_target_encoding should be of type float 
        batch_target_encoding = batch_target_encoding.float()

        batch = batch[0]
        z_rep = self.encode(batch)

        self.z_rep = z_rep
        self.dyn_interp_bool = self.interp_samping and z_rep.size(0) == self.batch_size
        if self.dyn_interp_bool:
            z_i_rep = self.interpolation_sampling(z_rep)
            interp_z_rep = torch.cat((z_rep, z_i_rep), 0)
            x_hat = self.decode(interp_z_rep)

        else:
            x_hat = self.decode(z_rep)

        # regardless of interpolative sampling, assesor x_hat is based on actual z_rep
        x_hat_assesor = self.decode(z_rep)

        self.dyn_neg_bool = self.negative_sampling and z_rep.size(0) == self.batch_size
        if self.dyn_neg_bool:
            z_n_rep = self.add_negative_samples()
            neg_z_rep = torch.cat((z_rep, z_n_rep), 0)

            # extend x_hat_assesor with negative samples
            z_n_rep_dec = self.decode(z_n_rep)

            x_hat_assesor = torch.cat((x_hat_assesor, z_n_rep_dec), 0)

            if self.assesor_type == "fusion_activity_assesor":
                y_hat = self.assesor_module(neg_z_rep, batch_target_encoding)
            elif self.assesor_type == "seq_concat_activity_assesor":
                y_hat = self.assesor_module(x_hat_assesor, batch_target_encoding)
            else:
                raise ValueError(f"No assesor type {self.assesor_type} found")

        else:
            if self.assesor_type == "fusion_activity_assesor":
                y_hat = self.assesor_module(z_rep, batch_target_encoding)
            elif self.assesor_type == "seq_concat_activity_assesor":
                y_hat = self.assesor_module(x_hat_assesor, batch_target_encoding)
            else:
                raise ValueError(f"No assesor type {self.assesor_type} found")

        return [x_hat, y_hat], z_rep


    
    def loss_function(self, predictions, targets, valid_step = False):

        x_hat, y_hat = predictions
        x_true, y_true = targets

        total_loss = 0.0

        # TODO: Implement loss function
        mloss_dict = {
        }
        return total_loss, mloss_dict