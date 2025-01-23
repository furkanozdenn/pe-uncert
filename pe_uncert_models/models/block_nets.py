'''NNs used in CRISPRoposer

- ConvNets
- (bi)-LSTMs
- Transformers
- BottleNeck layer
- FC Layers
- Assesors for cleavage activity prediction
'''

import torch
from torch.nn import functional as F
from torch.nn import MultiheadAttention 
from torch import nn

import math
import numpy as np

import pdb
from collections import OrderedDict

import argparse
import wandb

from pe_uncert_models.models.math_utils import scaled_dot_product


## ConvNets
class ConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """ ConvNet with residual connections
        # TODO: args
        Args: 
        """
        super(ConvNet, self).__init__()
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels))

    def forward(self, x):

        identity = x

        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out

class LSTMNet(nn.Module):
    """Takes output of ConvNet as input 
    and returns last hidden state of LSTM
    bidirectional = True by default 
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def forward(self, x):
        # pdb.set_trace()
        # x = x.view(x.size(0), 1, -1)
        # print(x.shape)
        out, (h_n, c_n) = self.lstm(x)
        # print(out.shape)
        # print(h_n.shape)
        # print(c_n.shape)
        return h_n




# BottleNeck class
class BottleNeck(nn.Module):
    """ FCN BottleNeck layer
    TODO: Args
    """
    def __init__(self, input_dim, bottleneck_dim):
        super(BottleNeck, self).__init__()

        self.fc = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h): # h for latent space representation

        z_rep = self.fc(h)

        return z_rep


class FusionActivityAssesor(nn.Module):

    """bi-LSTM, CNN Fusion nn to assess cleavage activity of the interface (cnn lstm order)
    Inputs: 
    z_latent, 4-dim ohe DNA 
    Out:
    Activity score

    Uses the serial combination of ConvNet (residual connections) and LSTMNet (bidirectional)
    Final MLP layer predicts the cleavage activity score
    """
    def __init__(
        self,
        bottleneck_dim,
        target_seq_len, 
        in_channels = 5,
        out_channels = 16,
        kernel_size = 1,
        stride = 1,
        lstm_hidden = 64,
        linear_hidden = 64, # reduce from 128 to 64
        num_lstm_layers = 1
    ):
        super(FusionActivityAssesor, self).__init__()

        self.bottleneck_dim = bottleneck_dim
        self.lstm_hidden = lstm_hidden
        self.linear_hidden = linear_hidden
        self.num_lstm_layers = num_lstm_layers

        self.target_seq_len = target_seq_len

        self.convnet = ConvNet(in_channels, out_channels, stride)
        self.lstmnet = LSTMNet(out_channels, lstm_hidden, num_lstm_layers)

        self.fc1 = nn.Linear(lstm_hidden * 2, linear_hidden) # bidirectional lstm
        self.fc2 = nn.Linear(linear_hidden, bottleneck_dim) # reduce to bottleneck dim
        self.fc3 = nn.Linear(bottleneck_dim*2, 1)

    def forward(self, z_latent, target_encoding):
            pdb.set_trace()
            x = target_encoding
            # ConvNet pass: expects 3d input (batch_size, in_channels, target_seq_len)
            x = x.permute(0, 2, 1)
            # pdb.set_trace()
            x = self.convnet(x)
            x = x.permute(0, 2, 1)

            # LSTMNet pass, is bidirectional by default - output shape: (2, batch_size, hidden_dim)
            x = self.lstmnet(x)

            # flatten before dense layers
            x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1) # concats hidden states in both directions

            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)

            # concat with z_latent, TODO: try summation and other variations
            x = torch.cat((x, z_latent), dim=1)

            x = self.fc3(x)
            # apply ReLU
            x = F.relu(x)

            return x


class SeqConcatActivityAssesor(nn.Module):

    """bi-LSTM, CNN Fusion nn to assess cleavage activity of the interface (cnn lstm order)
    Inputs: 
    decoded_sgRNA: instead of z_latent, this model uses the decoded sgRNA sequence
    target_seq: 5-dim ohe DNA, instead of 4-dim ohe DNA, now includes N to match sgRNA vocab
    Out:
    Activity score
    """
    def __init__(
        self,
        target_seq_len, 
        in_channels = 10, # 10 instead of 4, decoded sgRNA and target seq are concatenated each with 5-dim ohe 
        out_channels = 32,
        kernel_size = 1,
        stride = 1,
        lstm_hidden = 64,
        linear_hidden = 64, # reduce from 128 to 64
        num_lstm_layers = 1
    ):
        super(SeqConcatActivityAssesor, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.linear_hidden = linear_hidden
        self.num_lstm_layers = num_lstm_layers

        self.target_seq_len = target_seq_len

        self.convnet = ConvNet(in_channels, out_channels, stride)
        self.lstmnet = LSTMNet(out_channels, lstm_hidden, num_lstm_layers)

        self.fc1 = nn.Linear(lstm_hidden * 2, linear_hidden) # bidirectional lstm
        self.fc2 = nn.Linear(linear_hidden, 1)

    def forward(self, decoded_sgRNA, target_encoding):
        # decoded_sgRNA_soft = F.softmax(decoded_sgRNA, dim=1)
        # use F.gumbel_softmax instead 

        # TODO: optimize temprature, usually T < 1
        decoded_sgRNA_gs = F.gumbel_softmax(decoded_sgRNA, tau=0.1, dim =1, hard=True)
        # decoded_sgRNA = decoded_sgRNA_soft # use softmax to keep function differentiable
        decoded_sgRNA = decoded_sgRNA_gs

        # reshape to match target_encoding shape (batch_size, target_seq_len, vocab_size)
        # change axis 1 and 2
        # pdb.set_trace()
        decoded_sgRNA = decoded_sgRNA.permute(0, 2, 1)

        # concat decoded_sgRNA with target_encoding
        x = torch.cat((decoded_sgRNA, target_encoding), dim=2)
        # ConvNet pass: expects 3d input (batch_size, in_channels, target_seq_len)

        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)

        # LSTMNet pass, is bidirectional by default - output shape: (2, batch_size, hidden_dim)
        x = self.lstmnet(x)

        # flatten before dense layers
        x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # apply ReLU as final activation
        # x = F.relu(x)
        # apply tanh 
        x = torch.tanh(x) * 4

        return x

class MLP(nn.Module):

    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()

        q = []

        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i==len(hidden_size) - 2) and last_activation):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))

        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)

class Z_Encoder(nn.Module):
    """Encoder for latent space representation
    Example dimensions:
    input_dim: 16 (sgRNA encoding) + 16 (target encoding) + 1 (activity score) = 33
    hidden_dim: 16
    latent_dim: 8
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Z_Encoder, self).__init__()

        self.calc_mean = MLP([input_dim, hidden_dim, latent_dim], last_activation = False)
        self.calc_logvar = MLP([input_dim, hidden_dim, latent_dim], last_activation = False)

    def forward(self, x, y = None):

        if y is not None:
            x = torch.cat((x, y), dim=1)
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(x), self.calc_logvar(x) 
        

class Z_Decoder(nn.Module):
    """Decoder for latent space representation
    Decodes into sgRNA sequence only
            layers = [
            nn.Linear(self.latent_dim, self.sgrna_seq_len * (self.hidden_dim // 2)),
            ConvNet(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size = 3, padding = 1),
        ]
        self.decoder = nn.ModuleList(layers)

        fwd: 
            h_rep = z_rep

        for indx, layer in enumerate(self.decoder):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.sgrna_seq_len)
            
            h_rep = layer(h_rep)
        
        return h_rep
    """

    def __init__(self, latent_dim, sgrna_seq_len, seq_encoding_dim, input_dim = 5):
        super(Z_Decoder, self).__init__()

        self.sgrna_seq_len = sgrna_seq_len
        self.seq_encoding_dim = seq_encoding_dim
        self.input_dim = input_dim

        # below architecture assumes activity_score != None TODO: refactor
        layers = [
            nn.Linear(latent_dim + seq_encoding_dim + 1, sgrna_seq_len * (seq_encoding_dim // 2)),
            ConvNet(seq_encoding_dim // 2, seq_encoding_dim),
            nn.Conv1d(seq_encoding_dim, input_dim, kernel_size = 3, padding = 1),
        ]
        self.decoder = nn.ModuleList(layers)

    def forward(self, z_latent, target_encoding, activity_score = None):

        if activity_score is not None:
            x = torch.cat((z_latent, target_encoding, activity_score), dim=1)
        else:
            x = torch.cat((z_latent, target_encoding), dim=1)

        h_rep = x

        for indx, layer in enumerate(self.decoder):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.seq_encoding_dim // 2, self.sgrna_seq_len)
            
            h_rep = layer(h_rep)

        # TODO: consider softmax here
        # add softmax
        h_rep = F.softmax(h_rep, dim=1)

        return h_rep


class TargetConvNetEncoder(nn.Module):

    """ConvNet with residual connections to encode target sequence
    Inputs: 
    target_seq: 5-dim ohe DNA seq
    Out:
    Encoding of target sequence
    """
    def __init__(
        self,
        target_seq_len, 
        in_channels = 5, # 10 instead of 4, decoded sgRNA and target seq are concatenated each with 5-dim ohe 
        out_channels = 16,
        kernel_size = 1,
        stride = 1,
        linear_hidden = 32, 
        out_dim = 16
    ):
        super(TargetConvNetEncoder, self).__init__()
        self.target_seq_len = target_seq_len

        self.linear_hidden = linear_hidden
        self.convnet = ConvNet(in_channels, out_channels, stride)
        self.fc1 = nn.Linear(out_channels * target_seq_len, linear_hidden) # bidirectional lstm
        self.fc2 = nn.Linear(linear_hidden, out_dim)

    def forward(self, target_encoding):
        # ConvNet pass: expects 3d input (batch_size, in_channels, target_seq_len)

        x = target_encoding
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)

        # flatten for dense layers, use reshape instead of view
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x



