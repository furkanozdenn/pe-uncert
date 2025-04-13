'''NNs used in CRISPRoposer

- ConvNets
- (bi)-LSTMs
- FC Layers
- Assesors for cleavage activity prediction
'''

import torch
from torch.nn import functional as F    
from torch import nn


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





