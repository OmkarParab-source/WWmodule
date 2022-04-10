import torch
import torch.nn as nn

class lstm(nn.Module):

    def __init__(
        self,
        num_classes,
        features,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        device='cpu'
    ):
        super(lstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1
        self.device = device
        self.layer_norm = nn.LayerNorm(features)
        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.classifier = nn.Linear(
            hidden_size*self.directions,
            num_classes
        )
    
    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return (torch.zeros(n*d, batch_size, hs).to(self.device),
                torch.zeros(n*d, batch_size, hs).to(self.device))

    def forward(self, x):
        x = self.layer_norm(x)
        hidden = self._init_hidden(x.size()[1])
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn)
        return out