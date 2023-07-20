import torch
import torch.nn as nn
import dgl
from model.graph import DebateGraph
from model.lstm import UtterEncoder
from collections import OrderedDict

class GraphGRUArgument(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.nlast = 3*config.nhid
        self.utter_enc = UtterEncoder(config, vocab) # sent-level encoder
        self.graph = DebateGraph(config)
        self.score1 = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm1d(1)),
                ('linear1', nn.Linear(self.nlast, config.nhid//2)),
                ('relu1', nn.ReLU()), 
                ('drop1', nn.Dropout(p=0.2)),
                ('bn2', nn.BatchNorm1d(1)),
                ('linear2', nn.Linear(config.nhid//2, 1)),
                ('tanh', nn.Tanh()),
            ]))
        self.score2 = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm1d(1)),
                ('linear1', nn.Linear(self.nlast, config.nhid//2)),
                ('relu1', nn.ReLU()), 
                ('drop1', nn.Dropout(p=0.2)),
                ('bn2', nn.BatchNorm1d(1)),
                ('linear2', nn.Linear(config.nhid//2, 1)),
                ('tanh', nn.Tanh()),
            ]))
        
    def forward(self, batch):
        x = self.utter_enc(batch)
        x1, x2 = self.graph(batch, x)
        x2 = self.score1(x2)
        x1 = self.score1(x1)
        return x1.squeeze(), x2.squeeze()