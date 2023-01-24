import torch
import dgl


class GraphArguments(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, batch):
        for ith in turns:
            # node propagation 
    