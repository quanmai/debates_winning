import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from model.gat import GAT, CrossGAT

class GraphArguments(nn.Module):
    def __init__(self, config):
        # TODO: do we need different attns for different debaters?
        super().__init__()
        self.config = config
        self.attn1 = GAT(config.nfeat, config.nhid, config.nhead, config.alpha, config.dropout) # Debater#1
        self.attn2 = GAT(config.nfeat, config.nhid, config.nhead, config.alpha, config.dropout) # Debater#1
        self.counter_attn = CrossGAT(config.nhid, config.nhead, config.alpha, config.dropout) # CrossGAT does not change the dimension
        self.fc = nn.Linear(config.nhid, config.nhid*2) # real value score
        self.score = nn.Linear(config.nhid*2, 1) # real value score
    
    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        turns = torch.unique(g.ndata['ids'], sorted=True) 
        for t in turns:
            # First self-attn
            speaker = t % 2
            self.attn1(g,t) if speaker==0 else self.attn2(g,t)
            # Then Counter attn
            if self.config.is_counter and t > 0:
                h = self.counter_attn(g, t-1)
        # read-out
        # with g.local_scope():
        h1 = self._read_out(g, -2, op='mean')
        h2 = self._read_out(g, -1, op='mean')
        # Classification
        # speaker#1 wins if s1>s2
        s1 = self.score(F.relu(self.fc(h1)))
        s2 = self.score(F.relu(self.fc(h2)))
        return s1.squeeze(), s2.squeeze()
    
    def _read_out(self, gg, t, op='mean'):
        gl = dgl.unbatch(gg)
        res = [] # list of tensor
        for g in gl:
            turns = torch.max(g.ndata['ids']).item()+1
            node_idx = g.filter_nodes(lambda nodes: nodes.data['ids']==turns+t)
            h = g.nodes[node_idx].data['hp']
            # nicer way: https://stackoverflow.com/questions/23131594/choose-which-function-to-execute-based-on-a-parameter-its-name
            # TODO: back later
            if op == 'mean':
                a = torch.mean(h, dim=-2, keepdim=True)
            elif op == 'max':
                a = torch.max(h, dim=-2, keepdim=True)
            else:
                raise AssertionError("Unexpected value of 'op'!", op)
            res.append(a)
        r = torch.cat(res, dim=0).unsqueeze(1)
        return r
