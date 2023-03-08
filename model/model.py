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
        self.score = nn.Linear(config.nhid, 1) # real value score
    
    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        turns = torch.unique(g.ndata['ids'], sorted=True) 
        # print(f'number of turns: {turns}')
        for t in turns:
            # First self-attn
            speaker = t % 2
            self.attn1(g,t) if speaker==0 else self.attn2(g,t)
            # Then Counter attn
            if self.config.is_counter and t > 0:
                h = self.counter_attn(g, t-1)
        # read-out
        h1 = self._read_out(g, turns[-2], op='mean')
        # print(f'h1 shape: {h1.shape}')
        h2 = self._read_out(g, turns[-1], op='mean')
        # print(f'h2 shape: {h2.shape}')
        # Classification
        # speaker#1 wins if s1>s2
        s1 = self.score(h1)
        s2 = self.score(h2)
        return s1, s2
    
    def _read_out(self, g, t, op='mean'):
        node_idx = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        h = g.nodes[node_idx].data['hp']
        # nicer way: https://stackoverflow.com/questions/23131594/choose-which-function-to-execute-based-on-a-parameter-its-name
        # be back later
        if op == 'mean':
            return torch.mean(h, dim=-2, keepdim=True)
        elif op == 'max':
            return torch.max(h, dim=-2, keepdim=True)
        else:
            raise AssertionError("Unexpected value of 'op'!", op)

    # def _get_nodes(self, graph, turn, name) -> Tensor: 
    #     """ return tensor of nodes each turn"""
    #     return graph.filter_nodes(lambda nodes: nodes.data[name] == turn)
