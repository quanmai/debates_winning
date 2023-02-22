import torch
import torch.nn.function as F
import dgl
import torch.nn as nn
from model.gat import GAT, CrossGAT

class GraphArguments(nn.Module):
    def __init__(self, config, nfeat, nhid, nheads, nclass, alpha, dropout, theta, is_counter, is_support):
        # TODO: do we need different attns for different debaters?
        super().__init__()
        self.is_counter = is_counter
        self.is_support = is_support
        self.config = config
        self.theta = theta
        self.attn1 = GAT(nfeat, nhid, nheads, alpha, dropout) # Debater#1
        self.attn2 = GAT(nfeat, nhid, nheads, alpha, dropout) # Debater#2
        self.counter_attn = CrossGAT(nhid, nheads, alpha, dropout) # CrossGAT does not change the dimension
        self.score = nn.Linear(config.embed, 1) # real value score
        # self.classifier2 = nn.Linear(config.embed, nclass)
    
    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        num_turns = g.ndata['ids'] #TODO: check!
        for t in range(num_turns):
            # First self-attn
            speaker = t % 2
            self.attn1(g,t) if speaker==0 else self.attn2(g,t)
            # Then Counter attn
            if self.is_counter and t > 0:
                h = self.counter_attn(g, t-1)
        # read-out
        h1 = self._read_out(g, num_turns-2, op='mean')
        h2 = self._read_out(g, num_turns-1, op='mean')
        # Classification
        # speaker#1 wins if s1>s2
        s1 = F.log_softmax(F.relu(self.score(h1)))
        s2 = F.log_softmax(F.relu(self.score(h2)))
        return s1, s2
    
    def _read_out(self, g, t, op='mean'):
        node_idx = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        h = node_idx['h']
        # nicer way: https://stackoverflow.com/questions/23131594/choose-which-function-to-execute-based-on-a-parameter-its-name
        # be back later
        if op == 'mean':
            return torch.mean(h, dim=-1, keepdim=True)
        elif op == 'max':
            return torch.max(h, dim=-1, keepdim=True)
        else:
            raise AssertionError("Unexpected value of 'op'!", op)

    # def _get_nodes(self, graph, turn, name) -> Tensor: 
    #     """ return tensor of nodes each turn"""
    #     return graph.filter_nodes(lambda nodes: nodes.data[name] == turn)
