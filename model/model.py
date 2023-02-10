import torch
import torch.nn.function as F
import dgl
import torch.nn as nn
from model.gat import IntraGAT, CrossGRU

class GraphArguments(nn.Module):
    def __init__(self, config, nfeat, nhid, nheads, nclass, alpha, dropout, theta):
        # TODO: do we need different attns for different debaters?
        super().__init__()
        self.config = config
        self.theta= theta
        self.attn1 = IntraGAT(nfeat, nhid, nheads, alpha, dropout) # speaker#1
        self.attn2 = IntraGAT(nfeat, nhid, nheads, alpha, dropout) # speaker#2
        self.gru = CrossGRU(nhid, nhid, nheads, )
        self.classifier = nn.Linear(config.embed, nclass)
    
    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        num_turns = g.ndata['ids'] #TODO: check!
        for t in range(num_turns):
            speaker = t % 2
            if t > 0:
                h = self.cross_attn(g, t)

            self.attn1(g,t) if speaker==0 else self.attn2(g,t)
            node_id = g.filter_nodes(lambda nodes: nodes.data['ids'==t])


            

    def _get_nodes(self, graph, turn, name) -> Tensor: 
        """ return tensor of nodes each turn"""
        return graph.filter_nodes(lambda nodes: nodes.data[name] == turn)    
