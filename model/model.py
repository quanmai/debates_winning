import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from model.gat import GAT, CrossGAT, GRUMultiplexer, CrossGG, GATGRUCell

MAX_TURNS = 10


class GraphGRUArguments(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gatgrucell = GATGRUCell(config.nfeat, config.nhid, config.nhead, config.alpha, config.dropout)
        self.fc = nn.Linear(config.nhid, config.nhid//2)
        self.score = nn.Linear(config.nhid//2, 1) # real value score
        # self.register_buffer('read_out_loud', torch.zeros(1, config.nhid))
    
    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        turns = torch.unique(g.ndata['ids'], sorted=True)
        if len(turns) > 10 or len(turns) < 6:
            print(f'Argument has {len(turns)} turns')
        for t in turns:
            _ = self.gatgrucell(g, t)
        # read-out
        # if self.config.loss=='pair':
        # h1 = self._read_out(g, -2, op='mean') # second-last argument
        # h2 = self._read_out(g, -1, op='mean') # last argument
        h1, h2 = self._read_out_loud(g, op='mean')
        # Classification: speaker#1 wins if s1>s2
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
            # TODO: back later
            if op == 'mean':
                a = torch.mean(h, dim=-2, keepdim=True)
                # print(a.shape)
            elif op == 'max':
                a = torch.max(h, dim=-2, keepdim=True).values
            else:
                raise AssertionError("Unexpected value of 'op'!", op)
            res.append(a)
        r = torch.cat(res, dim=0).unsqueeze(1)
        return r

    def _read_out_loud(self, gg, op='mean'):
        gl = dgl.unbatch(gg)
        res1 = [] # list of tensor
        res2 = [] # list of tensor
        for g in gl:
            turns = torch.max(g.ndata['ids']).item()+1
            readout_s1 = torch.zeros((1, self.config.nhid), device=self.config.device)
            readout_s2 = torch.zeros((1, self.config.nhid), device=self.config.device)
            # readout_s2 = self.read_out_loud
            # print(readout_s1)
            for turn in range(turns):
                node_idx = g.filter_nodes(lambda nodes: nodes.data['ids']==turn)
                h = g.nodes[node_idx].data['hp']
                if op == 'mean':
                    a = torch.mean(h, dim=-2, keepdim=True)
                elif op == 'max':
                    a = torch.max(h, dim=-2, keepdim=True).values
                else:
                    raise AssertionError("Unexpected value of 'op'!", op)
                if turn % 2 == 0:
                    readout_s1 += a
                else:
                    readout_s2 += a
            res1.append(readout_s1)
            res2.append(readout_s2)
        r1 = torch.cat(res1, dim=0).unsqueeze(1)
        r2 = torch.cat(res2, dim=0).unsqueeze(1)
        return r1, r2


class GraphArguments(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attns = nn.ModuleList([GAT(config.nfeat, config.nhid, config.nhead, config.alpha, config.dropout) for _ in range(MAX_TURNS)])
        if config.is_counter:
            self.cnter_attns = nn.ModuleList([CrossGG(config.nhid, config.nhid, config.nhead, config.alpha, config.dropout) for _ in range(MAX_TURNS-1)])
            # self.cnter_attns = nn.ModuleList([CrossGAT(config.nhid, config.nhead, config.alpha, config.dropout) for _ in range(MAX_TURNS-1)])
        if config.is_support:
            self.spprt_attns = nn.ModuleList([CrossGG(config.nhid, config.nhid, config.nhead, config.alpha, config.dropout) for _ in range(MAX_TURNS-2)])
            # self.spprt_attns = nn.ModuleList([CrossGAT(config.nhid, config.nhead, config.alpha, config.dropout) for _ in range(MAX_TURNS-2)])
        if config.is_counter or config.is_support:
            self.gru = nn.ModuleList([GRUMultiplexer(config.nhid, config.nhid) for _ in range(MAX_TURNS-1)])
        self.fc = nn.Linear(config.nhid, config.nhid*2) # real value score
        self.score = nn.Linear(config.nhid*2, 1) # real value score
    
    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        turns = torch.unique(g.ndata['ids'], sorted=True)
        if len(turns) > 10:
            print(f'Argument has {len(turns)} turns')
        for t in turns:
            cnter_feat = None
            spprt_feat = None
            # Intra-argument (self) attention
            self.attns[t](g,t)
            # Inter-argument (counter-support) attention
            # TODO: check why t-2, t-1 instead of t
            if self._do_counter(t):
                cnter_feat, node_idx = self.cnter_attns[t-1](g, t-1, 'counter')
            if self._do_support(t):
                spprt_feat, node_idx = self.spprt_attns[t-2](g, t-2, 'support') 
            if self._do_counter(t) or self._do_counter(t):
                intra_feat = g.nodes[node_idx].data['hp']
                self.gru[t-1](g, node_idx, intra_feat, spprt_feat, cnter_feat)
        # read-out
        h1 = self._read_out(g, -2, op='mean') # second-last argument
        h2 = self._read_out(g, -1, op='mean') # last argument
        # Classification: speaker#1 wins if s1>s2
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

    def _do_counter(self, t):
        return self.config.is_counter and t > 0
    
    def _do_support(self, t):
        return self.config.is_support and t > 1