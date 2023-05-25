import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from model.gat import GATGRUCell, GATGRUCellInversed
from model.degree_encoder import DegreeEncoder
from collections import OrderedDict

MAX_TURNS = 10


class GraphGRUArguments(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nlast = int(1.5*config.nhid) if config.mode=='bidirection' else config.nhid
        self.degree_encoder = DegreeEncoder(max_degree=20, 
                                            embedding_dim=config.nfeat if config.node_encoder == 'first' else config.nhid, 
                                            direction=config.node_encoder_direction)
        self.gatgrucell = GATGRUCell(config.nfeat, config.nhid, config.nhead, config.alpha, config.dropout)
        if config.mode=='bidirection':
            self.gatgrucellback = GATGRUCellInversed(config.nhid, config.nhid//2, config.nhead, config.alpha, config.dropout)
        if config.loss == 'binary':
            self.layers = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm1d(1)),
                ('linear1', nn.Linear(2*self.nlast, config.nhid//2)),
                ('relu1', nn.ReLU()),
                ('drop1', nn.Dropout(p=0.2)),
                ('bn2', nn.BatchNorm1d(1)),
                # ('linear2', nn.Linear(config.nhid//2, config.nhid//4)),
                # ('relu2', nn.ReLU()),
                # ('drop2', nn.Dropout(p=0.2)),
                # ('linear3', nn.Linear(config.nhid//4, 16)),
                # ('relu3', nn.ReLU()),
                # ('drop3', nn.Dropout(p=0.2)),
                ('linear4', nn.Linear(config.nhid//2, 1)),
            ]))
            # nn.init.xavier_uniform_(self.layers.linear1.weight, gain=1.414)
            # nn.init.xavier_uniform_(self.layers.linear2.weight, gain=1.414)
            # nn.init.xavier_uniform_(self.layers.linear3.weight, gain=1.414)
            # nn.init.xavier_uniform_(self.layers.linear4.weight, gain=1.414)
            nn.init.xavier_normal_(self.layers.linear1.weight)
            # nn.init.xavier_normal_(self.layers.linear2.weight)
            # nn.init.xavier_normal_(self.layers.linear3.weight)
            nn.init.xavier_normal_(self.layers.linear4.weight)
        else:
            self.gatgrucell2 = GATGRUCell(config.nfeat, config.nhid, config.nhead, config.alpha, config.dropout)
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

    def forward(self, g):
        """ Each turn do the following things:
            1. update node representation using intra-attention
            2. aggregate node representation with previous argument
        """
        # turns = torch.unique(g.ndata['ids'], sorted=True).tolist()
        # if len(turns) > 10 or len(turns) < 6:
        #     print(f'Argument has {len(turns)} turns')
        # if self.config.mode=='bidirection':
        turns = [0,1,2,3,4,5]
        if self.config.node_encoder == 'first':
            d_enc = self.degree_encoder(g)
            g.ndata['h'] += d_enc
        # for t in turns:
        #     if t % 2 == 0:
        #         _ = self.gatgrucell(g, t)
        #     else:
        #         _ = self.gatgrucell2(g, t)
        # if self.config.mode=='bidirection':
        #     for t in turns[::-1]:
        #         _ = self.gatgrucellback(g, t, len(turns))
        # read-out
        # h1 = self._read_out(g, -2, op='mean') # second-last argument
        # h2 = self._read_out(g, -1, op='mean') # last argument
        # Classification: speaker#1 wins if s1>s2
        if self.config.loss == 'binary':
            for t in turns:
                _ = self.gatgrucell(g, t)
            if self.config.node_encoder != 'first':
                d_enc = self.degree_encoder(g)
                g.ndata['hp'] += d_enc
            # for t in turns:
            #     if t % 2 == 0:
            #         _ = self.gatgrucell(g, t)
            #     else:
            #         _ = self.gatgrucell2(g, t)
            if self.config.mode=='bidirection':
                for t in turns[::-1]:
                    _ = self.gatgrucellback(g, t, len(turns))
            # h2 = (h1+h2)/2
            # x = self._read_out(g, -1, op='mean') # last argument
            x1, _ = self._read_out_loud(g, op='max')
            _, x2 = self._read_out_loud(g, op='min')
            x = torch.cat((x1, x2), dim=-1)
            # x = torch.ones_like(x)
            # print(f'{x=}')
            x = self.layers(x)
            return 0, x.squeeze()
        else:
            for t in turns:
                if t % 2 == 0:
                    _ = self.gatgrucell(g, t)
                else:
                    _ = self.gatgrucell2(g, t)
            if self.config.mode=='bidirection':
                for t in turns[::-1]:
                    _ = self.gatgrucellback(g, t, len(turns))
            h1, h2 = self._read_out_loud(g, op='max')
            s1 = self.score1(h1)
            s2 = self.score2(h2)
            return s1.squeeze(), s2.squeeze()

    def _read_out(self, gg, t, op='mean'):
        gl = dgl.unbatch(gg)
        res = [] # list of tensor
        for g in gl:
            # turns = torch.max(g.ndata['ids']).item()+1
            turns = 6
            node_idx = g.filter_nodes(lambda nodes: nodes.data['ids']==turns+t)
            h = g.nodes[node_idx].data['hp']

            # TODO: back later
            if op == 'mean':
                a = torch.mean(h, dim=-2, keepdim=True)
                # print(a.shape)
            elif op == 'max':
                a = torch.max(h, dim=-2, keepdim=True).values
            elif op == 'min':
                a = torch.min(h, dim=-2, keepdim=True).values
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
            # turns = torch.max(g.ndata['ids']).item()+1
            turns = 6 # if self.config.mode=='bidirection' else torch.max(g.ndata['ids']).item()+1
            readout_s1 = torch.zeros((1, self.nlast), device=self.config.device)
            readout_s2 = torch.zeros((1, self.nlast), device=self.config.device)

            for turn in range(turns):
                node_idx = g.filter_nodes(lambda nodes: nodes.data['ids']==turn)
                # h = g.nodes[node_idx].data['hp']
                if self.config.mode=='bidirection':
                    h = torch.cat((g.nodes[node_idx].data['hp'], g.nodes[node_idx].data['hb']), dim=1)
                else:
                    h = g.nodes[node_idx].data['hp']
                if op == 'mean':
                    a = torch.mean(h, dim=-2, keepdim=True)
                elif op == 'max':
                    a = torch.max(h, dim=-2, keepdim=True).values
                elif op == 'min':
                    a = torch.min(h, dim=-2, keepdim=True).values
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