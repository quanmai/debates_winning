import torch
import torch.nn as nn
import dgl
from model.gat import GATGRUCell


class DebateGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gatgru_cell = GATGRUCell(config.nfeat,
                                    config.nhid,
                                    nheads=config.nhead, 
                                    alpha= config.alpha, 
                                    dropout=config.dropout,
                                    res=False)
        self.config = config
        self.turns = 6

    def forward(self, batch, emb):
        g = batch['graph']
        nmask = batch['nmask']
        nodeidx = batch['nodeidx']
        B, T, N, _ = batch['utter'].shape
        self.set_node_feature(g, nmask, nodeidx, (B, T, N), emb)
        self.propagation(g)
        self.pooling(g)
        x1, x2 = self.read_out_loud(g)
        return x1, x2

    def propagation(self, g):
        for t in range(6):
            self.gatgru_cell(g, t)

    def set_node_feature(self, graph, nmask, nodeidx, ushape, emb):
        """
        Set node feature after Utterance encoding...
        """
        # -> view(B, TxN, -1) 
        B, T, N = ushape
        emb = (emb * nmask.unsqueeze(-1)
               ).reshape(B*T*N, -1)
        feat = emb[nodeidx]
        graph.ndata['h'] = feat

    def read_out_loud(self, graph, op1='mean', op2='mean', feat='hp', attn_softmax=True):
        def _helper(h, op):
            if op == 'mean':
                a = torch.mean(h, dim=-2, keepdim=True)
            elif op == 'sum':
                a = torch.sum(h, dim=-2, keepdim=True)
            elif op == 'max':
                a = torch.max(h, dim=-2, keepdim=True).values
            elif op == 'min':
                a = torch.min(h, dim=-2, keepdim=True).values
            else:
                raise AssertionError("Unexpected value of 'op'!", op)
            return a

        gl = dgl.unbatch(graph)
        res1 = [] # list of tensor
        res2 = [] # list of tensor
        for g in gl:
            # turns = torch.max(g.ndata['ids']).item()+1
            readout_s1 = torch.zeros((1, self.config.nhid), device=self.config.device)
            readout_s2 = torch.zeros((1, self.config.nhid), device=self.config.device)

            for turn in range(self.turns):
                node_idx = g.filter_nodes(lambda nodes: nodes.data['tid']==turn)
                h = g.nodes[node_idx].data[feat]
                attn = g.nodes[node_idx].data['attn_score']
                attn = attn.squeeze()
                if attn_softmax:
                    attn = torch.nn.functional.softmax(attn, dim=0)
                h = h*attn.unsqueeze(-1)
                if turn % 2 == 0:
                    readout_s1 += _helper(h, op1)
                else:
                    readout_s2 += _helper(h, op2)

            res1.append(readout_s1)
            res2.append(readout_s2)
        r1 = torch.cat(res1, dim=0).unsqueeze(1)
        r2 = torch.cat(res2, dim=0).unsqueeze(1)
        return r1, r2

    def pooling(self, g, attn='e'):
        def _reducer(nodes):
            return {'attn_score': nodes.mailbox[attn].sum(1)}
        g.update_all(dgl.function.copy_e(attn, attn), reduce_func=_reducer)