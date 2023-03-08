import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# class GRUCell(nn.Module):
#     # https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
#     def __init__(self, nfeat, nhid, bias=True):
#         super(GRUCell, self).__init__()
#         self.nfeat = nfeat
#         self.nhid = nhid
#         self.bias = bias
#         self.x2h = nn.Linear(nfeat, 3*nhid, bias=bias)
#         self.h2h = nn.Linear(nhid, 3*nhid, bias=bias)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         std = 1.0 / math.sqrt(self.out_features)
#         for w in self.parameters():
#             w.data.uniform_(-std, std)
    
#     def forward(self, x, hidden):
#         """ hidden: cross-attention features: hidden
#             x: intra-attention features: input

#             z = sigmoid(W_z . [h,x])
#             r = sigmoid(W_r . [h,x])
#             h' = tanh(W . [r*h,x])
#             _h = (1-z)*h + z*h'
#         """

#         gate_x = self.x2h(x).squeeze()
#         gate_h = self.h2h(hidden).squeeze()

#         i_r, i_u, i_n = gate_x.chunk(3, 1)
#         h_r, h_u, h_n = gate_h.chunk(3, 1)
        
#         update_gate = F.sigmoid(i_u + h_u) # z
#         reset_gate = F.sigmoid(i_r + h_r)  # r
#         new_gate = F.tanh((reset_gate*h_n)+i_n)

#         h = new_gate + input_gate * (hidden-new_gate)

#         return h


class CrossGAT(nn.Module):
    def __init__(self, nhid, nheads, alpha, dropout):
        super(CrossGAT, self).__init__()
        # TODO: check dimensions
        self.gru = nn.GRUCell(nhid, nhid)
        self.attentions = [DirectedGATLayer(nhid, nhid//nheads, alpha=alpha, dropout=dropout) for _ in range(nheads)]

    def forward(self, g, t):
        # print('Doing CrossGAT')
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+100)
        _, dst_nodes = g.find_edges(edge_id)
        dst_nodes = dst_nodes.unique()
        # print(f'dst_nodes: {dst_nodes}')

        h = torch.cat([att(g, t) for att in self.attentions], dim=1)
        # only update dst node
        feat = g.nodes[dst_nodes].data['hp']
        # print(f'h shape CGAT: {h.shape}')
        # print(f'feat shape {feat.shape}')
        g.nodes[dst_nodes].data['hp'] = self.gru(feat, h)

        return g.nodes[dst_nodes].data['hp']

class GAT(nn.Module):
    """ Take, aggregate, plug back """
    def __init__(self, nfeat, nhid, nheads, alpha, dropout):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(nfeat, nhid//nheads, alpha=alpha, dropout=dropout) for _ in range(nheads)]
        for i, attn in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attn)

    def forward(self, g, t):
        # print('Doing GAT')
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        h = torch.cat([att(g, t) for att in self.attentions], dim=1)
        # g.nodes[node_id].data.pop('h')
        g.nodes[node_id].data['hp'] = h
        return h
        

class DirectedGATLayer(nn.Module):
    """ Attention layer for cross-argument nodes.
        This is directed graph attention, which means dst-nodes pay attention to src-nodes.
        We can define this as a bipartite graph, one side has N_(t-1) nodes, the other has N_(t)
        We do not shrink the dimension here: out_features = in_features.

        TODO (try later): Attention mechanism using Key, Query & Value matrices
    """
    def __init__(self, in_features, out_features, alpha, dropout):
        super(DirectedGATLayer, self).__init__()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.W = nn.Parameter(torch.empty((in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty((2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.out_features = out_features

    def forward(self, g, t):
        """ Compute attention score from turn t to turn t-1 """
        # TODO: Can we use dgl.subgraph() instead? Rep: Later

        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+100) # cross_argument
        # g.find_edges(eid): Given an edge ID array, return the source and destination node ID array s and d. 
        # Only update representation of destination node
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        node_id = torch.cat((src_nodes, dst_nodes), dim=0)
        h = g.nodes[node_id].data['hp']
        h = F.dropout(h, self.dropout, training=self.training)

        Wh = torch.mm(h, self.W) # Wh = h x W
        g.nodes[node_id].data['Wh'] = Wh

        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=dst_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[dst_nodes].data.pop('Wh')
        h_prime = g.nodes[dst_nodes].data.pop('h_prime') # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox['Wh'], dim=1)
        return {'h_prime': h_prime}

    def _message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        """ We only compute attention score for 1 direction """
        # e_1 = Wh * a[:self.out_features, :]
        # e_2 = Wh * a[self.out_features:, :]
        # e = e_1 + e_2
        Wh1 = torch.matmul(edges.src['Wh'], self.a[:self.out_features, :])
        Wh2 = torch.matmul(edges.dst['Wh'], self.a[self.out_features:, :])
        e = Wh1 + Wh2
        return {'e': self.leakyrelu(e)}


class GATLayer(nn.Module):
    """ Attention layer for intra-argument nodes """
    def __init__(self, in_features, out_features, alpha, dropout):
        # https://github.com/Diego999/pyGAT/blob/master/layers.py
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.W = nn.Parameter(torch.empty((in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty((2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = dropout

    def forward(self, g, t):
        # h is nodes feature, h0 = x
        # ALWAYS `specify` NODE_IDS
        # print(f'at time step: {t}')
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        edge_id = g.filter_edges(lambda edges: edges.data['turn']==t) # intra_argument
        # edge_id = g.filter_edges(lambda edges: )
        # print(f'node_id: {node_id}')
        # print(f'edge_id: {edge_id}')
        # print(f'edge_turn: {g.edges[edge_id].data["turn"]}')
        # print(g.nodes[node_id].data)
        h = g.nodes[node_id].data['h']
        # print(f'W shape: {self.W.shape}')
        # print(f'h shape: {h.shape}')
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = torch.mm(h, self.W) # Wh = h x W

        g.nodes[node_id].data['Wh'] = Wh
        # print(f'WH shape: {Wh.shape}')
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_id, message_func=self._message_func, reduce_func=self._reduce_func)
        g.ndata.pop('Wh') # remove 'Wh'
        h_prime = g.ndata.pop('h_prime') # get h'
        # g.nodes[node_id].data['h'] = h_prime # will assign in multi-head
        return h_prime[node_id]

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        # print(nodes.mailbox['e'].shape)
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        # print(f'attention shape: {attention.shape}')
        h_prime = torch.sum(attention * nodes.mailbox['Wh'], dim=1)
        return {'h_prime': h_prime}

    def _message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        # e_1 = Wh * a[:self.out_features, :]
        # e_2 = Wh * a[self.out_features:, :]
        # e = e_1 + e_2
        Wh1 = torch.matmul(edges.src['Wh'], self.a[:self.out_features, :])
        Wh2 = torch.matmul(edges.dst['Wh'], self.a[self.out_features:, :])
        e = Wh1 + Wh2
        return {'e': self.leakyrelu(e)}
    
    def _edge_filter(self, edges):
        return edges.data['turn'] == t