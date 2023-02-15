import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, nfeat, nhid, nheads, alpha, dropout):
        super(CrossGAT, self).__init__()
        # TODO: check dimensions
        self.gru = nn.GRUCell(nhid, nhid)
        self.attentions = [DirectedGATLayer(nfeat, alpha=alpha, dropout=dropout) for _ in nheads]

    def forward(self, g, t):
        edge_id = g.filter_edges(lambda edges: edges.data['etype'] == 1, \
                                        lambda edges: edges.data['turn']==t)
        _, dst_nodes = g.find_edges(edge_id)

        h = torch.cat([att(g, t) for att in self.attentions], dim=1)
        # only update dst node
        feat = g.nodes[dst_nodes].data['h']
        g.nodes[dst_nodes].data['h'] = self.gru(feat, h)

        return g.nodes[dst_nodes].data['h']

class GAT(nn.Module):
    """ Take, aggregate, plug back """
    def __init__(self, nfeat, nhid, nheads, alpha, dropout):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(nfeat, nhid, alpha=alpha, dropout=dropout) for _ in nheads]
        for i, attn in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attn)

    def forward(self, g, t):
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        h = torch.cat([att(g, t) for att in self.attentions], dim=1)
        g.nodes[node_id].data['h'] = h
        return h
        

class DirectedGATLayer(nn.Module):
    """ Attention layer for cross-argument nodes.
        This is directed graph attention, which means dst-nodes pay attention to src-nodes.
        We can define this as a bipartite graph, one side has N_(t-1) nodes, the other has N_(t)
        We do not shrink the dimension here: out_features = in_features.

        TODO (try later): Attention mechanism using Key, Query & Value matrices
    """
    def __init__(self, in_features, alpha, dropout):
        super(DirectedGATLayer, self).__init__()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.W = nn.Parameter(torch.empty((in_features, in_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty((2*in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout

    def forward(self, g, t):
        """ Compute attention score from turn t to turn t-1 """
        # TODO: Can we use dgl.subgraph() instead? Rep: Later

        edge_id = g.filter_edges(lambda edges: edges.data['etype'] == 1, \
                                lambda edges: edges.data['turn']==t) # cross_argument
        # g.find_edges(eid): Given an edge ID array, return the source and destination node ID array s and d. 
        # Only update representation of destination node
        src_nodes, dst_nodes = g.find_edges(edge_id)
        h_src = g.nodes[src_nodes].data['h']
        h_dst = g.nodes[dst_nodes].data['h']
        h = torch.cat((h_src, h_dst), dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        assert h.shape[1] == src_nodes.shape[1]+dst_nodes.shape[1]

        Wh = torch.mm(h, self.W) # Wh = h x W
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=dst_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[dst_nodes].data.pop('Wh')
        h_prime = g.nodes[dst_nodes].data.pop('h_prime') # get h'
        # g.nodes[dst_nodes].data['h'] = h_prime # will assign in multi-head
        return h_prime[dst_nodes]

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
        e = Wh1 + Wh2.T
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
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        edge_id = g.filter_edges(lambda edges: edges.data['etype'] == 0) # intra_argument
        
        h = g.nodes[node_id].data['h']
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = torch.mm(h, self.W) # Wh = h x W

        g.nodes[node_id].data['Wh'] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_id, message_func=self._message_func, reduce_func=self._reduce_func)
        g.ndata.pop('Wh') # remove 'Wh'
        h_prime = g.ndata.pop('h_prime') # get h'
        # g.nodes[node_id].data['h'] = h_prime # will assign in multi-head
        return h_prime[node_id]

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        attention = F.softmax(nodes.mailbox['e'], dim=1)
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
        e = Wh1 + Wh2.T
        return {'e': self.leakyrelu(e)}