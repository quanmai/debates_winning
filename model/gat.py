import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from utils.config import config


"""
    How about bi-GRU? 
        - Bi-directional information flow!
        - Forward: conversation flow
        - Backward: inversed flow
    Option 2:
        - update both dst and src nodes in forward path
"""


class GATGRUCell(nn.Module):
    """ 
        The cell is GRUCell-like, as the output of the GATGRU is not 100% used
        for next argument, the GRU(counter, support) (or self.gruinter()) is.
    """
    def __init__(self, nfeats, nhids, nheads, alpha, dropout):
        super(GATGRUCell, self).__init__()
        self.grucell = nn.GRUCell(nhids, nhids) #self.grucell(input, hidden)
        self.gruinter = nn.GRUCell(nhids, nhids)
        self.gat = GAT(nfeats, nhids, nheads, alpha, dropout)
        self.xgat = CrossGG(nhids, nhids, nheads, alpha, dropout)

    def forward(self, g, t):
        x = self.gat(g, t)
        if t > 0:
            h_c, node_idx, src_nodes = self.xgat(g, t-1, 'counter')
            h = h_c
            if t > 1:
                h_s, node_idx_sp, _ = self.xgat(g, t-2, 'support')
                assert torch.all(node_idx.eq(node_idx_sp))
                h = 0.5*h_c+0.5*h_s
                # h = h_c + self.gruinter(h_c, h_s)
            # print(f'{x.shape=}')
            # print(f'{h.shape=}')
            # hprime = self.grucell(x, h)
            hprime = 0.5*x + 0.5*h
            g.nodes[node_idx].data['hp'] = hprime
            return hprime
        return x


class CrossGG(nn.Module):
    def __init__(self, nfeats, nhid, nheads, alpha, dropout):
        super(CrossGG, self).__init__()
        self.attentions = nn.ModuleList([DirectedGATLayer(nfeats, nhid//nheads, alpha=alpha, dropout=dropout) for _ in range(nheads)])

    def forward(self, g, t, itype):
        """
            itype (interaction type): counter/support
        """
        offset = config.EDGE_OFFSET[itype]
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset)
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()

        h_itype = torch.cat([att(g, t, offset) for att in self.attentions], dim=1)
        # only update dst node
        # g.nodes[dst_nodes].data[itype] = h_itype
        return h_itype, dst_nodes, src_nodes


class GAT(nn.Module):
    """ Take, aggregate, plug back """
    def __init__(self, nfeat, nhid, nheads, alpha, dropout, direction='forward', feat_name='hp'):
        super(GAT, self).__init__()
        self.attentions = [GATLayer(nfeat, nhid//nheads, alpha=alpha, dropout=dropout, direction=direction) for _ in range(nheads)]
        for i, attn in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attn)
        self.feat_name = feat_name
        self.direction = direction

    def forward(self, g, t):
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        h = torch.cat([att(g, t) for att in self.attentions], dim=1)
        g.nodes[node_id].data[self.feat_name] = h
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
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(alpha)
        # self.W = nn.Parameter(torch.empty((in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.empty((2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        self.dropout = dropout
        self.out_features = out_features

    def forward(self, g, t, offset):
        """ Compute attention score from turn t to turn t-1 """
        # TODO: Can we use dgl.subgraph() instead? Rep: Later
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset) # cross_argument
        # g.find_edges(eid): Given an edge ID array, return the source and destination node ID array s and d. 
        # Only update representation of destination node
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()

        # print(f'{g.batch_num_nodes()=}')
        # print(f'{g.batch_num_edges()=}')
        # print(f'{src_nodes=}')
        # print(f'{dst_nodes=}')
        # assert torch.max(src_nodes) < torch.min(dst_nodes)
        node_id = torch.cat((src_nodes, dst_nodes), dim=0)
        # print(f'{node_id.shape=}')
        h = g.nodes[node_id].data['hp']
        # print(f'{h=}')
        h = F.dropout(h, self.dropout, training=self.training)
        # Wh = torch.mm(h, self.W) # Wh = h x W
        Wh = self.W(h)
        g.nodes[node_id].data['Wh'] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=dst_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[node_id].data.pop('Wh')

        if config.sparsify == 'threshold':
            node_idx = g.filter_nodes(lambda nodes: nodes.data['ids'] == t+1)
            node_idx = node_idx.unique()
            num_nodes = node_idx.shape.item()
            h_prime = torch.zeros(num_nodes, self.out_features)
            for node in dst_nodes:
                # does this assign zero to non-zero column?
                h_prime[node] = None
        else:
            h_prime = g.nodes[dst_nodes].data.pop('h_prime') # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox['Wh'], dim=1)
        return {'h_prime': h_prime}

    def _message_func(self, edges):
        turn = edges.data['turn']
        # print(f'{turn=}')
        # print(f'{turn.shape=}')
        assert torch.all(turn >= 0)
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        """ We only compute attention score for 1 direction """
        # Wh1 = torch.matmul(edges.src['Wh'], self.a[:self.out_features, :])
        # Wh2 = torch.matmul(edges.dst['Wh'], self.a[self.out_features:, :])
        # e = Wh1 + Wh2
        # return {'e': self.leakyrelu(e)}
        Wh = torch.cat([edges.src['Wh'], edges.dst['Wh']], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        return {'e': self.leakyrelu(e)} # assign to each edge



class GATLayer(nn.Module):
    """ Attention layer for intra-argument nodes """
    def __init__(self, in_features, out_features, alpha, dropout, direction='forward'):
        # https://github.com/Diego999/pyGAT/blob/master/layers.py
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

        self.dropout = dropout
        self.h_feat_name = 'h' if direction=='forward' else 'hp'

    def forward(self, g, t):
        # h is nodes feature, h0 = x
        # ALWAYS `specify` NODE_IDS
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        # print(f'{node_id.shape=}')
        edge_id = g.filter_edges(lambda edges: edges.data['turn']==t) # intra_argument
        # print(f'{edge_id.shape=}')
        h = g.nodes[node_id].data[self.h_feat_name]
        # print(h)
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = self.W(h)
        # print(f'{Wh.shape=}')

        g.nodes[node_id].data['Wh'] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_id, message_func=self._message_func, reduce_func=self._reduce_func)
        g.ndata.pop('Wh') # remove 'Wh'
        h_prime = g.nodes[node_id].data.pop('h_prime') # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        mail_e = nodes.mailbox['e']
        # print(f'{mail_e.shape=}')
        # print(mail_e)
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox['Wh'], dim=1)
        # wh = nodes.mailbox['Wh']
        # print(f'{wh.shape=}')
        # print(wh)
        # breakpoint()
        return {'h_prime': h_prime}

    def _message_func(self, edges):
        # TODO: check! inter-edges are DISPLAYED here!!!
        # DONE: yeah, but inter-edges Wh and other values are 0
        turn = edges.data['turn']
        # print(f'{turn=}')
        # print(f'{turn.shape=}')
        assert torch.all(turn >= 0)
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        # e_1 = Wh * a[:self.out_features, :]
        # e_2 = Wh * a[self.out_features:, :]
        # e = e_1 + e_2
        # Wh1 = torch.matmul(edges.src['Wh'], self.a[:self.out_features, :])
        # Wh2 = torch.matmul(edges.dst['Wh'], self.a[self.out_features:, :])
        # e = Wh1 + Wh2.T
        Wh = torch.cat([edges.src['Wh'], edges.dst['Wh']], dim=1) # [#edges, 2*out_features]
        # print(f'{Wh.shape=}')
        e = self.a(Wh)  # [#edges, 1]
        # print(f'{e.shape=}')
        # print(e)
        # sids = edges.src['ids']
        # dids = edges.dst['ids']
        # print(f'{sids.shape=}')
        # print(f'{sids=}')
        # print(f'{dids.shape=}')
        # print(f'{dids=}')
        return {'e': self.leakyrelu(e)} # assign to each edge

class Pooling(nn.Module):
    """ Should do pooling layer before the readout"""
    def __init__(self,):
        pass

    def forward(self, g, t):
        # TODO: complete later 
        # node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        # feat = g.nodes[node_id].data['hp']
        pass


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


class GATGRUCellInversed(nn.Module):
    """ 
        TODO: update description
    """
    def __init__(self, nfeats, nhids, nheads, alpha, dropout, direction='backward', feat_name='hb'):
        super(GATGRUCellInversed, self).__init__()
        self.grucell = nn.GRUCell(nhids, nhids)
        self.gruinter = nn.GRUCell(nhids, nhids)
        self.gat = GAT(nfeats, nhids, nheads, alpha, dropout, direction, feat_name)
        self.xgat = CrossGGInversed(nhids, nhids, nheads, alpha, dropout, direction)
        self.feat_name = feat_name

    def forward(self, g, t, num_turns):
        """
            From time t-1 back to 0
        """
        #TODO: check here, there seems a shift in edge offset, as we're supposed to put t, not t-1
        x = self.gat(g, t)
        if t < num_turns-1:
            h_c, node_idx = self.xgat(g, t, 'counter_bw')
            h = h_c
            if t < num_turns - 2:
                h_s, node_idx_sp = self.xgat(g, t, 'support_bw')
                assert torch.all(node_idx.eq(node_idx_sp))
                h = 0.5*h_c + 0.5*h_s
                # h = h_c + self.gruinter(h_c, h_s)
            hprime = self.grucell(x, h)
            g.nodes[node_idx].data[self.feat_name] = hprime
            return hprime
        return x


class CrossGGInversed(nn.Module):
    def __init__(self, nfeats, nhid, nheads, alpha, dropout, direction='backward'):
        super(CrossGGInversed, self).__init__()
        self.feat_name = 'hb' if direction == 'backward' else 'hp'
        self.attentions = nn.ModuleList([DirectedGATLayerInversed(nfeats, 
                                                                nhid//nheads, 
                                                                alpha=alpha, 
                                                                dropout=dropout, 
                                                                direction=direction,
                                                                feat_name=self.feat_name) for _ in range(nheads)])
        self.direction = direction

    def forward(self, g, t, itype):
        """
            itype (interaction type): counter/support backward
        """
        offset = config.EDGE_OFFSET[itype]
        itype_feat_name = itype + '_' +self.feat_name
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset)
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        h_itype = torch.cat([att(g, t, offset) for att in self.attentions], dim=1)
        # updated_nodes = src_nodes if self.direction=='backward' else dst_nodes
        updated_nodes = dst_nodes
        # only update dst node

        g.nodes[updated_nodes].data[itype_feat_name] = h_itype
        return h_itype, updated_nodes


class DirectedGATLayerInversed(nn.Module):
    """ Attention layer for cross-argument nodes.
        This is directed graph attention, which means dst-nodes pay attention to src-nodes.
        We can define this as a bipartite graph, one side has N_(t-1) nodes, the other has N_(t)
        We do not shrink the dimension here: out_features = in_features.

        TODO (try later): Attention mechanism using Key, Query & Value matrices
    """
    def __init__(self, in_features, out_features, alpha, dropout, direction='backward', feat_name='hb'):
        super(DirectedGATLayerInversed, self).__init__()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.W = nn.Parameter(torch.empty((in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty((2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.out_features = out_features
        self.direction = direction
        self.feat_name = feat_name

    def forward(self, g, t, offset):
        """ Compute attention score from turn t to turn t-1 """
        # TODO: Can we use dgl.subgraph() instead? Rep: Later
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset) # cross_argument
        # g.find_edges(eid): Given an edge ID array, return the source and destination node ID array s and d. 
        # Only update representation of destination node
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        # TODO: check if needed to swap src and dst in node_id
        node_id = torch.cat((src_nodes, dst_nodes), dim=0)
        h = g.nodes[node_id].data[self.feat_name]
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = torch.mm(h, self.W) # Wh = h x W
        g.nodes[node_id].data['Wh'] = Wh

        updated_nodes = dst_nodes
        # updated_nodes = src_nodes if self.direction=='backward' else dst_nodes
        
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=updated_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[updated_nodes].data.pop('Wh')
        h_prime = g.nodes[updated_nodes].data.pop('h_prime') # get h'
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
        Wh1 = torch.matmul(edges.src['Wh'], self.a[:self.out_features, :])
        Wh2 = torch.matmul(edges.dst['Wh'], self.a[self.out_features:, :])
        e = Wh1 + Wh2
        return {'e': self.leakyrelu(e)}