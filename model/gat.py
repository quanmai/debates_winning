import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from utils.config import config


FEAT_NAME = 'h' # node feature = sentence embeddings
HPRIME = 'h_prime' # h'
HID_FEAT_NAME = 'hp' # same as HPRIME, but just a name convention
FEAT_WH = 'Wh' # W*h 


"""
    How about bi-GRU? 
        - Bi-directional information flow!
        - Forward: conversation flow
        - Backward: inversed flow
    Option 2:
        - update both dst and src nodes in forward path (v. 457)
"""


class GATGRUCell(nn.Module):
    """ 
        The cell is GRUCell-like, as the output of the GATGRU is not 100% used
        for next argument, the GRU(counter, support) (or self.gruinter()) is.
    """
    def __init__(self, nfeats, nhids, nheads, alpha, dropout):
        super(GATGRUCell, self).__init__()
        self.grucell = nn.GRUCell(nhids, nhids) #self.grucell(input, hidden)
        # self.gruinter = nn.GRUCell(nhids, nhids)
        self.gat = GAT(nfeats, nhids, nheads, alpha, dropout)
        self.xgat = CrossGAT(nhids, nhids, nheads, alpha, dropout)
        # self.xgat_s = CrossGAT(nhids, nhids, nheads, alpha, dropout)

    def forward(self, g, t):
        x = self.gat(g, t)
        if t > 0:
            h_c, node_idx, _ = self.xgat(g, t-1, itype='counter')
            h = h_c
            if t > 1:
                h_s, node_idx_sp, _ = self.xgat(g, t-2, itype='support')
                assert torch.all(node_idx.eq(node_idx_sp))
                h = config.counter_coeff*h_c+(1-config.counter_coeff)*h_s
                # h = self.gruinter(h_c, h_s)
            hprime = self.grucell(x, h)
            # hprime = 0.5*x + 0.5*h
            g.nodes[node_idx].data[HID_FEAT_NAME] = hprime
            return hprime
        return x


class CrossGAT(nn.Module):
    def __init__(self, nfeats, nhid, nheads, alpha, dropout):
        super(CrossGAT, self).__init__()
        self.attentions = nn.ModuleList([
            DirectedGATLayer(nfeats, nhid//nheads, alpha=alpha, dropout=dropout) for _ in range(nheads)
            ])
        if config.gat_layers != 1 :
            self.out_att = DirectedGATLayer(nhid, nhid, alpha=alpha, dropout=dropout)

    def forward(self, g, t, itype):
        """
            itype (interaction type): counter/support
        """
        offset = config.EDGE_OFFSET[itype]
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset)
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        h_itype = torch.cat([att(g, t, offset) for att in self.attentions], dim=1)
        if config.gat_layers != 1 :
            h_itype = self.out_att(g, t, offset, h=h_itype)
        return h_itype, dst_nodes, src_nodes


class GAT(nn.Module):
    """ Take, aggregate, plug back """
    def __init__(self, nfeat, nhid, nheads, alpha, dropout, direction='forward', feat_name=HID_FEAT_NAME):
        super(GAT, self).__init__()
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid//nheads, alpha=alpha, dropout=dropout, layer=1) for _ in range(nheads)
            ])
        # self.attentions = [GATLayer(nfeat, nhid//nheads, alpha=alpha, dropout=dropout, direction=direction) for _ in range(nheads)]
        # for i, attn in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attn)
        if config.gat_layers > 1 :
            self.out_att = GATLayer(nhid, nhid, alpha=alpha, dropout=dropout, layer=2)
        self.feat_name = feat_name
        self.direction = direction

    def forward(self, g, t):
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        h = torch.cat([att(g, t) for att in self.attentions], dim=1)
        if config.gat_layers > 1 :
            h = self.out_att(g, t, h=h)
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

    def forward(self, g, t, offset, h=None):
        """ Compute attention score from turn t to turn t-1 
        """
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
        if h is None: # 1 GAT Layer
            h = g.nodes[node_id].data[HID_FEAT_NAME]
        else: # 2 layers
            # h.shape = (#dst_node, nhid) --> concatenate with src_nodes
            h = torch.cat((g.nodes[src_nodes].data[HID_FEAT_NAME], h), dim=0)
        # print(f'{h=}')
        h = F.dropout(h, self.dropout, training=self.training)
        # Wh = torch.mm(h, self.W) # Wh = h x W
        Wh = self.W(h)
        g.nodes[node_id].data[FEAT_WH] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=dst_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[node_id].data.pop(FEAT_WH)

        # if config.sparsify == 'threshold':
        #     # the problem with thresholding sparsification is
        #     # that GAT applys to full nodes while xGAT doesnot -> shape missmatched
        #     # here we do padding
        #     node_idx = g.filter_nodes(lambda nodes: nodes.data['ids'] == t+1) 
        #     node_idx = node_idx.unique()
        #     node_offset = torch.min(node_idx, dim=-1).values()
        #     num_nodes = node_idx.shape[0]
        #     print(f'{dst_nodes=}')  # nodes exist due to thresholding
        #     print(f'{node_idx=}') # whole nodes
        #     h_prime = torch.zeros(num_nodes, self.out_features)
        #     for node in dst_nodes:
        #         h_prime[node-node_offset] = g.nodes[node].data[HPRIME]
        #     return h_prime
        
        h_prime = g.nodes[dst_nodes].data.pop(HPRIME) # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox[FEAT_WH], dim=1)
        return {HPRIME: h_prime}

    def _message_func(self, edges):
        turn = edges.data['turn']
        # print(f'{turn=}')
        # print(f'{turn.shape=}')
        assert torch.all(turn >= 0)
        return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        """ We only compute attention score for 1 direction """
        # Wh1 = torch.matmul(edges.src[FEAT_WH], self.a[:self.out_features, :])
        # Wh2 = torch.matmul(edges.dst[FEAT_WH], self.a[self.out_features:, :])
        # e = Wh1 + Wh2
        # return {'e': self.leakyrelu(e)}
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        return {'e': self.leakyrelu(e)} # assign to each edge



class GATLayer(nn.Module):
    """ Attention layer for intra-argument nodes """
    def __init__(self, in_features, out_features, alpha, dropout, layer=1, direction='forward'):
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
        if layer==1:
            self.h_feat_name = FEAT_NAME if direction=='forward' else HID_FEAT_NAME
        else:
            self.h_feat_name = HID_FEAT_NAME


    def forward(self, g, t, h=None):
        # h is nodes feature, h0 = x
        # ALWAYS `specify` NODE_IDS
        node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        edge_id = g.filter_edges(lambda edges: edges.data['turn']==t) # intra_argument
        if h is None:
            h = g.nodes[node_id].data[self.h_feat_name]
        # print(h)
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = self.W(h)
        # print(f'{Wh.shape=}')

        g.nodes[node_id].data[FEAT_WH] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_id, message_func=self._message_func, reduce_func=self._reduce_func)
        g.ndata.pop(FEAT_WH) # remove FEAT_WH
        h_prime = g.nodes[node_id].data.pop(HPRIME) # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        mail_e = nodes.mailbox['e']
        # print(f'{mail_e.shape=}')
        # print(mail_e)
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox[FEAT_WH], dim=1)
        # wh = nodes.mailbox[FEAT_WH]
        # print(f'{wh.shape=}')
        # print(wh)
        # breakpoint()
        return {HPRIME: h_prime}

    def _message_func(self, edges):
        # TODO: check! inter-edges are DISPLAYED here!!!
        # DONE: yeah, but inter-edges Wh and other values are 0
        turn = edges.data['turn']
        # print(f'{turn=}')
        # print(f'{turn.shape=}')
        assert torch.all(turn >= 0)
        return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        # e_1 = Wh * a[:self.out_features, :]
        # e_2 = Wh * a[self.out_features:, :]
        # e = e_1 + e_2
        # Wh1 = torch.matmul(edges.src[FEAT_WH], self.a[:self.out_features, :])
        # Wh2 = torch.matmul(edges.dst[FEAT_WH], self.a[self.out_features:, :])
        # e = Wh1 + Wh2.T
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
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
    """ Should do pooling layer before the readout
        IDEAS: should we choose only nodes that have high attention score?
    """
    def __init__(self,):
        pass

    def forward(self, g, t):
        # TODO: complete later 
        # node_id = g.filter_nodes(lambda nodes: nodes.data['ids']==t)
        # feat = g.nodes[node_id].data[HID_FEAT_NAME]
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
        self.xgat = CrossGATInversed(nhids, nhids, nheads, alpha, dropout, direction)
        self.feat_name = feat_name

    def forward(self, g, t, num_turns):
        """
            From time t-1 back to 0
        """
        #TODO: check here, there seems a shift in edge offset, as we're supposed to put t, not t-1: DONE
        x = self.gat(g, t)
        if t < num_turns-1:
            h_c, node_idx = self.xgat(g, t, 'counter_bw')
            h = h_c
            if t < num_turns - 2:
                h_s, node_idx_sp = self.xgat(g, t, 'support_bw')
                assert torch.all(node_idx.eq(node_idx_sp))
                h = 0.5*h_c + 0.5*h_s
            hprime = self.grucell(x, h)
            g.nodes[node_idx].data[self.feat_name] = hprime
            return hprime
        return x


class CrossGATInversed(nn.Module):
    def __init__(self, nfeats, nhid, nheads, alpha, dropout, direction='backward'):
        super(CrossGATInversed, self).__init__()
        self.feat_name = 'hb' if direction == 'backward' else HID_FEAT_NAME
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
        updated_nodes = dst_nodes
        # only update dst node

        # g.nodes[updated_nodes].data[itype_feat_name] = h_itype
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
        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a = nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        self.dropout = dropout
        self.out_features = out_features
        self.direction = direction
        self.feat_name = feat_name

    def forward(self, g, t, offset):
        """ Compute attention score from turn t to turn t-1 """
        # TODO: Can we use dgl.subgraph() instead? Rep: Later
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset) # cross_argument
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        # TODO: check if needed to swap src and dst in node_id
        node_id = torch.cat((src_nodes, dst_nodes), dim=0)
        h = g.nodes[node_id].data[self.feat_name]
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = self.W(h)
        g.nodes[node_id].data[FEAT_WH] = Wh
        updated_nodes = dst_nodes
        # updated_nodes = src_nodes if self.direction=='backward' else dst_nodes
        
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=updated_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[node_id].data.pop(FEAT_WH)
        h_prime = g.nodes[updated_nodes].data.pop(HPRIME) # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox[FEAT_WH], dim=1)
        return {HPRIME: h_prime}

    def _message_func(self, edges):
        turn = edges.data['turn']
        assert torch.all(turn >= 0)
        return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        """ We only compute attention score for 1 direction """
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        return {'e': self.leakyrelu(e)} # assign to each edge



## Final GAT Layers?