import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from utils.config import config


FEAT_NAME = 'h' # node feature = sentence embeddings
HPRIME = 'h_prime' # h'
HID_FEAT_NAME = 'hp' # same as HPRIME, but just a name convention
HID_FEAT_NAME_BACK = 'hb' # same as HPRIME, but just a name convention
FEAT_WH = 'Wh' # W*h 


"""
    3 source of information coming to the node:
        + information among intra-turn nodes (h_i) = GAT(vi, vl l \in L): L is set of neighbor nodes of i in same turn
        + information from supporting nodes (h_s) = DirectGAT(vi, vj j \in J): J is set of nodes vi supporting
        + information from attacking nodes  (h_c) = DirectGAT(vi, vk k \in K): K is set of nodes vi attacking
        + information aggregating:
            Debater has to think 
            -> h'_i = GRU(h_i, h_s+h_c)
"""


class GATGRUCell(nn.Module):
    """ 
    The cell is GRUCell-like, as the output of the GATGRU is not 100% used
    for next argument, the GRU(counter, support) (or self.gruinter()) is.
    """
    def __init__(self, nfeats, nhids, nheads, alpha, dropout, feat_name=HID_FEAT_NAME,res=False, update='both'):
        super(GATGRUCell, self).__init__()
        if config.v1 or config.v3:
            self.grucell = nn.GRUCell(nhids, nhids) #self.grucell(input, hidden)
        if config.v2 or config.v3:
            self.gruinter = nn.GRUCell(nhids, nhids)
        self.gat = GAT(nfeats, nhids, nheads, alpha, dropout)
        self.cgat = CrossGAT(nhids, nhids, nheads, alpha, dropout, feat_name, res, update)
        self.updateFeature = HID_FEAT_NAME
        self.update = update

    def forward(self, g, t):
        # x: feature update among intra-turn nodes
        x = self.gat(g, t)
        if t > 0:
            node_idx = g.filter_nodes(lambda nodes: nodes.data['tid']==t)
            h = torch.zeros_like(x)
            h, h_src, src_nodes = self.counter(g, t, node_idx, h)
            # if t > 1:
            #     h_s, node_idx_sp, _ = self.sgat(g, t-2, itype='support')
            #     assert torch.all(node_idx.eq(node_idx_sp))
            #     if config.v1: #v1
            #         h = config.counter_coeff*h_c+(1-config.counter_coeff)*h_s
            #     if config.v2 or config.v3: #v2, v3
            #         h = self.gruinter(h_c, h_s)
            if config.v1 or config.v3: #v1
                hprime = self.grucell(x, h)
            if config.v2: #v2
                hprime = 0.5*x + 0.5*h
            g.nodes[node_idx].data[self.updateFeature] = hprime
            if self.update == 'both':
                g.nodes[src_nodes].data[self.updateFeature] -= h_src
            return hprime
        return x
    
    def counter(self, g, t, node_idx, h):
        """ Feature for dest nodes 
            h_src will be used to update src nodes (depends on self.update settings)
        """
        if self.update=='both':
            h_c, h_src, dst_nodes, src_nodes = self.cgat(g, t-1, itype='counter')
        else:
            h_c, dst_nodes, src_nodes = self.cgat(g, t-1, itype='counter')
        _map = {x.item(): i for i, x in enumerate(node_idx)}
        node_c_idx_map = torch.tensor([_map[i.item()] for i in dst_nodes])
        h[node_c_idx_map] = h_c
        h_src = h_src if self.update =='both' else None
        src_nodes = src_nodes if self.update =='both' else None
        return h, h_src, src_nodes

    def support(self, g, t, node_idx, h):
        h_s, node_c_idx, _ = self.sgat(g, t-2, itype='counter')
        _map = {x.item(): i for i, x in enumerate(node_idx)}
        node_c_idx_map = torch.tensor([_map[i.item()] for i in node_c_idx])
        h[node_c_idx_map] = h_s
        return h


class CrossGAT(nn.Module):
    def __init__(self, nfeats, nhid, nheads, alpha, dropout, feat_name, res, update):
        super(CrossGAT, self).__init__()
        self.attentions = nn.ModuleList([
            DirectedGATLayer(nfeats, 
                             nhid//nheads, 
                             alpha=alpha, 
                             dropout=dropout, 
                             feat_name=feat_name, 
                             res=res,
                             update=update) for _ in range(nheads)
            ])
        self.update = update
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
        if self.update == 'both':
            num_src_nodes = src_nodes.shape[0]
            return h_itype[num_src_nodes:, :], h_itype[:num_src_nodes, :], dst_nodes, src_nodes
        return h_itype, dst_nodes, src_nodes


class GAT(nn.Module):
    """ Take, aggregate, plug back """
    def __init__(self, nfeat, nhid, nheads, alpha, dropout, direction='forward', feat_name=HID_FEAT_NAME):
        super(GAT, self).__init__()
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid//nheads, alpha=alpha, dropout=dropout, layer=1, direction=direction) for _ in range(nheads)
            ])
        if config.gat_layers > 1 :
            self.out_att = GATLayer(nhid, nhid, alpha=alpha, dropout=dropout, layer=2)
        self.feat_name = HID_FEAT_NAME if direction=='forward' else HID_FEAT_NAME_BACK
        self.direction = direction

    def forward(self, g, t):
        node_id = g.filter_nodes(lambda nodes: nodes.data['tid']==t)
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
    def __init__(self, in_features, out_features, alpha, dropout, feat_name = HID_FEAT_NAME, res=False, update='both'):
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
        if res: #residual
            self.res_fc = nn.Linear(in_features, out_features)
        self.res = res
        self.feat_name = feat_name
        self.update = update

    def forward(self, g, t, offset, h=None):
        """ Compute attention score from turn t to turn t-1 
        """
        # TODO: Can we use dgl.subgraph() instead? Rep: Later
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset) # cross_argument
        # g.find_edges(eid): Given an edge ID array, return the source and destination node ID array s and d. 
        # Only update representation of destination node
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        node_id = torch.cat((src_nodes, dst_nodes), dim=0)
        node_updated = node_id if self.update == 'both' else dst_nodes
        if h is None: # 1 GAT Layer
            h = g.nodes[node_id].data[self.feat_name]
        else: # 2 layers
            # h.shape = (#dst_node, nhid) --> concatenate with src_nodes
            h = torch.cat((g.nodes[src_nodes].data[self.feat_name], h), dim=0)
        # print(f'{h=}')
        h = F.dropout(h, self.dropout, training=self.training)
        # Wh = torch.mm(h, self.W) # Wh = h x W
        Wh = self.W(h)
        g.nodes[node_id].data[FEAT_WH] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_updated, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[node_id].data.pop(FEAT_WH)        
        h_prime = g.nodes[node_updated].data.pop(HPRIME) # get h'
        if self.res:
            h_dst = g.nodes[node_updated].data[self.feat_name]
            h_dst = F.dropout(h_dst, self.dropout, training=self.training)
            res = self.res_fc(h_dst)
            h_prime += res
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
        # return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e'], 'ew': edges.data['ew']}

    def _edge_attn(self, edges):
        """ We only compute attention score for 1 direction """
        # Wh1 = torch.matmul(edges.src[FEAT_WH], self.a[:self.out_features, :])
        # Wh2 = torch.matmul(edges.dst[FEAT_WH], self.a[self.out_features:, :])
        # e = Wh1 + Wh2
        # return {'e': self.leakyrelu(e)}
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        e_attn = self.leakyrelu(e)
        return {'e': e_attn} # assign to each edge
        # return {'e': e_attn, 'ew': edges.data['w']*e_attn} # assign to each edge



class GATLayer(nn.Module):
    """ Attention layer for intra-argument nodes """
    def __init__(self, in_features, out_features, alpha, dropout, res=False, layer=1, direction='forward'):
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
        if res: #residual
            self.res_fc = nn.Linear(in_features, out_features)
        self.res = res

    def forward(self, g, t, h=None):
        # ALWAYS `specify` NODE_IDS
        node_id = g.filter_nodes(lambda nodes: nodes.data['tid']==t)
        edge_id = g.filter_edges(lambda edges: edges.data['turn']==t) # intra_argument
        if h is None:
            h = g.nodes[node_id].data[self.h_feat_name]
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = self.W(h)

        g.nodes[node_id].data[FEAT_WH] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_id, message_func=self._message_func, reduce_func=self._reduce_func)
        g.ndata.pop(FEAT_WH) # remove FEAT_WH
        h_prime = g.nodes[node_id].data.pop(HPRIME) # get h'
        if self.res:
            res = self.res_fc(h)
            h_prime += res
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        # mail_e = nodes.mailbox['e']
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox[FEAT_WH], dim=1)
        return {HPRIME: h_prime}

    def _message_func(self, edges):
        # TODO: check! inter-edges are DISPLAYED here!!!
        # DONE: yeah, but inter-edges Wh and other values are 0
        turn = edges.data['turn']
        assert torch.all(turn >= 0)
        return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e']}
        # return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e'], 'ew': edges.data['ew']}

    def _edge_attn(self, edges):
        # e_1 = Wh * a[:self.out_features, :]
        # e_2 = Wh * a[self.out_features:, :]
        # e = e_1 + e_2
        # Wh1 = torch.matmul(edges.src[FEAT_WH], self.a[:self.out_features, :])
        # Wh2 = torch.matmul(edges.dst[FEAT_WH], self.a[self.out_features:, :])
        # e = Wh1 + Wh2.T
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        e_attn = self.leakyrelu(e)
        return {'e': e_attn} # assign to each edge
        # return {'e': e_attn, 'ew': edges.data['w']*e_attn} # assign to each edge


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
        # TODO: Can we use dgl.subgraph() instead? Rep: No need
        edge_id = g.filter_edges(lambda edges: edges.data['turn'] == t+offset) # cross_argument
        src_nodes, dst_nodes = g.find_edges(edge_id)
        src_nodes, dst_nodes = src_nodes.unique(), dst_nodes.unique()
        # TODO: check if needed to swap src and dst in node_id -> DONE, update dst_nodes for both cases
        node_id = torch.cat((src_nodes, dst_nodes), dim=0)
        h = g.nodes[node_id].data[self.feat_name]
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = self.W(h)
        g.nodes[node_id].data[FEAT_WH] = Wh
        
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=dst_nodes, message_func=self._message_func, reduce_func=self._reduce_func)
        g.nodes[node_id].data.pop(FEAT_WH)
        h_prime = g.nodes[dst_nodes].data.pop(HPRIME) # get h'
        return h_prime

    def _reduce_func(self, nodes):
        # mailbox: return the received messages
        attention = F.softmax(nodes.mailbox['eb'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox[FEAT_WH], dim=1)
        return {HPRIME: h_prime}

    def _message_func(self, edges):
        turn = edges.data['turn']
        assert torch.all(turn >= 0)
        return {FEAT_WH: edges.src[FEAT_WH], 'eb': edges.data['eb']}

    def _edge_attn(self, edges):
        """ We only compute attention score for 1 direction """
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        return {'eb': self.leakyrelu(e)} # assign to each edge



## Final GAT Layers?
class GATConv(nn.Module):
    """ Take, aggregate, plug back """
    def __init__(self, nfeat, nhid, nheads, alpha, dropout, feat_name=HID_FEAT_NAME):
        super(GATConv, self).__init__()
        self.attentions = nn.ModuleList([
            FinalGATLayer(nfeat, nhid//nheads, alpha=alpha, dropout=dropout, layer=1, feat_name='h') for _ in range(nheads)
            ])
        self.feat_name = feat_name

    def forward(self, g):
        node_id = g.nodes()
        h = torch.cat([att(g) for att in self.attentions], dim=1)
        g.nodes[node_id].data[self.feat_name] = h
        return h
    

class FinalGATLayer(nn.Module):
    """ Attention layer for intra-argument nodes """
    def __init__(self, in_features, out_features, alpha, dropout, layer=1, direction='forward', feat_name='hp'):
        # https://github.com/Diego999/pyGAT/blob/master/layers.py
        super(FinalGATLayer, self).__init__()
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
        self.feat_name = feat_name

    def forward(self, g):
        # h is nodes feature, h0 = x
        # ALWAYS `specify` NODE_IDS
        node_id = g.nodes()
        edge_id = g.edges()
        h = g.nodes[node_id].data[self.feat_name]
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = self.W(h)

        g.nodes[node_id].data[FEAT_WH] = Wh
        g.apply_edges(self._edge_attn, edges=edge_id)
        g.pull(v=node_id, message_func=self._message_func, reduce_func=self._reduce_func)
        g.ndata.pop(FEAT_WH) # remove FEAT_WH
        h_prime = g.nodes[node_id].data.pop(HPRIME) # get h'
        # print(f'{h_prime.shape}')
        return h_prime

    def _reduce_func(self, nodes):
        attention = F.softmax(nodes.mailbox['e'], dim=1)
        h_prime = torch.sum(attention * nodes.mailbox[FEAT_WH], dim=1)
        return {HPRIME: h_prime}

    def _message_func(self, edges):
        turn = edges.data['turn']
        assert torch.all(turn >= 0)
        return {FEAT_WH: edges.src[FEAT_WH], 'e': edges.data['e']}

    def _edge_attn(self, edges):
        Wh = torch.cat([edges.src[FEAT_WH], edges.dst[FEAT_WH]], dim=1) # [#edges, 2*out_features]
        e = self.a(Wh)  # [#edges, 1]
        return {'e': self.leakyrelu(e)} # assign to each edge