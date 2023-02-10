import torch
import dgl
import pickle
import torch.utils.data as data
from config import config
from itertools import accumulate

# https://www.dgl.ai/blog/2019/01/25/batch.html

class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def _add_node(self, G, turn, argument, offset=None):
        """ Add nodes & edges for each turn
            Edges are added to corresponding pair of node 
            
            G: dgl.graph
            argument: list[sent_emb]
            offset: 
        """
        speaker_id = turn%2 # 0 or 1
        num_nodes = len(argument)
        node_feature = {}
        node_feature['speaker'] = torch.zeros(num_nodes) if speaker_id == 0 else torch.ones(num_nodes)
        node_feature['h'] = torch.LongTensor(argument)
        node_feature['ids'] = torch.ones(num_nodes)*turn
        G = dgl.add_nodes(g=G, num=num_nodes, data=node_feature)

    def _add_edges(self, G, adj, src_offset, dst_offset, turn):
        #TODO: add turn info: DONE!, but need to double check!
        src, dst = np.nonzero(adj)
        num_edges = src.shape[0]
        if src_offset == dst_offset: #intra-argument edges
            edge_type = torch.zeros([num_edges], dtype=torch.int8)
        else: #cross-arguments edges
            edge_type = torch.ones([num_edges], dtype=torch.int8)
        edge_feature = {}
        edge_feature['etype'] = edge_type
        edge_feature['turn'] = torch.ones([num_edges], dtype=torch.int8)*turn
        G.add_edges(src+src_offset, dst+dst_offset, data=edge_feature)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Get the idx-th sample
            return graph & corresponding label """
        data = self.data[idx]
        attn_list = data['adj']['intra_adj']
        cross_attn_list = data['adj']['inter_adj']
        arg_embed = data['graph']
        label = data['label']
        graph = self.create_graph(arg_embed, attn_list, cross_attn_list)
        return graph, label

    def create_graph(self, arg_embed, attn_list, cross_attn_list ):
        """ Create graph for each conversation 
            We did not implement for speaker relation
            TODO: speaker information flow """

        G = dgl.graph([])

        # offset: list of #sent in each turn/argument
        offset = [len(argument) for argument in arg_embed]
        # calculate prefix sum
        # will be used as offset for edge construction
        # [1,3,5,9] -> [1,4,9,18]
        offset = list(accumulate(offset))

        for turn, argument in enumerate(arg_embed):
            self._add_node(G, turn, argument)
            assert G.num_nodes() == offset[-1], "Fail constructing graph!"

        # Edge adding by g.add_edges([scr_nodes], [dst_nodes], {edge_feature})
        # Or we can use ADJ matrices!!
        # e.g., with first turn, #sents = 10, offset = 0
        # source = list(range(offset, offset+#sents))
        # g = dgl.add_edges(g, source, source, {'feature': intr_adj} )
        # https://github.com/dmlc/dgl/issues/3364

        for i in range(len(attn_list)):
            attn = attn_list[i]
            o = 0 if i==0 else offset[i-1]  # [1,4,9,18] -> [0,1,4,9]
            self._add_edges(G, attn, o, o) # intra-edges
            if i < len(attn_list)-1: # len(cross) = len(self) - 1
                cross_attn = cross_attn_list[i]
                #TODO: need to know what turn?
                self._add_edges(G, cross_attn, o, offsset[i], i)
        return G
    
def collate_fn(data):
    """ Padding sequences of various length """
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

