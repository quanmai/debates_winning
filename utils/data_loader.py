import torch
import dgl
import pickle
from torch.utils.data import Dataset
from utils.config import config
from itertools import accumulate
import numpy as np
from utils.helpers import top_k_sparsify

# https://www.dgl.ai/blog/2019/01/25/batch.html
# https://discuss.dgl.ai/t/create-dataset-from-dglgraphs-in-memory/904/5

class ArgDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def _load(self, filename):
        pass

    def _add_node(self, G, turn, argument, offset=None):
        """ Add nodes & edges for each turn
            Edges are added to corresponding pair of node 
            
            G: dgl.graph
            argument: list[sent_emb]
            offset: 
        """
        speaker_id = turn%2 # 0 or 1
        num_nodes = len(argument)
        # print(f'argument shape: {}')
        # print(f"Hey, adding {num_nodes} node")
        node_feature = {}
        node_feature['speaker'] = torch.zeros(num_nodes, dtype=torch.int8) if speaker_id == 0 else torch.ones(num_nodes, dtype=torch.int8)
        node_feature['h'] = torch.tensor(argument)
        node_feature['ids'] = torch.ones(num_nodes, dtype=torch.int8)*turn
        G.add_nodes(num=num_nodes, data=node_feature)
        # print(f"Current # nodes: {G.num_nodes()}")
        # print(G.ndata['h'])

    def _add_edges(self, G, adj: np.ndarray, src_offset, dst_offset, turn):
        # pick top k similarity score
        adj = top_k_sparsify(adj, k=3)
        # add egdes
        src, dst = np.nonzero(adj)
        # print(f'src: {src}, dst: {dst}')
        # print(adj)
        # if src_offset==dst_offset: 
        #     print(f'adj: {adj}')
        #     print(f'src node: {src}')
        #     print(f'dst node: {dst}')
        num_edges = src.shape[0]
        # print(f'adding {num_edges} edges')
        # if src_offset == dst_offset: #intra-argument edges
        #     edge_type = torch.zeros(num_edges, dtype=torch.int8)
        # else: #cross-arguments edges
        #     edge_type = torch.ones(num_edges, dtype=torch.int8)
        edge_feature = {}
        # edge_feature['etype'] = edge_type
        if src_offset == dst_offset:
            edge_feature['turn'] = torch.ones(num_edges, dtype=torch.int8)*turn
        else:
            edge_feature['turn'] = torch.ones(num_edges, dtype=torch.int8)*turn + torch.tensor(100)

        G.add_edges(src+src_offset, dst+dst_offset, data=edge_feature)
        # breakpoint()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Get the idx-th sample
            return graph & corresponding label """
        data = self.data[0] if config.debug else self.data[idx]
        attn_list = data['adj']['intra_adj']
        cross_attn_list = data['adj']['inter_adj']
        arg_embed = data['graph']
        label = data['label']
        graph = self.create_graph(arg_embed, attn_list, cross_attn_list)
        return graph, label

    def create_graph(self, arg_embed, attn_list, cross_attn_list):
        """ Create graph for each conversation 
            We did not implement for speaker relation
            TODO: speaker information flow """

        G = dgl.graph([])

        # offset: list of #sent in each turn/argument
        offset = [len(argument) for argument in arg_embed]
        print(offset)
        # calculate prefix sum
        # will be used as offset for edge construction
        # [1,3,5,9] -> [1,4,9,18]
        offset = list(accumulate(offset))
        # print(offset)
        for turn, argument in enumerate(arg_embed):
            self._add_node(G, turn, argument)
        assert G.num_nodes() == offset[-1], \
                f"Fail constructing graph, #nodes = {G.num_nodes()}, get {offset[-1]} !"

        # Edge adding by g.add_edges([scr_nodes], [dst_nodes], {edge_feature})
        # Or we can use ADJ matrices!!
        # e.g., with first turn, #sents = 10, offset = 0
        # source = list(range(offset, offset+#sents))
        # g = dgl.add_edges(g, source, source, {'feature': intr_adj} )
        # https://github.com/dmlc/dgl/issues/3364

        for i in range(len(attn_list)):
            attn = attn_list[i]
            o = 0 if i==0 else offset[i-1]  # [1,4,9,18] -> [0,1,4,9]
            self._add_edges(G, attn, o, o, i) # intra-edges
            if i < len(attn_list)-1: # len(cross) = len(self) - 1
                cross_attn = cross_attn_list[i]
                # print(f'cross_attn: {cross_attn}')
                self._add_edges(G, cross_attn, o, offset[i], i)
        # print(f'num_edges: {G.num_edges()}')
        return G
    
def collate_fn(data):
    """ Padding sequences of various length """
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)