import torch
import dgl
import pickle
from torch.utils.data import Dataset
from utils.config import config
from itertools import accumulate
import numpy as np
from utils.helpers import top_k_sparsify, threshold_sparsity

# https://www.dgl.ai/blog/2019/01/25/batch.html
# https://discuss.dgl.ai/t/create-dataset-from-dglgraphs-in-memory/904/5

class ArgDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def _add_node(self, G, turn, argument, offset=None):
        """ Add nodes & edges for each turn """
        speaker_id = turn%2 # 0 or 1
        num_nodes = len(argument)
        node_feature = {}
        node_feature['speaker'] = torch.zeros(num_nodes, dtype=torch.int8) if speaker_id == 0 else torch.ones(num_nodes, dtype=torch.int8)
        node_feature['h'] = torch.tensor(argument)
        node_feature['ids'] = torch.ones(num_nodes, dtype=torch.int8)*turn
        G.add_nodes(num=num_nodes, data=node_feature)

    def _add_edges(self, G, adj: np.ndarray, src_offset, dst_offset, turn, edge_type=''):
        # pick top k similarity score
        if config.sparsify=='topk':
            adj = top_k_sparsify(adj, k=3)
        else: # thresholding
            adj = threshold_sparsity(adj, thres=0.7)
        # add egdes
        dst, src = np.nonzero(adj) # dst is turn t, src is turn t-1 (b.c.of transposition)
        num_edges = src.shape[0]
        edge_feature = {}
        if edge_type == 'self': #src_offset == dst_offset:
            edge_feature['turn'] = torch.ones(num_edges, dtype=torch.int8)*turn
        elif edge_type == 'counter':
            edge_feature['turn'] = torch.ones(num_edges, dtype=torch.int8)*turn + torch.tensor(100)
        else: #support
            edge_feature['turn'] = torch.ones(num_edges, dtype=torch.int8)*turn + torch.tensor(200)
        G.add_edges(src+src_offset, dst+dst_offset, data=edge_feature)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Get the idx-th sample
            return graph & corresponding label """
        # data = self.data[0] if config.debug else self.data[idx]
        data = self.data[idx]
        attn_list = data['adj']['intra_adj']
        counter_attn_list = data['adj']['counter_adj']
        support_attn_list = data['adj']['support_adj']
        arg_embed = data['graph']
        label = data['label']
        if label==0: label=-1
        graph = self.create_graph(arg_embed, attn_list, counter_attn_list, support_attn_list)
        return graph, label

    def create_graph(self, arg_embed, attn_list, counter_attn_list, support_attn_list):
        """ Create graph for each conversation 
            We did not implement for speaker relation
            TODO: speaker information flow """

        G = dgl.graph([])

        # offset: list of #sent in each turn/argument
        offset = [len(argument) for argument in arg_embed]
        # calculate prefix sum, will be used as offset for edge construction
        # [1,3,5,9] -> [1,4,9,18]
        offset = list(accumulate(offset))
        for turn, argument in enumerate(arg_embed):
            self._add_node(G, turn, argument)
        assert G.num_nodes() == offset[-1], \
                f"Fail constructing graph, #nodes = {G.num_nodes()}, get {offset[-1]} !"

        # https://github.com/dmlc/dgl/issues/3364
        for i in range(len(attn_list)):
            attn = attn_list[i]
            o = 0 if i==0 else offset[i-1]  # [1,4,9,18] -> [0,1,4,9]
            self._add_edges(G, attn, o, o, i, 'self') # intra-edges
            if i < len(attn_list)-1 and config.is_counter: # len(counter) = len(self) - 1
                counter_attn = counter_attn_list[i]
                self._add_edges(G, counter_attn, o, offset[i], i, 'counter')
            if i < len(attn_list)-2 and config.is_support: # len(support) = len(self) - 2
                support_attn = support_attn_list[i]
                self._add_edges(G, support_attn, o, offset[i+1], i, 'support')
        return G
    

def collate_fn(data):
    """ Padding sequences of various length """
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)