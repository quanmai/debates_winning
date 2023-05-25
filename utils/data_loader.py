import torch
import dgl
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
        # node_feature['speaker'] = torch.zeros(num_nodes, dtype=torch.int8) if speaker_id == 0 else torch.ones(num_nodes, dtype=torch.int8)
        node_feature['h'] = torch.tensor(argument)
        node_feature['ids'] = torch.ones(num_nodes, dtype=torch.int8)*turn
        G.add_nodes(num=num_nodes, data=node_feature)

    def _add_edges(self, G, adj: np.ndarray, src_offset, dst_offset, turn, edge_type):
        """
            G: dgl.graph
            adj: adjacency matrix
            src_offset: source node offset
            dst_offset: destination node offset
            turn:
            edge_type: interaction types

            - for forward, dst is turn t, src is turn t-1 (b.c.of transposition)
            - for backward, dst is turn t-1, src is turn t
        """
        if '_bw' not in edge_type:
            adj = adj.T
        # pick top k similarity score
        if config.sparsify=='topk':
            adj = top_k_sparsify(adj, k=3)
        else: # thresholding
            # we do thresholding here
            # if node is isolated, connect it with highest score node
            adj = threshold_sparsity(adj, thres=0.9)

        dst, src = np.nonzero(adj)
        num_edges = src.shape[0]
        # print(f'{edge_type= }, {num_edges=}')
        edge_feature = {}
        edge_feature['turn'] = torch.ones(num_edges, dtype=torch.float)*turn \
                                + config.EDGE_OFFSET[edge_type]
        # assert(edge_feature['turn'] > 0)

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
        # print(f'Winning side: {label}')

        title = data['title']
        if config.loss != 'binary':
            if label==0: label=-1 
        graph = self.create_graph(arg_embed, attn_list, counter_attn_list, support_attn_list, title)
        return graph, label

    def create_graph(self, arg_embed, attn_list, counter_attn_list, support_attn_list, title = ''):
        """ Create graph for each conversation """

        G = dgl.graph([])

        # offset: list of #sent in each turn/argument
        offset = [len(argument) for argument in arg_embed]
        # print(offset)
        # calculate prefix sum, will be used as offset for edge construction
        # [1,3,5,9] -> [1,4,9,18]
        offset = list(accumulate(offset))
        for turn, argument in enumerate(arg_embed):
            self._add_node(G, turn, argument)
        assert G.num_nodes() == offset[-1], \
                f"Fail constructing graph, #nodes = {G.num_nodes()}, get {offset[-1]} !"
        assert len(arg_embed) == len(attn_list)

        # https://github.com/dmlc/dgl/issues/3364
        for i in range(len(attn_list)):
            attn = attn_list[i] #self attn
            o = 0 if i==0 else offset[i-1]  # offset: [1,4,9,18] -> o: [0,1,4,9]
            self._add_edges(G, attn, o, o, i, 'self') # intra-edges
            if i < len(attn_list)-1 and config.is_counter: # len(counter) = len(self) - 1
                counter_attn = counter_attn_list[i]
                self._add_edges(G, counter_attn, o, offset[i], i, 'counter')
                if config.mode == 'bidirection':
                    self._add_edges(G, counter_attn, offset[i], o, i, 'counter_bw')
            if i < len(attn_list)-2 and config.is_support: # len(support) = len(self) - 2
                support_attn = support_attn_list[i]
                self._add_edges(G, support_attn, o, offset[i+1], i, 'support')
                if config.mode == 'bidirection':
                    self._add_edges(G, support_attn, offset[i+1], o, i, 'support_bw')
        return G
    

def collate_fn(data):
    """ Padding sequences of various length """
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    # print(f'{batched_graph.batch_size=}')
    return batched_graph, torch.tensor(labels)