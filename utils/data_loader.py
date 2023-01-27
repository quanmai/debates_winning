import torch
import dgl
import pickle

import torch.utils.data as data
from config import config
from itertools import accumulate


class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def _load_file(self):
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
        node_feature = {}
        node_feature['speaker'] = torch.zeros(num_nodes) if speaker_id == 0 else torch.ones(num_nodes)
        node_feature['h'] = torch.LongTensor(argument)
        G = dgl.add_nodes(g=G, num=num_nodes, data=node_feature)

    def _add_edges(self, G, argument, offset, self_attn, cross_attn):
        num_nodes = len(argument)
        for i in range(offset, offset+num_nodes):
            for j in range(offset, offset+num_nodes):
                G.add_edges(i, j, data={'h': torch.LongTensor(intra_adj[i][j]), 'type': torch.CharTensor([0])}) # type = 0: intra-edge


    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        """ Take conversation as input
            and return Graph & Label """

        graph = self.create_graph()
        return graph, label

    def create_graph(self, ):
        """ Create graph for each conversation
            
            data['graph'] is a list of arguments (list(list(sents)))  

        """

        G = dgl.graph([])
        G.set_n_initializer(dgl.inti.zero_initializer)

        # offset: list of #sent in each turn/argument
        offset = [len(argument) for argument in data['graph']]
        # calculate prefix sum
        # will be used as offset for edge construction
        # [1,3,5,9] -> [1,4,9,18]
        offset = list(accumulate(offset))

        for turn, argument in enumerate(data['graph']):
            self._add_node(G, turn, argument)
            assert G.num_nodes() == offset[-1], "Fail constructing graph!"

        # self._add_edges(G)
        for i, o in enumerate(offset):
            G.add_edges()

    
def collate_fn(data):
    """ Padding sequences of various length"""
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    
    return batched_graph, torch.tensor(labels)

