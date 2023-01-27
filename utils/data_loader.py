'''
    Data loader

    How many graph?
    - In the end of the day, graph is just nodes and their connectivity
    - So, in each turn :
        + in argument: nodes represent sentences, nodes attributes are sentence embeddings using BERT
        + 


    To feed into the model, data chunks have to have SAME SIZE! Guess why? 
    - Need PADDING technique

'''



import torch
import dgl
import pickle

import torch.utils.data as data
from config import config


class Dataset(data.Dataset):
    def __init__(self, data):
        # self.filename = filename
        self.data = data

    def _load_file(self):
        pass

    def _add_node(self, G, num_nodes):
        G.add_nodes(num_nodes)


    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        """ Take conversation as input
            and return Graph & Label """

        graph = self.create_graph()


        return graph, label


        
    
    def create_graph(self, ):
        """ Create graph for each conversation
            - Node types:
                1. Speaker nodes
                2. Turn nodes
        """

        G = dgl.graph([])
        G.set_n_initializer(dgl.inti.zero_initializer)
        # G.add_node
        # dgl.add_nodes(g, num, data=None, ntype=None)
        # Each sentence is a node, add node here 
        # data['graph'] is a list of arguments (list(list(sents)))
        for turn, argument in enumerate(data['graph']):

            # Need to get node IDs
            num_nodes = len(argument)
            data_dict = {}
            data_dict['speaker'] = torch.zeros(num_nodes) if turn%2==0 else torch.ones(num_nodes)
            data_dict['h'] = torch.LongTensor(argument)
            data_dict['turn'] = torch.ByteTensor(turn)
            G = dgl.add_nodes(g=G, num=num_nodes, data=data_dict)


            for i in range(len(argument)):
                for j in range(i+1, len(argument)):
                    G = dgl.add_edges()

                
    
def collate_fn(data):
    """ Padding sequences of various length"""
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    
    return batched_graph, torch.tensor(labels)

