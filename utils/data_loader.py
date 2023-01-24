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

    def add_node(self, G, num_nodes):
        G.add_nodes(num_nodes)

    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        """ Return Conversation and Label """



        
    
    def create_graph(self, ):
        """ Create graph for each conversation
            - Node types:
                1. Speaker nodes
                2. Turn nodes
        """
        
        G = dgl.graph([])
        G.set_n_initializer(dgl.inti.zero_initializer)
        # G.add_node

        # Each sentence is a node, add node here 
        
                
    
def collate_fn(data):
    """ Padding sequences of various length"""
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    
    return batched_graph, torch.tensor(labels)

