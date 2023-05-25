import os
import sys
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='dataset/')
parser.add_argument('--nhid', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.02, help='leaky relu slope' )
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout prob')
parser.add_argument('--run100', action='store_true', default=False, help='run 100 samples, for debug')
parser.add_argument('--is-counter', action='store_true', default=False)
parser.add_argument('--is-support', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--gendata', action='store_true', default=False)
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'adamax'], default='adam', help='Optimizer: sgd, adamw, adamax, adam')
parser.add_argument('--loss', type=str, choices=['pair', 'binary', 'ranking'], default='pair', help='Loss types')
parser.add_argument('--mode', type=str, choices=['bidirection', 'unidirection'], default='unidirection', help='Mode types')
parser.add_argument('--scheduler', type=str, choices=['', 'exp', 'cyclic'], default='', help='use scheduler')
parser.add_argument('--lr_decay', type=float, default=0.98, help='scheduler decay')
parser.add_argument('--accelerator', type=str, choices=['gpu','cpu'], default='cpu')
parser.add_argument('--sparsify', type=str, choices=['topk', 'threshold'], default='topk', help='Adjacency Sparsification')
parser.add_argument('--node-encoder', type=str, default='', help='Node encoder type')
parser.add_argument('--node-encoder-direction', type=str, choices=['in', 'out', 'both'], default='both', help='Degree encoder directioin')
parser.add_argument('--k', type=int, default=3, help='Top-k Sparsification')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch-size', type=int, default=100, help='batch size cuda can support')
parser.add_argument('--gat-layers', type=int, choices=[1, 2], default=1, help='Number GAT Layers')
parser.add_argument('--test-ver', type=int, default=0, help='Test Model Version')
parser.add_argument('--counter-coeff', type=float, default=0.5, help='Counter argument coefficient')

args = parser.parse_args()
class Config:
    def __init__(self, args) -> None:
        self.__dict__.update(vars(args))
        self.train_f, self.dev_f, self.test_f = (os.path.join(self.data_dir, o) for o in ['train.json','dev.json','test.json'])
        # self.proce_f = os.path.join(self.data_dir, 'dataset_preproc.p')
        if self.run100:
            self.proce_f = os.path.join(self.data_dir, 'dataset_preproc_100.p')
        else:
            # self.proce_f = os.path.join(self.data_dir, 'dataset_preproc_full.p')
            self.proce_f = os.path.join(self.data_dir, 'dataset_preproc_6_turns.p')
        self.filtered_f = os.path.join(self.data_dir, 'data_all_argument.json')
        self.gen_f = os.path.join(self.data_dir, 'data_gen.json')
        self.original_f = 'ddo/debates.json'
        self.bert = 'bert-base-uncased'
        self.max_length = 128
        self.device = 'cuda' if torch.cuda.is_available() and self.accelerator == 'gpu' else 'cpu'
        self.word_pairs = {
            "it's": "it is", 
            "don't": "do not", 
            "doesn't": "does not", 
            "didn't": "did not", 
            "you'd": "you would",
            "you're": "you are", 
            "'ll": " will",
            "i'm": "i am", 
            "they're": "they are", 
            "that's": "that is", 
            "what's": "what is", 
            "couldn't": "could not", 
            "'ve": " have", 
            "can't": "cannot", 
            "i'd": "i would",  
            "aren't" :"are not", 
            "isn't" :"is not", 
            "wasn't": "was not", 
            "weren't": "were not", 
            "won't": "will not", 
            "there's": "there is", 
            "there're": "there are"
            }
        self.nfeat = 384 #768 - Bert last hidden layer
        self.num_workers = 64
        self.shuffle = True # shuffle Training dataset
        self.precision = 32
        if self.debug:
            self.batch_size = 1 # set batch-size to 1
            self.dropout = 0 # no dropout
            self.shuffle = False # shuffle Training dataset
            self.num_workers = 1

        self.EDGE_OFFSET = {
            'self': 0,
            'counter': 1000,
            'counter_bw': 100000,
            'support': 10000,
            'support_bw': 200000
        }
config = Config(args)