import os
import sys
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='dataset/')
parser.add_argument('--nhid', type=int, default=64)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--alpha', type=float, default=0.02, help='leaky relu slope' )
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout prob')
parser.add_argument('--not-counter', action='store_true', default=False)
parser.add_argument('--not-support', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--gendata', action='store_true', default=False)
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'adamax'], default='adam', help='Optimizer: sgd, adamw, adamax, adam')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--scheduler', type=str, choices=['', 'exp', 'cyclic'], default='', help='use scheduler')
parser.add_argument('--batch-size', type=int, default=16, help='batch size cuda can support')
args = parser.parse_args()
class Config:
    def __init__(self, args) -> None:
        self.__dict__.update(vars(args))
        self.train_f, self.dev_f, self.test_f = (os.path.join(self.data_dir, o) for o in ['train.json','dev.json','test.json'])
        self.proce_f = os.path.join(self.data_dir, 'dataset_preproc.p')
        self.filtered_f = os.path.join(self.data_dir, 'data_all_argument.json')
        self.gen_f = os.path.join(self.data_dir, 'data_gen.json')
        self.original_f = 'ddo/debates.json'
        self.bert = 'bert-base-uncased'
        self.max_length = 128
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
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
            # "we've": "we have", 
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
        self.is_counter = not self.not_counter
        self.is_suport = not self.not_support
        self.nfeat = 768 # Bert last hidden layer
        self.num_workers = 64
        self.shuffle = True # shuffle Training dataset
        if self.debug:
            self.batch_size = 1 # set batch-size to 1
            self.dropout = 0 # no dropout
            self.shuffle = False # shuffle Training dataset
            self.num_workers = 1

config = Config(args)