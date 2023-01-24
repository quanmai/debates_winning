import os
import sys
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/')

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

config = Config(args)