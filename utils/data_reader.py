import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import itertools
import re
import random
import json
import math
from sklearn.metrics.pairwise import cosine_similarity
from utils.sentence_embedding import sentence_embedding
import pickle
from utils.config import config
from utils.helpers import top_k_sparsify
import os
# QUO = '<quote>'
# CORENLP_PATH = os.path.join('/Users/quanmai/Software/stanford-corenlp-4.5.1','*')
# nlp = spacy.load("en_core_web_sm")    

def _process_text(text: str) -> str:
    url = r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{1,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    # text = re.sub(r'^[^a-zA-Z]+','', text) # Remove non-alphabetic char before alphabetic chars
    text = re.sub(url, ' [URL] ', text)
    text = re.sub(r'-?\d+(\.\d+)?(?!\w)', ' [NUM] ', text)
    text = re.sub(r'[-+]?.?[0-9]+[a-zA-Z]*', ' [NUM] ', text) #like 19th
    text = re.sub(r"&gt;.*\n\n", ' [QUOTE] ', text)
    text = re.sub(r'\r\n', '', text)
    # text = re.sub(r'\r\n', '', text)
    text = text.lower()

    for k, v in config.word_pairs.items():
        text = text.replace(k,v)

    return text

def _get_label(debate):
    temp_pro = 0
    temp_con = 0
    Pro = debate['participant_1_name']
    Con = debate['participant_2_name']
    for user in debate['votes']:
        for name, attitude in user['votes_map'].items():
            if 'Made more convincing arguments' in attitude:
                if name == Pro and attitude['Made more convincing arguments'] == True:
                    temp_pro = temp_pro + 1
                elif name == Con and attitude['Made more convincing arguments'] == True:
                    temp_con = temp_con + 1
    if temp_pro > temp_con:
        return 1
    elif temp_pro < temp_con:
        return 0
    else:
        return -1

def _preprocess(title_list, min_len: int = 3, is_to_sparse=False):
    """ Sentence embedding using BERT
        Edges are defined by cosine similarity among sentence"""
    print(title_list)

    with open(config.filtered_f, 'r') as f:
        debates = json.load(f)
    f.close()

    D = []
    # debate = debates['rounds']
    for key, debate in debates.items():
        if key in title_list:
            low_qual = False # flag for low-quality argument
            print('Processing debates: {}'.format(key))
            d = {}
            
            title = debate['title']
            # title_embeddings = sentence_embedding([title])
            d['title'] = _process_text(title)
            # d['title_emb'] = title_embeddings

            arguments_embed_list = []
            arguments_len_list = [] # to track argument length of each turn
            for round in debate['rounds']:
                for side in round:
                    text = side['text']
                    text = _process_text(text)
                    sents = nltk.tokenize.sent_tokenize(text)
                    sents = [s for s in sents if len([w for w in s.split(' ') if w.isalpha()]) > min_len]
                    # print(sents)
                    if not sents: # low quality arguments
                        low_qual = True
                        break
                    sents_embeddings = sentence_embedding(sents)
                    arguments_embed_list.append(sents_embeddings)
                    arguments_len_list.append(len(sents))
            if low_qual:
                continue
            # print(len(arguments_embed_list))
                
            # node attribute
            # arguments_embed_list = list(itertools.chain(*arguments_embed_list)) # list of lists into a list
            d['graph'] = arguments_embed_list 
            d['arg_len'] = arguments_len_list
            
            intra_sim_list, counter_sim_list, support_sim_list = [], [], []
            for i in range(len(arguments_embed_list)):
                intra_sim = cosine_similarity(arguments_embed_list[i]) # ndarray
                intra_sim_list.append(intra_sim) # list of ndarrays :D
                if i != len(arguments_embed_list)-1:
                    counter_sim = cosine_similarity(arguments_embed_list[i], arguments_embed_list[i+1])
                    counter_sim_list.append(counter_sim)
                if i < len(arguments_embed_list)-2:
                    support_sim = cosine_similarity(arguments_embed_list[i], arguments_embed_list[i+2])
                    support_sim_list.append(support_sim)

            d['adj'] = {'intra_adj': intra_sim_list,
                        'counter_adj': counter_sim_list, 
                        'support_adj': support_sim_list}    
            d['label'] = _get_label(debate)
            d['turns'] = len(arguments_embed_list) // 2
            D.append(d)

    return D

def generate_data(titles, seed=42):
    debates = _preprocess(titles)
    random.Random(seed).shuffle(debates)
    l = len(debates)
    idx_train = int(0.6*l)
    idx_dev = int(0.8*l)

    train, dev, test = debates[:idx_train], debates[idx_train:idx_dev], debates[idx_dev:]

    print('Dumping to files')
    with open(config.proce_full, 'wb') as f:
        pickle.dump([train,dev,test], f)

    print('Done')

def generate_data_test(titles, seed=4):
    debates = _preprocess(titles[:3])
    random.Random(seed).shuffle(debates)
    
    l = len(debates)
    idx_train = int(0.6*l)
    idx_dev = int(0.8*l)

    train, dev, test = debates[0], debates[1], debates[2]

    print('Dumping to files')
    with open(config.proce_f, 'wb') as f:
        pickle.dump([train,dev,test], f)

    print('Done')
    
def load_dataset():
    if os.path.exists(config.proce_f):
        print('Loading dataset...')
        with open(config.proce_f, 'rb') as f:
            train, dev, test = pickle.load(f)
        f.close()
        return train, dev, test
    
    # generating file
    # TODO: calling genrate_data()
    generate_data()


if __name__ == '__main__':
    # load_data()
    with open('title.txt','r') as rf:
        titles = rf.readlines()
    
    # titles = titles[:100]
    titles = [t.replace('\n','') for t in titles]
    
    generate_data(titles)