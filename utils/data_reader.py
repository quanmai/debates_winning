NUM_TURNS = 6
import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")
# nlp = English()  # just the language with no pipeline
# nlp.add_pipe("sentencizer")
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
import os
from sentence_transformers import SentenceTransformer
import sklearn


nlp = spacy.load("en_core_web_sm")
from spacy.language import Language
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "..." or token.text == "\n":
            doc[token.i + 1].is_sent_start = True
    return doc

nlp.add_pipe("set_custom_boundaries", before="parser")


model = SentenceTransformer('all-MiniLM-L6-v2')
# QUO = '<quote>'
# CORENLP_PATH = os.path.join('/Users/quanmai/Software/stanford-corenlp-4.5.1','*')
# nlp = spacy.load("en_core_web_sm")    

URL = ' url ' #' [URL] '
NUM = ' number ' #' [NUM] '
QUO = ' quote ' #' [QUOTE] '

def _process_text(text: str) -> str:
    url = r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{1,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    # text = re.sub(r'^[^a-zA-Z]+','', text) # Remove non-alphabetic char before alphabetic chars
    text = re.sub(url, URL , text)
    text = re.sub(r'-?\d+(\.\d+)?(?!\w)', NUM, text)
    text = re.sub(r'[-+]?.?[0-9]+[a-zA-Z]*', NUM, text) #like 19th
    text = re.sub(r"&gt;.*\n\n", QUO, text)
    text = re.sub(r'\r\n', '', text)
    text = re.sub(r'%','',  text)
    text = text.lower()

    for k, v in config.word_pairs.items():
        text = text.replace(k,v)

    return text

def _get_label(debate):
    """ PRO:1, CON: 0"""
    temp_pro = 0
    temp_con = 0
    if debate['participant_1_position'] == 'Pro' and debate['participant_2_position'] == 'Con':
        Pro = debate['participant_1_name']
        Con = debate['participant_2_name']
    elif debate['participant_1_position'] == 'Con' and debate['participant_2_position'] == 'Pro':
        Pro = debate['participant_2_name']
        Con = debate['participant_1_name']
    else:
        raise Exception('Cannot determine Pro and Con')
    
    for user in debate['votes']:
        for name, attitude in user['votes_map'].items():
            if 'Made more convincing arguments' in attitude:
                if name == Pro and attitude['Made more convincing arguments'] == True:
                    temp_pro += 1
                elif name == Con and attitude['Made more convincing arguments'] == True:
                    temp_con += 1
    if temp_pro > temp_con:
        return 1
    elif temp_pro < temp_con:
        return 0
    else:
        raise Exception('Cannot determine winner')

def _preprocess(title_list, min_len: int = 3):
    """ Sentence embedding using Sentence-BERT
        Edges are defined by cosine similarity among sentence"""
    print(title_list)
    pros, cons = 0, 0
    with open(config.filtered_f, 'r') as f:
        debates = json.load(f)
    f.close()

    D = []
    # debate = debates['rounds']
    for key, debate in debates.items():
        if key in title_list:
            okay_arg = True # flag for low-quality argument
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
                    doc = nlp(text)
                    sents = [s.text for s in doc.sents]
                    # sents = nltk.tokenize.sent_tokenize(text)
                    sents = [s for s in sents if len([w for w in s.split(' ') if w.isalpha()]) >= min_len]
                    # print(sents)
                    if sents and len(sents) >= 3: # high quality arguments
                        # sents_embeddings = sentence_embedding(sents)
                        sents_embeddings = model.encode(sents)
                        # print(sents_embeddings)
                        sents_embeddings_t = sents_embeddings.T
                        scaler = sklearn.preprocessing.StandardScaler().fit(sents_embeddings_t)
                        sents_embeddings_scaled = scaler.transform(sents_embeddings_t)
                        arguments_embed_list.append(sents_embeddings_scaled.T)
                        arguments_len_list.append(len(sents))
                    else:
                        okay_arg = False
            if not okay_arg:
                continue
            # arguments_embed_list = list(itertools.chain(*arguments_embed_list)) # list of lists into a list
            arguments_embed_list = arguments_embed_list[-NUM_TURNS:]
            d['graph'] = arguments_embed_list 
            # d['arg_len'] = arguments_len_list
            assert len(arguments_embed_list) == NUM_TURNS, f'Number of turns is {len(arguments_embed_list)}'
            # print(len(arguments_len_list))
            
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
            label = _get_label(debate)
            if label == 1:
                pros += 1
            else: 
                cons += 1
            d['label'] = label
            print(f'Winning side: {label}')
            d['turns'] = len(arguments_embed_list) // 2
            D.append(d)
    print(f'Number of debates Pros win: {pros}')
    print(f'Number of debates Cons win: {cons}')
    return D

def generate_data(titles, file_name=config.proce_f, seed=42):
    debates = _preprocess(titles)
    random.Random(seed).shuffle(debates)
    num = len(debates)
    print(f'Number of qualified debates are: {num}')
    idx_train = int(0.6*num)
    idx_dev = int(0.8*num)

    train, dev, test = debates[:idx_train], debates[idx_train:idx_dev], debates[idx_dev:]
    # for debug
    # train, dev, test = debates[0], debates[1], debates[2]

    print(f'Dumping to files {file_name}')
    with open(file_name, 'wb') as f:
        pickle.dump([train,dev,test], f)

    print('Done')
    
def load_dataset():
    if os.path.exists(config.proce_f):
        print('Loading dataset...')
        with open(config.proce_f, 'rb') as f:
            train, dev, test = pickle.load(f)
        f.close()
        return train, dev, test
    else:
        print('Generating data...')
        generate_data()


if __name__ == '__main__':
    # load_data()
    with open('title.txt','r') as rf:
        titles = rf.readlines()
    
    # titles = titles[:100]
    titles = [t.replace('\n','') for t in titles]
    
    generate_data(titles)