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
    text = re.sub(url, URL , text)
    text = re.sub(r'-?\d+(\.\d+)?(?!\w)', NUM, text)
    text = re.sub(r'[-+]?.?[0-9]+[a-zA-Z]*', NUM, text) #like 19th
    text = re.sub(r"&gt;.*\n\n", QUO, text)
    text = re.sub(r'\r\n', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'^[^a-zA-Z]+','', text) # Remove non-alphabetic char before alphabetic chars
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
        Edges are defined by cosine similarity among sentence
        Remove sentence less than min_len words, 
        remove debate if a turn have less than 3 sentence"""
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
                    print('---------')
                    print(text)
                    print('---------')
                    text = _process_text(text)
                    doc = nlp(text)
                    sents = [s.text for s in doc.sents]
                    print(sents)
                    print('---------')
                    # sents = nltk.tokenize.sent_tokenize(text)
                    _sents = []
                    for s in sents:
                        _count = 0
                        for w in s.split(' '):
                            if w.isalpha():
                                _count += 1
                        if _count >= min_len:
                            # print(s)
                            ss = ' '.join([sss for sss in s.split(' ') if sss.isalpha()])
                            if s[-1][-1] == '.':
                                last_word = s.split(' ')[-1]
                                ss += ' ' + last_word
                            _sents.append(ss)
                    # sents = [s for s in sents if len([w for w in s.split(' ') if w.isalpha()]) >= min_len]
                    sents = _sents
                    print(sents)
                    breakpoint()
                    if sents and len(sents) >= 10: # high quality arguments
                        sents_embeddings = model.encode(sents)
                        # transform scale
                        sents_embeddings_t = sents_embeddings.T
                        scaler = sklearn.preprocessing.StandardScaler().fit(sents_embeddings_t)
                        sents_embeddings_scaled = scaler.transform(sents_embeddings_t)
                        arguments_embed_list.append(sents_embeddings_scaled.T)
                        # arguments_embed_list.append(sents_embeddings)
                        arguments_len_list.append(len(sents))
                    else:
                        okay_arg = False
            if not okay_arg:
                continue
            abel = _get_label(debate)
            print(f'Winner is: {abel}')
            breakpoint()
            # arguments_embed_list = list(itertools.chain(*arguments_embed_list)) # list of lists into a list
            arguments_embed_list = arguments_embed_list[-NUM_TURNS:]
            # arguments_embed_list = arguments_embed_list[:NUM_TURNS]
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

def get_stats(file=config.proce_f):
    """
        This function to get stats of debate, including:
            - Number of sentence made by Winner/Loser
            - Number of countering/supporting connections made by Winner/Loser
    """
    import numpy as np
    def get_connection(connection_type, connection_mat, threshold):
        n = NUM_TURNS-1 if connection_type=='counter' else NUM_TURNS-2
        first_connection, second_connection = 0, 0
        for i in range(n):
            # Pro made 2 counters, 2 supports
            # Con made 3 counters, 2 supports
            connection = connection_mat[i].T # Transposition to get similarity score from t -> t-1
            A = np.where(connection>=threshold, 1, 0)
            num_connections = np.sum(A) # / connection.shape[0]
            if i % 2:
                second_connection += num_connections
            else:
                first_connection += num_connections
        if connection_type == 'counter':
            pro_connection, con_connection = first_connection/2, second_connection/3
        else:
            pro_connection, con_connection = second_connection/2, first_connection/2
        return pro_connection, con_connection 
    
    winner_stats = {
        'number_of_sentence_per_turn': 0.0,
        'number_of_counter_edges': 0.0,
        'number_of_support_edges': 0.0
    }
    loser_stats = {
        'number_of_sentence_per_turn': 0.0,
        'number_of_counter_edges': 0.0,
        'number_of_support_edges': 0.0
    }

    with open(file, 'rb') as f:
        train, dev, test = pickle.load(f)
        dataset = train + dev + test
        num_debates = len(dataset)
        for d in dataset:
            arguments_embed_list = d['graph']
            winner = d['label'] # 1: Pro, 0: Con
            # intra_sim_list = d['adj']['intra_adj']
            counter_sim_list = d['adj']['counter_adj']
            support_sim_list = d['adj']['support_adj']

            len_sents = [len(a) for a in arguments_embed_list]
            number_sent_pro, number_sent_con = 0 , 0

            # number of sentence per turn
            for i in range(NUM_TURNS):
                if i % 2:
                    number_sent_con += len_sents[i]
                else:
                    number_sent_pro += len_sents[i]
            winner_stats['number_of_sentence_per_turn'] += number_sent_pro if winner == 1 else number_sent_con
            loser_stats['number_of_sentence_per_turn'] += number_sent_con if winner == 1 else number_sent_pro

            # number of counter edges
            pros_counter, cons_counter = get_connection('counter', counter_sim_list, 0.85)
            pros_support, cons_support = get_connection('support', support_sim_list, 0.85)
            winner_stats['number_of_counter_edges'] += pros_counter if winner == 1 else cons_counter
            loser_stats['number_of_counter_edges'] += cons_counter if winner == 1 else pros_counter
            winner_stats['number_of_support_edges'] += pros_support if winner == 1 else cons_support
            loser_stats['number_of_support_edges'] += cons_support if winner == 1 else pros_support
        f.close()
        
        for k in winner_stats.keys():
            winner_stats[k] /= num_debates
            loser_stats[k] /= num_debates
        winner_stats['number_of_sentence_per_turn'] /= 3
        loser_stats['number_of_sentence_per_turn'] /= 3

        print('Winner #sentences: {number_of_sentence_per_turn}, Winner #Countering_edges: {number_of_counter_edges}, \
              Winner #Supporting_edges: {number_of_support_edges}'.format(**winner_stats))
        print('Loser #sentences: {number_of_sentence_per_turn}, Loser #Countering_edges: {number_of_counter_edges}, \
              Loser #Supporting_edges: {number_of_support_edges}'.format(**loser_stats))

if __name__ == '__main__':
    print('Getting stats of dataset ---')
    get_stats()