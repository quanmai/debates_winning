
from utils.config import config
from utils import constant
import json
from collections import defaultdict, Counter
import numpy as np
import spacy


## spacy/ntlk settings ##
from spacy.language import Language
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "\n" or token.text == "\r":
            doc[token.i + 1].is_sent_start = True
        # if token.text == ":":
        #     doc[token.i + 1].is_sent_start = False
    return doc
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("set_custom_boundaries", before="parser")

from nltk.corpus import stopwords
def set_stopwords():
    sw = set(stopwords.words('english'))
    for w in ['!',',','.','?', '(', ')', '"', "'", ';', ':']:
        sw.add(w)
    return sw

##########
STOPWORDS = set_stopwords()
URL_REPLACE = ' website '
NUM_REPLACE = ' number '
NEWLINE = r"[\n\r]"
####

class Vocab(object):
    def __init__(self, wordlist):
        self.w2idx = {w: i for i, w in enumerate(wordlist)}
        self.idx2w = {i: w for i, w in enumerate(wordlist)}
        self.num_words = len(wordlist)

    def map(self, token_list):
        """
        Map a list of tokens to their idx
        """
        return [self.w2idx[w] if w in self.w2idx else constant.UNK_ID for w in token_list]


def build_embedding(wv_file, vocab, wv_dim=300):
    vocab_size = len(vocab)
    emb = np.random.randn(vocab_size, wv_dim) * 0.01
    emb[constant.PAD_ID] = 0
    wid = {w: i for i, w in enumerate(vocab)}

    with open(file=wv_file, mode='r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            token, vec = values[0], values[1:]
            if token in wid:
                idx = wid[token]
                emb[idx] = [float(v) for v in vec]
    return emb

def load_glove_vocab(gv_file, wv_dim=None):
    """ Build vocab from glove word embedding file """
    vocab = set()
    with open(gv_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            token, vec = values[0], values[1:]
            vocab.add(token)
    return vocab

def cleaner(text):
    """
        Clean the following things:
            - url -> 'website'
            - number/digit/percentage -> 'number'
        We do not remove stopwords/puntuations here!
    """
    import re
    url = r"\b(?:(?:https?|ftp):\/\/|www\.)[\w-]+(\.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?\b"
    num = r"\b(?:\d+(?:\.\d*)?|\.\d+)\b(?:%|\b)"
    text = re.sub(url, URL_REPLACE, text) # replace url by 'website'
    text = re.sub(num, NUM_REPLACE, text) # replace digits by 'number'
    text = re.sub(NEWLINE, "", text) # remove newline characters
    text = re.sub(r"/", " / ", text) #
    text = re.sub(r"\s+", " ", text) # remove duplicated whitespaces
    text = re.sub(r"^\s+", "", text) # remove SOS whitespace
    text = re.sub(r'=+', "=", text) # remove =====

    text = text.lower()
    for k, v in config.word_pairs.items():
        text = text.replace(k,v)
    return text

def preprocess(debate_name):
    # load debate:
    with open(file=config.filtered_f, mode='r') as rf:
        debates = json.load(rf)
    rf.close()

    D = [] # list of all debates list[dict]
    T = [] # list of all tokens list[str]
    pros, cons = 0, 0
    for name, debate in debates.items():
        if name in debate_name:
            print('Processing debates: {}'.format(name))
            label = get_label(debate)
            debate_info = {}
            intra_adjac = []
            inter_adjac = []
            lemma_list = [] # for cross-arg lemma co-occur
            utter_list = [] # list[list[dict()]]
            for round in debate['rounds']:
                for side in round:
                    utter = [] # utter of 1 turn
                    lemma = [] # lemma 1 turn
                    arguments = side['text']
                    arguments = cleaner(arguments)
                    doc = nlp(arguments)
                    sentences = [a.text for a in doc.sents] # list of sentences
                    for sentence in sentences:
                        # a = (sentence, len(sentence.split(' ')))
                        nlp_feat = extract_features(sentence)
                        # a = (len(nlp_feat['tokens']), nlp_feat['tokens'])
                        # if len(nlp_feat['tokens']) > 100:
                        #     print(a)
                        utter.append(nlp_feat)
                        lemma.append(nlp_feat['lem'])
                    utter_list.append(utter)
                    lemma_list.append(lemma)
            utter_list = utter_list[:6]
            lemma_list = lemma_list[:6]
            assert len(lemma_list) == 6 
            for i, lem in enumerate(lemma_list):
                intra = adj_matrix(lem, dist=3)
                intra_adjac.append(intra)
                if i + 1 < len(lemma_list):
                    inter = adj_matrix(lem, lemma_list[i+1], co=2)
                    inter_adjac.append(inter)
            debate_info['utter'] = utter_list # T x N x M
            debate_info['title'] = debate['title']
            debate_info['label'] = label
            debate_info['intra'] = intra_adjac
            debate_info['inter'] = inter_adjac
            D.append(debate_info)

            T += [tok 
                for qq in utter_list # turn-level
                for q in qq     # sentence-level
                for tok in q['tokens']]  # list-of-tokens
            pros += int(label==1)
            cons += int(label==0)
    print(f'Number of debates Pros win: {pros}')
    print(f'Number of debates Cons win: {cons}')
    return T, D

def extract_features(text: str):
    """
    Extract linguistic characteristics of a sentence
    """

    d = defaultdict(list)
    text = nlp(text)
    d['tokens'] = [str(token) for token in text]
    d['pos'] = [token.pos_ for token in text]
    d['dep'] = [token.dep_ for token in text]
    d['ner'] = [token.ent_type_ for token in text]
    d['lem'] = [token.lemma_ for token in text]
    return d

def adj_matrix(sent_list_a, sent_list_b=None, dist=3, co=3):
    """
    Return the adjacency matrix (edges), 
    should we call it in data_loader?
    """

    import numpy as np
    if not sent_list_b:
        # intra-argument connection
        # each sentence will connect to its "dist" nearest sentence
        # adj.shape = (n, n)
        nums = len(sent_list_a)
        adj = np.zeros((nums, nums), dtype=int)
        for i in range(nums):
            left, right = max(0, i-dist), min(nums, i+dist+1)
            adj[i][left:right] = 1
        return adj
    else:
        # inter-argument adjacency matrix using word/lemma co-occurence
        # adj.shape = (n_a, n_b)
        nums_a, nums_b = len(sent_list_a), len(sent_list_b)
        adj = np.zeros((nums_a, nums_b), dtype=int)
        for i, sent_a in enumerate(sent_list_a):
            sent_a = clean_text(sent_a)
            words_dict_a = Counter(sent_a)
            for j, sent_b in enumerate(sent_list_b):
                sent_a = clean_text(sent_b)
                words_dict_b = Counter(sent_b)
                co_occur = words_dict_a & words_dict_b
                if len(co_occur) >=  co:
                    adj[i][j] = 1
        return adj

def clean_text(sentence):
    return [token 
            for token in sentence 
            if token not in STOPWORDS and token.isalpha()]
 
def get_label(debate):
    """ PRO:1, 
        CON: 0
    """
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

def generate_data(titles, seed=42):
    """
    To save training time, this function generates datasets,
    including pickle files containing `graph` & vocab, and embedding file 
    """

    import random
    import pickle

    T, D = preprocess(titles)

    # train, dev, test split
    random.Random(seed).shuffle(D)
    num = len(D)
    print(f'Number of qualified debates are: {num}')
    idx_train = int(0.6*num)
    idx_dev = int(0.8*num)
    train, dev, test = D[:idx_train], D[idx_train:idx_dev], D[idx_dev:]

    # vocab generation
    glove_vocab = load_glove_vocab(config.glove_f)
    v, _ = build_vocab(T, glove_vocab)
    vocab = Vocab(v)

    # embedding construction
    embedding = build_embedding(config.glove_f, v)

    # save to files
    print('Saving embeddings...')
    np.save(config.embed_f, embedding)
    #
    print(f'Dumping to files {config.proce_f} ...')
    with open(config.proce_f, 'wb') as f:
        pickle.dump([train, dev, test, vocab], f)
        
    print('Done, Good luck :-)')

def generate_data_3_splits(titles, seed=42):
    """
    Basically this has the same functionalities of the generate_data(),
    except this function seperates 3 vocab sets for the ease of memory
    """

    import random
    import pickle

    num = len(titles)
    random.Random(seed).shuffle(titles)
    idx_train = int(0.6*num)
    idx_dev = int(0.8*num)

    # vocab generation
    glove_vocab = load_glove_vocab(config.glove_f)

    def _generate(split_type):
        if split_type == 'train':
            T, D = preprocess(titles[:idx_train])
            efile = config.embed_f_train
        elif split_type == 'dev':
            T, D = preprocess(titles[idx_train:idx_dev])
            efile = config.embed_f_dev
        else:
            T, D = preprocess(titles[idx_dev:])
            efile = config.embed_f_test

        v, _ = build_vocab(T, glove_vocab)
        vocab = Vocab(v)

        # embedding construction
        embedding = build_embedding(config.glove_f, v)

        # save to files
        print(f'Saving embeddings {split_type} ...')
        np.save(efile, embedding)

        return D, vocab

    train, vocab_train = _generate('train')
    dev, vocab_dev = _generate('dev')
    test, vocab_test = _generate('test')

    #
    print(f'Dumping to files {config.proce_f} ...')
    with open(config.proce_f, 'wb') as f:
        pickle.dump([train, dev, test, vocab_train, vocab_dev, vocab_test], f)
        
    print('Done, Good luck :-)')


def load_dataset():
    """
    Loading dataset files, if files do not exist, generate them ... 
    """
    import os
    import pickle

    if not os.path.exists(config.proce_f):
        print(f'{config.proce_f} does not exist. Generating data...')
        with open('title.txt','r') as rf:
            titles = rf.readlines()
        titles = [t.replace('\n','') for t in titles]
        generate_data_3_splits(titles)

    print('Loading dataset...')
    with open(config.proce_f, 'rb') as f:
        train, dev, test, vocab_train, vocab_dev, vocab_test = pickle.load(f)
    f.close()
    return train, dev, test, vocab_train, vocab_dev, vocab_test


def build_vocab(tokens, glove_vocab):
    """
    Build vocab from tokens and glove vocab
    :Example::
        ['age', 'learn', 'play' ...., 'specialoneasje']
    """
    counter = Counter(tokens)
    v = sorted([t for t in counter if t in glove_vocab
                and counter.get(t) > 1], key=counter.get, reverse=True)
    v += constant.SPECIAL_TOKENS
    return v, counter


if __name__ == '__main__':
    generate_data()