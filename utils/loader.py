import torch
from torch.utils import data
import dgl
from utils import constant
from itertools import accumulate
import numpy as np
from utils.config import config

MAX_TOKS= 175


class Dataset(data.Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        utter, title, label, intra, inter = map(data.get, ('utter', 
                                                           'title',
                                                           'label',
                                                           'intra',
                                                           'inter'))
        # utters: list[list[list[str]]] = debate <- turn <- sent <- token
        tok = [[torch.LongTensor(self.vocab.map(o['tokens'][:MAX_TOKS])) # each token
                                for o in u]        # each sent
                                for u in utter]    # each turn
        
        pos = [[torch.LongTensor([constant.pos_dict[_pos]
                                for _pos in o['pos'][:MAX_TOKS]]) # each token
                                for o in u]         # each sent
                                for u in utter]       # each turn
        ner = [[torch.LongTensor([constant.ner_dict[_ner]
                                for _ner in o['ner'][:MAX_TOKS]]) # each token
                                for o in u]         # each sent
                                for u in utter]       # each turn
        
        # nsents: number of sentences in each turn: list[int]
        # e.g., [40, 31, 24, 52, 90, 12]
        nsents = [len(u) for u in utter] # list[int], len(list) = n
        ntoks = [[len(u['tokens'][:MAX_TOKS]) for u in uu] for uu in utter] # list[list[int]], len(list[0]) = m
        tmasks =  [[torch.LongTensor([1 for _ in range(o)])
                    for o in u]
                    for u in ntoks] # token masking

        maxnsents = max(nsents)
        maxntoks = max([max(a) for a in ntoks])
        # a = (maxntoks, title)
        # print(a)
        # wembs = [[self.wembs[o] # each token
        #             for o in u]        # each sent
        #             for u in tok]    # each turn
        graph = self._create_graph(intra, inter, utter)

        return dict(
            utter=tok,
            pos=pos,
            ner=ner,
            graph=graph,
            nsents=nsents,
            ntoks=ntoks,
            tmasks=tmasks,
            maxnsents=maxnsents,
            maxntoks=maxntoks,
            # wembs=wembs,
            label=torch.tensor(label),
        )
    
    def _add_node(self, graph, turn, utter, offset=None):
        """
        Add nodes to graph based on their turn and offset
        """
        num_nodes = len(utter)
        node_feature = dict(
            tid=torch.ones(num_nodes, dtype=torch.int32)*turn,
            uid=torch.ones(num_nodes, dtype=torch.int32
                           ) if turn%2 else torch.zeros(num_nodes, dtype=torch.int32),
            nid=torch.LongTensor([i for i in range(num_nodes)]),
        )
        graph.add_nodes(num=num_nodes, data=node_feature)

    def _add_edge(self, graph, adj_matrix, src_offset, dst_offset, turn, edge_type):
        """
        Add nodes to graph based on their turn and offset
        """
        src, dst = np.nonzero(adj_matrix)
        a = src, dst
        num_edges = src.shape[0]
        edge_offset = constant.EDGE_OFFSET[edge_type]
        edge_feature = dict(
            turn=torch.ones(num_edges, dtype=torch.float)*turn + edge_offset
        )

        graph.add_edges(src+src_offset, dst+dst_offset, data=edge_feature)

    def _create_graph(self, intra_edge, inter_edge, utter):
        G = dgl.graph([])
        
        # offset: list of #sent in each turn
        offset = [len(argument) for argument in utter]
        # calculate prefix sum, will be used as offset for edge construction
        # [1,3,5,9] -> [1,4,9,18]
        offset = list(accumulate(offset))
        # Adding nodes
        for t, u in enumerate(utter):
            o = 0 if t==0 else offset[t-1]
            self._add_node(G, t, u, o)
        # Adding edges
        T = len(intra_edge)
        for t, attn in enumerate(intra_edge):
            o = 0 if t==0 else offset[t-1]
            # intra-arg edge
            self._add_edge(G, attn, o, o, t, 'self')
            # cross-arg edge
            if t < T - 1 and config.is_counter:
                self._add_edge(G, inter_edge[t], o, offset[t], t, 'counter')
            # if t < T - 2 and config.is_support:
            #     self._add_edge(G, inter_edge[t], o, offset[t], t, 'support')

        # check if graph is correctly constructed
        assert G.num_nodes() == offset[-1], \
                f"Fail constructing graph, #nodes = {G.num_nodes()}, get {offset[-1]} !"
        return G
    
def collate_fn(data):
    """
        utter, pos, ner
    """
    item = {}
    for k in data[0].keys():
        item[k] = [d[k] for d in data] # B x ...
    
    # tokens -> sentences -> turns -> debates -> batch of debates
    max_num_sents = max(item['maxnsents']) # for padding
    max_num_tokens = max(item['maxntoks']) # for padding
    # max_num_utter = max(item['nutter']) # for node feature setting
    B, T, N, M, H = len(item['utter']), 6, max_num_sents, max_num_tokens, config.emb_dim
    debate_len=item['ntoks']

    def _padding(x):
        """
        x: B x T x n x m
        Padding to the maximum N and M in each batch&turn
        N, M = list[int] -> BxT
        Keeping T=6, We yet to pad turns in this phase
        debate: list[list[list[list[int]]]] -> debate_len: list[list[list[int]]]  (token-level)
        
        Return: B x T x N x M
        T: num_turns, N: num_sent/turn, M: num_words/sent
        """
        padded_debate = torch.zeros(B, T, N, M, dtype=torch.int64)
        for batch_idx, (debate_l, debate) in enumerate(zip(debate_len, x)): # B
            for turn_idx, (n_sents, argument) in enumerate(zip(debate_l, debate)): # T
                for sent_idx, (ntok, sent) in enumerate(zip(n_sents, argument)): # N
                    padded_debate[batch_idx, turn_idx, sent_idx, :ntok] = sent # M
        return padded_debate

    def _node_masking(x):
        node_masked = torch.zeros(B, T, N, dtype=torch.int64)
        for bidx, nsents in enumerate(x):
            for tidx, ns in enumerate(nsents):
                node_masked[bidx, tidx, :ns] = 1
        return node_masked

    def _pad_wembs(x):
        padded_wembs = torch.zeros(B, T, N, M, H, dtype=torch.float)
        for batch_idx, (debate_l, debate) in enumerate(zip(debate_len, x)): # B
            for turn_idx, (n_sents, argument) in enumerate(zip(debate_l, debate)): # T
                for sent_idx, (ntok, sent) in enumerate(zip(n_sents, argument)): # N
                    padded_wembs[batch_idx, turn_idx, sent_idx, :ntok] = sent # M
        return padded_wembs

    utter = _padding(item['utter'])
    pos = _padding(item['pos'])
    ner = _padding(item['ner'])
    mask = _padding(item['tmasks']) # assert torch.all(utter - utter*mask) == 0
    nmask = _node_masking(item['nsents'])
    # Q: why we have to identify the nodeidx?
    # A: as we pad dummy sentences to have the same # sentence every turn
    # indentify nodeidx in a padded graph
    # turn level: ------------------    tidx + bidx*T
    # sent level: ------------------    sidx + tidx*N
    nodeidx = [sidx + (tidx + bidx*T)*N
               for bidx, oo in enumerate(item['nsents']) # bidx
               for tidx, o in enumerate(oo) # tidx
               for sidx in range(o)] # sidx
    # wembs = _pad_wembs(item['wembs'])

    return dict(
        utter=utter,
        pos=pos,
        ner=ner,
        mask=mask, # token masking
        nmask=nmask, # node masking
        nodeidx=nodeidx,
        # wembs=wembs,
        graph=dgl.batch(item['graph']),
        label=torch.stack(item['label'])
    )