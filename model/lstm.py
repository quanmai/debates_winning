import torch
import numpy as np
import torch.nn as nn
from utils import constant


def _keep_partial_grad(grad, topk):
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

class uLSTM(nn.Module):
    """
    Encode utterance using RNN, 
    each word vection is a concatenation of: wv_emb, ent_, dep_
    """

    def __init__(self, config, embeddings):
        super().__init__()
        self.config = config
        self.in_dim = config.emb_dim + config.pos_emb_dim + config.ner_emb_dim
        self.rnn = nn.LSTM(input_size=self.in_dim,
                           hidden_size=config.rnn_hidden_dim,
                           num_layers=config.rnn_layers,
                           bias=True,
                           batch_first=True,
                           dropout=config.rnn_dropout,
                           bidirectional=True)
        self.linear = nn.Linear(2*config.rnn_hidden_dim, config.nfeat) # map lstm dim to hid dim
        self.pos_emb, self.ner_emb = embeddings
        self.in_drop = nn.Dropout(config.input_dropout)
        self.rnn_drop = nn.Dropout(config.rnn_dropout)

    def rnn_encoder(self, emb, masks):
        _in = emb * masks.unsqueeze(-1)
        _out, (ht, ct) = self.rnn(_in)
        _out = _out * masks.unsqueeze(-1)
        return _out # (BxTxN) x M x 2 x H_

    def forward(self, batch):
        words, masks, pos, ner, wembs = batch
        # word_emb = self.wrd_emb(words)
        # embs = [word_emb]
        embs = [wembs]
        if self.config.pos_emb_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.config.ner_emb_dim > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        embs = self.rnn_drop(self.rnn_encoder(embs, masks))
        embs = self.linear(embs) * masks.unsqueeze(-1)
        return embs # (BxTxN) x M x H


class UtterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # if self.config.fine_tune_we:
        #     self.wrd_emb = nn.Embedding(
        #         num_embeddings=self.vocab.num_words,
        #         embedding_dim=self.config.emb_dim
        #         )   
        #     self._init_word_embedding(np.load(open(config.embed_f, 'rb'),
        #                                     allow_pickle=True))
        # else:
        #     ww = torch.from_numpy(np.load(
        #                             open(config.embed_f, 'rb'),
        #                             allow_pickle=True)
        #                             ).float()
        #     self.wrd_emb = nn.Embedding.from_pretrained(ww)
        pos_emb = nn.Embedding(
                num_embeddings=len(constant.POS),
                embedding_dim=self.config.pos_emb_dim
            ) if self.config.pos_emb_dim > 0 else None
        ner_emb = nn.Embedding(
                num_embeddings=len(constant.NER),
                embedding_dim=self.config.ner_emb_dim
            ) if self.config.ner_emb_dim > 0 else None
        self.embeddings = (pos_emb, ner_emb)
        self.lstm = uLSTM(config, self.embeddings)

    # def _init_word_embedding(self, wv_emb):
    #     self.wrd_emb.weight = nn.Parameter(torch.from_numpy(wv_emb).float())
    #     if self.config.tune_topk <= 0:
    #         print('Do not fine-tune word embedding layer')
    #         self.wrd_emb.weight.requires_grad = False
    #     elif self.config.tune_topk < self.vocab.num_words:
    #         print('Fine-tune top {self.config.tune_topk} word embeddings')
    #         self.wrd_emb.weight.register_hook(
    #             lambda x: _keep_partial_grad(x, self.config.tune_topk)
    #         )
    #     else:
    #         print('Finetune all word embeddings')
    
    def forward(self, batch):
        conv, masks, ner, pos, wembs = batch['utter'], batch['mask'], batch['ner'], batch['pos'], batch['wembs']
        B, T, N, M, H = wembs.shape
        conv = conv.view(-1, M)   # -> (BxTxN, M)
        masks = masks.view(-1, M)
        ner = ner.view(-1, M)
        pos = pos.view(-1, M)
        wembs = wembs.view(-1, M, H)
        h = self.lstm((conv, masks, pos, ner, wembs)) # (BxTxN) x M x H
        h = torch.mean(h, dim=1) # (BxTxN)x H
        h = h.reshape(B, T, N, -1)
        return h