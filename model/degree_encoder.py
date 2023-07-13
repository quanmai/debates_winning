"""Utterance Encoder"""

import torch
import torch.nn as nn
from utils.config import config

class DegreeEncoder(nn.Module):
    def __init__(self, max_degree, embedding_dim, direction="both"):
        super(DegreeEncoder, self).__init__()
        self.direction = direction
        if direction == "both":
            self.encoder1 = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
            self.encoder2 = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
        else:
            self.encoder = nn.Embedding(
                max_degree + 1, embedding_dim, padding_idx=0
            )
        self.max_degree = max_degree

    def forward(self, g):
        in_degree = torch.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = torch.clamp(g.out_degrees(), min=0, max=self.max_degree)

        if self.direction == "in":
            degree_embedding = self.encoder(in_degree)
        elif self.direction == "out":
            degree_embedding = self.encoder(out_degree)
        else: #self.direction == "both":
            degree_embedding = self.encoder1(in_degree) + self.encoder2(
                out_degree
            )
        return degree_embedding

class PosEncoder(nn.Module):
    def __init__(self, max_pos, embedding_dim):
        super(PosEncoder, self).__init__()
        self.pos_embedding = nn.Embedding(max_pos+1, embedding_dim, padding_idx=0)
        self.max_pos = max_pos
    def forward(self, g):
        node_idx = g.ndata['nid']
        node_idx = torch.clamp(node_idx, min=0, max=self.max_pos)
        pos_embedding = self.pos_embedding(node_idx)
        return pos_embedding

class NodeEncoder(nn.Module):
    def __init__(self, max_degree, max_pos, embedding_dim, dropout=0.1, direction='both', op='add'):
        super(NodeEncoder, self).__init__()
        self.degree_embeddings = DegreeEncoder(max_degree, embedding_dim, direction)
        if config.pos_emb:
            self.pos_embeddings = PosEncoder(max_pos, embedding_dim)
        if config.turn_emb:
            self.turn_embeddings = nn.Embedding(6, embedding_dim, padding_idx=0)
        if config.user_emb:
            self.debater_embeddings = nn.Embedding(2, embedding_dim, padding_idx=0)
        if op=='cat':
            self.layer_norm = nn.LayerNorm(2*embedding_dim, eps=1e-12)
        else:
            self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.op = op
    def forward(self, g, feat='h'):
        dEmbed = self.degree_embeddings(g)
        nodeEmbed = dEmbed
        if config.turn_emb:
            tTensor = g.ndata['ids']
            tEmbed = self.turn_embeddings(tTensor)
            nodeEmbed += tEmbed
        if config.user_emb:
            uTensor = g.ndata['uid']
            uEmbed = self.debater_embeddings(uTensor)
            nodeEmbed += uEmbed
        if config.pos_emb:
            pEmbed = self.pos_embeddings(g)
            nodeEmbed += pEmbed

        x = g.ndata[feat]
        if self.op == 'add':
            x += nodeEmbed
        else:
            x = torch.cat((x, nodeEmbed), dim=1)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x