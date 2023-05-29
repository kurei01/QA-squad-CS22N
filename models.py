"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c_emb, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(att, c_len)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class LSTM(nn.Module):
    """Baseline LSTM model for SQuAD."""

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding.from_pretrained(word_vectors)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.out = layers.BiDAFOutput(hidden_size=hidden_size, drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc, _ = self.lstm(c_emb)       # (batch_size, c_len, hidden_size)
        q_enc, _ = self.lstm(q_emb)       # (batch_size, q_len, hidden_size)

        out = self.out(c_enc, c_mask)     # 2 tensors, each (batch_size, c_len)

        return out

class BiLSTM(nn.Module):
    """Baseline BiLSTM model for SQuAD."""

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiLSTM, self).__init__()
        self.emb = nn.Embedding.from_pretrained(word_vectors)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.out = layers.BiDAFOutput(hidden_size=2 * hidden_size, drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc, _ = self.lstm(c_emb)       # (batch_size, c_len, 2 * hidden_size)
        q_enc, _ = self.lstm(q_emb)       # (batch_size, q_len, 2 * hidden_size)

        out = self.out(c_enc, c_mask)     # 2 tensors, each (batch_size, c_len)

        return out

class LSTMAttention(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, cw_idxs, qw_idxs):
        c_emb = self.embedding(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.embedding(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc, _ = self.lstm(c_emb.transpose(0, 1))  # (c_len, batch_size, 2 * hidden_size)
        q_enc, _ = self.lstm(q_emb.transpose(0, 1))  # (q_len, batch_size, 2 * hidden_size)

        c_len = c_enc.size(0)
        q_len = q_enc.size(0)

        c_enc = c_enc.transpose(0, 1)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = q_enc.transpose(0, 1)  # (batch_size, q_len, 2 * hidden_size)

        c_att = self.attention(c_enc)  # (batch_size, c_len, 1)
        q_att = self.attention(q_enc)  # (batch_size, q_len, 1)

        c_att = c_att.transpose(1, 2)  # (batch_size, 1, c_len)
        q_att = q_att.transpose(1, 2)  # (batch_size, 1, q_len)

        c_att = torch.softmax(c_att, dim=2)  # (batch_size, 1, c_len)
        q_att = torch.softmax(q_att, dim=2)  # (batch_size, 1, q_len)

        c_rep = torch.bmm(c_att, c_enc)  # (batch_size, 1, 2 * hidden_size)
        q_rep = torch.bmm(q_att, q_enc)  # (batch_size, 1, 2 * hidden_size)

        c_rep = c_rep.squeeze(1)  # (batch_size, 2 * hidden_size)
        q_rep = q_rep.squeeze(1)  # (batch_size, 2 * hidden_size)

        concat_rep = torch.cat((c_rep, q_rep), dim=1)  # (batch_size, 4 * hidden_size)
        concat_rep = self.dropout(concat_rep)

        out = self.fc(concat_rep)  # (batch_size, hidden_size)

        return out
    
class BiLSTMAttention(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(word_vectors)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, cw_idxs, qw_idxs):
        c_emb = self.embedding(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.embedding(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc, _ = self.lstm(c_emb.transpose(0, 1))  # (c_len, batch_size, 2 * hidden_size)
        q_enc, _ = self.lstm(q_emb.transpose(0, 1))  # (q_len, batch_size, 2 * hidden_size)

        c_len = c_enc.size(0)
        q_len = q_enc.size(0)

        c_enc = c_enc.transpose(0, 1)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = q_enc.transpose(0, 1)  # (batch_size, q_len, 2 * hidden_size)

        c_att = self.attention(c_enc)  # (batch_size, c_len, 1)
        q_att = self.attention(q_enc)  # (batch_size, q_len, 1)

        c_att = c_att.transpose(1, 2)  # (batch_size, 1, c_len)
        q_att = q_att.transpose(1, 2)  # (batch_size, 1, q_len)

        c_att = torch.softmax(c_att, dim=2)  # (batch_size, 1, c_len)
        q_att = torch.softmax(q_att, dim=2)  # (batch_size, 1, q_len)

        c_rep = torch.bmm(c_att, c_enc)  # (batch_size, 1, 2 * hidden_size)
        q_rep = torch.bmm(q_att, q_enc)  # (batch_size, 1, 2 * hidden_size)

        c_rep = c_rep.squeeze(1)  # (batch_size, 2 * hidden_size)
        q_rep = q_rep.squeeze(1)  # (batch_size, 2 * hidden_size)

        concat_rep = torch.cat((c_rep, q_rep), dim=1)  # (batch_size, 4 * hidden_size)
        concat_rep = self.dropout(concat_rep)

        out = self.fc(concat_rep)  # (batch_size, hidden_size)

        return out


# returns an instance of the appropriate model


def init_model(name, split, **kwargs):
    name = name.lower()
    if name == 'bidaf':
        return BiDAF(word_vectors=kwargs['word_vectors'],
                     hidden_size=kwargs['hidden_size'],
                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)

    raise ValueError(f'No model named {name}')
