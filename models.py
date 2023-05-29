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


class BIDAFWithoutAttention(nn.Module):

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BIDAFWithoutAttention, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, context, question):
        context_out, _ = self.context_lstm(context)
        question_out, _ = self.question_lstm(question)

        context_len = context.size(1)
        question_len = question.size(1)

        context_question_mul = torch.matmul(
            context_out, question_out.transpose(1, 2))
        context_attention_weights = torch.softmax(context_question_mul, dim=2)
        context_attention_out = torch.matmul(
            context_attention_weights, question_out)

        modeling_input = torch.cat([context_out, context_attention_out], dim=2)
        modeling_out, _ = self.modeling_lstm(modeling_input)

        output_input = torch.cat([context_out, modeling_out], dim=2)
        output_out = self.output_linear(output_input)

        return output_out

# returns an instance of the appropriate model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Only use the last time step output
        return out.squeeze()


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Only use the last time step output
        return out.squeeze()
    
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        attention_weights = torch.softmax(self.attention(out), dim=1)
        attended_out = torch.sum(attention_weights * out, dim=1)
        out = self.fc(attended_out)
        return out.squeeze()
    
def init_model(name, split, **kwargs):
    name = name.lower()
    if name == 'bidaf':
        return BiDAF(word_vectors=kwargs['word_vectors'],
                     hidden_size=kwargs['hidden_size'],
                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'bidaf_no_attention':
        return BIDAFWithoutAttention(word_vectors=kwargs['word_vectors'],
                                     hidden_size=kwargs['hidden_size'],
                                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'lstm':
        return LSTMModel(word_vectors=kwargs['word_vectors'],
                                     hidden_size=kwargs['hidden_size'],
                                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'lstm_attention':
        return LSTMAttentionModel(word_vectors=kwargs['word_vectors'],
                                     hidden_size=kwargs['hidden_size'],
                                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    elif name == 'bi_lstm':
        return BiLSTMModel(word_vectors=kwargs['word_vectors'],
                                     hidden_size=kwargs['hidden_size'],
                                     drop_prob=kwargs['drop_prob'] if split == 'train' else 0)
    raise ValueError(f'No model named {name}')
