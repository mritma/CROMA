import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_embed, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_embed % heads == 0, "d_embed can't be divisible by heads!"

        self.d_embed = d_embed
        self.d_k = d_embed // heads
        self.h = heads

        self.q_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_embed, d_embed)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # (bs, seq_len, heads, d_embed/heads)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # (bs, heads, seq_len, d_embed/heads)
        q, k, v = map(lambda x: x.transpose(1, 2), [q, k, v])

        scores = self.attention(q, k, v, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_embed)
        output = self.out(concat)

        return output

    def attention(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores.scores.masked_fill(mask == 0, -1e25)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_embed, d_ff=2048, dropout=0.0):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_embed, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_embed)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_embed, eps=1e-6):
        super(Norm, self).__init__()

        self.size = d_embed
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_embed, heads, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_embed)
        self.norm_2 = Norm(d_embed)
        self.attn = MultiHeadAttention(heads, d_embed)
        self.ffn = FeedForward(d_embed)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ffn(x2))
        return x
