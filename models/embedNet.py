import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedNet(nn.Module):
    """ Embedding layer
    Input: raw sequence
    Output: Embedding vectors
    Share Embedding between policy net and classifier.
    """

    def __init__(self, vocab_size, d_embed, drop):
        super(EmbedNet, self).__init__()
        self.p = drop
        self.embed = nn.Embedding(vocab_size, d_embed)
        for param in self.embed.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)

    def forward(self, sequence):
        return F.dropout(self.embed(sequence), p=self.p)
