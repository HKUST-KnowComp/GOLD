import torch
import torch.nn as nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        h = torch.matmul(inp, self.W)
        N = h.size()[1]
        B = h.size()[0]

        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)
        a_input = torch.cat((h, a), dim=2)
        e = self.leakyrelu(torch.matmul(a_input, self.a))
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention.view(B, 1, N)
        return torch.matmul(attention, h).squeeze(1)
