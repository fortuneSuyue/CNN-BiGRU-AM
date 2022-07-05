import math

import torch
from torch import nn


class AttentionModule_V1(nn.Module):
    def __init__(self, input_size, batch_first=True, reduce_sum=True):
        """
        This is a attention module for RNN. Better suited to classified tasks. Paper Reference:
        https://aclanthology.org/P16-2034.pdf. Code Reference:
        https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction/blob/master/attention.py.

        :param input_size:
        :param batch_first:
        :param reduce_sum: If attention is used for Classification or Regression, reduce_sum is better to be set as True.
            It is better to be False while used for Seq2Seq.
        """
        super(AttentionModule_V1, self).__init__()
        self.batch_first = batch_first
        self.w = nn.Parameter(torch.Tensor(1, input_size))
        nn.init.uniform_(self.w, -0.1, 0.1)
        self.reduce_sum = reduce_sum

    def forward(self, h):
        """
        batch first or not.
        :param h:
        :return:
        """
        if not self.batch_first:
            h = h.permute(1, 0, 2)
        m = torch.tanh(h)
        m = torch.matmul(self.w, m.transpose(1, 2)).squeeze(1)  # b, t
        alpha = torch.softmax(m, dim=1).unsqueeze(-1)
        m = h * alpha
        if self.reduce_sum:
            m = torch.sum(m, dim=1)
        m = torch.tanh(m)
        return m, alpha


class AttentionModule_V2(nn.Module):
    def __init__(self, reduce_sum=True):
        """
        This is implemented according to the definition of the attention (self-attention).
        """
        super(AttentionModule_V2, self).__init__()
        self.reduce_sum=reduce_sum

    def forward(self, x, query):
        """
        :param x: key=value=x
        :param query:
        :return:
        """
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(query.size(-1))
        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, x)
        if self.reduce_sum:
            return attn_output.sum(1), attn
        else:
            return attn_output, attn


class AttentionModule_V3(nn.Module):
    def __init__(self, input_dim: int, reduce_sum=True):
        """
        Code reference: https://github1s.com/WHLYA/text-classification
        Formula reference: https://www.cnblogs.com/cxq1126/p/13504437.html
        Paper: https://aclanthology.org/N16-1174.pdf
        :param input_dim:
        :param reduce_sum:
        """
        super(AttentionModule_V3, self).__init__()
        self.w = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.u = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.uniform_(self.u, -0.1, 0.1)
        nn.init.uniform_(self.w, -0.1, 0.1)
        self.reduce_sum = reduce_sum

    def forward(self, x):
        u = torch.tanh(torch.matmul(x, self.w))
        attn = torch.matmul(u, self.u)
        attn = torch.softmax(attn, dim=1)
        attn_output = x * attn
        if self.reduce_sum:
            return attn_output.sum(1), attn
        else:
            return attn_output, attn


if __name__ == '__main__':
    a = torch.ones(size=(3, 15, 5), dtype=torch.float32)
    a, w = AttentionModule_V1(input_size=5, reduce_sum=True)(a)
    print(a.shape, w.shape)
