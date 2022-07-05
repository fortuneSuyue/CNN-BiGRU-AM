import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayerV1(nn.Module):
    """
    Simple Graph Attention Network (GAT) layer, similar to https://arxiv.org/abs/1710.10903
    Code Reference: https://github.com/Diego999/pyGAT/blob/3664f2dc90cbf971564c0bf186dc794f12446d0c/layers.py#L7
                    https://zhuanlan.zhihu.com/p/128072201
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=1e-2, concat=True):
        """
        :param in_features:
        :param out_features:
        :param dropout:
        :param alpha: parameter for LeakyReLU
        :param concat: determine whether to use ELU at the end.
        """
        super(GraphAttentionLayerV1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.w = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyRelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h, adj=None):
        wh = torch.matmul(h, self.w)  # h.shape: (B, N, in_features), wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(wh)
        zero_vec = -9e15 * torch.ones_like(e)
        if adj is None:
            adj = torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h_prime = torch.matmul(attention, wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        wh1 = torch.matmul(wh, self.a[:self.out_features, :])
        wh2 = torch.matmul(wh, self.a[self.out_features:, :])
        # broadcast add
        e = wh1 + wh2.transpose(-1, -2)
        return self.leakyRelu(e)


class GraphAttentionLayerV2(nn.Module):
    """
    Graph Attention Network (GAT) layer from
                                        MTAD-GAT：Multivariate Time-series Anomaly Detection via Graph Attention Network,
                                        similar to https://arxiv.org/abs/2009.02040
    Code Reference: https://github.com/mangushev/mtad-gat/blob/63c1fe48567bd77c2299f4445280a0b1b8ad8496/model.py#L156
                    https://zhuanlan.zhihu.com/p/385812392
    """

    def __init__(self, num_node, node_size, dropout=0.5, alpha=0.2, concat=False):
        """
        :param dropout:
        :param alpha: parameter for LeakyReLU
        """
        super(GraphAttentionLayerV2, self).__init__()
        self.num_node = num_node
        self.node_size = node_size
        self.concat = concat

        self.a = nn.Parameter(torch.empty(size=(2 * node_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.layerNorm = nn.LayerNorm(normalized_shape=[num_node, num_node], eps=1e-14)
        self.leakyRelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, input_tensor: torch.Tensor):
        """
        :param input_tensor: (B, N, D), D is node_size, the dimension of the node.
        :return: tensor
        """
        e = self._prepare_attentional_mechanism_input(input_tensor)
        alpha = torch.softmax(e, dim=1)
        if self.dropout > 1e-3:
            alpha = torch.dropout(alpha, self.dropout, train=self.training).unsqueeze(-1)
        else:
            alpha = alpha.unsqueeze(-1)
        input_tensor = torch.unsqueeze(input_tensor, 1).repeat(1, self.num_node, 1, 1)  # (B, N, N, D)
        h_prime = torch.sigmoid(torch.sum(alpha * input_tensor, dim=2))
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, input_tensor):
        """
        :param input_tensor: (B, N, D)
        :return:
        """
        i_dim = torch.unsqueeze(input_tensor, 2).repeat(1, 1, self.num_node, 1)  # (B, N, N, D)
        j_dim = i_dim.transpose(2, 1)
        ij_dim = torch.cat([i_dim, j_dim], dim=-1).unsqueeze(-1)  # (B, N, N, 2D, 1)
        e = torch.matmul(self.a.T, ij_dim).squeeze(-1).squeeze(-1)
        e = self.layerNorm(e)
        return self.leakyRelu(e)


if __name__ == '__main__':
    x = torch.randn(2, 5, 10)
    # print(x)
    # gal = GraphAttentionLayerV1(10, 10, 0.2, 0.2)
    gal = GraphAttentionLayerV2(5, 10)
    y = gal(x)
    if y is not None:
        print(y.shape)
    print(y)
