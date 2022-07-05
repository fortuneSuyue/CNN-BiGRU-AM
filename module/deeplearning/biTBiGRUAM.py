import torch
from torch import nn
from module.deeplearning.tcn import TemporalBlock


def flip(x, dim):
    x_size = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *x_size[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(x_size)


class BiTemporalBlock(nn.Module):
    def __init__(self, in_dim=5, adjust_dim=16):
        super(BiTemporalBlock, self).__init__()
        self.tcn1 = TemporalBlock(in_dim=in_dim, out_dim=adjust_dim, kernel_size=3, padding=2, dilation=1)
        self.tcn2 = TemporalBlock(in_dim=in_dim, out_dim=adjust_dim, kernel_size=3, padding=2, dilation=1)
        adjust_dim *= 2
        self.bn_relu_12 = nn.Sequential(
            nn.BatchNorm1d(adjust_dim),
            nn.ReLU(inplace=True)
        )
        self.tcn3 = TemporalBlock(in_dim=adjust_dim, out_dim=adjust_dim, kernel_size=3, padding=2, dilation=1)
        self.tcn4 = TemporalBlock(in_dim=adjust_dim, out_dim=adjust_dim, kernel_size=3, padding=2, dilation=1)
        adjust_dim *= 2
        self.bn_relu_34 = nn.Sequential(
            nn.BatchNorm1d(adjust_dim),
            nn.ReLU(inplace=True)
        )
        self.sub_branch = nn.Conv1d(in_dim, adjust_dim, 1) if in_dim != adjust_dim else None

    def forward(self, x):
        y = self.bn_relu_12(torch.cat([self.tcn1(x), self.tcn2(flip(x, -1))], dim=-2))
        y = self.bn_relu_34(torch.cat([self.tcn3(y), self.tcn4(flip(y, -1))], dim=-2))
        if self.sub_branch is not None:
            x = self.sub_branch(x)
        return x+y


class BiT_GRUClassifier(nn.Module):
    def __init__(self, in_dim=5, adjust_dim=16, hidden_size=32, num_layers=1, dropout=0.1, n_classes=4,
                 bidirectional=False, attention_type: str = None):
        """
        :param in_dim:
        :param adjust_dim:
        :param hidden_size:
        :param num_layers:
        :param dropout:
        :param n_classes:
        :param bidirectional:
        :param attention_type: attention_type are supposed to be in ['V1', 'V2', 'V3']. Otherwise, None.
        """
        super(BiT_GRUClassifier, self).__init__()
        self.tcns = BiTemporalBlock(in_dim=in_dim, adjust_dim=adjust_dim)
        adjust_dim *= 4
        if num_layers == 1:
            dropout = 0.
        self.rnn = nn.GRU(input_size=adjust_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        self.attention_type = attention_type
        if attention_type is None or attention_type.upper() not in ['V1', 'V2', 'V3']:
            self.att = None
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * bidirectional * num_layers, hidden_size),
                nn.Linear(hidden_size, n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * bidirectional, hidden_size),
                nn.Linear(hidden_size, n_classes)
            )
            if attention_type == 'V1':
                from module.deeplearning.attention import AttentionModule_V1
                self.att = AttentionModule_V1(input_size=hidden_size * bidirectional, batch_first=True, reduce_sum=True)
            elif attention_type == 'V2':
                from module.deeplearning.attention import AttentionModule_V2
                self.att = AttentionModule_V2(reduce_sum=True)
            elif attention_type == 'V3':
                from module.deeplearning.attention import AttentionModule_V3
                self.att = AttentionModule_V3(input_dim=hidden_size * bidirectional, reduce_sum=True)

    def forward(self, x, need_alpha=False):
        x = self.tcns(x.transpose(1, 2).contiguous()).transpose(1, 2)
        if self.att is not None:
            out, _ = self.rnn(x)
            if self.attention_type == 'V2':
                out, alpha = self.att(out, self.dropout(out))
            else:
                out, alpha = self.att(out)
            if need_alpha:
                return self.classifier(out), alpha
            else:
                return self.classifier(out)
        else:
            _, h = self.rnn(x)
            h = h.transpose(0, 1).contiguous().view(x.size(0), -1)
            return self.classifier(h)


if __name__ == '__main__':
    sl = 15
    m = BiT_GRUClassifier(in_dim=5, hidden_size=16, num_layers=2, dropout=0.1, n_classes=8, bidirectional=True,
                          attention_type='V1')
    a = torch.ones(size=(3, sl, 5), dtype=torch.float32)
    b = m(a, need_alpha=True)
    if isinstance(b, tuple):
        print(*[item.shape for item in b])
    else:
        print(b.shape)
