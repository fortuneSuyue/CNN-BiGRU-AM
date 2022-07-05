import torch
from torch import nn
from module.deeplearning.gat import GraphAttentionLayerV2


class GRUClassifier(nn.Module):
    def __init__(self, in_dim, hidden_size=32, num_layers=1, dropout=0.1, n_classes=4, bidirectional=False,
                 attention_type: str = None):
        super(GRUClassifier, self).__init__()
        if num_layers == 1:
            dropout = 0.
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout)
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


class CNN_GRUClassifier(nn.Module):
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
        super(CNN_GRUClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=adjust_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(adjust_dim),
            nn.ReLU(inplace=True)
        )
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
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        if self.att is not None:
            out, _ = self.rnn(x)
            # print(out.shape, '...')
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


class CNN_BiGRUPlusClassifier(nn.Module):
    def __init__(self, in_dim=5, adjust_dim=16, hidden_size=32, num_layers=1, dropout=0.1, n_classes=4,
                 attention_type: str = None):
        """
        :param in_dim:
        :param adjust_dim:
        :param hidden_size:
        :param num_layers:
        :param dropout:
        :param n_classes:
        :param attention_type: attention_type are supposed to be in ['V1', 'V2', 'V3']. Otherwise, None.
        """
        super(CNN_BiGRUPlusClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=adjust_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(adjust_dim),
            nn.ReLU(inplace=True)
        )
        self.hidden_size = hidden_size
        if num_layers == 1:
            dropout = 0.
        self.rnn = nn.GRU(input_size=adjust_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_type = attention_type
        if attention_type is None or attention_type.upper() not in ['V1', 'V2', 'V3']:
            self.att = None
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * num_layers, hidden_size),
                nn.Linear(hidden_size, n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, n_classes)
            )
            if attention_type == 'V1':
                from module.deeplearning.attention import AttentionModule_V1
                self.att = AttentionModule_V1(input_size=hidden_size, batch_first=True, reduce_sum=True)
            elif attention_type == 'V2':
                from module.deeplearning.attention import AttentionModule_V2
                self.att = AttentionModule_V2(reduce_sum=True)
            elif attention_type == 'V3':
                from module.deeplearning.attention import AttentionModule_V3
                self.att = AttentionModule_V3(input_dim=hidden_size, reduce_sum=True)

    def forward(self, x, need_alpha=False):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        if self.att is not None:
            out, _ = self.rnn(x)
            out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
            # print(out.shape, '...')
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
            h = h[::2, :, :] + h[1::2, :, :]
            # print(h.shape)
            h = h.transpose(0, 1).contiguous().view(x.size(0), -1)
            return self.classifier(h)


class CNN_GAT_GRUClassifier(nn.Module):
    def __init__(self, in_dim=5, adjust_dim=16, hidden_size=32, num_layers=1, dropout=0.1, n_classes=4,
                 bidirectional=False, attention_type: str = None, seq_length=19):
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
        super(CNN_GAT_GRUClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=adjust_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(adjust_dim),
            nn.ReLU(inplace=True)
        )
        self.gat = GraphAttentionLayerV2(num_node=adjust_dim, node_size=seq_length, dropout=dropout)
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
        x = self.gat(self.cnn(x.transpose(1, 2))).transpose(1, 2)
        if self.att is not None:
            out, _ = self.rnn(x)
            # print(out.shape, '...')
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
    m = CNN_GAT_GRUClassifier(in_dim=5, hidden_size=16, num_layers=2, dropout=0.1, n_classes=8, attention_type=None,
                              seq_length=sl)
    a = torch.ones(size=(3, sl, 5), dtype=torch.float32)
    b = m(a, need_alpha=True)
    if isinstance(b, tuple):
        print(*[item.shape for item in b])
    else:
        print(b.shape)
