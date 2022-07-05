import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden_size=32, num_layers=1, dropout=0.1, n_classes=4, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        if num_layers == 1:
            dropout = 0.
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * bidirectional * num_layers, hidden_size),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        h = h.transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.classifier(h)


class CNN_LSTMClassifier(nn.Module):
    def __init__(self, in_dim, adjust_dim=16, hidden_size=32, num_layers=1, dropout=0.1, n_classes=8,
                 bidirectional=False):
        super(CNN_LSTMClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=adjust_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(adjust_dim),
            nn.ReLU(inplace=True)
        )
        if num_layers == 1:
            dropout = 0.
        self.rnn = nn.LSTM(input_size=adjust_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * bidirectional * num_layers, hidden_size),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.rnn(x)
        h = h.transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.classifier(h)


if __name__ == '__main__':
    sl = 15
    m = CNN_LSTMClassifier(in_dim=5, hidden_size=16, num_layers=2, dropout=0.1, n_classes=4)
    a = torch.ones(size=(3, sl, 5), dtype=torch.float32)
    b = m(a)
    print(b.shape)
