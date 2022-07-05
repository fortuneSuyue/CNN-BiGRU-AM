import torch
from torch import nn


class VanillaRNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_size=32, num_layers=1, dropout=0.1, n_classes=4, bidirectional=False):
        super(VanillaRNNClassifier, self).__init__()
        if num_layers == 1:
            dropout = 0.
        self.rnn = nn.RNN(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
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
        _, h = self.rnn(x)
        h = h.transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.classifier(h)


if __name__ == '__main__':
    sl = 15
    m = VanillaRNNClassifier(in_dim=5, hidden_size=16, num_layers=2, dropout=0.1, n_classes=8, bidirectional=False)
    a = torch.ones(size=(3, sl, 5), dtype=torch.float32)
    b = m(a)
    if isinstance(b, tuple):
        print(*[item.shape for item in b])
    else:
        print(b.shape)
