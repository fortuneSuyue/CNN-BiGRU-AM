import torch
from torch import nn


class FNN(nn.Sequential):
    def __init__(self, in_c, layers, hidden=None, dropout=0.1, n_classes=4):
        super(FNN, self).__init__()
        if hidden is None:
            hidden = [in_c]
            for i in range(layers):
                hidden.append(pow(2, i) * 16)
        for i in range(1, len(hidden)):
            self.add_module('layer{}'.format(i - 1), nn.Linear(hidden[i - 1], hidden[i]))
            self.add_module('dropout{}'.format(i - 1), nn.Dropout(p=dropout))
        self.add_module('classifier', nn.Linear(hidden[-1], n_classes))


if __name__ == '__main__':
    m = FNN(in_c=5, layers=3)
    a = torch.ones(size=(3, 5), dtype=torch.float32)
    b = m(a)
    print(b.shape)
