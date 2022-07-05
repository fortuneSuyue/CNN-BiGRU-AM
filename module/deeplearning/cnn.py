from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self, in_dim=5, channels=None, n_classes=8):
        super(CNN, self).__init__()
        if channels is None or not isinstance(channels, (list, tuple)):
            channels = [16, 32]
        channels.insert(0, in_dim)
        self.features = nn.Sequential(*[self._make_layer(channels[i-1], channels[i]) for i in range(1, len(channels))])
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=7)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*channels[-1], out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=n_classes, bias=True)
        )

    @staticmethod
    def _make_layer(in_c: int, out_c: int):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x, channel_last=True):
        if channel_last:
            x = x.transpose(1, 2)
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    import numpy as np
    m = CNN()
    a = torch.from_numpy(np.random.uniform(-1., 1., size=(3, 15, 5))).float()
    print(m(a).shape)
    print(m)
