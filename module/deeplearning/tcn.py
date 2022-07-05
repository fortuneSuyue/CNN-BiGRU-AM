import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int, padding=1, dropout=0.2, stride=1):
        """
        The same as a residual block with weight_norm from "Weight Normalization: A Simple Reparameterization to
        Accelerate Training of Deep Neural Networks" (https://arxiv.org/abs/1602.07868).
        :param in_dim:
        :param out_dim:
        :param kernel_size:
        :param dilation:
        :param padding:
        :param dropout:
        :param stride:
        """
        super(TemporalBlock, self).__init__()
        self.main_branch = nn.Sequential(
            weight_norm(nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.sub_branch = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else None
        self.relu = nn.ReLU(inplace=True)
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.main_branch(x)
        if self.sub_branch is not None:  # residual channel adjustment.
            x = self.sub_branch(x)
        return self.relu(out + x)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim=5, num_channels: list or tuple = None, kernel_size=2, dropout=0.2):
        """
        Lea, Colin, et al. “Temporal convolutional networks: A unified approach to action segmentation.” European
        Conference on Computer Vision. Springer, Cham, 2016.

        :param num_channels: including the in_dim and the channels of each layer (residual block).
        :param kernel_size:
        :param dropout:
        """
        super(TemporalConvNet, self).__init__()
        if num_channels is None or not isinstance(num_channels, (list, tuple)):
            num_channels = [16, 32]
        num_channels.insert(0, input_dim)
        modules = []
        for i in range(1, len(num_channels)):
            dilation_size = 2 ** (i - 1)
            modules.append(TemporalBlock(in_dim=num_channels[i - 1], out_dim=num_channels[i],
                                         kernel_size=kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.tcn_modules = nn.Sequential(*modules)

    def forward(self, x, channel_last=True):
        if channel_last:
            x = x.transpose(1, 2)
        return self.tcn_modules(x)


class TemporalConvNetClassifier(nn.Module):
    def __init__(self, input_dim=5, num_channels: list or tuple = None, kernel_size=2, dropout=0.2, n_classes=8):
        """
        Lea, Colin, et al. “Temporal convolutional networks: A unified approach to action segmentation.” European
        Conference on Computer Vision. Springer, Cham, 2016.

        :param num_channels: including the in_dim and the channels of each layer (residual block).
        :param kernel_size:
        :param dropout:
        """
        super(TemporalConvNetClassifier, self).__init__()
        if num_channels is None or not isinstance(num_channels, (list, tuple)):
            num_channels = [16, 32]
        num_channels.insert(0, input_dim)
        modules = []
        for i in range(1, len(num_channels)):
            dilation_size = 2 ** (i - 1)
            modules.append(TemporalBlock(in_dim=num_channels[i - 1], out_dim=num_channels[i],
                                         kernel_size=kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.tcn_modules = nn.Sequential(*modules)
        self.tcn_modules.add_module('avg_pool', nn.AdaptiveAvgPool1d(output_size=7))
        self.tcn_modules.add_module('flatten', nn.Flatten())
        self.tcn_modules.add_module('classifier', nn.Sequential(
            nn.Linear(in_features=7 * num_channels[-1], out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=n_classes, bias=True)
        ))

    def forward(self, x, channel_last=True):
        if channel_last:
            x = x.transpose(1, 2)
        return self.tcn_modules(x)


if __name__ == '__main__':
    sl = 15
    model = TemporalConvNetClassifier(input_dim=5, num_channels=[16, 32])
    a = torch.ones(size=(3, sl, 5), dtype=torch.float32)
    b = model(a)
    if isinstance(b, tuple):
        print(*[item.shape for item in b])
    else:
        print(b.shape)
    # print(models)
