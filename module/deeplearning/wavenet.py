import torch
from torch import nn


class CausalConv1d(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation=1, stride=1):
        """
        Input and output sizes will be the same.
            [in_dim+2*pad-dilation*(kernel_size-1)-1]/stride+1
        if stride ==1:
            out_size = in_size+2*pad-dilation*(kernel_size-1)
        finally:
            out_size = in_size
            padding = (kernel_size-1)//2*dilation
        recommend:
            stride = 1
            kernel_size % 2 =1

        :param in_dim:
        :param out_dim:
        :param kernel_size:
        :param dilation:
        :param stride:
        """
        super(CausalConv1d, self).__init__()
        self.add_module('causalConv1d', nn.Conv1d(in_dim, out_dim, kernel_size, padding=(kernel_size-1)//2*dilation,
                                                  stride=stride, dilation=dilation))


class ResidualLayer(nn.Module):
    def __init__(self, residual_dim: int, skip_dim: int, dilation: int):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_dim, residual_dim, kernel_size=3, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_dim, residual_dim, kernel_size=3, dilation=dilation)
        self.res_conv1x1 = nn.Conv1d(residual_dim, residual_dim, kernel_size=1)
        self.skip_conv1x1 = nn.Conv1d(residual_dim, skip_dim, kernel_size=1)

    def forward(self, x):
        filter_value = self.conv_filter(x)
        gate_value = self.conv_gate(x)
        fx = torch.tanh(filter_value) * torch.sigmoid(gate_value)
        fx = self.res_conv1x1(fx)
        skip = self.skip_conv1x1(fx)
        residual = fx + x  # residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_dim: int, skip_dim: int, dilation_depth: int):
        super(DilatedStack, self).__init__()
        self.residual_stack = nn.ModuleList([
            ResidualLayer(residual_dim, skip_dim, 2**layer) for layer in range(dilation_depth)
        ])

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))  # skip =[1, batch, skip_size, seq_len]
        return torch.cat(skips, dim=0), x    # [layers, batch, skip_size, seq_len]


class WaveNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual_dim: int, skip_dim: int, dilation_cycles: int,
                 dilation_depth: int):
        super(WaveNet, self).__init__()
        self.input_conv = CausalConv1d(in_dim, residual_dim, kernel_size=3)
        self.dilated_stacks = nn.ModuleList([
            DilatedStack(residual_dim, skip_dim, dilation_depth) for _ in range(dilation_cycles)
        ])
        self.relu = nn.ReLU()
        self.out_conv1 = nn.Conv1d(skip_dim, skip_dim, kernel_size=1)
        self.final = nn.Conv1d(skip_dim, out_dim, kernel_size=1)

    def forward(self, x, channel_last=True):
        if channel_last:
            x = x.transpose(1, 2)
        x = self.input_conv(x)  # [batch, residual_dim, seq_len]
        skip_connections = []
        for cycle in self.dilated_stacks:
            skips, x = cycle(x)
            skip_connections.append(skips)
        # skip_connections: [total_layers, batch, skip_size, seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)
        # gather all the skip_connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0)  # [batch, skip_size, seq_len]
        out = self.relu(out)
        out = self.out_conv1(out)   # [batch, skip_dim, seq_len]
        return self.final(out)  # [batch, n_classes]


class WaveNetClassifier(nn.Module):
    def __init__(self, in_dim: int, residual_dim: int = 16, skip_dim: int = 32, dilation_cycles: int = 2,
                 dilation_depth: int = 2, n_classes=8):
        super(WaveNetClassifier, self).__init__()
        self.input_conv = CausalConv1d(in_dim, residual_dim, kernel_size=3)
        self.dilated_stacks = nn.ModuleList([
            DilatedStack(residual_dim, skip_dim, dilation_depth) for _ in range(dilation_cycles)
        ])
        self.relu = nn.ReLU()
        self.out_conv1 = nn.Conv1d(skip_dim, skip_dim, kernel_size=1)
        self.final = nn.Linear(skip_dim, n_classes)

    def forward(self, x, channel_last=True):
        if channel_last:
            x = x.transpose(1, 2)
        x = self.input_conv(x)  # [batch, residual_dim, seq_len]
        skip_connections = []
        for cycle in self.dilated_stacks:
            skips, x = cycle(x)
            skip_connections.append(skips)
        # skip_connections: [total_layers, batch, skip_size, seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)
        # gather all the skip_connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0)  # [batch, skip_size, seq_len]
        out = self.relu(out)
        out = self.out_conv1(out)   # [batch, skip_dim, seq_len]
        out = torch.mean(out, dim=2)  # =[batch, skip_dim]
        return self.final(out)  # [batch, n_classes]


if __name__ == '__main__':
    sl = 15
    model = WaveNetClassifier(in_dim=5, n_classes=8, residual_dim=32, skip_dim=256, dilation_cycles=3, dilation_depth=8)
    a = torch.ones(size=(3, sl, 5), dtype=torch.float32)
    b = model(a)
    if isinstance(b, tuple):
        print(*[item.shape for item in b])
    else:
        print(b.shape)
    # print(models)


