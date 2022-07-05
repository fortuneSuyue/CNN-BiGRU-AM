import math

from torch import nn
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim, enc_num_layers=4, n_head=8, dropout=0.1, seq_length=15, n_classes=4):
        super(TransformerClassifier, self).__init__()
        self.pos_encoder = PositionalEncoding(in_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_head,
                                                                        dropout=dropout,
                                                                        dim_feedforward=4 * in_dim),
                                             num_layers=enc_num_layers)
        self.decoder = nn.Linear(seq_length, 1)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        x = self.pos_encoder(x.transpose(0, 1))
        x = self.encoder(x).transpose(0, 1).transpose(1, 2)
        x = self.decoder(x).squeeze(-1)
        return self.fc(x)


class CNNTransformer(nn.Module):
    def __init__(self, input_dim, adjust_dim=16, kernel_size=3, padding=1,
                 enc_num_layers=4, n_head=8, dropout=0.1, seq_length=15, n_classes=8, using_pe=True):
        """
        using CNN to adjust the input dimension.
        :param input_dim:
        :param adjust_dim:
        :param kernel_size:
        :param padding:
        :param enc_num_layers:
        :param n_head:
        :param dropout:
        :param seq_length:
        :param n_classes:
        """
        super(CNNTransformer, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=adjust_dim, kernel_size=kernel_size, padding=padding,
                             bias=False)
        self.bn = nn.BatchNorm1d(adjust_dim)
        self.relu = nn.ReLU()
        if using_pe:
            self.pos_encoder = PositionalEncoding(adjust_dim)
        else:
            self.pos_encoder = None
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=adjust_dim, nhead=n_head,
                                                                        dropout=dropout,
                                                                        dim_feedforward=4 * input_dim),
                                             num_layers=enc_num_layers)
        self.decoder = nn.Linear(seq_length+padding*2-kernel_size+1, 1)
        self.fc = nn.Linear(adjust_dim, n_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.cnn(x.transpose(1, 2)))).transpose(1, 2)
        x = x.transpose(0, 1)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.encoder(x).transpose(0, 1).transpose(1, 2)
        x = self.decoder(x).squeeze(-1)
        return self.fc(x)


if __name__ == '__main__':
    import numpy as np

    a = torch.from_numpy(np.random.uniform(-1., 1., size=(3, 5, 5))).float()
    m = TransformerClassifier(in_dim=5, enc_num_layers=6, n_head=5, n_classes=4, seq_length=5)(a)
    # m = CNNTransformer(input_dim=5, adjust_dim=32, kernel_size=3, padding=0, enc_num_layers=6, n_head=8,
    #                    seq_length=5, n_classes=8, using_pe=True)(a)
    print(m.shape)
