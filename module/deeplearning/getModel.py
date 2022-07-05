def get_classifier(tag: str, in_dim=5, n_classes=8, dropout=0.4, num_layers=2, base_dim=16, bidirectional=True,
                   seq_length=15):
    """
    ['CNN', 'BiLSTM', 'CNN-BiGRU', 'BiGRU', 'CNN-BiGRU', 'CNN-BiGRU-AM',
    'BiT-BiGRU-AM', 'WaveNet', 'TCN', 'Transformer', 'CnnTransformer']

    :param attention_type:
    :param tag:
    :param in_dim: 5
    :param n_classes: 8
    :param dropout: 0.4
    :param num_layers: 2
    :param base_dim: 16
    :param bidirectional: True
    :param seq_length: 15, just for transformer models.
    :return:
    """
    if tag.upper() == 'CNN'.upper():
        from module.deeplearning.cnn import CNN
        return CNN(in_dim=in_dim, n_classes=n_classes, channels=[base_dim * 2 ** num for num in range(num_layers)])
    elif tag.upper() == 'RNN'.upper():
        from module.deeplearning.rnn import VanillaRNNClassifier
        return VanillaRNNClassifier(in_dim=in_dim, hidden_size=base_dim * 2 ** (num_layers - 1), num_layers=num_layers,
                                    dropout=dropout, n_classes=n_classes, bidirectional=bidirectional)
    if tag.upper() in ('BiLSTM'.upper(), 'LSTM'):
        from module.deeplearning.lstm import LSTMClassifier
        return LSTMClassifier(in_dim=in_dim, hidden_size=base_dim * 2 ** (num_layers - 1), num_layers=num_layers,
                              dropout=dropout, n_classes=n_classes, bidirectional=bidirectional)
    if tag.upper() in ('CNN-BiLSTM'.upper(), 'CNN-LSTM'):
        from module.deeplearning.lstm import CNN_LSTMClassifier
        return CNN_LSTMClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=16 * 2 ** (num_layers - 1),
                                  num_layers=num_layers,
                                  dropout=dropout, n_classes=n_classes, bidirectional=bidirectional)

    if tag.upper() in ('BiGRU'.upper(), 'GRU'):
        from module.deeplearning.gru import GRUClassifier
        return GRUClassifier(in_dim=in_dim, hidden_size=base_dim * 2 ** (num_layers - 1), num_layers=num_layers,
                             dropout=dropout, n_classes=n_classes, bidirectional=bidirectional)
    if tag.upper() in ('CNN-BiGRU'.upper(), 'CNN-GRU'):
        from module.deeplearning.gru import CNN_GRUClassifier
        return CNN_GRUClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                                 num_layers=num_layers,
                                 dropout=dropout, n_classes=n_classes, bidirectional=bidirectional)
    if tag.upper() in ('BiGRU-AM'.upper(), 'GRU-AM'):
        from module.deeplearning.gru import GRUClassifier
        return GRUClassifier(in_dim=in_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                             num_layers=num_layers, dropout=dropout, n_classes=n_classes,
                             bidirectional=bidirectional,
                             attention_type=tag.upper()[-2:] if tag.upper()[-2:] in ['V1', 'V2', 'V3'] else 'V1')
    if 'CNN-BiGRU-AM'.upper() in tag.upper() or 'CNN-GRU-AM'.upper() in tag.upper():
        from module.deeplearning.gru import CNN_GRUClassifier
        return CNN_GRUClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                                 num_layers=num_layers, dropout=dropout, n_classes=n_classes,
                                 bidirectional=bidirectional,
                                 attention_type=tag.upper()[-2:] if tag.upper()[-2:] in ['V1', 'V2', 'V3'] else 'V1')
    if 'CNN-GAT-BiGRU-AM'.upper() in tag.upper() or 'CNN-GAT-GRU-AM'.upper() in tag.upper():
        from module.deeplearning.gru import CNN_GAT_GRUClassifier
        return CNN_GAT_GRUClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                                     num_layers=num_layers, dropout=dropout, n_classes=n_classes,
                                     bidirectional=bidirectional,
                                     attention_type=tag.upper()[-2:] if tag.upper()[-2:] in ['V1', 'V2',
                                                                                             'V3'] else 'V1',
                                     seq_length=seq_length)
    if 'CNN-BiGRUPlus-AM'.upper() in tag.upper():
        from module.deeplearning.gru import CNN_BiGRUPlusClassifier
        return CNN_BiGRUPlusClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                                       num_layers=num_layers, dropout=dropout, n_classes=n_classes,
                                       attention_type=tag.upper()[-2:] if tag.upper()[-2:] in ['V1', 'V2',
                                                                                               'V3'] else 'V1')
    if 'BiT-BiGRU-AM'.upper() in tag.upper():
        from module.deeplearning.biTBiGRUAM import BiT_GRUClassifier
        return BiT_GRUClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                                 num_layers=num_layers, dropout=dropout, n_classes=n_classes,
                                 bidirectional=bidirectional,
                                 attention_type=tag.upper()[-2:] if tag.upper()[-2:] in ['V1', 'V2', 'V3'] else 'V1')
    if 'BiT-BiGRU'.upper() in tag.upper() or 'BiT-GRU'.upper():
        from module.deeplearning.biTBiGRUAM import BiT_GRUClassifier
        return BiT_GRUClassifier(in_dim=in_dim, adjust_dim=base_dim, hidden_size=base_dim * 2 ** (num_layers - 1),
                                 num_layers=num_layers, dropout=dropout, n_classes=n_classes,
                                 bidirectional=bidirectional)
    if tag.upper() == 'WaveNet'.upper():
        from module.deeplearning.wavenet import WaveNetClassifier
        return WaveNetClassifier(in_dim=in_dim, n_classes=n_classes, residual_dim=base_dim, dilation_cycles=num_layers,
                                 skip_dim=base_dim * 2 ** (num_layers - 1), dilation_depth=num_layers)
    if tag.upper() == 'TCN'.upper():
        from module.deeplearning.tcn import TemporalConvNetClassifier
        return TemporalConvNetClassifier(input_dim=in_dim,
                                         num_channels=[base_dim * 2 ** num for num in range(num_layers)],
                                         kernel_size=2, dropout=dropout, n_classes=n_classes)
    if tag.upper() == 'CnnTransformer'.upper():
        from module.deeplearning.transformer import CNNTransformer
        return CNNTransformer(input_dim=in_dim, adjust_dim=base_dim, enc_num_layers=num_layers, dropout=dropout,
                              seq_length=seq_length, n_classes=n_classes, using_pe=True, n_head=8)

    if tag.upper() == 'Transformer'.upper():
        from module.deeplearning.transformer import TransformerClassifier
        return TransformerClassifier(in_dim=in_dim, enc_num_layers=num_layers, n_head=in_dim, n_classes=n_classes,
                                     dropout=dropout, seq_length=seq_length)
    raise ValueError(f'{tag} is valid. No such model...')
