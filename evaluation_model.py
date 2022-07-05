import socket

import torch
from torch.utils.data import DataLoader

from module.deeplearning.getModel import get_classifier
from script.dataset import getDataset
from script.configurationInitializer import init_random_seed
from script.train import BasicTrainer


def reload_evaluation(window_size=15, model_tag='CNN', use_cuda=False, experiment_id='0', return_all=True,
                      random_state=256):
    is_normalization = False
    batch_size = 256
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = FocalLoss()
    logs_path = f'tb_loggers/{model_tag}/{experiment_id}'
    num_epoch = 500
    if 'Transformer' in model_tag:
        num_epoch = 1000

    print(f'experiment_id: {experiment_id}')

    print(model_tag)
    print(logs_path)
    print(f'is_normalization: {is_normalization}')
    print(f'window_size: {window_size}')

    train_data, test_data = getDataset(window_size=window_size, test_size=0.3, value_range_correction=True,
                                       is_normalization=is_normalization, random_state=random_state)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_data = DataLoader(test_data, batch_size=len(test_data), shuffle=False, drop_last=False)
    trainer = BasicTrainer(loss_func=loss_fn, n_classes=8, cuda=use_cuda, need_test=False)
    results = []
    for m_id in range(10):
        if 'Bi' in model_tag:
            print(f'Bi, {model_tag}')
            m = get_classifier(tag=model_tag, in_dim=5, n_classes=8, dropout=0.4, num_layers=2, base_dim=16,
                               bidirectional=True, seq_length=window_size)
        else:
            print(f'Si, {model_tag}')
            m = get_classifier(tag=model_tag, in_dim=5, n_classes=8, dropout=0.4, num_layers=2, base_dim=16,
                               bidirectional=False, seq_length=window_size)
        m.load_state_dict(torch.load(f'{logs_path}/checkpoint_epoch_{num_epoch - 9 + m_id}.pth.tar')['state_dict'])
        m.cpu().eval()
        results.append(trainer.validation(test_data, m, cuda=use_cuda, val_test='val'))
    if return_all:
        return results
    else:
        return results[-1][-1]


if __name__ == '__main__':
    # key: train, val test
    # Hyperparameters
    rand_state = 256
    init_random_seed(rand_state)
    host_name = socket.gethostname()
    model_tag_list = ['CNN', 'BiLSTM', 'CNN-BiLSTM', 'BiGRU', 'CNN-BiGRU', 'CNN-BiGRU-AM', 'WaveNet', 'TCN',
                      'CnnTransformer']
    w_size = 5
    valuable_metrics = ['Acc', 'Precision', 'Recall', 'F1']
    for _, item in reload_evaluation(window_size=w_size, model_tag='BiGRU', use_cuda=False,
                                     experiment_id='51645877632'):
        print({val_metrics: item[val_metrics] for val_metrics in valuable_metrics})
