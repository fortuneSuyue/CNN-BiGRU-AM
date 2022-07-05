import logging
import math
import os
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tensorboard_logger import Logger
from torch import optim
from torch.utils.data import DataLoader

from module.deeplearning.getModel import get_classifier
from script.dataset import LithologyIdentificationDataset
from script.loadData import load_wx
from script.loadTFEvent import load_tf_event
from script.configurationInitializer import init_random_seed, init_matplotlib_style
from script.train import BasicTrainer
from module.deeplearning.loss.focalLoss import FocalLoss


def main(window_size=15, model_tag='CNN', use_cuda=False, show_pic=False, h_name='DESKTOP-JCJIVK8'):
    experiment_id = int(time.time())
    experiment_id += window_size * 10 ** (int(math.log10(experiment_id)) + 1)
    is_normalization = False
    num_epoch = 500
    if 'Transformer' in model_tag:
        num_epoch = 1000
    lr = 0.001
    batch_size = 128
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = FocalLoss()
    metrics_keys = ['loss', 'Acc', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC']
    logs_path = f'tb_loggers/{model_tag}/{experiment_id}'
    tb_logger = Logger(logdir=logs_path, dummy_time=experiment_id)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logs_path, f'training.logs.{experiment_id}.{h_name}.log')),
            logging.StreamHandler()
        ])
    print = logging.info
    print(f'experiment_id: {experiment_id}')
    print(f'host_name: {host_name}')

    print(model_tag)
    print(logs_path)
    print(f'random_state: {random_state}')
    print(f'lr: {lr}; ce, adam, {batch_size}, {num_epoch}')
    print(f'is_normalization: {is_normalization}')
    print(f'window_size: {window_size}')

    # Load data and split it to train/validation (7/3) items, where category_code is the code of category name.
    '''col_names ['深度', 'RHOB', 'GR', 'SP', 'CN', 'RT', '_岩性_主名']
        Counter before dealing:  Counter({'泥岩': 8893, '粉砂岩': 970, '细砂岩': 512, '泥灰岩': 104, '中砂岩': 96, 
                                            '粗砂岩': 50, '细砾岩': 48, '煤层': 28, '煤': 8})
        Counter after dealing Counter({1: 8893, 0: 970, 2: 512, 6: 104, 3: 96, 4: 50, 5: 48, 7: 36})
        588 items have been corrected...
    '''
    df, category_code = load_wx(path='dataset/SouthSea/WX4-2-1(2).xlsx', value_range_correction=True)
    label = df.pop('_岩性_主名')
    train_index, test_index = train_test_split(pd.DataFrame(np.arange(len(df)).reshape(-1, 1)),
                                               test_size=0.3, shuffle=True, stratify=label, random_state=random_state)
    train_index = train_index.values.squeeze(-1)
    test_index = test_index.values.squeeze(-1)
    df.pop('深度')
    if is_normalization:
        max_val, min_val = df.iloc[train_index].max(), df.iloc[train_index].min()
        print('max_val-min_val:', max_val - min_val)
        df = df.apply(lambda x: (x - min_val) / (max_val - min_val), axis=1)  # 归一化
    df = pd.concat([df, label], axis=1)
    train_data = LithologyIdentificationDataset(index=train_index, origin_length=10709, data=df,
                                                window_size=window_size)
    test_data = LithologyIdentificationDataset(index=test_index, origin_length=10709, data=df, window_size=window_size)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    trainer = BasicTrainer(loss_func=loss_fn, n_classes=8, cuda=use_cuda)
    if 'Bi' in model_tag:
        print(f'Bi, {model_tag}')
        m = get_classifier(tag=model_tag, in_dim=5, n_classes=8, dropout=0.4, num_layers=2, base_dim=16,
                           bidirectional=True, seq_length=window_size)
    else:
        print(f'Si, {model_tag}')
        m = get_classifier(tag=model_tag, in_dim=5, n_classes=8, dropout=0.4, num_layers=2, base_dim=16,
                           bidirectional=False, seq_length=window_size)
    optimiser = optim.Adam(m.parameters(), lr=lr)
    for i in range(num_epoch):
        for procedure in ['train', 'val']:
            if procedure == 'train':
                common_loss, evaluation_values = trainer.train_per_epoch(train_data, m, optimiser, cuda=use_cuda,
                                                                         info=f'epoch: {i + 1}/{num_epoch}')
            else:
                common_loss, evaluation_values = trainer.validation(test_data, m, cuda=use_cuda, val_test=procedure)
            print(f'{model_tag} epoch: {i + 1}/{num_epoch}  {procedure}  '
                  f'loss: {common_loss}, {metrics_keys[1]}: {evaluation_values[metrics_keys[1]]}')
            tb_logger.log_value(f'{procedure}_loss', value=common_loss, step=i + 1)
            for key, val in evaluation_values.items():
                tb_logger.log_value('_'.join((procedure, key)), value=val, step=i + 1)
        if num_epoch - i <= 10:  # save last 10
            torch.save({
                'loss': common_loss,
                'evaluation': evaluation_values,
                'epoch': i + 1,
                'state_dict': m.state_dict()
            }, f'{logs_path}/checkpoint_epoch_{i + 1}.pth.tar')

    evaluation_scalars = load_tf_event(path=logs_path, event_id=experiment_id, host_name=h_name)
    plt.rcParams['font.sans-serif'] = ['Arial']
    color_dict = {
        'train': 'red',
        'val': 'blue',
        'test': 'green'
    }
    for evaluation_key in metrics_keys:
        plt.figure(dpi=300)
        plt.xlabel('Epoch')
        plt.ylabel(evaluation_key)
        for procedure in ['train', 'val']:
            evaluation_scalar = evaluation_scalars['_'.join((procedure, evaluation_key))]
            steps, values = evaluation_scalar['steps'], evaluation_scalar['values']
            plt.plot(steps, values, label=f'{procedure} {evaluation_key}', color=color_dict[procedure], linewidth=0.8)
        plt.legend()
        plt.savefig(os.path.join(logs_path, f'{evaluation_key}.tif'), bbox_inches='tight')
        if show_pic:
            plt.show()


if __name__ == '__main__':
    # key: train, val test
    # Hyperparameters
    random_state = 256
    init_random_seed(random_state)
    init_matplotlib_style(style='science')
    host_name = socket.gethostname()
    model_tag_list = ['CNN', 'RNN', 'GRU', 'BiGRU', 'CNN-GRU', 'CNN-BiGRU', 'BiGRU-AM', 'TCN',
                      'CNN-BiGRU-AM']
    # for w_size in [23]:
    #     for i in range(len(model_tag_list)):
    #         main(window_size=w_size, model_tag=model_tag_list[i], h_name=host_name)
    #         print(model_tag_list[i], w_size)
    # 5, 7, 9, 11, 13, 15, 17, 19, 21, 23

    for w_size in [5, 7, 9, 11, 13, 15]:
        for model_tag in ['CNN-GAT-BiGRU-AM']:
            print(model_tag, w_size)
            main(window_size=w_size, model_tag=model_tag, h_name=host_name)
