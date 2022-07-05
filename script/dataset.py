import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from script.loadData import load_wx


class LithologyIdentificationDataset(Dataset):
    def __init__(self, index: np.ndarray, origin_length: int, data: pd.DataFrame or np.ndarray, window_size: int):
        super(LithologyIdentificationDataset, self).__init__()
        index = index[np.where(index >= window_size // 2)]
        index = index[np.where(index < origin_length - window_size // 2)]
        assert index.size > 0
        self.index = index
        self.data = data if isinstance(data, np.ndarray) else data.values
        self.window_size = window_size
        self.num_f = self.data.shape[1] - 1

    def __getitem__(self, item):
        index = self.index[item]
        label = self.data[index, self.num_f:]
        feature = self.data[index - self.window_size // 2: index - self.window_size // 2 + self.window_size,
                  : self.num_f]
        return torch.from_numpy(feature).to(torch.float32), torch.from_numpy(label).long()

    def __len__(self):
        return self.index.size


def getDataset(path='dataset/SouthSea/WX4-2-1(2).xlsx', window_size=9, test_size=0.3, value_range_correction=True,
               is_normalization=False, random_state=256):
    """
      Load data and split it to train/validation (7/3) items, where category_code is the code of category name.
       col_names ['深度', 'RHOB', 'GR', 'SP', 'CN', 'RT', '_岩性_主名']
        Counter before dealing:  Counter({'泥岩': 8893, '粉砂岩': 970, '细砂岩': 512, '泥灰岩': 104, '中砂岩': 96,
                                            '粗砂岩': 50, '细砾岩': 48, '煤层': 28, '煤': 8})
        Counter after dealing Counter({1: 8893, 0: 970, 2: 512, 6: 104, 3: 96, 4: 50, 5: 48, 7: 36})
        588 items have been corrected...

    :param test_size:
    :param path:
    :param window_size:
    :param value_range_correction:
    :param is_normalization:
    :param random_state:
    :return:
    """
    df, category_code = load_wx(path=path, value_range_correction=value_range_correction)
    origin_length = len(df)
    print('origin_length:', origin_length)
    label = df.pop('_岩性_主名')
    train_index, test_index = train_test_split(pd.DataFrame(np.arange(len(df)).reshape(-1, 1)),
                                               test_size=test_size, shuffle=True, stratify=label,
                                               random_state=random_state)
    train_index = train_index.values.squeeze(-1)
    test_index = test_index.values.squeeze(-1)
    df.pop('深度')
    if is_normalization:
        max_val, min_val = df.iloc[train_index].max(), df.iloc[train_index].min()
        print('max_val-min_val:', max_val - min_val)
        df = df.apply(lambda x: (x - min_val) / (max_val - min_val), axis=1)  # ->[0, 1]
    df = pd.concat([df, label], axis=1)
    train_index = LithologyIdentificationDataset(index=train_index, origin_length=origin_length, data=df,
                                                 window_size=window_size)
    test_index = LithologyIdentificationDataset(index=test_index, origin_length=origin_length, data=df,
                                                window_size=window_size)
    return train_index, test_index


if __name__ == '__main__':
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    a = a[np.where(a >= 3 // 2)]
    print(a[np.where(a < 9 - 3 // 2)])
    print(type(a))
    print(isinstance(a, np.ndarray))
