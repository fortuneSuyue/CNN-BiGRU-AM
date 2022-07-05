import pandas as pd
import numpy as np
from collections import Counter


def load_wx(path='dataset/SouthSea/WX4-2-1(2).xlsx', value_range_correction=False):
    df = pd.read_excel(path)  # 10709 rows x 7 columns
    col_names = list(df.columns)  # ['深度', 'RHOB', 'GR', 'SP', 'CN', 'RT', '_岩性_主名']
    print('col_names', col_names)
    # dealing with lithology
    lithology = list(df[col_names[-1]])
    category_code = {
        '粉砂岩': 0,
        '泥岩': 1,
        '细砂岩': 2,
        '中砂岩': 3,
        '粗砂岩': 4,
        '细砾岩': 5,
        '泥灰岩': 6,
        '煤层': 7
    }
    print('Counter before dealing: ', Counter(lithology))
    for i in range(len(lithology)):
        for key, val in category_code.items():
            if lithology[i] in key:
                lithology[i] = category_code[key]
                break
    print('Counter after dealing', Counter(lithology))
    df.loc[:, col_names[-1]] = lithology
    # Value range correction
    if value_range_correction:
        value_range = {
            'SP': (-20., 100.),
            'GR': (0., 150.),
            'RHOB': (1., 5.),
            'CN': (1., 50.),
            'RT': (1., 500.)
        }
        num_change = 0
        for index, val in value_range.items():
            line = list(df[index])
            for i in range(len(line)):
                if line[i] > val[1]:
                    line[i] = val[1]
                    num_change += 1
                    continue
                if line[i] < val[0]:
                    line[i] = val[0]
                    num_change += 1
            df.loc[:, index] = line
        print(f'{num_change} items have been corrected...')
    return df, category_code


def decoder_category_code(origin_code: dict):
    return {
        value: key for key, value in origin_code.items()
    }


if __name__ == '__main__':
    print(decoder_category_code(load_wx('../dataset/SouthSea/WX4-2-1(2).xlsx', value_range_correction=True)[1]))
