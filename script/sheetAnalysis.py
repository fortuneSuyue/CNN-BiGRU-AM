import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from script.configurationInitializer import init_matplotlib_style

matplotlib.rcParams['font.sans-serif'] = ['Arial']


def analysis(path='../dataset/SouthSea/WX4-2-1(2).xlsx', use_plot=False, use_save=False,
             save_path='../procedure/pic.tif', need_depth=False):
    data = pd.read_excel(path)
    depth = data.values[:, 0]
    lithology = data.values[:, -1]
    print(np.unique(lithology))
    count = {}
    for k in lithology:
        if k in count.keys():
            count[k] = count[k] + 1
        else:
            count[k] = 1
    print(count)
    # input('stop***********************')

    features = data.values[:, 1: data.shape[1] - 1]
    features_name = data.keys().values[1: 6]
    print(features_name)
    h, w = features.shape
    print(h, w)
    plt.figure(dpi=300)
    for c in range(w):
        top = 1.0
        bottom = 0.0
        if features_name[c] == 'SP':
            top = 100.0
            bottom = -20.0
        elif features_name[c] == 'GR':
            top = 150.0
            bottom = 0.0
        elif features_name[c] == 'RHOB':
            top = 5.0
            bottom = 1.0
        elif features_name[c] == 'CN':
            top = 50.0
            bottom = 1.0
        elif features_name[c] == 'RT':
            top = 500.0
            bottom = 1.0
        for i in range(h):
            if features[i, c] < bottom:
                features[i, c] = bottom
            elif features[i, c] > top:
                features[i, c] = top

        print(features_name[c], features[:, c].max(), features[:, c].min(), features[:, c].mean())
        plt.subplot(3, 2, c + 1)
        plt.ylabel(features_name[c])
        plt.plot(depth, features[:, c])
    if use_save:
        plt.tight_layout()
        plt.savefig(save_path)
    if use_plot:
        print('...')
        plt.show()
    category = np.zeros(shape=(h, 1), dtype=np.int32)
    count = {}
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
    for i in range(h):
        for key, val in category_code.items():
            if lithology[i] in key:
                lithology[i] = '煤层'
                category[i][0] = category_code[key]
                break
            if lithology[i] not in count:
                count[lithology[i]] = 1
            else:
                count[lithology[i]] += 1
    print(np.unique(lithology))
    print(count)
    if need_depth:
        return (features, category), depth
    return features, category


def analysis_Cha1(path='../dataset/NorthEast/Cha 1.xlsx', use_plot=False):
    data = pd.read_excel(path)
    keys = data.keys().values[: 7]
    lithology = data.values[:, 7].astype(int)
    data = data.values[:, : 7]
    print(np.unique(lithology))
    print(data.shape, type(data))
    print(keys)
    print(1884.75 - 0.125 * (data.shape[0]))
    x = [1884.75 - 0.125 * (data.shape[0] - i) for i in range(data.shape[0])]
    plt.figure()
    for i in range(len(keys)):
        if keys[i] == 'GR':
            for j in range(data.shape[0]):
                if data[j][i] > 150.:
                    data[j][i] = 150.
        print(keys[i], data[:, i].max(), data[:, i].min(), data[:, i].mean())
        plt.subplot(3, 3, i + 1)
        plt.ylabel(keys[i])
        plt.xlabel('depth')
        plt.plot(x, data[:, i])
    if use_plot:
        plt.show()
    category = np.zeros(shape=(data.shape[0], 1), dtype=np.int32)
    count = {}
    for i in range(data.shape[0]):
        if lithology[i] == 7:
            category[i][0] = 5
        elif lithology[i] == 16:
            category[i][0] = 6
        else:
            category[i][0] = lithology[i]
        if category[i][0] not in count:
            count[category[i][0]] = 1
        else:
            count[category[i][0]] += 1
    print(np.unique(category))
    print(count)
    return np.concatenate([data[:, :4], data[:, 5:]], 1), category


if __name__ == '__main__':
    init_matplotlib_style(style='science')
    analysis(use_plot=True, use_save=False)
    # analysis_Cha1(use_plot=True)
    # data, category = analysis_Cha1(use_plot=True)
    # print(data.shape, category.shape)

    # np.save('../dataset/SouthSea/data_SouthSea.npy', data)
    # np.save('../dataset/SouthSea/category_SouthSea.npy', category)
    # a = np.load('../dataset/SouthSea/data_SouthSea.npy', allow_pickle=True)
    # b = np.load('../dataset/SouthSea/category_SouthSea.npy', allow_pickle=True)
    # print(a.shape, b.shape)
    # print(a==data, b==category)
