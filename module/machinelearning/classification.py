from datetime import datetime

import joblib
import sklearn.linear_model as sk_linear
import sklearn.model_selection as sk_model_selection
import sklearn.naive_bayes as sk_bayes
import sklearn.neural_network as sk_nn
import sklearn.tree as sk_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


# 模型保存
def save_model(model, is_save=False):
    if is_save:
        path = input('Please input the path while file suffix will be automatically given: ')
        joblib.dump(model, path)  # 保存
    else:
        print('Fail to save the models...')


def commonProcess(model, x_train, y_train, x_test=None, y_test=None, is_save=False):
    model.fit(x_train, y_train)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def svc(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = make_pipeline(StandardScaler(),
                          SVC(probability=True, gamma='auto'))
    model.fit(x_train, y_train)
    # models = SVC(gamma=87.1, C=98.9, probability=True)
    model.fit(x_train, y_train)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def bestSVC0(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': np.linspace(50, 200, 100), 'gamma': np.linspace(0.01, 150, 100)}
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1) 
    grid_search.fit(x_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    return commonProcess(model, x_train, y_train, x_test, y_test, is_save)


def bestSVC1(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': np.linspace(0.01, 150, 10), 'gamma': np.linspace(200, 400, 10)}
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    return commonProcess(model, x_train, y_train, x_test, y_test, is_save)


def knn(x_train, y_train, x_test=None, y_test=None, n_neighbors=3, is_save=False):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    model.fit(x_train, y_train)
    print('KNN models:', model)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def logisticsRegression(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = sk_linear.LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=-1,
                                         random_state=20, fit_intercept=True, max_iter=10000)
    model.fit(x_train, y_train)  # 对模型进行训练
    print(model)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def naiveBayes(x_train, y_train, x_test=None, y_test=None, para_type=3, is_save=False):
    if type == 1:
        model = sk_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)  # 多项式分布的朴素贝叶斯
    elif type == 2:
        model = sk_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)  # 伯努利分布的朴素贝叶斯
    else:
        model = sk_bayes.GaussianNB()  # 高斯分布的朴素贝叶斯
    model.fit(x_train, y_train)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def ID3_tree(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = sk_tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                           max_features=None, max_leaf_nodes=None, min_impurity_decrease=0)
    model.fit(x_train, y_train)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def GPR(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = GaussianProcessClassifier(random_state=1)
    model.fit(x_train, y_train)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def decisionTree(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = DecisionTreeClassifier(random_state=1)
    return commonProcess(model, x_train, y_train, x_test, y_test, is_save)


def RF(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = RandomForestClassifier(random_state=1)
    return commonProcess(model, x_train, y_train, x_test, y_test, is_save)


def GBDT(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = GradientBoostingClassifier(random_state=256)
    return commonProcess(model, x_train, y_train, x_test, y_test, is_save)


def AdaBoost(x_train, y_train, x_test=None, y_test=None, is_save=False):
    model = AdaBoostClassifier(random_state=1)
    return commonProcess(model, x_train, y_train, x_test, y_test, is_save)


def nn(x_train, y_train, x_test=None, y_test=None, activation='tanh', epochs=2000, is_save=False):
    model = sk_nn.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
                                learning_rate_init=0.001, max_iter=epochs)
    model.fit(x_train, y_train)
    print('train_dataset_acc：', model.score(x_train, y_train))
    if x_test is not None:
        print('test_dataset_acc：', model.score(x_test, y_test))
    save_model(model, is_save)
    return model


def cross_val(x, y, model):
    acc = sk_model_selection.cross_val_score(model, x, y=y, scoring=None, cv=10, n_jobs=-1)
    print('交叉验证结果:', acc)
