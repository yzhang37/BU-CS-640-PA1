import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


absPath = os.path.abspath
joinPath = os.path.join
DATA_PATH = absPath(joinPath(os.curdir, "Data"))


def getDataset1(is_linear=True):
    """
    Load Dataset 1, without Cross-validation segmentation.
    :param is_linear: True to load `LinearX/Y.csv`; False to load `NonlinearX/Y.csv`.
    :return:
    """
    d1 = joinPath(DATA_PATH, "dataset1")
    if is_linear:
        X = np.genfromtxt(joinPath(d1, "LinearX.csv"), delimiter=",")
        y = np.genfromtxt(joinPath(d1, "LinearY.csv"), delimiter=",").astype(np.int64)
    else:
        X = np.genfromtxt(joinPath(d1, "NonlinearX.csv"), delimiter=",")
        y = np.genfromtxt(joinPath(d1, "NonlinearY.csv"), delimiter=",").astype(np.int64)
    assert (X.shape[0] == y.shape[0])
    return X, y


def make_cross_validation_data(X, y, fold=10, shuffle=True):
    assert (X.shape[0] == y.shape[0])
    skf = StratifiedKFold(n_splits=fold, shuffle=shuffle)
    for train, test in skf.split(X, y):
        yield X[train], X[test], y[train], y[test]


def make_one_hot(y):
    num_classes = np.max(y) + 1
    num_samples = y.shape[0]
    one_hot_y = np.zeros((num_samples, num_classes))
    for i, y_i in enumerate(y):
        one_hot_y[i][y_i] = 1
    return one_hot_y


def getDataset2():
    d2 = joinPath(DATA_PATH, "dataset2")
    XTrain = np.genfromtxt(joinPath(d2, "Digit_X_train"), delimiter=",")
    yTrain = np.genfromtxt(joinPath(d2, "Digit_y_train"), delimiter=",")
    XTest = np.genfromtxt(joinPath(d2, "Digit_X_test"), delimiter=",")
    yTest = np.genfromtxt(joinPath(d2, "Digit_y_test"), delimiter=",")
    return XTrain, XTest, yTrain, yTest
