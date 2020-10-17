import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


class DataDepot:
    def __init__(self, filenames: list):
        self.x_dim = None
        self.y_dim = None
        self.all_records_num = None
        self.label_data = dict()
        self.index_data = []
        self.full_data_x = None
        self.full_data_y = None
        has_data = False
        for file_id, filename in enumerate(filenames):
            if isinstance(filename, (tuple, list)):
                if len(filename) >= 3:
                    name_x, name_y, label = filename
                    label = str(label)
                elif len(filename) == 2:
                    name_x, name_y = filename
                    label = None
                else:
                    raise Exception("Incomplete data.")
                dat_x = np.genfromtxt(name_x, delimiter=",")
                dat_y = np.genfromtxt(name_y, delimiter=",")
                if dat_y.ndim == 1:
                    dat_y = dat_y[:, np.newaxis]
                if dat_x.shape[0] != dat_y.shape[0]:
                    raise Exception("In file<{0}>, X and Y has different records count.".format(file_id))
                if not has_data:
                    self.x_dim = dat_x.shape[1:]
                    self.y_dim = dat_y.shape[1:]
                    self.all_records_num = dat_x.shape[0]
                    self.full_data_x = dat_x
                    self.full_data_y = dat_y
                else:
                    if dat_x.shape[1:] != self.x_dim or dat_y.shape[1:] != self.y_dim:
                        raise Exception("In file<{0}>, X or Y data dim doesn't match with other files.".format(file_id))
                    self.all_records_num += dat_x.shape[0]
                    self.full_data_x = np.vstack(self.full_data_x, dat_x)
                    self.full_data_y = np.vstack(self.full_data_y, dat_y)

                cur_data_dict = {"X": dat_x, "Y": dat_y}
                self.index_data.append(cur_data_dict)
                if label is not None:
                    self.label_data[label] = cur_data_dict
            else:
                raise Exception("<{0}>, filename type not supported.".format(file_id))

    def get(self, label=None):
        if label is None:
            return self.full_data_x, self.full_data_y
        elif type(label) == int:
            dat = self.index_data[label]
            return dat["X"], dat["Y"]
        elif type(label) == str:
            dat = self.label_data[label]
            return dat["X"], dat["Y"]
        else:
            raise Exception("type {0} is not supported.".format(type(label)))

    def getStratifiedKFold(self, fold=10, shuffle=True):
        skf = StratifiedKFold(n_splits=fold, shuffle=shuffle)
        for train, test in skf.split(self.full_data_x, self.full_data_y):
            yield self.full_data_x[train], self.full_data_y[train], self.full_data_x[test], self.full_data_y[test]


absPath = os.path.abspath
joinPath = os.path.join
DATA = absPath(joinPath(os.curdir, "Data"))
D1 = joinPath(DATA, "dataset1")
D2 = joinPath(DATA, "dataset2")


def getD1Linear():
    d1Linear = DataDepot([(joinPath(D1, "LinearX.csv"), joinPath(D1, "LinearY.csv"))])
    return d1Linear


def getD1NonLinear():
    d1NonLinear = DataDepot([(joinPath(D1, "NonlinearX.csv"), joinPath(D1, "NonlinearY.csv"))])
    return d1NonLinear


def getD2():
    return DataDepot([
        (joinPath(D2, "Digit_X_train.csv"), joinPath(D2, "Digit_y_train.csv"), "train"),
        (joinPath(D2, "Digit_X_test.csv"), joinPath(D2, "Digit_y_test.csv"), "test"),
    ])


def make_one_hot(y):
    num_classes = np.max(y) + 1
    num_samples = y.shape[0]
    one_hot_y = np.zeros((num_samples, num_classes))
    for i, y_i in enumerate(y):
        one_hot_y[i][y_i] = 1
    return one_hot_y
