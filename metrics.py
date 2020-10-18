import numpy as np
import prettytable


class ConfusionMatrix:
    def __init__(self, YTrue, YPredict, labels: list=None, is_one_hot=True):
        self.__labels = None
        if labels is not None:
            self.labels = labels
        YTrue = np.array(YTrue)
        YPredict = np.array(YPredict)
        if is_one_hot:
            assert(YTrue.shape == YPredict.shape)
            # here we assume: row is samples, and column is dimensions.
            self.__cls_num = YTrue.shape[1]
        else:
            YTrue = YTrue.flatten().astype(np.int64)
            YPredict = YPredict.flatten().astype(np.int64)
            self.__cls_num = np.max([np.max(YTrue), np.max(YPredict)]) + 1
        self.__conf_mat = np.zeros((self.__cls_num, self.__cls_num), dtype=np.int64)
        self.__acc = 0
        self.__tp_sum = np.zeros(self.__cls_num).astype(np.int64)
        self.__fp_sum = np.zeros(self.__cls_num).astype(np.int64)
        self.__fn_sum = np.zeros(self.__cls_num).astype(np.int64)

        self.__pr_cache = [None] * self.__cls_num
        self.__rc_cache = [None] * self.__cls_num

        if is_one_hot:
            for yTrue, yPredict in zip(YTrue, YPredict):
                t = np.argmax(yTrue)
                p = np.argmax(yPredict)
                self.__conf_mat[t, p] += 1
                if t == p:
                    self.__acc += 1
                    self.__tp_sum[t] += 1
                else:
                    self.__fp_sum[p] += 1
                    self.__fn_sum[t] += 1
        else:
            for t, p in zip(YTrue, YPredict):
                self.__conf_mat[t, p] += 1
                if t == p:
                    self.__acc += 1
                    self.__tp_sum[t] += 1
                else:
                    self.__fp_sum[p] += 1
                    self.__fn_sum[t] += 1

    @property
    def matrix(self):
        return self.__conf_mat

    def __len__(self):
        return self.__cls_num

    @property
    def accuracy(self):
        return self.__acc / np.sum(self.__conf_mat)

    def get_precision(self, index=None):
        if index is None and self.__cls_num == 2:
            index = 1
        assert (0 <= index < self.__cls_num)
        if self.__pr_cache[index] is None:
            tp = self.__tp_sum[index]
            fp = self.__fp_sum[index]
            z = tp + fp
            self.__pr_cache[index] = tp / z if tp != 0 else 0.0
        return self.__pr_cache[index]

    def get_recall(self, index=None):
        if index is None and self.__cls_num == 2:
            index = 1
        assert (0 <= index < self.__cls_num)
        if self.__rc_cache[index] is None:
            tp = self.__tp_sum[index]
            fn = self.__fn_sum[index]
            z = tp + fn
            self.__rc_cache[index] = tp / z if tp != 0 else 0.0
        return self.__rc_cache[index]

    def get_f1(self, index=None):
        if index is None and self.__cls_num == 2:
            index = 1
        assert (0 <= index < self.__cls_num)
        p = self.get_precision(index)
        r = self.get_recall(index)
        z = p * r
        return 2 * z / (p + r) if z != 0.0 else 0.0

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, new_labels):
        assert(isinstance(new_labels, list))
        self.__labels = new_labels

    @labels.deleter
    def labels(self):
        self.__labels = None

    def __repr__(self):
        return "<ConfMat Size=({0}, {0}), acc={1:0.6f}({2:0.2f}%)>".format(self.__cls_num, self.accuracy, self.accuracy * 100)

    def __str__(self):
        return self.output_matrix()

    def output_matrix(self):
        fields = [r"real \ pred"]
        if self.labels is not None:
            fields.extend(map(lambda x: self.labels[x], range(self.__cls_num)))
        else:
            fields.extend(map(str, range(self.__cls_num)))
        table = prettytable.PrettyTable(fields)
        table.set_style(prettytable.MSWORD_FRIENDLY)
        for i in range(self.__cls_num):
            row = [i] + list(self.matrix[i])
            if self.labels is not None:
                row[0] = self.labels[i]
            table.add_row(row)
        return table.get_string()

    def output_metrics(self):
        if self.__cls_num == 2:
            table = prettytable.PrettyTable()
            table.set_style(prettytable.PLAIN_COLUMNS)
            table.field_names = ["precision:",
                                 "{0:0.6f}".format(self.get_precision()),
                                 "({0:0.2f}%)".format(100* self.get_precision())]
            table.add_row(["recall:",
                           "{0:0.6f}".format(self.get_recall()),
                           "({0:0.2f}%)".format(100* self.get_recall())])
            table.add_row(["f1 score:",
                           "{0:0.6f}".format(self.get_f1()),
                           "({0:0.2f}%)".format(100* self.get_f1())])
            return table.get_string()
        else:
            table = prettytable.PrettyTable()
            table.set_style(prettytable.PLAIN_COLUMNS)
            table.field_names = ["", "precision", "recall", "f1"]
            for i in range(self.__cls_num):
                table.add_row([str(i) if self.labels is None else self.__labels[i],
                               "{0:0.6f}".format(self.get_precision(i)),
                               "{0:0.6f}".format(self.get_recall(i)),
                               "{0:0.6f}".format(self.get_f1(i))])
            return table.get_string()


if __name__ == "__main__":
    def utility_create_matrix(samples, classes):
        np.random.seed()
        dat = np.zeros((samples, classes), dtype=np.int64)
        for i, j in enumerate(np.random.randint(0, classes, (samples,))):
            dat[i, j] = 1
        return dat

    a = utility_create_matrix(5000, 10)
    b = utility_create_matrix(5000, 10)
    cmat = ConfusionMatrix(a, b)
    cmat.labels = ["〇", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    print(cmat)
    print(cmat.output_metrics())
