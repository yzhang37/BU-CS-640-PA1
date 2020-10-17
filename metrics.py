import numpy as np


def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM, labels, label2id : numpy matrix, labels, labels2id
        The confusion matrix.
    """
    # first we fetch the data from
    assert(YTrue.shape == YPredict.shape)
    labels = sorted(set(YTrue) | set(YPredict))
    label_2_id = dict()
    for i, label in enumerate(labels):
        label_2_id[label] = i
    conf_mat = np.ones((len(labels), len(labels)), dtype=np.int64)
    for yTrue, yPredict in zip(YTrue, YPredict):
        t = label_2_id[yTrue]
        p = label_2_id[yPredict]
        conf_mat[t, p] += 1
    return conf_mat, labels, label_2_id


def getPerformanceScores(YTrue, YPredict, true_label=1):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """
    conf_mat, _, label_2_id = getConfusionMatrix(YTrue, YPredict)
    eye_sum = 0
    for i in range(conf_mat.shape[0]):
        eye_sum += conf_mat[i, i]
    accuracy = eye_sum / np.sum(conf_mat)
    if true_label not in label_2_id.keys():
        raise Exception("unexpected label")
    lt = label_2_id[true_label]
    precision = conf_mat[lt, lt] / np.sum(conf_mat[:, lt])
    recall = conf_mat[lt, lt] / np.sum(conf_mat[lt, :])
    f1 = precision * recall / (precision + recall)
    return {
        "CM": conf_mat,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
