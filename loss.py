import numpy as np
import acti


def dEuclideanLoss(YPredict, YTrue):
    if YPredict.shape != YTrue.shape:
        YTrue = YTrue.reshape(YPredict.shape)
    return YPredict - YTrue


def euclideanLoss(YPredict, YTrue):
    YDiff = dEuclideanLoss(YPredict, YTrue)
    YDiffPwr = np.power(YDiff, 2)
    return 0.5 * np.sum(YDiffPwr, axis=0)


def dSoftmaxLoss(YPredict, YTrueOneHot):
    YTrueOneHot = YTrueOneHot.reshape(YPredict.shape)
    pred_argmax = np.argmax(YPredict, axis=1)
    pred = np.zeros_like(YTrueOneHot)
    for i, argmax in enumerate(pred_argmax):
        pred[i, argmax] = 1
    delta = np.equal(pred, YTrueOneHot).astype(np.int64)
    return delta


def crossEntropyLoss(YPredict, YTrue):
    if YPredict.shape != YTrue.shape:
        YTrue = YTrue.reshape(YPredict.shape)
    return -np.sum(np.multiply(np.log(YPredict), YTrue))


def dCrossEntropyLoss_dSoftmax(YPredict, YTrueOneHot):
    if YPredict.shape != YTrueOneHot.shape:
        YTrueOneHot = YTrueOneHot.reshape(YPredict.shape)
    return np.multiply(YPredict, np.sum(YTrueOneHot, axis=1, keepdims=True)) - YTrueOneHot


def crossEntropyPredict(YPredict):
    YPredict = np.atleast_2d(YPredict)
    return np.argmax(YPredict, axis=1)
