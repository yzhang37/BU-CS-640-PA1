import numpy as np


def dEuclideanLoss(YPredict, YTrue):
    if YTrue.ndim == 0 or YPredict.ndim == 0:
        ans = YPredict - YTrue
        if not (ans.ndim == 0 or np.prod(ans.shape) == 1):
            raise Exception("dimension error")
    elif YTrue.ndim == 1 and YPredict.ndim == 1:
        ans = YPredict - YTrue
    elif np.prod(YTrue.ndim) == np.prod(YPredict.ndim):
        if YPredict.ndim == 1:
            ans = YPredict.reshape(1,) - YTrue[np.newaxis, :]
        elif YTrue.ndim == 1:
            ans = YPredict.reshape[np.newaxis, :] - YTrue(1,)
        elif YPredict.shape == YTrue.shape:
            ans = YPredict - YTrue
        else:
            raise Exception("dimension error")
    else:
        raise Exception("dimension error")
    return ans


def euclideanLoss(YPredict, YTrue):
    YDiff = dEuclideanLoss(YPredict, YTrue)
    YDiffPwr = np.power(YDiff, 2)
    return 1 / 2 * np.sum(YDiffPwr, axis=0)


def dSoftmaxLoss(YPredict, YTrueOneHot):
    YTrueOneHot = YTrueOneHot.reshape(YPredict.shape)
    pred_argmax = np.argmax(YPredict, axis=1)
    pred = np.zeros_like(YTrueOneHot)
    for i, argmax in enumerate(pred_argmax):
        pred[i, argmax] = 1
    delta = np.equal(pred, YTrueOneHot).astype(np.int64)
    return delta


def softmaxLoss(YPredict, YTrueOneHot):
    YTrueOneHot = YTrueOneHot.reshape(YPredict.shape)
    pred_argmax = np.argmax(YPredict, axis=1)
    ytrue_argmax = np.argmax(YTrueOneHot, axis=1)
    return np.sum(1 - np.equal(pred_argmax, ytrue_argmax).astype(np.int64))