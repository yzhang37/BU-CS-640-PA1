import numpy as np


def dEuclideanLoss(YPredict, YTrue):
    """
    Compute dEuclideanLoss. Only used by the backward propagation training process.
    :param YPredict: The predicted Y.
    :param YTrue: The golden y
    :return: The -benefits.
    """
    if YPredict.shape != YTrue.shape:
        YTrue = YTrue.reshape(YPredict.shape)
    return YPredict - YTrue


def euclideanLoss(YPredict, YTrue):
    """
    Compute Euclidean Loss.
    Only used when you want to evaluate the model's performance, and used by the getLoss function
    This function's not needed by the backward propagation training process.
    :param YPredict: The predicted Y.
    :param YTrue: The golden y
    :return: The Euclidean Loss.
    """
    YDiff = dEuclideanLoss(YPredict, YTrue)
    YDiffPwr = np.power(YDiff, 2)
    return 0.5 * np.sum(YDiffPwr, axis=0)


def crossEntropyLoss(YPredict, YTrueOneHot):
    """
    Compute the cross entropy loss.
    Only used when you want to evaluate the model's performance, and used by the getLoss function
    This function's not needed by the backward propagation training process.
    :param YPredict: The predicted Y.
    :param YTrueOneHot: The golden y. It's worth mentioning that here it's an one-hot vector.
    :return: The cross entropy loss.
    """
    if YPredict.shape != YTrueOneHot.shape:
        YTrueOneHot = YTrueOneHot.reshape(YPredict.shape)
    return -np.sum(np.multiply(np.log(YPredict), YTrueOneHot))


def dCrossEntropyLoss_dSoftmax(YPredict, YTrueOneHot):
    """
    Compute dCrossEntropyLoss + dSoftmax. Only used by the backward propagation training process.

    Notes: Why don't I write `dCrossEntropyLoss` and `dSoftmax` separately here?
    Because here the function compute the value based on `YPredict`, which is a vector already passed
    through the `softmax` activation function. We assumed it as $\vec{S} \in \R^{n}$, and the vector before the `softmax` is $D \in \R^{n}$. So when we compute $\dfrac{\partial Loss}{\partial S_{j}}$, what we got is $-\dfrac{y_{j}^{true}}{S_{j}}\cdot \dfrac{\partial S_j}{\partial \vec{D}}$.

    When compute $\dfrac{\partial S_j}{\partial \vec{D}}$, assumed we compute each $\dfrac{\partial S_j}{\partial D_{i}} =\left\{\begin{array}{rcl}S_{j} (1-S_i)  & ,i = j\\-S_{j} S_i & ,i\neq j\\\end{array}\right.$.

    During the debugging process, I found that this $s_j$ is likely to be 0, and division here can easily lead to NaN errors. At the same time, if the formula of the chain rule is multiplied, this $s_j$ here can be reduced. So I finally decided to write these two steps together. At the same time, the upper layer still needs a `dActivation` function. I use `dLinearUnit` directly there, and the problem can be solved perfectly.

    :param YPredict: The predicted Y, which is already passed out from the `softmax` activation function.
    :param YTrueOneHot: The golden Y. It's worth mentioning that here it's an one-hot vector.
    :return: The -benefits.
    """
    if YPredict.shape != YTrueOneHot.shape:
        YTrueOneHot = YTrueOneHot.reshape(YPredict.shape)
    return np.multiply(YPredict, np.sum(YTrueOneHot, axis=1, keepdims=True)) - YTrueOneHot


def crossEntropyPredict(YPredict):
    """
    This is only a function, which can be useful when make prediction.
    It will find the largest value's index.
    Only used in network's predict method.
    :param YPredict:
    :return:
    """
    YPredict = np.atleast_2d(YPredict)
    return np.argmax(YPredict, axis=1)
