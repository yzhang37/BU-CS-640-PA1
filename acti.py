import numpy as np


def sigmoid(x):
    """
    Calculate the Sigmoid activation function.
    :param x: Scalar, Array or Matrix
    :return: result, of the same dimension of x.
    """
    x = np.atleast_2d(x)
    return 1 / (1 + np.exp(-x))


def dSigmoid(output, beta):
    # calc the result of derivative of sigmoid.
    coef = output * (1 - output)
    return np.multiply(beta, coef)


def linearUnit(x):
    # Unlike ReLU, LinearUnit really passes the previous data
    # directly to the next layer without any activation.
    return np.atleast_2d(x)


def dLinearUnit(output, beta):
    # no activation, so return 1
    return beta


def softmax(x):
    # use the softmax as the activation function.
    x = np.atleast_2d(x)
    shift = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

