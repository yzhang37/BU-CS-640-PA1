import numpy as np
import scipy as sp
import os, sys
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import pandas

################################################################################
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
    # now we make raw_y to one-hot vectors
    # num_classes = np.max(raw_y) + 1
    # num_samples = X.shape[0]
    # one_hot_y = np.zeros((num_samples, num_classes))
    # for i, y_i in enumerate(raw_y):
    #     one_hot_y[i][y_i] = 1
    return X, y


def make_cross_validation_data(X, y, fold=10, shuffle=True):
    assert (X.shape[0] == y.shape[0])
    skf = StratifiedKFold(n_splits=fold, shuffle=shuffle)
    for train, test in skf.split(X, y):
        yield X[train], X[test], y[train], y[test]


def getDataset2():
    d2 = joinPath(DATA_PATH, "dataset2")


################################################################################
# Define Activation and Loss Functions


def sigmoid(x):
    """
    Calculate the Sigmoid activation function.
    :param x: Scalar, Array or Matrix
    :return: result, of the same dimension of x.
    """
    exp_x = np.exp(x)
    return exp_x / (exp_x + 1)


def dSigmoid(x):
    # calc the result of derivative of sigmoid.
    return x * (1 - x)


def no_activation(x):
    # just no activation
    return x


def dno_activation(x):
    # no activation, so return 1
    return 1


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


################################################################################
# Define the Layer Class

class Layer:
    def __init__(self, NInput, NOutput, bias=True):
        self.NInput = NInput
        self.NOutput = NOutput
        self.useBias = bias
        self.initializeWeights()

    def initializeWeights(self):
        """
        Initializes the weights and biases with small random values.
        """
        # create a weights matrix, of Input x Output size.
        self.weights = np.random.randn(self.NInput, self.NOutput)

        # if use Bias, then create another
        if self.useBias:
            self.bias = np.zeros((1, self.NOutput))


################################################################################
# Define the Network Class

class Network:
    def __init__(self, layers, activationList, dActivationList, loss, dLoss):
        """
        :param layers: List[Layers]
               This should be a list of Layers objects.
        :param activationList: List[functions].
               a list of activation functions (e.g. sigmoid).
        :param dActivationList: List[functions].
               a list of derivative functions (e.g. dSigmoid).
        :param loss: function.
               a loss function (e.g. enclideanLoss).
        :param dLoss: function.
               a loss derivative function (e.g. dEuclideanLoss).
        """
        assert len(layers) == len(activationList) == len(dActivationList)
        self.layers = layers
        self.activationList = activationList
        self.dActivationList = dActivationList
        self.loss = loss
        self.dLoss = dLoss

        self.gradients = []
        self.grad_bias = []
        self.grad_zero()

    def grad_zero(self):
        """
        Make the gradients zero
        """
        self.gradients.clear()
        self.grad_bias.clear()
        for layer in self.layers[::-1]:
            self.gradients.append(np.zeros_like(layer.weights))
            if layer.useBias:
                self.grad_bias.append(np.zeros_like(layer.bias))
            else:
                self.grad_bias.append(None)

    def fit(self, X, Y, learningRate, regLambda, regMethod="l2"):
        """
        Fit the model with input features and targets.
        Parameters
        ----------
        X, Y : array-like
            X contains the input features while Y contains the target values.
        learningRate, regLambda : float
            Basic hyperparameters for the model.
        """

        if regMethod not in ("l1", "l2"):
            raise Exception("invalid reg method.")

        # First, initialize zero gradients.
        self.grad_zero()

        # Next, use the forward and backprog functions to accumulate gradients
        # sample by sample.
        # Hint:
        # For each sample, do:
        #     Forward pass once, using the forward function defined below.
        #     Backward propagation, using the backprog function defined below.

        for x, y in zip(X, Y):
            all_layers = self.forward(x)
            self.backprog(all_layers, y)

        # Lastly, update weights and biases using the gradients; don't forget to
        # take the mean before updating the weights. Note that both the learning
        # rate and the regularization should appear here.

        num_sample = X.shape[0]
        c = learningRate / num_sample
        for rev_i, layer in enumerate(self.layers[::-1]):
            if layer.useBias:
                layer.bias -= c * self.grad_bias[rev_i]
            if regMethod == "l1":
                layer.weights = layer.weights - c * regLambda * np.sign(layer.weights) - c * self.gradients[rev_i]
            elif regMethod == "l2":
                layer.weights = (1 - c * regLambda) * layer.weights - c * self.gradients[rev_i]

    def predict(self, X):
        """
        Predict the outcome of given input features. This should be as easy as
        Parameters
        ----------
        X : array-like
        """
        y_output = []
        for x in X:
            y = (self.forward(x)[-1]).flatten()
            if y.shape[0] == 1:
                y = y[0]
            y_output.append(y)
        return np.array(y_output)

    def forward(self, x):
        """
        Performs the forward propagation for ONE sample.
        x  : array-like
            Note that this is just ONE single sample.
        """
        working = np.atleast_1d(x)[np.newaxis, :]
        all_layers = [working]
        for i, (layer, act_func) in enumerate(zip(self.layers, self.activationList)):
            working = act_func(np.dot(working, layer.weights) + layer.bias)
            all_layers.append(working)
        return all_layers

    def backprog(self, all_layers, y):
        """
        Performs the backward propagation. This is the core part of the model.
        Parameters
        ----------
        None, but feel free to add anything you like.
        """
        yPred = all_layers[-1]
        # First, compute the derivative of the loss and the output activation, as beta
        beta = self.dLoss(yPred, y)

        rev_hidden = all_layers[::-1]
        rev_dActLists = self.dActivationList[::-1]
        # Then, compute the gradient layer by layer in the reverse order.
        for rev_i, layer_i in enumerate(self.layers[::-1]):
            # dAct_layer_i is, the dActivation of the current layer_(i+1)
            dAct_layer_i = rev_dActLists[rev_i](rev_hidden[rev_i])
            beta = np.multiply(beta, dAct_layer_i)

            if layer_i.useBias:
                self.grad_bias[rev_i] += beta

            new_gradient = np.dot(rev_hidden[rev_i + 1].T, beta)
            self.gradients[rev_i] += new_gradient
            beta = np.dot(beta, layer_i.weights.T)

    def getLoss(self, X, Y, regLambda=0.0, regMethod="l2"):
        """
        Compute and return the loss given X and Y. This function is helpful for
        you to visualize the training process.
        """
        if regMethod not in ("l1", "l2"):
            raise Exception("invalid reg method.")
        y_pred = self.predict(X)
        num_samples = X.shape[0]
        _loss = self.loss(y_pred, Y) / num_samples
        _reg = 0.0
        if regLambda != 0:
            if regMethod == "l1":
                for layer in self.layers:
                    _reg += np.sum(np.abs(layer.weights))
            elif regMethod == "l2":
                for layer in self.layers:
                    _reg += np.sum(np.square(layer.weights))
                _reg /= 2.0
        return _loss + regLambda * _reg / num_samples


#######################################################################################
# Define Evaluation Functions
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


#######################################################################################
# !! this code is provided by the instructor
def plotDecisionBoundary(model, X, Y):
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.bwr)
    plt.show()

# continue writing your code in this block
#######################################################################################
# Test Model

def main():
    AllX, AllY = getDataset1(is_linear=True)
    XTrain, XTest, YTrain, YTest = next(make_cross_validation_data(AllX, AllY))

    # assemble your model
    layers = [Layer(2, 4), Layer(4, 1)]
    model = Network(layers, [sigmoid, sigmoid], [dSigmoid, dSigmoid], euclideanLoss, dEuclideanLoss)

    # specify training parameters
    epochs = 2000
    learningRate = 0.05
    regLambda = 0

    # capture the loss values during training
    loss = {"train": [0.0] * epochs, "test": [0.0] * epochs}

    # start training
    for epoch in range(epochs):
        model.fit(XTrain, YTrain, learningRate, regLambda)
        loss["train"][epoch] = model.getLoss(XTrain, YTrain, regLambda)
        loss["test"][epoch] = model.getLoss(XTest, YTest, regLambda)

    print(YTest)
    print(model.predict(XTest))

    # plot the losses, both curves should be decreasing
    plt.plot([i for i in range(epochs)], loss["train"], label="train")
    plt.plot([i for i in range(epochs)], loss["test"], label="test")
    plt.legend()
    plt.title("Loss Plot")
    plt.xlabel("Epoch")
    plt.show()


if __name__ == "__main__":
    main()
