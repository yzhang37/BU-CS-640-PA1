from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from get_data import getD1Linear, getD1NonLinear, getD2, make_one_hot
from layer import Layer
from network import Network
import acti
import loss as lossFuncs


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


def train_dat1():
    # data = getD1Linear()
    # XTrain, YTrain, XTest, YTest = next(data.getStratifiedKFold())
    # AllX, AllY = data.get()
    data = getD2()
    XTrain, YTrain = data.get("train")
    XTest, YTest = data.get("test")
    AllX, AllY = data.get()

    # layers = [Layer(2, 4), Layer(4, 1)]
    # model = Network(layers, [sigmoid, sigmoid], [dSigmoid, dSigmoid], euclideanLoss, dEuclideanLoss)
    layers = [Layer(64, 40), Layer(40, 10)]
    activations = [acti.sigmoid, acti.softmax]
    dActivations = [acti.dSigmoid, acti.dLinearUnit]
    model = Network(layers, activations, dActivations,
                    lossFuncs.crossEntropyLoss, lossFuncs.dCrossEntropyLoss_dSoftmax,
                    lossFuncs.crossEntropyPredict)

    epochs = 10000
    learningRate = 0.01
    regLambda = 0

    loss = {"train": [0.0] * epochs, "test": [0.0] * epochs}

    YTrainOneHot = make_one_hot(YTrain)
    YTestOneHot = make_one_hot(YTest)
    for epoch in tqdm(range(epochs)):
        # model.fit(XTrain, YTrain, learningRate, regLambda)
        model.fit(XTrain, YTrainOneHot, learningRate, regLambda)
        loss["train"][epoch] = model.getLoss(XTrain, YTrainOneHot, regLambda)
        loss["test"][epoch] = model.getLoss(XTest, YTestOneHot, regLambda)
        if (epoch + 1) % 1000 == 0:
            print("epoch {0}: ltrain: {1:0.6f}, ltest: {2:0.6f}".format(epoch + 1, loss["train"][epoch], loss["test"][epoch]))
    print(YTest.T)
    print(model.predict(XTest).T)

    plt.plot([i for i in range(epochs)], loss["train"], label="train")
    plt.plot([i for i in range(epochs)], loss["test"], label="test")
    plt.legend()
    plt.title("Loss Plot")
    plt.xlabel("Epoch")
    plt.show()

    # plotDecisionBoundary(model, AllX, AllY)


# def train_dat2():
#     data = getD2()
#     XTrain, YTrain = data.get("train")
#     XTest, YTest = data.get("test")
#     AllX, AllY = data.get()
#
#     # assemble your model
#     layers = [Layer(2, 4), Layer(4, 1)]
#     model = Network(layers, [sigmoid, sigmoid], [dSigmoid, dSigmoid], euclideanLoss, dEuclideanLoss)
#
#     # specify training parameters
#     epochs = 50
#     learningRate = 0.001
#     regLambda = 0
#
#     # capture the loss values during training
#     loss = {"train": [0.0] * epochs, "test": [0.0] * epochs}
#
#     # start training
#     for epoch in range(epochs):
#         model.fit(XTrain, YTrain, learningRate, regLambda)
#         loss["train"][epoch] = model.getLoss(XTrain, YTrain, regLambda)
#         loss["test"][epoch] = model.getLoss(XTest, YTest, regLambda)
#     print(YTest)
#     print(model.predict(XTest))
#
#     # plot the losses, both curves should be decreasing
#     plt.plot([i for i in range(epochs)], loss["train"], label="train")
#     plt.plot([i for i in range(epochs)], loss["test"], label="test")
#     plt.legend()
#     plt.title("Loss Plot")
#     plt.xlabel("Epoch")
#     plt.show()
#
#     plotDecisionBoundary(model, AllX, AllY)


if __name__ == "__main__":
    train_dat1()
