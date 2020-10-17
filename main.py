import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas


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


def main():
    AllX, AllY = getDataset1(is_linear=True)
    XTrain, XTest, YTrain, YTest = next(make_cross_validation_data(AllX, AllY))

    # assemble your model
    layers = [Layer(2, 4), Layer(4, 1)]
    model = Network(layers, [sigmoid, sigmoid], [dSigmoid, dSigmoid], euclideanLoss, dEuclideanLoss)

    # specify training parameters
    epochs = 50
    learningRate = 0.001
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
